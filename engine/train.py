"""
Minimal training script for Diffusion Track Prediction (DiT).

Uses pipeline.py standalone functions, Lightning Fabric for DDP,
and optional EMA. Metrics are logged to stdout / the Hydra log file.

Trains on demo-format example folders (see motion_forecasting/data);
to train on your own data, either export it in that format or swap in
any dataloader that yields the batch dict described in the README.

Run via:
    python scripts/train.py [hydra config overrides]
"""

import os
import time
import logging

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy
from datetime import timedelta
import numpy as np

from motion_forecasting.data import ExampleDataset
from motion_forecasting.model.pipeline import (
    create_model,
    create_diffusion,
    create_dino_extractor,
    load_checkpoint,
    save_checkpoint,
    train_step,
)
from motion_forecasting.utils.train_utils import setup_lr_scheduler, setup_optimizer
from motion_forecasting.utils.ema import EMAModel
from engine.train_utils import (
    _seed_worker,
    _move_optimizer_state_to_device,
    build_model_cfg,
)

logger = logging.getLogger(__name__)


# ======================================================================
# Deterministic seeding
# ======================================================================

def setup(cfg):
    """Set global random seeds for reproducibility."""
    import random
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================================================================
# Training epoch
# ======================================================================

def run_one_epoch(
    fabric, model, diffusion, dino_extractor, dataloader, optimizer,
    cfg, scheduler=None, clip_grad=1.0, limit_batches=None,
    ema=None, epoch=None, global_step_container=None,
    pipeline_cfg=None, log_every=50,
):
    """Run one training epoch.

    Args:
        fabric: Lightning Fabric instance.
        model: DiT_MammalNet_SingleImage wrapped by Fabric.
        diffusion: GaussianDiffusion instance.
        dino_extractor: DINOFeatureExtractor or None.
        dataloader: Training dataloader wrapped by Fabric.
        optimizer: Optimizer wrapped by Fabric.
        cfg: Full Hydra config.
        scheduler: Optional LR scheduler.
        clip_grad: Max gradient norm.
        limit_batches: Optional max number of batches per epoch.
        ema: Optional EMAModel.
        epoch: Current epoch index.
        global_step_container: Optional [step_count] for global step tracking.
        pipeline_cfg: Dict passed to train_step as cfg.
        log_every: Log step metrics every N optimizer steps (rank 0).

    Returns:
        dict with epoch metrics (avg_loss, etc.).
    """
    if epoch is not None and hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'set_epoch'):
        dataloader.sampler.set_epoch(epoch)

    model.train()
    total_loss = 0.0
    total_items = 0
    rank = fabric.global_rank

    for i, batch in enumerate(tqdm(dataloader, disable=(rank != 0))):
        if limit_batches is not None and i >= limit_batches:
            break

        try:
            pipeline_batch = {
                "video": batch["video"],
                "track": batch["tracks"],
                "visibility": batch.get("visibility", None),
                "point_mask": batch.get("point_mask", None),
                "total_displacement": batch.get("total_displacement", None),
            }
        except Exception as e:
            logger.warning("Error loading batch %d: %s", i, e)
            continue

        result = train_step(model, diffusion, pipeline_batch, dino_extractor, pipeline_cfg)
        loss = result["loss"]

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            logger.warning("NaN/Inf loss at batch %d, skipping", i)
            continue

        optimizer.zero_grad()
        fabric.backward(loss)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        if ema is not None:
            ema.update()

        total_loss += loss.item() * pipeline_batch["video"].shape[0]
        total_items += pipeline_batch["video"].shape[0]

        # Per-step logging (rank 0)
        if global_step_container is not None and fabric.is_global_zero:
            step = global_step_container[0]
            if step % max(log_every, 1) == 0:
                gn = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                logger.info(
                    "step %d: loss=%.6f grad_norm=%.4f lr=%.2e",
                    step, loss.item(), gn, optimizer.param_groups[0]["lr"])
            global_step_container[0] += 1

    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / max(total_items, 1)
    return {"avg_loss": avg_loss, "total_items": total_items}


# ======================================================================
# Main
# ======================================================================

@hydra.main(config_path="../conf", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    # Determine work directory
    if cfg.get("resume_dir") is not None and os.path.isdir(cfg.resume_dir):
        work_dir = cfg.resume_dir
        saved_config_path = os.path.join(work_dir, "config.yaml")
        if os.path.exists(saved_config_path):
            logger.info("RESUME MODE: Loading config from %s", saved_config_path)
            saved_cfg = OmegaConf.load(saved_config_path)
            for key in ("load_path", "resume_dir", "epochs", "train_gpus", "lr", "clip_grad"):
                if cfg.get(key) != saved_cfg.get(key) and cfg.get(key) is not None:
                    OmegaConf.update(saved_cfg, key, cfg.get(key))
            cfg = saved_cfg
    else:
        work_dir = HydraConfig.get().runtime.output_dir

    setup(cfg)

    os.makedirs(work_dir, exist_ok=True)
    OmegaConf.save(config=cfg, f=os.path.join(work_dir, "config.yaml"))

    # Fabric setup
    nccl_timeout = int(cfg.get("nccl_timeout_seconds", 1800))
    ddp_strategy = DDPStrategy(timeout=timedelta(seconds=nccl_timeout))
    fabric = Fabric(
        accelerator="cuda",
        devices=list(cfg.train_gpus),
        precision="bf16-mixed" if cfg.mix_precision else None,
        strategy=ddp_strategy,
    )
    fabric.launch()

    # Build model config
    model_cfg = build_model_cfg(cfg)
    pipeline_cfg = dict(model_cfg)  # train_step uses the same flat dict

    # Dataset: demo-format example folders (image.png + tracks.npz).
    # To train on your own data, swap in any dataset/dataloader that yields
    # the batch dict documented in the README.
    train_dataset = ExampleDataset(
        cfg.data_root,
        num_points=model_cfg["num_points"],
        horizon=model_cfg["horizon"],
        num_point_cond=model_cfg["num_point_cond"],
        img_size=model_cfg["img_size"],
    )
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, worker_init_fn=_seed_worker,
        pin_memory=True, drop_last=False,
    )
    logger.info("Train dataset: %d samples from %s", len(train_dataset), cfg.data_root)

    # Create model + diffusion
    logger.info("Creating model: %s", model_cfg.get("model_name"))
    model = create_model(model_cfg)

    logger.info("Creating diffusion (steps=%d)", model_cfg.get("diffusion_steps", 1000))
    diffusion = create_diffusion(model_cfg)

    # DINO extractor
    dino_extractor = None
    if model_cfg.get("use_dino_features", False):
        logger.info("Creating DINO extractor")
        dino_extractor = create_dino_extractor({
            "dino_layer": model_cfg.get("dino_layer", 23),
            "dino_device": f"cuda:{fabric.local_rank}",
        })

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %s", f"{total_params:,}")

    optimizer = setup_optimizer(cfg.optimizer_cfg, model)
    scheduler = setup_lr_scheduler(optimizer, cfg.scheduler_cfg)

    # Load checkpoint weights.
    # - load_path alone = finetune mode: weights only, fresh optimizer/epoch.
    # - resume_dir set  = full resume: also restore optimizer/scheduler/epoch/EMA
    #   (only valid for checkpoints saved by this codebase).
    start_epoch = 0
    ckpt = None
    full_resume = cfg.get("resume_dir") is not None
    if cfg.load_path is not None and os.path.exists(cfg.load_path):
        logger.info("Loading checkpoint: %s", cfg.load_path)
        metadata = load_checkpoint(model, cfg.load_path, strict=True)
        if metadata.get("dino_scale_factor") is not None:
            pipeline_cfg["dino_scale_factor"] = metadata["dino_scale_factor"]
        if metadata.get("occlusion_scaling") is not None:
            pipeline_cfg["occlusion_scaling"] = metadata["occlusion_scaling"]

        if full_resume:
            ckpt = torch.load(cfg.load_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict):
                try:
                    if "optimizer_state_dict" in ckpt:
                        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
                        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                    start_epoch = ckpt.get("epoch", 0)
                    logger.info("Resumed training state from epoch %d", start_epoch)
                except (ValueError, RuntimeError, KeyError) as e:
                    logger.warning(
                        "Could not restore optimizer/scheduler state (%s). "
                        "Continuing with fresh training state.", e)
                    start_epoch = 0
        else:
            logger.info("Finetune mode: loaded weights only (fresh optimizer, start_epoch=0)")

    if start_epoch >= cfg.epochs:
        logger.warning(
            "start_epoch (%d) >= epochs (%d): no training will run. "
            "Increase epochs or unset resume_dir for finetuning.",
            start_epoch, cfg.epochs)

    # Fabric wrapping
    model, optimizer = fabric.setup(model, optimizer)
    train_loader = fabric.setup_dataloaders(train_loader)

    if isinstance(ckpt, dict) and "optimizer_state_dict" in ckpt:
        _move_optimizer_state_to_device(optimizer, fabric.device)

    # EMA (created after fabric.setup so shadow params live on the right device)
    ema = None
    ema_cfg = cfg.get("ema", {})
    if ema_cfg.get("enabled", False):
        ema = EMAModel(model, decay=ema_cfg.get("decay", 0.9999),
                       start_step=ema_cfg.get("start_step", 0))
        logger.info("EMA enabled (decay=%s)", ema_cfg.get("decay"))
        if full_resume and isinstance(ckpt, dict) and "ema_state_dict" in ckpt:
            try:
                ema.load_state_dict(ckpt["ema_state_dict"])
                logger.info("Loaded EMA state from checkpoint")
            except (RuntimeError, KeyError, IndexError) as e:
                logger.warning("Could not restore EMA state (%s); starting EMA fresh.", e)

    # Training loop
    global_step = [0]
    clip_grad = cfg.get("clip_grad", 1.0)
    log_every = cfg.get("log_every", 50)

    for epoch in range(start_epoch, cfg.epochs):
        logger.info("Epoch %d/%d", epoch + 1, cfg.epochs)
        t0 = time.time()

        metrics = run_one_epoch(
            fabric=fabric,
            model=model,
            diffusion=diffusion,
            dino_extractor=dino_extractor,
            dataloader=train_loader,
            optimizer=optimizer,
            cfg=cfg,
            scheduler=scheduler,
            clip_grad=clip_grad,
            limit_batches=cfg.get("limit_train_batches", None),
            ema=ema,
            epoch=epoch,
            global_step_container=global_step,
            pipeline_cfg=pipeline_cfg,
            log_every=log_every,
        )

        epoch_time = time.time() - t0
        logger.info("Epoch %d: avg_loss=%.6f, time=%.1fs", epoch + 1, metrics["avg_loss"], epoch_time)

        # Save checkpoint
        if fabric.is_global_zero:
            ckpt_path = os.path.join(work_dir, "latest.pt")
            unwrapped = model
            if hasattr(unwrapped, "_forward_module"):
                unwrapped = unwrapped._forward_module
            if hasattr(unwrapped, "module"):
                unwrapped = unwrapped.module

            save_checkpoint(
                unwrapped, ckpt_path,
                optimizer=optimizer, scheduler=scheduler,
                epoch=epoch + 1,
                ema=ema,
                dino_scale_factor=pipeline_cfg.get("dino_scale_factor"),
                occlusion_scaling=pipeline_cfg.get("occlusion_scaling"),
            )

            # Periodic checkpoint
            save_every = cfg.get("save_freq", 25)
            if (epoch + 1) % save_every == 0:
                periodic_path = os.path.join(work_dir, f"epoch_{epoch + 1:04d}.pt")
                save_checkpoint(
                    unwrapped, periodic_path,
                    optimizer=optimizer, scheduler=scheduler,
                    epoch=epoch + 1,
                    ema=ema,
                    dino_scale_factor=pipeline_cfg.get("dino_scale_factor"),
                    occlusion_scaling=pipeline_cfg.get("occlusion_scaling"),
                )
                logger.info("Saved periodic checkpoint: %s", periodic_path)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
