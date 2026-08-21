"""
Minimal evaluation script for Diffusion Track Prediction (DiT).

Loads a model checkpoint, runs predict() over a folder of demo-format
examples (see motion_forecasting/data), computes ADE/FDE, and prints
results. Optionally renders per-sample visualizations.

Configuration comes from conf/eval.yaml; any key can be overridden on the
command line in OmegaConf dotlist syntax:

    python -m engine.eval checkpoint=/path/to/model.ckpt
    python -m engine.eval checkpoint=/path/to/model.ckpt noise_seed=0 \
        visualize=true viz_style=animal_color viz_dir=viz_out

Checkpoints saved by this codebase embed the model_cfg used to build the
architecture, so only `checkpoint` and `data_root` are required.
"""

import logging
import os

import torch
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from motion_forecasting.data import ExampleDataset
from motion_forecasting.model.pipeline import (
    create_model,
    create_diffusion,
    create_dino_extractor,
    load_checkpoint,
    load_ema_weights,
    predict,
)

logger = logging.getLogger(__name__)


def _compute_ade_fde(pred, gt):
    """Simple ADE/FDE in normalized [0,1] space.

    Args:
        pred: (T, N, 2) predicted tracks
        gt:   (T, N, 2) ground truth tracks

    Returns:
        (ade, fde) floats.
    """
    diffs = np.linalg.norm(pred - gt, axis=-1)  # (T, N)
    ade = float(np.mean(diffs))
    fde = float(np.mean(diffs[-1]))
    return ade, fde


_EVAL_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "conf", "eval.yaml")


def load_eval_cfg():
    """Load conf/eval.yaml and merge command-line dotlist overrides."""
    cfg = OmegaConf.load(_EVAL_CONFIG_PATH)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())
    if not cfg.get("checkpoint"):
        raise ValueError("Set `checkpoint` in conf/eval.yaml or pass checkpoint=/path/to.ckpt")
    return cfg


def _batch_to_pipeline(batch, device):
    """Map a (collated) dataloader batch to the pipeline batch format."""
    pipeline_batch = {
        "video": batch["video"].to(device),
        "track": batch["tracks"].to(device),
        "visibility": batch.get("visibility", None),
        "point_mask": batch.get("point_mask", None),
        "total_displacement": batch.get("total_displacement", None),
    }
    for key in ("visibility", "point_mask", "total_displacement"):
        if pipeline_batch[key] is not None:
            pipeline_batch[key] = pipeline_batch[key].to(device)
    return pipeline_batch


@torch.no_grad()
def evaluate(model, diffusion, dino_extractor, dataloader, cfg, device="cuda", max_samples=None):
    """Run batched evaluation and compute metrics.

    Args:
        model: DiT_MammalNet_SingleImage instance.
        diffusion: GaussianDiffusion instance.
        dino_extractor: DINOFeatureExtractor or None.
        dataloader: Evaluation dataloader.
        cfg: Pipeline config dict.
        device: Device string.
        max_samples: Optional limit on number of batches to evaluate.

    Returns:
        dict of aggregated metrics.
    """
    model.eval()
    num_cond = cfg.get("num_point_cond", 1)
    all_ade = []
    all_fde = []

    for i, batch in enumerate(dataloader):
        if max_samples is not None and i >= max_samples:
            break

        pipeline_batch = _batch_to_pipeline(batch, device)

        result = predict(model, diffusion, pipeline_batch, dino_extractor, cfg)

        pred = result["pred_tracks"].cpu().numpy()  # (B, T, N, 2)
        gt = result["gt_tracks"].cpu().numpy()       # (B, T, N, 2)

        for b in range(pred.shape[0]):
            pred_future = pred[b, num_cond:]
            gt_future = gt[b, num_cond:]
            ade, fde = _compute_ade_fde(pred_future, gt_future)
            all_ade.append(ade)
            all_fde.append(fde)

        if (i + 1) % 10 == 0:
            logger.info("Evaluated %d batches...", i + 1)

    return {
        "ade": float(np.mean(all_ade)) if all_ade else 0.0,
        "fde": float(np.mean(all_fde)) if all_fde else 0.0,
        "num_samples": len(all_ade),
    }


@torch.no_grad()
def evaluate_examples(model, diffusion, dino_extractor, dataset, cfg,
                      device="cuda", max_samples=None, visualize=False,
                      viz_dir="eval_viz", viz_style="tracks"):
    """Evaluate examples one at a time, optionally rendering visualizations.

    Iterates samples at batch size 1 so per-sample names stay aligned with
    outputs.

    viz_style:
        "tracks"       — 3x3 GT-vs-pred debug grid (plot_stabilized_tracks_video).
        "animal_color" — paper-style texture-colored GT + prediction videos.
    """
    model.eval()
    num_cond = cfg.get("num_point_cond", 1)
    all_ade, all_fde = [], []

    if visualize:
        if viz_style == "animal_color":
            from motion_forecasting.model.animal_color_viz import (
                generate_animal_color_videos_for_sample,
            )
        elif viz_style == "tracks":
            import mediapy as media
            from motion_forecasting.model.track_visualization import plot_stabilized_tracks_video
        else:
            raise ValueError(f"Unknown viz_style {viz_style!r} (expected 'tracks' or 'animal_color')")
        os.makedirs(viz_dir, exist_ok=True)

    indices = range(len(dataset))
    if max_samples is not None:
        indices = range(min(len(dataset), max_samples))

    for n in indices:
        sample = dataset[n]
        sample_name = sample.get("name", f"idx{n}")
        batch = default_collate([sample])
        pipeline_batch = _batch_to_pipeline(batch, device)

        result = predict(model, diffusion, pipeline_batch, dino_extractor, cfg)

        pred = result["pred_tracks"].cpu()
        gt = result["gt_tracks"].cpu()

        ade, fde = _compute_ade_fde(pred[0, num_cond:].numpy(), gt[0, num_cond:].numpy())
        all_ade.append(ade)
        all_fde.append(fde)

        logger.info("[%d/%d] %s  ADE=%.5f FDE=%.5f",
                    n + 1, len(indices), sample_name, ade, fde)

        if visualize and viz_style == "animal_color":
            written = generate_animal_color_videos_for_sample(
                batch, gt, pred, viz_dir, f"{n:03d}_{sample_name}",
                num_point_cond=num_cond)
            for path in written:
                logger.info("  saved %s", path)
        elif visualize:
            # Demo-example tracks are normalized to the example image itself,
            # so the renderer's no-bbox fallback (tracks * image size) applies.
            composite = plot_stabilized_tracks_video(
                batch["video"].float(), gt, pred,
                condition_on_displacement=cfg.get("condition_on_displacement", False),
                point_mask=batch.get("point_mask", None),
                gt_visibility=batch.get("visibility", None),
                pred_visibility=batch.get("visibility", None),
            )
            # composite: (1, T, 3, H, W) -> (T, H, W, 3) for mediapy
            frames = composite[0].transpose(0, 2, 3, 1).astype(np.uint8)
            out_path = os.path.join(viz_dir, f"{n:03d}_{sample_name}.mp4")
            media.write_video(out_path, frames, fps=4)
            logger.info("  saved %s", out_path)

    return {
        "ade": float(np.mean(all_ade)) if all_ade else 0.0,
        "fde": float(np.mean(all_fde)) if all_fde else 0.0,
        "num_samples": len(all_ade),
    }


def main():
    eval_cfg = load_eval_cfg()
    logging.basicConfig(level=logging.INFO)

    # Checkpoints saved by this codebase embed the flat model_cfg they were
    # built with, making them self-describing.
    checkpoint = torch.load(eval_cfg.checkpoint, map_location="cpu", weights_only=False)
    if not (isinstance(checkpoint, dict) and "model_cfg" in checkpoint):
        raise ValueError(
            "Checkpoint has no embedded model_cfg. Evaluation requires a "
            "checkpoint saved by this codebase (scripts/inference.py supports "
            "external configs via --config).")
    pipeline_cfg = dict(checkpoint["model_cfg"])
    if eval_cfg.get("motion_history_conditioning") is not None:
        pipeline_cfg["motion_history_conditioning"] = str(
            eval_cfg.motion_history_conditioning)
    pipeline_cfg.update({
        "use_ddim": bool(eval_cfg.use_ddim),
        "ddim_timesteps": int(eval_cfg.ddim_steps),
        "ddim_eta": float(eval_cfg.ddim_eta),
    })
    if eval_cfg.get("noise_seed") is not None:
        pipeline_cfg["noise_seed"] = int(eval_cfg.noise_seed)
    use_null_displacement = bool(eval_cfg.use_null_displacement)
    pipeline_cfg["use_null_displacement"] = use_null_displacement
    logger.info("Sampling mode: %s",
                "UNCONDITIONAL (null displacement)" if use_null_displacement
                else "conditional (GT displacement)")

    model = create_model(pipeline_cfg)

    logger.info("Loading checkpoint: %s", eval_cfg.checkpoint)
    metadata = load_checkpoint(model, checkpoint)
    if metadata.get("dino_scale_factor") is not None:
        pipeline_cfg["dino_scale_factor"] = metadata["dino_scale_factor"]
    if metadata.get("occlusion_scaling") is not None:
        pipeline_cfg["occlusion_scaling"] = metadata["occlusion_scaling"]

    device = eval_cfg.device
    if bool(eval_cfg.no_ema):
        logger.info("Weights: live model_state_dict (no_ema=true)")
    else:
        if load_ema_weights(model, checkpoint):
            logger.info("Weights: EMA shadow (default; set no_ema=true for live weights)")
        else:
            logger.warning("Weights: checkpoint has no EMA shadow — using live weights")
    model.to(device)

    diffusion = create_diffusion(pipeline_cfg)

    dino_extractor = None
    if pipeline_cfg.get("use_dino_features", False):
        dino_extractor = create_dino_extractor({
            "dino_layer": pipeline_cfg.get("dino_layer", 23),
            "dino_device": device,
        })

    dataset = ExampleDataset(
        str(eval_cfg.data_root),
        num_points=pipeline_cfg.get("num_points", 320),
        horizon=pipeline_cfg.get("horizon", 32),
        num_point_cond=pipeline_cfg.get("num_point_cond", 4),
        img_size=pipeline_cfg.get("img_size", 256),
    )

    visualize = bool(eval_cfg.visualize)
    max_samples = eval_cfg.get("max_samples")
    max_samples = int(max_samples) if max_samples is not None else None

    if visualize:
        logger.info("Evaluating %d examples (per-sample, with visualization)...",
                    len(dataset))
        metrics = evaluate_examples(model, diffusion, dino_extractor, dataset,
                                    pipeline_cfg, device=device,
                                    max_samples=max_samples,
                                    visualize=True,
                                    viz_dir=str(eval_cfg.viz_dir),
                                    viz_style=str(eval_cfg.viz_style))
    else:
        loader = DataLoader(dataset, batch_size=int(eval_cfg.batch_size),
                            shuffle=False, num_workers=int(eval_cfg.num_workers))
        logger.info("Evaluating %d examples...", len(dataset))
        metrics = evaluate(model, diffusion, dino_extractor, loader, pipeline_cfg,
                           device=device, max_samples=max_samples)

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k:>20s}: {v:.6f}")
        else:
            print(f"  {k:>20s}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
