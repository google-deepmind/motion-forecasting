"""
Standalone pipeline functions for diffusion-based track prediction.

Provides composable functions for model creation, checkpoint loading,
training, and inference without any wrapper class.
"""

import logging
import os
import torch
import torch.nn as nn
import numpy as np

from .dit_backbone import (
    DiT_MammalNet_SingleImage,
    DiT_MammalNet_models,
    TimestepEmbedder,
)
from .diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    ModelMeanType,
    ModelVarType,
    LossType,
    get_named_beta_schedule,
)
from .dino_feature_extractor import DINOFeatureExtractor, DINOV3_VITL_FEATURE_DIM
from .coordinate_utils import (
    get_channel_layout,
    build_fixed_channels_mask,
    coords_to_velocity,
    velocity_to_coords,
    compute_velocity_displacement_from_tracks,
    tracks_to_model_format,
    model_to_tracks_format,
    MAX_MOTION_HISTORY_COND,
)

logger = logging.getLogger(__name__)


# ======================================================================
# Factory functions
# ======================================================================

def create_model(cfg: dict) -> DiT_MammalNet_SingleImage:
    """Instantiate a DiT_MammalNet_SingleImage model from a config dict.

    The config dict should contain keys matching the constructor arguments.
    If ``model_name`` is provided, the corresponding factory function from
    ``DiT_MammalNet_models`` is used (which sets depth/hidden_size/num_heads).

    Args:
        cfg: dict with model configuration. Recognized keys:
            model_name, horizon, num_points, img_size, dino_feature_dim,
            handle_occlusions, diffuse_on_velocity, use_initial_pos_encoding,
            use_image_conditioning, vel_disp_conditioning_mode, motion_history_dim,
            condition_on_displacement, velocity_conditioning_type, vel_disp_dropout,
            single_cond_dropout_prob, null_embedding_type,
            vel_disp_token_type_embeddings, motion_history_scale,
            hidden_size, depth, num_heads (used only if model_name is absent).

    Returns:
        DiT_MammalNet_SingleImage instance.
    """
    model_name = cfg.get("model_name", None)

    model_kwargs = {}
    for key in (
        "horizon", "num_points", "img_size", "dino_feature_dim",
        "handle_occlusions", "diffuse_on_velocity", "use_initial_pos_encoding",
        "use_image_conditioning", "vel_disp_conditioning_mode", "motion_history_dim",
        "condition_on_displacement", "velocity_conditioning_type", "vel_disp_dropout",
        "single_cond_dropout_prob", "null_embedding_type",
        "vel_disp_token_type_embeddings", "motion_history_scale",
        "motion_history_conditioning",
        "learn_sigma", "cond_dim", "mlp_ratio",
    ):
        if key in cfg:
            model_kwargs[key] = cfg[key]

    if model_name and model_name in DiT_MammalNet_models:
        model = DiT_MammalNet_models[model_name](**model_kwargs)
    else:
        for key in ("hidden_size", "depth", "num_heads"):
            if key in cfg:
                model_kwargs[key] = cfg[key]
        model = DiT_MammalNet_SingleImage(**model_kwargs)

    return model


def create_diffusion(cfg: dict) -> GaussianDiffusion:
    """Instantiate a GaussianDiffusion object from a config dict.

    Args:
        cfg: dict with keys: diffusion_steps, noise_schedule, beta_start,
             beta_end, model_mean_type, model_var_type, loss_type.

    Returns:
        GaussianDiffusion instance.
    """
    diffusion_steps = cfg.get("diffusion_steps", 1000)
    noise_schedule = cfg.get("noise_schedule", "linear")
    beta_start = cfg.get("beta_start", 0.0001)
    beta_end = cfg.get("beta_end", 0.02)
    model_mean_type = cfg.get("model_mean_type", "epsilon")
    model_var_type = cfg.get("model_var_type", "fixed_small")
    loss_type = cfg.get("loss_type", "mse")

    if noise_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, diffusion_steps)
    else:
        betas = get_named_beta_schedule(noise_schedule, diffusion_steps)

    return GaussianDiffusion(
        betas=betas,
        model_mean_type=getattr(ModelMeanType, model_mean_type.upper()),
        model_var_type=getattr(ModelVarType, model_var_type.upper()),
        loss_type=getattr(LossType, loss_type.upper()),
    )


def create_dino_extractor(cfg: dict) -> DINOFeatureExtractor:
    """Instantiate a frozen DINOFeatureExtractor from a config dict.

    Args:
        cfg: dict with optional keys: dino_layer, dino_device.

    Returns:
        DINOFeatureExtractor instance.
    """
    dino_layer = cfg.get("dino_layer", 23)
    device = cfg.get("dino_device", "cuda" if torch.cuda.is_available() else "cpu")
    return DINOFeatureExtractor(device=device, dino_layer=dino_layer)


# ======================================================================
# Checkpoint loading
# ======================================================================

def _load_checkpoint_dict(path_or_checkpoint):
    """Accept a checkpoint path or an already-loaded checkpoint dict."""
    if isinstance(path_or_checkpoint, (str, os.PathLike)):
        return torch.load(path_or_checkpoint, map_location="cpu", weights_only=False)
    return path_or_checkpoint


def load_checkpoint(model: DiT_MammalNet_SingleImage, path, strict: bool = True):
    """Load a checkpoint whose keys match DiT_MammalNet_SingleImage directly.

    Expects the released checkpoint format (or any checkpoint saved by
    save_checkpoint()): ``model_state_dict`` holds the model's own keys
    with no wrapper prefixes.

    Args:
        model: target model.
        path: checkpoint path, or an already-loaded checkpoint dict.
        strict: passed to load_state_dict (default True).

    Returns:
        dict with any extra metadata from the checkpoint (e.g. dino_scale_factor).
    """
    checkpoint = _load_checkpoint_dict(path)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    remapped = dict(state_dict)

    # Guard against silently loading a checkpoint with a different channel
    # layout. Same total width with different semantics is possible (e.g. the
    # "channel" layout with DINO and the "embedded" layout without DINO
    # both have 94 + 1024 channels), so also require the motion-history
    # embedder presence to match the model's conditioning style.
    ckpt_in_channels = None
    if "x_embedder.weight" in remapped:
        ckpt_in_channels = remapped["x_embedder.weight"].shape[1]
    if ckpt_in_channels is not None and ckpt_in_channels != model.in_channels:
        raise RuntimeError(
            f"Checkpoint input width ({ckpt_in_channels} channels) does not match "
            f"the model ({model.in_channels} channels, "
            f"motion_history_conditioning={model.motion_history_conditioning!r}). "
            f"Channel-layout checkpoints ([XVel|YVel|Occ|DINO], used by the "
            f"released checkpoints) need model_cfg.motion_history_conditioning: "
            f"channel; embedded-layout checkpoints need 'embedded'. Also verify "
            f"dino_feature_dim/horizon."
        )
    ckpt_has_mh_embedder = any(k.startswith("motion_history_embedder.")
                               for k in remapped)
    model_is_embedded = getattr(model, "motion_history_conditioning", "embedded") == "embedded"
    if ckpt_has_mh_embedder and not model_is_embedded:
        raise RuntimeError(
            "Checkpoint contains motion_history_embedder keys (embedded layout) "
            "but the model uses motion_history_conditioning='channel'. "
            "Set model_cfg.motion_history_conditioning: embedded."
        )
    if not ckpt_has_mh_embedder and model_is_embedded and ckpt_in_channels is not None:
        raise RuntimeError(
            "Checkpoint has no motion_history_embedder keys, which indicates a "
            "channel-layout checkpoint, but the model uses "
            "motion_history_conditioning='embedded'. Even when channel counts "
            "coincide, the channel semantics differ. "
            "Set model_cfg.motion_history_conditioning: channel."
        )

    missing, unexpected = model.load_state_dict(remapped, strict=strict)
    if missing:
        logger.warning("Missing keys when loading checkpoint: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys when loading checkpoint: %s", unexpected)

    metadata = {}
    if isinstance(checkpoint, dict):
        for meta_key in ("dino_scale_factor", "occlusion_scaling", "epoch",
                         "best_loss", "best_epoch"):
            if meta_key in checkpoint:
                metadata[meta_key] = checkpoint[meta_key]
        metadata["has_ema"] = (
            "ema_state_dict" in checkpoint
            and "shadow_params" in checkpoint.get("ema_state_dict", {})
        )

    return metadata


def load_ema_weights(model: DiT_MammalNet_SingleImage, path) -> bool:
    """Copy the checkpoint's EMA shadow weights into the model's parameters.

    Checkpoints store ``model_state_dict`` (live weights) plus
    ``ema_state_dict.shadow_params``: a name-less list of EMA-averaged tensors,
    one per trainable parameter, in the model's parameter-iteration order
    (the EMAModel convention). Evaluations were historically run with EMA
    weights swapped in, so inference should prefer them.

    Call AFTER load_checkpoint() (non-trainable params keep their loaded
    values, which is correct: EMA never tracked them).

    Args:
        model: target model (already loaded with live weights).
        path: checkpoint path, or an already-loaded checkpoint dict.

    Returns:
        True if EMA weights were applied, False if the checkpoint has no EMA.
    """
    checkpoint = _load_checkpoint_dict(path)
    if not (isinstance(checkpoint, dict)
            and "ema_state_dict" in checkpoint
            and "shadow_params" in checkpoint["ema_state_dict"]):
        logger.warning("No EMA shadow in checkpoint — keeping live weights.")
        return False

    shadow = checkpoint["ema_state_dict"]["shadow_params"]
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]

    if len(shadow) != len(trainable):
        raise RuntimeError(
            f"EMA shadow has {len(shadow)} entries but the model has "
            f"{len(trainable)} trainable parameters — parameter sets do not "
            f"match. The checkpoint was not saved by this codebase's "
            f"save_checkpoint().")

    with torch.no_grad():
        for (name, p), s in zip(trainable, shadow):
            if p.shape != s.shape:
                raise RuntimeError(
                    f"EMA shadow shape mismatch at {name!r}: "
                    f"param={tuple(p.shape)} shadow={tuple(s.shape)} — "
                    f"alignment is wrong, refusing to load EMA weights.")
            p.data.copy_(s.to(device=p.device, dtype=p.dtype))

    logger.info("EMA: applied shadow weights to %d parameter tensors.", len(trainable))
    return True


def save_checkpoint(model: DiT_MammalNet_SingleImage, path: str,
                    optimizer=None, scheduler=None, epoch=None,
                    best_loss=None, best_epoch=None, ema=None,
                    dino_scale_factor=None, occlusion_scaling=None):
    """Save model checkpoint with optional training state."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
    }
    if dino_scale_factor is not None:
        checkpoint["dino_scale_factor"] = dino_scale_factor
    if occlusion_scaling is not None:
        checkpoint["occlusion_scaling"] = occlusion_scaling
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if best_loss is not None:
        checkpoint["best_loss"] = best_loss
    if best_epoch is not None:
        checkpoint["best_epoch"] = best_epoch
    if ema is not None:
        checkpoint["ema_state_dict"] = ema.state_dict()
    torch.save(checkpoint, path)


# ======================================================================
# Model function factory (for diffusion sampling loops)
# ======================================================================

def create_model_fn(model, y, model_dtype, vel_disp_embedding=None,
                    initial_pos_for_encoding=None, use_initial_pos_encoding=False,
                    vel_disp_tokens=None, attn_mask=None, track_type_emb=None):
    """Create a closure compatible with GaussianDiffusion's sampling loops.

    Returns:
        callable(x_t, timesteps, **kwargs) -> model prediction
    """
    initial_pos = initial_pos_for_encoding if use_initial_pos_encoding else None

    def model_fn(x_t, timesteps, **model_kwargs):
        if x_t.dtype != model_dtype:
            x_t = x_t.to(model_dtype)
        if timesteps.dtype != torch.int64:
            timesteps = timesteps.long()
        return model(x_t, timesteps, y,
                     vel_disp_embedding=vel_disp_embedding,
                     initial_pos=initial_pos,
                     vel_disp_tokens=vel_disp_tokens,
                     attn_mask=attn_mask,
                     track_type_emb=track_type_emb)

    return model_fn


# ======================================================================
# Training
# ======================================================================

def _make_attn_mask(point_mask):
    """Convert point_mask (B, N) to key-padding attention mask (B, 1, 1, N)."""
    if point_mask is None:
        return None
    return point_mask[:, None, None, :]


def prepare_conditioning(model, vid, track, total_displacement,
                         visibility, dino_extractor, dino_scale_factor,
                         training=True):
    """Prepare model inputs and conditioning for a training or inference step.

    Converts tracks to model format, computes displacement conditioning,
    and handles dropout during training.

    Returns:
        x: (B, C, N) model input tensor
        y: (B, C, H, W) conditioning image
        vel_disp_embedding: (B, D) or None (AdaLN mode)
        vel_disp_tokens: (B, K, D) or None (token mode)
        token_attn_mask: (B, 1, S, S) or None (token mode)
        track_type_emb: (1, D) or None (token mode)
        initial_pos_for_encoding: (B, N, 2) or None
        initial_pos_mode2: (B, 1, N, 2) or None
        new_dino_scale_factor: float or None
        channel_layout: dict
    """
    x, y, initial_pos_for_encoding, initial_pos_mode2, new_dino_scale_factor = (
        tracks_to_model_format(
            track=track, vid=vid, horizon=model.horizon,
            diffuse_on_velocity=model.diffuse_on_velocity,
            num_point_cond=MAX_MOTION_HISTORY_COND,
            velocity_scale=getattr(model, '_velocity_scale', 1.0),
            handle_occlusions=model.handle_occlusions,
            occlusion_scaling=getattr(model, '_occlusion_scaling', 1.0),
            use_dino_features=(model.dino_feature_dim > 0),
            dino_feature_dim=model.dino_feature_dim,
            dino_extractor=dino_extractor,
            dino_scale_factor=dino_scale_factor,
            img_size=model.img_size,
            visibility=visibility,
            motion_history_embedder=model.motion_history_embedder,
            motion_history_dim=model.motion_history_dim,
            motion_history_conditioning=model.motion_history_conditioning,
        )
    )

    vel_disp_embedding = None
    vel_disp_tokens = None
    token_attn_mask = None
    track_type_emb = None

    if model.displacement_embedder is not None:
        dropout_mask = None
        if model.vel_disp_dropout and training:
            dropout_mask = model.sample_vel_disp_dropout_mask(x.shape[0], x.device)

        if model.vel_disp_conditioning_mode == "token":
            vel_disp_tokens, token_attn_mask, track_type_emb = (
                model.embed_velocity_displacement_as_tokens(
                    total_displacement, dropout_mask=dropout_mask,
                )
            )
        else:
            vel_disp_embedding = model.embed_velocity_displacement(
                total_displacement, dropout_mask=dropout_mask,
            )

    channel_layout = get_channel_layout(
        model.horizon, MAX_MOTION_HISTORY_COND, model.diffuse_on_velocity,
        model.handle_occlusions, model.dino_feature_dim > 0, model.dino_feature_dim,
        motion_history_dim=model.motion_history_dim,
        motion_history_conditioning=model.motion_history_conditioning,
    )

    return (x, y, vel_disp_embedding, vel_disp_tokens, token_attn_mask,
            track_type_emb, initial_pos_for_encoding, initial_pos_mode2,
            new_dino_scale_factor, channel_layout)


def apply_motion_history_dropout(x, K, channel_layout, motion_history_dim=None):
    """Zero out motion-history conditioning channels not available at level K.

    Only meaningful for the "embedded" layout; a no-op for "channel" (where
    history level K is expressed by fixing fewer leading channels instead).
    """
    if channel_layout["motion_history_conditioning"] != "embedded":
        return x

    c_occ_start = channel_layout["c_occ_start"]
    c_vel_starts = channel_layout["c_vel_starts"]
    mhd = channel_layout["motion_history_dim"]

    if K < MAX_MOTION_HISTORY_COND:
        x[:, c_occ_start + K : c_occ_start + MAX_MOTION_HISTORY_COND, :] = 0

    for ci in range(len(c_vel_starts)):
        needed_K = ci + 2
        if K < needed_K:
            start = c_vel_starts[ci]
            x[:, start : start + mhd, :] = 0

    return x


def train_step(model, diffusion, batch, dino_extractor, cfg):
    """Execute a single training step.

    Args:
        model: DiT_MammalNet_SingleImage instance.
        diffusion: GaussianDiffusion instance.
        batch: dict with keys: video (B,T,C,H,W), track (B,T,N,2),
               total_displacement (B,2),
               visibility (B,T,N) optional, point_mask (B,N) optional.
        dino_extractor: DINOFeatureExtractor or None.
        cfg: dict with keys: num_point_cond, velocity_scale, occlusion_scaling,
             dino_scale_factor, track_error_metric, simultaneous_steps.

    Returns:
        dict with 'loss' (scalar tensor), 'metrics' (dict of floats),
        'dino_scale_factor' (possibly updated float).
    """
    vid = batch["video"]
    track = batch["track"]
    total_displacement = batch.get("total_displacement", None)
    visibility = batch.get("visibility", None)
    point_mask = batch.get("point_mask", None)

    num_point_cond = cfg.get("num_point_cond", 1)
    velocity_scale = cfg.get("velocity_scale", 1.0)
    occlusion_scaling = cfg.get("occlusion_scaling", 1.0)
    dino_scale_factor = cfg.get("dino_scale_factor", None)
    track_error_metric = cfg.get("track_error_metric", "l2")
    simultaneous_steps = cfg.get("simultaneous_steps", 2)

    # Temporarily set scale attributes the coordinate utils expect
    model._velocity_scale = velocity_scale
    model._occlusion_scaling = occlusion_scaling

    (x, y, vel_disp_embedding, vel_disp_tokens, token_attn_mask,
     track_type_emb, initial_pos_for_encoding, initial_pos_mode2,
     new_dino_scale_factor, channel_layout) = prepare_conditioning(
        model, vid, track, total_displacement,
        visibility, dino_extractor, dino_scale_factor, training=True,
    )

    sc = channel_layout["supervised_channels"]
    attn_mask = _make_attn_mask(point_mask)

    K = max(1, simultaneous_steps) if model.training else 1
    model_dtype = next(model.parameters()).dtype

    if K > 1:
        x_expanded = x.repeat_interleave(K, dim=0)
        y_expanded = y.repeat_interleave(K, dim=0)
        pm_expanded = point_mask.repeat_interleave(K, dim=0) if point_mask is not None else None
        vel_disp_expanded = vel_disp_embedding.repeat_interleave(K, dim=0) if vel_disp_embedding is not None else None
        init_pos_expanded = initial_pos_for_encoding.repeat_interleave(K, dim=0) if initial_pos_for_encoding is not None else None
        attn_mask_expanded = attn_mask.repeat_interleave(K, dim=0) if attn_mask is not None else None
        vel_disp_tokens_expanded = vel_disp_tokens.repeat_interleave(K, dim=0) if vel_disp_tokens is not None else None
        token_attn_mask_expanded = token_attn_mask.repeat_interleave(K, dim=0) if token_attn_mask is not None else None
    else:
        x_expanded, y_expanded, pm_expanded = x, y, point_mask
        vel_disp_expanded = vel_disp_embedding
        init_pos_expanded = initial_pos_for_encoding
        attn_mask_expanded = attn_mask
        vel_disp_tokens_expanded = vel_disp_tokens
        token_attn_mask_expanded = token_attn_mask

    effective_attn_mask = token_attn_mask_expanded if token_attn_mask_expanded is not None else attn_mask_expanded
    model_fn = create_model_fn(
        model,
        y_expanded if y_expanded.dtype == model_dtype else y_expanded.to(model_dtype),
        model_dtype,
        vel_disp_embedding=vel_disp_expanded,
        initial_pos_for_encoding=init_pos_expanded,
        use_initial_pos_encoding=model.use_initial_pos_encoding,
        attn_mask=effective_attn_mask,
        vel_disp_tokens=vel_disp_tokens_expanded,
        track_type_emb=track_type_emb,
    )

    loss, model_output, metrics = _compute_diffusion_loss(
        model, diffusion, x_expanded, y_expanded, model_fn, channel_layout,
        num_point_cond, pm_expanded, track_error_metric,
    )

    return {
        "loss": loss,
        "metrics": metrics,
        "dino_scale_factor": new_dino_scale_factor if new_dino_scale_factor is not None else dino_scale_factor,
    }


def _compute_diffusion_loss(model, diffusion, x, y, model_fn, layout,
                            num_point_cond, point_mask, track_error_metric):
    """Compute the core diffusion training loss.

    "embedded" layout: channels >= supervised_channels are conditioning and
    never noised; history dropout zeroes unused C-blocks.

    "channel" layout (released checkpoints): history conditioning fixes the leading
    num_point_cond-1 velocity channels per axis and num_point_cond occlusion
    channels to GT; DINO channels are never noised. The loss supervises all
    coord+occ channels (including the conditioned leading ones), matching the
    original training code.
    """
    model_dtype = next(model.parameters()).dtype
    if x.dtype != model_dtype:
        x = x.to(model_dtype)
    if y.dtype != model_dtype:
        y = y.to(model_dtype)

    B = x.shape[0]
    device = x.device
    sc = layout["supervised_channels"]
    is_embedded = layout["motion_history_conditioning"] == "embedded"

    t = torch.randint(0, diffusion.num_timesteps, (B,), device=device)
    noise = torch.randn_like(x)

    if point_mask is not None:
        noise = noise * point_mask.float().unsqueeze(1)

    if is_embedded:
        noise[:, sc:, :] = 0
        x_t = diffusion.q_sample(x, t, noise=noise)
        x_t[:, sc:, :] = x[:, sc:, :].clone()

        if num_point_cond < MAX_MOTION_HISTORY_COND:
            apply_motion_history_dropout(x_t, num_point_cond, layout)
    else:
        # Channel-layout conditioning: DINO never noised, leading channels GT.
        if layout["dino_start"] is not None:
            noise[:, layout["dino_start"]:, :] = 0
        x_t = diffusion.q_sample(x, t, noise=noise)

        n_vel_cond = max(0, num_point_cond - 1) if layout["diffuse_on_velocity"] else num_point_cond
        y_start = layout["y_start"]
        x_t[:, :n_vel_cond, :] = x[:, :n_vel_cond, :].clone()
        x_t[:, y_start:y_start + n_vel_cond, :] = x[:, y_start:y_start + n_vel_cond, :].clone()
        if layout["occ_start"] is not None:
            occ_start = layout["occ_start"]
            x_t[:, occ_start:occ_start + num_point_cond, :] = \
                x[:, occ_start:occ_start + num_point_cond, :].clone()
        if layout["dino_start"] is not None:
            x_t[:, layout["dino_start"]:, :] = x[:, layout["dino_start"]:, :].clone()

    model_output = model_fn(x_t, t)

    model_output_supervised = model_output[:, :sc, :]

    if diffusion.model_mean_type.name == "START_X":
        target = x[:, :sc, :]
    elif diffusion.model_mean_type.name == "EPSILON":
        target = noise[:, :sc, :]
    elif diffusion.model_mean_type.name == "PREVIOUS_X":
        x_sup = x[:, :sc, :]
        x_t_sup = x_t[:, :sc, :]
        target = diffusion.q_posterior_mean_variance(x_start=x_sup, x_t=x_t_sup, t=t)[0]
    else:
        raise ValueError(f"Unknown model_mean_type: {diffusion.model_mean_type}")

    if track_error_metric == "l1":
        element_loss = torch.abs(target - model_output_supervised)
    elif track_error_metric == "l2":
        element_loss = (target - model_output_supervised) ** 2
    else:
        raise ValueError(f"Unknown track_error_metric: {track_error_metric}")

    if point_mask is not None:
        B_mask, N = point_mask.shape
        C = element_loss.shape[1]
        point_mask_flat = point_mask.unsqueeze(1).expand(B_mask, C, N)
        masked_loss = element_loss * point_mask_flat.float()
        num_real_points = point_mask_flat.sum(dim=1, keepdim=True).clamp(min=1)
        per_point_avg = masked_loss.sum(dim=1) / num_real_points.squeeze(1)
        loss = per_point_avg.mean()
    else:
        loss = element_loss.mean()

    metrics = {
        "loss": loss.item(),
        "track_loss": loss.item(),
    }

    return loss, model_output, metrics


# ======================================================================
# Inference
# ======================================================================

def predict(model, diffusion, batch, dino_extractor, cfg):
    """Run diffusion sampling to predict future tracks.

    Args:
        model: DiT_MammalNet_SingleImage instance.
        diffusion: GaussianDiffusion instance.
        batch: dict with keys: video (B,T,C,H,W), track (B,T,N,2),
               total_displacement (B,2),
               visibility (B,T,N) optional, point_mask (B,N) optional.
        dino_extractor: DINOFeatureExtractor or None.
        cfg: dict with keys: num_point_cond, velocity_scale, occlusion_scaling,
             dino_scale_factor, clip_denoised_multiplier,
             use_ddim, ddim_eta, ddim_timesteps,
             use_null_displacement, motion_history_K.

    Returns:
        dict with 'pred_tracks' (B,T,N,2), 'gt_tracks' (B,T,N,2),
        and optionally 'pred_visibility' (B,T,N).
    """
    vid = batch["video"]
    track = batch["track"]
    total_displacement = batch.get("total_displacement", None)
    visibility = batch.get("visibility", None)
    point_mask = batch.get("point_mask", None)

    num_point_cond = cfg.get("num_point_cond", 1)
    velocity_scale = cfg.get("velocity_scale", 1.0)
    occlusion_scaling = cfg.get("occlusion_scaling", 1.0)
    dino_scale_factor = cfg.get("dino_scale_factor", None)
    clip_denoised_multiplier = cfg.get("clip_denoised_multiplier", 3.0)
    use_ddim = cfg.get("use_ddim", False)
    ddim_eta = cfg.get("ddim_eta", 0.0)
    ddim_timesteps = cfg.get("ddim_timesteps", 50)
    use_null_displacement = cfg.get("use_null_displacement", False)
    motion_history_K = cfg.get("motion_history_K", None)
    noise_seed = cfg.get("noise_seed", None)

    model._velocity_scale = velocity_scale
    model._occlusion_scaling = occlusion_scaling

    (x_gt, y, vel_disp_embedding, vel_disp_tokens, token_attn_mask,
     track_type_emb, initial_pos_for_encoding, initial_pos_mode2,
     new_dino_scale_factor, channel_layout) = prepare_conditioning(
        model, vid, track, total_displacement,
        visibility, dino_extractor, dino_scale_factor, training=False,
    )

    # Override conditioning for null-displacement inference
    if use_null_displacement and model.displacement_embedder is not None:
        if model.vel_disp_conditioning_mode == "token":
            vel_disp_tokens, token_attn_mask, track_type_emb = (
                model.embed_velocity_displacement_as_tokens(
                    total_displacement,
                    use_null_displacement=True,
                )
            )
            vel_disp_embedding = None
        else:
            vel_disp_embedding = model.embed_velocity_displacement(
                total_displacement,
                use_null_displacement=True,
            )
            vel_disp_tokens = None
            token_attn_mask = None
            track_type_emb = None

    sc = channel_layout["supervised_channels"]
    is_embedded = channel_layout["motion_history_conditioning"] == "embedded"
    model_dtype = next(model.parameters()).dtype
    if x_gt.dtype != model_dtype:
        x_gt = x_gt.to(model_dtype)
    if y.dtype != model_dtype:
        y = y.to(model_dtype)

    device = x_gt.device
    x_shape = x_gt.shape

    if noise_seed is not None:
        generator = torch.Generator(device=device).manual_seed(noise_seed)
        x_pred = torch.randn(x_shape, device=device, dtype=model_dtype, generator=generator)
    else:
        x_pred = torch.randn(x_shape, device=device, dtype=model_dtype)

    K = motion_history_K if motion_history_K is not None else num_point_cond

    if is_embedded:
        fixed_channels_start = sc
        fixed_channels_mask = None
        x_pred[:, sc:, :] = x_gt[:, sc:, :].clone()
        if K < MAX_MOTION_HISTORY_COND:
            apply_motion_history_dropout(x_pred, K, channel_layout)

        img_for_loop = x_gt
        if K < MAX_MOTION_HISTORY_COND:
            img_for_loop = x_gt.clone()
            apply_motion_history_dropout(img_for_loop, K, channel_layout)
    else:
        # Channel layout: fix the K leading conditioning channels + DINO
        # at every denoising step via a non-contiguous channel mask.
        fixed_channels_start = None
        fixed_channels_mask = build_fixed_channels_mask(
            channel_layout, num_point_cond=K, device=device)
        x_pred[:, fixed_channels_mask, :] = x_gt[:, fixed_channels_mask, :].clone()
        img_for_loop = x_gt

    attn_mask = _make_attn_mask(point_mask)
    effective_attn_mask = token_attn_mask if token_attn_mask is not None else attn_mask

    model_fn = create_model_fn(
        model, y, model_dtype,
        vel_disp_embedding=vel_disp_embedding,
        initial_pos_for_encoding=initial_pos_for_encoding,
        use_initial_pos_encoding=model.use_initial_pos_encoding,
        vel_disp_tokens=vel_disp_tokens,
        attn_mask=effective_attn_mask,
        track_type_emb=track_type_emb,
    )

    sample_loop_fn = diffusion.ddim_sample_loop if use_ddim else diffusion.p_sample_loop

    sampling_kwargs = {
        "model": model_fn,
        "shape": x_shape,
        "noise": x_pred,
        "device": device,
        "progress": False,
        "clip_denoised_range": clip_denoised_multiplier,
        "fixed_channels_start": fixed_channels_start,
        "fixed_channels_mask": fixed_channels_mask,
        "img": img_for_loop,
        "motion_history_K": K,
        "point_mask": point_mask,
        "model_kwargs": {},
    }
    if use_ddim:
        sampling_kwargs["eta"] = ddim_eta
        sampling_kwargs["ddim_timesteps"] = ddim_timesteps

    x_sampled = sample_loop_fn(**sampling_kwargs)

    ip = initial_pos_mode2
    pred_result = model_to_tracks_format(
        x_dit=x_sampled, horizon=model.horizon,
        diffuse_on_velocity=model.diffuse_on_velocity,
        num_point_cond=num_point_cond,
        velocity_scale=velocity_scale,
        handle_occlusions=model.handle_occlusions,
        occlusion_scaling=occlusion_scaling,
        use_dino_features=(model.dino_feature_dim > 0),
        dino_feature_dim=model.dino_feature_dim,
        initial_pos_override=ip,
        motion_history_dim=model.motion_history_dim,
    )
    gt_result = model_to_tracks_format(
        x_dit=x_gt, horizon=model.horizon,
        diffuse_on_velocity=model.diffuse_on_velocity,
        num_point_cond=num_point_cond,
        velocity_scale=velocity_scale,
        handle_occlusions=model.handle_occlusions,
        occlusion_scaling=occlusion_scaling,
        use_dino_features=(model.dino_feature_dim > 0),
        dino_feature_dim=model.dino_feature_dim,
        initial_pos_override=ip,
        motion_history_dim=model.motion_history_dim,
    )

    result = {}
    if isinstance(pred_result, tuple):
        result["pred_tracks"] = pred_result[0]
        if len(pred_result) > 2:
            result["pred_visibility"] = pred_result[2]
    else:
        result["pred_tracks"] = pred_result

    result["gt_tracks"] = gt_result[0] if isinstance(gt_result, tuple) else gt_result
    result["pred_video"] = vid

    return result
