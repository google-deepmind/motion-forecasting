"""
Training utility functions: seeding, optimizer helpers, config building.
"""

import random
import numpy as np
import torch


def _seed_worker(worker_id):
    """Seed each dataloader worker from PyTorch's assigned seed (reproducible augmentations)."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _move_optimizer_state_to_device(optimizer, device):
    """Move all optimizer state tensors to the given device.
    Required after resume: checkpoint is loaded with map_location='cpu', then fabric.setup()
    moves the model to GPU but the optimizer's internal buffers (e.g. Adam exp_avg, exp_avg_sq)
    stay on CPU, causing device mismatch in optimizer.step()."""
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device, non_blocking=True)


def _compute_avg_speed(tracks, visibility, point_mask):
    """Average per-frame speed (L2 of consecutive-frame deltas, bbox-normalized).

    Args:
        tracks:     (B, T, N, 2) tensor
        visibility: (B, T, N) tensor
        point_mask: (B, N) tensor

    Returns:
        (B,) tensor of avg speed per sample.
    """
    deltas = tracks[:, 1:, :, :] - tracks[:, :-1, :, :]       # (B, T-1, N, 2)
    speeds = torch.sqrt((deltas ** 2).sum(dim=-1))             # (B, T-1, N)
    vis_mask = visibility[:, 1:, :] * visibility[:, :-1, :]    # (B, T-1, N)
    valid = vis_mask * point_mask.unsqueeze(1).float()         # (B, T-1, N)
    denom = valid.sum(dim=(1, 2)).clamp(min=1)                 # (B,)
    return (speeds * valid).sum(dim=(1, 2)) / denom            # (B,)


def _assign_motion_bucket(avg_speed, low_thresh, high_thresh):
    """Assign a motion bucket label based on avg speed and thresholds."""
    if avg_speed < low_thresh:
        return 'low_motion'
    elif avg_speed < high_thresh:
        return 'mid_motion'
    else:
        return 'high_motion'


def build_model_cfg(cfg):
    """Build the config dict for model creation from a training config.

    Returns a flat dict usable by pipeline.create_model(), pipeline.create_diffusion(),
    and pipeline.train_step()/predict() cfg arguments.
    """
    from motion_forecasting.model.dino_feature_extractor import DINOV3_VITL_FEATURE_DIM

    use_dino = cfg.model_cfg.get('use_dino_features', True)
    return {
        'model_name': cfg.model_cfg.get('dit_model_name', 'DiT-L-MammalNet'),
        'horizon': cfg.model_cfg.get('horizon', 16),
        'num_points': cfg.model_cfg.get('num_points', 32),
        'img_size': cfg.model_cfg.get('img_size', 256),
        'diffusion_steps': cfg.model_cfg.get('diffusion_steps', 1000),
        'noise_schedule': cfg.model_cfg.get('noise_schedule', 'linear'),
        'beta_start': cfg.model_cfg.get('beta_start', 0.0001),
        'beta_end': cfg.model_cfg.get('beta_end', 0.02),
        'model_mean_type': cfg.model_cfg.get('model_mean_type', 'epsilon'),
        'model_var_type': cfg.model_cfg.get('model_var_type', 'fixed_small'),
        'loss_type': cfg.model_cfg.get('loss_type', 'mse'),
        'track_error_metric': cfg.model_cfg.get('track_error_metric', 'l2'),
        'num_point_cond': cfg.model_cfg.get('num_point_cond', 1),
        'motion_history_dim': cfg.model_cfg.get('motion_history_dim', 340),
        'motion_history_scale': cfg.model_cfg.get('motion_history_scale', 0.19092),
        'condition_on_displacement': cfg.model_cfg.get('condition_on_displacement', False),
        'velocity_conditioning_type': cfg.model_cfg.get('velocity_conditioning_type', 'linear'),
        'vel_disp_dropout': cfg.model_cfg.get('vel_disp_dropout', False),
        'single_cond_dropout_prob': cfg.model_cfg.get('single_cond_dropout_prob', 0.3),
        'null_embedding_type': cfg.model_cfg.get('null_embedding_type', 'learned'),
        'clip_denoised_multiplier': cfg.model_cfg.get('clip_denoised_multiplier', 3.0),
        'use_dino_features': use_dino,
        # CRITICAL: the model's input width depends on dino_feature_dim; if this
        # is omitted the model silently builds without DINO channels.
        'dino_feature_dim': cfg.model_cfg.get(
            'dino_feature_dim', DINOV3_VITL_FEATURE_DIM if use_dino else 0),
        'dino_layer': cfg.model_cfg.get('dino_layer', 23),
        'dino_scale_factor': cfg.model_cfg.get('dino_scale_factor', None),
        'handle_occlusions': cfg.model_cfg.get('handle_occlusions', False),
        'occlusion_scaling': cfg.model_cfg.get('occlusion_scaling', 1.0),
        'diffuse_on_velocity': cfg.model_cfg.get('diffuse_on_velocity', False),
        'velocity_scale': cfg.model_cfg.get('velocity_scale', 16.0),
        'use_initial_pos_encoding': cfg.model_cfg.get('use_initial_pos_encoding', False),
        'use_image_conditioning': cfg.model_cfg.get('use_image_conditioning', True),
        'vel_disp_conditioning_mode': cfg.model_cfg.get('vel_disp_conditioning_mode', 'adaln'),
        'vel_disp_token_type_embeddings': cfg.model_cfg.get('vel_disp_token_type_embeddings', False),
        'simultaneous_steps': cfg.model_cfg.get('simultaneous_steps', 2),
        # "embedded" = sin-cos conditioning blocks (training default); "channel" =
        # layout used by the released checkpoints (conditioning via leading channels).
        'motion_history_conditioning': cfg.model_cfg.get('motion_history_conditioning', 'embedded'),
    }

