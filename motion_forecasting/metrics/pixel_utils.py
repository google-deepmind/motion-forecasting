"""
Canonical pixel-space conversion utilities for track metrics.

Provides functions to convert tracks between [0,1] normalized space and
original video pixel space, handling all bbox formats consistently.

Also provides the DiT round-trip conversion simulation for matching
training-time precision.
"""

import torch
import numpy as np
from typing import Dict, Optional


def tracks_to_pixel_space(
    gt_tracks: torch.Tensor,
    pred_tracks: torch.Tensor,
    gt_vis: torch.Tensor = None,
    img_size: int = 256,
    normalization: str = None,
    bbox: dict = None,
    frame_start: int = None,
    orig_width: int = None,
    orig_height: int = None
) -> tuple:
    """
    Convert tracks from [0,1] normalized space to original video pixel space.

    Handles all bbox formats:
    - Training format: bbox[frame_start]['padded_square'] (keyed by frame number)
    - Extracted pkl format: bbox['padded_square'] (already extracted for frame)

    Args:
        gt_tracks: (B, T, N, 2) ground truth tracks in [0,1] space
        pred_tracks: (B, T, N, 2) predicted tracks in [0,1] space
        gt_vis: (B, T, N) visibility mask, or None
        img_size: Image size used for training (default 256, fallback)
        normalization: 'first_bbox', 'per_frame_bbox', or 'whole_image_dim'
        bbox: Bounding box info dict
        frame_start: Starting frame index
        orig_width: Original image width
        orig_height: Original image height

    Returns:
        (gt_tracks_pixel, pred_tracks_pixel, gt_vis) -- all cloned/safe to modify
    """
    # Extract scalar values from DataLoader-collated lists/tensors
    if isinstance(normalization, (list, tuple)):
        normalization = normalization[0]
    if isinstance(frame_start, (list, tuple)):
        frame_start = frame_start[0]
    if hasattr(frame_start, 'item'):
        frame_start = frame_start.item()

    gt_tracks_pixel = gt_tracks.clone()
    pred_tracks_pixel = pred_tracks.clone()

    if normalization == "first_bbox" and bbox is not None:
        bbox_padded = _extract_padded_bbox(bbox, frame_start)
        if bbox_padded is not None:
            x, y, w, h = _bbox_to_tensors(bbox_padded, gt_tracks.device, gt_tracks.dtype)
            gt_tracks_pixel[:, :, :, 0] = gt_tracks_pixel[:, :, :, 0] * w + x
            gt_tracks_pixel[:, :, :, 1] = gt_tracks_pixel[:, :, :, 1] * h + y
            pred_tracks_pixel[:, :, :, 0] = pred_tracks_pixel[:, :, :, 0] * w + x
            pred_tracks_pixel[:, :, :, 1] = pred_tracks_pixel[:, :, :, 1] * h + y
        else:
            print(f"Warning: Could not extract bbox, using img_size={img_size} fallback")
            gt_tracks_pixel = gt_tracks_pixel * img_size
            pred_tracks_pixel = pred_tracks_pixel * img_size

    elif normalization == "per_frame_bbox" and bbox is not None:
        # Try to use per-frame bboxes; fall back to first_bbox style
        bbox_padded = _extract_padded_bbox(bbox, frame_start)
        if bbox_padded is not None:
            x, y, w, h = _bbox_to_tensors(bbox_padded, gt_tracks.device, gt_tracks.dtype)
            gt_tracks_pixel[:, :, :, 0] = gt_tracks_pixel[:, :, :, 0] * w + x
            gt_tracks_pixel[:, :, :, 1] = gt_tracks_pixel[:, :, :, 1] * h + y
            pred_tracks_pixel[:, :, :, 0] = pred_tracks_pixel[:, :, :, 0] * w + x
            pred_tracks_pixel[:, :, :, 1] = pred_tracks_pixel[:, :, :, 1] * h + y
        else:
            # Try per-frame iteration (training format)
            B, T, N, _ = gt_tracks.shape
            applied = False
            for i in range(T):
                frame_idx = (frame_start + i) if frame_start is not None else i
                if isinstance(bbox, dict) and frame_idx in bbox:
                    padded = bbox[frame_idx].get('padded_square')
                    if padded is not None:
                        xi, yi, wi, hi = _bbox_to_tensors(padded, gt_tracks.device, gt_tracks.dtype)
                        gt_tracks_pixel[:, i, :, 0] = gt_tracks_pixel[:, i, :, 0] * wi + xi
                        gt_tracks_pixel[:, i, :, 1] = gt_tracks_pixel[:, i, :, 1] * hi + yi
                        pred_tracks_pixel[:, i, :, 0] = pred_tracks_pixel[:, i, :, 0] * wi + xi
                        pred_tracks_pixel[:, i, :, 1] = pred_tracks_pixel[:, i, :, 1] * hi + yi
                        applied = True
            if not applied:
                print(f"Warning: Could not extract bbox for per_frame_bbox, using img_size={img_size} fallback")
                gt_tracks_pixel = gt_tracks.clone() * img_size
                pred_tracks_pixel = pred_tracks.clone() * img_size

    elif normalization == "whole_image_dim" and orig_width is not None and orig_height is not None:
        scale = torch.tensor([orig_width, orig_height], device=gt_tracks.device, dtype=gt_tracks.dtype)
        gt_tracks_pixel = gt_tracks_pixel * scale
        pred_tracks_pixel = pred_tracks_pixel * scale

    elif normalization == "fixed_256":
        gt_tracks_pixel = gt_tracks_pixel * 256
        pred_tracks_pixel = pred_tracks_pixel * 256

    else:
        # Fallback
        gt_tracks_pixel = gt_tracks_pixel * img_size
        pred_tracks_pixel = pred_tracks_pixel * img_size

    return gt_tracks_pixel, pred_tracks_pixel, gt_vis


def _extract_padded_bbox(bbox: dict, frame_start=None) -> Optional[dict]:
    """
    Extract the padded_square bbox dict, handling both formats:
    - Extracted pkl format: bbox['padded_square']
    - Training format: bbox[frame_start]['padded_square']
    """
    if bbox is None:
        return None

    # Try extracted pkl format first
    if 'padded_square' in bbox:
        return bbox['padded_square']

    # Try training format (keyed by frame number)
    if frame_start is not None:
        frame_key = frame_start
        if hasattr(frame_start, 'item'):
            frame_key = frame_start.item()
        frame_key = int(frame_key)
        if frame_key in bbox and isinstance(bbox[frame_key], dict):
            return bbox[frame_key].get('padded_square')

    return None


def _bbox_to_tensors(bbox_padded: dict, device, dtype) -> tuple:
    """Extract x, y, w, h from bbox_padded dict and convert to tensors."""
    x = bbox_padded['x']
    y = bbox_padded['y']
    w = bbox_padded['w']
    h = bbox_padded['h']

    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, device=device, dtype=dtype)
        y = torch.tensor(y, device=device, dtype=dtype)
        w = torch.tensor(w, device=device, dtype=dtype)
        h = torch.tensor(h, device=device, dtype=dtype)
    else:
        x = x.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)
        w = w.to(device=device, dtype=dtype)
        h = h.to(device=device, dtype=dtype)

    return x, y, w, h


def apply_dit_roundtrip_conversion(tracks: np.ndarray, horizon: int = None,
                                    use_bfloat16: bool = True) -> np.ndarray:
    """
    Apply the same coordinate round-trip conversion that happens during training.

    During training, gt_tracks go through:
    1. tracks_to_model_format: (B, T, N, 2) -> (B, 2*T, N), [0,1] -> [-1,1]
    2. Dtype conversion to model dtype (bfloat16 with mix_precision)
    3. model_to_tracks_format: (B, 2*T, N) -> (B, T, N, 2), [-1,1] -> [0,1]

    This function replicates that exact transformation to ensure metrics match.

    Args:
        tracks: (T, N, 2) tracks in [0, 1] space (no batch dimension)
        horizon: expected horizon (T). If None, uses tracks.shape[0]
        use_bfloat16: If True, simulate the bfloat16 conversion that happens during
                      mixed precision training.

    Returns:
        tracks_roundtrip: (T, N, 2) tracks after round-trip conversion
    """
    T, N, _ = tracks.shape
    if horizon is None:
        horizon = T

    tracks_batched = tracks[np.newaxis, ...]  # (1, T, N, 2)
    tracks_t = torch.from_numpy(tracks_batched).float()

    # Track to model: [0, 1] -> [-1, 1], reshape
    track_diffusion = tracks_t * 2.0 - 1.0
    x_dit = track_diffusion.permute(0, 3, 1, 2).contiguous()  # (B, 2, T, N)
    x_dit = x_dit.view(1, 2 * T, N)

    # Simulate bfloat16 precision loss
    if use_bfloat16:
        x_dit = x_dit.to(torch.bfloat16).to(torch.float32)

    # Model to track: reshape, [-1, 1] -> [0, 1]
    coords = x_dit[:, :2 * horizon, :]
    coords = coords.view(1, 2, horizon, N)
    tracks_out = coords.permute(0, 2, 3, 1).contiguous()
    tracks_01 = (tracks_out + 1.0) / 2.0

    return tracks_01[0].numpy()


def compute_frac_within_thresh(
    gt_tracks: torch.Tensor,
    pred_tracks: torch.Tensor,
    gt_vis: torch.Tensor = None,
    thresholds: list = None,
    img_size: int = 256,
    normalization: str = None,
    bbox: dict = None,
    frame_start: int = None,
    orig_width: int = None,
    orig_height: int = None
) -> dict:
    """
    Compute fraction of points within distance thresholds in pixel space.

    This is the canonical implementation handling all bbox formats.

    Args:
        gt_tracks: (B, T, N, 2) in [0, 1] normalized space
        pred_tracks: (B, T, N, 2) in [0, 1] normalized space
        gt_vis: (B, T, N) visibility mask, or None
        thresholds: distance thresholds in pixels (default: [1, 2, 4, 8, 16])
        img_size: fallback image size
        normalization: bbox normalization type
        bbox: bounding box info
        frame_start: starting frame index
        orig_width, orig_height: original image dimensions

    Returns:
        dict with 'pts_within_X' for each threshold and 'average_pts_within_thresh'
    """
    if thresholds is None:
        thresholds = [1, 2, 4, 8, 16]

    b, t, n, _ = gt_tracks.shape

    gt_pixel, pred_pixel, _ = tracks_to_pixel_space(
        gt_tracks, pred_tracks, gt_vis,
        img_size=img_size, normalization=normalization,
        bbox=bbox, frame_start=frame_start,
        orig_width=orig_width, orig_height=orig_height
    )

    if gt_vis is None:
        gt_vis = torch.ones((b, t, n), device=gt_tracks.device, dtype=torch.bool)
    else:
        gt_vis = gt_vis.bool()

    metrics = {}
    all_frac_within = []

    for thresh in thresholds:
        squared_dist = torch.sum((pred_pixel - gt_pixel) ** 2, dim=-1)
        within_thresh = squared_dist < (thresh ** 2)
        correct_and_visible = within_thresh & gt_vis
        count_correct = torch.sum(correct_and_visible, dim=(1, 2))
        count_visible = torch.sum(gt_vis, dim=(1, 2))

        frac_within = torch.where(
            count_visible > 0,
            count_correct.float() / count_visible.float(),
            torch.zeros_like(count_correct, dtype=torch.float32)
        )

        metrics[f'pts_within_{thresh}'] = frac_within
        all_frac_within.append(frac_within)

    all_frac_within = torch.stack(all_frac_within, dim=1)
    metrics['average_pts_within_thresh'] = torch.mean(all_frac_within, dim=1)

    return metrics


def compute_track_metrics_exact(
    gt_tracks: torch.Tensor,
    pred_tracks: torch.Tensor,
    gt_vis: torch.Tensor = None,
    point_mask: torch.Tensor = None,
    img_size: int = 256,
    normalization: str = None,
    bbox: dict = None,
    frame_start: int = None,
    orig_width: int = None,
    orig_height: int = None,
    num_point_cond: int = None,
    pwt_normalization: str = 'fixed_256',
) -> dict:
    """
    Compute track prediction metrics: ADE, FDE, and fraction of points within thresholds.

    - ADE/FDE are computed in normalized [0, 1] space
    - pts_within_thresh is computed in pixel space using pwt_normalization

    Args:
        gt_tracks: (B, T, N, 2) in [0, 1] normalized coords
        pred_tracks: (B, T, N, 2) in [0, 1] normalized coords
        gt_vis: (B, T, N) visibility, or None
        point_mask: (B, N) mask for real (non-padded) points, or None
        img_size: image size (default 256)
        normalization: bbox normalization type (used as fallback if pwt_normalization is None)
        bbox: bounding box info dict
        frame_start: starting frame index
        orig_width, orig_height: original image dimensions
        num_point_cond: number of conditioning timesteps (used to compute ADE window)
        pwt_normalization: normalization used for PWT pixel conversion
            (default 'fixed_256'; set to None to use `normalization` instead)

    Returns:
        dict with track_ade, track_fde, pts_within_X, average_pts_within_thresh
    """
    B, T, N, _ = gt_tracks.shape

    # === ADE on predicted timesteps (normalized space) ===
    num_ade_timesteps = (T - num_point_cond) if num_point_cond is not None else T
    gt_ade = gt_tracks[:, -num_ade_timesteps:, :, :]
    pred_ade = pred_tracks[:, -num_ade_timesteps:, :, :]

    if gt_vis is not None:
        vis_mask_ade = gt_vis[:, -num_ade_timesteps:, :]
    else:
        vis_mask_ade = torch.ones(B, num_ade_timesteps, N, device=gt_tracks.device)

    if point_mask is not None:
        pm_ade = point_mask.unsqueeze(1).expand(B, num_ade_timesteps, N)
        combined_ade = (vis_mask_ade.bool() & pm_ade.bool()).float()
    else:
        combined_ade = vis_mask_ade.float()

    # Per-point Euclidean distance (standard ADE definition)
    euclidean_dist_ade = torch.sqrt(((pred_ade - gt_ade) ** 2).sum(dim=-1))  # (B, T, N)
    valid_ade = combined_ade.sum()
    if valid_ade > 0:
        track_ade = (euclidean_dist_ade * combined_ade).sum() / valid_ade
    else:
        track_ade = torch.tensor(0.0, device=gt_tracks.device)

    # === FDE on last timestep (normalized space) ===
    gt_end = gt_tracks[:, -1, :, :]
    pred_end = pred_tracks[:, -1, :, :]

    if gt_vis is not None:
        vis_fde = gt_vis[:, -1, :]
    else:
        vis_fde = torch.ones(B, N, device=gt_tracks.device)

    if point_mask is not None:
        combined_fde = (vis_fde.bool() & point_mask.bool()).float()
    else:
        combined_fde = vis_fde.float()

    # Per-point Euclidean distance (standard FDE definition)
    euclidean_dist_fde = torch.sqrt(((gt_end - pred_end) ** 2).sum(dim=-1))  # (B, N)
    valid_fde = combined_fde.sum()
    if valid_fde > 0:
        track_fde = (euclidean_dist_fde * combined_fde).sum() / valid_fde
    else:
        track_fde = torch.tensor(0.0, device=gt_tracks.device)

    # === PWT in pixel space ===
    if point_mask is not None:
        pm_exp = point_mask.unsqueeze(1).unsqueeze(-1).expand(B, T, N, 2)
        gt_frac = gt_tracks * pm_exp.float()
        pred_frac = pred_tracks * pm_exp.float()
    else:
        gt_frac = gt_tracks
        pred_frac = pred_tracks

    if gt_vis is not None:
        if point_mask is not None:
            pm_vis = point_mask.unsqueeze(1).expand(B, T, N)
            combined_vis = (gt_vis.bool() & pm_vis.bool()).float()
        else:
            combined_vis = gt_vis
    else:
        if point_mask is not None:
            combined_vis = point_mask.unsqueeze(1).expand(B, T, N).float()
        else:
            combined_vis = torch.ones(B, T, N, device=gt_tracks.device)

    pwt_norm = pwt_normalization if pwt_normalization is not None else normalization
    frac_metrics = compute_frac_within_thresh(
        gt_tracks=gt_frac.clone(),
        pred_tracks=pred_frac.clone(),
        gt_vis=combined_vis,
        thresholds=[1, 2, 4, 8, 16],
        img_size=img_size,
        normalization=pwt_norm,
        bbox=bbox,
        frame_start=frame_start,
        orig_width=orig_width,
        orig_height=orig_height
    )

    metrics = {
        'track_ade': track_ade.item() if hasattr(track_ade, 'item') else float(track_ade),
        'track_fde': track_fde.item() if hasattr(track_fde, 'item') else float(track_fde),
        'pts_within_1': frac_metrics['pts_within_1'].mean().item(),
        'pts_within_2': frac_metrics['pts_within_2'].mean().item(),
        'pts_within_4': frac_metrics['pts_within_4'].mean().item(),
        'pts_within_8': frac_metrics['pts_within_8'].mean().item(),
        'pts_within_16': frac_metrics['pts_within_16'].mean().item(),
        'average_pts_within_thresh': frac_metrics['average_pts_within_thresh'].mean().item(),
    }

    return metrics


def compute_metrics_with_roundtrip(
    pred_tracks: np.ndarray,
    gt_tracks: np.ndarray,
    visibility: np.ndarray,
    point_mask: np.ndarray,
    pixel_info: dict,
    apply_gt_roundtrip: bool = True,
    use_bfloat16: bool = True,
    predicted_only: bool = False,
    num_point_cond: int = None,
    pwt_normalization: str = 'fixed_256',
) -> Dict[str, float]:
    """
    Compute all metrics using the EXACT same logic as training.

    This wrapper:
    1. Optionally applies the DiT round-trip conversion to gt_tracks
    2. Optionally slices to predicted-only timesteps
    3. Calls compute_track_metrics_exact

    Args:
        pred_tracks: (T, N, 2) predicted tracks in [0,1] space
        gt_tracks: (T, N, 2) ground truth tracks in [0,1] space
        visibility: (T, N) visibility mask
        point_mask: (N,) boolean mask for real points
        pixel_info: dict with bbox, frame_start, orig_width, orig_height, img_size, normalization
        apply_gt_roundtrip: If True, apply DiT format round-trip to gt_tracks
        use_bfloat16: If True, simulate bfloat16 precision loss
        predicted_only: If True, only compute on predicted timesteps
        num_point_cond: Number of conditioning timesteps
        pwt_normalization: normalization for PWT pixel conversion
            (default 'fixed_256'; set to None to use pixel_info['normalization'])

    Returns:
        dict with all metrics
    """
    if apply_gt_roundtrip:
        gt_tracks = apply_dit_roundtrip_conversion(gt_tracks, use_bfloat16=use_bfloat16)

    # If predicted_only, slice off conditioning frames. After slicing,
    # conditioning frames are gone so don't forward num_point_cond to
    # compute_track_metrics_exact (would cause double-subtraction in ADE).
    effective_num_point_cond = num_point_cond
    if predicted_only and num_point_cond is not None and num_point_cond > 0:
        pred_tracks = pred_tracks[num_point_cond:, :, :]
        gt_tracks = gt_tracks[num_point_cond:, :, :]
        if visibility is not None:
            visibility = visibility[num_point_cond:, :]
        effective_num_point_cond = None  # already sliced, don't subtract again

    T, N, _ = gt_tracks.shape

    pred_t = torch.from_numpy(pred_tracks).unsqueeze(0).float()
    gt_t = torch.from_numpy(gt_tracks).unsqueeze(0).float()
    vis_t = torch.from_numpy(visibility).unsqueeze(0).float() if visibility is not None else None
    pm_t = torch.from_numpy(point_mask).unsqueeze(0) if point_mask is not None else None

    metrics = compute_track_metrics_exact(
        gt_tracks=gt_t,
        pred_tracks=pred_t,
        gt_vis=vis_t,
        point_mask=pm_t,
        img_size=pixel_info.get('img_size', 256),
        normalization=pixel_info.get('normalization'),
        bbox=pixel_info.get('bbox'),
        frame_start=pixel_info.get('frame_start'),
        orig_width=pixel_info.get('orig_width'),
        orig_height=pixel_info.get('orig_height'),
        num_point_cond=effective_num_point_cond,
        pwt_normalization=pwt_normalization,
    )

    return metrics
