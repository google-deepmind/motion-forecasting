"""
Points-within-threshold (PWT) metrics and acceleration regression distance.

PWT measures the fraction of predicted points within various pixel-distance
thresholds of the ground truth (predicted timesteps only).

Acceleration regression distance fits a model:
    GT ~ t^2 * (Translation + Scale * Pred)
and measures the average per-point Euclidean distance between GT and the fitted model.

Acceleration regression PWT is the same regression fit, but instead of reporting
mean Euclidean distance it reports average_pts_within_thresh on the fitted tracks.
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple

from motion_forecasting.metrics.pixel_utils import (
    compute_metrics_with_roundtrip,
    apply_dit_roundtrip_conversion,
    compute_frac_within_thresh,
)


# =============================================================================
# BATCH: Predicted-only PWT metrics (one call to compute_metrics_with_roundtrip)
# =============================================================================

_PRED_PWT_NAMES = [
    'pred_pts_within_1', 'pred_pts_within_2', 'pred_pts_within_4',
    'pred_pts_within_8', 'pred_pts_within_16', 'pred_average_pts_within_thresh',
]

_PRED_PWT_KEY_MAP = {
    'pred_pts_within_1': 'pts_within_1',
    'pred_pts_within_2': 'pts_within_2',
    'pred_pts_within_4': 'pts_within_4',
    'pred_pts_within_8': 'pts_within_8',
    'pred_pts_within_16': 'pts_within_16',
    'pred_average_pts_within_thresh': 'average_pts_within_thresh',
}


def compute_pred_only_pwt_batch(pred_tracks, gt_tracks, visibility,
                                pred_visibility, point_mask,
                                pixel_info, use_bfloat16=True, num_point_cond=None):
    """Compute all predicted-only PWT metrics in a single call."""
    if num_point_cond is None or num_point_cond <= 0:
        return {name: None for name in _PRED_PWT_NAMES}
    metrics = compute_metrics_with_roundtrip(
        pred_tracks, gt_tracks, visibility, point_mask, pixel_info,
        apply_gt_roundtrip=use_bfloat16, use_bfloat16=use_bfloat16,
        predicted_only=True, num_point_cond=num_point_cond
    )
    return {pred_name: metrics[raw_key] for pred_name, raw_key in _PRED_PWT_KEY_MAP.items()}


# =============================================================================
# SHARED HELPER: Acceleration regression fitting
# =============================================================================

def _prepare_accel_regression_inputs(pred_tracks, gt_tracks, visibility,
                                     pred_visibility, point_mask,
                                     use_bfloat16, num_point_cond):
    """
    Shared preprocessing for acceleration regression metrics:
    applies bfloat16 roundtrip to GT, slices to predicted-only frames.

    Returns:
        (pred_tracks, gt_tracks, visibility, pred_visibility) after slicing
    """
    if use_bfloat16:
        gt_tracks = apply_dit_roundtrip_conversion(gt_tracks, use_bfloat16=True)

    if num_point_cond is not None and num_point_cond > 0:
        pred_tracks = pred_tracks[num_point_cond:, :, :]
        gt_tracks = gt_tracks[num_point_cond:, :, :]
        if visibility is not None:
            visibility = visibility[num_point_cond:, :]
        if pred_visibility is not None:
            pred_visibility = pred_visibility[num_point_cond:, :]

    return pred_tracks, gt_tracks, visibility, pred_visibility


def _fit_acceleration_regression(pred_tracks, gt_tracks, visibility,
                                 pred_visibility, point_mask):
    """
    Fit the acceleration regression model: GT ~ pred + t^2 * (Translation + Scale * pred).

    Uses two masks:
    - gt_mask (visibility & point_mask): positions where GT is available for evaluation
    - reg_mask (gt_mask & pred_vis): positions used for fitting the regression
      (restricted to where the model was confident in its predictions)

    Returns:
        fitted_tracks: (T, N, 2) with regression-corrected predictions
        gt_mask: (T, N) boolean mask for GT-visible real points
        success: bool indicating whether the fit succeeded
    """
    T, N, _ = pred_tracks.shape

    if visibility is None:
        visibility = np.ones((T, N), dtype=bool)
    else:
        visibility = visibility > 0.5

    if pred_visibility is None:
        pred_vis = np.ones((T, N), dtype=bool)
    else:
        pred_vis = pred_visibility > 0.5

    if point_mask is None:
        point_mask = np.ones((N,), dtype=bool)

    gt_mask = visibility & point_mask[np.newaxis, :]
    reg_mask = gt_mask & pred_vis

    if np.sum(reg_mask) < 3:
        return pred_tracks.copy(), gt_mask, False

    reg_rows, reg_cols = np.where(reg_mask)
    reg_t_vals = reg_rows.astype(np.float32)
    reg_t_sq = reg_t_vals ** 2

    reg_p_x = pred_tracks[reg_rows, reg_cols, 0]
    reg_p_y = pred_tracks[reg_rows, reg_cols, 1]
    reg_g_x = gt_tracks[reg_rows, reg_cols, 0]
    reg_g_y = gt_tracks[reg_rows, reg_cols, 1]

    K = len(reg_rows)
    A = np.zeros((2 * K, 3), dtype=np.float32)
    b = np.zeros((2 * K,), dtype=np.float32)

    A[:K, 0] = reg_t_sq
    A[:K, 2] = reg_t_sq * reg_p_x
    b[:K] = reg_g_x - reg_p_x

    A[K:, 1] = reg_t_sq
    A[K:, 2] = reg_t_sq * reg_p_y
    b[K:] = reg_g_y - reg_p_y

    try:
        solution, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        X_coef, Y_coef, S_coef = solution
    except np.linalg.LinAlgError:
        return pred_tracks.copy(), gt_mask, False

    if np.sum(gt_mask) == 0:
        return pred_tracks.copy(), gt_mask, False

    fitted_tracks = pred_tracks.copy()
    t_grid = np.arange(T, dtype=np.float32)[:, np.newaxis]  # (T, 1)
    t_sq_grid = t_grid ** 2  # (T, 1)

    fitted_tracks[:, :, 0] = pred_tracks[:, :, 0] + t_sq_grid * (X_coef + S_coef * pred_tracks[:, :, 0])
    fitted_tracks[:, :, 1] = pred_tracks[:, :, 1] + t_sq_grid * (Y_coef + S_coef * pred_tracks[:, :, 1])

    return fitted_tracks, gt_mask, True


# =============================================================================
# BATCH: All acceleration regression metrics (single fit, 3 metrics)
# =============================================================================

_ACCEL_REG_NAMES = [
    'acceleration_regression_distance',
    'acceleration_regression_pwt',
    'acceleration_regression_pwt_diff',
]


def compute_accel_regression_batch(pred_tracks: np.ndarray, gt_tracks: np.ndarray,
                                   visibility: np.ndarray, pred_visibility: np.ndarray,
                                   point_mask: np.ndarray,
                                   pixel_info: dict, use_bfloat16: bool = True,
                                   num_point_cond: int = None) -> Dict[str, float]:
    """Compute all acceleration regression metrics with a single regression fit.

    Metrics:
    - acceleration_regression_distance: mean Euclidean distance between GT and fitted
    - acceleration_regression_pwt: average_pts_within_thresh on fitted tracks
    - acceleration_regression_pwt_diff: PWT improvement from regression (fitted - raw)
    """
    pred_tracks, gt_tracks, visibility, pred_visibility = _prepare_accel_regression_inputs(
        pred_tracks, gt_tracks, visibility, pred_visibility, point_mask,
        use_bfloat16, num_point_cond
    )

    fitted_tracks, gt_mask, success = _fit_acceleration_regression(
        pred_tracks, gt_tracks, visibility, pred_visibility, point_mask
    )

    if not success:
        return {name: 0.0 for name in _ACCEL_REG_NAMES}

    # --- Distance: mean Euclidean distance on gt_mask positions ---
    rows, cols = np.where(gt_mask)
    diff_x = gt_tracks[rows, cols, 0] - fitted_tracks[rows, cols, 0]
    diff_y = gt_tracks[rows, cols, 1] - fitted_tracks[rows, cols, 1]
    dists = np.sqrt(diff_x ** 2 + diff_y ** 2)
    accel_dist = float(np.mean(dists))

    # --- PWT and PWT diff: needs frac_within_thresh on fitted and raw ---
    gt_vis = gt_mask.astype(np.float32)
    fitted_t = torch.from_numpy(fitted_tracks).unsqueeze(0).float()
    pred_t = torch.from_numpy(pred_tracks).unsqueeze(0).float()
    gt_t = torch.from_numpy(gt_tracks).unsqueeze(0).float()
    vis_t = torch.from_numpy(gt_vis).unsqueeze(0).float()

    frac_fitted = compute_frac_within_thresh(
        gt_tracks=gt_t, pred_tracks=fitted_t, gt_vis=vis_t,
        thresholds=[1, 2, 4, 8, 16], normalization='fixed_256',
    )
    frac_raw = compute_frac_within_thresh(
        gt_tracks=gt_t, pred_tracks=pred_t, gt_vis=vis_t,
        thresholds=[1, 2, 4, 8, 16], normalization='fixed_256',
    )

    fitted_pwt = float(frac_fitted['average_pts_within_thresh'].mean().item())
    raw_pwt = float(frac_raw['average_pts_within_thresh'].mean().item())

    return {
        'acceleration_regression_distance': accel_dist,
        'acceleration_regression_pwt': fitted_pwt,
        'acceleration_regression_pwt_diff': fitted_pwt - raw_pwt,
    }


