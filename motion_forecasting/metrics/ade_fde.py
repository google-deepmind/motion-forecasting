"""
ADE (Average Displacement Error) and FDE (Final Displacement Error) metrics.

track_ade / track_fde: Predicted-only (excludes conditioning timesteps).
These are the primary metrics since conditioning frames are copied from GT.
"""

import numpy as np
from typing import Dict, Optional

from motion_forecasting.metrics.pixel_utils import compute_metrics_with_roundtrip


def compute_ade_fde_batch(pred_tracks: np.ndarray, gt_tracks: np.ndarray,
                          visibility: np.ndarray, pred_visibility: np.ndarray,
                          point_mask: np.ndarray,
                          pixel_info: dict, use_bfloat16: bool = True,
                          num_point_cond: int = None) -> Dict[str, Optional[float]]:
    """Compute predicted-only ADE/FDE metrics."""
    if num_point_cond is not None and num_point_cond > 0:
        pred_metrics = compute_metrics_with_roundtrip(
            pred_tracks, gt_tracks, visibility, point_mask, pixel_info,
            apply_gt_roundtrip=use_bfloat16, use_bfloat16=use_bfloat16,
            predicted_only=True, num_point_cond=num_point_cond
        )
        return {
            'track_ade': pred_metrics['track_ade'],
            'track_fde': pred_metrics['track_fde'],
        }
    return {'track_ade': None, 'track_fde': None}
