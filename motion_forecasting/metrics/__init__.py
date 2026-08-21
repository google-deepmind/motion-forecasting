"""
Track prediction metrics.

Direct imports of metric functions — no registry machinery.
"""

from motion_forecasting.metrics.ade_fde import compute_ade_fde_batch
from motion_forecasting.metrics.pts_within_thresh import (
    compute_pred_only_pwt_batch,
    compute_accel_regression_batch,
)
from motion_forecasting.metrics.frechet_distance import (
    compute_frechet_distance,
    compute_frechet_distance_core,
    compute_frechet_distance_velocity,
    compute_frechet_distance_acceleration,
    compute_variance_pred,
    compute_variance_gt,
    extract_perfect_tracks_flattened,
    extract_perfect_tracks_velocity,
    extract_perfect_tracks_acceleration,
    _fd_from_matrices,
)
from motion_forecasting.metrics.mvd import compute_mvd_distance
from motion_forecasting.metrics.pixel_utils import (
    compute_track_metrics_exact,
    compute_metrics_with_roundtrip,
    compute_frac_within_thresh,
)

# FVMD has optional scipy dependency
try:
    from motion_forecasting.metrics.fvmd import (
        compute_fvmd,
        compute_fvmd_velocity,
        compute_fvmd_acceleration,
        compute_fvmd_tw4,
        compute_fvmd_tw8,
        compute_fvmd_spatiotemporal,
    )
except ImportError:
    pass
