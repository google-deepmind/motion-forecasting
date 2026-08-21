"""
FVMD (Frechet Video Motion Distance) metrics for point tracks.

Adapted from https://github.com/jzhangbs/FVMD-frechet-video-motion-distance

FVMD computes HOG-like histogram features from velocity and acceleration,
then measures distribution similarity via Frechet distance.

Variants:
- Standard FVMD with different temporal windows (tw4, tw8)
- Overlapping windows (tw4_stride2)
- Velocity-only and acceleration-only components
- Perfect tracks only
- No temporal windowing
- Spatiotemporal with anchor points and k-NN neighborhoods
"""

import numpy as np
from typing import List, Tuple, Optional

from motion_forecasting.metrics.frechet_distance import compute_frechet_distance_core


# =============================================================================
# FVMD histogram computation
# =============================================================================

def _compute_fvmd_histogram(vectors: np.ndarray, angle_bins: int = 8,
                            magnitude_bins: int = 256) -> np.ndarray:
    """
    Compute HOG-like histogram for motion vectors.

    Matches the original FVMD implementation:
    - arctan2(vector[0], vector[1]) argument order
    - Magnitude clipped to [0, magnitude_bins-1], then log-scaled with ceil
    - No normalization of individual histograms

    Args:
        vectors: (num_vectors, 2) array of motion vectors [x, y]
        angle_bins: Number of angle bins (default: 8)
        magnitude_bins: Max magnitude for clipping (default: 256)

    Returns:
        histogram: (angle_bins,)
    """
    if vectors.size == 0:
        return np.zeros(angle_bins)

    angles = np.arctan2(vectors[:, 0], vectors[:, 1])
    angle_bins_idx = (angles + np.pi) // (2 * np.pi / angle_bins)
    angle_bins_idx = np.clip(angle_bins_idx, 0, angle_bins - 1).astype(int)

    magnitudes = np.linalg.norm(vectors, axis=1)
    magnitudes = np.clip(magnitudes, 0, magnitude_bins - 1)
    magnitude_weights = magnitudes + 1
    magnitude_weights = np.log2(magnitude_weights)
    magnitude_weights = np.clip(magnitude_weights, 0, int(np.log2(magnitude_bins)))
    magnitude_weights = np.ceil(magnitude_weights)
    magnitude_weights = magnitude_weights / np.log2(magnitude_bins)

    histogram = np.zeros(angle_bins)
    for i in range(len(vectors)):
        histogram[angle_bins_idx[i]] += magnitude_weights[i]

    return histogram


# =============================================================================
# Feature extraction
# =============================================================================

def _extract_fvmd_features(
    all_data: List[dict],
    angle_bins: int = 8,
    magnitude_bins: int = 256,
    temporal_window: int = 4,
    stride: int = None,
    require_full_visibility: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract FVMD-style features for all samples.

    For each sample:
    1. Compute velocity and acceleration from tracks (in pixel space)
    2. Build HOG-like histograms per temporal window
    3. Concatenate velocity and acceleration histograms

    Args:
        all_data: List of sample dicts
        angle_bins: Number of angle bins (default: 8)
        magnitude_bins: Max magnitude for clipping (default: 256)
        temporal_window: Temporal window size (default: 4, 0/None = all frames)
        stride: Stride between windows (default: same as temporal_window)
        require_full_visibility: If True, only use tracks visible in ALL frames

    Returns:
        pred_features: (num_samples, feature_dim)
        gt_features: (num_samples, feature_dim)
    """
    if stride is None:
        stride = temporal_window if temporal_window else 1

    pred_features_list = []
    gt_features_list = []

    for sample_data in all_data:
        pred_tracks = sample_data['pred_tracks']
        gt_tracks = sample_data['gt_tracks']
        visibility = sample_data.get('visibility')
        point_mask = sample_data['point_mask']
        num_point_cond = sample_data.get('num_point_cond', 0)
        pixel_info = sample_data.get('pixel_info', {})

        img_size = pixel_info.get('img_size', 256) if pixel_info else 256
        T, N, _ = pred_tracks.shape

        if num_point_cond is not None and num_point_cond > 0:
            pred_pred = pred_tracks[num_point_cond:, :, :]
            gt_pred = gt_tracks[num_point_cond:, :, :]
            vis_pred = visibility[num_point_cond:, :] if visibility is not None else None
        else:
            pred_pred = pred_tracks
            gt_pred = gt_tracks
            vis_pred = visibility

        T_pred = pred_pred.shape[0]
        if T_pred < 3:
            continue

        # Velocities in pixel space
        pred_vel = np.diff(pred_pred, axis=0) * img_size
        gt_vel = np.diff(gt_pred, axis=0) * img_size
        pred_acc = np.diff(pred_vel, axis=0)
        gt_acc = np.diff(gt_vel, axis=0)

        # Determine valid points
        valid_points = []
        for n in range(N):
            if point_mask is not None and not point_mask[n]:
                continue
            if vis_pred is not None:
                if require_full_visibility:
                    if not np.all(vis_pred[:, n]):
                        continue
                    valid_vel = list(range(T_pred - 1))
                    valid_acc = list(range(T_pred - 2))
                else:
                    valid_vel = [t for t in range(T_pred - 1)
                                 if vis_pred[t, n] and vis_pred[t + 1, n]]
                    valid_acc = [t for t in range(T_pred - 2)
                                 if vis_pred[t, n] and vis_pred[t + 1, n] and vis_pred[t + 2, n]]
            else:
                valid_vel = list(range(T_pred - 1))
                valid_acc = list(range(T_pred - 2))

            if valid_vel:
                valid_points.append((n, valid_vel, valid_acc))

        # Determine temporal windows
        n_vel_frames = T_pred - 1
        n_acc_frames = T_pred - 2

        if temporal_window is not None and temporal_window > 0:
            num_vel_windows = max(1, (n_vel_frames - temporal_window) // stride + 1) if n_vel_frames >= temporal_window else 1
            num_acc_windows = max(1, (n_acc_frames - temporal_window) // stride + 1) if n_acc_frames >= temporal_window else 1
        else:
            num_vel_windows = 1
            num_acc_windows = 1

        # Collect vectors per window
        pred_vel_by_w = [[] for _ in range(num_vel_windows)]
        gt_vel_by_w = [[] for _ in range(num_vel_windows)]
        pred_acc_by_w = [[] for _ in range(num_acc_windows)]
        gt_acc_by_w = [[] for _ in range(num_acc_windows)]

        for n, valid_vel, valid_acc in valid_points:
            for t in valid_vel:
                if temporal_window is not None and temporal_window > 0:
                    w_min = max(0, (t - temporal_window + 1 + stride - 1) // stride)
                    w_max = min(num_vel_windows - 1, t // stride)
                    for w in range(w_min, w_max + 1):
                        ws = w * stride
                        if ws <= t < ws + temporal_window:
                            pred_vel_by_w[w].append(pred_vel[t, n, :])
                            gt_vel_by_w[w].append(gt_vel[t, n, :])
                else:
                    pred_vel_by_w[0].append(pred_vel[t, n, :])
                    gt_vel_by_w[0].append(gt_vel[t, n, :])

            for t in valid_acc:
                if temporal_window is not None and temporal_window > 0:
                    w_min = max(0, (t - temporal_window + 1 + stride - 1) // stride)
                    w_max = min(num_acc_windows - 1, t // stride)
                    for w in range(w_min, w_max + 1):
                        ws = w * stride
                        if ws <= t < ws + temporal_window:
                            pred_acc_by_w[w].append(pred_acc[t, n, :])
                            gt_acc_by_w[w].append(gt_acc[t, n, :])
                else:
                    pred_acc_by_w[0].append(pred_acc[t, n, :])
                    gt_acc_by_w[0].append(gt_acc[t, n, :])

        # Compute histograms
        pred_vel_hists = []
        gt_vel_hists = []
        for w in range(num_vel_windows):
            pv = np.array(pred_vel_by_w[w]) if pred_vel_by_w[w] else np.array([]).reshape(0, 2)
            gv = np.array(gt_vel_by_w[w]) if gt_vel_by_w[w] else np.array([]).reshape(0, 2)
            pred_vel_hists.append(_compute_fvmd_histogram(pv, angle_bins, magnitude_bins))
            gt_vel_hists.append(_compute_fvmd_histogram(gv, angle_bins, magnitude_bins))

        pred_acc_hists = []
        gt_acc_hists = []
        for w in range(num_acc_windows):
            pa = np.array(pred_acc_by_w[w]) if pred_acc_by_w[w] else np.array([]).reshape(0, 2)
            ga = np.array(gt_acc_by_w[w]) if gt_acc_by_w[w] else np.array([]).reshape(0, 2)
            pred_acc_hists.append(_compute_fvmd_histogram(pa, angle_bins, magnitude_bins))
            gt_acc_hists.append(_compute_fvmd_histogram(ga, angle_bins, magnitude_bins))

        pred_features_list.append(np.concatenate(pred_vel_hists + pred_acc_hists))
        gt_features_list.append(np.concatenate(gt_vel_hists + gt_acc_hists))

    if not pred_features_list:
        feature_dim = 2 * angle_bins
        return np.array([]).reshape(0, feature_dim), np.array([]).reshape(0, feature_dim)

    return np.stack(pred_features_list, axis=0), np.stack(gt_features_list, axis=0)


def _compute_fvmd_from_features(pred_features: np.ndarray, gt_features: np.ndarray,
                                metric_name: str = "FVMD") -> Optional[float]:
    """Helper to compute Frechet distance from extracted features."""
    if pred_features.size == 0 or gt_features.size == 0:
        return None
    if pred_features.shape[0] < 2:
        return None

    mu_pred = np.mean(pred_features, axis=0)
    sigma_pred = np.cov(pred_features, rowvar=False)
    mu_gt = np.mean(gt_features, axis=0)
    sigma_gt = np.cov(gt_features, rowvar=False)

    if sigma_pred.ndim == 0:
        sigma_pred = np.array([[sigma_pred]])
        sigma_gt = np.array([[sigma_gt]])

    return compute_frechet_distance_core(mu_pred, sigma_pred, mu_gt, sigma_gt, regularize=True)


# =============================================================================
# Core FVMD computation functions (parameterized)
# =============================================================================

def compute_fvmd(all_data, angle_bins=8, magnitude_bins=256,
                 temporal_window=4, stride=None, require_full_visibility=False):
    """Compute FVMD between predicted and GT track distributions."""
    if not all_data:
        return None
    pred_f, gt_f = _extract_fvmd_features(
        all_data, angle_bins, magnitude_bins, temporal_window, stride, require_full_visibility)
    return _compute_fvmd_from_features(pred_f, gt_f, f"FVMD (tw={temporal_window})")


def compute_fvmd_velocity(all_data, angle_bins=8, magnitude_bins=256,
                          temporal_window=4, stride=None, require_full_visibility=False):
    """Compute FVMD using only velocity histogram features."""
    if not all_data:
        return None
    pred_f, gt_f = _extract_fvmd_features(
        all_data, angle_bins, magnitude_bins, temporal_window, stride, require_full_visibility)
    if pred_f.size == 0:
        return None
    vel_dim = pred_f.shape[1] // 2 if temporal_window else angle_bins
    return _compute_fvmd_from_features(pred_f[:, :vel_dim], gt_f[:, :vel_dim], "FVMD velocity")


def compute_fvmd_acceleration(all_data, angle_bins=8, magnitude_bins=256,
                              temporal_window=4, stride=None, require_full_visibility=False):
    """Compute FVMD using only acceleration histogram features."""
    if not all_data:
        return None
    pred_f, gt_f = _extract_fvmd_features(
        all_data, angle_bins, magnitude_bins, temporal_window, stride, require_full_visibility)
    if pred_f.size == 0:
        return None
    vel_dim = pred_f.shape[1] // 2 if temporal_window else angle_bins
    return _compute_fvmd_from_features(pred_f[:, vel_dim:], gt_f[:, vel_dim:], "FVMD acceleration")


# =============================================================================
# Spatiotemporal FVMD (anchor points + k-NN)
# =============================================================================

def _extract_fvmd_features_spatiotemporal(
    all_data: List[dict],
    angle_bins: int = 8,
    magnitude_bins: int = 256,
    temporal_window: int = 4,
    num_anchors: int = 100,
    k_neighbors: int = 10,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract FVMD features using spatiotemporal subcubes with anchor points and k-NN."""
    from scipy.spatial import cKDTree

    rng = np.random.RandomState(seed)
    pred_features_list = []
    gt_features_list = []

    for sample_data in all_data:
        pred_tracks = sample_data['pred_tracks']
        gt_tracks = sample_data['gt_tracks']
        visibility = sample_data.get('visibility')
        point_mask = sample_data['point_mask']
        num_point_cond = sample_data.get('num_point_cond', 0)
        pixel_info = sample_data.get('pixel_info', {})
        img_size = pixel_info.get('img_size', 256) if pixel_info else 256

        T, N, _ = pred_tracks.shape

        if num_point_cond is not None and num_point_cond > 0:
            pred_pred = pred_tracks[num_point_cond:, :, :]
            gt_pred = gt_tracks[num_point_cond:, :, :]
            vis_pred = visibility[num_point_cond:, :] if visibility is not None else None
        else:
            pred_pred = pred_tracks
            gt_pred = gt_tracks
            vis_pred = visibility

        T_pred = pred_pred.shape[0]
        if T_pred < 3:
            continue

        valid_indices = np.array([n for n in range(N) if point_mask is None or point_mask[n]])
        num_valid = len(valid_indices)
        if num_valid < k_neighbors + 1:
            continue

        if num_valid >= num_anchors:
            anchor_local = rng.choice(num_valid, size=num_anchors, replace=False)
        else:
            anchor_local = rng.choice(num_valid, size=num_anchors, replace=True)
        anchor_global = valid_indices[anchor_local]

        first_pos = gt_pred[0, valid_indices, :]
        kdtree = cKDTree(first_pos)
        _, neighbor_local = kdtree.query(first_pos[anchor_local], k=min(k_neighbors + 1, num_valid))
        if neighbor_local.ndim == 1:
            neighbor_local = neighbor_local.reshape(1, -1)
        neighbor_global = valid_indices[neighbor_local]

        pred_vel = np.diff(pred_pred, axis=0) * img_size
        gt_vel = np.diff(gt_pred, axis=0) * img_size
        pred_acc = np.diff(pred_vel, axis=0)
        gt_acc = np.diff(gt_vel, axis=0)

        n_vel_frames = T_pred - 1
        n_acc_frames = T_pred - 2
        num_vel_windows = max(1, n_vel_frames // temporal_window) if temporal_window > 0 else 1
        num_acc_windows = max(1, n_acc_frames // temporal_window) if temporal_window > 0 else 1

        pred_vel_hists = []
        gt_vel_hists = []
        pred_acc_hists = []
        gt_acc_hists = []

        for ai in range(num_anchors):
            neighbors = neighbor_global[ai]
            for w in range(num_vel_windows):
                t_start = w * temporal_window if temporal_window > 0 else 0
                t_end = min(t_start + temporal_window, n_vel_frames) if temporal_window > 0 else n_vel_frames
                pv, gv = [], []
                for t in range(t_start, t_end):
                    for n in neighbors:
                        if vis_pred is not None and not (vis_pred[t, n] and vis_pred[t + 1, n]):
                            continue
                        pv.append(pred_vel[t, n, :])
                        gv.append(gt_vel[t, n, :])
                pv = np.array(pv) if pv else np.array([]).reshape(0, 2)
                gv = np.array(gv) if gv else np.array([]).reshape(0, 2)
                pred_vel_hists.append(_compute_fvmd_histogram(pv, angle_bins, magnitude_bins))
                gt_vel_hists.append(_compute_fvmd_histogram(gv, angle_bins, magnitude_bins))

        for ai in range(num_anchors):
            neighbors = neighbor_global[ai]
            for w in range(num_acc_windows):
                t_start = w * temporal_window if temporal_window > 0 else 0
                t_end = min(t_start + temporal_window, n_acc_frames) if temporal_window > 0 else n_acc_frames
                pa, ga = [], []
                for t in range(t_start, t_end):
                    for n in neighbors:
                        if vis_pred is not None and not (vis_pred[t, n] and vis_pred[t + 1, n] and vis_pred[t + 2, n]):
                            continue
                        pa.append(pred_acc[t, n, :])
                        ga.append(gt_acc[t, n, :])
                pa = np.array(pa) if pa else np.array([]).reshape(0, 2)
                ga = np.array(ga) if ga else np.array([]).reshape(0, 2)
                pred_acc_hists.append(_compute_fvmd_histogram(pa, angle_bins, magnitude_bins))
                gt_acc_hists.append(_compute_fvmd_histogram(ga, angle_bins, magnitude_bins))

        pred_features_list.append(np.concatenate(pred_vel_hists + pred_acc_hists))
        gt_features_list.append(np.concatenate(gt_vel_hists + gt_acc_hists))

    if not pred_features_list:
        return np.array([]).reshape(0, 1), np.array([]).reshape(0, 1)

    return np.stack(pred_features_list, axis=0), np.stack(gt_features_list, axis=0)


def compute_fvmd_spatiotemporal(all_data, angle_bins=8, magnitude_bins=256,
                                temporal_window=4, num_anchors=100, k_neighbors=10):
    """Compute FVMD using spatiotemporal subcubes with anchor points and k-NN."""
    if not all_data:
        return None
    pred_f, gt_f = _extract_fvmd_features_spatiotemporal(
        all_data, angle_bins, magnitude_bins, temporal_window, num_anchors, k_neighbors)
    return _compute_fvmd_from_features(pred_f, gt_f, "FVMD spatiotemporal")


def compute_fvmd_spatiotemporal_velocity(all_data, angle_bins=8, magnitude_bins=256,
                                         temporal_window=4, num_anchors=100, k_neighbors=10):
    """Compute FVMD spatiotemporal using only velocity histograms."""
    if not all_data:
        return None
    pred_f, gt_f = _extract_fvmd_features_spatiotemporal(
        all_data, angle_bins, magnitude_bins, temporal_window, num_anchors, k_neighbors)
    if pred_f.size == 0:
        return None
    vel_dim = pred_f.shape[1] // 2
    return _compute_fvmd_from_features(pred_f[:, :vel_dim], gt_f[:, :vel_dim], "FVMD spatiotemporal velocity")


def compute_fvmd_spatiotemporal_acceleration(all_data, angle_bins=8, magnitude_bins=256,
                                              temporal_window=4, num_anchors=100, k_neighbors=10):
    """Compute FVMD spatiotemporal using only acceleration histograms."""
    if not all_data:
        return None
    pred_f, gt_f = _extract_fvmd_features_spatiotemporal(
        all_data, angle_bins, magnitude_bins, temporal_window, num_anchors, k_neighbors)
    if pred_f.size == 0:
        return None
    vel_dim = pred_f.shape[1] // 2
    return _compute_fvmd_from_features(pred_f[:, vel_dim:], gt_f[:, vel_dim:], "FVMD spatiotemporal acceleration")


# =============================================================================
# Registered FVMD metrics
# =============================================================================

# -- Temporal Window 4 (default, matches original FVMD) --
def compute_fvmd_tw4(all_data):
    """FVMD with temporal_window=4 (matches original FVMD's cube_frames=4)."""
    return compute_fvmd(all_data, temporal_window=4)

def compute_fvmd_velocity_tw4(all_data):
    """FVMD velocity component with temporal_window=4."""
    return compute_fvmd_velocity(all_data, temporal_window=4)

def compute_fvmd_acceleration_tw4(all_data):
    """FVMD acceleration component with temporal_window=4."""
    return compute_fvmd_acceleration(all_data, temporal_window=4)

# -- Temporal Window 8 --
def compute_fvmd_tw8(all_data):
    """FVMD with temporal_window=8 (longer temporal context)."""
    return compute_fvmd(all_data, temporal_window=8)

def compute_fvmd_velocity_tw8(all_data):
    """FVMD velocity component with temporal_window=8."""
    return compute_fvmd_velocity(all_data, temporal_window=8)

def compute_fvmd_acceleration_tw8(all_data):
    """FVMD acceleration component with temporal_window=8."""
    return compute_fvmd_acceleration(all_data, temporal_window=8)

# -- Temporal Window 4, Stride 2 (overlapping) --
def compute_fvmd_tw4_stride2(all_data):
    """FVMD with temporal_window=4 and stride=2 (50% overlap)."""
    return compute_fvmd(all_data, temporal_window=4, stride=2)

def compute_fvmd_velocity_tw4_stride2(all_data):
    """FVMD velocity with temporal_window=4 and stride=2."""
    return compute_fvmd_velocity(all_data, temporal_window=4, stride=2)

def compute_fvmd_acceleration_tw4_stride2(all_data):
    """FVMD acceleration with temporal_window=4 and stride=2."""
    return compute_fvmd_acceleration(all_data, temporal_window=4, stride=2)

# -- Additional variants --
def compute_fvmd_perfect_tracks(all_data):
    """FVMD using only tracks visible in ALL prediction frames."""
    return compute_fvmd(all_data, temporal_window=4, require_full_visibility=True)

def compute_fvmd_no_temporal_window(all_data):
    """FVMD without temporal windowing (single histogram for all frames)."""
    return compute_fvmd(all_data, temporal_window=0)

# -- Spatiotemporal --
def compute_fvmd_spatiotemporal_default(all_data):
    """FVMD with spatiotemporal subcubes (100 anchors, k=10)."""
    return compute_fvmd_spatiotemporal(all_data, temporal_window=4, num_anchors=100, k_neighbors=10)

def compute_fvmd_spatiotemporal_velocity_default(all_data):
    """FVMD spatiotemporal velocity component."""
    return compute_fvmd_spatiotemporal_velocity(all_data, temporal_window=4, num_anchors=100, k_neighbors=10)

def compute_fvmd_spatiotemporal_acceleration_default(all_data):
    """FVMD spatiotemporal acceleration component."""
    return compute_fvmd_spatiotemporal_acceleration(all_data, temporal_window=4, num_anchors=100, k_neighbors=10)
