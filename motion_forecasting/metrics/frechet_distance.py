"""
Frechet Distance metrics for track distributions.

Computes Frechet Distance between predicted and ground truth track distributions
using position, velocity (first-order differences), and acceleration (second-order
differences) representations.

Also provides variance metrics for predicted and GT trajectories.
"""

import numpy as np
from typing import List, Tuple, Optional

# =============================================================================
# Track extraction helpers (shared with FVMD)
# =============================================================================

def extract_perfect_tracks_flattened(all_data: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract "perfect" tracks (visible in ALL prediction frames) and flatten them.

    For each sample:
    1. Filter for tracks visible in ALL prediction frames (after num_point_cond)
    2. Flatten each track to a vector of size (T_pred * 2)

    Returns:
        pred_matrix: (total_perfect_tracks, T_pred * 2)
        gt_matrix: (total_perfect_tracks, T_pred * 2)
    """
    all_pred_flat = []
    all_gt_flat = []

    for sample_data in all_data:
        pred_tracks = sample_data['pred_tracks']  # (T, N, 2)
        gt_tracks = sample_data['gt_tracks']  # (T, N, 2)
        visibility = sample_data.get('visibility')
        point_mask = sample_data['point_mask']
        num_point_cond = sample_data.get('num_point_cond', 0)

        T, N, _ = pred_tracks.shape

        if num_point_cond is not None and num_point_cond > 0:
            pred_pred = pred_tracks[num_point_cond:, :, :]
            gt_pred = gt_tracks[num_point_cond:, :, :]
            vis_pred = visibility[num_point_cond:, :] if visibility is not None else None
        else:
            pred_pred = pred_tracks
            gt_pred = gt_tracks
            vis_pred = visibility

        for n in range(N):
            if point_mask is not None and not point_mask[n]:
                continue
            if vis_pred is not None and not np.all(vis_pred[:, n]):
                continue

            all_pred_flat.append(pred_pred[:, n, :].flatten())
            all_gt_flat.append(gt_pred[:, n, :].flatten())

    if not all_pred_flat:
        return np.array([]), np.array([])

    return np.stack(all_pred_flat, axis=0), np.stack(all_gt_flat, axis=0)


def extract_perfect_tracks_velocity(all_data: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract "perfect" tracks and compute their velocities (first-order differences).

    Velocity: v_t = position_{t+1} - position_t

    Returns:
        pred_matrix: (total_perfect_tracks, (T_pred-1) * 2)
        gt_matrix: (total_perfect_tracks, (T_pred-1) * 2)
    """
    all_pred_vel = []
    all_gt_vel = []

    for sample_data in all_data:
        pred_tracks = sample_data['pred_tracks']
        gt_tracks = sample_data['gt_tracks']
        visibility = sample_data.get('visibility')
        point_mask = sample_data['point_mask']
        num_point_cond = sample_data.get('num_point_cond', 0)

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
        if T_pred < 2:
            continue

        pred_vel = np.diff(pred_pred, axis=0)
        gt_vel = np.diff(gt_pred, axis=0)

        for n in range(N):
            if point_mask is not None and not point_mask[n]:
                continue
            if vis_pred is not None and not np.all(vis_pred[:, n]):
                continue

            all_pred_vel.append(pred_vel[:, n, :].flatten())
            all_gt_vel.append(gt_vel[:, n, :].flatten())

    if not all_pred_vel:
        return np.array([]), np.array([])

    return np.stack(all_pred_vel, axis=0), np.stack(all_gt_vel, axis=0)


def extract_perfect_tracks_acceleration(all_data: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract "perfect" tracks and compute their accelerations (second-order differences).

    Acceleration: a_t = v_{t+1} - v_t = pos_{t+2} - 2*pos_{t+1} + pos_t

    Returns:
        pred_matrix: (total_perfect_tracks, (T_pred-2) * 2)
        gt_matrix: (total_perfect_tracks, (T_pred-2) * 2)
    """
    all_pred_acc = []
    all_gt_acc = []

    for sample_data in all_data:
        pred_tracks = sample_data['pred_tracks']
        gt_tracks = sample_data['gt_tracks']
        visibility = sample_data.get('visibility')
        point_mask = sample_data['point_mask']
        num_point_cond = sample_data.get('num_point_cond', 0)

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

        pred_acc = np.diff(pred_pred, n=2, axis=0)
        gt_acc = np.diff(gt_pred, n=2, axis=0)

        for n in range(N):
            if point_mask is not None and not point_mask[n]:
                continue
            if vis_pred is not None and not np.all(vis_pred[:, n]):
                continue

            all_pred_acc.append(pred_acc[:, n, :].flatten())
            all_gt_acc.append(gt_acc[:, n, :].flatten())

    if not all_pred_acc:
        return np.array([]), np.array([])

    return np.stack(all_pred_acc, axis=0), np.stack(all_gt_acc, axis=0)


# =============================================================================
# Core Frechet Distance computation
# =============================================================================

def compute_frechet_distance_core(mu1: np.ndarray, sigma1: np.ndarray,
                                  mu2: np.ndarray, sigma2: np.ndarray,
                                  regularize: bool = False, eps: float = 1e-6) -> float:
    """
    Compute the Frechet Distance between two multivariate Gaussians.

    FD = ||mu1 - mu2||^2 + Tr(sigma1) + Tr(sigma2) - 2*Tr(sqrt(sigma1 @ sigma2))

    Args:
        mu1, mu2: Mean vectors
        sigma1, sigma2: Covariance matrices
        regularize: If True, add eps * I to covariances
        eps: Regularization constant

    Returns:
        Frechet distance (float)
    """
    from scipy.linalg import sqrtm

    if regularize:
        d = sigma1.shape[0]
        sigma1 = sigma1 + eps * np.eye(d)
        sigma2 = sigma2 + eps * np.eye(d)

    mean_diff_sq = np.sum((mu1 - mu2) ** 2)
    cov_product_sqrt = sqrtm(sigma1 @ sigma2)

    if np.iscomplexobj(cov_product_sqrt):
        cov_product_sqrt = cov_product_sqrt.real

    fd = mean_diff_sq + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(cov_product_sqrt)
    return float(fd)


def _fd_from_matrices(pred_matrix: np.ndarray, gt_matrix: np.ndarray,
                      regularize: bool = False, name: str = "FD") -> Optional[float]:
    """Compute Frechet Distance from feature matrices, with validation."""
    if pred_matrix.size == 0 or gt_matrix.size == 0:
        return None
    if pred_matrix.shape[0] < 2:
        return None

    mu_pred = np.mean(pred_matrix, axis=0)
    sigma_pred = np.cov(pred_matrix, rowvar=False)
    mu_gt = np.mean(gt_matrix, axis=0)
    sigma_gt = np.cov(gt_matrix, rowvar=False)

    if sigma_pred.ndim == 0:
        sigma_pred = np.array([[sigma_pred]])
        sigma_gt = np.array([[sigma_gt]])

    return compute_frechet_distance_core(mu_pred, sigma_pred, mu_gt, sigma_gt, regularize=regularize)


# =============================================================================
# Registered metrics
# =============================================================================

def compute_frechet_distance(all_data: List[dict], regularize: bool = False) -> Optional[float]:
    """
    Frechet Distance between predicted and GT track distributions (position space).

    Filters for "perfect" tracks (visible in ALL prediction frames), flattens each
    track to (T_pred * 2), fits Gaussians, computes FD.
    """
    if not all_data:
        return None
    pred_matrix, gt_matrix = extract_perfect_tracks_flattened(all_data)
    return _fd_from_matrices(pred_matrix, gt_matrix, regularize, "Frechet Distance")


def compute_frechet_distance_velocity(all_data: List[dict], regularize: bool = False) -> Optional[float]:
    """
    Frechet Distance on track VELOCITIES (first-order differences).

    Captures similarity in motion dynamics (speed, direction changes).
    """
    if not all_data:
        return None
    pred_matrix, gt_matrix = extract_perfect_tracks_velocity(all_data)
    return _fd_from_matrices(pred_matrix, gt_matrix, regularize, "Velocity FD")


def compute_frechet_distance_acceleration(all_data: List[dict], regularize: bool = False) -> Optional[float]:
    """
    Frechet Distance on track ACCELERATIONS (second-order differences).

    Captures similarity in motion changes (jerk, smoothness).
    """
    if not all_data:
        return None
    pred_matrix, gt_matrix = extract_perfect_tracks_acceleration(all_data)
    return _fd_from_matrices(pred_matrix, gt_matrix, regularize, "Acceleration FD")


def compute_variance_pred(all_data: List[dict]) -> Optional[float]:
    """
    Variance of predicted trajectories (position space).

    Captures both between-track spatial spread and within-track temporal variation.
    """
    if not all_data:
        return None
    pred_matrix, _ = extract_perfect_tracks_flattened(all_data)
    if pred_matrix.size == 0:
        return None
    return float(np.var(pred_matrix))


def compute_variance_gt(all_data: List[dict]) -> Optional[float]:
    """
    Variance of ground truth trajectories (position space).

    Reference variance -- predictions should ideally match this level of dynamism.
    """
    if not all_data:
        return None
    _, gt_matrix = extract_perfect_tracks_flattened(all_data)
    if gt_matrix.size == 0:
        return None
    return float(np.var(gt_matrix))


def compute_variance_pred_velocity(all_data: List[dict]) -> Optional[float]:
    """
    Variance of predicted trajectory velocities (first-order differences).

    Zero for a no-motion baseline, since velocity is zero everywhere.
    Captures how much motion/dynamics the predictions contain.
    """
    if not all_data:
        return None
    pred_matrix, _ = extract_perfect_tracks_velocity(all_data)
    if pred_matrix.size == 0:
        return None
    return float(np.var(pred_matrix))


def compute_variance_gt_velocity(all_data: List[dict]) -> Optional[float]:
    """Variance of ground truth trajectory velocities (first-order differences)."""
    if not all_data:
        return None
    _, gt_matrix = extract_perfect_tracks_velocity(all_data)
    if gt_matrix.size == 0:
        return None
    return float(np.var(gt_matrix))


def compute_variance_pred_acceleration(all_data: List[dict]) -> Optional[float]:
    """
    Variance of predicted trajectory accelerations (second-order differences).

    Zero for both no-motion and constant-velocity baselines.
    Captures how much the predictions change speed/direction.
    """
    if not all_data:
        return None
    pred_matrix, _ = extract_perfect_tracks_acceleration(all_data)
    if pred_matrix.size == 0:
        return None
    return float(np.var(pred_matrix))


def compute_variance_gt_acceleration(all_data: List[dict]) -> Optional[float]:
    """Variance of ground truth trajectory accelerations (second-order differences)."""
    if not all_data:
        return None
    _, gt_matrix = extract_perfect_tracks_acceleration(all_data)
    if gt_matrix.size == 0:
        return None
    return float(np.var(gt_matrix))
