"""
MVD (Motion Vector Distance) metric.

Per-example metric that measures how well predicted track motion patterns match
ground truth by comparing spatiotemporal motion histograms. For each set of tracks:

1. The spatial domain is divided into a grid of cells (grid_size / cell_size).
2. The temporal domain is divided into cubes of cube_frames consecutive frames.
3. For each point (assigned to a cell by its initial position), velocity (frame-to-frame
   displacement) and acceleration (velocity differences) vectors are computed.
4. Each motion vector is decomposed into an angle bin and a log-scaled magnitude.
5. A histogram over angle bins (weighted by magnitude) is accumulated per
   (time_cube, spatial_cell) and normalized by the number of visible points.

The final MVD score is the L2 distance between the concatenated
(velocity + acceleration) histogram features of the predicted and GT tracks.
Lower is better.
"""

import numpy as np

def _compute_track_motion_histogram(
   tracks: np.ndarray,
   visibility: np.ndarray,
   point_mask: np.ndarray,
   grid_size: int = 256,
   cell_size: int = 8,
   cube_frames: int = 8,
   angle_bins: int = 8,
   magnitude_bins: int = 256,
   motion_type: str = 'velocity',
) -> np.ndarray:
 T, N, _ = tracks.shape


 grid_cells = grid_size // cell_size
 num_time_cubes = (
     max(1, (T - 1) // cube_frames)
     if motion_type == 'velocity'
     else max(1, (T - 2) // cube_frames)
 )


 vis = (
     visibility > 0.5
     if visibility is not None
     else np.ones((T, N), dtype=bool)
 )
 pmask = point_mask if point_mask is not None else np.ones((N,), dtype=bool)


 init_pos = tracks[0, :, :]
 cell_x = np.clip(
     (init_pos[:, 0] // cell_size).astype(np.int32), 0, grid_cells - 1
 )
 cell_y = np.clip(
     (init_pos[:, 1] // cell_size).astype(np.int32), 0, grid_cells - 1
 )


 if motion_type == 'velocity':
   motion = tracks[1:] - tracks[:-1]
   motion_vis = vis[1:] & vis[:-1]
 else:
   vel = tracks[1:] - tracks[:-1]
   motion = vel[1:] - vel[:-1]
   motion_vis = vis[2:] & vis[1:-1] & vis[:-2]


 T_motion = motion.shape[0]
 usable_frames = num_time_cubes * cube_frames
 if usable_frames > T_motion:
   num_time_cubes = T_motion // cube_frames
   usable_frames = num_time_cubes * cube_frames
 if num_time_cubes == 0:
   return np.zeros(grid_cells * grid_cells * angle_bins)


 motion = motion[:usable_frames]
 motion_vis = motion_vis[:usable_frames]
 motion_vis = motion_vis & pmask[np.newaxis, :]


 angles = np.arctan2(motion[:, :, 0], motion[:, :, 1])
 angle_bin_arr = np.clip(
     ((angles + np.pi) // (2 * np.pi / angle_bins)).astype(np.int32),
     0,
     angle_bins - 1,
 )


 mag = np.linalg.norm(motion, axis=2)
 mag = np.clip(mag, 0, magnitude_bins - 1)
 mag = np.log2(mag + 1)
 mag = np.clip(mag, 0, np.log2(magnitude_bins))
 mag = np.ceil(mag)
 mag = mag / np.log2(magnitude_bins)
 mag = mag * motion_vis


 time_cube_idx = np.arange(usable_frames) // cube_frames
 tc_arr = np.broadcast_to(time_cube_idx[:, np.newaxis], (usable_frames, N))
 cx_arr = np.broadcast_to(cell_x[np.newaxis, :], (usable_frames, N))
 cy_arr = np.broadcast_to(cell_y[np.newaxis, :], (usable_frames, N))


 flat_cell_idx = (
     tc_arr * (grid_cells * grid_cells) + cy_arr * grid_cells + cx_arr
 )
 num_cells = num_time_cubes * grid_cells * grid_cells


 flat_cell_idx_flat = flat_cell_idx.ravel()
 motion_vis_flat = motion_vis.ravel()
 mag_flat = mag.ravel()
 angle_bin_flat = angle_bin_arr.ravel()


 counts = np.bincount(
     flat_cell_idx_flat,
     weights=motion_vis_flat.astype(np.float64),
     minlength=num_cells,
 ).reshape(num_time_cubes, grid_cells, grid_cells)


 hist = np.zeros(
     (num_time_cubes, grid_cells, grid_cells, angle_bins), dtype=np.float64
 )
 for b in range(angle_bins):
   bin_mask = angle_bin_flat == b
   weights = mag_flat * bin_mask
   hist[:, :, :, b] = np.bincount(
       flat_cell_idx_flat,
       weights=weights,
       minlength=num_cells,
   ).reshape(num_time_cubes, grid_cells, grid_cells)


 nonzero = counts > 0
 for b in range(angle_bins):
   hist[:, :, :, b][nonzero] /= counts[nonzero]


 return hist.reshape(-1)




def compute_mvd_distance(
   pred_tracks: np.ndarray,
   gt_tracks: np.ndarray,
   visibility: np.ndarray,
   pred_visibility: np.ndarray,
   point_mask: np.ndarray,
   pixel_info: dict,
   use_bfloat16: bool = True,
   num_point_cond: int = None,
) -> float:
 grid_size = 256
 cell_size = 8
 cube_frames = 8
 angle_bins = 8


 T, N, _ = pred_tracks.shape
 gt_vis = (
     visibility > 0.5
     if visibility is not None
     else np.ones((T, N), dtype=bool)
 )
 pred_vis = (
     pred_visibility > 0.5
     if pred_visibility is not None
     else np.ones((T, N), dtype=bool)
 )
 pmask = point_mask if point_mask is not None else np.ones((N,), dtype=bool)


 all_vis_mask = np.zeros((T, N), dtype=bool)
 all_vis_mask |= gt_vis & pmask[np.newaxis, :]
 all_vis_mask |= pred_vis & pmask[np.newaxis, :]


 all_visible_pts = np.concatenate(
     [
         gt_tracks[all_vis_mask],
         pred_tracks[all_vis_mask],
     ],
     axis=0,
 )


 if all_visible_pts.shape[0] == 0:
   return 0.0


 coord_min = all_visible_pts.min(axis=0)
 coord_max = all_visible_pts.max(axis=0)
 coord_range = coord_max - coord_min
 coord_range = np.where(coord_range < 1e-8, 1.0, coord_range)


 pred_tracks = (pred_tracks - coord_min) / coord_range * grid_size
 gt_tracks = (gt_tracks - coord_min) / coord_range * grid_size


 vel_hist_pred = _compute_track_motion_histogram(
     pred_tracks,
     pred_visibility,
     point_mask,
     grid_size=grid_size,
     cell_size=cell_size,
     cube_frames=cube_frames,
     angle_bins=angle_bins,
     motion_type='velocity',
 )
 vel_hist_gt = _compute_track_motion_histogram(
     gt_tracks,
     visibility,
     point_mask,
     grid_size=grid_size,
     cell_size=cell_size,
     cube_frames=cube_frames,
     angle_bins=angle_bins,
     motion_type='velocity',
 )
 acc_hist_pred = _compute_track_motion_histogram(
     pred_tracks,
     pred_visibility,
     point_mask,
     grid_size=grid_size,
     cell_size=cell_size,
     cube_frames=cube_frames,
     angle_bins=angle_bins,
     motion_type='acceleration',
 )
 acc_hist_gt = _compute_track_motion_histogram(
     gt_tracks,
     visibility,
     point_mask,
     grid_size=grid_size,
     cell_size=cell_size,
     cube_frames=cube_frames,
     angle_bins=angle_bins,
     motion_type='acceleration',
 )


 feat_pred = np.concatenate([vel_hist_pred, acc_hist_pred])
 feat_gt = np.concatenate([vel_hist_gt, acc_hist_gt])


 return float(np.linalg.norm(feat_pred - feat_gt))



