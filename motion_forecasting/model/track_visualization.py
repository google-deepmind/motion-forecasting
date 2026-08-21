"""
Track visualization utilities for diffusion track prediction.

Provides standalone visualization functions for creating track overlay videos,
denoising process videos, and trajectory drawings. These functions operate on
tensors/arrays directly and do not depend on the model class.
"""

import torch
import numpy as np
import cv2

from .coordinate_utils import (
    model_to_tracks_format,
    compute_velocity_displacement_from_tracks,
)


def draw_track_trajectories(image, tracks, color=(255, 0, 0), thickness=3,
                            bbox_info=None, frame_start=None, orig_width=None, orig_height=None,
                            velocity_info=None, displacement_info=None, label=""):
    """
    Draw track trajectories showing how points move over time.
    Tracks are in [0,1] normalized coordinates, need to convert to pixel coordinates.
    
    Args:
        image: (H, W, 3) numpy array (modified in-place)
        tracks: (T, N, 2) track coordinates in [0,1] space
        color: RGB color tuple
        thickness: line thickness
        bbox_info: bbox metadata dictionary
        frame_start: starting frame index
        orig_width: original video width
        orig_height: original video height
        velocity_info: (2,) [vx, vy] average velocity or None
        displacement_info: (2,) [dx, dy] total displacement or None
        label: Text label to show
    """
    H, W = image.shape[:2]
    T, N, _ = tracks.shape
    
    # Convert normalized coordinates to pixel coordinates
    pixel_tracks = tracks.copy()
    
    if bbox_info is not None and frame_start is not None and orig_width is not None and orig_height is not None:
        frame_start_idx = frame_start if isinstance(frame_start, int) else frame_start.item()
        if frame_start_idx in bbox_info:
            bbox_padded = bbox_info[frame_start_idx]['padded_square']
            bbox_x = bbox_padded['x'].item() if hasattr(bbox_padded['x'], 'item') else bbox_padded['x']
            bbox_y = bbox_padded['y'].item() if hasattr(bbox_padded['y'], 'item') else bbox_padded['y']
            bbox_w = bbox_padded['w'].item() if hasattr(bbox_padded['w'], 'item') else bbox_padded['w']
            bbox_h = bbox_padded['h'].item() if hasattr(bbox_padded['h'], 'item') else bbox_padded['h']
            
            orig_w = orig_width if isinstance(orig_width, (int, float)) else orig_width.item()
            orig_h = orig_height if isinstance(orig_height, (int, float)) else orig_height.item()
            
            actual_crop_x = max(0, bbox_x)
            actual_crop_y = max(0, bbox_y)
            actual_crop_w = min(bbox_w, orig_w - actual_crop_x)
            actual_crop_h = min(bbox_h, orig_h - actual_crop_y)
            
            scale_x = (bbox_w / actual_crop_w) * W
            scale_y = (bbox_h / actual_crop_h) * H
            
            pixel_tracks[:, :, 0] *= scale_x
            pixel_tracks[:, :, 1] *= scale_y
        else:
            pixel_tracks[:, :, 0] *= W
            pixel_tracks[:, :, 1] *= H
    else:
        pixel_tracks[:, :, 0] *= W
        pixel_tracks[:, :, 1] *= H
    
    # Draw trajectories for each track point
    for n in range(N):
        trajectory_points = []
        for t in range(T):
            if not np.isnan(pixel_tracks[t, n]).any():
                pt = (int(pixel_tracks[t, n, 0]), int(pixel_tracks[t, n, 1]))
                if 0 <= pt[0] < W and 0 <= pt[1] < H:
                    trajectory_points.append(pt)
        
        # Draw trajectory line
        if len(trajectory_points) > 1:
            for i in range(len(trajectory_points) - 1):
                cv2.line(image, trajectory_points[i], trajectory_points[i+1], color, thickness)
        
        # Draw individual points with time-varying intensity
        for t, pt in enumerate(trajectory_points):
            intensity = int(255 * (t / max(1, T-1)))
            point_color = (int(color[0] * intensity/255), int(color[1] * intensity/255), int(color[2] * intensity/255))
            cv2.circle(image, pt, thickness + 1, point_color, -1)
    
    # Add text overlay
    if label or velocity_info is not None or displacement_info is not None:
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)
        font_thickness = 1
        
        if label:
            cv2.putText(image, label, (10, y_offset), font, font_scale, font_color, font_thickness)
            y_offset += 20
        
        if velocity_info is not None:
            vel_text = f"Vel: ({velocity_info[0]:.4f}, {velocity_info[1]:.4f})"
            cv2.putText(image, vel_text, (10, y_offset), font, font_scale, font_color, font_thickness)
            y_offset += 20
        
        if displacement_info is not None:
            disp_text = f"Disp: ({displacement_info[0]:.4f}, {displacement_info[1]:.4f})"
            cv2.putText(image, disp_text, (10, y_offset), font, font_scale, font_color, font_thickness)


def _convert_tracks_to_pixel(tracks_np, bbox_info, frame_start_info, orig_width_info, orig_height_info,
                              W_orig, H_orig, offset_x=0, offset_y=0):
    """
    Convert [0,1]-normalized tracks to pixel coordinates, accounting for bbox crop and canvas offset.
    
    Args:
        tracks_np: (T, N, 2) track coordinates in [0,1] space
        bbox_info, frame_start_info, orig_width_info, orig_height_info: metadata
        W_orig, H_orig: original image dimensions
        offset_x, offset_y: adaptive canvas offsets
        
    Returns:
        tracks_pixel: (T, N, 2) in pixel coordinates
    """
    tracks_pixel = tracks_np.copy()
    
    if bbox_info is not None and frame_start_info is not None and orig_width_info is not None and orig_height_info is not None:
        frame_start_idx = frame_start_info[0].item() if hasattr(frame_start_info[0], 'item') else frame_start_info[0]
        if frame_start_idx in bbox_info:
            bbox_padded = bbox_info[frame_start_idx]['padded_square']
            bbox_x = bbox_padded['x'].item() if hasattr(bbox_padded['x'], 'item') else bbox_padded['x']
            bbox_y = bbox_padded['y'].item() if hasattr(bbox_padded['y'], 'item') else bbox_padded['y']
            bbox_w = bbox_padded['w'].item() if hasattr(bbox_padded['w'], 'item') else bbox_padded['w']
            bbox_h = bbox_padded['h'].item() if hasattr(bbox_padded['h'], 'item') else bbox_padded['h']
            
            orig_w = orig_width_info[0].item() if hasattr(orig_width_info[0], 'item') else orig_width_info[0]
            orig_h = orig_height_info[0].item() if hasattr(orig_height_info[0], 'item') else orig_height_info[0]
            
            actual_crop_x = max(0, bbox_x)
            actual_crop_y = max(0, bbox_y)
            actual_crop_w = min(bbox_w, orig_w - actual_crop_x)
            actual_crop_h = min(bbox_h, orig_h - actual_crop_y)
            
            scale_x = (bbox_w / actual_crop_w) * W_orig
            scale_y = (bbox_h / actual_crop_h) * H_orig
            
            tracks_pixel[:, :, 0] *= scale_x
            tracks_pixel[:, :, 1] *= scale_y
        else:
            tracks_pixel[:, :, 0] *= W_orig
            tracks_pixel[:, :, 1] *= H_orig
    else:
        tracks_pixel[:, :, 0] *= W_orig
        tracks_pixel[:, :, 1] *= H_orig
    
    tracks_pixel[:, :, 0] += offset_x
    tracks_pixel[:, :, 1] += offset_y
    
    return tracks_pixel


def create_denoising_video(intermediate_results, gt_tracks, conditioning_image, 
                           final_tracks, bbox=None, frame_start=None,
                           orig_width=None, orig_height=None):
    """
    Create a video showing the track denoising process step by step.
    
    Args:
        intermediate_results: list of dicts with 'timestep' and 'tracks' keys
        gt_tracks: (B, T, N, 2) ground truth tracks in [0,1] space
        conditioning_image: (C, H, W) or (B, C, H, W) conditioning image tensor
        final_tracks: (B, T, N, 2) final predicted tracks in [0,1] space
        bbox, frame_start, orig_width, orig_height: metadata for coordinate conversion
        
    Returns:
        denoising_video: (1, T, C, H, W) numpy array
    """
    # Convert conditioning image to proper format
    if conditioning_image.dim() == 4:
        conditioning_image = conditioning_image[0]
    
    conditioning_image_np = conditioning_image.permute(1, 2, 0).cpu().numpy()
    
    if conditioning_image_np.max() <= 1.0:
        conditioning_image_np = (conditioning_image_np * 255).astype(np.uint8)
    else:
        conditioning_image_np = (conditioning_image_np / conditioning_image_np.max() * 255).astype(np.uint8)
    
    H, W = conditioning_image_np.shape[:2]
    
    frames = []
    gt_tracks_np = gt_tracks[0].cpu().numpy()
    
    for step_data in intermediate_results:
        timestep = step_data['timestep']
        pred_tracks = step_data['tracks'][0].cpu().numpy()
        
        frame = np.zeros((H, W * 2, 3), dtype=np.uint8)
        
        # Left side: predicted tracks
        left_img = conditioning_image_np.copy()
        draw_track_trajectories(left_img, pred_tracks, color=(255, 0, 0), thickness=3,
                               bbox_info=bbox, frame_start=frame_start,
                               orig_width=orig_width, orig_height=orig_height)
        frame[:, :W] = left_img
        
        # Right side: ground truth tracks
        right_img = conditioning_image_np.copy()
        draw_track_trajectories(right_img, gt_tracks_np, color=(0, 255, 0), thickness=3,
                               bbox_info=bbox, frame_start=frame_start,
                               orig_width=orig_width, orig_height=orig_height)
        frame[:, W:] = right_img
        
        # Add text overlay
        cv2.putText(frame, f"Diffusion Timestep: t={timestep}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Predicted Tracks", (10, H-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, "Ground Truth Tracks", (W+10, H-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        frames.append(frame)
    
    # Add final result frame
    final_tracks_np = final_tracks[0].cpu().numpy()
    final_frame = np.zeros((H, W * 2, 3), dtype=np.uint8)
    
    left_img = conditioning_image_np.copy()
    draw_track_trajectories(left_img, final_tracks_np, color=(255, 0, 0), thickness=3,
                           bbox_info=bbox, frame_start=frame_start,
                           orig_width=orig_width, orig_height=orig_height)
    final_frame[:, :W] = left_img
    
    right_img = conditioning_image_np.copy()
    draw_track_trajectories(right_img, gt_tracks_np, color=(0, 255, 0), thickness=3,
                           bbox_info=bbox, frame_start=frame_start,
                           orig_width=orig_width, orig_height=orig_height)
    final_frame[:, W:] = right_img
    
    cv2.putText(final_frame, "Final Result (t=0)", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(final_frame, "Predicted Tracks", (10, H-20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(final_frame, "Ground Truth Tracks", (W+10, H-20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    frames.append(final_frame)
    
    # Convert to video format: (1, T, C, H, W)
    denoising_video = np.stack(frames, axis=0)
    denoising_video = denoising_video.transpose(0, 3, 1, 2)  # (T, C, H, W)
    denoising_video = denoising_video[np.newaxis, ...]  # (1, T, C, H, W)
    
    
    return denoising_video


def plot_stabilized_tracks_video(video_frames, gt_tracks, pred_tracks,
                                  condition_on_displacement,
                                  point_mask=None,
                                  bbox_info=None, frame_start_info=None,
                                  orig_width_info=None, orig_height_info=None,
                                  fps_stride_info=None,
                                  gt_visibility=None, pred_visibility=None):
    """
    Create a 3-row, 3-column video visualization for stabilized tracks with ADAPTIVE CANVAS SIZING.
    
    Top row: Original video (animated)
    Middle row: GT tracks (green/blue for visible/occluded) - black bg, first frame, with tails
    Bottom row: Predicted tracks (red/blue) - black bg, first frame, with tails
    
    Args:
        video_frames: (B, T, C, H, W) video tensor in [0, 255] range
        gt_tracks: (B, T, N, 2) ground truth tracks in [0,1] normalized space
        pred_tracks: (B, T, N, 2) predicted tracks in [0,1] normalized space
        condition_on_displacement: bool, whether displacement conditioning is enabled
        point_mask: Optional (B, N) boolean mask for valid points
        bbox_info, frame_start_info, orig_width_info, orig_height_info: metadata
        fps_stride_info: FPS stride for mapping video frames to track timesteps
        gt_visibility: Optional (B, T, N) visibility mask for GT tracks
        pred_visibility: Optional (B, T, N) visibility mask for predicted tracks
        
    Returns:
        composite_video: (1, T, 3, H*3, W*3) numpy array
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from motion_forecasting.utils.flow_utils import compute_adaptive_canvas_size
    
    B, T, C, H, W = video_frames.shape
    H_orig, W_orig = H, W
    
    # Extract FPS stride
    fps_stride = 1
    if fps_stride_info is not None:
        if isinstance(fps_stride_info, (list, tuple, torch.Tensor)):
            fps_stride = fps_stride_info[0].item() if hasattr(fps_stride_info[0], 'item') else int(fps_stride_info[0])
        else:
            fps_stride = fps_stride_info.item() if hasattr(fps_stride_info, 'item') else int(fps_stride_info)
    
    num_track_ts = gt_tracks.shape[1]
    # Compute adaptive canvas size
    canvas_size, offset_x, offset_y = compute_adaptive_canvas_size(
        gt_tracks, img_size=H_orig, normalization='first_bbox',
        bbox=bbox_info, orig_width=orig_width_info, orig_height=orig_height_info,
        frame_start=frame_start_info, padding_ratio=0.2
    )
    
    # Pad video frames
    if canvas_size != H_orig:
        pad_left = int(np.floor(offset_x))
        pad_right = canvas_size - H_orig - pad_left
        pad_top = int(np.floor(offset_y))
        pad_bottom = canvas_size - W_orig - pad_top
        
        video_frames_padded = torch.nn.functional.pad(
            video_frames.reshape(B * T, C, H_orig, W_orig),
            (pad_left, pad_right, pad_top, pad_bottom),
            mode='constant', value=0
        ).reshape(B, T, C, canvas_size, canvas_size)
        H, W = canvas_size, canvas_size
    else:
        video_frames_padded = video_frames
    
    # Extract data as numpy
    video_np = video_frames_padded[0].cpu().numpy()  # (T, C, H, W)
    gt_tracks_np = gt_tracks[0].cpu().numpy()
    pred_tracks_np = pred_tracks[0].cpu().numpy()
    
    # Handle visibility masks
    if gt_visibility is not None:
        gt_visibility_np = gt_visibility[0].cpu().numpy()
    else:
        gt_visibility_np = np.ones((gt_tracks_np.shape[0], gt_tracks_np.shape[1]), dtype=np.float32)
    
    if pred_visibility is not None:
        pred_visibility_np = pred_visibility[0].cpu().numpy()
    else:
        pred_visibility_np = np.ones((pred_tracks_np.shape[0], pred_tracks_np.shape[1]), dtype=np.float32)
    
    # Handle point masking
    if point_mask is not None:
        mask_np = point_mask[0].cpu().numpy()
        gt_tracks_np = gt_tracks_np[:, mask_np, :]
        pred_tracks_np = pred_tracks_np[:, mask_np, :]
        gt_visibility_np = gt_visibility_np[:, mask_np]
        pred_visibility_np = pred_visibility_np[:, mask_np]
    
    # Convert video from (T, C, H, W) to (T, H, W, C)
    video_np = video_np.transpose(0, 2, 3, 1)
    if video_np.max() <= 1.0:
        video_np = (video_np * 255).astype(np.uint8)
    else:
        video_np = video_np.astype(np.uint8)
    
    # Pre-compute pixel coordinates
    gt_tracks_pixel = _convert_tracks_to_pixel(
        gt_tracks_np, bbox_info, frame_start_info, orig_width_info, orig_height_info,
        W_orig, H_orig, offset_x, offset_y
    )
    pred_tracks_pixel = _convert_tracks_to_pixel(
        pred_tracks_np, bbox_info, frame_start_info, orig_width_info, orig_height_info,
        W_orig, H_orig, offset_x, offset_y
    )
    
    # Compute velocity and displacement for text overlay
    gt_tracks_batch = torch.from_numpy(gt_tracks_np).unsqueeze(0)
    pred_tracks_batch = torch.from_numpy(pred_tracks_np).unsqueeze(0)
    gt_vis_t = torch.ones(1, num_track_ts, gt_tracks_np.shape[1])
    pred_vis_t = torch.ones(1, num_track_ts, pred_tracks_np.shape[1])
    gt_pm = torch.ones(1, gt_tracks_np.shape[1], dtype=torch.bool)
    pred_pm = torch.ones(1, pred_tracks_np.shape[1], dtype=torch.bool)
    
    gt_velocity, gt_displacement = compute_velocity_displacement_from_tracks(gt_tracks_batch, gt_vis_t, gt_pm)
    pred_velocity, pred_displacement = compute_velocity_displacement_from_tracks(pred_tracks_batch, pred_vis_t, pred_pm)
    
    gt_velocity_np = gt_velocity[0].cpu().numpy()
    gt_displacement_np = gt_displacement[0].cpu().numpy()
    pred_velocity_np = pred_velocity[0].cpu().numpy()
    pred_displacement_np = pred_displacement[0].cpu().numpy()
    
    # Setup matplotlib
    figure_dpi = 64
    composite_frames = []
    black_bg = np.zeros((H, W, 3), dtype=np.uint8)
    
    for frame_idx in range(T):
        track_idx = min(frame_idx // fps_stride, num_track_ts - 1)
        
        fig = plt.figure(
            figsize=(W * 3 / figure_dpi, (H * 3) / figure_dpi),
            dpi=figure_dpi, frameon=False, facecolor='w',
        )
        
        # --- Top row: Original video ---
        for col_idx in range(1, 4):
            ax = plt.subplot(3, 3, col_idx)
            ax.axis('off')
            ax.imshow(video_np[frame_idx])
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)
        
        # --- Middle row: GT tracks ---
        gt_x = gt_tracks_pixel[track_idx, :, 0]
        gt_y = gt_tracks_pixel[track_idx, :, 1]
        gt_vis_current = gt_visibility_np[track_idx, :]
        visible_mask = gt_vis_current > 0.5
        occluded_mask = ~visible_mask
        
        # Middle left: black bg with GT points
        ax_ml = plt.subplot(3, 3, 4)
        ax_ml.axis('off')
        ax_ml.imshow(black_bg)
        ax_ml.set_xlim(0, W); ax_ml.set_ylim(H, 0)
        if visible_mask.any():
            ax_ml.scatter(gt_x[visible_mask], gt_y[visible_mask], c='green', s=30, alpha=0.8, edgecolors='white', linewidths=1)
        if occluded_mask.any():
            ax_ml.scatter(gt_x[occluded_mask], gt_y[occluded_mask], c='deepskyblue', s=30, alpha=0.8, edgecolors='white', linewidths=1)
        ax_ml.text(10, 30, 'Ground Truth', fontsize=12, color='white',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='green', alpha=0.7), verticalalignment='top')
        
        # Middle center: first frame with GT points
        ax_mc = plt.subplot(3, 3, 5)
        ax_mc.axis('off')
        ax_mc.imshow(video_np[0])
        ax_mc.set_xlim(0, W); ax_mc.set_ylim(H, 0)
        if visible_mask.any():
            ax_mc.scatter(gt_x[visible_mask], gt_y[visible_mask], c='green', s=30, alpha=0.8, edgecolors='white', linewidths=1)
        if occluded_mask.any():
            ax_mc.scatter(gt_x[occluded_mask], gt_y[occluded_mask], c='deepskyblue', s=30, alpha=0.8, edgecolors='white', linewidths=1)
        ax_mc.text(10, 30, 'Ground Truth', fontsize=12, color='white',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='green', alpha=0.7), verticalalignment='top')
        
        # Middle right: first frame with GT tails
        ax_mr = plt.subplot(3, 3, 6)
        ax_mr.axis('off')
        ax_mr.imshow(video_np[0])
        ax_mr.set_xlim(0, W); ax_mr.set_ylim(H, 0)
        
        if track_idx > 0:
            green_segments = []
            blue_segments = []
            for n in range(gt_tracks_pixel.shape[1]):
                trajectory = gt_tracks_pixel[:track_idx + 1, n, :]
                vis = gt_visibility_np[:track_idx + 1, n]
                if len(trajectory) > 1:
                    points = trajectory.reshape(-1, 1, 2)
                    segs = np.concatenate([points[:-1], points[1:]], axis=1)
                    if vis.min() < 0.5:
                        blue_segments.append(segs)
                    else:
                        green_segments.append(segs)
            
            if green_segments:
                lc = LineCollection(np.concatenate(green_segments, axis=0), colors='green', linewidths=2, alpha=0.6)
                ax_mr.add_collection(lc)
            if blue_segments:
                lc = LineCollection(np.concatenate(blue_segments, axis=0), colors='deepskyblue', linewidths=2, alpha=0.6)
                ax_mr.add_collection(lc)
        
        if visible_mask.any():
            ax_mr.scatter(gt_x[visible_mask], gt_y[visible_mask], c='green', s=30, alpha=0.8, edgecolors='white', linewidths=1)
        if occluded_mask.any():
            ax_mr.scatter(gt_x[occluded_mask], gt_y[occluded_mask], c='deepskyblue', s=30, alpha=0.8, edgecolors='white', linewidths=1)
        
        y_text = 30
        ax_mr.text(10, y_text, 'Ground Truth', fontsize=10, color='white',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7), verticalalignment='top')
        y_text += 25
        if condition_on_displacement and gt_displacement_np is not None:
            ax_mr.text(10, y_text, f'Disp: ({gt_displacement_np[0]:.4f}, {gt_displacement_np[1]:.4f})', fontsize=8, color='white',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7), verticalalignment='top')
        
        # --- Bottom row: Predicted tracks ---
        pred_x = pred_tracks_pixel[track_idx, :, 0]
        pred_y = pred_tracks_pixel[track_idx, :, 1]
        pred_vis_current = pred_visibility_np[track_idx, :]
        pred_visible_mask = pred_vis_current > 0.5
        pred_occluded_mask = ~pred_visible_mask
        
        # Bottom left: black bg with pred points
        ax_bl = plt.subplot(3, 3, 7)
        ax_bl.axis('off')
        ax_bl.imshow(black_bg)
        ax_bl.set_xlim(0, W); ax_bl.set_ylim(H, 0)
        if pred_visible_mask.any():
            ax_bl.scatter(pred_x[pred_visible_mask], pred_y[pred_visible_mask], c='red', s=30, alpha=0.8, edgecolors='white', linewidths=1)
        if pred_occluded_mask.any():
            ax_bl.scatter(pred_x[pred_occluded_mask], pred_y[pred_occluded_mask], c='deepskyblue', s=30, alpha=0.8, edgecolors='white', linewidths=1)
        ax_bl.text(10, 30, 'Predicted', fontsize=12, color='white',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.7), verticalalignment='top')
        
        # Bottom center: first frame with pred points
        ax_bc = plt.subplot(3, 3, 8)
        ax_bc.axis('off')
        ax_bc.imshow(video_np[0])
        ax_bc.set_xlim(0, W); ax_bc.set_ylim(H, 0)
        if pred_visible_mask.any():
            ax_bc.scatter(pred_x[pred_visible_mask], pred_y[pred_visible_mask], c='red', s=30, alpha=0.8, edgecolors='white', linewidths=1)
        if pred_occluded_mask.any():
            ax_bc.scatter(pred_x[pred_occluded_mask], pred_y[pred_occluded_mask], c='deepskyblue', s=30, alpha=0.8, edgecolors='white', linewidths=1)
        ax_bc.text(10, 30, 'Predicted', fontsize=12, color='white',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.7), verticalalignment='top')
        
        # Bottom right: first frame with pred tails
        ax_br = plt.subplot(3, 3, 9)
        ax_br.axis('off')
        ax_br.imshow(video_np[0])
        ax_br.set_xlim(0, W); ax_br.set_ylim(H, 0)
        
        if track_idx > 0:
            red_segments = []
            blue_segments = []
            for n in range(pred_tracks_pixel.shape[1]):
                trajectory = pred_tracks_pixel[:track_idx + 1, n, :]
                vis = pred_visibility_np[:track_idx + 1, n]
                if len(trajectory) > 1:
                    points = trajectory.reshape(-1, 1, 2)
                    segs = np.concatenate([points[:-1], points[1:]], axis=1)
                    if vis.min() < 0.5:
                        blue_segments.append(segs)
                    else:
                        red_segments.append(segs)
            
            if red_segments:
                lc = LineCollection(np.concatenate(red_segments, axis=0), colors='red', linewidths=2, alpha=0.6)
                ax_br.add_collection(lc)
            if blue_segments:
                lc = LineCollection(np.concatenate(blue_segments, axis=0), colors='deepskyblue', linewidths=2, alpha=0.6)
                ax_br.add_collection(lc)
        
        if pred_visible_mask.any():
            ax_br.scatter(pred_x[pred_visible_mask], pred_y[pred_visible_mask], c='red', s=30, alpha=0.8, edgecolors='white', linewidths=1)
        if pred_occluded_mask.any():
            ax_br.scatter(pred_x[pred_occluded_mask], pred_y[pred_occluded_mask], c='deepskyblue', s=30, alpha=0.8, edgecolors='white', linewidths=1)
        
        y_text = 30
        ax_br.text(10, y_text, 'Predicted', fontsize=10, color='white',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7), verticalalignment='top')
        y_text += 25
        if condition_on_displacement and pred_displacement_np is not None:
            ax_br.text(10, y_text, f'Disp: ({pred_displacement_np[0]:.4f}, {pred_displacement_np[1]:.4f})', fontsize=8, color='white',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7), verticalalignment='top')
        
        # Tight layout
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.05, wspace=0.05)
        plt.margins(0, 0)
        
        # Convert to numpy
        fig.canvas.draw()
        try:
            buf = fig.canvas.buffer_rgba()
            img = np.asarray(buf)[:, :, :3]
        except AttributeError:
            width, height = fig.get_size_inches() * fig.get_dpi()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        composite_frames.append(np.copy(img))
        
        plt.close(fig)
    
    # Stack frames and convert to (1, T, C, H, W)
    composite_video = np.stack(composite_frames, axis=0)
    composite_video = composite_video.transpose(0, 3, 1, 2)  # (T, C, H, W)
    composite_video = composite_video[np.newaxis, ...]  # (1, T, C, H, W)
    
    pass
    
    return composite_video
