"""
Track canvas utilities: compute pixel-space track bounds and the adaptive
canvas needed to render tracks that extend beyond the image crop.

Used by track_visualization.plot_stabilized_tracks_video.
"""

import numpy as np
import torch


def compute_track_bounds_in_pixels(tracks, img_size, normalization=None, bbox=None, orig_width=None, orig_height=None, frame_start=None):
    """
    Compute the bounding box of tracks in pixel space (after unnormalization).

    Args:
        tracks: (B, T, N, 2) normalized tracks
        img_size: target image size (usually 256)
        normalization: normalization method used
        bbox, orig_width, orig_height, frame_start: unnormalization parameters

    Returns:
        min_x, min_y, max_x, max_y: bounds in pixel space (256x256 coordinate system)
    """
    B, T, N, C = tracks.shape
    tracks_copy = tracks.clone()
    generation_size = 256

    # Handle normalization parameter
    if isinstance(normalization, (list, tuple, torch.Tensor)):
        normalization = normalization[0]

    # Convert normalized tracks to pixel space (bbox-aware, never tracks * img_size)
    if normalization == "first_bbox" or normalization == "per_frame_bbox":
        if bbox is not None and frame_start is not None and orig_width is not None and orig_height is not None:
            frame_start_idx = frame_start[0].item() if hasattr(frame_start[0], 'item') else frame_start[0]
            if frame_start_idx in bbox:
                bbox_padded = bbox[frame_start_idx]['padded_square']
                bbox_x = bbox_padded['x'].item() if hasattr(bbox_padded['x'], 'item') else bbox_padded['x']
                bbox_y = bbox_padded['y'].item() if hasattr(bbox_padded['y'], 'item') else bbox_padded['y']
                bbox_w = bbox_padded['w'].item() if hasattr(bbox_padded['w'], 'item') else bbox_padded['w']
                bbox_h = bbox_padded['h'].item() if hasattr(bbox_padded['h'], 'item') else bbox_padded['h']

                orig_w = orig_width[0].item() if hasattr(orig_width[0], 'item') else orig_width[0]
                orig_h = orig_height[0].item() if hasattr(orig_height[0], 'item') else orig_height[0]

                # Compute actual extracted crop size
                actual_crop_x = max(0, bbox_x)
                actual_crop_y = max(0, bbox_y)
                actual_crop_w = min(bbox_w, orig_w - actual_crop_x)
                actual_crop_h = min(bbox_h, orig_h - actual_crop_y)

                # Scale factor
                scale_x = (bbox_w / actual_crop_w) * generation_size
                scale_y = (bbox_h / actual_crop_h) * generation_size

                tracks_copy[:, :, :, 0] = tracks_copy[:, :, :, 0] * scale_x
                tracks_copy[:, :, :, 1] = tracks_copy[:, :, :, 1] * scale_y
            else:
                tracks_copy = tracks_copy * generation_size
        else:
            tracks_copy = tracks_copy * generation_size
    elif normalization == "whole_image_dim":
        frame_start_val = frame_start[0].item()
        bbox_padded = bbox[frame_start_val]['padded_square']
        x, y, w, h = bbox_padded['x'], bbox_padded['y'], bbox_padded['w'], bbox_padded['h']
        x, y, w, h = x.to(tracks.device), y.to(tracks.device), w.to(tracks.device), h.to(tracks.device)

        tracks_copy[:, :, :, 0] = tracks_copy[:, :, :, 0] * orig_width
        tracks_copy[:, :, :, 1] = tracks_copy[:, :, :, 1] * orig_height
        tracks_copy[:, :, :, 0] = tracks_copy[:, :, :, 0] - x
        tracks_copy[:, :, :, 1] = tracks_copy[:, :, :, 1] - y
        tracks_copy[:, :, :, 0] = tracks_copy[:, :, :, 0] / w
        tracks_copy[:, :, :, 1] = tracks_copy[:, :, :, 1] / h
        tracks_copy = tracks_copy * generation_size
    else:
        tracks_copy = tracks_copy * generation_size

    # Compute bounds
    min_x = tracks_copy[:, :, :, 0].min().item()
    max_x = tracks_copy[:, :, :, 0].max().item()
    min_y = tracks_copy[:, :, :, 1].min().item()
    max_y = tracks_copy[:, :, :, 1].max().item()

    return min_x, min_y, max_x, max_y


def compute_adaptive_canvas_size(gt_tracks, img_size=256, normalization=None, bbox=None, orig_width=None, orig_height=None, frame_start=None, padding_ratio=0.2):
    """
    Compute adaptive canvas size based on GT track extent.

    Args:
        gt_tracks: (B, T, N, 2) ground truth tracks
        img_size: base image size (256)
        padding_ratio: fraction of extent to add as padding (default 0.2 = 20%)

    Returns:
        adaptive_size: int, the canvas size to use
        offset_x, offset_y: offsets to center the original 256x256 region
    """
    min_x, min_y, max_x, max_y = compute_track_bounds_in_pixels(
        gt_tracks, img_size, normalization, bbox, orig_width, orig_height, frame_start
    )

    # Compute extent
    extent_x = max_x - min_x
    extent_y = max_y - min_y

    # Add padding
    padding_x = extent_x * padding_ratio
    padding_y = extent_y * padding_ratio

    # Compute required canvas size (must fit both original 256 region and all tracks with padding)
    min_x_padded = min(min_x - padding_x, 0)
    max_x_padded = max(max_x + padding_x, img_size)
    min_y_padded = min(min_y - padding_y, 0)
    max_y_padded = max(max_y + padding_y, img_size)

    canvas_width = int(np.ceil(max_x_padded - min_x_padded))
    canvas_height = int(np.ceil(max_y_padded - min_y_padded))

    # Use square canvas (max dimension)
    adaptive_size = max(canvas_width, canvas_height, img_size)

    # Compute offsets for track positioning on the canvas
    # This handles: (1) negative track coords, (2) centering content on square canvas
    # Tracks in [0, img_size] pixel space will be positioned at [offset_x, offset_x + img_size]
    offset_x = -min_x_padded + (adaptive_size - canvas_width) / 2.0
    offset_y = -min_y_padded + (adaptive_size - canvas_height) / 2.0

    return adaptive_size, offset_x, offset_y
