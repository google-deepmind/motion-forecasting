"""
Animal-color track video rendering (paper-style visualization).

Point colors are texture-sampled: for each point, a small circle around its
t=0 position on the first frame is sampled and the mean RGB becomes that
point's color for the whole video (GT and predictions).

Video timeline (at video_fps, default 30):
  Phase 1  [0.0s-0.5s]:   Static first frame at full opacity.
  Phase 2  [0.5s-1.25s]:  First frame fades to 0.3 alpha (toward white);
                          texture-colored starting-point circles fade in.
  Phase 3  [1.25s-1.5s]:  Dots remain texture-colored; static pause.
  Phase 4  [1.5s-...]:    Play track timesteps at track_fps (default 15)
                          with tail_length trailing timesteps (default 5).

Entry points:
  - generate_track_video(): render one video from pixel-space tracks.
  - generate_animal_color_videos_for_sample(): batch-format driver used by
    engine/eval.py with visualize=true (GT + prediction videos).
"""

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


# =====================================================================
# Color helpers
# =====================================================================

def _expand_colors(colors_real: np.ndarray, point_mask: np.ndarray) -> np.ndarray:
    """Map (N_real, 3) colours to (N_total, 3) using point_mask."""
    N_total = point_mask.shape[0]
    full = np.zeros((N_total, 3))
    full[np.where(point_mask)[0]] = colors_real
    return full


def _color_to_rgb_uint8(c) -> Tuple[int, int, int]:
    """Convert a [0,1] float RGB triplet to an int tuple for cv2."""
    return (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))


def sample_texture_colors(
    first_frame: np.ndarray,
    positions_xy: np.ndarray,
    point_mask: np.ndarray,
    radius: int,
) -> np.ndarray:
    """Sample mean RGB in a small circle around each point's t=0 position.

    Used to assign a single texture-derived color per point for the whole
    video (GT and predictions). Out-of-bounds or empty patches get neutral grey.

    Args:
        first_frame: (H, W, 3) uint8 RGB.
        positions_xy: (N, 2) pixel coords (x, y) at t=0.
        point_mask: (N,) bool — which points to sample.
        radius: radius of sampling circle in pixels (0 = center pixel only).

    Returns:
        (N, 3) float in [0, 1]; masked-out indices get (0.5, 0.5, 0.5).
    """
    H, W = first_frame.shape[:2]
    N = positions_xy.shape[0]
    fallback = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    real_idx = np.where(point_mask)[0]
    if len(real_idx) == 0:
        return np.tile(fallback, (N, 1))

    colors_real = np.zeros((len(real_idx), 3), dtype=np.float64)
    radius = max(0, int(radius))

    for i, n in enumerate(real_idx):
        cx = int(round(positions_xy[n, 0]))
        cy = int(round(positions_xy[n, 1]))
        y_lo = max(0, cy - radius)
        y_hi = min(H, cy + radius + 1)
        x_lo = max(0, cx - radius)
        x_hi = min(W, cx + radius + 1)
        if y_lo >= y_hi or x_lo >= x_hi:
            colors_real[i] = fallback
            continue
        patch = first_frame[y_lo:y_hi, x_lo:x_hi].astype(np.float64)
        if radius > 0:
            yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
            in_circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
            patch = patch.copy()
            patch[~in_circle] = np.nan
        with np.errstate(invalid='ignore'):
            mean_rgb = np.nanmean(patch.reshape(-1, 3), axis=0)
        if np.any(np.isnan(mean_rgb)):
            colors_real[i] = fallback
        else:
            colors_real[i] = np.clip(mean_rgb / 255.0, 0.0, 1.0)

    return _expand_colors(colors_real, point_mask)


# =====================================================================
# Texture dot splatting (bigger points with sampled patches from first frame)
# =====================================================================

def _plot_tracks_dots_inner(
    rgb: np.ndarray,
    points: np.ndarray,
    occluded: np.ndarray,
    dot_size: float,
    bg: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Render track dots by splatting texture patches from rgb[0] at each timestep.

    Args:
        rgb: (T, H, W, 3) uint8; patches are sampled from rgb[0] at each point's t=0 position.
        points: (N, T, 2) pixel coords (x, y) per point per frame.
        occluded: (N, T) float; >0.5 means occluded (dot not drawn).
        dot_size: diameter of each dot in pixels (soft circle).
        bg: optional (T, H, W, 3) uint8; if set, composite splatted dots onto bg instead of white.

    Returns:
        (T, H, W, 3) uint8 RGB video.
    """
    num_points, num_frames = points.shape[:2]
    height, width = rgb.shape[1], rgb.shape[2]
    patch_diam = int(np.ceil(dot_size))
    if patch_diam % 2 == 0:
        patch_diam += 1
    radius = patch_diam // 2

    yy, xx = np.mgrid[:patch_diam, :patch_diam]
    dist = np.sqrt((yy - radius) ** 2 + (xx - radius) ** 2)
    circle_mask = np.clip(dot_size / 2.0 - dist, 0.0, 1.0)

    cx = np.round(points[:, 0, 0]).astype(np.int32)
    cy = np.round(points[:, 0, 1]).astype(np.int32)
    y0 = cy - radius
    x0 = cx - radius
    valid_start = (
        (y0 >= 0)
        & (x0 >= 0)
        & (y0 + patch_diam <= height)
        & (x0 + patch_diam <= width)
    )

    y0_safe = np.clip(y0, 0, height - patch_diam)
    x0_safe = np.clip(x0, 0, width - patch_diam)
    # Sample patches from first frame (rgb[0])
    patches = (
        np.stack(
            [
                rgb[
                    0,
                    y0_safe[n] : y0_safe[n] + patch_diam,
                    x0_safe[n] : x0_safe[n] + patch_diam,
                    :,
                ]
                for n in range(num_points)
            ],
            axis=0,
        ).astype(np.float64)
        / 255.0
    )
    patches *= valid_start[:, None, None, None]

    mask_y, mask_x = np.where(circle_mask)

    output_video = np.empty_like(rgb)
    for t in range(num_frames):
        buf = np.zeros((height, width, 4), dtype=np.float64)

        dy = points[:, t, 1] - points[:, 0, 1]
        dx = points[:, t, 0] - points[:, 0, 0]
        vis = ((occluded[:, t] < 0.5) & valid_start).astype(np.float64)

        for mi in range(len(mask_y)):
            py, px = int(mask_y[mi]), int(mask_x[mi])

            out_y = (cy - radius + py).astype(np.float64) + dy
            out_x = (cx - radius + px).astype(np.float64) + dx

            iy = np.floor(out_y).astype(np.int32)
            ix = np.floor(out_x).astype(np.int32)

            in_bounds = (iy >= 0) & (ix >= 0) & (iy + 1 < height) & (ix + 1 < width)
            w = vis * in_bounds.astype(np.float64) * circle_mask[py, px]

            fy = out_y - iy
            fx = out_x - ix

            iy = np.clip(iy, 0, height - 2)
            ix = np.clip(ix, 0, width - 2)

            pval = patches[:, py, px, :]
            rgba = np.concatenate([pval, np.ones((num_points, 1))], axis=1)

            w00 = (w * (1 - fy) * (1 - fx))[:, None]
            w01 = (w * (1 - fy) * fx)[:, None]
            w10 = (w * fy * (1 - fx))[:, None]
            w11 = (w * fy * fx)[:, None]

            np.add.at(buf, (iy, ix), w00 * rgba)
            np.add.at(buf, (iy, ix + 1), w01 * rgba)
            np.add.at(buf, (iy + 1, ix), w10 * rgba)
            np.add.at(buf, (iy + 1, ix + 1), w11 * rgba)

        alpha = buf[..., 3:]
        if bg is not None:
            # dot_rgb is in [0, 1]; bg is uint8 [0, 255] — scale dot to 255 for correct composite
            dot_rgb = buf[..., :3] / np.maximum(alpha, 1e-9)
            blend = np.minimum(alpha, 1.0)
            out_frame = (
                bg[t].astype(np.float64) * (1.0 - blend)
                + (dot_rgb * 255.0) * blend
            ).clip(0, 255).astype(np.uint8)
        else:
            rgb_out = buf[..., :3] / np.maximum(1.0, alpha)
            rgb_out = rgb_out + (1.0 - np.minimum(1.0, alpha))
            out_frame = np.clip(rgb_out * 255.0, 0, 255).astype(np.uint8)
        output_video[t] = out_frame

    return output_video


def plot_tracks_dots(
    rgb: np.ndarray,
    points: np.ndarray,
    occluded: np.ndarray,
    dot_size: float,
    bg: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Wrapper that pads rgb/points so dots can extend slightly out of frame, then splats."""
    patch_diam = int(np.ceil(dot_size))
    if patch_diam % 2 == 0:
        patch_diam += 1
    radius = patch_diam // 2
    margin = radius + 1

    # points shape: (N, T, 2) with last dim (x, y)
    min_y = np.min(points[..., 1]) - margin
    max_y = np.max(points[..., 1]) + margin
    min_x = np.min(points[..., 0]) - margin
    max_x = np.max(points[..., 0]) + margin

    height, width = rgb.shape[1], rgb.shape[2]
    pad_top = max(0, int(np.ceil(-min_y)))
    pad_bottom = max(0, int(np.ceil(max_y - height + 1)))
    pad_left = max(0, int(np.ceil(-min_x)))
    pad_right = max(0, int(np.ceil(max_x - width + 1)))

    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        padded_rgb = np.pad(
            rgb,
            [(0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)],
            mode="constant",
            constant_values=255,
        )
        padded_points = points.copy()
        padded_points[..., 0] += pad_left
        padded_points[..., 1] += pad_top
        if bg is not None:
            padded_bg = np.pad(
                bg,
                [(0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)],
                mode="constant",
                constant_values=255,
            )
        else:
            padded_bg = None
    else:
        padded_rgb = rgb
        padded_points = points
        padded_bg = bg

    out = _plot_tracks_dots_inner(
        padded_rgb, padded_points, occluded, dot_size, bg=padded_bg
    )

    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        out = out[
            :,
            pad_top : out.shape[1] - pad_bottom,
            pad_left : out.shape[2] - pad_right,
            :,
        ]
    return out


# =====================================================================
# Canvas expansion
# =====================================================================

def _compute_adaptive_canvas_size_pixels(
    tracks_pixel: np.ndarray,
    img_h: int,
    img_w: int,
    padding_ratio: float = 0.1,
) -> Tuple[int, int, float, float]:
    """Canvas (canvas_w, canvas_h, offset_x, offset_y) fitting all pixel tracks.

    Pixel-space variant (the flow_utils version takes normalized tracks).
    """
    min_x = tracks_pixel[:, :, 0].min()
    max_x = tracks_pixel[:, :, 0].max()
    min_y = tracks_pixel[:, :, 1].min()
    max_y = tracks_pixel[:, :, 1].max()
    pad_x = (max_x - min_x) * padding_ratio
    pad_y = (max_y - min_y) * padding_ratio
    min_xp = min(min_x - pad_x, 0)
    max_xp = max(max_x + pad_x, img_w)
    min_yp = min(min_y - pad_y, 0)
    max_yp = max(max_y + pad_y, img_h)
    cw = max(int(np.ceil(max_xp - min_xp)), img_w)
    ch = max(int(np.ceil(max_yp - min_yp)), img_h)
    return cw, ch, -min_xp, -min_yp


def expand_canvas_for_tracks(
    first_frame: np.ndarray,
    tracks_pixel: np.ndarray,
    point_mask: np.ndarray,
    visibility: np.ndarray,
    padding_ratio: float = 0.1,
    bg_color: int = 255,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Expand the frame canvas so that all visible track positions fit.

    If every visible track position already lies within [0, W) x [0, H),
    the original frame and tracks are returned unchanged.

    Args:
        first_frame: (H, W, 3) uint8 image.
        tracks_pixel: (T, N, 2) pixel-space track coordinates.
        point_mask: (N,) bool — which points are real.
        visibility: (T, N) float — >0.5 means visible.
        padding_ratio: extra padding as a fraction of the track extent.
        bg_color: fill value for the expanded canvas (0=black, 255=white).

    Returns:
        expanded_frame: (H', W', 3) uint8  (H' >= H, W' >= W).
        shifted_tracks: (T, N, 2) tracks offset so the original frame's
            top-left corner sits at (offset_x, offset_y) in the new canvas.
        offset_x: horizontal offset applied.
        offset_y: vertical offset applied.
    """
    H, W = first_frame.shape[:2]

    active = point_mask[np.newaxis, :] & (visibility > 0.5)  # (T, N)
    if not active.any():
        return first_frame, tracks_pixel, 0.0, 0.0

    cw, ch, ox, oy = _compute_adaptive_canvas_size_pixels(
        tracks_pixel, H, W, padding_ratio=padding_ratio)

    ox, oy = float(ox), float(oy)
    if cw == W and ch == H:
        return first_frame, tracks_pixel, 0.0, 0.0

    canvas = np.full((ch, cw, 3), bg_color, dtype=np.uint8)
    pl, pt = int(round(ox)), int(round(oy))
    canvas[pt:pt + H, pl:pl + W] = first_frame

    shifted = tracks_pixel.copy()
    shifted[:, :, 0] += ox
    shifted[:, :, 1] += oy

    return canvas, shifted, ox, oy


# =====================================================================
# Single-frame renderer
# =====================================================================

def render_track_frame(
    bg: np.ndarray,
    tracks_pixel: np.ndarray,
    visibility: np.ndarray,
    point_mask: np.ndarray,
    colors_int: List[Tuple[int, int, int]],
    current_t: int,
    tail_length: int = 5,
    thickness: int = 2,
    circle_radius: int = 3,
) -> np.ndarray:
    """Render one video frame showing track state at timestep *current_t*.

    Draws:
      - Polyline tail over [current_t - tail_length, current_t] with fading
        alpha (older segments more transparent).
      - Filled circle at the current position.

    Args:
        bg: (H, W, 3) uint8 background image (faded first frame).
        tracks_pixel: (T, N, 2) pixel-space tracks.
        visibility: (T, N) float.
        point_mask: (N,) bool.
        colors_int: list of (R, G, B) uint8 tuples, length N.
        current_t: which track timestep to render.
        tail_length: how many past timesteps the tail covers.
        thickness: polyline thickness.
        circle_radius: endpoint circle radius.

    Returns:
        (H, W, 3) uint8 RGB frame.
    """
    frame = bg.copy()
    T = tracks_pixel.shape[0]
    N = tracks_pixel.shape[1]

    t_start = max(current_t - tail_length, 0)
    tail_len = current_t - t_start

    # Draw all tail segments on a single overlay, using per-segment alpha
    # as colour intensity (draw newest/brightest last so it wins on overlap).
    if tail_len > 0:
        overlay = np.zeros_like(frame)
        mask = np.zeros(frame.shape[:2], dtype=np.float32)
        for seg_t in range(t_start, current_t):
            fade = (seg_t - t_start + 1) / tail_len
            alpha = 0.25 + 0.55 * fade
            for n in range(N):
                if not point_mask[n]:
                    continue
                if seg_t + 1 >= T:
                    continue
                if visibility[seg_t, n] < 0.5 or visibility[seg_t + 1, n] < 0.5:
                    continue
                pt1 = (int(round(tracks_pixel[seg_t, n, 0])),
                       int(round(tracks_pixel[seg_t, n, 1])))
                pt2 = (int(round(tracks_pixel[seg_t + 1, n, 0])),
                       int(round(tracks_pixel[seg_t + 1, n, 1])))
                cv2.line(overlay, pt1, pt2, colors_int[n],
                         thickness=thickness, lineType=cv2.LINE_AA)
                cv2.line(mask, pt1, pt2, alpha,
                         thickness=thickness, lineType=cv2.LINE_AA)
        # Composite: where mask > 0, blend overlay onto frame
        m3 = mask[:, :, np.newaxis]
        np.clip(m3, 0, 1, out=m3)
        frame = (frame * (1 - m3) + overlay * m3).astype(np.uint8)

    # Draw filled circles at current_t directly (fully opaque)
    if 0 <= current_t < T:
        for n in range(N):
            if not point_mask[n]:
                continue
            if visibility[current_t, n] < 0.5:
                continue
            pt = (int(round(tracks_pixel[current_t, n, 0])),
                  int(round(tracks_pixel[current_t, n, 1])))
            _draw_circle_aa(frame, pt, circle_radius, colors_int[n])

    return frame


def _make_faded_bg(first_frame: np.ndarray, alpha: float) -> np.ndarray:
    """Blend *first_frame* toward white at the given alpha."""
    white = np.full_like(first_frame, 255)
    blended = cv2.addWeighted(first_frame, alpha, white, 1.0 - alpha, 0)
    return blended


def _draw_circle_aa(
    frame: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    color: Tuple[int, int, int],
) -> None:
    """Draw a filled circle with anti-aliasing. For radius <= 2, uses upscale-then-downscale
    so the dot appears round instead of diamond-shaped (OpenCV's circle at r=2 has few pixels)."""
    if radius >= 3:
        cv2.circle(frame, center, radius, color, thickness=-1, lineType=cv2.LINE_AA)
        return
    # Small radius: draw at higher resolution then downscale for a smooth round shape
    H, W = frame.shape[:2]
    x0 = center[0] - radius
    y0 = center[1] - radius
    x1 = center[0] + radius + 1
    y1 = center[1] + radius + 1
    sx0 = max(0, -x0)
    sy0 = max(0, -y0)
    x0_clip = max(0, x0)
    y0_clip = max(0, y0)
    x1_clip = min(W, x1)
    y1_clip = min(H, y1)
    roi_h = y1_clip - y0_clip
    roi_w = x1_clip - x0_clip
    if roi_w <= 0 or roi_h <= 0:
        return
    bg = frame[y0_clip:y1_clip, x0_clip:x1_clip].copy()
    scale = 4
    patch_size = (2 * radius + 1) * scale
    patch = cv2.resize(bg, (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)
    cx, cy = patch_size // 2, patch_size // 2
    cv2.circle(patch, (cx, cy), radius * scale, color, thickness=-1, lineType=cv2.LINE_AA)
    small = cv2.resize(patch, (2 * radius + 1, 2 * radius + 1), interpolation=cv2.INTER_AREA)
    src = small[sy0 : sy0 + roi_h, sx0 : sx0 + roi_w]
    frame[y0_clip:y1_clip, x0_clip:x1_clip] = src


# =====================================================================
# Video writers
# =====================================================================

def _save_gif_fast(frames: List[np.ndarray], path: str, fps: int = 30):
    """Write frames to GIF using Pillow with a shared palette (much faster than imageio)."""
    duration_ms = int(1000 / fps)
    pil_frames = [Image.fromarray(f).quantize(colors=256, method=Image.Quantize.FASTOCTREE)
                  for f in frames]
    pil_frames[0].save(
        path, save_all=True, append_images=pil_frames[1:],
        duration=duration_ms, loop=0, optimize=False,
    )


def _save_frames_mp4(frames: List[np.ndarray], path: str, fps: int = 30):
    """Write (T, H, W, 3) uint8 RGB frames to MP4. Ensures even dimensions for codec."""
    if not frames:
        return
    arr = np.stack(frames, axis=0)
    T, H, W = arr.shape[:3]
    out_h = H if H % 2 == 0 else H + 1
    out_w = W if W % 2 == 0 else W + 1
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, fps, (out_w, out_h))
    for f in arr:
        if f.shape[0] != out_h or f.shape[1] != out_w:
            padded = np.full((out_h, out_w, 3), 0, dtype=np.uint8)
            padded[:f.shape[0], :f.shape[1]] = f
            f = padded
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()


# =====================================================================
# Full video generation (4-phase timeline)
# =====================================================================

def generate_track_video(
    first_frame: np.ndarray,
    tracks_pixel: np.ndarray,
    visibility: np.ndarray,
    point_mask: np.ndarray,
    grey_colors: np.ndarray,
    rainbow_colors: Optional[np.ndarray],
    num_point_cond: int,
    is_gt: bool,
    output_path_gif: Optional[str],
    output_path_mp4: Optional[str] = None,
    video_fps: int = 30,
    track_fps: int = 15,
    tail_length: int = 5,
    thickness: int = 2,
    circle_radius: int = 3,
    use_texture_dots: bool = False,
    dot_size: float = 10.0,
    cond_bbox_outline: bool = True,
    padding_ratio: float = 0.1,
    white_bg: bool = False,
    first_frame_image_rect_xywh: Optional[Tuple[int, int, int, int]] = None,
    show_cond_label: bool = False,
    expand_canvas: bool = True,
):
    """Render the 4-phase video and write to GIF (and optionally MP4).

    Phases:
      1. 0.5s  — static first frame.
      2. 0.75s — fade frame to 0.3 alpha; texture-colored dots fade in.
      3. 0.5s  — dots remain texture-colored; static pause.
      4. rest  — play track timesteps with tails (same texture-derived colors).

    Args:
        first_frame: (H, W, 3) uint8 RGB.
        tracks_pixel: (T, N, 2).
        visibility: (T, N).
        point_mask: (N,) bool.
        grey_colors: (N, 3) float [0,1] — point colors (e.g. texture-sampled).
        rainbow_colors: (N, 3) float [0,1] — same as grey when using texture;
                        ignored when is_gt=True.
        num_point_cond: number of conditioning timesteps.
        is_gt: if True, all timesteps use grey_colors (GT video).
        output_path_gif: path for .gif output (None to skip).
        output_path_mp4: path for .mp4 output (None to skip).
        video_fps: output video frame rate.
        track_fps: how many track timesteps per second.
        tail_length: trailing timesteps for tails.
        thickness: polyline thickness.
        circle_radius: endpoint circle radius (color mode only).
        use_texture_dots: if True, use texture splatting (bigger dots) instead of small colored circles.
        dot_size: diameter in pixels for texture dots (used when use_texture_dots=True).
        cond_bbox_outline: if True and not is_gt, use animal-colored intro/pause and draw light grey outline around first-frame image during conditioning frames only.
        padding_ratio: padding as fraction of track extent for adaptive canvas (default 0.1).
        first_frame_image_rect_xywh: when set, (x, y, w, h) of the first-frame image in input first_frame coords
            (so the outline is drawn only around that rect after expand). When None, outline uses full input frame.
        expand_canvas: if False, do not expand canvas; use original frame size only (no padding for tracks outside).
    """
    H_orig, W_orig = first_frame.shape[:2]
    if expand_canvas:
        first_frame, tracks_pixel, ox, oy = expand_canvas_for_tracks(
            first_frame, tracks_pixel, point_mask, visibility,
            padding_ratio=padding_ratio)
    else:
        ox, oy = 0.0, 0.0

    H, W = first_frame.shape[:2]
    T = tracks_pixel.shape[0]
    N = tracks_pixel.shape[1]

    frames_per_track_step = max(video_fps // track_fps, 1)
    phase1_frames = int(round(video_fps * 0.5))
    phase2_frames = int(round(video_fps * 0.75))
    phase3_frames = int(round(video_fps * 0.5))

    grey_int = [_color_to_rgb_uint8(grey_colors[n]) for n in range(N)]
    if rainbow_colors is not None:
        rainbow_int = [_color_to_rgb_uint8(rainbow_colors[n]) for n in range(N)]
    else:
        rainbow_int = grey_int

    # For GT videos, intro dots are always grey. With cond_bbox_outline (pred only), use animal color for intro/pause.
    if cond_bbox_outline and not is_gt:
        intro_dot_colors = rainbow_int
        phase3_dot_colors = rainbow_int
    else:
        intro_dot_colors = grey_int if is_gt else rainbow_int
        phase3_dot_colors = grey_int

    # Outline around the first-frame image region (for cond_bbox_outline); thickness 3 for visibility
    bbox_outline_thickness = 3
    # First-frame image rect: when caller passes first_frame_image_rect_xywh (x,y,w,h in input frame coords),
    # use it shifted by (ox,oy) after expand; else use the placed input frame (ox, oy, W_orig, H_orig)
    if cond_bbox_outline and not is_gt:
        if first_frame_image_rect_xywh is not None:
            rx, ry, rw, rh = first_frame_image_rect_xywh
            bbox_xywh = (int(round(rx + ox)), int(round(ry + oy)), rw, rh)
        else:
            bbox_xywh = (int(round(ox)), int(round(oy)), W_orig, H_orig)
    else:
        bbox_xywh = None
    bbox_outline_color = (140, 140, 140)  # light grey (RGB)

    if white_bg:
        faded_bg = np.full((H, W, 3), 255, dtype=np.uint8)
    else:
        faded_bg = _make_faded_bg(first_frame, 0.3)

    def _render_dots(bg_img, dot_colors, dot_alpha=1.0):
        """Draw t=0 dots in *dot_colors* onto bg_img with given opacity."""
        overlay = bg_img.copy()
        for n in range(N):
            if not point_mask[n]:
                continue
            if visibility[0, n] > 0.5:
                pt = (int(round(tracks_pixel[0, n, 0])),
                      int(round(tracks_pixel[0, n, 1])))
                _draw_circle_aa(overlay, pt, circle_radius, dot_colors[n])
        if dot_alpha < 1.0:
            return cv2.addWeighted(overlay, dot_alpha, bg_img,
                                   1.0 - dot_alpha, 0)
        return overlay

    def _render_texture_dots_one_frame(bg_img, dot_alpha=1.0):
        """Splat texture dots at t=0 only onto bg_img (one-frame version)."""
        rgb_one = np.expand_dims(first_frame, 0)
        points_one = np.expand_dims(tracks_pixel[0], axis=1)  # (N, 1, 2)
        occluded_one = (1.0 - visibility[0:1, :]).T  # (N, 1)
        occluded_one[~point_mask, 0] = 1.0  # don't draw masked-out points
        bg_one = np.expand_dims(bg_img, 0)
        out = plot_tracks_dots(rgb_one, points_one, occluded_one, dot_size, bg=bg_one)
        frame = out[0]
        if dot_alpha < 1.0:
            return cv2.addWeighted(frame, dot_alpha, bg_img, 1.0 - dot_alpha, 0)
        return frame

    all_frames: List[np.ndarray] = []

    # --- Phase 1: static first frame (0.5 s) ---
    # Same image repeated — share reference (never mutated after append).
    ff = first_frame.copy()
    for _ in range(phase1_frames):
        all_frames.append(ff)

    # --- Phase 2: fade frame + dots fade in (0.75 s) ---
    if white_bg:
        for i in range(phase2_frames):
            progress = (i + 1) / phase2_frames
            if use_texture_dots:
                frame = _render_texture_dots_one_frame(faded_bg, dot_alpha=progress)
            else:
                frame = _render_dots(faded_bg, intro_dot_colors, dot_alpha=progress)
            all_frames.append(frame)
    else:
        first_f32 = first_frame.astype(np.float32)
        faded_f32 = faded_bg.astype(np.float32)
        for i in range(phase2_frames):
            progress = (i + 1) / phase2_frames      # 0→1
            bg = np.clip(first_f32 * (1.0 - progress) + faded_f32 * progress,
                         0, 255).astype(np.uint8)
            if use_texture_dots:
                frame = _render_texture_dots_one_frame(bg, dot_alpha=progress)
            else:
                frame = _render_dots(bg, intro_dot_colors, dot_alpha=progress)
            all_frames.append(frame)

    # --- Phase 3: dots pause (0.5 s) ---
    if use_texture_dots:
        phase3_frame = _render_texture_dots_one_frame(faded_bg, dot_alpha=1.0)
    else:
        phase3_frame = _render_dots(faded_bg, phase3_dot_colors, dot_alpha=1.0)
    for _ in range(phase3_frames):
        all_frames.append(phase3_frame)

    def _draw_cond_bbox_if_needed(frame: np.ndarray, t: int) -> None:
        """Draw light grey outline around the first-frame image when in conditioning range (in-place)."""
        if bbox_xywh is not None and t < num_point_cond:
            x, y, w, h = bbox_xywh
            cv2.rectangle(
                frame, (x, y), (x + w, y + h), bbox_outline_color,
                thickness=bbox_outline_thickness, lineType=cv2.LINE_AA,
            )

    def _draw_cond_label(frame: np.ndarray, t: int) -> None:
        """Draw 'CONDITIONING' in black at top-left, under the border, for conditioning frames only (when show_cond_label)."""
        if not show_cond_label or t >= num_point_cond:
            return
        if bbox_xywh is not None:
            x, y, _, _ = bbox_xywh
            tx, ty = x + bbox_outline_thickness + 4, y + bbox_outline_thickness + 20
        else:
            tx, ty = 8, 24
        cv2.putText(
            frame, "CONDITIONING", (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA,
        )

    # --- Phase 4: play track timesteps ---
    if use_texture_dots:
        rgb_stack = np.stack([first_frame] * T)
        points_n_t = np.transpose(tracks_pixel, (1, 0, 2))
        occluded_n_t = (1.0 - visibility).T
        occluded_n_t[~point_mask, :] = 1.0  # don't draw masked-out points
        bg_stack = np.stack([faded_bg] * T)
        phase4_frames = plot_tracks_dots(
            rgb_stack, points_n_t, occluded_n_t, dot_size, bg=bg_stack
        )
        for t in range(T):
            frame = phase4_frames[t].copy()
            _draw_cond_bbox_if_needed(frame, t)
            _draw_cond_label(frame, t)
            for _ in range(frames_per_track_step):
                all_frames.append(frame)
    else:
        for t in range(T):
            if is_gt:
                colors_for_t = grey_int
            elif t < num_point_cond and not cond_bbox_outline:
                colors_for_t = grey_int
            else:
                colors_for_t = rainbow_int

            frame = render_track_frame(
                faded_bg, tracks_pixel, visibility, point_mask,
                colors_for_t, t,
                tail_length=tail_length,
                thickness=thickness,
                circle_radius=circle_radius,
            )
            _draw_cond_bbox_if_needed(frame, t)
            _draw_cond_label(frame, t)
            for _ in range(frames_per_track_step):
                all_frames.append(frame)

    # --- Write GIF ---
    if output_path_gif:
        _save_gif_fast(all_frames, output_path_gif, fps=video_fps)

    # --- Write MP4 (optional) ---
    if output_path_mp4:
        out_h = H if H % 2 == 0 else H + 1
        out_w = W if W % 2 == 0 else W + 1

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path_mp4, fourcc, video_fps,
                                 (out_w, out_h))
        for f in all_frames:
            if f.shape[0] != out_h or f.shape[1] != out_w:
                padded = np.full((out_h, out_w, 3), 255, dtype=np.uint8)
                padded[:f.shape[0], :f.shape[1]] = f
                f = padded
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()


# =====================================================================
# Batch-format driver (used by engine/eval.py visualization)
# =====================================================================

def generate_animal_color_videos_for_sample(
    batch,
    gt_tracks,
    pred_tracks,
    output_dir: str,
    sample_name: str,
    num_point_cond: int = 4,
    texture_radius: int = 3,
    video_fps: int = 30,
    track_fps: int = 15,
    tail_length: int = 5,
    thickness: int = 2,
    circle_radius: int = 5,
    save_gif: bool = False,
):
    """Render GT + prediction animal-color videos for one collated sample.

    Tracks are in [0,1] space normalized to the bbox crop; conversion to pixel
    space uses the batch's bbox/orig-dim metadata (requires the dataset to be
    built with vis=True).

    Args:
        batch: collated batch of size 1 with 'video' (B, T, C, H, W) in
            [0, 255], plus 'bbox', 'frame_start', 'orig_width', 'orig_height',
            'point_mask', 'visibility' metadata.
        gt_tracks: (B, T, N, 2) torch tensor, [0,1] normalized.
        pred_tracks: (B, T, N, 2) torch tensor, [0,1] normalized.
        output_dir: directory for output files.
        sample_name: basename for outputs (<name>_gt.mp4, <name>_pred.mp4).
        num_point_cond: number of conditioning timesteps (for outline/colors).
        texture_radius: radius of the texture color sampling circle.
        save_gif: also write GIFs next to the MP4s.

    Returns:
        list of written file paths.
    """
    from motion_forecasting.model.track_visualization import _convert_tracks_to_pixel

    os.makedirs(output_dir, exist_ok=True)

    video = batch.get("video_30fps_vis", batch["video"])
    frame0 = video[0, 0].permute(1, 2, 0).cpu().numpy()
    if frame0.max() <= 1.0:
        frame0 = (frame0 * 255.0)
    first_frame = np.clip(frame0, 0, 255).astype(np.uint8)
    H_img, W_img = first_frame.shape[:2]

    gt_np = gt_tracks[0].cpu().numpy()
    pred_np = pred_tracks[0].cpu().numpy()
    T, N = gt_np.shape[:2]

    pm = batch.get("point_mask", None)
    point_mask = (pm[0].cpu().numpy().astype(bool) if pm is not None
                  else np.ones(N, dtype=bool))
    vis = batch.get("visibility", None)
    visibility = (vis[0].cpu().numpy().astype(np.float32) if vis is not None
                  else np.ones((T, N), dtype=np.float32))

    bbox_info = batch.get("bbox", None)
    frame_start_info = batch.get("frame_start", None)
    orig_width_info = batch.get("orig_width", None)
    orig_height_info = batch.get("orig_height", None)

    # CRITICAL: bbox-aware conversion — tracks are normalized to the bbox
    # crop, never `tracks * img_size`.
    gt_pix = _convert_tracks_to_pixel(
        gt_np, bbox_info, frame_start_info, orig_width_info, orig_height_info,
        W_img, H_img, 0.0, 0.0)
    pred_pix = _convert_tracks_to_pixel(
        pred_np, bbox_info, frame_start_info, orig_width_info, orig_height_info,
        W_img, H_img, 0.0, 0.0)

    # One texture color per point, sampled at the GT t=0 position; shared
    # between the GT and prediction videos so points are identifiable.
    texture_colors = sample_texture_colors(
        first_frame, gt_pix[0], point_mask, texture_radius)

    written = []
    for tracks_pix, is_gt, tag in ((gt_pix, True, "gt"), (pred_pix, False, "pred")):
        mp4_path = os.path.join(output_dir, f"{sample_name}_{tag}.mp4")
        gif_path = (os.path.join(output_dir, f"{sample_name}_{tag}.gif")
                    if save_gif else None)
        generate_track_video(
            first_frame, tracks_pix, visibility, point_mask,
            texture_colors, texture_colors, num_point_cond,
            is_gt=is_gt,
            output_path_gif=gif_path,
            output_path_mp4=mp4_path,
            video_fps=video_fps,
            track_fps=track_fps,
            tail_length=tail_length,
            thickness=thickness,
            circle_radius=circle_radius,
            cond_bbox_outline=not is_gt,
        )
        written.append(mp4_path)
        if gif_path:
            written.append(gif_path)
    return written
