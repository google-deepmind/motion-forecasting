"""
Coordinate conversion utilities for diffusion track prediction.

Provides standalone functions for converting between:
- Track format (B, T, N, 2) in [0,1] space (dataloader format)
- Model format (B, C, N) in [-1,1] space (DiT internal format)
- Velocity representations (multiple modes)

These functions take explicit configuration arguments instead of reading from self.*,
making them testable and reusable outside the wrapper class.
"""

import math
import torch
import torch.nn as nn
import numpy as np


# Maximum number of conditioning timesteps supported by motion history
MAX_MOTION_HISTORY_COND = 4


class MotionHistorySinCosEmbedding(nn.Module):
    """Non-learned sin-cos embedding that maps a raw 2D velocity to a fixed-dim vector.

    For each velocity component (x, y) we produce ``half_per_component`` cosine values
    followed by ``half_per_component`` sine values, then concatenate x and y parts.
    Total output dim = ``embed_dim_per_vel`` (default 340 = 2 * 2 * 85).

    Frequencies are geometrically spaced between wavelengths ``min_period`` (finest
    detail, default 1/1024) and ``max_period`` (full input range, default 0.5).

    The ``scale`` parameter is applied after the sin/cos computation and should
    already incorporate the sqrt(2) unit-variance correction (default 0.135 * sqrt(2)).
    """

    def __init__(self, embed_dim_per_vel: int = 340,
                 min_period: float = 1.0 / 1024,
                 max_period: float = 0.5,
                 scale: float = 0.19092):
        super().__init__()
        assert embed_dim_per_vel % 4 == 0, "embed_dim_per_vel must be divisible by 4"
        half_per_component = embed_dim_per_vel // 4  # 85 for default 340
        wavelengths = torch.exp(torch.linspace(
            math.log(min_period), math.log(max_period), half_per_component
        ))
        freqs = 2.0 * math.pi / wavelengths
        self.register_buffer('freqs', freqs)
        self.scale = scale
        self.embed_dim_per_vel = embed_dim_per_vel

    def forward(self, vel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vel: (B, N, 2) raw velocity in [-1,1] coord space (before velocity_scale).
        Returns:
            (B, N, embed_dim_per_vel) scaled sin-cos embedding.
        """
        vx = vel[..., 0]  # (B, N)
        vy = vel[..., 1]  # (B, N)
        args_x = vx.unsqueeze(-1) * self.freqs  # (B, N, half_per_component)
        args_y = vy.unsqueeze(-1) * self.freqs
        emb = torch.cat([
            torch.cos(args_x), torch.sin(args_x),
            torch.cos(args_y), torch.sin(args_y),
        ], dim=-1)  # (B, N, embed_dim_per_vel)
        return emb * self.scale


def get_channel_layout(horizon, num_point_cond, diffuse_on_velocity,
                       handle_occlusions, use_dino_features, dino_feature_dim,
                       motion_history_dim=340, motion_history_conditioning="embedded"):
    """
    Get the channel layout for one of the two supported conditioning schemes.

    "embedded" (current scheme):
      [XVel, YVel, Occ, C_occ0:3, C1, C2, C3, DINO]
      The first ``supervised_channels`` channels (XVel + YVel + Occ) are always
      noised during training. Everything from ``supervised_channels`` onward is
      never noised and acts as conditioning. Always allocates the full
      MAX_MOTION_HISTORY_COND=4 slots for C_occ and 3 sin-cos velocity
      embedding slots regardless of ``num_point_cond``; unused slots are zeroed.

    "channel" (used by the released checkpoints):
      [XVel, YVel, Occ, DINO]
      No dedicated conditioning blocks. Motion-history conditioning works by
      fixing the leading channels to ground truth during training/sampling:
      the first ``num_cond_per_axis`` velocity channels per axis and the first
      ``num_point_cond`` occlusion channels. DINO channels are never noised.
      The loss supervises all coord+occ channels (including the conditioned
      leading ones).
    """
    assert motion_history_conditioning in ("embedded", "channel"), \
        f"motion_history_conditioning must be 'embedded' or 'channel', got {motion_history_conditioning!r}"
    assert 1 <= num_point_cond <= MAX_MOTION_HISTORY_COND, \
        f"num_point_cond must be in [1, {MAX_MOTION_HISTORY_COND}], got {num_point_cond}"

    if diffuse_on_velocity:
        num_coord_channels = horizon - 1
        num_cond_per_axis = max(0, num_point_cond - 1)
    else:
        num_coord_channels = horizon
        num_cond_per_axis = num_point_cond

    # --- Supervised (noised) channels ---
    x_start = 0
    x_end = num_coord_channels
    y_start = num_coord_channels
    y_end = 2 * num_coord_channels

    occ_start = occ_end = None
    if handle_occlusions:
        occ_start = y_end
        occ_end = occ_start + horizon

    supervised_end = occ_end if handle_occlusions else y_end
    supervised_channels = supervised_end

    if motion_history_conditioning == "embedded":
        # --- Dedicated conditioning (never-noised) channels ---
        num_c_occ = MAX_MOTION_HISTORY_COND          # always 4 slots
        num_c_vel = MAX_MOTION_HISTORY_COND - 1      # always 3 slots

        c_occ_start = supervised_end
        c_occ_end = c_occ_start + num_c_occ

        c_vel_starts = []
        cursor = c_occ_end
        for _ in range(num_c_vel):
            c_vel_starts.append(cursor)
            cursor += motion_history_dim
    else:
        # Channel layout: no dedicated conditioning channels.
        num_c_occ = 0
        num_c_vel = 0
        c_occ_start = c_occ_end = None
        c_vel_starts = []
        cursor = supervised_end

    dino_start = dino_end = None
    if use_dino_features and dino_feature_dim > 0:
        dino_start = cursor
        dino_end = dino_start + dino_feature_dim
        cursor = dino_end

    total_channels = cursor

    return {
        'motion_history_conditioning': motion_history_conditioning,
        'diffuse_on_velocity': diffuse_on_velocity,
        'num_coord_channels_per_axis': num_coord_channels,
        'x_start': x_start,
        'x_end': x_end,
        'y_start': y_start,
        'y_end': y_end,
        'occ_start': occ_start,
        'occ_end': occ_end,
        'supervised_channels': supervised_channels,
        # Channel-conditioning ranges (leading channels fixed to GT)
        'num_cond_per_axis': num_cond_per_axis,
        'num_cond_occ': num_point_cond if handle_occlusions else 0,
        # Embedded motion history conditioning ranges
        'c_occ_start': c_occ_start,
        'c_occ_end': c_occ_end,
        'num_c_occ': num_c_occ,
        'c_vel_starts': c_vel_starts,           # list of 3 start indices ("embedded" only)
        'num_c_vel': num_c_vel,
        'motion_history_dim': motion_history_dim,
        # DINO
        'dino_start': dino_start,
        'dino_end': dino_end,
        'total_channels': total_channels,
    }


def build_fixed_channels_mask(layout, num_point_cond=None, device=None):
    """Boolean (C,) mask of channels that are held fixed (never noised / re-imposed
    from GT at every denoising step).

    For "embedded": everything from supervised_channels onward.
    For "channel": the leading conditioned velocity channels per axis, the
    leading conditioned occlusion channels, and the DINO tail.

    Args:
        layout: dict from get_channel_layout().
        num_point_cond: history level K. Defaults to the layout's value.
            For "channel" mode, K determines how many leading channels are fixed
            (K-1 velocity channels per axis in velocity mode, K occ channels).

    Returns:
        torch.BoolTensor of shape (total_channels,)
    """
    C = layout['total_channels']
    mask = torch.zeros(C, dtype=torch.bool, device=device)

    if layout['motion_history_conditioning'] == "embedded":
        mask[layout['supervised_channels']:] = True
        return mask

    # "channel" mode
    if num_point_cond is None:
        n_vel = layout['num_cond_per_axis']
        n_occ = layout['num_cond_occ']
    else:
        # Mirror the layout's velocity-mode adjustment: in velocity diffusion,
        # K history positions correspond to K-1 leading velocity channels.
        n_vel = max(0, num_point_cond - 1) if layout['diffuse_on_velocity'] else num_point_cond
        n_occ = num_point_cond if layout['occ_start'] is not None else 0

    y_start = layout['y_start']
    mask[0:n_vel] = True
    mask[y_start:y_start + n_vel] = True
    if layout['occ_start'] is not None and n_occ > 0:
        mask[layout['occ_start']:layout['occ_start'] + n_occ] = True
    if layout['dino_start'] is not None:
        mask[layout['dino_start']:layout['dino_end']] = True
    return mask


def coords_to_velocity(tracks, num_point_cond, velocity_scale, debug_mode=False):
    """
    Convert coordinate tracks to pure velocity representation.

    Produces [v_0, v_1, ..., v_{T-2}] = T-1 tokens per axis.
    Initial position is returned separately.

    Args:
        tracks: (B, T, N, 2) normalized coordinates in [0, 1] or [-1, 1] space
        num_point_cond: number of conditioned points
        velocity_scale: scale factor for velocities

    Returns:
        all_velocities_scaled: (B, T-1, N, 2) scaled velocities
        initial_pos: (B, 1, N, 2) initial position (stored externally)
    """
    B, T, N, _ = tracks.shape
    initial_pos = tracks[:, :1, :, :].clone()
    all_velocities = tracks[:, 1:, :, :] - tracks[:, :-1, :, :]
    all_velocities_scaled = all_velocities * velocity_scale
    return all_velocities_scaled, initial_pos


def velocity_to_coords(combined, num_point_cond, velocity_scale,
                        initial_pos_override=None, debug_mode=False):
    """
    Convert pure velocity representation back to coordinates.

    Args:
        combined: (B, T-1, N, 2) scaled velocities
        num_point_cond: number of conditioned points
        velocity_scale: scale factor for velocities
        initial_pos_override: (B, 1, N, 2) initial position (required)

    Returns:
        tracks: (B, T, N, 2) reconstructed coordinates
    """
    velocities_unscaled = combined / velocity_scale
    if initial_pos_override is None:
        raise ValueError("Initial position required but none was provided")
    cumsum_vel = torch.cumsum(velocities_unscaled, dim=1)
    subsequent_positions = initial_pos_override + cumsum_vel
    return torch.cat([initial_pos_override, subsequent_positions], dim=1)


def compute_velocity_displacement_from_tracks(tracks, visibility=None, point_mask=None):
    """
    Compute velocity and displacement from tracks (for visualization/comparison).
    This mirrors the logic in the dataloader but works with torch tensors.
    
    Args:
        tracks: (B, T, N, 2) track coordinates
        visibility: (B, T, N) visibility mask or None
        point_mask: (B, N) point mask (True for real points) or None
        
    Returns:
        avg_velocity: (B, 2) average velocity [vx, vy]
        total_displacement: (B, 2) total displacement [dx, dy]
    """
    B, T, N, _ = tracks.shape
    device = tracks.device
    
    tracks_np = tracks.detach().cpu().numpy()
    if visibility is not None:
        visibility_np = visibility.detach().cpu().numpy()
    else:
        visibility_np = np.ones((B, T, N))
    if point_mask is not None:
        point_mask_np = point_mask.detach().cpu().numpy()
    else:
        point_mask_np = np.ones((B, N), dtype=bool)
    
    batch_velocities = []
    batch_displacements = []
    
    for b in range(B):
        valid_point_indices = np.where(point_mask_np[b])[0]
        
        if len(valid_point_indices) == 0:
            batch_velocities.append(np.zeros(2, dtype=np.float32))
            batch_displacements.append(np.zeros(2, dtype=np.float32))
            continue
        
        velocities = []
        displacements = []
        
        for point_idx in valid_point_indices:
            visible_frames = np.where(visibility_np[b, :, point_idx] > 0)[0]
            
            if len(visible_frames) < 2:
                continue
            
            point_track = tracks_np[b, visible_frames, point_idx]
            frame_velocities = np.diff(point_track, axis=0)
            point_avg_velocity = np.mean(frame_velocities, axis=0)
            velocities.append(point_avg_velocity)
            
            point_displacement = point_track[-1] - point_track[0]
            displacements.append(point_displacement)
        
        if len(velocities) > 0:
            avg_velocity = np.mean(velocities, axis=0).astype(np.float32)
            avg_displacement = np.mean(displacements, axis=0).astype(np.float32)
        else:
            avg_velocity = np.zeros(2, dtype=np.float32)
            avg_displacement = np.zeros(2, dtype=np.float32)
        
        batch_velocities.append(avg_velocity)
        batch_displacements.append(avg_displacement)
    
    avg_velocity = torch.from_numpy(np.stack(batch_velocities)).to(device)
    total_displacement = torch.from_numpy(np.stack(batch_displacements)).to(device)
    
    return avg_velocity, total_displacement


def tracks_to_model_format(track, vid, horizon, diffuse_on_velocity,
                               num_point_cond, velocity_scale, handle_occlusions, occlusion_scaling,
                               use_dino_features, dino_feature_dim, dino_extractor, dino_scale_factor,
                               img_size, visibility=None, skip_dino_sync=False,
                               motion_history_embedder=None, motion_history_dim=340,
                               motion_history_conditioning="embedded",
                               debug_mode=False):
    """
    Convert track format [0,1] to model format [-1,1].

    "embedded" layout: [XVel, YVel, Occ, C_occ0:3, C1, C2, C3, DINO]
    "channel" layout (released checkpoints): [XVel, YVel, Occ, DINO]

    Args:
        track: (B, T_track, N, 2) track coordinates in [0, 1] space
        vid: (B, T, C, H, W) video frames
        motion_history_embedder: MotionHistorySinCosEmbedding instance
        motion_history_dim: int, dimension per velocity embedding

    Returns:
        x: (B, C, N) DiT input tensor
        y: (B, C, H, W) conditioning image (first frame)
        initial_pos_for_encoding: (B, N, 2) initial position for sinusoidal encoding
        initial_pos_mode2: (B, 1, N, 2) or None, initial position for velocity mode 2
        dino_scale_factor: float or None, updated DINO scale factor
    """
    B, T_track, N, _ = track.shape
    B_vid, T_vid, C, H, W = vid.shape

    y = vid[:, 0]  # (B, C, H, W)

    track_diffusion = track * 2.0 - 1.0

    initial_pos_for_encoding = track_diffusion[:, 0, :, :].clone()  # (B, N, 2)

    # --- Raw velocities (before scaling) for motion history conditioning ---
    all_raw_velocities = track_diffusion[:, 1:, :, :] - track_diffusion[:, :-1, :, :]  # (B, T-1, N, 2)

    # --- Velocity mode conversion (scaled velocities for diffused channels) ---
    initial_pos_mode2 = None
    layout = get_channel_layout(horizon, num_point_cond, diffuse_on_velocity,
                                 handle_occlusions,
                                 use_dino_features, dino_feature_dim,
                                 motion_history_dim=motion_history_dim,
                                 motion_history_conditioning=motion_history_conditioning)
    if diffuse_on_velocity:
        combined, initial_pos_mode2 = coords_to_velocity(
            track_diffusion, num_point_cond, velocity_scale
        )
        num_channels_per_axis = layout['num_coord_channels_per_axis']

        x = combined.permute(0, 3, 1, 2).contiguous()  # (B, 2, num_channels, N)
        x = x.view(B, 2 * num_channels_per_axis, N)
    else:
        x = track_diffusion.permute(0, 3, 1, 2).contiguous()  # (B, 2, T_track, N)
        x = x.view(B, 2 * T_track, N)

    # Pad or trim coordinate channels
    expected_coord_channels = 2 * layout['num_coord_channels_per_axis']
    if x.shape[1] != expected_coord_channels:
        if x.shape[1] < expected_coord_channels:
            padding = torch.zeros(B, expected_coord_channels - x.shape[1], N, device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=1)
        else:
            x = x[:, :expected_coord_channels]

    # --- Concatenate full-horizon occlusion channels (always noised) ---
    if handle_occlusions and visibility is not None:
        occlusion_float = visibility * 2.0 - 1.0
        occlusion_float = occlusion_float * occlusion_scaling

        if T_track != horizon:
            if T_track < horizon:
                occluded_value = -occlusion_scaling
                occlusion_padding = torch.full((B, horizon - T_track, N), occluded_value,
                                               device=occlusion_float.device, dtype=occlusion_float.dtype)
                occlusion_float = torch.cat([occlusion_float, occlusion_padding], dim=1)
            else:
                occlusion_float = occlusion_float[:, :horizon]

        x = torch.cat([x, occlusion_float], dim=1)

    # At this point x has shape (B, supervised_channels, N)

    if motion_history_conditioning == "embedded":
        # --- Motion history conditioning: C_occ0:3 ---
        # Always allocate MAX_MOTION_HISTORY_COND=4 occ slots
        if handle_occlusions and visibility is not None:
            vis_scaled = (visibility * 2.0 - 1.0) * occlusion_scaling  # (B, T_track, N)
            c_occ = torch.zeros(B, MAX_MOTION_HISTORY_COND, N, device=x.device, dtype=x.dtype)
            n_avail = min(T_track, MAX_MOTION_HISTORY_COND)
            c_occ[:, :n_avail, :] = vis_scaled[:, :n_avail, :]
        else:
            c_occ = torch.zeros(B, MAX_MOTION_HISTORY_COND, N, device=x.device, dtype=x.dtype)
        x = torch.cat([x, c_occ], dim=1)

        # --- Motion history conditioning: C1, C2, C3 (sin-cos embeddings) ---
        num_c_vel = MAX_MOTION_HISTORY_COND - 1  # 3
        if motion_history_embedder is not None:
            for ci in range(num_c_vel):
                if ci < all_raw_velocities.shape[1]:
                    vel_ci = all_raw_velocities[:, ci, :, :]  # (B, N, 2)
                    emb_ci = motion_history_embedder(vel_ci)   # (B, N, motion_history_dim)
                    emb_ci = emb_ci.permute(0, 2, 1)           # (B, motion_history_dim, N)
                else:
                    emb_ci = torch.zeros(B, motion_history_dim, N, device=x.device, dtype=x.dtype)
                x = torch.cat([x, emb_ci], dim=1)
        else:
            zeros = torch.zeros(B, num_c_vel * motion_history_dim, N, device=x.device, dtype=x.dtype)
            x = torch.cat([x, zeros], dim=1)
    # "channel" mode: no dedicated conditioning blocks; history conditioning
    # happens by fixing the leading channels (see pipeline.train_step/predict).

    # --- DINO features ---
    if use_dino_features and dino_extractor is not None:
        first_timestep_tracks_dit = track_diffusion[:, 0, :, :]
        first_frames = vid[:, 0]

        dino_per_point_batch = dino_extractor.extract_point_features_batch(
            first_frames, first_timestep_tracks_dit,
            img_size=img_size, coord_space="-1_1"
        )  # (B, N, dino_feature_dim)

        dino_per_point_batch = dino_per_point_batch.permute(0, 2, 1)  # (B, dino_feature_dim, N)

        if dino_scale_factor is None:
            # Use only the supervised portion to compute variance
            coord_var_per_dim = x[:, :layout['supervised_channels'], :].var().item()
            dino_var_per_dim = dino_per_point_batch.var().item()

            num_coord_channels_for_var = 2 * horizon
            coord_total_var = coord_var_per_dim * num_coord_channels_for_var
            dino_total_var = dino_var_per_dim * dino_feature_dim

            scale_factor_local = (coord_total_var / dino_total_var) ** 0.5 if dino_total_var > 0 else 1.0

            if not skip_dino_sync and torch.distributed.is_available() and torch.distributed.is_initialized():
                scale_factor_tensor = torch.tensor(scale_factor_local, device=x.device)
                torch.distributed.all_reduce(scale_factor_tensor, op=torch.distributed.ReduceOp.AVG)
                dino_scale_factor = scale_factor_tensor.item()
            else:
                dino_scale_factor = scale_factor_local

        dino_per_point_batch = dino_per_point_batch * dino_scale_factor
        x = torch.cat([x, dino_per_point_batch], dim=1)

    return x, y, initial_pos_for_encoding, initial_pos_mode2, dino_scale_factor


def model_to_tracks_format(x_dit, horizon, diffuse_on_velocity,
                                num_point_cond, velocity_scale, handle_occlusions, occlusion_scaling,
                                use_dino_features, dino_feature_dim,
                                initial_pos_override=None, motion_history_dim=340,
                                debug_mode=False):
    """
    Convert model format [-1,1] back to track format [0,1] for evaluation.

    Only the supervised channels (XVel, YVel, Occ) are used; conditioning
    and DINO channels are ignored.
    """
    B, C_total, N = x_dit.shape

    layout = get_channel_layout(horizon, num_point_cond, diffuse_on_velocity,
                                 handle_occlusions,
                                 use_dino_features, dino_feature_dim,
                                 motion_history_dim=motion_history_dim)
    num_coord_channels_per_axis = layout['num_coord_channels_per_axis']

    coords = x_dit[:, :2 * num_coord_channels_per_axis, :]

    occlusion_pred = None
    if handle_occlusions:
        occ_start = layout['occ_start']
        occ_end = layout['occ_end']
        occlusion_pred = x_dit[:, occ_start:occ_end, :]
        if occlusion_pred.numel() == 0:
            occlusion_pred = None
        else:
            occlusion_pred = occlusion_pred / occlusion_scaling

    coords = coords.view(B, 2, num_coord_channels_per_axis, N)
    combined = coords.permute(0, 2, 3, 1).contiguous()

    if diffuse_on_velocity:
        tracks = velocity_to_coords(combined, num_point_cond,
                                     velocity_scale, initial_pos_override)
    else:
        tracks = combined

    tracks_01 = (tracks + 1.0) / 2.0

    if handle_occlusions and occlusion_pred is not None:
        occlusion_binary = (occlusion_pred > 0).float()
        return tracks_01, occlusion_pred, occlusion_binary

    return tracks_01


