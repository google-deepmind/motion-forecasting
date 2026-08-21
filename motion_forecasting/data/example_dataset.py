"""Dataset and input-preparation utilities for the demo-example format.

An example is a folder containing what a user would have after running a
point tracker (e.g. TAPIR) on a square crop around an animal:

    examples_root/
        <example_name>/
            image.png   first frame of the crop (any square size; resized
                        to the model input size internally)
            tracks.npz  'tracks' (T, N, 2) float32 in [0, 1] normalized to
                        the crop, and optionally 'visibility' (T, N)
                        float32 (1=visible; defaults to all-visible)
            meta.json   optional free-form provenance (ignored by loading)

`ExampleDataset` serves these folders as training/eval batches matching the
pipeline batch contract. `prepare_inputs` (also used directly by
scripts/inference.py) adapts raw tracker output of any (T, N) to the model's
fixed horizon and point count.
"""

import json
import logging
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from motion_forecasting.model.coordinate_utils import (
    compute_velocity_displacement_from_tracks,
)

logger = logging.getLogger(__name__)


def load_image(path, img_size=256):
    """Load an image and convert to the expected tensor format.

    Returns:
        (1, 1, 3, H, W) float32 tensor in [0, 255] range.
    """
    from PIL import Image
    import torchvision.transforms.functional as TF

    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w != h:
        logger.warning(
            "Input image is %dx%d (not square) and will be squashed to "
            "%dx%d. The model expects a SQUARE crop around the animal; "
            "non-square inputs distort the predicted motion geometry.",
            w, h, img_size, img_size)
    if (w, h) != (img_size, img_size):
        logger.info("Resizing image %dx%d -> %dx%d (model input size).",
                    w, h, img_size, img_size)
    img = img.resize((img_size, img_size), Image.BILINEAR)
    tensor = TF.to_tensor(img) * 255.0  # (3, H, W) in [0, 255]
    return tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, H, W)


def load_tracks(path):
    """Load the bundled tracks file (.npz).

    Expected keys:
      - 'tracks':     (T, N, 2) point coordinates
      - 'visibility': (T, N), 1=visible, 0=occluded (optional;
                      defaults to all-visible)

    A plain .npy of shape (T, N, 2) is also accepted (all points assumed
    visible).

    Returns:
        (tracks float32 (T, N, 2), visibility float32 (T, N) or None)
    """
    data = np.load(path)
    visibility = None
    if isinstance(data, np.lib.npyio.NpzFile):
        if "tracks" not in data:
            raise ValueError(f"{path} is an .npz without a 'tracks' key "
                             f"(found: {list(data.keys())})")
        tracks = data["tracks"]
        if "visibility" in data:
            visibility = data["visibility"].astype(np.float32)
    else:
        tracks = data
    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        raise ValueError(f"Tracks must have shape (T, N, 2); got {tracks.shape}")
    return tracks.astype(np.float32), visibility


# Normalized tracks are [0, 1] relative to the crop, but stabilized tracks
# can legitimately leave the crop (values of roughly +-5 are normal). Only
# magnitudes far beyond that indicate pixel coordinates.
_PIXEL_COORD_THRESHOLD = 10.0


def normalize_tracks_if_needed(tracks, image_path):
    """Auto-normalize pixel-space tracks to [0, 1] using the source image size.

    Normalized tracks may extend well outside [0, 1] (points that leave the
    crop after stabilization), so only clearly pixel-scale magnitudes
    (> _PIXEL_COORD_THRESHOLD) trigger normalization.
    """
    if np.abs(tracks).max() <= _PIXEL_COORD_THRESHOLD:
        return tracks
    from PIL import Image
    with Image.open(image_path) as img:
        w, h = img.size
    logger.info("Tracks look like pixel coordinates (max |coord|=%.1f); "
                "normalizing by the image size (%dx%d).",
                np.abs(tracks).max(), w, h)
    out = tracks.copy()
    out[..., 0] /= float(w)
    out[..., 1] /= float(h)
    return out


def prepare_inputs(tracks, visibility, num_points, horizon, num_point_cond,
                   seed=0, verbose=True):
    """Adapt raw tracker output (T, N, 2) to the model's fixed input shape.

    - T >= num_point_cond required. If T < horizon, the last observed
      position/visibility is repeated forward (those timesteps are
      forecast by the model anyway).
    - N > num_points: uniform random downsample (seeded, sorted indices).
    - N < num_points: zero-pad with point_mask=False.

    Returns:
        dict with batched tensors: track (1, horizon, N_model, 2),
        visibility (1, horizon, N_model), point_mask (1, N_model),
        plus kept_indices (N_kept,) into the input points, n_real,
        and history_only (bool: no future timesteps were provided).
    """
    log = logger.info if verbose else logger.debug

    T, N, _ = tracks.shape
    if T < num_point_cond:
        raise ValueError(
            f"Tracks have {T} timesteps but the model needs at least "
            f"num_point_cond={num_point_cond} observed timesteps of history.")

    if visibility is None:
        visibility = np.ones((T, N), dtype=np.float32)
    if visibility.shape != (T, N):
        raise ValueError(
            f"Visibility shape {visibility.shape} does not match tracks {(T, N)}")
    visibility = visibility.astype(np.float32)

    # --- Point count: downsample or pad to num_points ---
    if N > num_points:
        rng = np.random.RandomState(seed)
        kept_indices = np.sort(rng.choice(N, size=num_points, replace=False))
        tracks = tracks[:, kept_indices]
        visibility = visibility[:, kept_indices]
        log("Downsampled %d -> %d points (seed=%d).", N, num_points, seed)
        n_real = num_points
    else:
        kept_indices = np.arange(N)
        n_real = N

    point_mask = np.zeros(num_points, dtype=bool)
    point_mask[:n_real] = True
    if n_real < num_points:
        pad_n = num_points - n_real
        tracks = np.concatenate(
            [tracks, np.zeros((T, pad_n, 2), dtype=np.float32)], axis=1)
        visibility = np.concatenate(
            [visibility, np.zeros((T, pad_n), dtype=np.float32)], axis=1)
        log("Padded %d -> %d points (point_mask marks the %d real ones).",
            n_real, num_points, n_real)

    # --- Timesteps: extend history to the model horizon ---
    history_only = T <= num_point_cond
    if T < horizon:
        reps = horizon - T
        tracks = np.concatenate(
            [tracks, np.repeat(tracks[-1:], reps, axis=0)], axis=0)
        visibility = np.concatenate(
            [visibility, np.repeat(visibility[-1:], reps, axis=0)], axis=0)
        log("Extended %d observed timesteps to horizon %d "
            "(future is forecast by the model).", T, horizon)
    elif T > horizon:
        logger.warning("Tracks have %d timesteps; using the first %d (horizon).",
                       T, horizon)
        tracks = tracks[:horizon]
        visibility = visibility[:horizon]

    return {
        "track": torch.from_numpy(tracks).float().unsqueeze(0),
        "visibility": torch.from_numpy(visibility).float().unsqueeze(0),
        "point_mask": torch.from_numpy(point_mask).unsqueeze(0),
        "kept_indices": kept_indices,
        "n_real": n_real,
        "history_only": history_only,
    }


class ExampleDataset(Dataset):
    """Dataset over a folder of demo-format examples.

    Each sample is a dict matching the pipeline batch contract
    (unbatched; use a standard DataLoader / default_collate to batch):

        video              (1, 3, H, W) float32 in [0, 255]
        tracks             (horizon, num_points, 2) float32 in [0, 1]
        visibility         (horizon, num_points) float32
        point_mask         (num_points,) bool
        total_displacement (2,) float32, computed from the tracks
        name               example folder name (provenance)

    Args:
        root: directory containing one subfolder per example
              (image.png + tracks.npz).
        num_points: model's fixed point count (pad/downsample target).
        horizon: model horizon (timesteps).
        num_point_cond: observed history length the model expects.
        img_size: model input image size.
        seed: seed for point downsampling when an example has more
              points than num_points.
    """

    def __init__(self, root, num_points=320, horizon=32, num_point_cond=4,
                 img_size=256, seed=0):
        self.root = root
        self.num_points = num_points
        self.horizon = horizon
        self.num_point_cond = num_point_cond
        self.img_size = img_size
        self.seed = seed

        if not os.path.isdir(root):
            raise FileNotFoundError(f"Example root does not exist: {root}")
        self.example_dirs = sorted(
            os.path.join(root, d) for d in os.listdir(root)
            if os.path.isfile(os.path.join(root, d, "tracks.npz"))
            and os.path.isfile(os.path.join(root, d, "image.png"))
        )
        if not self.example_dirs:
            raise ValueError(
                f"No examples found under {root} (expected subfolders "
                f"containing image.png and tracks.npz).")
        logger.info("Found %d examples under %s", len(self.example_dirs), root)

    def __len__(self):
        return len(self.example_dirs)

    def load_meta(self, idx):
        """Return the example's meta.json contents ({} if absent)."""
        meta_path = os.path.join(self.example_dirs[idx], "meta.json")
        if not os.path.isfile(meta_path):
            return {}
        with open(meta_path) as f:
            return json.load(f)

    def __getitem__(self, idx):
        example_dir = self.example_dirs[idx]
        image_path = os.path.join(example_dir, "image.png")
        tracks_path = os.path.join(example_dir, "tracks.npz")

        image = load_image(image_path, img_size=self.img_size)  # (1,1,3,H,W)
        tracks_np, visibility_np = load_tracks(tracks_path)
        tracks_np = normalize_tracks_if_needed(tracks_np, image_path)

        prepared = prepare_inputs(
            tracks_np, visibility_np,
            num_points=self.num_points,
            horizon=self.horizon,
            num_point_cond=self.num_point_cond,
            seed=self.seed,
            verbose=False,
        )
        tracks = prepared["track"][0]          # (horizon, num_points, 2)
        visibility = prepared["visibility"][0]  # (horizon, num_points)
        point_mask = prepared["point_mask"][0]  # (num_points,)

        _, total_displacement = compute_velocity_displacement_from_tracks(
            tracks.unsqueeze(0), visibility.unsqueeze(0),
            point_mask.unsqueeze(0),
        )

        return {
            "video": image[0],                       # (1, 3, H, W)
            "tracks": tracks,
            "visibility": visibility,
            "point_mask": point_mask,
            "total_displacement": total_displacement[0].float(),  # (2,)
            "name": os.path.basename(example_dir),
        }
