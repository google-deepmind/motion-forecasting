#!/usr/bin/env python3
"""
Minimal inference script for track prediction.

Usage:
    python scripts/inference.py \
        --checkpoint path/to/checkpoint.pt \
        [--config path/to/config.yaml] \
        --image path/to/image.png \
        --tracks path/to/tracks.npz \
        --output predictions.npy \
        [--use_ddim] [--ddim_steps 50] [--ddim_eta 0.0] \
        [--visualize --viz_output viz.mp4]

Model architecture:
    Released checkpoints embed the model_cfg used to build the
    architecture, so only --checkpoint is needed. --config (the training
    run's config.yaml) overrides the embedded config; if neither is
    available, DiT-B defaults are used (only correct for DiT-B checkpoints).

Expected workflow:
    You have a video, cropped it around the animal (square crop), and ran a
    point tracker (e.g. TAPIR) on the crop. That gives per-point (x, y) and
    visibility over time. Feed the model the first frame of the crop plus the
    observed track history; it forecasts the future motion.

Input formats:
    --image:      First frame of the crop (PNG/JPG). Must be a SQUARE crop
                  around the animal; it is resized to the model's input size
                  (256x256) internally. Non-square images are squashed (a
                  warning is logged) and will distort the predicted motion.
    --tracks:     Bundled .npz with 'tracks' (T, N, 2) and optionally
                  'visibility' (T, N, 1=visible; defaults to all-visible) —
                  the demo-example format (see docs/DATA_PREPROCESSING.md).
                  Coordinates normalized to [0, 1] relative to the image
                  (pixel coordinates are auto-detected and normalized).
                  T must be at least num_point_cond (the observed history,
                  default 4). If T reaches the model horizon, the extra
                  timesteps are shown as ground truth in the visualization.
                  A plain (T, N, 2) .npy is also accepted.

    Any number of points N is accepted: more than the model's num_points
    (320) are uniformly downsampled (seeded); fewer are padded internally
    and stripped from the output.

Displacement conditioning:
    Unconditional by default (no displacement conditioning). Pass
    --displacement DX DY to steer the forecast: values are in PIXELS of the
    256x256 model crop, so 256 256 means "move by the full image diagonal"
    and 64 0 means "move right by a quarter of the image".

Output:
    A numpy array of shape (horizon, N_real, 2) with predicted tracks in
    [0, 1] normalized space, saved to --output (N_real = your real points,
    after any downsampling; padding is stripped). If points were
    downsampled, the kept indices are saved next to it as *_indices.npy.

Optional visualization:
    When --visualize is set, overlays predicted tracks on the input image
    and saves to --viz_output (default: viz.mp4).
"""

import argparse
import logging
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motion_forecasting.data import (
    load_image,
    load_tracks,
    normalize_tracks_if_needed,
    prepare_inputs,
)
from motion_forecasting.model.pipeline import (
    create_model,
    create_diffusion,
    create_dino_extractor,
    load_checkpoint,
    load_ema_weights,
    predict,
)
from motion_forecasting.model.dino_feature_extractor import DINOV3_VITL_FEATURE_DIM

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _default_model_cfg():
    """Default model config matching the production training setup."""
    return dict(
        model_name="DiT-B-MammalNet",
        horizon=32,
        num_points=320,
        img_size=256,
        condition_on_displacement=True,
        velocity_conditioning_type="linear",
        vel_disp_dropout=True,
        single_cond_dropout_prob=0.5,
        null_embedding_type="zero",
        vel_disp_conditioning_mode="adaln",
        vel_disp_token_type_embeddings=False,
        motion_history_dim=340,
        motion_history_scale=0.19092,
        handle_occlusions=True,
        diffuse_on_velocity=True,
        use_initial_pos_encoding=True,
        use_image_conditioning=False,
        use_dino_features=True,
        dino_feature_dim=DINOV3_VITL_FEATURE_DIM,
    )


def _default_diffusion_cfg():
    """Default diffusion config."""
    return dict(
        diffusion_steps=1000,
        noise_schedule="linear",
        beta_start=0.0001,
        beta_end=0.02,
        model_mean_type="start_x",
        model_var_type="fixed_small",
        loss_type="mse",
    )


def _default_pipeline_cfg():
    """Default pipeline config for inference."""
    return dict(
        num_point_cond=4,
        velocity_scale=12.0,
        occlusion_scaling=0.1,
        dino_scale_factor=None,
        clip_denoised_multiplier=3.0,
        track_error_metric="l1",
        use_ddim=True,
        ddim_eta=0.0,
        ddim_timesteps=50,
    )


def build_batch(image_tensor, prepared, displacement=None):
    """Construct a pipeline batch dict from the image and prepared inputs.

    displacement: optional (dx, dy) tuple in [0,1] units to steer the
    forecast. When None (unconditional sampling) the displacement values
    are zeros — they are ignored because the null embedding is used.
    """
    tracks_tensor = prepared["track"]
    B, T, N, _ = tracks_tensor.shape
    batch = {
        "video": image_tensor,
        "track": tracks_tensor,
        "visibility": prepared["visibility"],
        "point_mask": prepared["point_mask"],
    }

    if displacement is not None:
        batch["total_displacement"] = torch.tensor([list(displacement)], dtype=torch.float32)
    else:
        batch["total_displacement"] = torch.zeros(B, 2)

    return batch


def visualize_predictions(image_tensor, pred_tracks, gt_tracks, output_path):
    """Create a simple visualization overlaying tracks on the image."""
    try:
        from motion_forecasting.model.track_visualization import draw_track_trajectories
        import cv2
    except ImportError:
        logger.warning("Visualization requires opencv-python. Skipping.")
        return

    img = image_tensor[0, 0].permute(1, 2, 0).cpu().numpy().astype(np.uint8).copy()
    H, W = img.shape[:2]

    gt_np = gt_tracks[0].cpu().numpy()
    pred_np = pred_tracks[0].cpu().numpy()

    img_gt = img.copy()
    draw_track_trajectories(img_gt, gt_np, color=(0, 255, 0), thickness=2, label="GT")

    img_pred = img.copy()
    draw_track_trajectories(img_pred, pred_np, color=(255, 0, 0), thickness=2, label="Pred")

    combined = np.concatenate([img_gt, img_pred], axis=1)

    if output_path.endswith(".mp4"):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, 1, (combined.shape[1], combined.shape[0]))
        writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        writer.release()
    else:
        cv2.imwrite(output_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    logger.info("Visualization saved to %s", output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Run track prediction inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", default=None,
                        help="Path to the training run's config.yaml (defines model "
                             "architecture). If omitted, DiT-B defaults are used.")
    parser.add_argument("--image", required=True, help="Path to input image (first frame of the crop)")
    parser.add_argument("--tracks", required=True,
                        help="Path to tracks bundle (.npz with 'tracks' (T, N, 2) "
                             "and optional 'visibility' (T, N)), in [0,1] "
                             "coordinates (pixel coords auto-normalized). "
                             "T >= num_point_cond. Plain (T, N, 2) .npy also "
                             "accepted (all-visible).")
    parser.add_argument("--displacement", nargs=2, type=float, default=None,
                        metavar=("DX", "DY"),
                        help="Steer the forecast with a target total displacement "
                             "in PIXELS of the 256x256 model crop (e.g. 64 0 = "
                             "move right by a quarter of the image; 256 256 = "
                             "full image diagonal). Default: unconditional "
                             "sampling (no displacement conditioning).")
    parser.add_argument("--noise_seed", type=int, default=None,
                        help="Fixed diffusion noise seed for reproducible "
                             "sampling (useful when comparing displacements)")
    parser.add_argument("--point_seed", type=int, default=0,
                        help="Seed for downsampling when N > model num_points")
    parser.add_argument("--output", default="predictions.npy", help="Output path for predicted tracks")

    parser.add_argument("--use_ddim", action="store_true", default=True, help="Use DDIM sampling (default: True)")
    parser.add_argument("--no_ddim", action="store_true", help="Disable DDIM, use full DDPM sampling")
    parser.add_argument("--ddim_steps", type=int, default=50, help="Number of DDIM steps")
    parser.add_argument("--ddim_eta", type=float, default=0.0, help="DDIM eta parameter")

    parser.add_argument("--no_ema", action="store_true",
                        help="Use live weights instead of the checkpoint's EMA shadow (default: EMA)")
    parser.add_argument("--no_dino", action="store_true", help="Disable DINO features")
    parser.add_argument("--dino_layer", type=int, default=23, help="DINO transformer layer")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--img_size", type=int, default=None,
                        help="Image size (default: from --config, else 256)")

    parser.add_argument("--visualize", action="store_true", help="Generate visualization")
    parser.add_argument("--viz_output", default="viz.png", help="Visualization output path")

    args = parser.parse_args()

    # Load the checkpoint once; checkpoints saved by this codebase embed the
    # flat model_cfg used to build the architecture, so --config is usually
    # unnecessary.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    if args.config is not None:
        # Build the architecture from the training run's config.yaml
        # (same flattening path as engine/train.py and engine/eval.py).
        from omegaconf import OmegaConf
        from engine.train_utils import build_model_cfg

        logger.info("Loading model config from %s", args.config)
        train_cfg = OmegaConf.load(args.config)
        flat_cfg = build_model_cfg(train_cfg)
        # build_model_cfg includes model, diffusion, and pipeline keys
        model_cfg = flat_cfg
        diffusion_cfg = flat_cfg
        pipeline_cfg = dict(flat_cfg)
    elif isinstance(checkpoint, dict) and "model_cfg" in checkpoint:
        logger.info("Using the model_cfg embedded in the checkpoint.")
        flat_cfg = dict(checkpoint["model_cfg"])
        model_cfg = flat_cfg
        diffusion_cfg = flat_cfg
        pipeline_cfg = dict(flat_cfg)
    else:
        logger.warning("No --config given and the checkpoint has no embedded "
                       "model_cfg; using DiT-B default architecture. "
                       "This is only correct for DiT-B checkpoints.")
        model_cfg = _default_model_cfg()
        diffusion_cfg = _default_diffusion_cfg()
        pipeline_cfg = _default_pipeline_cfg()

    if args.img_size is not None:
        model_cfg["img_size"] = args.img_size
    args.img_size = model_cfg.get("img_size", 256)

    if args.no_dino:
        model_cfg["use_dino_features"] = False
        model_cfg["dino_feature_dim"] = 0

    pipeline_cfg["use_ddim"] = not args.no_ddim
    pipeline_cfg["ddim_timesteps"] = args.ddim_steps
    pipeline_cfg["ddim_eta"] = args.ddim_eta

    logger.info("Creating model...")
    model = create_model(model_cfg)

    logger.info("Loading checkpoint from %s", args.checkpoint)
    metadata = load_checkpoint(model, checkpoint)
    if "dino_scale_factor" in metadata:
        pipeline_cfg["dino_scale_factor"] = metadata["dino_scale_factor"]
    if "occlusion_scaling" in metadata:
        pipeline_cfg["occlusion_scaling"] = metadata["occlusion_scaling"]

    if not args.no_ema:
        load_ema_weights(model, checkpoint)

    model.to(args.device)
    model.eval()

    logger.info("Creating diffusion...")
    diffusion = create_diffusion(diffusion_cfg)

    dino_extractor = None
    if model_cfg.get("use_dino_features", False):
        logger.info("Creating DINO extractor (layer %d)...", args.dino_layer)
        dino_extractor = create_dino_extractor({"dino_layer": args.dino_layer, "dino_device": args.device})

    logger.info("Loading image from %s", args.image)
    image_tensor = load_image(args.image, img_size=args.img_size).to(args.device)

    logger.info("Loading tracks from %s", args.tracks)
    tracks_np, visibility_np = load_tracks(args.tracks)
    tracks_np = normalize_tracks_if_needed(tracks_np, args.image)

    T, N = tracks_np.shape[:2]
    num_point_cond = pipeline_cfg.get("num_point_cond", 4)
    logger.info("Input: %d timesteps, %d points (model: horizon=%d, num_points=%d, "
                "num_point_cond=%d)", T, N, model_cfg["horizon"],
                model_cfg["num_points"], num_point_cond)

    prepared = prepare_inputs(
        tracks_np, visibility_np,
        num_points=model_cfg["num_points"],
        horizon=model_cfg["horizon"],
        num_point_cond=num_point_cond,
        seed=args.point_seed,
    )
    for key in ("track", "visibility", "point_mask"):
        prepared[key] = prepared[key].to(args.device)

    # Displacement conditioning: unconditional by default; --displacement
    # (pixels of the model crop) switches to conditional sampling.
    displacement = None
    if args.displacement is not None:
        crop_size = float(model_cfg.get("img_size", 256))
        displacement = (args.displacement[0] / crop_size,
                        args.displacement[1] / crop_size)
        pipeline_cfg["use_null_displacement"] = False
        logger.info("Conditioning on displacement (%.1f, %.1f) px of the "
                    "%dx%d crop = (%.4f, %.4f) normalized.",
                    args.displacement[0], args.displacement[1],
                    int(crop_size), int(crop_size), *displacement)
    else:
        pipeline_cfg["use_null_displacement"] = True
        logger.info("Unconditional sampling (no displacement conditioning). "
                    "Pass --displacement DX DY (pixels out of %d) to steer "
                    "the forecast.", int(model_cfg.get("img_size", 256)))

    if args.noise_seed is not None:
        pipeline_cfg["noise_seed"] = args.noise_seed
        logger.info("Fixed noise seed: %d", args.noise_seed)

    batch = build_batch(image_tensor, prepared, displacement=displacement)

    logger.info("Running inference%s...", " (DDIM)" if pipeline_cfg["use_ddim"] else " (DDPM)")
    with torch.no_grad():
        result = predict(model, diffusion, batch, dino_extractor=dino_extractor, cfg=pipeline_cfg)

    # Strip padding: keep only the user's real points.
    n_real = prepared["n_real"]
    pred_tracks = result["pred_tracks"][0, :, :n_real].cpu().numpy()

    np.save(args.output, pred_tracks)
    logger.info("Predictions saved to %s (shape: %s)", args.output, pred_tracks.shape)
    if len(prepared["kept_indices"]) != N:
        idx_path = os.path.splitext(args.output)[0] + "_indices.npy"
        np.save(idx_path, prepared["kept_indices"])
        logger.info("Input points were downsampled; kept indices saved to %s", idx_path)

    if args.visualize:
        visualize_predictions(
            image_tensor.cpu(),
            result["pred_tracks"][:, :, :n_real].cpu(),
            result["gt_tracks"][:, :, :n_real].cpu(),
            args.viz_output,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
