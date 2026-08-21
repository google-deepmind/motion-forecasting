# Data preprocessing

This document describes (1) the example format that training, evaluation,
and inference consume, (2) how to produce that format from your own video,
and (3) how the MammalMotion training data was constructed, so the
preprocessing is reproducible end to end.

## 1. The example format

Training (`engine/train.py`) and evaluation (`engine/eval.py`) read a folder
of examples, one subfolder per example:

```
examples_root/
    <example_name>/
        image.png    first frame of a square crop around the animal
        tracks.npz   'tracks' (T, N, 2) float32, normalized [0, 1]
                     'visibility' (T, N) float32, 1 = visible (optional)
        meta.json    optional free-form provenance (ignored by loading)
```

Conventions:

- **`image.png`** is a square crop. Any resolution works — it is resized to
  the model input size (256×256) on load. Non-square images are squashed, always crop square.
- **`tracks.npz` coordinates are normalized to the crop**: `(0, 0)` is the
  crop's top-left corner, `(1, 1)` its bottom-right. `x` is horizontal.
  Pixel coordinates are auto-detected (magnitudes > 10; normalized tracks
  can legitimately leave the crop, so moderate out-of-range values are kept
  as-is) and normalized by the image size on load.
- **`T` (timesteps)**: at least `num_point_cond` (the observed history, 4 by
  default). Inference extends shorter sequences to the model horizon (32) by
  repeating the last observation; sequences longer than the horizon are
  truncated. For training and evaluation, provide the full horizon so the
  future timesteps act as ground truth.
- **`N` (points)**: any count. More than the model's `num_points` (320) are
  uniformly downsampled (seeded); fewer are zero-padded internally with a
  `point_mask` marking the real points. Padding is stripped from outputs.
- **`visibility`** marks occluded timesteps with 0. If omitted, all points
  are treated as visible.

`scripts/inference.py` accepts the same `image.png` + `tracks.npz` pair
directly (a plain `(T, N, 2)` `.npy` also works).

## 2. Producing examples from your own video

The intended workflow mirrors what the model saw in training:

1. **Crop**: pick the animal, take a square crop around it with generous
   margin (training crops padded the detection box by 50% of its larger
   side before squaring). Save the first frame as `image.png`.
2. **Track**: run a point tracker (e.g. CoTracker / BootsTAPIR) on the cropped
   clip. Query points on the animal, not the background. Keep the tracker's
   per-point visibility/occlusion output.
3. **Normalize**: divide track coordinates by the crop size so they live in
   `[0, 1]`, and save `tracks.npz` with `tracks` and `visibility`.
4. **Frame rate**: training tracks were sampled at 15 fps with a 32-step
   horizon (~2 s of motion). Subsample your tracks to a similar rate for
   best results.

If the camera moves, stabilize the tracks first (see below) — the model was
trained on camera-stabilized trajectories, so unstabilized tracks conflate
camera and animal motion.

## 3. How the MammalMotion training data was built

The released dataset was constructed from MammalNet videos with the
following pipeline. Numbers in parentheses are the values used for the
release.

**Upstream inputs, per video:**

- Per-animal segmentation masks from a detector + video segmenter
  (GroundingDINO + VideoSAM, detection confidence 0.4), stored as RLE.
- Dense point tracks from BootsTAPIR (500 query points per animal, run on
  8-second chunks at 256×256 track resolution).
- Per-chunk homographies estimated from background points (RANSAC) for
  camera stabilization.

**Crop construction, per frame and animal:**

1. Take the animal's mask bounding box.
2. Pad it on every side by 50% of `max(width, height)`.
3. Expand the padded box to a square; clamp to the image bounds.
4. Crop and resize to 256×256.
5. Quality filters: crops smaller than 200×200 px or with aspect ratio
   above 3 (before squaring) are discarded.

The resulting square box (the `padded_square` box of the **first frame** of
each sample window) defines the coordinate frame: all track coordinates are
normalized to it, and `image.png` is that crop.

**Track processing, per sample:**

1. **Stabilization**: tracks are mapped through the per-frame homographies
   so coordinates are expressed relative to a static camera (stabilized to
   the first frame of the sample window).
2. **Normalization**: stabilized tracks are normalized to `[0, 1]` using the
   first frame's `padded_square` crop box.
3. **Temporal sampling**: 30 fps tracks are subsampled to 15 fps (every 2nd
   frame) and windowed to a 32-step horizon with a stride of 8 frames
   between windows. Chunks shorter than 16 frames are skipped.
4. **Occlusions**: occluded timesteps are linearly interpolated in
   coordinate space; the visibility mask still marks them as occluded so
   the model can learn occlusion patterns.
5. **Point sampling**: up to 320 points per sample; samples with fewer are
   zero-padded with `point_mask` marking the real points.
6. **Displacement**: the mean per-point displacement from first to last
   visible position, `total_displacement` `(2,)`, computed from the
   normalized tracks (the released code recomputes this on load, so it does
   not need to be stored).
