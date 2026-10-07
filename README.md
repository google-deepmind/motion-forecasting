# Forecasting Motion in the Wild

[Neerja Thakkar](https://neerjathakkar.github.io)<sup>1,2</sup>,
[Shiry Ginosar](https://people.eecs.berkeley.edu/~shiry/)<sup>3</sup>,
[Jacob Walker](https://scholar.google.com/citations?user=DBVqHRMAAAAJ)<sup>2</sup>,
[Jitendra Malik](https://people.eecs.berkeley.edu/~malik/)<sup>1</sup>,
[Joao Carreira](https://scholar.google.com/citations?user=IUZ-7_cAAAAJ)<sup>2</sup>,
[Carl Doersch](https://scholar.google.com/citations?user=Mv-dVOMAAAAJ)<sup>2</sup>

<sup>1</sup> UC Berkeley &nbsp; <sup>2</sup> Google DeepMind &nbsp; <sup>3</sup> TTIC

**ECCV 2026**

[[Paper (Arxiv Version)]](https://arxiv.org/abs/2604.01015) [[Project Page]](https://motion-forecasting.github.io/)

## Overview

We propose dense point trajectories as **visual tokens for behavior**, a structured mid-level representation that disentangles motion from appearance and generalizes across diverse non-rigid agents. Building on this abstraction, we design a **diffusion transformer** that models unordered sets of trajectories and explicitly reasons about occlusion, enabling coherent forecasts of complex motion patterns.

Given a single input image and a short motion history, our model forecasts future animal motion as a set of point trajectories.

## Installation

```bash
# Clone the repository
git clone https://github.com/google-deepmind/motion-forecasting.git
cd motion-forecasting

# Install dependencies
pip install torch torchvision timm hydra-core omegaconf lightning tqdm scipy
```

### DINOv3 Setup

Our model uses frozen [DINOv3](https://arxiv.org/abs/2508.10104) ViT-L features for per-point visual conditioning. You need:

1. A local clone of the DINOv3 repository
2. The pretrained weights (`dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`)

Set the environment variable before training or inference:
```bash
export DINOV3_REPO_DIR=/path/to/your/dinov3
```

## Inference

**Expected workflow:** you have a video, cropped it (square) around the animal, and ran a point tracker (e.g. TAPIR) on the crop — giving per-point `(x, y)` coordinates and visibility over time. Feed the model the first frame of the crop plus the observed track history; it forecasts the future motion.

```bash
python scripts/inference.py \
    --checkpoint /path/to/checkpoint.ckpt \
    --image /path/to/image.png \
    --tracks /path/to/tracks.npz \
    --output predictions.npy \
    --visualize --viz_output viz.png
```

**Input format:**
- `--checkpoint`: Released checkpoints are self-describing (they embed the model config), so no config file is needed. For other checkpoints, pass `--config` (the training run's `config.yaml`) to define the architecture.
- `--image`: First frame of the crop (PNG/JPG). Must be a **square** crop around the animal — it is resized to the model's 256×256 input internally, so non-square images get squashed and distort the predicted motion.
- `--tracks`: Bundled `.npz` with `tracks` `(T, N, 2)` and optionally `visibility` `(T, N)` (1=visible; defaults to all-visible). Coordinates in `[0, 1]` normalized to the image (pixel coordinates are auto-detected and normalized). `T` must be at least the observed history length (`num_point_cond`, default 4). Any `N` works: more points than the model's 320 are uniformly downsampled, fewer are padded internally and stripped from the output. A plain `(T, N, 2)` `.npy` is also accepted.

**Displacement conditioning:** unconditional by default. Pass `--displacement DX DY` to steer the forecast — values are in pixels of the 256x256 model crop (`64 0` = move right by a quarter of the image, `256 256` = the full image diagonal).

**Demo examples:** the released demo examples (one folder per example with `image.png`, `tracks.npz`, `meta.json`) are in exactly this format — point `--image`/`--tracks` at any of them. The format is documented in [docs/DATA_PREPROCESSING.md](docs/DATA_PREPROCESSING.md).

**Output:**
- `predictions.npy`: Predicted future tracks, shape `(T, N, 2)`
- `viz.png` (optional): Visualization of observed + predicted tracks overlaid on the image

**DDIM options:**
- `--use_ddim` / `--no_ddim`: Toggle DDIM sampling (default: on)
- `--ddim_steps 50`: Number of DDIM steps
- `--ddim_eta 0.0`: DDIM stochasticity parameter

## Training

Training uses [Hydra](https://hydra.cc/) for configuration and [Lightning Fabric](https://lightning.ai/docs/fabric/) for distributed training (DDP).

The training config is `conf/train.yaml`; any key can be overridden on the command line (Hydra syntax):

```bash
python scripts/train.py data_root=/path/to/examples epochs=10
```

`data_root` points to a folder of demo-format examples (one subfolder per example with `image.png` + `tracks.npz`; see [docs/DATA_PREPROCESSING.md](docs/DATA_PREPROCESSING.md)).

To finetune from the released checkpoint, set `load_path` (see the comments in `conf/train.yaml`).

### Training with a custom dataloader

The training loop accepts any dataloader that yields batches as a dict with these keys:

| Key | Shape | Description |
|-----|-------|-------------|
| `video` | `(B, T, C, H, W)` | RGB frames, float `[0, 255]` (T=1 frame is used) |
| `tracks` | `(B, T, N, 2)` | Point tracks, normalized `[0, 1]` to the crop |
| `visibility` | `(B, T, N)` | 1=visible, 0=occluded (optional) |
| `point_mask` | `(B, N)` | True=real point, False=padding (optional) |
| `total_displacement` | `(B, 2)` | Mean per-point displacement (optional) |

To use your own dataloader, replace the `ExampleDataset` / `DataLoader` construction in `engine/train.py` with anything that yields this dict.

## Evaluation

Evaluation is configured by `conf/eval.yaml`; any key can be overridden on the command line (OmegaConf dotlist syntax). It runs over a folder of demo-format examples (`data_root`); released checkpoints embed their model config, so only the checkpoint path is needed:

```bash
python -m engine.eval \
    checkpoint=/path/to/model.ckpt \
    data_root=/path/to/examples ddim_steps=100

# With paper-style visualization:
python -m engine.eval checkpoint=/path/to/model.ckpt data_root=/path/to/examples \
    visualize=true viz_style=animal_color viz_dir=viz_out noise_seed=0
```

## Pretrained Checkpoints

| Model | Dataset | Download |
|-------|---------|----------|
| DiT-B (Ours) | MammalMotion | Coming soon |

## MammalMotion Dataset

We release **MammalMotion**, a large-scale dataset of camera-stabilized point trajectories extracted from ~300 hours of unconstrained animal video (MammalNet). The dataset includes:

- Dense point tracks (BootsTAPIR) with visibility annotations
- Camera-stabilized coordinates via RANSAC homography estimation
- Per-animal bounding boxes and segmentation masks (GroundingDINO + VideoSAM)

**Download**:

You can download the full dataset using the Google Cloud CLI. No authentication is required.

**Using gcloud:**
```bash
gcloud storage cp -r gs://representations4d/mammalnet_data/
```

**With wget**

```bash
wget -i https://storage.googleapis.com/representations4d/mammalnet_data/segmentation_manifest.txt
wget -i https://storage.googleapis.com/representations4d/mammalnet_data/animal_manifest.txt
```

## Method

Our approach consists of:

1. **Trajectory Token Construction**: Each point track is encoded as a single token containing DINOv3 features at the initial location, sinusoidal embeddings of the motion history velocities, occlusion indicators, and the noisy diffusion target.

2. **Diffusion Transformer (DiT)**: A standard DiT architecture with adaptive layer norm (AdaLN) conditioning on the diffusion timestep and an optional displacement vector. Tokens attend to each other via self-attention, enabling the model to reason about coherent group motion.

3. **Velocity Parameterization**: We reparameterize tracks as velocities (frame-to-frame deltas) scaled by σ_v=12.0, with occlusions scaled by σ_o=0.1. Occluded velocities are linearly interpolated.

4. **DDIM Sampling**: Inference uses 50 DDIM steps (from 1000 training steps) with deterministic sampling (η=0).

## Citation

```bibtex
@inproceedings{thakkar2026forecasting,
    title={Forecasting Animal Motion in the Wild},
    author={Thakkar, Neerja and Ginosar, Shiry and Walker, Jacob and Malik, Jitendra and Carreira, Joao and Doersch, Carl},
    booktitle={arxiv},
    year={2026}
}
```

## Acknowledgments

We thank Noah Snavely, Andrew Zisserman, Drew Purves, Aleksander Holynski, Linyi Jin, Sander Dieleman, Mark Hamilton, and Jathushan Rajasegeran for helpful discussions and feedback. This work was supported by ONR MURI N00014-21-1-280 and a NSF Graduate Fellowship to NT.

## License & Disclaimer

Copyright 2026 Google LLC  
All materials are licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC-BY-NC). You may obtain a copy of the CC-BY-NC license at: https://creativecommons.org/licenses/by-nc/4.0/legalcode.en
Unless required by applicable law or agreed to in writing, all software and materials distributed here under the Apache 2.0 or CC-BY licenses are distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, 
either express or implied. See the licenses for the specific language governing permissions and limitations under those licenses.
This is not an official Google product.

See [LICENSE](LICENSE) for details.
