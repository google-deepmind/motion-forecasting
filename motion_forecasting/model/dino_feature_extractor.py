"""
DINOv3 Feature Extractor for point-level feature extraction.

Extracts dense DINO features from images and samples them at specific point locations.
Supports both per-image and batched GPU extraction for efficient training.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import os

# DINOv3 configuration
DINOV3_VITL_PATCH_SIZE = 16
DINOV3_VITL_FEATURE_DIM = 1024
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Pre-computed tensors for GPU normalization (created once, cached per device)
_NORM_MEAN_CACHE = {}
_NORM_STD_CACHE = {}


def _get_norm_tensors(device):
    """Get cached normalization tensors for the given device."""
    if device not in _NORM_MEAN_CACHE:
        _NORM_MEAN_CACHE[device] = torch.tensor(IMAGENET_MEAN, device=device, dtype=torch.float32).view(1, 3, 1, 1)
        _NORM_STD_CACHE[device] = torch.tensor(IMAGENET_STD, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    return _NORM_MEAN_CACHE[device], _NORM_STD_CACHE[device]


class DINOFeatureExtractor:
    """
    Wrapper for extracting dense DINOv3 features from images.
    EXACT copy from phase1_dino_feature_validation.py for consistency.
    """
    
    # ViT-L has 24 transformer blocks (layers 0-23).
    # Layer 23 is the last layer (deepest, most semantic features).
    # Earlier layers (e.g. 11) give mid-level features and are ~2x faster
    # since get_intermediate_layers can skip the remaining blocks.
    VITL_NUM_LAYERS = 24  # total layers in ViT-L
    VITL_LAST_LAYER = 23  # 0-indexed

    def __init__(self, model_name="dinov3_vitl16", checkpoint_path=None, device="cuda",
                 dino_layer=23, repo_dir=None):
        """
        Args:
            model_name: DINO model variant name
            checkpoint_path: Path to model weights (None for default)
            device: Device to load model on
            dino_layer: Which transformer layer to extract features from (0-indexed).
                        23 = last layer (default, deepest/most semantic).
                        11 = mid-level features (~2x faster, skips layers 12-23).
            repo_dir: Path to local DINOv3 repo clone. If None, uses DINOV3_REPO_DIR
                      environment variable.
        """
        self.device = device
        self.patch_size = DINOV3_VITL_PATCH_SIZE
        self.feature_dim = DINOV3_VITL_FEATURE_DIM
        self.dino_layer = dino_layer
        
        if dino_layer < 0 or dino_layer >= self.VITL_NUM_LAYERS:
            raise ValueError(f"dino_layer must be in [0, {self.VITL_NUM_LAYERS - 1}], got {dino_layer}")
        
        if repo_dir is None:
            repo_dir = os.environ.get("DINOV3_REPO_DIR")
        if repo_dir is None or not os.path.exists(repo_dir):
            raise ValueError(
                f"DINOv3 repo not found at '{repo_dir}'. Set the DINOV3_REPO_DIR "
                "environment variable to point to your local dinov3 repo clone."
            )
        
        if checkpoint_path is None:
            checkpoint_path = "dino/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        
        self.model = torch.hub.load(
            repo_or_dir=repo_dir,
            model=model_name,
            source="local",
            weights=checkpoint_path
        )
        self.model.to(device)
        self.model.eval()
        
    def preprocess_image(self, image_tensor, img_size=256):
        """
        Preprocess image for DINO.
        
        Args:
            image_tensor: (C, H, W) tensor in [0, 255] range (from dataloader)
            img_size: Target size (should be divisible by patch_size)
            
        Returns:
            preprocessed: (1, C, H, W) normalized tensor ready for DINO
        """
        # Ensure image is divisible by patch size
        if img_size % self.patch_size != 0:
            img_size = ((img_size + self.patch_size - 1) // self.patch_size) * self.patch_size
        
        # Convert to PIL for easier resizing
        if image_tensor.dim() == 3:
            # Convert from (C, H, W) to (H, W, C) for PIL
            image_np = image_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            image_pil = Image.fromarray(image_np)
        else:
            raise ValueError(f"Expected 3D tensor, got shape {image_tensor.shape}")
        
        # Resize to target size
        image_pil = image_pil.resize((img_size, img_size), Image.BILINEAR)
        
        # Convert to tensor and normalize
        image_tensor = TF.to_tensor(image_pil)  # (C, H, W) in [0, 1]
        image_tensor = TF.normalize(image_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        
        # Add batch dimension
        return image_tensor.unsqueeze(0)  # (1, C, H, W)
    
    def _extract_features_early_exit(self, image_tensor):
        """
        Extract DINO features with early exit: stops after self.dino_layer
        instead of running all 24 ViT-L blocks.

        The upstream get_intermediate_layers runs ALL blocks and just
        collects output at requested indices. This method replicates that
        logic but breaks out of the loop once the target layer is reached,
        giving ~2x speedup when using mid-level features (e.g. layer 11).

        Args:
            image_tensor: (B, C, H, W) preprocessed tensor

        Returns:
            features: (B, feature_dim, H_patches, W_patches) feature map
        """
        model = self.model

        # --- prepare_tokens_with_masks (no masks) ---
        x = model.patch_embed(image_tensor)
        B, H, W, _ = x.shape
        x = x.flatten(1, 2)
        cls_token = model.cls_token + 0 * model.mask_token
        if model.n_storage_tokens > 0:
            storage_tokens = model.storage_tokens
        else:
            storage_tokens = torch.empty(
                1, 0, cls_token.shape[-1],
                dtype=cls_token.dtype, device=cls_token.device,
            )
        x = torch.cat([
            cls_token.expand(B, -1, -1),
            storage_tokens.expand(B, -1, -1),
            x,
        ], dim=1)

        # --- run blocks with early exit ---
        target_layer = self.dino_layer
        for i, blk in enumerate(model.blocks):
            if model.rope_embed is not None:
                rope_sincos = model.rope_embed(H=H, W=W)
            else:
                rope_sincos = None
            x = blk(x, rope_sincos)
            if i == target_layer:
                break  # early exit — skip remaining blocks

        # --- norm + reshape (same as get_intermediate_layers with norm=True, reshape=True) ---
        if hasattr(model, 'untie_cls_and_patch_norms') and model.untie_cls_and_patch_norms:
            x_norm_cls_reg = model.cls_norm(x[:, : model.n_storage_tokens + 1])
            x_norm_patch = model.norm(x[:, model.n_storage_tokens + 1 :])
            x = torch.cat((x_norm_cls_reg, x_norm_patch), dim=1)
        else:
            x = model.norm(x)

        # strip cls + storage tokens, keep only patch tokens
        x = x[:, model.n_storage_tokens + 1 :]
        # reshape to spatial: (B, H*W, D) -> (B, D, H, W)
        # H, W are already patch grid dims from patch_embed (e.g. 16x16 for 256px / 16px patch)
        x = x.reshape(B, H, W, -1)
        x = x.permute(0, 3, 1, 2).contiguous()

        return x  # (B, feature_dim, H_patches, W_patches)

    def extract_features(self, image_tensor):
        """
        Extract dense DINO features from image.
        
        Args:
            image_tensor: (1, C, H, W) preprocessed tensor
            
        Returns:
            features: (1, feature_dim, H_patches, W_patches) feature map
        """
        with torch.inference_mode():
            features = self._extract_features_early_exit(image_tensor.to(self.device))
        return features
    
    def sample_features_at_points(self, features, point_coords, coord_space="0_1"):
        """
        Sample DINO features at point locations using bilinear interpolation.
        Uses patch-center-based interpolation: points at patch centers get 100% weight.
        
        Supports both single-image and batched inputs:
          - Single: features (1, D, Hp, Wp) + point_coords (N, 2) -> (N, D)
          - Batched: features (B, D, Hp, Wp) + point_coords (B, N, 2) -> (B, N, D)
        
        F.grid_sample processes each batch element independently, so batched output[i]
        is identical to calling this function with features[i:i+1] and point_coords[i].
        
        Args:
            features: (1, feature_dim, H_patches, W_patches) or (B, ...) feature map
            point_coords: (N, 2) or (B, N, 2) point coordinates in [X, Y] format
            coord_space: "0_1" if coords in [0,1], "-1_1" if in [-1,1]
            
        Returns:
            sampled_features: (N, feature_dim) or (B, N, feature_dim) features at each point
        """
        if point_coords.dim() == 2:
            point_coords = point_coords.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B, N, _ = point_coords.shape
        _, _, H_patches, W_patches = features.shape
        
        # Convert coordinates to grid_sample format [-1, 1] with patch-center alignment
        if coord_space == "0_1":
            # Convert to patch coordinates (where patch centers are at integer values)
            patch_coords_x = point_coords[:, :, 0] * W_patches - 0.5
            patch_coords_y = point_coords[:, :, 1] * H_patches - 0.5
            
            # Convert to grid_sample coordinates [-1, 1]
            grid_coords_x = (patch_coords_x / (W_patches - 1)) * 2.0 - 1.0
            grid_coords_y = (patch_coords_y / (H_patches - 1)) * 2.0 - 1.0
            
            grid_coords = torch.stack([grid_coords_x, grid_coords_y], dim=-1)
            
        elif coord_space == "-1_1":
            # Convert to [0, 1] first
            coords_01 = (point_coords + 1.0) / 2.0
            
            # Apply same transformation
            patch_coords_x = coords_01[:, :, 0] * W_patches - 0.5
            patch_coords_y = coords_01[:, :, 1] * H_patches - 0.5
            
            grid_coords_x = (patch_coords_x / (W_patches - 1)) * 2.0 - 1.0
            grid_coords_y = (patch_coords_y / (H_patches - 1)) * 2.0 - 1.0
            
            grid_coords = torch.stack([grid_coords_x, grid_coords_y], dim=-1)
        else:
            raise ValueError(f"Unknown coord_space: {coord_space}")
        
        # grid_sample expects (B, H, W, 2) format
        grid_coords = grid_coords.unsqueeze(1)  # (B, 1, N, 2)
        
        # Ensure same device and dtype
        features = features.to(grid_coords.device)
        if features.dtype != torch.float32:
            features = features.float()
        if grid_coords.dtype != torch.float32:
            grid_coords = grid_coords.float()
        
        # Sample features using bilinear interpolation
        sampled = F.grid_sample(
            features,
            grid_coords,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        # Output: (B, feature_dim, 1, N)
        
        # Reshape to (B, N, feature_dim)
        sampled = sampled.squeeze(2).permute(0, 2, 1)  # (B, N, feature_dim)
        
        if squeeze_output:
            sampled = sampled.squeeze(0)  # (N, feature_dim)
        
        return sampled

    # ================================================================
    # Batched GPU methods — equivalent to per-sample methods but faster
    # ================================================================

    def preprocess_batch(self, image_tensors, img_size=256):
        """
        Batch preprocess images for DINO entirely on GPU.
        Produces results equivalent to calling preprocess_image() on each image.

        The original per-image path does:
          1. tensor -> numpy uint8 (truncation) -> PIL Image
          2. PIL BILINEAR resize to img_size
          3. TF.to_tensor (PIL uint8 -> float32 / 255.0)
          4. TF.normalize with ImageNet mean/std

        This batched version replicates that on GPU:
          1. clamp + uint8 truncation (matches numpy astype(uint8))
          2. F.interpolate if resize needed
          3. / 255.0
          4. normalize with ImageNet mean/std

        Args:
            image_tensors: (B, C, H, W) tensor in [0, 255] range (any device/dtype)
            img_size: Target size (should be divisible by patch_size)

        Returns:
            preprocessed: (B, C, H, W) normalized float32 tensor on self.device
        """
        if img_size % self.patch_size != 0:
            img_size = ((img_size + self.patch_size - 1) // self.patch_size) * self.patch_size

        # Move to GPU and ensure float32
        x = image_tensors.to(device=self.device, dtype=torch.float32)
        B, C, H, W = x.shape

        # Match uint8 truncation from original PIL pipeline:
        # Original does .cpu().numpy().astype(np.uint8) which truncates float -> uint8.
        # torch.uint8 conversion also truncates towards zero, matching numpy behavior.
        x = x.clamp(0, 255).to(torch.uint8).float()

        # Resize if needed
        if H != img_size or W != img_size:
            # Note: PIL BILINEAR on uint8 and F.interpolate on float may differ by
            # up to ~1/255 per pixel. In practice images are already img_size x img_size
            # so this branch rarely executes.
            x = F.interpolate(x, size=(img_size, img_size), mode='bilinear',
                              align_corners=False, antialias=False)
            x = x.clamp(0, 255)

        # Convert to [0, 1] range (equivalent to TF.to_tensor dividing PIL uint8 by 255)
        x = x / 255.0

        # Normalize with ImageNet mean/std (equivalent to TF.normalize)
        mean, std = _get_norm_tensors(x.device)
        x = (x - mean) / std

        return x

    def extract_features_batch(self, image_batch):
        """
        Extract dense DINO features from a batch of images.
        Equivalent to calling extract_features() on each image individually,
        since ViT processes each sample in the batch independently.

        Args:
            image_batch: (B, C, H, W) preprocessed (normalized) tensor

        Returns:
            features: (B, feature_dim, H_patches, W_patches) feature map
        """
        with torch.inference_mode():
            features = self._extract_features_early_exit(image_batch.to(self.device))
        return features

    def extract_point_features_batch(self, images, point_coords, img_size=256,
                                     coord_space="-1_1"):
        """
        All-in-one batched DINO feature extraction at point locations.
        Equivalent to the per-sample loop:
            for each sample: preprocess_image -> extract_features -> sample_features_at_points

        Args:
            images: (B, C, H, W) tensor in [0, 255] range
            point_coords: (B, N, 2) point coordinates in [X, Y] format
            img_size: Target size for DINO preprocessing
            coord_space: "0_1" or "-1_1" for point coordinate space

        Returns:
            point_features: (B, N, feature_dim) DINO features sampled at each point
        """
        preprocessed = self.preprocess_batch(images, img_size=img_size)
        features = self.extract_features_batch(preprocessed)
        point_features = self.sample_features_at_points(features, point_coords,
                                                        coord_space=coord_space)
        return point_features
