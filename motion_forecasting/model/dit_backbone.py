"""
DiT architecture for diffusion-based track prediction.

Contains the core DiT_MammalNet_SingleImage model with:
- Transformer backbone (DiTBlock, MaskedAttention, TimestepEmbedder)
- Displacement conditioning (AdaLN and token modes)
- Motion history sin-cos embedding
- ResNet vision encoder for image conditioning
- Factory functions and model registry
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision
from timm.models.vision_transformer import Mlp
from typing import Callable

from .coordinate_utils import (
    get_channel_layout,
    MotionHistorySinCosEmbedding,
    MAX_MOTION_HISTORY_COND,
)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


#################################################################################
#                          ResNet helpers                                        #
#################################################################################

def get_resnet(name: str, weights=None, **kwargs) -> nn.Module:
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = torch.nn.Identity()
    return resnet


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module


def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int = 16) -> nn.Module:
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features // features_per_group,
            num_channels=x.num_features)
    )
    return root_module


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class MaskedAttention(nn.Module):
    """Multi-head self-attention with optional attention mask support."""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DiTBlock(nn.Module):
    """A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = MaskedAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, attn_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """The final layer of DiT."""
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT_MammalNet_SingleImage(nn.Module):
    """
    Diffusion Transformer for point track forecasting.

    Owns all learned parameters: the transformer backbone, vision encoder,
    motion history embedder, and displacement conditioning modules.
    Diffusion logic and DINO feature extraction are external.
    """
    def __init__(
        self,
        horizon=16,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=False,
        cond_dim=512,
        num_points=32,
        img_size=256,
        dino_feature_dim=0,
        handle_occlusions=False,
        diffuse_on_velocity=False,
        use_initial_pos_encoding=False,
        use_image_conditioning=True,
        vel_disp_conditioning_mode="adaln",
        motion_history_dim=340,
        # Conditioning parameters (absorbed from wrapper)
        condition_on_displacement=False,
        velocity_conditioning_type="linear",
        vel_disp_dropout=False,
        single_cond_dropout_prob=0.3,
        null_embedding_type="learned",
        vel_disp_token_type_embeddings=False,
        motion_history_scale=0.19092,
        motion_history_conditioning="embedded",
    ):
        super().__init__()

        self.learn_sigma = learn_sigma
        self.horizon = horizon
        self.num_points = num_points
        self.img_size = img_size
        self.hidden_size = hidden_size
        self.dino_feature_dim = dino_feature_dim
        self.handle_occlusions = handle_occlusions
        self.diffuse_on_velocity = diffuse_on_velocity
        self.use_initial_pos_encoding = use_initial_pos_encoding
        self.use_image_conditioning = use_image_conditioning
        self.vel_disp_conditioning_mode = vel_disp_conditioning_mode
        self.motion_history_dim = motion_history_dim
        self.motion_history_conditioning = motion_history_conditioning

        # Conditioning config
        self.condition_on_displacement = condition_on_displacement
        self.velocity_conditioning_type = velocity_conditioning_type
        self.vel_disp_dropout = vel_disp_dropout
        self.single_cond_dropout_prob = single_cond_dropout_prob
        self.null_embedding_type = null_embedding_type
        self.vel_disp_token_type_embeddings = vel_disp_token_type_embeddings

        # Channel layout
        layout = get_channel_layout(
            horizon, MAX_MOTION_HISTORY_COND, diffuse_on_velocity,
            handle_occlusions, dino_feature_dim > 0, dino_feature_dim,
            motion_history_dim=motion_history_dim,
            motion_history_conditioning=motion_history_conditioning,
        )
        self.in_channels = layout['total_channels']
        self.out_channels = (self.in_channels * 2) if learn_sigma else self.in_channels
        self.num_heads = num_heads

        # --- Core backbone layers ---
        self.x_embedder = nn.Linear(self.in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)

        if use_image_conditioning:
            vision_encoder = get_resnet('resnet18')
            vision_encoder = replace_bn_with_gn(vision_encoder)
            self.vision_encoder = vision_encoder
            self.y_embedder = nn.Linear(cond_dim, hidden_size)
        else:
            self.vision_encoder = None
            self.y_embedder = None

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size=1, out_channels=self.out_channels)

        # --- Motion history embedder (absorbed from wrapper) ---
        # Only the "embedded" scheme uses sin-cos velocity embeddings; the
        # "channel" checkpoints have no embedder (keeps state_dict compatible).
        if motion_history_conditioning == "embedded":
            self.motion_history_embedder = MotionHistorySinCosEmbedding(
                embed_dim_per_vel=motion_history_dim,
                scale=motion_history_scale,
            )
        else:
            self.motion_history_embedder = None

        # --- Displacement conditioning (absorbed from wrapper) ---
        self.displacement_embedder = None
        self.displacement_proj = None

        if condition_on_displacement:
            if velocity_conditioning_type == "linear":
                self.displacement_embedder = nn.Linear(2, hidden_size)
            elif velocity_conditioning_type == "timestep_embedder":
                self.displacement_embedder = TimestepEmbedder(hidden_size)
                self.displacement_proj = nn.Linear(2 * hidden_size, hidden_size)

            if vel_disp_dropout:
                if null_embedding_type == "zero":
                    self.register_buffer("null_displacement_embedding", torch.zeros(1, hidden_size))
                else:
                    self.null_displacement_embedding = nn.Parameter(torch.randn(1, hidden_size) * 0.02)

        # --- Token type embeddings (token mode) ---
        self.token_type_embedding = None
        if vel_disp_conditioning_mode == "token" and condition_on_displacement:
            self.token_type_embedding = nn.Embedding(3, hidden_size)
            nn.init.normal_(self.token_type_embedding.weight, std=0.02)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        x = torch.einsum('npc->ncp', x)
        return x

    @staticmethod
    def coordinate_embedding(coord, dim, max_period=10000):
        coord_scaled = (coord + 1.0) / 2.0 * 1000.0
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=coord.device)
        args = coord_scaled.unsqueeze(-1) * freqs
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[..., :1])], dim=-1)
        return embedding

    # ------------------------------------------------------------------
    # Displacement conditioning methods (absorbed from ConditioningMixin)
    # ------------------------------------------------------------------

    def embed_velocity_displacement(
        self, total_displacement,
        dropout_mask=None, use_null_displacement=False,
    ):
        """Embed displacement into a single (B, D) vector (AdaLN mode)."""
        if not self.condition_on_displacement:
            return None

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        hidden_size = self.hidden_size

        if total_displacement is not None:
            B = total_displacement.shape[0]
        else:
            return torch.zeros(1, hidden_size, device=device, dtype=dtype)

        embedding = torch.zeros(B, hidden_size, device=device, dtype=dtype)

        if self.displacement_embedder is not None:
            disp_emb = self._embed_single(
                self.displacement_embedder, self.displacement_proj,
                total_displacement, B, device, dtype, hidden_size,
            )
            disp_emb = self._apply_null(
                disp_emb, B, device, dtype, hidden_size,
                use_null_displacement,
                self._disp_null_mask(dropout_mask),
                "null_displacement_embedding",
            )
            embedding = embedding + disp_emb

        return embedding

    def embed_velocity_displacement_as_tokens(
        self, total_displacement,
        dropout_mask=None, use_null_displacement=False,
    ):
        """Embed displacement as sequence tokens (token mode).

        Returns (vel_disp_tokens, attn_mask, track_type_emb).
        """
        if not self.condition_on_displacement:
            return None, None, None

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        hidden_size = self.hidden_size
        P = self.num_points

        if total_displacement is not None:
            B = total_displacement.shape[0]
        else:
            return None, None, None

        tokens_list = []
        token_names = []

        if self.displacement_embedder is not None:
            disp_emb = self._embed_single(
                self.displacement_embedder, self.displacement_proj,
                total_displacement, B, device, dtype, hidden_size,
            )
            disp_use_null = self._per_sample_null_flag(
                B, device, use_null_displacement, dropout_mask,
            )
            if disp_use_null.any() and hasattr(self, "null_displacement_embedding"):
                null_disp = self.null_displacement_embedding.expand(B, -1).to(device=device, dtype=dtype)
                disp_emb = torch.where(disp_use_null.unsqueeze(-1).expand(-1, hidden_size), null_disp, disp_emb)
            tokens_list.append(disp_emb.unsqueeze(1))
            token_names.append(("disp", disp_use_null))

        if len(tokens_list) == 0:
            return None, None, None

        vel_disp_tokens = torch.cat(tokens_list, dim=1)
        K = vel_disp_tokens.shape[1]
        S = K + P

        track_type_emb = None
        if self.token_type_embedding is not None and self.vel_disp_token_type_embeddings:
            type_id_map = {"disp": 1}
            for token_idx, (name, _null_mask) in enumerate(token_names):
                type_emb = self.token_type_embedding.weight[type_id_map[name]]
                vel_disp_tokens[:, token_idx, :] = vel_disp_tokens[:, token_idx, :] + type_emb.to(dtype=dtype)
            track_type_emb = self.token_type_embedding.weight[2].unsqueeze(0).to(dtype=dtype)

        attn_mask = torch.ones(B, 1, S, S, dtype=torch.bool, device=device)
        for token_idx, (_name, null_mask) in enumerate(token_names):
            if null_mask.any():
                attn_mask[null_mask, :, :, token_idx] = False
                attn_mask[null_mask, :, token_idx, :] = False

        return vel_disp_tokens, attn_mask, track_type_emb

    def sample_vel_disp_dropout_mask(self, batch_size, device):
        """Sample per-sample dropout mask for displacement conditioning."""
        rand = torch.rand(batch_size, device=device)
        mask = torch.where(
            rand < self.single_cond_dropout_prob,
            torch.tensor(0, device=device),
            torch.tensor(1, device=device),
        )
        return mask

    def _embed_single(self, embedder, proj, values, B, device, dtype, hidden_size):
        if values is not None:
            vals = values.to(device=device, dtype=dtype)
        else:
            vals = torch.zeros(B, 2, device=device, dtype=dtype)

        if self.velocity_conditioning_type == "linear":
            return embedder(vals)
        elif self.velocity_conditioning_type == "timestep_embedder":
            emb_x = embedder(vals[:, 0])
            emb_y = embedder(vals[:, 1])
            return proj(torch.cat([emb_x, emb_y], dim=-1))
        else:
            raise ValueError(f"Unknown velocity_conditioning_type: {self.velocity_conditioning_type}")

    def _disp_null_mask(self, dropout_mask):
        if dropout_mask is None or not self.vel_disp_dropout:
            return None
        return dropout_mask == 0

    def _apply_null(self, emb, B, device, dtype, hidden_size, use_null_flag, null_mask, attr_name):
        if use_null_flag and hasattr(self, attr_name):
            return getattr(self, attr_name).expand(B, -1).to(device=device, dtype=dtype)
        if null_mask is not None and hasattr(self, attr_name):
            null_emb = getattr(self, attr_name).expand(B, -1).to(device=device, dtype=dtype)
            return torch.where(null_mask.unsqueeze(-1).expand(-1, hidden_size), null_emb, emb)
        return emb

    def _per_sample_null_flag(self, B, device, use_null_all, dropout_mask):
        flag = torch.zeros(B, dtype=torch.bool, device=device)
        if use_null_all:
            flag[:] = True
        elif dropout_mask is not None and self.vel_disp_dropout:
            flag = dropout_mask == 0
        return flag

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x, t, y, vel_disp_embedding=None, initial_pos=None,
                vel_disp_tokens=None, attn_mask=None, track_type_emb=None):
        if len(y.shape) == 5:
            y = y[:, 0]

        x = torch.einsum('ncp->npc', x)
        x = self.x_embedder(x)

        if self.use_initial_pos_encoding and initial_pos is not None:
            half_dim = self.hidden_size // 2
            pos_enc_x = self.coordinate_embedding(initial_pos[..., 0], half_dim)
            pos_enc_y = self.coordinate_embedding(initial_pos[..., 1], half_dim)

            if pos_enc_x.dtype != x.dtype:
                pos_enc_x = pos_enc_x.to(x.dtype)
                pos_enc_y = pos_enc_y.to(x.dtype)

            pos_enc = torch.cat([pos_enc_x, pos_enc_y], dim=-1)
            x = x + pos_enc

        if track_type_emb is not None:
            if track_type_emb.dtype != x.dtype:
                track_type_emb = track_type_emb.to(x.dtype)
            x = x + track_type_emb

        num_cond_tokens = 0
        if self.vel_disp_conditioning_mode == "token" and vel_disp_tokens is not None:
            num_cond_tokens = vel_disp_tokens.shape[1]
            x = torch.cat([vel_disp_tokens, x], dim=1)

        t_freq = TimestepEmbedder.timestep_embedding(t, self.t_embedder.frequency_embedding_size)
        if t_freq.dtype != next(self.t_embedder.parameters()).dtype:
            model_dtype = next(self.t_embedder.parameters()).dtype
            t_freq = t_freq.to(model_dtype)
        t = self.t_embedder.mlp(t_freq)

        if self.use_image_conditioning:
            y_features = self.vision_encoder(y)
            y_emb = self.y_embedder(y_features)
            c = t + y_emb
        else:
            c = t

        if self.vel_disp_conditioning_mode == "adaln" and vel_disp_embedding is not None:
            c = c + vel_disp_embedding

        for i, block in enumerate(self.blocks):
            x = block(x, c, attn_mask=attn_mask)

        if num_cond_tokens > 0:
            x = x[:, num_cond_tokens:, :]

        x = self.final_layer(x, c)
        x = self.unpatchify(x)

        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        return self.forward(x, t, y)


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_MammalNet(**kwargs):
    return DiT_MammalNet_SingleImage(depth=28, hidden_size=1152, num_heads=16, **kwargs)

def DiT_L_MammalNet(**kwargs):
    return DiT_MammalNet_SingleImage(depth=24, hidden_size=1024, num_heads=16, **kwargs)

def DiT_B_MammalNet(**kwargs):
    return DiT_MammalNet_SingleImage(depth=12, hidden_size=768, num_heads=12, **kwargs)

def DiT_S_MammalNet(**kwargs):
    return DiT_MammalNet_SingleImage(depth=12, hidden_size=384, num_heads=6, **kwargs)

def DiT_XS_MammalNet(**kwargs):
    return DiT_MammalNet_SingleImage(depth=6, hidden_size=256, num_heads=4, **kwargs)

def DiT_S_half_MammalNet(**kwargs):
    return DiT_MammalNet_SingleImage(depth=6, hidden_size=384, num_heads=6, **kwargs)

DiT_MammalNet_models = {
    'DiT-XL-MammalNet': DiT_XL_MammalNet,
    'DiT-L-MammalNet': DiT_L_MammalNet,
    'DiT-B-MammalNet': DiT_B_MammalNet,
    'DiT-S-MammalNet': DiT_S_MammalNet,
    'DiT-S/2-MammalNet': DiT_S_half_MammalNet,
    'DiT-XS-MammalNet': DiT_XS_MammalNet,
}
