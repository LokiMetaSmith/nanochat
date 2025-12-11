"""
Vision Encoder Module for Nanochat (NanoBot/NanoJanus).

This module implements a lightweight Vision Transformer (ViT) and a Projector
to interface visual data with the LLM's latent space.
It is designed to be:
1.  Lightweight (pure PyTorch, no heavy deps like timm/torchvision).
2.  Modular (easy to swap out for SigLIP or other encoders).
3.  Compatible with Strix Halo (ROCm) via standard PyTorch ops.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionConfig:
    def __init__(self,
                 image_size=224,
                 patch_size=14,
                 width=768,
                 layers=12,
                 heads=12,
                 mlp_ratio=4.0,
                 channels=3,
                 output_dim=None):
        self.image_size = image_size
        self.patch_size = patch_size
        self.width = width
        self.layers = layers
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.channels = channels
        self.output_dim = output_dim # If None, defaults to width

class PatchEmbed(nn.Module):
    """ Turn images into patch embeddings. """
    def __init__(self, img_size=224, patch_size=14, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."

        x = self.proj(x) # (B, embed_dim, grid_h, grid_w)
        x = x.flatten(2).transpose(1, 2) # (B, num_patches, embed_dim)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Flash Attention is standard in Nanochat
        # We use the scaled_dot_product_attention which handles Flash internally
        x = F.scaled_dot_product_attention(q, k, v, scale=self.scale)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class VisionTransformer(nn.Module):
    """
    A minimal Vision Transformer.
    """
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config

        self.patch_embed = PatchEmbed(
            img_size=config.image_size,
            patch_size=config.patch_size,
            in_chans=config.channels,
            embed_dim=config.width
        )

        # Learnable position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, config.width))

        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "norm1": nn.LayerNorm(config.width),
                "attn": Attention(config.width, num_heads=config.heads),
                "norm2": nn.LayerNorm(config.width),
                "mlp": MLP(config.width, int(config.width * config.mlp_ratio))
            }) for _ in range(config.layers)
        ])

        self.norm = nn.LayerNorm(config.width)

        self.init_weights()

    def init_weights(self):
        # Simple initialization
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.patch_embed(x) # (B, N, D)
        x = x + self.pos_embed

        for block in self.blocks:
            # Standard ViT Block
            x = x + block["attn"](block["norm1"](x))
            x = x + block["mlp"](block["norm2"](x))

        x = self.norm(x)
        return x

class VisionProjector(nn.Module):
    """
    Projects vision embeddings to the LLM's dimension.
    Based on Janus/LLaVA (typically a 2-layer MLP).
    """
    def __init__(self, vision_dim, llm_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = llm_dim

        self.net = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, llm_dim)
        )

    def forward(self, x):
        return self.net(x)
