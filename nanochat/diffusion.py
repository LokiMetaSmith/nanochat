"""
Diffusion Module for NanoBot Action Generation.

This module implements a minimal Diffusion Probabilistic Model (DDPM)
conditioned on LLM embeddings to generate continuous action/surface vectors.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class DiffusionConfig:
    timesteps: int = 100
    beta_start: float = 0.0001
    beta_end: float = 0.02
    hidden_dim: int = 256
    num_layers: int = 3

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ConditionalDenoiser(nn.Module):
    """
    MLP-based Denoiser.
    Inputs:
    - x_t: Noisy sample (Action/Surface vector)
    - t: Timestep
    - cond: Conditioning embedding (from LLM)
    """
    def __init__(self, action_dim, cond_dim, config: DiffusionConfig):
        super().__init__()

        # Time Embedding
        time_dim = config.hidden_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.Mish(),
            nn.Linear(time_dim, time_dim),
        )

        # Condition Embedding Projector
        self.cond_proj = nn.Linear(cond_dim, config.hidden_dim)

        # Main MLP
        # Input: x_t (action_dim) + time_emb (hidden) + cond_emb (hidden)
        input_dim = action_dim + time_dim + config.hidden_dim

        layers = []
        in_d = input_dim
        for _ in range(config.num_layers):
            layers.append(nn.Linear(in_d, config.hidden_dim))
            layers.append(nn.Mish())
            in_d = config.hidden_dim

        layers.append(nn.Linear(in_d, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t, cond):
        """
        x: (B, action_dim)
        t: (B,)
        cond: (B, cond_dim)
        """
        t_emb = self.time_mlp(t) # (B, hidden)
        c_emb = self.cond_proj(cond) # (B, hidden)

        # Concatenate all inputs
        # (B, action_dim + hidden + hidden)
        h = torch.cat([x, t_emb, c_emb], dim=-1)

        return self.net(h)

class NoiseScheduler(nn.Module):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.timesteps = config.timesteps

        # Define beta schedule
        betas = torch.linspace(config.beta_start, config.beta_end, config.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Register buffers for device handling
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def add_noise(self, x_start, t, noise=None):
        """
        Forward process: x_0 -> x_t
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_prod = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)

        x_t = sqrt_alpha_prod * x_start + sqrt_one_minus_alpha_prod * noise
        return x_t, noise

    @torch.no_grad()
    def sample(self, model, cond, action_dim):
        """
        Reverse process: Generate sample from noise.
        Simple DDPM sampling.
        """
        device = cond.device
        B = cond.shape[0]

        # Start from pure noise
        x = torch.randn(B, action_dim, device=device)

        for i in reversed(range(0, self.timesteps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)

            # Predict noise
            predicted_noise = model(x, t, cond)

            # Equation for x_{t-1}
            alpha = self.alphas[i]
            alpha_hat = self.alphas_cumprod[i]
            beta = self.betas[i]

            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

        return x
