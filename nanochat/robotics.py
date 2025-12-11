"""
Robotics Interface Module for Nanochat (NanoBot).

This module implements the interface for:
1.  Raw Sensor Telemetry (Proprioception: Joints, Accel).
2.  Latent Surface Vectors (Communication between External NN and LLM).

It treats these inputs as additional modalities that are projected into the
LLM's embedding space and prepended as tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class RoboticsConfig:
    sensor_dim: int = 64     # Dimension of raw sensor vector (e.g. 18 joints + 3 accel)
    surface_dim: int = 128   # Dimension of the neural surface vector (latent comms)
    sensor_tokens: int = 1   # Number of tokens to represent sensor state
    surface_tokens: int = 4  # Number of tokens to represent surface state
    projector_dim: int = 768 # Should match LLM n_embd (set dynamically if needed)

class Projector(nn.Module):
    """
    Generic MLP Projector: Input Dim -> [Hidden] -> Output Dim (n_tokens * n_embd)
    """
    def __init__(self, input_dim, output_dim, n_tokens=1):
        super().__init__()
        self.n_tokens = n_tokens
        self.output_dim = output_dim # This is per-token dim (n_embd)

        # We project to n_tokens * output_dim so we can reshape later
        total_out = n_tokens * output_dim

        # Simple 2-layer MLP
        self.net = nn.Sequential(
            nn.Linear(input_dim, total_out), # Expand/Compress directly
            nn.GELU(),
            nn.Linear(total_out, total_out)  # Refine
        )

    def forward(self, x):
        # x: (B, input_dim)
        B = x.shape[0]
        x = self.net(x) # (B, n_tokens * output_dim)
        x = x.view(B, self.n_tokens, self.output_dim) # (B, T, C)
        return x

class RoboticsInterface(nn.Module):
    """
    Combines Sensor and Surface inputs into a sequence of embeddings.
    """
    def __init__(self, config: RoboticsConfig, llm_dim: int):
        super().__init__()
        self.config = config
        self.sensor_dim = config.sensor_dim
        self.surface_dim = config.surface_dim

        # Sensor Projector (Raw Telemetry -> LLM Space)
        if self.sensor_dim > 0:
            self.sensor_proj = Projector(
                input_dim=config.sensor_dim,
                output_dim=llm_dim,
                n_tokens=config.sensor_tokens
            )
        else:
            self.sensor_proj = None

        # Surface Projector (Latent Neural State -> LLM Space)
        if self.surface_dim > 0:
            self.surface_proj = Projector(
                input_dim=config.surface_dim,
                output_dim=llm_dim,
                n_tokens=config.surface_tokens
            )
        else:
            self.surface_proj = None

    def forward(self, sensors=None, surface=None):
        """
        Args:
            sensors: (B, sensor_dim) or None
            surface: (B, surface_dim) or None
        Returns:
            embeddings: (B, T_robot, n_embd) where T_robot = T_sensor + T_surface
        """
        embeddings_list = []

        # Process Sensors
        if self.sensor_proj is not None and sensors is not None:
            # Check shape just in case
            assert sensors.shape[-1] == self.sensor_dim, f"Sensor dim mismatch: {sensors.shape[-1]} vs {self.sensor_dim}"
            sensor_emb = self.sensor_proj(sensors) # (B, T_sens, C)
            embeddings_list.append(sensor_emb)

        # Process Surface
        if self.surface_proj is not None and surface is not None:
            assert surface.shape[-1] == self.surface_dim, f"Surface dim mismatch: {surface.shape[-1]} vs {self.surface_dim}"
            surface_emb = self.surface_proj(surface) # (B, T_surf, C)
            embeddings_list.append(surface_emb)

        if not embeddings_list:
            return None

        # Concatenate along time dimension
        return torch.cat(embeddings_list, dim=1)

    def get_num_tokens(self):
        count = 0
        if self.sensor_proj is not None:
            count += self.config.sensor_tokens
        if self.surface_proj is not None:
            count += self.config.surface_tokens
        return count
