import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(
        self,
        linear_layer: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0
    ):
        """
        Wraps an existing nn.Linear layer with Low-Rank Adaptation (LoRA).

        Args:
            linear_layer: The frozen base linear layer.
            rank: The rank 'r' of the low-rank decomposition.
            alpha: The scaling factor alpha.
            dropout: Dropout probability for the LoRA input.
        """
        super().__init__()
        self.base_layer = linear_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Determine dimensions from the base layer
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features

        # LoRA weights
        # A: (in_features, rank) -> Kaiming Uniform
        # B: (rank, out_features) -> Zero
        self.lora_A = nn.Parameter(torch.zeros(self.in_features, self.rank))
        self.lora_B = nn.Parameter(torch.zeros(self.rank, self.out_features))

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Mark base layer parameters as frozen
        for param in self.base_layer.parameters():
            param.requires_grad = False

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize A with Kaiming Uniform (same as standard linear layer default for weight)
        # We follow the original paper: "We use a random Gaussian initialization for A and zero for B"
        # However, nn.Linear uses kaiming_uniform_ by default.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Base forward pass (frozen)
        # We reuse the base layer logic to handle bias etc.
        base_out = self.base_layer(x)

        # LoRA forward pass: x @ A @ B * scaling
        # x: (..., in_features)
        # A: (in_features, rank)
        # B: (rank, out_features)

        lora_in = self.dropout(x)

        # Low-rank path
        # (..., in) @ (in, r) -> (..., r)
        # (..., r) @ (r, out) -> (..., out)
        # We use F.linear for convenience: F.linear(input, weight) -> input @ weight.T
        # So for x @ A, we do F.linear(x, A.T)

        lora_out = (lora_in @ self.lora_A) @ self.lora_B

        return base_out + (lora_out * self.scaling)

    def merge(self):
        """
        Merges the LoRA weights into the base layer (for inference/export).
        This is permanent.
        """
        if self.rank > 0:
            # W_new = W_old + (A @ B) * scaling
            # Note: nn.Linear stores weight as (out, in)
            # A @ B gives (in, out). We need transpose.
            # (A @ B).T = B.T @ A.T

            delta_w = (self.lora_A @ self.lora_B).T * self.scaling
            self.base_layer.weight.data += delta_w
            self.rank = 0 # Mark as merged

    def __repr__(self):
        return f"LoRALinear(in={self.in_features}, out={self.out_features}, rank={self.rank}, alpha={self.alpha})"
