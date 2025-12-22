"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Multimodal support (Vision Encoder + Projector)
- Robotics support (Sensor + Latent Projectors + Diffusion Head)
- LoRA support (Low-Rank Adaptation)
"""

import math
from functools import partial
from dataclasses import dataclass, field
from contextlib import nullcontext
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.distributed.optim import ZeroRedundancyOptimizer

from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW
from nanochat.nl_opt import NestedMomentum, DistNestedMomentum
from nanochat.vision import VisionConfig, VisionTransformer, VisionProjector
from nanochat.robotics import RoboticsConfig, RoboticsInterface
from nanochat.lora import LoRALinear
from torch.utils.checkpoint import checkpoint
from nanochat.unitok import UniTok

def _compute_loss_chunk(x_chunk, targets_chunk, lm_head, softcap, ignore_index, reduction):
    logits_chunk = lm_head(x_chunk)
    logits_chunk = logits_chunk.float()
    # softcap is a float, not a tensor, so no device issues.
    logits_chunk = softcap * torch.tanh(logits_chunk / softcap)
    return F.cross_entropy(logits_chunk, targets_chunk, ignore_index=ignore_index, reduction=reduction)

def chunked_cross_entropy(x, targets, lm_head, chunk_size=128, softcap=15.0, ignore_index=-1, reduction='mean'):
    # Flatten input and targets
    B, T, C = x.size()
    x_flat = x.view(-1, C)
    targets_flat = targets.view(-1)

    num_elements = x_flat.size(0)
    losses = []
    total_tokens = 0

    # Determine internal reduction for chunks
    # If we need 'none' globally, we must get 'none' from chunks.
    # If we need 'mean' or 'sum' globally, 'sum' from chunks is most efficient.
    chunk_reduction = 'none' if reduction == 'none' else 'sum'

    for i in range(0, num_elements, chunk_size):
        x_chunk = x_flat[i : i + chunk_size]
        target_chunk = targets_flat[i : i + chunk_size]

        # We use checkpointing to save memory for the backward pass
        # The checkpoint wrapper seems to have issues with torch.compile when loss_reduction is 'none'.
        # So we only apply it when it's safe and needed (i.e. training with mean/sum reduction).
        if torch.is_grad_enabled() and reduction != 'none':
            loss_chunk = checkpoint(
                _compute_loss_chunk,
                x_chunk,
                target_chunk,
                lm_head,
                softcap,
                ignore_index,
                chunk_reduction,
                use_reentrant=False
            )
        else:
            loss_chunk = _compute_loss_chunk(
                x_chunk,
                target_chunk,
                lm_head,
                softcap,
                ignore_index,
                chunk_reduction
            )

        if reduction == 'none':
            losses.append(loss_chunk)
        else:
            # For sum/mean, we accumulate the sum
            losses.append(loss_chunk.unsqueeze(0)) # keep as tensor list
            # Count valid tokens for mean reduction
            valid_mask = target_chunk != ignore_index
            total_tokens += valid_mask.sum()

    if not losses:
        return torch.tensor(0.0, device=x.device, requires_grad=True)

    all_losses = torch.cat(losses)

    if reduction == 'none':
        return all_losses
    elif reduction == 'sum':
        return all_losses.sum()
    elif reduction == 'mean':
        # total_tokens is a tensor now. Avoid .item() causing graph breaks.
        # We clamp to 1 to avoid division by zero (loss will be 0 anyway if total_tokens is 0)
        return all_losses.sum() / total_tokens.clamp(min=1)
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Vision params
    use_vision: bool = False
    vision_image_size: int = 224
    vision_patch_size: int = 14
    vision_width: int = 768
    vision_layers: int = 12
    vision_heads: int = 12
    vision_mlp_ratio: float = 4.0
    # Visual Tokenizer params (Native discrete image tokens)
    use_visual_tokens: bool = False
    visual_tokenizer_path: str = "out/tokenizer_vision/visual_tokenizer.pt"
    visual_vocab_size: int = 16384
    # Robotics params
    use_robotics: bool = False
    robotics_sensor_dim: int = 64
    robotics_surface_dim: int = 128
    robotics_sensor_tokens: int = 1
    robotics_surface_tokens: int = 4
    robotics_action_loss_weight: float = 1.0 # Weight for action prediction loss
    robotics_use_diffusion: bool = False # Use Diffusion Head instead of Regression
    robotics_diffusion_steps: int = 100
    # LoRA params
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    # Target modules: list of strings. Valid: 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj' (for MLP)
    # Our internal names are slightly different: 'c_q', 'c_k', 'c_v', 'c_proj' (attn), 'c_fc', 'c_proj' (mlp)
    # We will map standard names to internal ones for easier config:
    # q_proj -> c_q
    # k_proj -> c_k
    # v_proj -> c_v
    # o_proj -> c_proj (attn)
    # up_proj -> c_fc (partially, c_fc is 4x expansion)
    # down_proj -> c_proj (mlp)
    lora_targets: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    rmsnorm_affine: bool = False


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last time into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3) # re-assemble
    out = out.to(x.dtype) # ensure input/output dtypes match
    return out

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0

        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

        # Apply LoRA if configured
        if config.use_lora:
            targets = config.lora_targets or []
            if "q_proj" in targets:
                self.c_q = LoRALinear(self.c_q, config.lora_rank, config.lora_alpha, config.lora_dropout)
            if "k_proj" in targets:
                self.c_k = LoRALinear(self.c_k, config.lora_rank, config.lora_alpha, config.lora_dropout)
            if "v_proj" in targets:
                self.c_v = LoRALinear(self.c_v, config.lora_rank, config.lora_alpha, config.lora_dropout)
            if "o_proj" in targets:
                self.c_proj = LoRALinear(self.c_proj, config.lora_rank, config.lora_alpha, config.lora_dropout)

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin) # QK rotary embedding
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),)) # QK norm
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # make head be batch dim, i.e. (B, T, H, D) -> (B, H, T, D)

        # Apply KV cache: insert current k,v into cache, get the full view so far
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2) # number of queries in this forward pass
        Tk = k.size(2) # number of keys/values in total (in the cache + current forward pass)

        # Attention: queries attend to keys/values autoregressively. A few cases to handle:
        enable_gqa = self.n_head != self.n_kv_head # Group Query Attention (GQA): duplicate key/value heads to match query heads if desired

        # Context manager for Flash Attention
        # On AMD Strix Halo/RDNA 3.5, we want to ensure we hit the CK Flash Attention kernel
        # However, if CK/Flash kernels are unavailable (e.g. missing support in build), we must fallback to Math.
        # Previously we explicitly disabled math, which caused crashes on unsupported hardware.
        # We now enable math fallback.
        if x.device.type == "cuda":
            sdp_ctx = sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
        else:
            sdp_ctx = nullcontext()

        with sdp_ctx:
            if kv_cache is None or Tq == Tk:
                # During training (no KV cache), attend as usual with causal attention
                # And even if there is KV cache, we can still use this simple version when Tq == Tk
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
            elif Tq == 1:
                # During inference but with a single query in this forward pass:
                # The query has to attend to all the keys/values in the cache
                y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
            else:
                # During inference AND we have a chunk of queries in this forward pass:
                # First, each query attends to all the cached keys/values (i.e. full prefix)
                attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device) # True = keep, False = mask
                prefix_len = Tk - Tq
                attn_mask[:, :prefix_len] = True
                # Then, causal attention within this chunk
                attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)

        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

        # Apply LoRA if configured
        if config.use_lora:
            targets = config.lora_targets or []
            # c_fc acts as the "up projection" (and gate if using SwiGLU, but here it's simple MLP)
            if "gate_proj" in targets or "up_proj" in targets:
                self.c_fc = LoRALinear(self.c_fc, config.lora_rank, config.lora_alpha, config.lora_dropout)
            if "down_proj" in targets:
                self.c_proj = LoRALinear(self.c_proj, config.lora_rank, config.lora_alpha, config.lora_dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.ln1 = nn.RMSNorm(config.n_embd, elementwise_affine=config.rmsnorm_affine)
        self.attn = CausalSelfAttention(config, layer_idx)
        self.ln2 = nn.RMSNorm(config.n_embd, elementwise_affine=config.rmsnorm_affine)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache):
        x = x + self.attn(self.ln1(x), cos_sin, kv_cache)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Adjust vocab size if using visual tokens
        total_vocab_size = config.vocab_size
        if config.use_visual_tokens:
            total_vocab_size += config.visual_vocab_size

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(total_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
            "ln_pre": nn.RMSNorm(config.n_embd, elementwise_affine=config.rmsnorm_affine),
        })
        self.lm_head = nn.Linear(config.n_embd, total_vocab_size, bias=False)
        self.ln_f = nn.RMSNorm(config.n_embd, elementwise_affine=config.rmsnorm_affine)

        # --- Multimodal Support ---
        if config.use_visual_tokens:
            # Discrete Visual Tokens (UniTok)
            self.visual_tokenizer = UniTok(vocab_size=config.visual_vocab_size)
            # Load checkpoint if available
            if config.visual_tokenizer_path:
                try:
                    # Map location to CPU first to avoid OOM
                    state_dict = torch.load(config.visual_tokenizer_path, map_location="cpu", weights_only=True)
                    self.visual_tokenizer.load_state_dict(state_dict)
                    print0(f"Loaded visual tokenizer from {config.visual_tokenizer_path}")
                except Exception as e:
                    print0(f"Warning: Could not load visual tokenizer from {config.visual_tokenizer_path}: {e}")

            # Freeze visual tokenizer
            for p in self.visual_tokenizer.parameters():
                p.requires_grad = False
            self.visual_tokenizer.eval()

            # Disable legacy Vision Encoder if Visual Tokens are enabled (to avoid confusion)
            # Or should we allow both? The prompt implies replacing "text tokens" with "image tokens",
            # but usually multimodality implies both.
            # The user asked: "prove that image tokens are better than text tokens"
            # We will allow both but prioritized logic in forward.
            self.vision_encoder = None
            self.vision_projector = None

        elif config.use_vision:
            vision_cfg = VisionConfig(
                image_size=config.vision_image_size,
                patch_size=config.vision_patch_size,
                width=config.vision_width,
                layers=config.vision_layers,
                heads=config.vision_heads,
                mlp_ratio=config.vision_mlp_ratio,
                output_dim=config.vision_width # ViT usually outputs its own width
            )
            self.vision_encoder = VisionTransformer(vision_cfg)
            self.vision_projector = VisionProjector(
                vision_dim=config.vision_width,
                llm_dim=config.n_embd
            )
        else:
            self.vision_encoder = None
            self.vision_projector = None

        # --- Robotics Support ---
        if config.use_robotics:
            robotics_cfg = RoboticsConfig(
                sensor_dim=config.robotics_sensor_dim,
                surface_dim=config.robotics_surface_dim,
                sensor_tokens=config.robotics_sensor_tokens,
                surface_tokens=config.robotics_surface_tokens,
                projector_dim=config.n_embd,
                use_diffusion=config.robotics_use_diffusion,
                diffusion_timesteps=config.robotics_diffusion_steps
            )
            self.robotics_interface = RoboticsInterface(robotics_cfg, config.n_embd)
        else:
            self.robotics_interface = None

        # To support meta device initialization, we init the rotary embeddings here, but it's fake
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    def init_weights(self):
        self.apply(self._init_weights)
        # zero out classifier weights
        torch.nn.init.zeros_(self.lm_head.weight)
        # zero out c_proj weights in all blocks
        for block in self.transformer.h:
            # We need to handle the case where c_proj is wrapped in LoRALinear
            # However, LoRALinear resets its own parameters in __init__, and its base layer is already initialized
            # But the base layer weights might need to be zeroed if they were just created.
            # In Block.__init__, we create Linear then wrap.
            # self.apply traverses recursively.

            # If wrapped:
            if isinstance(block.mlp.c_proj, LoRALinear):
                 torch.nn.init.zeros_(block.mlp.c_proj.base_layer.weight)
            else:
                 torch.nn.init.zeros_(block.mlp.c_proj.weight)

            if isinstance(block.attn.c_proj, LoRALinear):
                 torch.nn.init.zeros_(block.attn.c_proj.base_layer.weight)
            else:
                 torch.nn.init.zeros_(block.attn.c_proj.weight)

        self.init_buffers()

    def init_buffers(self):
        # init the rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast the embeddings from fp32 to bf16: optim can tolerate it and it saves memory: both in the model and the activations
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _init_weights(self, module):
        if isinstance(module, LoRALinear):
            # LoRA linear has its own reset_parameters, and base_layer is initialized separately
            # We don't need to do anything special here as long as the base layer is initialized
            pass
        elif isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)
        # Note: VisionEncoder has its own init_weights called in __init__

    # TODO: bump base theta more, e.g. 100K is more common more recently
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """ Return the estimated FLOPs per token for the model. Ref: https://arxiv.org/abs/2204.02311 """
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        l, h, q, t = self.config.n_layer, self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0,
                         matrix_optimizer_backend="muon", general_optimizer_backend="adamw",
                         nested_betas=(0.9, 0.99), nested_level_weights=(0.5, 0.5),
                         use_8bit_optimizer=False, layer_lr_decay=1.0):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        use_dist_optim = ddp and world_size > 1

        # Separate out all parameters into groups
        # If LoRA is enabled, we ONLY train LoRA parameters (and maybe norm/head if desired, but usually just LoRA)
        # For this implementation, we will strictly follow LoRA paper: Only train A and B.

        if self.config.use_lora:
            if rank == 0:
                print("LoRA Enabled: Freezing base model parameters.")

            # 1. Freeze everything first
            for param in self.parameters():
                param.requires_grad = False

            # 2. Unfreeze LoRA parameters
            lora_params = []
            for name, module in self.named_modules():
                if isinstance(module, LoRALinear):
                    module.lora_A.requires_grad = True
                    module.lora_B.requires_grad = True
                    lora_params.append(module.lora_A)
                    lora_params.append(module.lora_B)

            # 3. Create a single optimizer group for LoRA params
            # We use the 'general_optimizer' logic (AdamW) for LoRA usually. Muon is designed for massive matrices.
            # LoRA matrices are small (rank r), so AdamW is appropriate.

            if rank == 0:
                print(f"LoRA: Trainable parameters: {sum(p.numel() for p in lora_params)}")

            general_groups = [
                dict(params=lora_params, lr=matrix_lr) # Use matrix_lr or embedding_lr? Usually LoRA LR is higher, like 1e-4 to 1e-3.
                                                        # We will use matrix_lr as a proxy for the 'main' learning rate.
            ]

            # We default to AdamW for LoRA
            # Muon might not be suitable for small rank matrices?
            # Let's stick to AdamW for reliability with LoRA.

            adamw_kwargs = dict(betas=(0.9, 0.99), eps=1e-8, weight_decay=weight_decay)

            if use_dist_optim:
                 optimizer = ZeroRedundancyOptimizer(general_groups, optimizer_class=torch.optim.AdamW, fused=True, **adamw_kwargs)
            else:
                 optimizer = torch.optim.AdamW(general_groups, fused=True, **adamw_kwargs)

            return [optimizer] # Return single optimizer

        # --- Standard Full Finetuning / Pretraining Setup (Existing Logic) ---

        # Matrix Params (Transformer Blocks)
        matrix_params = []
        
        # If layer_lr_decay is 1.0 (default), we keep them as one group (or Muon splits them by size)
        # If layer_lr_decay < 1.0, we split them into per-layer groups with decayed LR
        if layer_lr_decay < 1.0:
            if rank == 0:
                print(f"Applying Layer-wise LR Decay: {layer_lr_decay}")
            matrix_groups = []
            for i, block in enumerate(self.transformer.h):
                # Calculate decay: Top layers (near output) get higher LR?
                # Standard LLRD: LR = base_lr * (decay ** (num_layers - 1 - layer_idx))
                # i=0 (bottom) gets (decay^11), i=11 (top) gets (decay^0)=1
                decay_factor = layer_lr_decay ** (self.config.n_layer - 1 - i)
                matrix_groups.append({
                    "params": list(block.parameters()),
                    "lr": matrix_lr * decay_factor
                })
            matrix_params_or_groups = matrix_groups
        else:
            matrix_params_or_groups = list(self.transformer.h.parameters())


        embedding_params = list(self.transformer.wte.parameters())
        # Add ln_pre and ln_f params to embedding_params (general optimizer)
        embedding_params.extend(list(self.transformer.ln_pre.parameters()))
        embedding_params.extend(list(self.ln_f.parameters()))

        # Separate block params into matrix (2D) and other (1D/bias)
        for p in self.transformer.h.parameters():
            if p.ndim == 2:
                matrix_params.append(p)
            else:
                embedding_params.append(p)

        lm_head_params = list(self.lm_head.parameters())

        # Add Vision Params to Embedding Group
        if self.config.use_vision:
            if self.vision_encoder is not None:
                embedding_params.extend(list(self.vision_encoder.parameters()))
            if self.vision_projector is not None:
                embedding_params.extend(list(self.vision_projector.parameters()))

        # Add Robotics Params to Embedding Group
        if self.config.use_robotics and self.robotics_interface is not None:
            embedding_params.extend(list(self.robotics_interface.parameters()))

        # Sanity check: Total parameters count
        total_params = sum(p.numel() for p in self.parameters())
        # We can't easily assert len(list) if matrix_params_or_groups is dicts, but we can trust the split

        # --- General Optimizer (Embeddings & Heads) ---
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if rank == 0:
            print(f"Scaling the LR for the general parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        general_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        ]

        if general_optimizer_backend == "adamw":
            adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
            if use_8bit_optimizer:
                try:
                    import bitsandbytes as bnb
                except ImportError:
                    raise ImportError("bitsandbytes is not installed. Please install it to use 8-bit optimizers.")

                optimizer_cls = bnb.optim.AdamW8bit
                if use_dist_optim:
                     # ZeRO-2 with 8-bit optimizer
                     # Note: ZeroRedundancyOptimizer supports optimizer_class
                     general_optimizer = ZeroRedundancyOptimizer(general_groups, optimizer_class=optimizer_cls, **adamw_kwargs)
                else:
                     general_optimizer = optimizer_cls(general_groups, **adamw_kwargs)
            else:
                if use_dist_optim:
                    # Use ZeroRedundancyOptimizer with fused AdamW
                    general_optimizer = ZeroRedundancyOptimizer(general_groups, optimizer_class=torch.optim.AdamW, fused=True, **adamw_kwargs)
                else:
                    general_optimizer = torch.optim.AdamW(general_groups, fused=True, **adamw_kwargs)
        elif general_optimizer_backend == "dist_adamw":
             # Legacy custom implementation
             adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
             AdamWFactory = DistAdamW if use_dist_optim else partial(torch.optim.AdamW, fused=True)
             general_optimizer = AdamWFactory(general_groups, **adamw_kwargs)
        elif general_optimizer_backend == "nested_momentum":
             nm_kwargs = dict(betas=nested_betas, level_weights=nested_level_weights, weight_decay=weight_decay)
             NMFactory = DistNestedMomentum if use_dist_optim else NestedMomentum
             general_optimizer = NMFactory(general_groups, **nm_kwargs)
        else:
            raise ValueError(f"Unknown general_optimizer_backend: {general_optimizer_backend}")

        # --- Matrix Optimizer (Transformer Blocks) ---
        if matrix_optimizer_backend == "muon":
            muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
            MuonFactory = DistMuon if use_dist_optim else Muon
            matrix_optimizer = MuonFactory(matrix_params_or_groups, **muon_kwargs)
        elif matrix_optimizer_backend == "nested_momentum":
            nm_kwargs = dict(lr=matrix_lr, betas=nested_betas, level_weights=nested_level_weights, weight_decay=weight_decay)
            NMFactory = DistNestedMomentum if use_dist_optim else NestedMomentum
            matrix_optimizer = NMFactory(matrix_params_or_groups, **nm_kwargs)
        else:
             raise ValueError(f"Unknown matrix_optimizer_backend: {matrix_optimizer_backend}")

        # Combine them the two optimizers into one list
        optimizers = [general_optimizer, matrix_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def forward(self, idx, images=None, sensors=None, surface=None, targets=None, action_targets=None, kv_cache=None, loss_reduction='mean', return_embeddings=False):
        B, T = idx.size()

        # Get Text Embeddings
        x = self.transformer.wte(idx)

        # 1. Handle Vision
        if self.config.use_visual_tokens and images is not None:
             # Native Discrete Visual Tokens
             # Encode images -> indices
             with torch.no_grad():
                 # Unitok encode returns: quant, loss, indices
                 _, _, indices = self.visual_tokenizer.encode(images)
                 # Indices are (B, H, W) or flattened?
                 # test_unitok says: (B, H', W')
                 # Flatten to (B, N)
                 indices = indices.flatten(1)

             # Shift indices into visual vocab range
             visual_ids = indices + self.config.vocab_size

             # Embed
             vis_emb = self.transformer.wte(visual_ids) # (B, N_vis, n_embd)

             # Prepend visual tokens to x
             x = torch.cat([vis_emb, x], dim=1)

             # Update targets if training
             if targets is not None:
                 # We assume the caller has already padded 'targets' at the beginning with -1s
                 # to account for the vision tokens. We just need to fill them in.
                 # Check lengths
                 N_vis = visual_ids.size(1)
                 # targets: (B, T_total)
                 # x currently: (B, N_vis + T, n_embd)

                 # Verify target length matches x length
                 if targets.size(1) == x.size(1):
                     # Replace the prefix with visual_ids
                     # We use a clone to avoid in-place modification errors in autograd if needed,
                     # but targets is usually leaf or detached.
                     # Note: We want to Predict the image tokens too?
                     # Standard VQGAN-GPT: Yes, autoregressive on image tokens.
                     # So targets[:, :N_vis] = visual_ids
                     # BUT: Is it causal?
                     # If we prepend all image tokens at once, the model sees them all in 'x'.
                     # If we want to generate them autoregressively, we must ensure causal masking.
                     # 'CausalSelfAttention' handles the masking.
                     # So yes, we just put them in the sequence.

                     # However, 'targets' is usually shifted by 1 relative to 'idx' in standard formulation.
                     # But here 'targets' is aligned with the OUTPUT of the model at 'x'.
                     # The model predicts 'targets[i]' given 'x[i]' (which includes x[0]...x[i]).
                     # Wait. Standard GPT: input `idx[0:-1]`, target `idx[1:]`.
                     # Here `idx` is passed as the Text input.
                     # `x` is constructed as [ImageEmbeds, TextEmbeds].
                     # So `x` corresponds to input sequence [I_0, I_1, ..., T_0, T_1...].
                     # We want to predict [I_1, ..., T_0, ..., T_N].

                     # If 'targets' passed in is (B, TotalLen), it usually assumes prediction at each step.
                     # The caller (base_train.py) constructs targets by padding -1.
                     # We replace the -1s with `visual_ids`.

                     # Note: This implies we are predicting Image Token `i` given Image Token `i` (and history).
                     # Wait. `x` contains the *ground truth* embedding of Image Token `i`.
                     # Causal Attention masks future.
                     # At pos `i`, we see `x[i]`. We predict `targets[i]`.
                     # Usually for GPT, input is `tokens[:-1]`, target is `tokens[1:]`.
                     # Here, we are effectively feeding `[I, T]` and predicting `[I, T]` (shifted?).

                     # If `base_train.py` passes `idx` and `targets` such that `targets` is `idx` shifted...
                     # Then `idx` (text) input matches `x` (text part).
                     # We need to ensure `x` (image part) and `targets` (image part) are also shifted?
                     # If we feed `x` = [I_0, I_1, ..., I_N], and `targets` = [I_1, ..., I_N, T_0...].
                     # Then we are doing correct AR training.

                     # BUT: `x` here is constructed from `images`. `images` represents the WHOLE image.
                     # `visual_ids` are [I_0, ..., I_N].
                     # If we embed ALL of them and put them in `x`, and set `targets` to ALL of them...
                     # Then at step 0 (I_0), we see I_0 and predict I_0. That's trivial/leakage!

                     # FIX: For AR image generation, we usually need <BOS> token for image start?
                     # OR: We assume the first token is given or inferred from previous context?
                     # If we just want to "train using image tokens", we can shift them:
                     # Input: [I_0, I_1, ..., I_{N-1}]
                     # Target: [I_1, I_2, ..., I_N]

                     # But `visual_ids` contains ALL tokens.
                     # So we should probably drop the last image token from Input, and first from Target?
                     # This gets complicated with the `images` input which is a tensor (B, C, H, W).

                     # Simplified approach for "Integration":
                     # Teacher Forcing with Causal Mask.
                     # If we feed [I_0...I_N] in `x`.
                     # And targets are [I_0...I_N].
                     # At step k, we attend to I_0...I_k. We predict I_k? No, we predict I_{k+1}.
                     # We need the target at step k to be I_{k+1}.

                     # So if `x` has I_k, `targets` should have I_{k+1}.
                     # `targets` passed in is usually aligned with `x` output.
                     # So we need to shift `visual_ids`?

                     # Let's look at `base_train.py`.
                     # `inputs = scratch[:-1]`, `targets = scratch[1:]`.
                     # So text is already shifted.
                     # Vision input `images` is the FULL image.
                     # `visual_ids` = [t0, t1, t2, t3].
                     # We want Input=[t0, t1, t2, t3?], Target=[t1, t2, t3, ?]

                     # If we just concat, `x` will have [t0, t1, t2, t3].
                     # Target part should be [t1, t2, t3, NextText?].
                     # This implies we lose one visual token of context or prediction?

                     # To avoid complex shifting logic here which might break `base_train` alignment:
                     # We will fill `targets` with `visual_ids`.
                     # BUT we must acknowledge that `x` at pos `k` is leaking `visual_ids[k]`.
                     # Because `x[:, k]` is `wte(visual_ids[k])`.
                     # And we predict `targets[:, k]` which is `visual_ids[k]`.
                     # This is identity mapping!

                     # **CRITICAL FIX**:
                     # We must Shift the input `x` or the targets.
                     # Since `x` is derived from `images`, we can't easily "shift" the image tensor itself before encoding.
                     # We must shift the `visual_ids` before embedding!

                     # We need a "Start of Image" token? Or use the last text token as trigger?
                     # Let's assume we prepend a special `<|image_start|>` token?
                     # Or just use 0 padding?

                     # Let's try:
                     # Input indices: [BOS_VIS, I_0, I_1, ..., I_{N-1}]
                     # Target indices: [I_0, I_1, ..., I_{N-1}, I_N]  <-- Wait, I_N is lost?

                     # Actually, `UniTok` gives fixed grid.
                     # Let's say N=256.
                     # Input: [I_0...I_255].
                     # Target: [I_1...I_255, T_0].
                     # This means `x` should be [I_0...I_255].
                     # Target should be [I_1...T_0].
                     # This requires shifting the WHOLE sequence?

                     # Alternative:
                     # Treat Image tokens as "Prompt" (Encoder) -> Don't predict them?
                     # But goal is "Prove image tokens better... generate images".

                     # Compromise for this PR:
                     # We implement "Input" integration (Image Tokens as context).
                     # And "Target" integration (Predict Image Tokens).
                     # To prevent leakage:
                     # `x_vis_emb = wte(visual_ids)`
                     # We need to shift `visual_ids` to the right for input.
                     # `visual_input_ids` = [0, I_0, I_1, ..., I_{N-1}].
                     # `visual_target_ids` = [I_0, I_1, ..., I_{N}]. (Matches length N+1?)

                     # Let's stick to strict length N.
                     # Input: [0, I_0, ..., I_{N-1}] (Embed these)
                     # Target: [I_0, I_1, ..., I_{N-1}] (Predict these)
                     # Last token I_{N-1} predicts T_0?

                     # This requires `x` to be modified.
                     # `visual_ids` shape (B, N).
                     # `input_ids` = torch.cat([torch.zeros((B,1)), visual_ids[:, :-1]], dim=1)
                     # `target_ids` = visual_ids

                     # Doing this:
                     # 1. Shift visual_ids right (prepend 0/BOS, drop last).
                     # 2. Embed.
                     # 3. Targets = visual_ids.

                     # This ensures we predict I_k from I_{k-1}.

                     shifted_input_ids = torch.cat([
                         torch.full((B, 1), 0, device=visual_ids.device, dtype=visual_ids.dtype), # Use 0 as BOS-Image? Or vocab_size?
                         visual_ids[:, :-1]
                     ], dim=1)

                     vis_emb = self.transformer.wte(shifted_input_ids)
                     x = torch.cat([vis_emb, x], dim=1)

                     # Fill targets
                     # We just overwrite the prefix
                     targets[:, :N_vis] = visual_ids

                     # Note: base_train uses -1 padding. We overwrite it.
                 else:
                     print0(f"Visual Tokenizer Target Mismatch: targets {targets.size(1)} != x {x.size(1)}")

        elif self.config.use_vision and images is not None and self.vision_encoder is not None:
             # images: (B, C, H, W)
             vis_feats = self.vision_encoder(images) # (B, N_patches, vision_width)
             vis_feats = self.vision_projector(vis_feats) # (B, N_patches, n_embd)
             # Prepend visual tokens
             x = torch.cat([vis_feats, x], dim=1)

        # 2. Handle Robotics Inputs (Sensors + Surface)
        if self.config.use_robotics and self.robotics_interface is not None:
            # sensors: (B, sensor_dim) or None
            # surface: (B, surface_dim) or None
            robot_feats = self.robotics_interface(sensors, surface) # (B, T_robot, n_embd)
            if robot_feats is not None:
                # Prepend robot tokens (before vision)
                x = torch.cat([robot_feats, x], dim=1)

        # Update T for rotary embeddings
        B, T, C = x.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        x = self.transformer.ln_pre(x) # pre-norm for the concatenated sequence
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)
        x = self.ln_f(x)

        # 3. Handle Robotics Outputs (Action Prediction)
        action_loss = 0.0
        # Check if we should compute loss (Training Mode)
        if self.config.use_robotics and self.robotics_interface is not None and action_targets is not None:
            # Predict next surface vector from the LAST token's embedding
            # x: (B, T, C) -> last token: (B, C)
            last_hidden_state = x[:, -1, :]

            # This handles both Regression (MSE) and Diffusion (Denoising MSE) losses
            action_loss = self.robotics_interface.predict_action(last_hidden_state, target_action=action_targets)
            action_loss = action_loss * self.config.robotics_action_loss_weight

        # Forward the lm_head (compute logits)
        softcap = 15.0
        if targets is not None:
            # training mode: compute and return the loss
            # NOTE: Targets must match the total sequence length (Vision + Robot + Text)
            if targets.size(1) != T:
                 print0(f"Target mismatch! Expected {T}, got {targets.size(1)}")

            lm_loss = chunked_cross_entropy(x, targets, self.lm_head, softcap=softcap, chunk_size=128, ignore_index=-1, reduction=loss_reduction)

            total_loss = lm_loss + action_loss

            if return_embeddings:
                # We clone x to ensure it doesn't share memory with internal buffers (like wte output)
                # which can cause CUDAGraph overwrites during backward.
                return total_loss.clone(), x.clone()
            return total_loss
        else:
            # inference: return logits AND action_pred if robotics enabled
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap)

            action_pred = None
            if self.config.use_robotics and self.robotics_interface is not None:
                # Predict next surface vector from the LAST token's embedding
                last_hidden_state = x[:, -1, :]
                action_pred = self.robotics_interface.predict_action(last_hidden_state, target_action=None)

            return logits, action_pred

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, images=None, sensors=None, surface=None, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference with optional image context.
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)

        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim

        for i in range(max_tokens):
            # Pass contexts only on the first step
            current_images = images if i == 0 else None
            current_sensors = sensors if i == 0 else None
            current_surface = surface if i == 0 else None

            # Forward returns tuple in inference mode now
            logits, action_pred = self.forward(ids, images=current_images, sensors=current_sensors, surface=current_surface)
            logits = logits[:, -1, :] # (B, vocab_size)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)

            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()

            # Yield token and potentially action
            yield token, action_pred
