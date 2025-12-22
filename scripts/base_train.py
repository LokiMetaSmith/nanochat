"""
Train model. Run as:

python base_train.py

or distributed as:

torchrun --nproc_per_node=8 base_train.py

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max_seq_len=512 --device_batch_size=1 --eval_tokens=512 --core_metric_every=-1 --total_batch_size=512 --num_iterations=20
"""

import os
import time
import shutil
import subprocess
from contextlib import nullcontext

import wandb
import torch
if torch.cuda.is_available() or (hasattr(torch.version, 'hip') and torch.version.hip):
    # Set precision high for TensorFloat32 if available (applies to both CUDA and ROCm)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    # Allow overriding PYTORCH_CUDA_ALLOC_CONF/PYTORCH_HIP_ALLOC_CONF for tuning
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Also set the HIP-specific env var if on ROCm, as suggested by OOM errors
    if hasattr(torch.version, 'hip') and torch.version.hip:
        if "PYTORCH_HIP_ALLOC_CONF" not in os.environ:
             os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"

        # Check for Strix Halo (gfx1151) and enable experimental features
        try:
            if shutil.which('rocminfo'):
                result = subprocess.run(['rocminfo'], capture_output=True, text=True)
                if 'gfx1151' in result.stdout and "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL" not in os.environ:
                    print("AMD Strix Halo (gfx1151) detected. Enabling experimental Triton support.")
                    os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
        except Exception:
            pass

        # Ensure Triton can find lld for torch.compile
        if "TRITON_HIP_LLD_PATH" not in os.environ:
             possible_paths = ["/opt/rocm/llvm/bin/ld.lld", "/usr/bin/ld.lld"]
             for p in possible_paths:
                 if os.path.exists(p):
                     os.environ["TRITON_HIP_LLD_PATH"] = p
                     break

from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader, tokenizing_distributed_data_loader_with_state
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.scheduler import get_lr_multiplier, get_muon_momentum
from scripts.base_eval import evaluate_model
from nanochat.infovore import Infovore
from nanochat.robotics import RoboticsConfig
print_banner()

# -----------------------------------------------------------------------------
# User settings
run = "dummy" # wandb run name default ("dummy" is special - we won't log to wandb)
# Runtime
device_type = "" # cuda|cpu|mps (empty => autodetect good device type default, in order: CUDA > MPS > CPU)
# Model architecture
depth = 20 # the depth of the Transformer model to train, rest of the kwargs are derived
max_seq_len = 2048 # max context length
# Vision Architecture (Multimodal)
use_vision = False # Enable Vision Encoder
vision_image_size = 224
vision_patch_size = 14
vision_width = 768
vision_layers = 12
vision_heads = 12
vision_mlp_ratio = 4.0
# Robotics Architecture (Multimodal)
use_robotics = False # Enable Robotics (Sensor + Latent)
robotics_sensor_dim = 64
robotics_surface_dim = 128
robotics_sensor_tokens = 1
robotics_surface_tokens = 4
robotics_action_loss_weight = 1.0 # Loss weight for latent action prediction
robotics_use_diffusion = False # Enable Diffusion Policy Head
robotics_diffusion_steps = 100
# LoRA (Low-Rank Adaptation)
use_lora = False # Enable LoRA
lora_rank = 8
lora_alpha = 16
lora_dropout = 0.05
lora_targets = ["q_proj", "v_proj"]
# Data Loading
continual = False # Enable continual learning (poll for new data)
max_chars = 0 # Ignored: Maximum characters for tokenizer training (shared config)
split_tokens = 0 # Ignored: Tokens per split for loss evaluation (shared config)

# Training horizon. Only one of these 3 will be used, in this order of precedence.
num_iterations = -1 # explicit number of steps of the optimization (-1 = disable)
target_flops = -1.0 # calculate num_iterations to reach target_flops. Useful for scaling laws experiments (-1 = disable)
target_param_data_ratio = 20 # calculate num_iterations to maintain fixed data:param ratio (Chinchilla=20) (-1 = disable)
# Optimization
device_batch_size = 16 # per-device batch size (reduced from 32 to avoid OOM on some GPUs)
total_batch_size = 524288 # total desired batch size, in #tokens
embedding_lr = 0.2 # learning rate for the embedding parameters (Adam)
unembedding_lr = 0.004 # learning rate for the unembedding parameters (Adam)
weight_decay = 0.0 # weight decay for the embedding/unembedding parameters (Adam)
matrix_lr = 0.02 # learning rate for the matrix parameters (Muon)
grad_clip = 1.0 # gradient clipping value (0.0 = disabled)
warmup_ratio = 0.0 # ratio of iterations for LR warmup
adam_warmup_ratio = 0.0 # ratio of iterations for AdamW LR warmup (0.0 = disabled). Independent of Muon warmup.
warmdown_ratio = 0.2 # ratio of iterations for LR warmdown
final_lr_frac = 0.0 # final LR is this fraction of the initial LR
resume_from_step = -1 # resume training from this step of the optimization (-1 = disable)
# Evaluation
eval_every = 250 # every how many steps to evaluate the model for val bpb
eval_tokens = 20*524288 # number of tokens to evaluate val loss on
core_metric_every = 2000 # every how many steps to evaluate the core metric (-1 = disable)
core_metric_max_per_task = 500 # examples per task in estimating the core metric
sample_every = 2000 # every how many steps to sample from the model
save_every = -1 # every how many steps to save model checkpoints (-1 = disable, and save only at the end of the run)
# Output
model_tag = "" # optionally override the model tag for the output checkpoint directory name
compile = True # whether to compile the model. On AMD ROCm this might be unstable.
compile_mode = "reduce-overhead" # torch.compile mode (e.g., "default", "reduce-overhead", "max-autotune")
compile_dynamic = False # whether to use dynamic shapes in torch.compile (dynamic=True/False)
# Optimizer backends
matrix_optimizer_backend = "muon" # "muon" or "nested_momentum"
general_optimizer_backend = "adamw" # "adamw" (ZeRO-2), "dist_adamw" (legacy custom), or "nested_momentum"
use_8bit_optimizer = False # use 8-bit optimizer (bitsandbytes) for general parameters
layer_lr_decay = 1.0 # Layer-wise Learning Rate Decay (1.0 = disabled)
nested_betas = (0.9, 0.99) # For nested_momentum
nested_level_weights = (0.5, 0.5) # For nested_momentum
# Infovore (Curriculum Learning)
use_infovore = False # Enable Novelty-Relation Quotient curriculum
infovore_beta = 0.99 # Momentum for memory manifold

# Auto-detect Strix Halo to enable specific optimizations
try:
    if shutil.which('rocminfo'):
        _res = subprocess.run(['rocminfo'], capture_output=True, text=True)
        if 'gfx1151' in _res.stdout:
            print("AMD Strix Halo (gfx1151) detected. Optimizing for APU.")
            # We no longer disable compile=True.
            # Instead we want to ensure experimental features and 8-bit optimizer if appropriate
            # (though 8-bit optimizer is opt-in via flag)
except Exception:
    pass

# now allow CLI to override the settings via the configurator lol
config_keys = {k: v for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None), tuple, list))}
from nanochat.configurator import get_config
config_updates = get_config(config_keys)
globals().update(config_updates)
user_config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

# Fallback for CPU compilation mode
if device.type == 'cpu' and compile_mode == 'reduce-overhead':
    print0("WARNING: 'reduce-overhead' compile mode is not compatible with CPU. Switching to 'default'.")
    compile_mode = 'default'

master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# wandb logging init
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=run, config=user_config)

# Tokenizer will be useful for evaluation, also we need the vocab size
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# Model kwargs are derived from the desired depth of the model
num_layers = depth
model_dim = depth * 64 # aspect ratio 64 (usually this is varied from 64 -> 128 as model size increases)
num_heads = max(1, (model_dim + 127) // 128) # head dim 128 (the division here is ceil div)
num_kv_heads = num_heads # default is 1:1 GQA (Group Query Attention) ratio (i.e. GQA is disabled)
print0(f"num_layers: {num_layers}")
print0(f"model_dim: {model_dim}")
print0(f"num_heads: {num_heads}")
print0(f"num_kv_heads: {num_kv_heads}")

# Optimizer / data / training length related hyperparameters
# figure out the needed gradient accumulation to reach the desired total batch size
tokens_per_fwdbwd = device_batch_size * max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# -----------------------------------------------------------------------------
# Initialize the Model

# Calculate total sequence length (Vision + Robotics + Text)
num_vision_patches = 0
if use_vision:
    num_vision_patches = (vision_image_size // vision_patch_size) ** 2
elif user_config.get("use_visual_tokens", False):
    # UniTok downsample factor is 16 (2*2*2*2)
    # Grid size = (224 // 16) * (224 // 16) = 14 * 14 = 196
    # Or should we use vision_image_size from config? Yes.
    num_vision_patches = (vision_image_size // 16) ** 2

num_robotics_tokens = 0
if use_robotics:
    if robotics_sensor_dim > 0:
        num_robotics_tokens += robotics_sensor_tokens
    if robotics_surface_dim > 0:
        num_robotics_tokens += robotics_surface_tokens

total_seq_len = max_seq_len + num_vision_patches + num_robotics_tokens
print0(f"Max text len: {max_seq_len}, Vision: {num_vision_patches}, Robot: {num_robotics_tokens} -> Total: {total_seq_len}")

# Create a new model with random weights
model_config_kwargs = dict(
    sequence_len=total_seq_len, # Increase capacity for vision/robot tokens
    vocab_size=vocab_size,
    n_layer=num_layers,
    n_head=num_heads,
    n_kv_head=num_kv_heads,
    n_embd=model_dim,
    use_vision=use_vision,
    vision_image_size=vision_image_size,
    vision_patch_size=vision_patch_size,
    vision_width=vision_width,
    vision_layers=vision_layers,
    vision_heads=vision_heads,
    vision_mlp_ratio=vision_mlp_ratio,
    use_robotics=use_robotics,
    robotics_sensor_dim=robotics_sensor_dim,
    robotics_surface_dim=robotics_surface_dim,
    robotics_sensor_tokens=robotics_sensor_tokens,
    robotics_surface_tokens=robotics_surface_tokens,
    robotics_action_loss_weight=robotics_action_loss_weight,
    use_lora=use_lora,
    lora_rank=lora_rank,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    lora_targets=lora_targets
)
with torch.device("meta"):
    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config)
model.to_empty(device=device)
model.init_weights()

# If we are resuming, overwrite the model parameters with those of the checkpoint
base_dir = get_base_dir()
output_dirname = model_tag if model_tag else f"d{depth}" # e.g. d12
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = resume_from_step != -1
if resuming:
    print0(f"Resuming optimization from step {resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data # free up this memory after the copy

orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
# If on ROCm (Strix Halo specifically), default compile to False if not explicitly set?
# Actually, let's just respect the flag. But if we are on ROCm, we might want to warn or default to False.
# For now, let's just make it conditional.
if compile:
    print0(f"compiling the model... (dynamic={compile_dynamic})...(mode={compile_mode})")
    model = torch.compile(model, dynamic=compile_dynamic, mode=compile_mode) # the inputs to model will never change shape so dynamic=False is safe
num_params = sum(p.numel() for p in model.parameters())
print0(f"Number of parameters: {num_params:,}")
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# Calculate number of iterations. Either it is given, or from target flops, or from target data:param ratio (in that order)
assert num_iterations > 0 or target_param_data_ratio > 0 or target_flops > 0
if num_iterations > 0:
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif target_flops > 0:
    # calculate the number of iterations from the target flops
    num_iterations = round(target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif target_param_data_ratio > 0:
    # calculate the number of iterations from the target param data ratio
    target_tokens = target_param_data_ratio * num_params
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Params ratio: {total_batch_size * num_iterations / num_params:.2f}") # Chinchilla is ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (Muon for Linear layers, AdamW for embedding and lm_head)
optimizers = model.setup_optimizers(
    unembedding_lr=unembedding_lr, embedding_lr=embedding_lr, matrix_lr=matrix_lr, weight_decay=weight_decay,
    matrix_optimizer_backend=matrix_optimizer_backend, general_optimizer_backend=general_optimizer_backend,
    nested_betas=nested_betas, nested_level_weights=nested_level_weights,
    use_8bit_optimizer=use_8bit_optimizer, layer_lr_decay=layer_lr_decay
)
# Note: optimizers[0] is general (AdamW/Nested), optimizers[1] is matrix (Muon/Nested)
general_optimizer, matrix_optimizer = optimizers

if resuming:
    for opt, dat in zip(optimizers, optimizer_data):
        try:
            opt.load_state_dict(dat)
        except Exception as e:
            print0(f"WARNING: Failed to load optimizer state: {e}. Starting with fresh optimizer state.")

    del optimizer_data # free up the memory

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val
tokens_dir = os.path.join(base_dir, "tokenized_data")
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]

# Vision Config for Dataloader
vision_loader_config = None
if use_vision or user_config.get("use_visual_tokens", False):
    vision_loader_config = {
        'image_size': vision_image_size,
        'channels': 3 # assume RGB
    }

# Robotics Config for Dataloader
robotics_loader_config = None
if use_robotics:
    robotics_loader_config = RoboticsConfig(
        sensor_dim=robotics_sensor_dim,
        surface_dim=robotics_surface_dim
    )

train_loader = tokenizing_distributed_data_loader_with_state(
    device_batch_size,
    max_seq_len, # Data loader still requests text of this length
    split="train",
    device=device,
    resume_state_dict=dataloader_resume_state_dict,
    vision_config=vision_loader_config,
    robotics_config=robotics_loader_config,
    continual=continual
)
# For val loader we don't support vision yet for bpb evaluation (standard text metric)
# TODO: Update evaluate_bpb to handle vision if desired, for now we disable vision in val
build_val_loader = lambda: tokenizing_distributed_data_loader(device_batch_size, max_seq_len, split="val", device=device)

# Kick off load of the very first batch
batch_data = next(train_loader)

# Unpack based on config
images = None
sensors = None
surface = None
action_targets = None # Only for training

if use_robotics:
    # Loader returns: inputs, targets, images (opt), sensors, surface, state
    # But wait, dataloader returns different tuples based on config!
    # Checking `dataloader.py`:
    # if robotics_config: yield inputs, targets, final_images, final_sensors, final_surface, state_dict
    x, y, images, sensors, surface, dataloader_state_dict = batch_data

    # Use surface as action target (predict next state)?
    # Or shift surface?
    # Current simplistic approach: predict current surface (reconstruction) or random noise?
    # In reality we want next step.
    # The dataloader needs to yield next step.
    # For now, let's use the current surface as target (autoencoder style)
    # or implement next-step in dataloader later.
    action_targets = surface

elif use_vision or user_config.get("use_visual_tokens", False):
    x, y, images, dataloader_state_dict = batch_data
else:
    x, y, dataloader_state_dict = batch_data
    images = None

# Initialize Infovore Agent if enabled
infovore_agent = None
if use_infovore:
    infovore_agent = Infovore(d_model=model_dim, device=device, beta=infovore_beta)
    print0("Initialized Infovore Agent for NRQ-weighted learning.")

# -----------------------------------------------------------------------------
# Loop state (variables updated by the training loop)

if not resuming:
    step = 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0 # EMA of training loss
    total_training_time = 0 # total wall-clock time of training
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# -----------------------------------------------------------------------------
# Training loop
while True:
    last_step = step == num_iterations # loop runs num_iterations+1 times so that we can eval/save at the end
    flops_so_far = num_flops_per_token * total_batch_size * step

    # once in a while: evaluate the val bpb (all ranks participate)
    if last_step or step % eval_every == 0:
        model.eval()
        val_loader = build_val_loader()
        eval_steps = eval_tokens // (device_batch_size * max_seq_len * ddp_world_size)
        with autocast_ctx:
            # Note: evaluate_bpb currently does not support vision, but since our val_loader is text-only, this is fine.
            # The model handles images=None automatically.
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    results = {}
    if core_metric_every > 0 and (last_step or (step > 0 and step % core_metric_every == 0)):
        model.eval()
        with autocast_ctx:
            # evaluate_model is text-only for now
            results = evaluate_model(orig_model, tokenizer, device, max_per_task=core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if master_process and (last_step or (step > 0 and step % sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with autocast_ctx:
                # generate_batch does not currently support images, which is fine for these text prompts
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
    if last_step or (step > 0 and step != resume_from_step and save_every > 0 and step % save_every == 0):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(), # model parameters
            [opt.state_dict() for opt in optimizers], # optimizer states
            { # metadata saved as json
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "model_config": model_config_kwargs,
                "user_config": user_config, # inputs to the training script
                "device_batch_size": device_batch_size,
                "max_seq_len": max_seq_len,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": { # all loop state (other than step) so that we can resume training
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )

    # termination conditions (TODO: possibly also add loss explosions etc.)
    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            # Prepare targets (pad for vision + robotics if needed)
            # Total padding = vision_patches + robotics_tokens
            padding_len = num_vision_patches + num_robotics_tokens

            # Mark the beginning of a step for CUDAGraphs (required for reduce-overhead mode)
            # This must be called before the model forward pass
            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()

            if padding_len > 0:
                # y shape is (B, max_seq_len)
                # We need to prepend padding
                vision_padding = torch.full((y.shape[0], padding_len), -1, dtype=y.dtype, device=y.device)
                y_padded = torch.cat([vision_padding, y], dim=1)
                final_targets = y_padded
            else:
                final_targets = y

            if use_infovore:
                loss, metrics = infovore_agent.compute_nrq_loss(
                    model, x, final_targets,
                    images=images, sensors=sensors, surface=surface, action_targets=action_targets
                )
                # We could log the metrics here, but we are inside a micro-step.
                # Just averaging them for logging might be tricky without plumbing.
                # For now, let's just use the weighted loss.
            else:
                loss = model(x, images=images, sensors=sensors, surface=surface, targets=final_targets, action_targets=action_targets)

        train_loss = loss.detach() # for logging
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        loss.backward()

        # Prefetch next batch
        batch_data = next(train_loader)
        if use_robotics:
            x, y, images, sensors, surface, dataloader_state_dict = batch_data
            action_targets = surface
        elif use_vision or user_config.get("use_visual_tokens", False):
            x, y, images, dataloader_state_dict = batch_data
        else:
            x, y, dataloader_state_dict = batch_data
            images = None

    # gradient clipping
    grad_clip_enabled = grad_clip > 0.0
    if grad_clip_enabled:
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(orig_model.parameters(), grad_clip)
        grad_norm = grad_norm_tensor.item() # GPU tensor -> CPU float (note: cpu-gpu sync point)
    # step the optimizers
    # AdamW might have a different warmup schedule than Muon
    adam_lrm = get_lr_multiplier(step, adam_warmup_ratio, num_iterations, warmdown_ratio, final_lr_frac)
    muon_lrm = get_lr_multiplier(step, warmup_ratio, num_iterations, warmdown_ratio, final_lr_frac)

    # Update LRs
    for group in general_optimizer.param_groups:
        group["lr"] = group["initial_lr"] * adam_lrm
    for group in matrix_optimizer.param_groups:
        group["lr"] = group["initial_lr"] * muon_lrm

    # Update Momentum (if needed, only Muon uses dynamic momentum in this repo)
    if matrix_optimizer_backend == "muon":
        muon_momentum = get_muon_momentum(step)
        for group in matrix_optimizer.param_groups:
            group["momentum"] = muon_momentum

    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging
    ema_beta = 0.9 # EMA decay factor for some smoothing just for nicer logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item() # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    promised_flops_per_sec_h100 = 989e12 * ddp_world_size # bfloat16 H100 SXM and without 2:4 sparsity
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100 # in %
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    print_grad_norm = f" grad norm: {grad_norm:.4f} |" if grad_clip_enabled else ""
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} |{print_grad_norm} adam_lrm: {adam_lrm:.2f} | muon_lrm: {muon_lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | total time: {total_training_time/60:.2f}m")
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/adam_lrm": adam_lrm,
            "train/muon_lrm": muon_lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
        }
        if grad_clip_enabled:
            log_data["train/grad_norm"] = grad_norm
        wandb_run.log(log_data)

    # state update
    step += 1

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Base model training", data=[
    user_config, # CLI args
    { # stats about the training setup
        "Number of parameters": num_params,
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Calculated number of iterations": num_iterations,
        "Number of training tokens": total_tokens,
        "Tokens : Params ratio": total_batch_size * num_iterations / num_params,
        "DDP world size": ddp_world_size,
        "warmup_ratio": warmup_ratio,
        "warmdown_ratio": warmdown_ratio,
        "final_lr_frac": final_lr_frac,
    },
    { # stats about training outcomes
        "Minimum validation bpb": min_val_bpb,
        "Final validation bpb": val_bpb,
        "CORE metric estimate": results.get("core_metric", None),
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
    }
])

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()
