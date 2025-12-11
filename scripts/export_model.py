"""
Script to export nanochat checkpoints to other formats.
Supports:
- HuggingFace Safetensors
- GGUF (for llama.cpp)

Usage:
python scripts/export_model.py --checkpoint_dir=path/to/checkpoint --format=safetensors
python scripts/export_model.py --checkpoint_dir=path/to/checkpoint --format=gguf
"""

import os
import argparse
import json
import torch
import numpy as np
from safetensors.torch import save_file
from nanochat.checkpoint_manager import load_checkpoint
from nanochat.common import get_base_dir, autodetect_device_type

def export_to_safetensors(model_state, model_config, output_path):
    print(f"Exporting to safetensors: {output_path}")

    # Flatten config to string metadata
    metadata = {k: str(v) for k, v in model_config.items() if v is not None}

    tensors = {}
    for k, v in model_state.items():
        if isinstance(v, torch.Tensor):
            tensors[k] = v.contiguous()
        else:
            print(f"Warning: Skipping non-tensor key {k}")

    save_file(tensors, output_path, metadata=metadata)
    print("Done.")

def export_to_gguf(model_state, model_config, output_path):
    try:
        import gguf
    except ImportError:
        print("Error: gguf package not found. Please install it.")
        return

    print(f"Exporting to GGUF: {output_path}")

    # Initialize GGUF writer
    gguf_writer = gguf.GGUFWriter(output_path, "nanochat")

    # --- Architecture Metadata ---
    # Standard LLM keys
    gguf_writer.add_string("general.architecture", "nanochat")
    gguf_writer.add_uint32("nanochat.context_length", model_config.get("sequence_len", 1024))
    gguf_writer.add_uint32("nanochat.embedding_length", model_config.get("n_embd", 768))
    gguf_writer.add_uint32("nanochat.block_count", model_config.get("n_layer", 12))
    gguf_writer.add_uint32("nanochat.feed_forward_length", 4 * model_config.get("n_embd", 768))
    gguf_writer.add_uint32("nanochat.attention.head_count", model_config.get("n_head", 12))
    gguf_writer.add_uint32("nanochat.attention.head_count_kv", model_config.get("n_kv_head", 12))

    # Vision Keys
    use_vision = model_config.get("use_vision", False)
    gguf_writer.add_bool("nanochat.use_vision", use_vision)
    if use_vision:
        gguf_writer.add_uint32("nanochat.vision.image_size", model_config.get("vision_image_size", 224))
        gguf_writer.add_uint32("nanochat.vision.patch_size", model_config.get("vision_patch_size", 14))
        gguf_writer.add_uint32("nanochat.vision.width", model_config.get("vision_width", 768))
        gguf_writer.add_uint32("nanochat.vision.layers", model_config.get("vision_layers", 12))
        gguf_writer.add_uint32("nanochat.vision.heads", model_config.get("vision_heads", 12))

    # Robotics Keys
    use_robotics = model_config.get("use_robotics", False)
    gguf_writer.add_bool("nanochat.use_robotics", use_robotics)
    if use_robotics:
        gguf_writer.add_uint32("nanochat.robotics.sensor_dim", model_config.get("robotics_sensor_dim", 64))
        gguf_writer.add_uint32("nanochat.robotics.surface_dim", model_config.get("robotics_surface_dim", 128))
        gguf_writer.add_uint32("nanochat.robotics.sensor_tokens", model_config.get("robotics_sensor_tokens", 1))
        gguf_writer.add_uint32("nanochat.robotics.surface_tokens", model_config.get("robotics_surface_tokens", 4))

        # Diffusion params
        use_diffusion = model_config.get("robotics_use_diffusion", False)
        gguf_writer.add_bool("nanochat.robotics.use_diffusion", use_diffusion)
        if use_diffusion:
            gguf_writer.add_uint32("nanochat.robotics.diffusion.timesteps", model_config.get("robotics_diffusion_steps", 100))

    # --- Tensor Data ---
    for k, v in model_state.items():
        if isinstance(v, torch.Tensor):
            # GGUF expects numpy arrays (fp32 or fp16)
            data = v.detach().cpu().float().numpy()
            gguf_writer.add_tensor(k, data)

    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    print("Done.")

def main():
    parser = argparse.ArgumentParser(description="Export Nanochat Model")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to checkpoint directory (e.g., base_checkpoints/d12)")
    parser.add_argument("--step", type=int, default=-1, help="Step to load (-1 for latest)")
    parser.add_argument("--format", type=str, choices=["safetensors", "gguf"], required=True, help="Export format")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--device", type=str, default="cpu", help="Device to load model on")

    args = parser.parse_args()

    # Determine paths
    if not os.path.exists(args.checkpoint_dir):
        # try relative to base dir
        base_dir = get_base_dir()
        candidate = os.path.join(base_dir, args.checkpoint_dir)
        if os.path.exists(candidate):
            args.checkpoint_dir = candidate
        else:
             # try relative to repo root if user passed e.g. "base_checkpoints/d12"
             # but get_base_dir returns ~/.cache/nanochat
             pass

    print(f"Loading checkpoint from {args.checkpoint_dir}...")

    # Load checkpoint
    try:
        model_state, _, meta_data = load_checkpoint(args.checkpoint_dir, args.step, args.device, load_optimizer=False)
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    model_config = meta_data.get("model_config", {})

    # Determine output path
    if args.output is None:
        filename = f"model_{meta_data['step']}.{args.format}"
        args.output = os.path.join(args.checkpoint_dir, filename)

    if args.format == "safetensors":
        export_to_safetensors(model_state, model_config, args.output)
    elif args.format == "gguf":
        export_to_gguf(model_state, model_config, args.output)

if __name__ == "__main__":
    main()
