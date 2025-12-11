import torch
import os
from nanochat.gpt import GPT, GPTConfig
from scripts.export_model import export_to_gguf, export_to_safetensors

def test_export():
    # 1. Create Dummy Model
    config = GPTConfig(
        n_layer=2, n_head=2, n_kv_head=2, n_embd=64, vocab_size=100, sequence_len=32,
        use_vision=True, vision_width=32,
        use_robotics=True, robotics_sensor_dim=16, robotics_surface_dim=32
    )
    model = GPT(config)

    # 2. Get State Dict
    state_dict = model.state_dict()

    # 3. Create Config Dict (simulating what's in checkpoint metadata)
    model_config = {
        "n_layer": 2, "n_head": 2, "n_kv_head": 2, "n_embd": 64, "vocab_size": 100, "sequence_len": 32,
        "use_vision": True, "vision_image_size": 224, "vision_patch_size": 14,
        "vision_width": 32, "vision_layers": 12, "vision_heads": 12,
        "use_robotics": True, "robotics_sensor_dim": 16, "robotics_surface_dim": 32,
        "robotics_use_diffusion": False
    }

    # 4. Export
    os.makedirs("test_export", exist_ok=True)
    export_to_safetensors(state_dict, model_config, "test_export/model.safetensors")
    export_to_gguf(state_dict, model_config, "test_export/model.gguf")

if __name__ == "__main__":
    test_export()
