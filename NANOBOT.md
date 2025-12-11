# NanoBot

**NanoBot** is a multimodal extension of Nanochat designed for **robotics** and **embodied AI**. It transforms the language model into a "brain" that can see, feel, and act by integrating visual perception and robotic telemetry directly into the latent space.

This architecture enables:
1.  **Vision:** Processing images via a lightweight Vision Transformer (ViT).
2.  **Proprioception:** Ingesting raw sensor data (joint angles, acceleration) as tokens.
3.  **Latent Communication:** Reading "surface vectors" from external neural networks (e.g., an on-device Hebbian controller on an ESP32).
4.  **Action Generation:** Outputting continuous control signals via a **Diffusion Policy Head** (Denoising Diffusion Probabilistic Model).

## Architecture

The NanoBot model prepends multimodal tokens to the text context:

```
[Robotics Tokens] [Vision Tokens] [Text Tokens] -> GPT -> [Next Token Prediction]
                                                       -> [Action Prediction (Diffusion)]
```

### Components

*   **Vision Encoder (`nanochat/vision.py`):** A custom, dependency-free implementation of a Vision Transformer (similar to ViT/SigLIP) that patches images and projects them to the LLM's embedding dimension.
*   **Robotics Interface (`nanochat/robotics.py`):**
    *   **Sensor Projector:** Maps raw float vectors (telemetry) to embeddings.
    *   **Surface Projector:** Maps latent vectors from external networks to embeddings.
    *   **Diffusion Head:** A Conditional Denoiser (MLP) that generates action vectors by denoising random noise, conditioned on the LLM's hidden state.

## Inference Kernel (Rust)

For deployment on robots (where Python overhead is undesirable), NanoBot includes a dedicated **Rust Inference Kernel** (`nanochat-rs`).

*   **Location:** `nanochat-rs/`
*   **Tech Stack:** Rust + [HuggingFace Candle](https://github.com/huggingface/candle).
*   **Capabilities:**
    *   Loads `safetensors` or `GGUF` models exported from the Python training pipeline.
    *   Runs the full multimodal forward pass.
    *   Executes the Diffusion Sampling loop to generate actions.
    *   Optimized for CPU (and CUDA/ROCm where supported).

### Building and Running

```bash
cd nanochat-rs
cargo build --release
./target/release/nanochat-rs --model-path /path/to/exported_model
```

## Training NanoBot

You can train NanoBot using the standard `scripts/base_train.py` by enabling the robotics flags.

**Configuration:**
The `GPTConfig` has been extended with:
*   `use_vision`: Enable vision encoder.
*   `use_robotics`: Enable robotics interface.
*   `robotics_use_diffusion`: Enable Diffusion Head for actions (default: `False`, uses Regression).
*   `robotics_diffusion_steps`: Number of denoising steps (default: 100).

**Example Command:**
```bash
python -m scripts.base_train \
    --use_vision=True \
    --use_robotics=True \
    --robotics_use_diffusion=True \
    --robotics_sensor_dim=32 \
    --robotics_surface_dim=64
```

*Note: The current dataloader generates synthetic noise for vision and robotics inputs to facilitate architectural testing. For real-world use, you must update `nanochat/dataloader.py` to load your specific dataset (e.g., Parquet files with image bytes and sensor floats).*

## Exporting Models

To move from Python training to Rust inference, export your checkpoint:

```bash
python scripts/export_model.py \
    --checkpoint_dir base_checkpoints/d12 \
    --format safetensors
```

This script automatically embeds the necessary Vision and Robotics configuration metadata into the file header, allowing the Rust kernel to reconstruct the correct model topology dynamically.
