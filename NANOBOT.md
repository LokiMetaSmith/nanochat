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

## Continual Learning Loop

NanoBot supports **Continual Learning**, allowing the model to train on new data streams in real-time as they are collected from the robot.

### 1. Data Collection
Use `scripts/collect_telemetry.py` to bridge the robot (via Serial or Mock) to the training pipeline. This script buffers high-frequency telemetry and writes chunks to Parquet files.

```bash
# Start Data Collector (Mock Mode)
bash run.sh scripts/collect_telemetry.py --mock --data_dir data/live_telemetry

# Start Data Collector (Real Robot)
# bash run.sh scripts/collect_telemetry.py --port /dev/ttyUSB0 --baud 115200 --data_dir data/live_telemetry
```

### 2. Online Training
Run the training script with the `--continual=True` flag. The dataloader will monitor the `data_dir` and ingest new Parquet files as they appear, instead of terminating when the dataset is exhausted.

```bash
# Point to the live data directory
export NANOCHAT_BASE_DIR=data/live_telemetry

# Start Online Training
bash run.sh -m scripts.base_train \
    --use_robotics=True \
    --continual=True \
    --robotics_use_diffusion=True \
    --device_batch_size=4
```

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
bash run.sh -m scripts.base_train \
    --use_vision=True \
    --use_robotics=True \
    --robotics_use_diffusion=True \
    --robotics_sensor_dim=32 \
    --robotics_surface_dim=64
```

## Exporting Models

To move from Python training to Rust inference, export your checkpoint:

```bash
bash run.sh scripts/export_model.py \
    --checkpoint_dir base_checkpoints/d12 \
    --format safetensors
```

This script automatically embeds the necessary Vision and Robotics configuration metadata into the file header, allowing the Rust kernel to reconstruct the correct model topology dynamically.
