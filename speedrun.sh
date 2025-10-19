#!/bin/bash
set -e

# This script is designed to be run inside the Docker container provided.

# Install dependencies
uv pip install -e .[dev]

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
# For newer AMD GPUs that are not yet officially supported by PyTorch ROCm builds,
# we can override the detected GPU architecture to a compatible one.
# For example, for a gfx1151 GPU, we can use gfx1100 (11.0.0).
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# wandb setup
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Report setup
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Build the rustbpe Tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# Download dataset and train tokenizer
python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 240 &
DATASET_DOWNLOAD_PID=$!
python -m scripts.tok_train --max_chars=2000000000
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)

# Download the eval_bundle
EVAL_BUNDLE_URL=https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
if [ ! -d "$NANOCHAT_BASE_DIR/eval_bundle" ]; then
    curl -L -o "$NANOCHAT_BASE_DIR/eval_bundle.zip" $EVAL_BUNDLE_URL
    unzip -q "$NANOCHAT_BASE_DIR/eval_bundle.zip" -d "$NANOCHAT_BASE_DIR"
    rm "$NANOCHAT_BASE_DIR/eval_bundle.zip"
fi

echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# Pretrain, evaluate, and assess the base model
python -m scripts.base_train -- --depth=20 --run=$WANDB_RUN --device_batch_size=16
python -m scripts.base_loss
python -m scripts.base_eval

# -----------------------------------------------------------------------------
# Midtraining
python -m scripts.mid_train --run=$WANDB_RUN --device_batch_size=16
python -m scripts.chat_eval -i mid

# -----------------------------------------------------------------------------
# Supervised Finetuning
python -m scripts.chat_sft --run=$WANDB_RUN
python -m scripts.chat_eval -i sft

# -----------------------------------------------------------------------------
# Generate final report
python -m nanochat.report generate