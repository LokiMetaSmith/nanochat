#!/bin/bash
set -e

# This script is the "Best ChatGPT clone that $100 can buy",
# It is designed to run in ~4 hours on 8XH100 node at $3/GPU/hour.

# 1) Example launch (simplest):
# bash speedrun.sh
# 2) Example launch in a screen session (because the run takes ~4 hours):
# screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=speedrun screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"

mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH")
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# --- PyTorch Installation ---
echo "🔍 Detecting hardware..."

FORCE_AMD=false
if [[ "$1" == "--force-amd" ]]; then
    FORCE_AMD=true
fi

DEVICE_FLAG=""
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected. Installing PyTorch for CUDA."
    uv pip install torch>=2.8.0 --extra-index-url https://download.pytorch.org/whl/cu128
    DEVICE_FLAG="--device=cuda"
elif [[ "$FORCE_AMD" = true ]] || command -v rocm-smi &> /dev/null; then
    echo "✅ AMD GPU detected. Installing ROCm 7.9.0 and PyTorch."
    uv pip install --index-url https://repo.amd.com/rocm/whl/gfx1151/ "rocm[libraries,devel]"
    uv pip install --index-url https://repo.amd.com/rocm/whl/gfx1151/ torch torchvision
    DEVICE_FLAG="--device=cuda"
    export ROCM_PATH=$(python -m rocm_sdk path --root)
    export LD_LIBRARY_PATH=$ROCM_PATH/lib
    export PATH=$PATH:$ROCM_PATH/bin
else
    echo "🤷 No GPU detected. Installing CPU-only PyTorch."
    uv pip install torch>=2.8.0
    DEVICE_FLAG="--device=cpu"
fi

echo "✅ PyTorch installation complete."

# --- Verification Step ---
echo "🔎 Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'ROCm available: {torch.version.hip is not None}')"
if python -c "import torch; exit(0) if torch.version.hip is not None else exit(1)"; then
    echo "✅ PyTorch ROCm installation verified."
else
    echo "❌ PyTorch ROCm installation failed."
    exit 1
fi

# --- Project Installation ---
echo "🚀 Installing nanochat project dependencies..."
uv pip install -e .[dev]

echo "✨ Setup complete!"

# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d26 bash speedrun.sh`
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Download the first ~200M characters of pretraining dataset
# look at dev/repackage_data_reference.py for details on how this data was prepared
# each data shard is ~250M chars, so we'll download 1 shard
# each shard is ~100MB of text (compressed)
python -m nanochat.dataset -n 1
# Immediately also kick off downloading more shards in the background while tokenizer trains
# See comment below for why 240 is the right number here
python -m nanochat.dataset -n 240 &
DATASET_DOWNLOAD_PID=$!
# train the tokenizer with vocab size 2**16 = 65536 on ~200M characters of data
python -m scripts.tok_train --max_chars=200000000
# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)

# Download the eval_bundle from s3 to evaluate CORE metric during training (~162MB)
EVAL_BUNDLE_URL=https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
if [ ! -d "$NANOCHAT_BASE_DIR/eval_bundle" ]; then
    curl -L -o eval_bundle.zip $EVAL_BUNDLE_URL
    unzip -q eval_bundle.zip
    rm eval_bundle.zip
    mv eval_bundle $NANOCHAT_BASE_DIR
fi

# The d20 model is 561M parameters.
# Chinchilla says #tokens = 20X #params, so we need 561e6 * 20 = 11.2B tokens.
# Assume our tokenizer is 4.8 chars/token, this is 11.2B * 4.8 ~= 54B chars.
# At 250M chars/shard, this is 54B / 250M ~= 216 shards needed for pretraining.
# Round up to 240 for safety. At ~100MB/shard, this downloads ~24GB of data to disk.
# (The total number of shards available in the entire dataset is 1822.)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# pretrain the d20 model
python -m scripts.base_train --depth=20 --run=$WANDB_RUN $DEVICE_FLAG
# evaluate the model on a larger chunk of train/val data and draw some samples
python -m scripts.base_loss
# evaluate the model on CORE tasks
python -m scripts.base_eval

# -----------------------------------------------------------------------------
# Midtraining (teach the model conversation special tokens, tool use, multiple choice)

# run midtraining and eval the model
python -m scripts.mid_train --run=$WANDB_RUN $DEVICE_FLAG
python -m scripts.chat_eval -i mid

# -----------------------------------------------------------------------------
# Supervised Finetuning (domain adaptation to each sequence all by itself per row)

# train sft and re-eval right away (should see a small bump)
python -m scripts.chat_sft --run=$WANDB_RUN $DEVICE_FLAG
python -m scripts.chat_eval -i sft

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"

# even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# Reinforcement Learning. Optional, and currently only on GSM8K
# (optional)

# run reinforcement learning
python -m scripts.chat_rl --run=$WANDB_RUN $DEVICE_FLAG
# eval the RL model only on GSM8K
# python -m scripts.chat_eval -- -i rl -a GSM8K

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
python -m nanochat.report generate
