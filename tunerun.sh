#!/bin/bash
set -e

# This script is designed to auto-tune the system performance for nanochat.
# It uses scripts/tune_system.py to find the best configuration (batch size, compilation flags, etc.)
# and reports the results.

# 1) Example launch:
# bash tunerun.sh
# 2) Launch with specific profile:
# bash tunerun.sh --profile tiny

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# Parse arguments
PROFILE="configs/medium.json" # Default profile
PY_ARGS=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --profile)
            PROFILE="configs/$2.json"
            shift
            ;;
        *)
            PY_ARGS="$PY_ARGS $1"
            ;;
    esac
    shift
done

if [ ! -f "$PROFILE" ]; then
    echo "Error: Configuration file $PROFILE not found!"
    exit 1
fi

echo "Using profile: $PROFILE"

# -----------------------------------------------------------------------------
# Tokenizer Setup (Needed for base_train to run)

# We use run.sh to handle environment, but we need to ensure the tokenizer exists first.
# run.sh automatically handles Rust builds.
# We just need to check for the pkl file and run generation if missing.

if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
    echo "Tokenizer not found. Training tokenizer on small data subset..."
    ./run.sh -m nanochat.dataset -n 1
    ./run.sh -m scripts.tok_train --max_chars=10000000
fi

# -----------------------------------------------------------------------------
# Run the System Tuner

# Delegate to run.sh for correct environment (including AMD specifics)
./run.sh -m scripts.tune_system --config "$PROFILE" $PY_ARGS
