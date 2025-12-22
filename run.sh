#!/bin/bash
set -e

# Usage: ./run.sh [python_script] [args...]
# Example: ./run.sh scripts/tune_system.py
# Example: ./run.sh -m scripts.tune_system

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 [python_script] [args...]"
    exit 1
fi

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Environment Setup

# install uv
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
source .venv/bin/activate

# Force sync of amd dependencies to ensure rocminfo is available
# Note: we must export the PATH to include the venv bin for rocminfo to be potentially found if it was installed there
export PATH="$(pwd)/.venv/bin:$PATH"
uv sync --extra amd > /dev/null 2>&1 || uv sync --extra amd

# Detect hardware
if command -v nvidia-smi &> /dev/null; then
    EXTRAS="gpu"
elif command -v rocminfo &> /dev/null; then
    EXTRAS="amd"
else
    # Double check if rocminfo is present in venv but not in PATH yet (though we added it)
    if [ -f ".venv/bin/rocminfo" ]; then
         EXTRAS="amd"
    else
         EXTRAS="cpu"
    fi
fi

# Sync dependencies
# Using silent sync if possible, but fallback to normal if it fails or first run
uv sync --extra $EXTRAS > /dev/null 2>&1 || uv sync --extra $EXTRAS
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
fi

# AMD Specifics
if [ "$EXTRAS" == "amd" ]; then
    # Get ROCm path from the installed package
    ROCM_PATH_SCRIPT="import sysconfig, os; p = f\"{sysconfig.get_paths()['purelib']}/_rocm_sdk_core\"; print(p) if os.path.exists(p) else print('')"
    ROCM_PATH=$(python -c "$ROCM_PATH_SCRIPT")

    if [ -n "$ROCM_PATH" ]; then
        export ROCM_PATH
        export LD_LIBRARY_PATH="$ROCM_PATH/lib:$LD_LIBRARY_PATH"
        export PATH="$ROCM_PATH/bin:$PATH"
    fi

    # Fix Triton conflicts if triton is present
    if uv pip show triton > /dev/null 2>&1; then
        uv pip uninstall -q triton
        # Restore environment to ensure pytorch-triton-rocm is intact and correct version
        uv sync --extra amd > /dev/null 2>&1
    fi
    # Ensure pytorch-triton-rocm is installed
    if ! uv pip show pytorch-triton-rocm > /dev/null 2>&1; then
        uv sync --extra amd > /dev/null 2>&1
    fi

    # LLD Path
    ROCM_LLD_PATH=$(python -c "import sysconfig; import os; p = f\"{sysconfig.get_paths()['purelib']}/_rocm_sdk_core/lib/llvm/bin/ld.lld\"; print(p) if os.path.exists(p) else print('')")
    if [ -n "$ROCM_LLD_PATH" ]; then
        export TRITON_HIP_LLD_PATH=$ROCM_LLD_PATH
    fi

    # Strix Halo
    IS_STRIX_HALO=0
    if command -v rocminfo &> /dev/null; then
        if rocminfo | grep -q "gfx1151"; then
            IS_STRIX_HALO=1
        fi
    fi

    if [ "$IS_STRIX_HALO" -eq 1 ]; then
        [ -z "$HSA_OVERRIDE_GFX_VERSION" ] && export HSA_OVERRIDE_GFX_VERSION=11.5.1
        export HSA_ENABLE_SDMA=0
    fi
fi

# Build Rust Tokenizer
uv run --no-sync --extra $EXTRAS maturin develop --release --manifest-path rustbpe/Cargo.toml > /dev/null 2>&1

# Run the command
export PYTHONPATH="$PYTHONPATH:$(pwd)"
exec python "$@"
