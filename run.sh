#!/bin/bash
set -e

# Usage: ./run.sh [python_script] [args...]
# Example: ./run.sh scripts/tune_system.py
# Example: ./run.sh -m scripts.tune_system
# Example: ./run.sh --deploy scripts/workflow.py --job tiny

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 [--deploy] [python_script] [args...]"
    exit 1
fi

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Flags
DEPLOY=0
if [ "$1" == "--deploy" ]; then
    DEPLOY=1
    shift
fi

# -----------------------------------------------------------------------------
# Environment Setup

# Detect hardware
# Logic shared between uv and deploy modes to select extras
if command -v nvidia-smi &> /dev/null; then
    EXTRAS="gpu"
elif command -v rocminfo &> /dev/null; then
    EXTRAS="amd"
else
    # Double check if rocminfo is present in venv but not in PATH yet (only relevant for uv mode really, but good to check)
    if [ -f ".venv/bin/rocminfo" ]; then
         EXTRAS="amd"
    else
         EXTRAS="cpu"
    fi
fi

if [ "$DEPLOY" -eq 1 ]; then
    echo "Deploy mode: Skipping uv/venv setup..."

    # 1. Install Project & Dependencies
    # Note: We assume python and pip are the system ones we want to use.
    echo "Installing project with extra: $EXTRAS"
    # Install in editable mode or standard? Using '.' usually implies editable if -e, but just . installs it.
    # The snippet used `pip install pyproject.toml` which implies installing the project defined there.
    # We use `.[amd]`, `.[gpu]`, etc.
    pip install ".[$EXTRAS]"

    # 2. Rust Setup
    if ! command -v cargo &> /dev/null; then
        echo "Cargo not found. Installing Rust..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source "$HOME/.cargo/env"
    elif [ -f "$HOME/.cargo/env" ]; then
         source "$HOME/.cargo/env"
    fi

    # 3. Build rustbpe
    echo "Building rustbpe..."
    # maturin should be installed by pip install above (it's in build-system requires, but maybe not in env)
    # The snippet had explicit `pip install maturin`.
    # Let's ensure maturin is there.
    pip install maturin

    # Build release
    # Snippet: pushd rustbpe; maturin develop --release; popd
    # We can do it from root with manifest-path
    maturin develop --release --manifest-path rustbpe/Cargo.toml

else
    # Normal UV Mode

    # install uv
    command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    [ -d ".venv" ] || uv venv
    source .venv/bin/activate

    # Force sync of amd dependencies to ensure rocminfo is available
    # Note: we must export the PATH to include the venv bin for rocminfo to be potentially found if it was installed there
    export PATH="$(pwd)/.venv/bin:$PATH"

    # Refine Hardware Detection with uv environment
    # Rerun detection logic in case rocminfo appeared in venv
    if command -v nvidia-smi &> /dev/null; then
        EXTRAS="gpu"
    elif command -v rocminfo &> /dev/null; then
        EXTRAS="amd"
    else
        EXTRAS="cpu"
    fi

    # Sync dependencies
    uv sync --extra $EXTRAS > /dev/null 2>&1 || uv sync --extra $EXTRAS
    if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    fi

    # Build Rust Tokenizer (if needed, but usually we run it here to be safe)
    uv run --no-sync --extra $EXTRAS maturin develop --release --manifest-path rustbpe/Cargo.toml > /dev/null 2>&1
fi

# -----------------------------------------------------------------------------
# Shared Hardware Config (AMD Specifics)

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
    if [ "$DEPLOY" -eq 0 ]; then
        if uv pip show triton > /dev/null 2>&1; then
            uv pip uninstall -q triton
            uv sync --extra amd > /dev/null 2>&1
        fi
        if ! uv pip show pytorch-triton-rocm > /dev/null 2>&1; then
            uv sync --extra amd > /dev/null 2>&1
        fi
    else
        # In deploy mode, we might need to manually handle this if pip installs standard triton?
        # For now, assume pip install .[amd] handles it correctly via pyproject.toml exclusions if configured,
        # but pyproject doesn't seem to exclude triton explicitly.
        # The snippet didn't handle triton, so we'll leave it simple for deploy unless requested.
        :
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

# -----------------------------------------------------------------------------
# WANDB Setup

if [ -z "$WANDB_RUN" ]; then
    export WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Execution

export PYTHONPATH="$PYTHONPATH:$(pwd)"
exec python "$@"
