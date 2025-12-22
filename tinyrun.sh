#!/bin/bash
set -e

# This script is the "Best ChatGPT clone that a crappy single GPU can buy",
# It is designed to run in ~1 hour on a single 3080 GPU with 10GB of VRAM.

# Wrapper around run.sh and workflow.py
./run.sh scripts/workflow.py --job tiny "$@"
