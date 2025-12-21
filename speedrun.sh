#!/bin/bash
set -e

# This script is the "Best ChatGPT clone that $100 can buy",
# It is designed to run in ~4 hours on 8XH100 node at $3/GPU/hour.

# Wrapper around run.sh and workflow.py
./run.sh scripts/workflow.py --job speed "$@"
