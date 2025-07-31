#!/bin/bash

# Basic RLOO training with k=4
# This uses the original RLOO implementation without optimization

set -e

RLOO_K=4
BATCH_SIZE=4
METHOD="basic"

echo "Starting Basic RLOO training with k=$RLOO_K, batch_size=$BATCH_SIZE"

# Ensure we're on main branch for basic implementation in TRL submodule
cd ../trl && git checkout main >/dev/null 2>&1 || true
cd ../experiments

# Setup output directory
OUTPUT_DIR="../results/data/results_${METHOD}/tiny-Qwen3ForCausalLM${RLOO_K}/logs"
mkdir -p "$OUTPUT_DIR"

# Start memory monitoring
python memory_monitor.py "$OUTPUT_DIR" 0.1 &
MONITOR_PID=$!

# Cleanup function
cleanup() {
    kill $MONITOR_PID 2>/dev/null || true
    wait $MONITOR_PID 2>/dev/null || true
}

trap cleanup EXIT

# Start training
echo "Starting training process..."
python train.py --rloo_k "$RLOO_K" --batch_size "$BATCH_SIZE"

echo "Basic RLOO k=$RLOO_K training completed!"