#!/bin/bash

# RepeatSampler RLOO training with k=8
# This uses RepeatSampler approach similar to GRPO for memory efficiency

set -e

RLOO_K=8
BATCH_SIZE=8
METHOD="repeatsampler"

echo "Starting RepeatSampler RLOO training with k=$RLOO_K, batch_size=$BATCH_SIZE"

# Switch to RepeatSampler branch in TRL submodule
cd ../trl && git checkout feature/rloo-repeat-sampler >/dev/null 2>&1 || echo "Warning: Could not checkout repeat sampler branch"
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

echo "RepeatSampler RLOO k=$RLOO_K training completed!"