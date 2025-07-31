#!/bin/bash

# Combine all experimental results into unified datasets
# This script aggregates GPU memory data from all experiments

set -e

echo "Combining experimental results..."

cd visualization

# Create combined CSV script
python combine_memory_data.py

echo "Results combined successfully!"
echo "Combined data saved in visualization/combined_gpu_memory.csv"