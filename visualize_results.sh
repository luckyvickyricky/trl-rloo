#!/bin/bash

# Generate visualization from experimental results
# This script creates all graphs and analysis charts

set -e

echo "Starting result visualization..."

cd visualization

echo "Step 1: Combining GPU memory data..."
python combine_memory_data.py

echo "Step 2: Generating visualization charts..."
python create_charts.py
