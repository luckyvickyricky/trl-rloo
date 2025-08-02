#!/bin/bash

# Run Basic RLOO experiments with k=2,4,8 sequentially
# This script executes all basic RLOO experiments without optimization

set -e

echo "Starting Basic RLOO experiment suite"
echo "This will run rloo_k=2, 4, 8 sequentially"

cd experiments

echo ""
echo "Phase 1/3: Running Basic RLOO with k=2"
./run_basic_rloo2.sh || echo "Warning: Basic k=2 failed, continuing to next experiment"

echo ""
echo "Phase 2/3: Running Basic RLOO with k=4"
./run_basic_rloo4.sh || echo "Warning: Basic k=4 failed, continuing to next experiment"

echo ""
echo "Phase 3/3: Running Basic RLOO with k=8"
./run_basic_rloo8.sh || echo "Warning: Basic k=8 failed, continuing to next experiment"

cd ..

echo ""
echo "Basic RLOO experiment suite completed"
echo "Results saved in results/data/results_basic/"