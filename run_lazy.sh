#!/bin/bash

# Run Lazy Generation RLOO experiments with k=2,4,8 sequentially
# This script executes all lazy generation RLOO experiments

set -e

echo "Starting Lazy Generation RLOO experiment suite"
echo "This will run rloo_k=2, 4, 8 sequentially"

cd experiments

echo ""
echo "Phase 1/3: Running Lazy Generation RLOO with k=2"
./run_lazy_rloo2.sh || echo "Warning: Lazy k=2 failed, continuing to next experiment"

echo ""
echo "Phase 2/3: Running Lazy Generation RLOO with k=4"
./run_lazy_rloo4.sh || echo "Warning: Lazy k=4 failed, continuing to next experiment"

echo ""
echo "Phase 3/3: Running Lazy Generation RLOO with k=8"
./run_lazy_rloo8.sh || echo "Warning: Lazy k=8 failed, continuing to next experiment"

cd ..

echo ""
echo "Lazy Generation RLOO experiment suite completed"
echo "Results saved in results/data/results_lazy/"