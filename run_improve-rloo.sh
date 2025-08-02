#!/bin/bash

# Run String-Level RLOO experiments with k=2,4,8 sequentially
# This script executes all String-Level RLOO experiments

set -e

echo "Starting String-Level RLOO experiment suite"
echo "This will run rloo_k=2, 4, 8 sequentially"

cd experiments

echo ""
echo "Phase 1/3: Running String-Level RLOO with k=2"
./run_string-level_rloo2.sh || echo "Warning: String-Level k=2 failed, continuing to next experiment"

echo ""
echo "Phase 2/3: Running String-Level RLOO with k=4"
./run_string-level_rloo4.sh || echo "Warning: String-Level k=4 failed, continuing to next experiment"

echo ""
echo "Phase 3/3: Running String-Level RLOO with k=8"
./run_string-level_rloo8.sh || echo "Warning: String-Level k=8 failed, continuing to next experiment"

cd ..

echo ""
echo "String-Level RLOO experiment suite completed"
echo "Results saved in results/data/results_string-level/"