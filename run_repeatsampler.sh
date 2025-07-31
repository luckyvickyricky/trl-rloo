#!/bin/bash

# Run RepeatSampler RLOO experiments with k=2,4,8 sequentially
# This script executes all RepeatSampler RLOO experiments

set -e

echo "Starting RepeatSampler RLOO experiment suite"
echo "This will run rloo_k=2, 4, 8 sequentially"

cd experiments

echo ""
echo "Phase 1/3: Running RepeatSampler RLOO with k=2"
./run_repeatsampler_rloo2.sh || echo "Warning: RepeatSampler k=2 failed, continuing to next experiment"

echo ""
echo "Phase 2/3: Running RepeatSampler RLOO with k=4"
./run_repeatsampler_rloo4.sh || echo "Warning: RepeatSampler k=4 failed, continuing to next experiment"

echo ""
echo "Phase 3/3: Running RepeatSampler RLOO with k=8"
./run_repeatsampler_rloo8.sh || echo "Warning: RepeatSampler k=8 failed, continuing to next experiment"

cd ..

echo ""
echo "RepeatSampler RLOO experiment suite completed"
echo "Results saved in results/data/results_repeatsampler/"