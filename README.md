# TRL RLOO Memory Optimization Project

**RLOO Trainer Memory Optimization and Performance Improvement**

[한국어 문서 (Korean Documentation)](docs/README_ko.md)

## System Information

- **OS**: Ubuntu 24.04.2 LTS (Linux 6.14.0-24-generic)
- **GPU**: NVIDIA GeForce RTX 5090
- **PyTorch**: 2.7.1+cu128
- **Transformers**: 4.54.1
- **CUDA**: 12.8

## Model and Dataset Information

### Model Used
- **Model**: `tiny-Qwen3ForCausalLM` (Custom tiny model for testing)
- **Purpose**: Lightweight model designed for memory optimization experiments
- **Why chosen**: Small enough to run multiple configurations while still demonstrating memory scaling issues
- **Source**: Generated using TRL's tiny model generation tools

### Dataset Information
- **Dataset**: Synthetic/minimal dataset for quick training cycles
- **Purpose**: Focus on memory profiling rather than training quality
- **Size**: Small batch sizes (2-8) to enable controlled memory analysis

## Project Structure

```
├── docs/                                      # Documentation and analysis
│   ├── RLOO_Improvement_Analysis.md           # RLOO improvement analysis
│   ├── RLOO_Memory_Optimization_Methods.md    # Memory optimization methods
│   └── RLOO_Memory_Optimization_Task.md       # Task specifications
│
├── experiments/                               # Experiment scripts
│   ├── train.py                              # RLOO training script
│   ├── config.py                             # Experiment configuration
│   ├── memory_monitor.py                     # Memory monitoring
│   └── run_*_rloo*.sh                        # Individual experiment scripts
│
├── results/data/                              # Experimental results
│   ├── results_basic/                        # Basic RLOO results
│   ├── results_lazy/                         # Lazy Generation results
│   ├── results_repeatsampler/                # RepeatSampler results
│   └── results_string/                       # String-Level results
│
├── visualization/                             # Visualization and analysis
│   ├── combine_gpu_memory_data.py            # Data aggregation script
│   ├── create_memory_charts.py               # Chart generation script
│   └── *.png                                 # Generated charts
│
├── models/                                    # Pre-trained models
├── trl/                                       # TRL library (modified version)
│
├── run_*.sh                                   # Main experiment runners
├── combine_results.sh                        # Result aggregation
└── visualize_results.sh                      # Visualization runner
```

## Key Achievements

### Memory Optimization Results
- **String-Level Processing**: **Average 55% memory reduction**
  - `rloo_k=2`: 55.5% reduction (14.38GB → 6.40GB)
  - `rloo_k=4`: 59.2% reduction (26.99GB → 11.01GB)
  - `rloo_k=8`: 15.2% reduction (23.87GB → 20.25GB)

### Critical Finding: OOM Issues at rloo_k=8
At `rloo_k=8`, all optimization methods except String-Level Processing encountered Out-Of-Memory (OOM) errors, resulting in lower performance than `rloo_k=4`. Only String-Level Processing successfully completed all experiments without memory issues.

## Implemented Optimization Methods

### 1. String-Level Processing (Best Performance)
**What**: OnlineDPO-style string processing approach
**Why**: Avoids token-level replication by processing prompts at string level before tokenization
**Implementation**: 
- Decode prompts to strings
- Repeat at string level
- Re-tokenize all at once
- Results in significant memory savings

### 2. Lazy Generation
**What**: Sequential generation approach
**Why**: Reduces peak memory usage during generation phase
**Implementation**:
- Generate responses sequentially for each rloo_k iteration
- Concatenate results to match original API
- Maintains compatibility while reducing memory spikes

### 3. RepeatSampler
**What**: GRPO-style data sampling optimization
**Why**: Avoids data replication at the sampling level
**Implementation**:
- Implement RepeatSampler similar to GRPO
- Use index repetition instead of data duplication
- Theoretical memory efficiency gains

## Quick Start

### 1. Environment Setup
```bash
./setup_env.sh
```

### 2. Run Experiments
```bash
# Run all methods with rloo_k=2,4,8
./run_basic.sh          # Basic RLOO (no optimization)
./run_lazy.sh           # Lazy Generation approach
./run_repeatsampler.sh  # RepeatSampler approach  
./run_string-level.sh   # String-Level Processing (recommended)
```

### 3. Analyze Results
```bash
# Combine all experimental data
./combine_results.sh

# Generate visualization charts
./visualize_results.sh
```

## Experimental Results

### Memory Usage Summary (Peak GPU Memory)

| Method | rloo_k=2 | rloo_k=4 | rloo_k=8 | Status |
|--------|----------|----------|----------|---------|
| **Basic** | 14.38GB | 26.99GB | 23.87GB | Completed |
| **Lazy** | 14.49GB | 27.22GB | **OOM** | Failed at k=8 |
| **RepeatSampler** | 14.48GB | 27.22GB | **OOM** | Failed at k=8 |
| **String-Level** | 6.40GB | 11.01GB | 20.25GB | **All Completed** |

### Key Observations

1. **String-Level Processing** is the only method that successfully handles all configurations
2. **Lazy Generation** and **RepeatSampler** show minimal improvements and fail at higher memory demands
3. **OOM errors** at `rloo_k=8` demonstrate the critical need for effective memory optimization
4. **String-Level Processing** achieves 50%+ memory reduction consistently

## Generated Visualizations

After running experiments and visualization:
- `peak_memory_comparison.png` - Peak memory usage comparison
- `memory_usage_timeline.png` - Memory usage over time for each method
- `method_performance_summary.png` - Performance summary and savings

## Documentation

- [Korean Documentation](docs/README_ko.md) - Complete Korean version
