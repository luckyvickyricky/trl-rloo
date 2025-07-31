#!/usr/bin/env python3
"""
Combine GPU memory data from all experimental results into a unified CSV
"""

import pandas as pd
import os
import glob
from pathlib import Path

def combine_memory_data():
    """Combine all memory usage CSV files into one unified dataset"""
    
    # Define method mapping
    methods = {
        'results_basic': 'Basic',
        'results_lazy': 'Lazy',
        'results_repeatsampler': 'RepeatSampler', 
        'results_string-level': 'String'
    }
    
    # Define rloo_k mapping (both old and new formats)
    rloo_configs = {
        'tiny-Qwen3ForCausalLM2': '2',
        'tiny-Qwen3ForCausalLM4': '4',
        'tiny-Qwen3ForCausalLM8': '8',
        'tiny-Qwen3ForCausalLM_rloo2': '2',
        'tiny-Qwen3ForCausalLM_rloo4': '4',
        'tiny-Qwen3ForCausalLM_rloo8': '8'
    }
    
    combined_data = []
    
    print("Scanning for memory usage data...")
    
    for method_dir, method_name in methods.items():
        method_path = f"../results/data/{method_dir}"
        
        if not os.path.exists(method_path):
            print(f"Warning: {method_path} not found")
            continue
            
        for config_dir, rloo_k in rloo_configs.items():
            csv_path = f"{method_path}/{config_dir}/logs/memory_usage.csv"
            
            if os.path.exists(csv_path):
                print(f"Loading {method_name} rloo_k={rloo_k}: {csv_path}")
                
                try:
                    df = pd.read_csv(csv_path)
                    
                    if len(df) > 0:
                        # Add metadata columns
                        df['method'] = method_name
                        df['rloo_k'] = rloo_k
                        df['method_rloo'] = f"{method_name}_rloo{rloo_k}"
                        
                        # Normalize timestamp
                        df['time_normalized'] = df['timestamp'] - df['timestamp'].iloc[0]
                        
                        # Convert to GB for better readability
                        df['gpu_memory_allocated_gb'] = df['gpu_memory_allocated_mb'] / 1024
                        df['gpu_memory_reserved_gb'] = df['gpu_memory_reserved_mb'] / 1024
                        df['gpu_memory_total_gb'] = df['gpu_memory_total_mb'] / 1024
                        
                        combined_data.append(df)
                        
                except Exception as e:
                    print(f"Error loading {csv_path}: {e}")
            else:
                print(f"File not found: {csv_path}")
    
    if not combined_data:
        print("No data found!")
        return
    
    # Combine all data
    final_df = pd.concat(combined_data, ignore_index=True)
    
    # Save combined data to results directory
    output_file = "../results/combined_gpu_memory.csv"
    final_df.to_csv(output_file, index=False)
    
    print(f"\nCombined data saved to: {output_file}")
    print(f"Total data points: {len(final_df)}")
    print(f"Methods: {final_df['method'].unique()}")
    print(f"RLOO_K values: {sorted(final_df['rloo_k'].unique())}")
    
    # Generate summary statistics
    print("\nSummary Statistics:")
    summary = final_df.groupby(['method', 'rloo_k']).agg({
        'gpu_memory_allocated_gb': ['count', 'max', 'mean'],
        'time_normalized': 'max'
    }).round(2)
    
    print(summary)
    
    return final_df

if __name__ == "__main__":
    combine_memory_data()