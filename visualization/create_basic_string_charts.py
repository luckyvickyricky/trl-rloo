#!/usr/bin/env python3
"""
Create memory usage visualization charts comparing Basic vs String-level optimization
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy.signal import savgol_filter

# Set style for clean visualization
plt.style.use('default')
sns.set_style("darkgrid")

def load_data():
    """Load combined memory data and filter for Basic and String methods only"""
    csv_path = "../results/combined_gpu_memory.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run combine_memory_data.py first.")
        return None
    
    df = pd.read_csv(csv_path)
    
    # Filter for Basic and String methods only
    filtered_df = df[df['method'].isin(['Basic', 'String'])].copy()
    
    print(f"Loaded {len(filtered_df)} data points for Basic and String methods")
    print(f"Methods: {filtered_df['method'].unique()}")
    print(f"RLOO_K values: {sorted(filtered_df['rloo_k'].unique())}")
    
    return filtered_df

def create_ultra_smooth_data(y_data, x_data):
    """Create ultra-smooth version using rolling mean + EWM"""
    if len(y_data) < 10:
        return y_data
    
    try:
        # Convert to pandas Series for easier manipulation
        series = pd.Series(y_data)
        
        # Apply rolling mean
        window_size = max(5, len(y_data) // 10)
        rolling_smooth = series.rolling(window=window_size, center=True, min_periods=1).mean()
        
        # Apply exponential weighted mean for final smoothing
        ultra_smooth = rolling_smooth.ewm(span=15, adjust=False).mean()
        
        return ultra_smooth.values
    except:
        return y_data

def create_peak_memory_comparison_basic_string(df):
    """Create peak memory usage comparison chart for Basic vs String-level only"""
    
    # Calculate peak memory for each configuration
    peak_data = []
    
    # Handle OOM cases - Basic rloo_k=8 shows OOM
    oom_cases = set()
    
    for method in ['Basic', 'String']:
        for rloo_k in [2, 4, 8]:
            subset = df[(df['method'] == method) & (df['rloo_k'] == rloo_k)]
            if len(subset) > 0:
                peak_memory = subset['gpu_memory_allocated_gb'].max()
                peak_data.append({
                    'Method': method,
                    'RLOO_K': f'rloo_k={rloo_k}',
                    'Peak_Memory_GB': peak_memory,
                    'rloo_k_num': rloo_k
                })
    
    peak_df = pd.DataFrame(peak_data)
    
    # Create the chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define colors for methods
    method_colors = {
        'Basic': '#e74c3c',
        'String': '#3498db'
    }
    
    # Create bar plot
    bars = sns.barplot(data=peak_df[peak_df['Peak_Memory_GB'] > 0], 
                      x='RLOO_K', y='Peak_Memory_GB', hue='Method', 
                      palette=method_colors, ax=ax)
    
    # Add title and labels
    ax.set_title('RLOO Memory Optimization: Basic vs String-Level Processing\nPeak GPU Memory Usage Comparison', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Peak GPU Memory (GB)', fontsize=14)
    ax.set_xlabel('RLOO_K Configuration', fontsize=14)
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f GB', fontsize=11, fontweight='bold')
    
    # Add OOM annotation if needed
    if oom_cases:
        ax.text(0.98, 0.95, 'Note: Basic method encountered OOM at rloo_k=8\nString-level optimization prevents OOM', 
                transform=ax.transAxes, fontsize=10, fontweight='bold',
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))
    
    # Add environment and source information
    env_text = ('Test Environment: RTX 5090, Ubuntu 24.04.2 LTS\n'
                'Model: tiny-Qwen3ForCausalLM\n'
                'Source: https://github.com/luckyvickyricky/trl-rloo\n'
                'Core Logic: https://github.com/luckyvickyricky/trl/blob/1ffaae468cb830176fb05075555054a52354e854/trl/trainer/rloo_trainer.py#L320-L342')
    
    ax.text(0.02, 0.02, env_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
    
    ax.legend(title='Method', title_fontsize=12, fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig('../results/peak_memory_comparison_basic_string.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created: ../results/peak_memory_comparison_basic_string.png")

def create_individual_ultra_smooth_basic_string(df):
    """Create individual method plots with ultra-smooth data for Basic vs String only"""
    
    # Define colors for rloo_k values
    rloo_colors = {
        2: '#e74c3c',    # Red
        4: '#f39c12',    # Orange
        8: '#9b59b6'     # Purple
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    methods = ['Basic', 'String']
    
    for idx, method in enumerate(methods):
        ax = axes[idx]
        
        method_data = df[df['method'] == method]
        
        for rloo_k in sorted(method_data['rloo_k'].unique()):
            rloo_k_int = int(rloo_k)
            subset = method_data[method_data['rloo_k'] == rloo_k].copy()
            if len(subset) > 0:
                # Sort by time for proper plotting
                subset = subset.sort_values('time_normalized')
                
                # Apply ultra-smooth processing
                ultra_smooth_memory = create_ultra_smooth_data(
                    subset['gpu_memory_allocated_gb'].values, 
                    subset['time_normalized'].values
                )
                
                ax.plot(subset['time_normalized'], ultra_smooth_memory,
                       color=rloo_colors.get(rloo_k_int, 'gray'),
                       linestyle='-',
                       label=f'rloo_k={rloo_k}',
                       linewidth=4, alpha=0.9)
        
        ax.set_xlabel('Training Time (seconds)', fontsize=12)
        ax.set_ylabel('GPU Memory Allocated (GB)', fontsize=12)
        ax.set_title(f'{method} Method - Memory Usage by RLOO_K\n(Ultra-Smooth Trend Visualization)', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
        # Add statistics
        stats_info = []
        for rloo_k in sorted(method_data['rloo_k'].unique()):
            subset = method_data[method_data['rloo_k'] == rloo_k]
            if len(subset) > 0:
                original_peak = subset['gpu_memory_allocated_gb'].max()
                smooth_avg = subset['gpu_memory_allocated_gb'].mean()
                stats_info.append(f"rloo_k={rloo_k}: Peak={original_peak:.1f}GB, Avg={smooth_avg:.1f}GB")
        
        stats_text = "\n".join([f"Memory Stats:"] + stats_info)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add OOM note for Basic method
        if method == 'Basic':
            oom_note = ("Note: Basic method may encounter\nOOM errors at higher rloo_k values\ndue to memory inefficiency")
            ax.text(0.98, 0.02, oom_note, transform=ax.transAxes, 
                   verticalalignment='bottom', horizontalalignment='right', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # Add environment and source information
    fig.text(0.5, 0.02, 
             'Test Environment: RTX 5090, Ubuntu 24.04.2 LTS | Model: tiny-Qwen3ForCausalLM | '
             'Source: https://github.com/luckyvickyricky/trl-rloo | '
             'Core Logic: https://github.com/luckyvickyricky/trl/blob/1ffaae468cb830176fb05075555054a52354e854/trl/trainer/rloo_trainer.py#L320-L342',
             ha='center', fontsize=9, style='italic',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for environment info
    plt.savefig('../results/rloo_individual_ultra_smooth_basic_string.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created: ../results/rloo_individual_ultra_smooth_basic_string.png")

def create_memory_reduction_summary(df):
    """Create memory reduction summary showing efficiency improvements"""
    
    basic_data = df[df['method'] == 'Basic']
    string_data = df[df['method'] == 'String']
    
    reduction_data = []
    
    for rloo_k in [2, 4, 8]:
        basic_subset = basic_data[basic_data['rloo_k'] == rloo_k]
        string_subset = string_data[string_data['rloo_k'] == rloo_k]
        
        if len(basic_subset) > 0 and len(string_subset) > 0:
            basic_peak = basic_subset['gpu_memory_allocated_gb'].max()
            string_peak = string_subset['gpu_memory_allocated_gb'].max()
            
            reduction_percent = ((basic_peak - string_peak) / basic_peak) * 100
            reduction_data.append({
                'RLOO_K': f'rloo_k={rloo_k}',
                'Basic_Memory_GB': basic_peak,
                'String_Memory_GB': string_peak,
                'Reduction_Percent': reduction_percent,
                'Memory_Saved_GB': basic_peak - string_peak
            })
    
    reduction_df = pd.DataFrame(reduction_data)
    
    # Check if we have data
    if len(reduction_df) == 0:
        print("No reduction data available - skipping memory reduction summary")
        return
    
    print(f"Reduction data shape: {reduction_df.shape}")
    print(f"Reduction data columns: {reduction_df.columns.tolist()}")
    print("Sample data:")
    print(reduction_df.head())
    
    # Create summary chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Memory usage comparison
    x_pos = np.arange(len(reduction_df))
    width = 0.35
    
    ax1.bar(x_pos - width/2, reduction_df['Basic_Memory_GB'], width, 
           label='Basic RLOO', color='#e74c3c', alpha=0.8)
    ax1.bar(x_pos + width/2, reduction_df['String_Memory_GB'], width,
           label='String-Level Optimization', color='#3498db', alpha=0.8)
    
    ax1.set_xlabel('RLOO_K Configuration')
    ax1.set_ylabel('Peak Memory Usage (GB)')
    ax1.set_title('Memory Usage Comparison: Basic vs String-Level')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(reduction_df['RLOO_K'])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (basic, string) in enumerate(zip(reduction_df['Basic_Memory_GB'], reduction_df['String_Memory_GB'])):
        ax1.text(i - width/2, basic + 0.3, f'{basic:.1f}GB', ha='center', fontweight='bold')
        ax1.text(i + width/2, string + 0.3, f'{string:.1f}GB', ha='center', fontweight='bold')
    
    # Memory reduction percentage
    bars = ax2.bar(reduction_df['RLOO_K'], reduction_df['Reduction_Percent'], 
                   color='#27ae60', alpha=0.8)
    
    ax2.set_xlabel('RLOO_K Configuration')
    ax2.set_ylabel('Memory Reduction (%)')
    ax2.set_title('Memory Efficiency Improvement')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add percentage labels
    for bar, percent in zip(bars, reduction_df['Reduction_Percent']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{percent:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../results/memory_reduction_summary_basic_string.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created: ../results/memory_reduction_summary_basic_string.png")
    
    # Print summary
    print("\n=== Memory Optimization Results ===")
    for _, row in reduction_df.iterrows():
        print(f"{row['RLOO_K']}: {row['Basic_Memory_GB']:.2f}GB → {row['String_Memory_GB']:.2f}GB "
              f"({row['Reduction_Percent']:.1f}% reduction)")

def main():
    """Main visualization function for Basic vs String comparison"""
    print("Creating Basic vs String-level memory optimization charts...")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    if df.empty:
        print("No data available for Basic and String methods")
        return
    
    # Create all charts
    create_peak_memory_comparison_basic_string(df)
    create_individual_ultra_smooth_basic_string(df)
    create_memory_reduction_summary(df)
    
    print("\nBasic vs String-level visualization charts created successfully!")
    print("\nGenerated files:")
    print("- peak_memory_comparison_basic_string.png")
    print("- rloo_individual_ultra_smooth_basic_string.png") 
    print("- memory_reduction_summary_basic_string.png")

if __name__ == "__main__":
    main()