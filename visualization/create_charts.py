#!/usr/bin/env python3
"""
Create memory usage visualization charts from combined data
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
# Set explicit color palette
custom_palette = ["#e74c3c", "#f39c12", "#27ae60", "#3498db", "#9b59b6", "#34495e"]
sns.set_palette(custom_palette)

# Force matplotlib color cycle
import matplotlib as mpl
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=custom_palette)

def load_data():
    """Load combined memory data"""
    csv_path = "../results/combined_gpu_memory.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run combine_memory_data.py first.")
        return None
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} data points")
    return df

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

def create_peak_preserving_smooth(y_data, window_length=None):
    """Create peak-preserving smoothed data using Savitzky-Golay filter with peak protection"""
    if len(y_data) < 10:
        return y_data
        
    if window_length is None:
        window_length = min(15, len(y_data) // 3)
        if window_length % 2 == 0:
            window_length += 1
        window_length = max(5, window_length)
    
    try:
        # Apply Savitzky-Golay filter
        sg_smooth = savgol_filter(y_data, window_length, 3)
        
        # Peak protection constraint: don't let smoothed values drop below 95% of original peaks
        peak_protected = np.maximum(sg_smooth, y_data * 0.95)
        
        return peak_protected
    except:
        return y_data

def create_peak_memory_comparison(df):
    """Create peak memory usage comparison chart"""
    
    # Calculate peak memory for each configuration
    peak_data = []
    for method in df['method'].unique():
        for rloo_k in df['rloo_k'].unique():
            subset = df[(df['method'] == method) & (df['rloo_k'] == rloo_k)]
            if len(subset) > 0:
                peak_memory = subset['gpu_memory_allocated_gb'].max()
                peak_data.append({
                    'Method': method,
                    'RLOO_K': f'rloo_k={rloo_k}',
                    'Peak_Memory_GB': peak_memory
                })
    
    peak_df = pd.DataFrame(peak_data)
    
    # Create the chart
    plt.figure(figsize=(12, 8))
    
    # Define colors for methods
    method_colors = {
        'Basic': '#e74c3c',
        'String': '#3498db'
    }
    
    sns.barplot(data=peak_df, x='RLOO_K', y='Peak_Memory_GB', hue='Method', 
               palette=method_colors)
    
    plt.title('RLOO Memory Optimization - Peak GPU Memory Usage Comparison', 
             fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Peak GPU Memory (GB)', fontsize=14)
    plt.xlabel('RLOO_K Configuration', fontsize=14)
    
    # Add value labels on bars
    ax = plt.gca()
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f GB', fontsize=11, fontweight='bold')
    
    plt.legend(title='Optimization Method', title_fontsize=12, fontsize=12)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig('../results/peak_memory_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created: ../results/peak_memory_comparison.png")

def create_individual_peak_preserving_plots(df):
    """Create individual method plots with peak-preserving smoothing"""
    
    # Define colors for rloo_k values (정수 키로 수정)
    rloo_colors = {
        2: '#e74c3c',    # Red
        4: '#f39c12',    # Orange
        8: '#9b59b6'     # Purple
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    
    methods = ['Basic', 'String']
    
    for idx, method in enumerate(methods):
        ax = axes[idx]
        
        method_data = df[df['method'] == method]
        
        for rloo_k in sorted(method_data['rloo_k'].unique()):
            subset = method_data[method_data['rloo_k'] == rloo_k].copy()
            if len(subset) > 0:
                # Sort by time for proper plotting
                subset = subset.sort_values('time_normalized')
                
                # Apply peak-preserving smoothing
                smoothed_memory = create_peak_preserving_smooth(subset['gpu_memory_allocated_gb'].values)
                
                ax.plot(subset['time_normalized'], smoothed_memory,
                       color=rloo_colors.get(rloo_k, 'gray'),
                       linestyle='-',  # All solid lines
                       label=f'rloo_k={rloo_k}',
                       linewidth=3, alpha=0.9)
        
        ax.set_xlabel('Training Time (seconds)', fontsize=12)
        ax.set_ylabel('GPU Memory Allocated (GB)', fontsize=12)
        ax.set_title(f'{method} Method - Memory Usage by RLOO_K\n(Peak-Preserving Smoothing)', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
        # Add statistics text using original peak values
        peak_memories = []
        for rloo_k in sorted(method_data['rloo_k'].unique()):
            subset = method_data[method_data['rloo_k'] == rloo_k]
            if len(subset) > 0:
                peak = subset['gpu_memory_allocated_gb'].max()
                peak_memories.append(f"rloo_k={rloo_k}: {peak:.1f}GB")
        
        stats_text = "\n".join([f"Peak Memory:"] + peak_memories)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('../results/rloo_individual_peak_preserving.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created: ../results/rloo_individual_peak_preserving.png")

def create_individual_ultra_smooth_plots(df):
    """Create individual method plots with ultra-smooth data"""
    
    # Define colors for rloo_k values (정수 키로 수정)
    rloo_colors = {
        2: '#e74c3c',    # Red
        4: '#f39c12',    # Orange
        8: '#9b59b6'     # Purple
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    
    methods = ['Basic', 'String']
    
    for idx, method in enumerate(methods):
        ax = axes[idx]
        
        method_data = df[df['method'] == method]
        
        for rloo_k in sorted(method_data['rloo_k'].unique()):
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
                       color=rloo_colors.get(rloo_k, 'gray'),
                       linestyle='-',  # All solid lines
                       label=f'rloo_k={rloo_k}',
                       linewidth=4, alpha=0.9)  # Thicker lines for smooth curves
        
        ax.set_xlabel('Training Time (seconds)', fontsize=12)
        ax.set_ylabel('GPU Memory Allocated (GB)', fontsize=12)
        ax.set_title(f'{method} Method - Memory Usage by RLOO_K\n(Ultra-Smooth Trend Visualization)', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
        # Add statistics: Show both original peak and smooth average
        stats_info = []
        for rloo_k in sorted(method_data['rloo_k'].unique()):
            subset = method_data[method_data['rloo_k'] == rloo_k]
            if len(subset) > 0:
                original_peak = subset['gpu_memory_allocated_gb'].max()
                smooth_avg = subset['gpu_memory_allocated_gb'].mean()
                stats_info.append(f"rloo_k={rloo_k}: Peak={original_peak:.1f}GB, Avg={smooth_avg:.1f}GB")
        
        stats_text = "\n".join([f"Memory Stats:"] + stats_info)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('../results/rloo_individual_ultra_smooth.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created: ../results/rloo_individual_ultra_smooth.png")

def create_performance_summary(df):
    """Create performance summary chart"""
    
    # Calculate memory savings relative to Basic
    basic_data = df[df['method'] == 'Basic']
    
    savings_data = []
    for method in ['String']:
        method_data = df[df['method'] == method]
        
        for rloo_k in ['2', '4', '8']:
            basic_subset = basic_data[basic_data['rloo_k'] == rloo_k]
            method_subset = method_data[method_data['rloo_k'] == rloo_k]
            
            if len(basic_subset) > 0 and len(method_subset) > 0:
                basic_peak = basic_subset['gpu_memory_allocated_gb'].max()
                method_peak = method_subset['gpu_memory_allocated_gb'].max()
                
                if basic_peak > 0:
                    savings_percent = ((basic_peak - method_peak) / basic_peak) * 100
                    savings_data.append({
                        'Method': method,
                        'rloo_k': f'rloo_k={rloo_k}',
                        'Memory_Savings_Percent': savings_percent,
                        'Basic_Peak_GB': basic_peak,
                        'Method_Peak_GB': method_peak
                    })
    
    savings_df = pd.DataFrame(savings_data)
    
    # Skip if no data available
    if len(savings_df) == 0:
        print("No performance comparison data available - skipping performance summary chart")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Create savings chart
    sns.barplot(data=savings_df, x='rloo_k', y='Memory_Savings_Percent', hue='Method')
    
    plt.title('Memory Savings vs Basic RLOO Implementation', 
             fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Memory Savings (%)', fontsize=14)
    plt.xlabel('RLOO_K Configuration', fontsize=14)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Add value labels
    ax = plt.gca()
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', fontsize=10)
    
    plt.legend(title='Optimization Method', title_fontsize=12, fontsize=12)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('../results/method_performance_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created: ../results/method_performance_summary.png")
    
    # Print summary statistics
    print("\nMemory Savings Summary:")
    for _, row in savings_df.iterrows():
        print(f"{row['Method']} {row['RLOO_K']}: {row['Memory_Savings_Percent']:+.1f}% "
              f"({row['Basic_Peak_GB']:.1f}GB → {row['Method_Peak_GB']:.1f}GB)")

def main():
    """Main visualization function"""
    print("Creating memory usage visualization charts...")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Create all charts
    create_peak_memory_comparison(df)
    create_individual_peak_preserving_plots(df)
    create_individual_ultra_smooth_plots(df)
    create_performance_summary(df)
    
    print("\nAll visualization charts created successfully!")

if __name__ == "__main__":
    main()