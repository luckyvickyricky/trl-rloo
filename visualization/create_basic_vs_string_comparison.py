#!/usr/bin/env python3
"""
Create comparison chart between Basic RLOO and String-level optimization
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_basic_vs_string_comparison():
    """Create focused comparison between Basic and String-level methods"""
    
    # Load the combined data
    df = pd.read_csv('../results/combined_gpu_memory.csv')
    
    # Filter for Basic and String methods only
    filtered_df = df[df['method'].isin(['Basic', 'String'])].copy()
    
    if filtered_df.empty:
        print("No Basic or String data found!")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('RLOO Memory Optimization: Basic vs String-Level Processing', fontsize=16, fontweight='bold')
    
    # Color palette
    colors = {'Basic': '#e74c3c', 'String': '#2ecc71'}
    
    # 1. Memory usage over time for each rloo_k
    ax1 = axes[0, 0]
    for rloo_k in sorted(filtered_df['rloo_k'].unique()):
        subset = filtered_df[filtered_df['rloo_k'] == rloo_k]
        for method in ['Basic', 'String']:
            method_data = subset[subset['method'] == method]
            if not method_data.empty:
                ax1.plot(method_data['time_normalized'], 
                        method_data['gpu_memory_allocated_gb'],
                        label=f'{method} (rloo_k={rloo_k})',
                        color=colors[method],
                        linestyle='-' if rloo_k == '2' else '--' if rloo_k == '4' else ':',
                        linewidth=2)
    
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('GPU Memory (GB)')
    ax1.set_title('Memory Usage Over Time')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Peak memory comparison by rloo_k
    ax2 = axes[0, 1]
    peak_memory = filtered_df.groupby(['method', 'rloo_k'])['gpu_memory_allocated_gb'].max().reset_index()
    
    # Create bar plot
    x_pos = np.arange(len(peak_memory['rloo_k'].unique()))
    width = 0.35
    
    for i, method in enumerate(['Basic', 'String']):
        method_peaks = peak_memory[peak_memory['method'] == method]
        values = []
        for rloo_k in sorted(peak_memory['rloo_k'].unique()):
            value = method_peaks[method_peaks['rloo_k'] == rloo_k]['gpu_memory_allocated_gb'].values
            values.append(value[0] if len(value) > 0 else 0)
        
        ax2.bar(x_pos + i*width, values, width, 
               label=method, color=colors[method], alpha=0.8)
    
    ax2.set_xlabel('RLOO K Value')
    ax2.set_ylabel('Peak GPU Memory (GB)')
    ax2.set_title('Peak Memory Usage Comparison')
    ax2.set_xticks(x_pos + width/2)
    ax2.set_xticklabels(sorted(peak_memory['rloo_k'].unique()))
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Memory efficiency (reduction percentage)
    ax3 = axes[1, 0]
    
    # Calculate memory reduction percentage
    reductions = []
    for rloo_k in sorted(peak_memory['rloo_k'].unique()):
        basic_peak = peak_memory[(peak_memory['method'] == 'Basic') & (peak_memory['rloo_k'] == rloo_k)]['gpu_memory_allocated_gb'].values
        string_peak = peak_memory[(peak_memory['method'] == 'String') & (peak_memory['rloo_k'] == rloo_k)]['gpu_memory_allocated_gb'].values
        
        if len(basic_peak) > 0 and len(string_peak) > 0:
            reduction = (basic_peak[0] - string_peak[0]) / basic_peak[0] * 100
            reductions.append({'rloo_k': rloo_k, 'reduction_percent': reduction})
    
    if reductions:
        reduction_df = pd.DataFrame(reductions)
        bars = ax3.bar(reduction_df['rloo_k'], reduction_df['reduction_percent'], 
                      color='#3498db', alpha=0.8)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax3.set_xlabel('RLOO K Value')
    ax3.set_ylabel('Memory Reduction (%)')
    ax3.set_title('Memory Efficiency Improvement')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, max([r['reduction_percent'] for r in reductions]) * 1.2 if reductions else 100)
    
    # 4. Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create summary statistics
    summary_stats = []
    for method in ['Basic', 'String']:
        for rloo_k in sorted(filtered_df['rloo_k'].unique()):
            subset = filtered_df[(filtered_df['method'] == method) & (filtered_df['rloo_k'] == rloo_k)]
            if not subset.empty:
                peak_mem = subset['gpu_memory_allocated_gb'].max()
                avg_mem = subset['gpu_memory_allocated_gb'].mean()
                summary_stats.append({
                    'Method': method,
                    'RLOO K': rloo_k,
                    'Peak (GB)': f'{peak_mem:.2f}',
                    'Average (GB)': f'{avg_mem:.2f}'
                })
    
    if summary_stats:
        summary_df = pd.DataFrame(summary_stats)
        
        # Create table
        table_data = []
        for _, row in summary_df.iterrows():
            table_data.append([row['Method'], row['RLOO K'], row['Peak (GB)'], row['Average (GB)']])
        
        table = ax4.table(cellText=table_data,
                         colLabels=['Method', 'RLOO K', 'Peak Memory (GB)', 'Average Memory (GB)'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0.3, 1, 0.6])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the table
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # Header row
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#95a5a6')
            elif j == 0:  # Method column
                if 'Basic' in cell.get_text().get_text():
                    cell.set_facecolor('#ffebee')
                elif 'String' in cell.get_text().get_text():
                    cell.set_facecolor('#e8f5e8')
        
        ax4.set_title('Memory Usage Summary', pad=20, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    output_file = '../results/basic_vs_string_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Comparison chart saved to: {output_file}")
    
    # Display summary
    print("\n=== RLOO Memory Optimization Summary ===")
    print("Method Comparison: Basic vs String-Level Processing")
    print("-" * 50)
    
    for rloo_k in sorted(peak_memory['rloo_k'].unique()):
        basic_peak = peak_memory[(peak_memory['method'] == 'Basic') & (peak_memory['rloo_k'] == rloo_k)]['gpu_memory_allocated_gb'].values
        string_peak = peak_memory[(peak_memory['method'] == 'String') & (peak_memory['rloo_k'] == rloo_k)]['gpu_memory_allocated_gb'].values
        
        if len(basic_peak) > 0 and len(string_peak) > 0:
            reduction = (basic_peak[0] - string_peak[0]) / basic_peak[0] * 100
            print(f"RLOO K={rloo_k}:")
            print(f"  Basic:  {basic_peak[0]:.2f} GB")
            print(f"  String: {string_peak[0]:.2f} GB")
            print(f"  Reduction: {reduction:.1f}%")
            print()
    
    return output_file

if __name__ == "__main__":
    create_basic_vs_string_comparison()