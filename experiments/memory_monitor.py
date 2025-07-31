#!/usr/bin/env python3
"""
External GPU memory monitoring service that runs independently and logs to CSV
"""

import csv
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil
import torch

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


class ExternalMemoryMonitor:
    """External memory monitor that logs GPU/CPU usage to CSV every 0.1 seconds"""
    
    def __init__(self, output_dir: str, interval: float = 0.1):
        self.output_dir = Path(output_dir)
        self.interval = interval
        self.csv_file = self.output_dir / "memory_usage.csv"
        self.running = True
        
        # Initialize NVML for system-wide GPU monitoring
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.use_nvml = True
            except:
                self.use_nvml = False
        else:
            self.use_nvml = False
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Initialize CSV file with headers
        self._initialize_csv()
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.running = False
        
    def _initialize_csv(self):
        """Initialize CSV file with headers"""
        headers = [
            'timestamp',
            'datetime',
            'gpu_memory_allocated_mb',
            'gpu_memory_reserved_mb', 
            'gpu_memory_free_mb',
            'gpu_memory_total_mb',
            'gpu_utilization_percent',
            'cpu_memory_used_mb',
            'cpu_memory_percent',
            'cpu_usage_percent'
        ]
        
        # Ensure parent directory exists
        self.csv_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            

        
    def _get_gpu_info(self):
        """Get GPU memory and utilization info using NVML for system-wide monitoring"""
        if self.use_nvml:
            try:
                # Get memory info from NVML (system-wide)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
                util_info = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                
                total_mb = mem_info.total / 1024**2
                used_mb = mem_info.used / 1024**2
                free_mb = mem_info.free / 1024**2
                
                return {
                    'allocated_mb': used_mb,  # System-wide used memory
                    'reserved_mb': 0,  # Not available in NVML
                    'free_mb': free_mb,
                    'total_mb': total_mb,
                    'utilization_percent': util_info.gpu
                }
            except Exception as e:
                # Fallback to torch if NVML fails
                pass
        
        # Fallback to PyTorch (process-specific)
        if not torch.cuda.is_available():
            return {
                'allocated_mb': 0,
                'reserved_mb': 0,
                'free_mb': 0,
                'total_mb': 0,
                'utilization_percent': 0
            }
            
        # GPU memory (PyTorch process-specific)
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        
        # Get total GPU memory
        gpu_props = torch.cuda.get_device_properties(0)
        total = gpu_props.total_memory / 1024**2
        free = total - allocated
        
        # GPU utilization (approximate based on memory usage)
        utilization = (allocated / total) * 100 if total > 0 else 0
        
        return {
            'allocated_mb': allocated,
            'reserved_mb': reserved,
            'free_mb': free,
            'total_mb': total,
            'utilization_percent': utilization
        }
        
    def _get_cpu_info(self):
        """Get CPU memory and usage info"""
        # Memory info
        memory = psutil.virtual_memory()
        memory_used_mb = memory.used / 1024**2
        memory_percent = memory.percent
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        return {
            'memory_used_mb': memory_used_mb,
            'memory_percent': memory_percent,
            'cpu_percent': cpu_percent
        }
        
    def log_memory_usage(self):
        """Log current memory usage to CSV"""
        current_time = time.time()
        current_datetime = datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')
        
        gpu_info = self._get_gpu_info()
        cpu_info = self._get_cpu_info()
        
        row = [
            current_time,
            current_datetime,
            f"{gpu_info['allocated_mb']:.1f}",
            f"{gpu_info['reserved_mb']:.1f}",
            f"{gpu_info['free_mb']:.1f}",
            f"{gpu_info['total_mb']:.1f}",
            f"{gpu_info['utilization_percent']:.1f}",
            f"{cpu_info['memory_used_mb']:.1f}",
            f"{cpu_info['memory_percent']:.1f}",
            f"{cpu_info['cpu_percent']:.1f}"
        ]
        
        # Check if file still exists (might be deleted by training process)
        if not self.csv_file.exists():
            self._initialize_csv()
        
        try:
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except FileNotFoundError:
            # File was deleted, reinitialize and retry
            self._initialize_csv()
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            
        # Silent logging - no console output
        
    def run(self):
        """Main monitoring loop"""
        # Silent monitoring - no console output during operation
        
        try:
            while self.running:
                self.log_memory_usage()
                
                # Sleep in small intervals to allow for graceful shutdown
                if not self.running:
                    break
                time.sleep(self.interval)
                    
        except KeyboardInterrupt:
            pass
            
        finally:
            self._generate_summary()
            
    def _generate_summary(self):
        """Generate summary statistics from the collected data"""
        try:
            import pandas as pd
            
            # Check if CSV file exists
            if not self.csv_file.exists():
                return
                
            # Read the CSV file
            df = pd.read_csv(self.csv_file)
            
            if len(df) == 0:
                return
                
            # Calculate summary statistics
            summary = {
                'monitoring_duration_seconds': df['timestamp'].max() - df['timestamp'].min(),
                'total_samples': len(df),
                'gpu_memory_stats': {
                    'allocated_mb': {
                        'mean': df['gpu_memory_allocated_mb'].mean(),
                        'max': df['gpu_memory_allocated_mb'].max(),
                        'min': df['gpu_memory_allocated_mb'].min(),
                        'std': df['gpu_memory_allocated_mb'].std()
                    },
                    'peak_usage_time': df.loc[df['gpu_memory_allocated_mb'].idxmax()]['datetime']
                },
                'cpu_memory_stats': {
                    'mean_percent': df['cpu_memory_percent'].mean(),
                    'max_percent': df['cpu_memory_percent'].max(),
                    'min_percent': df['cpu_memory_percent'].min()
                }
            }
            
            # Save summary to JSON
            import json
            summary_file = self.output_dir / "memory_monitoring_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
                
            # Summary saved silently
            
        except ImportError:
            pass  # pandas not available, skip summary
        except Exception as e:
            pass  # Error in summary generation


def main():
    """Main entry point for external memory monitoring"""
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python memory_monitor.py <output_directory> [interval_seconds]")
        print("Default interval: 0.1 seconds")
        sys.exit(1)
        
    output_dir = sys.argv[1]
    interval = float(sys.argv[2]) if len(sys.argv) == 3 else 0.1
    
    # Starting silent memory monitor
    monitor = ExternalMemoryMonitor(output_dir, interval)
    monitor.run()


if __name__ == "__main__":
    main()