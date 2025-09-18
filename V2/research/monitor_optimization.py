#!/usr/bin/env python3
"""Monitor optimization progress in real-time."""

import time
import psutil
import os
from pathlib import Path

def monitor_optimization():
    """Monitor the optimization process."""
    print("🔍 Monitoring hole optimization...")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        while True:
            # Find the optimization process
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['cmdline'] and 'hole_optimization.main' in ' '.join(proc.info['cmdline']):
                        # Get process info
                        cpu_percent = proc.cpu_percent()
                        memory_info = proc.memory_info()
                        memory_mb = memory_info.rss / 1024 / 1024
                        
                        print(f"\r🔄 Process {proc.info['pid']}: CPU {cpu_percent:.1f}% | Memory {memory_mb:.1f}MB", end='', flush=True)
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            else:
                print("\r❌ Optimization process not found", end='', flush=True)
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")

if __name__ == "__main__":
    monitor_optimization()
