#!/usr/bin/env python3
"""Watch optimization progress from log files."""

import time
import os
import re
from pathlib import Path
from datetime import datetime

def watch_optimization_progress():
    """Watch the optimization progress by monitoring log files."""
    print("🔍 Watching hole optimization progress...")
    print("Press Ctrl+C to stop monitoring")
    
    # Find the most recent log file
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        print("❌ No outputs directory found")
        return
    
    # Get the most recent date directory
    date_dirs = [d for d in outputs_dir.iterdir() if d.is_dir()]
    if not date_dirs:
        print("❌ No date directories found in outputs")
        return
    
    latest_date = max(date_dirs, key=lambda x: x.name)
    
    # Get the most recent time directory
    time_dirs = [d for d in latest_date.iterdir() if d.is_dir()]
    if not time_dirs:
        print("❌ No time directories found")
        return
    
    latest_time = max(time_dirs, key=lambda x: x.name)
    log_file = latest_time / "main.log"
    
    print(f"📁 Monitoring: {log_file}")
    
    # Track progress
    last_iteration = 0
    start_time = None
    
    try:
        with open(log_file, 'r') as f:
            # Go to end of file
            f.seek(0, 2)
            
            while True:
                line = f.readline()
                if line:
                    # Look for iteration progress
                    if "Iteration" in line and "/" in line:
                        match = re.search(r'Iteration (\d+)/(\d+)', line)
                        if match:
                            current_iter = int(match.group(1))
                            total_iter = int(match.group(2))
                            
                            if start_time is None:
                                start_time = time.time()
                            
                            if current_iter > last_iteration:
                                elapsed = time.time() - start_time
                                progress = (current_iter / total_iter) * 100
                                
                                if current_iter > 1:
                                    avg_time_per_iter = elapsed / current_iter
                                    remaining_iters = total_iter - current_iter
                                    eta_seconds = remaining_iters * avg_time_per_iter
                                    eta = datetime.fromtimestamp(time.time() + eta_seconds).strftime("%H:%M:%S")
                                else:
                                    eta = "Calculating..."
                                
                                print(f"\r🔄 Progress: {current_iter}/{total_iter} ({progress:.1f}%) | ETA: {eta}", end='', flush=True)
                                last_iteration = current_iter
                    
                    # Look for convergence
                    elif "Converged" in line:
                        print(f"\n✅ {line.strip()}")
                        break
                    
                    # Look for completion
                    elif "Final Results" in line:
                        print(f"\n🎉 {line.strip()}")
                        break
                        
                else:
                    time.sleep(1)
                    
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")
    except FileNotFoundError:
        print(f"❌ Log file not found: {log_file}")

if __name__ == "__main__":
    watch_optimization_progress()
