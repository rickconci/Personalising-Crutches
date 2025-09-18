#!/usr/bin/env python3
"""Monitor optimization performance and parallelization."""

import time
import psutil
import os
import threading
from pathlib import Path
import re
from datetime import datetime

class PerformanceMonitor:
    def __init__(self):
        self.running = True
        self.optimization_process = None
        self.start_time = None
        self.last_iteration = 0
        
    def find_optimization_process(self):
        """Find the running optimization process."""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
            try:
                if proc.info['cmdline'] and 'hole_optimization.main' in ' '.join(proc.info['cmdline']):
                    return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None
    
    def monitor_system_resources(self):
        """Monitor system resource usage."""
        while self.running:
            try:
                # Overall system stats
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                # Process-specific stats
                proc = self.find_optimization_process()
                if proc:
                    proc_cpu = proc.cpu_percent()
                    proc_memory = proc.memory_info()
                    proc_memory_mb = proc_memory.rss / 1024 / 1024
                    
                    print(f"\r🖥️  System: CPU {cpu_percent:.1f}% | RAM {memory.percent:.1f}% | "
                          f"Process: CPU {proc_cpu:.1f}% | RAM {proc_memory_mb:.1f}MB", end='', flush=True)
                else:
                    print(f"\r🖥️  System: CPU {cpu_percent:.1f}% | RAM {memory.percent:.1f}% | "
                          f"❌ Optimization process not found", end='', flush=True)
                
                time.sleep(2)
            except KeyboardInterrupt:
                break
    
    def monitor_log_progress(self):
        """Monitor progress from log files."""
        # Find the most recent log file
        outputs_dir = Path("outputs")
        if not outputs_dir.exists():
            return
        
        date_dirs = [d for d in outputs_dir.iterdir() if d.is_dir()]
        if not date_dirs:
            return
        
        latest_date = max(date_dirs, key=lambda x: x.name)
        time_dirs = [d for d in latest_date.iterdir() if d.is_dir()]
        if not time_dirs:
            return
        
        latest_time = max(time_dirs, key=lambda x: x.name)
        log_file = latest_time / "main.log"
        
        if not log_file.exists():
            return
        
        print(f"\n📁 Monitoring log: {log_file}")
        
        try:
            with open(log_file, 'r') as f:
                f.seek(0, 2)  # Go to end
                
                while self.running:
                    line = f.readline()
                    if line:
                        # Look for iteration progress
                        if "Iteration" in line and "/" in line:
                            match = re.search(r'Iteration (\d+)/(\d+)', line)
                            if match:
                                current_iter = int(match.group(1))
                                total_iter = int(match.group(2))
                                
                                if self.start_time is None:
                                    self.start_time = time.time()
                                
                                if current_iter > self.last_iteration:
                                    elapsed = time.time() - self.start_time
                                    progress = (current_iter / total_iter) * 100
                                    
                                    if current_iter > 1:
                                        avg_time_per_iter = elapsed / current_iter
                                        remaining_iters = total_iter - current_iter
                                        eta_seconds = remaining_iters * avg_time_per_iter
                                        eta = datetime.fromtimestamp(time.time() + eta_seconds).strftime("%H:%M:%S")
                                    else:
                                        eta = "Calculating..."
                                    
                                    print(f"\n🔄 Iteration {current_iter}/{total_iter} ({progress:.1f}%) | "
                                          f"Elapsed: {elapsed/60:.1f}min | ETA: {eta}")
                                    self.last_iteration = current_iter
                        
                        # Look for performance metrics
                        elif "Loss:" in line:
                            print(f"📊 {line.strip()}")
                        elif "Vocab Score:" in line:
                            print(f"📈 {line.strip()}")
                        elif "Truss Complexity:" in line:
                            print(f"🔧 {line.strip()}")
                        elif "Converged" in line or "Final Results" in line:
                            print(f"\n✅ {line.strip()}")
                            self.running = False
                            break
                    else:
                        time.sleep(1)
                        
        except (FileNotFoundError, KeyboardInterrupt):
            pass
    
    def run(self):
        """Run the performance monitor."""
        print("🔍 Performance Monitor for Hole Optimization")
        print("=" * 50)
        print("Monitoring system resources and optimization progress...")
        print("Press Ctrl+C to stop monitoring")
        print()
        
        # Start resource monitoring in a separate thread
        resource_thread = threading.Thread(target=self.monitor_system_resources)
        resource_thread.daemon = True
        resource_thread.start()
        
        # Monitor log progress in main thread
        try:
            self.monitor_log_progress()
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped")
        finally:
            self.running = False

if __name__ == "__main__":
    monitor = PerformanceMonitor()
    monitor.run()
