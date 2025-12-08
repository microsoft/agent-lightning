#!/usr/bin/env python3
"""Monitor CPU and memory usage of Agent-Lightning training processes.

Usage:
    # Monitor while training is running
    python monitor_resources.py
    
    # Monitor with custom interval
    python monitor_resources.py --interval 2
    
    # Export to CSV
    python monitor_resources.py --output resources.csv
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from typing import List, Dict, Optional

try:
    import psutil
except ImportError:
    print("Error: psutil not installed. Install it with: pip install psutil")
    sys.exit(1)


class ProcessMonitor:
    """Monitor resource usage of Agent-Lightning processes."""
    
    def __init__(self, interval: float = 1.0, output_file: Optional[str] = None):
        self.interval = interval
        self.output_file = output_file
        self.csv_writer = None
        self.csv_file = None
        
        if output_file:
            self.csv_file = open(output_file, 'w', newline='')
            fieldnames = [
                'timestamp', 'process_name', 'pid', 'cpu_percent', 
                'memory_mb', 'memory_percent', 'num_threads', 'status'
            ]
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
            self.csv_writer.writeheader()
    
    def find_agentlightning_processes(self) -> List[psutil.Process]:
        """Find all Agent-Lightning related processes."""
        processes = []
        keywords = [
            'train_calc_agent',
            'calc_agent',
            'runner-',
            'algorithm',
            'vllm',
            'python',  # Catch all python processes for filtering
        ]
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'status']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                
                # Check if it's an Agent-Lightning related process
                if any(keyword in cmdline.lower() for keyword in ['agentlightning', 'calc_agent', 'train_calc']):
                    processes.append(proc)
                # Check for runner/algorithm processes
                elif 'multiprocessing' in cmdline and 'python' in proc.info['name'].lower():
                    processes.append(proc)
                # Check for vllm
                elif 'vllm' in cmdline.lower():
                    processes.append(proc)
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
        return processes
    
    def get_process_info(self, proc: psutil.Process) -> Dict:
        """Get resource usage info for a process."""
        try:
            # Get process name from cmdline
            cmdline = proc.cmdline()
            if len(cmdline) > 1:
                # Extract meaningful name
                if 'train_calc_agent' in ' '.join(cmdline):
                    name = 'train_calc_agent (main)'
                elif 'runner-' in ' '.join(cmdline):
                    # Extract runner ID
                    for part in cmdline:
                        if 'runner-' in part:
                            name = f"Runner-{part.split('runner-')[-1]}"
                            break
                    else:
                        name = 'Runner'
                elif 'vllm' in ' '.join(cmdline).lower():
                    name = 'vLLM Server'
                elif 'algorithm' in ' '.join(cmdline):
                    name = 'Algorithm'
                else:
                    name = proc.name()
            else:
                name = proc.name()
            
            # Get resource usage
            cpu_percent = proc.cpu_percent(interval=0.1)
            memory_info = proc.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB
            memory_percent = proc.memory_percent()
            num_threads = proc.num_threads()
            status = proc.status()
            
            return {
                'name': name,
                'pid': proc.pid,
                'cpu_percent': cpu_percent,
                'memory_mb': memory_mb,
                'memory_percent': memory_percent,
                'num_threads': num_threads,
                'status': status,
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return None
    
    def get_system_summary(self) -> Dict:
        """Get overall system resource usage."""
        return {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
            'memory_available_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024,
            'memory_used_gb': psutil.virtual_memory().used / 1024 / 1024 / 1024,
            'memory_percent': psutil.virtual_memory().percent,
        }
    
    def print_header(self):
        """Print table header."""
        print("\n" + "=" * 120)
        print(f"{'Process Name':<30} {'PID':<10} {'CPU %':<10} {'Memory (MB)':<15} {'Mem %':<10} {'Threads':<10} {'Status':<10}")
        print("=" * 120)
    
    def print_process_info(self, info: Dict):
        """Print process information in a formatted table."""
        print(
            f"{info['name']:<30} "
            f"{info['pid']:<10} "
            f"{info['cpu_percent']:>8.1f}% "
            f"{info['memory_mb']:>13.1f} MB "
            f"{info['memory_percent']:>8.1f}% "
            f"{info['num_threads']:<10} "
            f"{info['status']:<10}"
        )
    
    def print_system_summary(self, summary: Dict, process_infos: List[Dict]):
        """Print system summary."""
        total_cpu = sum(p['cpu_percent'] for p in process_infos)
        total_memory_mb = sum(p['memory_mb'] for p in process_infos)
        
        print("-" * 120)
        print(f"{'TOTAL (Agent-Lightning)':<30} {'':<10} {total_cpu:>8.1f}% {total_memory_mb:>13.1f} MB")
        print("-" * 120)
        print(f"{'SYSTEM OVERALL':<30} "
              f"CPU: {summary['cpu_percent']:>5.1f}% ({summary['cpu_count']} cores) | "
              f"Memory: {summary['memory_used_gb']:.1f}/{summary['memory_total_gb']:.1f} GB "
              f"({summary['memory_percent']:.1f}%)")
        print("=" * 120)
    
    def save_to_csv(self, timestamp: str, info: Dict):
        """Save process info to CSV file."""
        if self.csv_writer:
            self.csv_writer.writerow({
                'timestamp': timestamp,
                'process_name': info['name'],
                'pid': info['pid'],
                'cpu_percent': f"{info['cpu_percent']:.2f}",
                'memory_mb': f"{info['memory_mb']:.2f}",
                'memory_percent': f"{info['memory_percent']:.2f}",
                'num_threads': info['num_threads'],
                'status': info['status'],
            })
            self.csv_file.flush()
    
    def monitor(self):
        """Main monitoring loop."""
        print(f"\n🔍 Monitoring Agent-Lightning processes (interval: {self.interval}s)")
        print("Press Ctrl+C to stop\n")
        
        if self.output_file:
            print(f"📊 Saving data to: {self.output_file}\n")
        
        iteration = 0
        try:
            while True:
                processes = self.find_agentlightning_processes()
                
                if not processes:
                    print(f"\r⏳ No Agent-Lightning processes found... (checked {iteration} times)", end='', flush=True)
                    time.sleep(self.interval)
                    iteration += 1
                    continue
                
                # Clear the waiting message
                if iteration > 0:
                    print("\r" + " " * 80 + "\r", end='')
                
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                self.print_header()
                print(f"⏰ Timestamp: {timestamp}")
                print("-" * 120)
                
                process_infos = []
                for proc in processes:
                    info = self.get_process_info(proc)
                    if info:
                        self.print_process_info(info)
                        process_infos.append(info)
                        
                        if self.output_file:
                            self.save_to_csv(timestamp, info)
                
                if process_infos:
                    system_summary = self.get_system_summary()
                    self.print_system_summary(system_summary, process_infos)
                
                time.sleep(self.interval)
                iteration += 1
                
        except KeyboardInterrupt:
            print("\n\n⛔ Monitoring stopped by user")
            if self.output_file:
                print(f"📁 Data saved to: {self.output_file}")
        finally:
            if self.csv_file:
                self.csv_file.close()


def main():
    parser = argparse.ArgumentParser(
        description='Monitor CPU and memory usage of Agent-Lightning training processes'
    )
    parser.add_argument(
        '--interval', 
        type=float, 
        default=1.0,
        help='Monitoring interval in seconds (default: 1.0)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path (optional)'
    )
    
    args = parser.parse_args()
    
    monitor = ProcessMonitor(interval=args.interval, output_file=args.output)
    monitor.monitor()


if __name__ == '__main__':
    main()

