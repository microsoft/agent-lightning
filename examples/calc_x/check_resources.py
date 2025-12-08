#!/usr/bin/env python3
"""Quick snapshot of current system resources."""

import sys

try:
    import psutil
except ImportError:
    print("Error: psutil not installed. Install it with: pip install psutil")
    sys.exit(1)

try:
    import GPUtil
    gpu_available = True
except ImportError:
    gpu_available = False


def format_bytes(bytes_value):
    """Format bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def main():
    print("\n" + "=" * 80)
    print("📊 System Resource Snapshot")
    print("=" * 80)
    
    # CPU Info
    print("\n🖥️  CPU Information:")
    print(f"   Logical CPUs: {psutil.cpu_count(logical=True)}")
    print(f"   Physical CPUs: {psutil.cpu_count(logical=False)}")
    print(f"   CPU Usage: {psutil.cpu_percent(interval=1)}%")
    
    # Per-CPU usage
    per_cpu = psutil.cpu_percent(interval=1, percpu=True)
    print(f"   Per-CPU Usage: {', '.join([f'{x:.1f}%' for x in per_cpu[:8]])}...")
    
    # Memory Info
    print("\n💾 Memory Information:")
    mem = psutil.virtual_memory()
    print(f"   Total: {format_bytes(mem.total)}")
    print(f"   Available: {format_bytes(mem.available)} ({mem.percent}% used)")
    print(f"   Used: {format_bytes(mem.used)}")
    print(f"   Free: {format_bytes(mem.free)}")
    
    # Swap Info
    swap = psutil.swap_memory()
    print(f"\n💿 Swap Memory:")
    print(f"   Total: {format_bytes(swap.total)}")
    print(f"   Used: {format_bytes(swap.used)} ({swap.percent}%)")
    print(f"   Free: {format_bytes(swap.free)}")
    
    # Disk Info
    print("\n💽 Disk Usage:")
    disk = psutil.disk_usage('/')
    print(f"   Total: {format_bytes(disk.total)}")
    print(f"   Used: {format_bytes(disk.used)} ({disk.percent}%)")
    print(f"   Free: {format_bytes(disk.free)}")
    
    # GPU Info (if available)
    if gpu_available:
        print("\n🎮 GPU Information:")
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                for i, gpu in enumerate(gpus):
                    print(f"   GPU {i}: {gpu.name}")
                    print(f"      Memory: {gpu.memoryUsed:.0f}MB / {gpu.memoryTotal:.0f}MB ({gpu.memoryUtil*100:.1f}%)")
                    print(f"      GPU Utilization: {gpu.load*100:.1f}%")
                    print(f"      Temperature: {gpu.temperature}°C")
            else:
                print("   No GPUs found")
        except Exception as e:
            print(f"   Error reading GPU info: {e}")
    else:
        print("\n🎮 GPU Information: (Install GPUtil for GPU stats: pip install gputil)")
    
    # Process Count
    print("\n🔢 Process Information:")
    python_procs = [p for p in psutil.process_iter(['name']) if 'python' in p.info['name'].lower()]
    print(f"   Total Processes: {len(list(psutil.process_iter()))}")
    print(f"   Python Processes: {len(python_procs)}")
    
    # Agent-Lightning processes
    agl_procs = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if 'agentlightning' in cmdline.lower() or 'train_calc' in cmdline.lower() or 'calc_agent' in cmdline.lower():
                agl_procs.append(proc)
        except:
            pass
    
    if agl_procs:
        print(f"   Agent-Lightning Processes: {len(agl_procs)}")
        print("\n   🎯 Active Agent-Lightning Processes:")
        for proc in agl_procs[:10]:  # Show first 10
            try:
                mem_mb = proc.memory_info().rss / 1024 / 1024
                cpu = proc.cpu_percent(interval=0.1)
                cmdline = ' '.join(proc.cmdline()[:3])  # First 3 parts
                print(f"      PID {proc.pid}: {cmdline[:60]}... (CPU: {cpu:.1f}%, Mem: {mem_mb:.1f}MB)")
            except:
                pass
    else:
        print(f"   Agent-Lightning Processes: None found")
    
    print("\n" + "=" * 80)
    print("💡 Tips:")
    print("   - Run 'python monitor_resources.py' to monitor resources in real-time")
    print("   - Run 'python monitor_resources.py --output stats.csv' to save data")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()

