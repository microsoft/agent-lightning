import subprocess

def run_cmd(cmd):
    """Execute a shell command and print its output"""
    print(f"👉 Running: {cmd}")
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result

def kill_process_on_port(port):
    result = subprocess.run(
        f"sudo lsof -t -i :{port}", 
        shell=True, 
        capture_output=True, 
        text=True
    )
    pids = result.stdout.strip().split("\n")
    for pid in pids:
        if pid:
            print(f"🔪 Killing process {pid} on port {port}")
            subprocess.run(f"sudo kill -9 {pid}", shell=True)