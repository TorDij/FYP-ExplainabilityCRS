import subprocess

# To check which user is running a specific PID
pid = 1239541  # Replace with the PID you want to check
try:
    result = subprocess.run(['ps', '-o', 'user=', '-p', str(pid)], capture_output=True, text=True)
    print(f"User running PID {pid}: {result.stdout.strip()}")
except Exception as e:
    print(f"Error checking PID {pid}: {e}")