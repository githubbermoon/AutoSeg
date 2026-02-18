import subprocess
import os

try:
    result = subprocess.run(['git', 'status'], cwd='/Users/pranjal/vscode_progs/cv_me/segT', capture_output=True, text=True)
    with open('/Users/pranjal/vscode_progs/cv_me/segT/status_log_py.txt', 'w') as f:
        f.write(result.stdout)
        f.write("\nSTDERR:\n")
        f.write(result.stderr)
except Exception as e:
    with open('/Users/pranjal/vscode_progs/cv_me/segT/status_error_py.txt', 'w') as f:
        f.write(str(e))
