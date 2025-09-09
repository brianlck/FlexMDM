import os
import subprocess
import time
import sys

def run_with_retry(cmd, max_retries=3):
    """Run command with retry logic for transient failures"""
    for attempt in range(max_retries):
        try:
            # Set environment variables to handle wandb issues
            env = os.environ.copy()
            env['WANDB_START_METHOD'] = 'thread'
            env['WANDB_INIT_TIMEOUT'] = '60'
            
            result = subprocess.run(cmd, env=env, check=True)
            return result
        except subprocess.CalledProcessError as e:
            print(f"Attempt {attempt + 1} failed with exit code {e.returncode}")
            if attempt < max_retries - 1:
                print(f"Retrying in 30 seconds...")
                time.sleep(30)
            else:
                print("All retry attempts failed")
                raise

if __name__ == "__main__":
    # Pass through all arguments to the original script
    cmd = [sys.executable, "sft_train.py"] + sys.argv[1:]
    run_with_retry(cmd)
