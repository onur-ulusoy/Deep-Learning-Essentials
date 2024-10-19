import subprocess
import os
import sys

# Set the directory where the scripts are located
script_directory = os.path.dirname(os.path.abspath(__file__))

# Set the root directory (two levels up from script directory)
root_directory = os.path.dirname(os.path.dirname(script_directory))

# Path to the virtual environment's Python executable
venv_python = os.path.join(root_directory, 'venv', 'Scripts', 'python.exe')

def run_script(script_name):
    script_path = os.path.join(script_directory, script_name)
    try:
        subprocess.run([venv_python, script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running {script_name}: {e}")

if __name__ == "__main__":
    if not os.path.exists(venv_python):
        print(f"Virtual environment Python not found at {venv_python}")
        sys.exit(1)
    
    run_script('train.py')
    run_script('test.py')