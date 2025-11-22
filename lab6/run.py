import subprocess
import sys
import time
import os

def run_script(script_name):
    """Runs a python script and waits for it to finish."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {script_name}")
    print(f"{'='*60}")

    if not os.path.exists(script_name):
        print(f"Error: File '{script_name}' not found.")
        print("Please ensure all 3 python files are in the same directory.")
        return

    try:
        subprocess.run([sys.executable, script_name], check=True)
        print(f"\n>>> Finished {script_name}")
        
    except subprocess.CalledProcessError as e:
        print(f"\nError occurred while running {script_name}.")
        print(e)
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
        sys.exit()

def main():
    print("Starting Lab Assignment 6 Execution...")
    print("NOTE: Close each pop-up plot window to proceed to the next problem.\n")
    scripts = [
        "assosiative_memory.py",
        "eight_rook.py",
        "tsp_hopfield.py"
    ]

    for script in scripts:
        run_script(script)
        time.sleep(1)

    print(f"\n{'='*60}")
    print("ALL PARTS COMPLETED.")
    print("Results and plots have been saved to the 'results/' folder.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()