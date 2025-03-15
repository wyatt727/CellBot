#!/usr/bin/env python3
"""
CellBot Main Entry Point

This script ensures that CellBot runs in a virtual environment with all dependencies installed.
It automatically sets up the environment if needed.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """
    Main entry point for CellBot that ensures it runs in a virtual environment.
    """
    # Get the directory where this script is located
    base_dir = Path(__file__).resolve().parent
    
    # Execute the setup.py script with the 'run' command
    setup_script = base_dir / "setup.py"
    
    if not setup_script.exists():
        print(f"Error: Setup script not found at {setup_script}")
        sys.exit(1)
    
    # Pass all command line arguments to the run command
    args = ["run"]
    if len(sys.argv) > 1:
        args.append(sys.argv[1])  # First arg is the script to run (default: run.py)
        args.extend(sys.argv[2:])  # Rest of args are passed to the script
    
    cmd = [sys.executable, str(setup_script)] + args
    
    try:
        # Execute the setup.py script with the run command
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\nCellBot terminated by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error running CellBot: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 