#!/usr/bin/env python3
"""
CellBot Setup Script

This script:
1. Creates a Python virtual environment if it doesn't exist
2. Installs dependencies inside the virtual environment
3. Provides a way to run the application using the virtual environment
4. Ensures Ollama is running if needed
"""

import os
import sys
import subprocess
import venv
import argparse
import time
from pathlib import Path

# Define paths
BASE_DIR = Path(__file__).resolve().parent
VENV_DIR = BASE_DIR / ".venv"
REQUIREMENTS_FILE = BASE_DIR / "requirements.txt"

# Check for possible typo in directory name (CellBet vs CellBot)
if str(BASE_DIR).endswith('CellBet') and not os.path.exists(VENV_DIR):
    possible_alt_path = str(BASE_DIR).replace('CellBet', 'CellBot')
    if os.path.exists(f"{possible_alt_path}/.venv"):
        print(f"Note: Found virtual environment in {possible_alt_path}/.venv instead of {VENV_DIR}")
        VENV_DIR = Path(f"{possible_alt_path}/.venv")

def create_venv():
    """Create virtual environment if it doesn't exist."""
    if not VENV_DIR.exists():
        print(f"Creating virtual environment in {VENV_DIR}...")
        venv.create(VENV_DIR, with_pip=True)
        return True
    return False

def get_venv_python():
    """Get path to the virtual environment's Python executable."""
    if os.name == 'nt':  # Windows
        return str(VENV_DIR / "Scripts" / "python.exe")
    else:  # Unix-like
        python_path = VENV_DIR / "bin" / "python"
        python3_path = VENV_DIR / "bin" / "python3"
        
        # Check if the virtual environment directory exists
        if not VENV_DIR.exists():
            print(f"Warning: Virtual environment directory not found at {VENV_DIR}")
            # Try to find the directory with a similar name
            parent_dir = VENV_DIR.parent
            if parent_dir.exists():
                venv_candidates = [d for d in parent_dir.glob(".venv")]
                if venv_candidates:
                    print(f"Found potential virtual environment at {venv_candidates[0]}")
                    # Use the found directory instead (without global)
                    new_venv_dir = venv_candidates[0]
                    python_path = new_venv_dir / "bin" / "python"
                    python3_path = new_venv_dir / "bin" / "python3"
        
        # Check if python3 exists when python doesn't
        if not python_path.exists() and python3_path.exists():
            print(f"Using python3 as python executable was not found")
            return str(python3_path)
        
        # If neither exists, provide a helpful error message but return the default path for further error handling
        if not python_path.exists() and not python3_path.exists():
            print(f"Warning: Neither python nor python3 found in {VENV_DIR}/bin/")
            print(f"Directories in virtual env: {list(VENV_DIR.glob('*'))}")
            if os.path.exists(f"{VENV_DIR}/bin"):
                print(f"Files in bin: {list(Path(f'{VENV_DIR}/bin').glob('*'))}")
        
        return str(python_path)

def get_venv_pip():
    """Get path to the virtual environment's pip executable."""
    if os.name == 'nt':  # Windows
        return str(VENV_DIR / "Scripts" / "pip.exe")
    else:  # Unix-like
        return str(VENV_DIR / "bin" / "pip")

def install_requirements():
    """Install requirements into the virtual environment."""
    if not REQUIREMENTS_FILE.exists():
        print(f"Error: Requirements file not found at {REQUIREMENTS_FILE}")
        return False
    
    print("Installing dependencies...")
    try:
        # Try using subprocess.run with stdout and stderr displayed
        result = subprocess.run(
            [get_venv_pip(), "install", "-r", str(REQUIREMENTS_FILE)],
            capture_output=False,  # Show output directly
            check=True  # Raise exception on error
        )
        print("Dependencies installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies (exit code: {e.returncode})")
        return False
    except Exception as e:
        print(f"Unexpected error during dependency installation: {e}")
        return False

def check_ollama():
    """Check if Ollama is installed and available."""
    try:
        # Try to find Ollama in PATH
        if os.name == 'nt':  # Windows
            # Check if ollama.exe is in PATH
            result = subprocess.run(
                ["where", "ollama"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return True
        else:  # Unix-like
            # Check if ollama is in PATH
            result = subprocess.run(
                ["which", "ollama"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return True
        
        return False
    except Exception:
        return False

def ensure_ollama_running():
    """Ensure Ollama is running, start it if needed."""
    print("\nChecking Ollama status...")
    
    # First, check if Ollama is installed
    if not check_ollama():
        print("❌ Ollama is not installed or not in PATH.")
        print("Please install Ollama from https://ollama.ai")
        return False
    
    # Try to list models to check if Ollama is running
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print("✅ Ollama is already running")
            return True
    except (subprocess.SubprocessError, FileNotFoundError):
        pass  # Ollama is not running or not found
    
    # If we're here, Ollama is installed but not running - try to start it
    print("Starting Ollama in the background...")
    
    try:
        # Start ollama serve in the background
        if os.name == 'nt':  # Windows
            subprocess.Popen(
                ["ollama", "serve"],
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:  # Unix-like
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
        
        # Wait a bit for Ollama to start
        print("Waiting for Ollama to start (this may take a few seconds)...")
        time.sleep(5)
        
        # Check if it started successfully
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print("✅ Ollama started successfully")
            return True
        else:
            print("⚠️ Failed to start Ollama automatically")
            print("Please start Ollama manually with 'ollama serve' in a separate terminal")
            return False
            
    except Exception as e:
        print(f"❌ Error starting Ollama: {e}")
        print("Please start Ollama manually with 'ollama serve' in a separate terminal")
        return False

def run_application(args):
    """Run the application inside the virtual environment."""
    if not args:
        # Default to run.py if no script is specified
        print("No script specified. Using default: run.py")
    
    script = args[0] if args else "run.py"
    script_path = BASE_DIR / script
    
    if not script_path.exists():
        print(f"Error: Script {script} not found")
        return False
        
    print(f"Running {script} with virtual environment...")
    
    # Get the Python executable path
    venv_python = get_venv_python()
    if not Path(venv_python).exists():
        print(f"Error: Python executable not found at {venv_python}")
        print("This might be due to the virtual environment not being set up correctly.")
        print("Checking for alternative Python executables...")
        
        # Try to find python3 if python doesn't exist
        venv_bin = VENV_DIR / "bin"
        if venv_bin.exists():
            python3_path = venv_bin / "python3"
            if python3_path.exists():
                venv_python = str(python3_path)
                print(f"Found alternative Python executable: {venv_python}")
            else:
                print("No alternative Python executable found in the virtual environment.")
                print("Please ensure your virtual environment is set up correctly.")
                return False
        else:
            print(f"Virtual environment bin directory not found at {venv_bin}")
            return False
    
    # Forward all arguments after the script name
    cmd = [venv_python, str(script_path)] + args[1:]
    try:
        result = subprocess.run(cmd)
        return result.returncode == 0
    except FileNotFoundError:
        print(f"Error: Could not execute {venv_python}")
        print("Please ensure your virtual environment is set up correctly.")
        return False
    except Exception as e:
        print(f"Error running application: {e}")
        return False

def print_env_info():
    """Print information about the environment."""
    venv_python = get_venv_python()
    
    print("\nEnvironment Information:")
    print(f"Virtual environment location: {VENV_DIR}")
    print(f"Python executable: {venv_python}")
    
    # Print Python version
    result = subprocess.run(
        [venv_python, "-V"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print(f"Python version: {result.stdout.strip()}")
    
    # Print installed packages
    print("\nInstalled packages:")
    subprocess.run([get_venv_pip(), "list"])
    
    # Check Ollama status
    print("\nOllama Status:")
    if check_ollama():
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                print("✅ Ollama is running")
                print("Available models:")
                # Skip the header line and print model names
                for line in result.stdout.strip().split('\n')[1:]:
                    if line:
                        parts = line.split()
                        if parts:
                            print(f"  - {parts[0]}")
            else:
                print("❌ Ollama is installed but not running")
        except:
            print("❌ Ollama is installed but not running")
    else:
        print("❌ Ollama is not installed or not in PATH")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="CellBot Setup and Run Script")
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Set up the virtual environment")
    setup_parser.add_argument("--start-ollama", action="store_true", help="Start Ollama if it's not running")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run the application")
    run_parser.add_argument("args", nargs="*", help="Arguments to pass to the application")
    run_parser.add_argument("--start-ollama", action="store_true", help="Start Ollama if it's not running")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show environment information")
    
    # Ollama command
    ollama_parser = subparsers.add_parser("ollama", help="Start Ollama if it's not running")
    
    args = parser.parse_args()
    
    if args.command == "setup" or not args.command:
        # Setup is the default if no command is specified
        created = create_venv()
        if created or not args.command:
            install_requirements()
        
        # Start Ollama if requested
        if hasattr(args, 'start_ollama') and args.start_ollama:
            ensure_ollama_running()
            
        print(f"\nSetup complete. You can now run CellBot with: {sys.executable} {__file__} run")
    
    elif args.command == "run":
        # Ensure venv exists
        if not VENV_DIR.exists():
            print("Virtual environment not found. Setting up...")
            create_venv()
            install_requirements()
        
        # Start Ollama if requested
        if hasattr(args, 'start_ollama') and args.start_ollama:
            ensure_ollama_running()
        
        # Run the application
        run_application(args.args)
    
    elif args.command == "info":
        if not VENV_DIR.exists():
            print("Virtual environment not found. Run setup first.")
            return
        print_env_info()
    
    elif args.command == "ollama":
        ensure_ollama_running()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 