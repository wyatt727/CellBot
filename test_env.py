#!/usr/bin/env python3
"""
CellBot Environment Test Script

This script checks if the environment is correctly set up for running CellBot.
It verifies:
1. Python version
2. Virtual environment
3. Required dependencies
4. File permissions
"""

import os
import sys
import platform
import subprocess
import importlib.util
from pathlib import Path


def print_header(title):
    """Print a section header."""
    print("\n" + "=" * 50)
    print(f" {title}")
    print("=" * 50)


def print_result(test, result, message=""):
    """Print a test result."""
    if result:
        print(f"✓ {test}: {message}" if message else f"✓ {test}")
    else:
        print(f"✗ {test}: {message}" if message else f"✗ {test}")
    return result


def check_python_version():
    """Check if Python version is compatible."""
    print_header("Python Environment")
    
    version = sys.version_info
    print(f"Python version: {platform.python_version()}")
    
    # Python 3.10+ is recommended
    result = version.major == 3 and version.minor >= 8
    return print_result("Python version compatible", result, 
                      "Python 3.8+ required" if not result else "")


def check_venv():
    """Check if running in a virtual environment."""
    in_venv = hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    venv_result = print_result("Virtual environment", in_venv, 
                             "Not running in a virtual environment" if not in_venv else f"Using {sys.prefix}")
    
    # Check if our .venv directory exists
    base_dir = Path(__file__).resolve().parent
    venv_dir = base_dir / ".venv"
    venv_exists = venv_dir.exists() and venv_dir.is_dir()
    venv_exists_result = print_result("Virtual environment directory", venv_exists,
                                    f"Virtual environment directory not found at {venv_dir}" if not venv_exists else f"Found at {venv_dir}")
    
    return venv_exists


def check_dependencies():
    """Check if required dependencies are installed."""
    print_header("Dependencies")
    
    required_packages = [
        "aiohttp", "aiofiles", "psutil", "markdown", 
        "diskcache", "urllib3", "argparse"
    ]
    
    all_found = True
    for package in required_packages:
        found = importlib.util.find_spec(package) is not None
        all_found = all_found and found
        print_result(f"Package {package}", found, "Not installed" if not found else "")
    
    return all_found


def check_file_permissions():
    """Check if important files have correct permissions."""
    print_header("File Permissions")
    
    base_dir = Path(__file__).resolve().parent
    files_to_check = [
        base_dir / "setup.py",
        base_dir / "cellbot.py",
        base_dir / "run.py"
    ]
    
    all_executable = True
    for file_path in files_to_check:
        if file_path.exists():
            is_executable = os.access(file_path, os.X_OK)
            all_executable = all_executable and is_executable
            print_result(f"File {file_path.name}", is_executable, 
                       f"Not executable. Run 'chmod +x {file_path.name}'" if not is_executable else "Executable")
        else:
            all_executable = False
            print_result(f"File {file_path.name}", False, "File not found")
    
    return all_executable


def check_running_application():
    """Try running the application with --help."""
    print_header("Application Test")
    
    base_dir = Path(__file__).resolve().parent
    setup_script = base_dir / "setup.py"
    
    try:
        result = subprocess.run(
            [sys.executable, str(setup_script), "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        success = result.returncode == 0
        print_result("Running setup.py --help", success, 
                   result.stdout.strip() if success else result.stderr.strip())
        return success
    except Exception as e:
        print_result("Running setup.py --help", False, str(e))
        return False


def main():
    """Run all tests and provide a summary."""
    print("CellBot Environment Test")
    print("------------------------")
    print(f"Running tests from: {Path(__file__).resolve().parent}")
    
    # Run all tests
    python_ok = check_python_version()
    venv_ok = check_venv()
    deps_ok = check_dependencies()
    perms_ok = check_file_permissions()
    app_ok = check_running_application()
    
    # Print summary
    print_header("Summary")
    total_tests = 5
    passed_tests = sum([python_ok, venv_ok, deps_ok, perms_ok, app_ok])
    print(f"Passed {passed_tests}/{total_tests} tests")
    
    if passed_tests == total_tests:
        print("\n✓ Environment is ready to run CellBot!")
        print("\nTry running:")
        print("  ./cellbot.py")
    else:
        print("\n✗ Some tests failed. Please fix the issues before running CellBot.")
        
        if not venv_ok:
            print("\nTo set up the virtual environment:")
            print("  ./setup.py setup")
        
        if not deps_ok:
            print("\nTo install dependencies:")
            print("  ./setup.py setup")
            print("  # or directly:")
            print("  .venv/bin/pip install -r requirements.txt")
        
        if not perms_ok:
            print("\nTo fix file permissions:")
            print("  chmod +x setup.py cellbot.py run.py")
    
    return 0 if passed_tests == total_tests else 1


if __name__ == "__main__":
    sys.exit(main()) 