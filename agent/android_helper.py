"""
Android Helper Module

Provides utility functions to help CellBot run more smoothly on Android devices.
This includes finding binary paths and managing environment variables.
"""

import os
import subprocess
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

def find_ollama_path():
    """
    Find the path to the Ollama binary on the system.
    
    Returns:
        str: The full path to the Ollama binary, or 'ollama' if not found
    """
    # First check if OLLAMA_PATH is already set
    if "OLLAMA_PATH" in os.environ:
        logger.info(f"Using OLLAMA_PATH from environment: {os.environ['OLLAMA_PATH']}")
        return os.environ["OLLAMA_PATH"]
    
    # Try to find ollama using the 'which' command
    try:
        result = subprocess.run(
            ["which", "ollama"], 
            capture_output=True, 
            text=True, 
            check=False
        )
        if result.returncode == 0 and result.stdout.strip():
            ollama_path = result.stdout.strip()
            logger.info(f"Found Ollama using 'which': {ollama_path}")
            os.environ["OLLAMA_PATH"] = ollama_path
            return ollama_path
    except Exception as e:
        logger.warning(f"Error running 'which ollama': {e}")
    
    # Check common locations
    common_paths = [
        "/usr/local/bin/ollama",
        "/usr/bin/ollama",
        "/data/data/com.termux/files/usr/bin/ollama",
        os.path.expanduser("~/bin/ollama")
    ]
    
    for path in common_paths:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            logger.info(f"Found Ollama in common location: {path}")
            os.environ["OLLAMA_PATH"] = path
            return path
    
    # Try to use shutil.which
    try:
        path = shutil.which("ollama")
        if path:
            logger.info(f"Found Ollama using shutil.which: {path}")
            os.environ["OLLAMA_PATH"] = path
            return path
    except Exception as e:
        logger.warning(f"Error using shutil.which: {e}")
    
    # If we couldn't find it, return the default
    logger.warning("Could not find Ollama path, using default 'ollama'")
    return "ollama"

def setup_android_environment():
    """
    Set up the environment for Android.
    """
    # Find and set up the Ollama path
    ollama_path = find_ollama_path()
    
    # Return a dictionary of environment variables to set
    return {
        "OLLAMA_PATH": ollama_path
    } 