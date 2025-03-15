# agent/__init__.py
# __version__ = "1.0.0"
# print("Initializing the tests/ package")

"""
CellBot AI Agent Package

This package contains the core agent functionality for the CellBot AI assistant.
"""

import os

# Try to import Android helper and configure environment first
try:
    from .android_helper import setup_android_environment
    
    # Set up environment variables for Android if needed
    env_vars = setup_android_environment()
    for key, value in env_vars.items():
        os.environ[key] = value
except ImportError:
    # Not a problem - the helper is optional
    pass

# Import the main agent class to expose it at the package level
from .agent import MinimalAIAgent

# Define what gets exported when using "from agent import *"
__all__ = ["MinimalAIAgent"]