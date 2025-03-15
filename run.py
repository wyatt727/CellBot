#!/usr/bin/env python3
# run.py - A simple runner for CellBot without database
import asyncio
import sys
import os
import subprocess
import time

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Check if Ollama is running and start it if needed
def ensure_ollama_running():
    """Check if Ollama is running and start it if needed."""
    print("Checking if Ollama is running...")
    
    # Get the Ollama path from environment or use default
    ollama_path = os.environ.get("OLLAMA_PATH", "ollama")
    
    try:
        # Try to list models to check if Ollama is running
        result = subprocess.run(
            [ollama_path, "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print("✅ Ollama is already running")
            return True
    except (subprocess.SubprocessError, FileNotFoundError):
        print("❌ Ollama is not running or not found")
    
    # If we're here, Ollama is not running - try to start it
    print("Attempting to start Ollama in the background...")
    
    try:
        # Start ollama serve in the background
        if os.name == 'nt':  # Windows
            subprocess.Popen(
                [ollama_path, "serve"],
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:  # Unix-like
            subprocess.Popen(
                [ollama_path, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
        
        # Wait a bit for Ollama to start
        print("Waiting for Ollama to start (this may take a few seconds)...")
        time.sleep(5)
        
        # Check if it started successfully
        result = subprocess.run(
            [ollama_path, "list"],
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

# Import Android helper and set up environment
try:
    from agent.android_helper import setup_android_environment
    
    # Configure environment for Android if needed
    env_vars = setup_android_environment()
    for key, value in env_vars.items():
        os.environ[key] = value
    print(f"Environment configured: OLLAMA_PATH={os.environ.get('OLLAMA_PATH', 'not set')}")
except ImportError as e:
    print(f"Warning: Android helper module not available: {e}")
    
from agent import MinimalAIAgent

async def main():
    """Run CellBot with in-memory storage only (no database)."""
    print("Starting CellBot with in-memory storage only (no database)")
    
    # Ensure Ollama is running before starting the agent
    ensure_ollama_running()
    
    # Create the agent with default parameters
    agent = MinimalAIAgent()
    
    # Run the agent
    await agent.run()

if __name__ == "__main__":
    # Run the asyncio event loop
    asyncio.run(main()) 