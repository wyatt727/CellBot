#!/usr/bin/env python3
"""
Test script for Ollama on Android

This script tests Ollama functionality on Android devices,
especially within a virtual environment.
"""

import os
import sys
import subprocess
import asyncio
import aiohttp
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import the Android helper
try:
    from agent.android_helper import find_ollama_path, setup_android_environment
    
    # Set up environment variables
    env_vars = setup_android_environment()
    for key, value in env_vars.items():
        os.environ[key] = value
    
    print(f"Environment configured: OLLAMA_PATH={os.environ.get('OLLAMA_PATH', 'not set')}")
except ImportError as e:
    print(f"Warning: Android helper module not available: {e}")
    print("Will try to find Ollama using basic methods")
    
    def find_ollama_path():
        """Basic fallback to find Ollama path"""
        # Try to find ollama using the 'which' command
        try:
            result = subprocess.run(
                ["which", "ollama"], 
                capture_output=True, 
                text=True, 
                check=False
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception as e:
            print(f"Error running 'which ollama': {e}")
        
        return "ollama"  # Default fallback

async def test_ollama_cli():
    """Test Ollama CLI command"""
    print("\n--- Testing Ollama CLI ---")
    
    ollama_path = os.environ.get("OLLAMA_PATH", find_ollama_path())
    print(f"Using Ollama path: {ollama_path}")
    
    try:
        # First check if Ollama is running
        process = await asyncio.create_subprocess_exec(
            "pgrep", "ollama",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await process.communicate()
        
        if stdout:
            print("✅ Ollama process is running")
        else:
            print("❌ Ollama process is not running!")
            return
        
        # Try to list models
        process = await asyncio.create_subprocess_exec(
            ollama_path, "list",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            print("✅ Ollama CLI command successful!")
            print("\nAvailable models (first few lines):")
            lines = stdout.decode().strip().split("\n")
            for line in lines[:3]:  # Show only first 3 lines
                print(line)
            if len(lines) > 3:
                print(f"... and {len(lines)-3} more models")
        else:
            print(f"❌ Ollama CLI error: {stderr.decode()}")
    except Exception as e:
        print(f"❌ Exception running Ollama CLI: {e}")

async def test_ollama_api():
    """Test Ollama API access"""
    print("\n--- Testing Ollama API ---")
    
    try:
        ollama_api_base = os.environ.get("OLLAMA_API_BASE", "http://127.0.0.1:11434")
        print(f"Using API base: {ollama_api_base}")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{ollama_api_base}/api/tags") as response:
                if response.status == 200:
                    print("✅ Successfully connected to Ollama API")
                    models_data = await response.json()
                    if models_data.get("models"):
                        print(f"API reports {len(models_data['models'])} available models")
                else:
                    print(f"❌ Ollama API error: {response.status}")
                    error_text = await response.text()
                    print(f"Error message: {error_text}")
    except Exception as e:
        print(f"❌ Failed to connect to Ollama API: {e}")

async def main():
    """Main test function"""
    print("=== Testing Ollama on Android ===")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Virtual env: {'VIRTUAL_ENV' in os.environ}")
    print(f"Current PATH: {os.environ.get('PATH', 'Not set')}")
    
    # Run the tests
    await test_ollama_cli()
    await test_ollama_api()
    
    print("\n=== Test completed ===")

if __name__ == "__main__":
    asyncio.run(main()) 