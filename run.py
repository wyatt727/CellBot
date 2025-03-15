#!/usr/bin/env python3
# run.py - A simple runner for CellBot without database
import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
    
    # Create the agent with default parameters
    agent = MinimalAIAgent()
    
    # Run the agent
    await agent.run()

if __name__ == "__main__":
    # Run the asyncio event loop
    asyncio.run(main()) 