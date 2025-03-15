#!/usr/bin/env python3
# run.py - A simple runner for CellBot without database
import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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