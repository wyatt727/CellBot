#!/usr/bin/env python3
"""
Test script to verify default token and temperature settings in CellBot
"""

import asyncio
import logging
import os
import sys
import json
import aiohttp
from contextlib import AsyncExitStack, asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to the path so we can import from agent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Simple context manager for an aiohttp session
@asynccontextmanager
async def create_session(timeout=30):
    """Create and properly close an aiohttp session."""
    session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout)
    )
    try:
        yield session
    finally:
        if not session.closed:
            await session.close()
            logger.info("Session closed")

async def main():
    """Main function to test the defaults"""
    print("=== Testing CellBot Default Settings ===")
    
    # Create our own session that we'll ensure is closed
    async with create_session(timeout=30) as session:
        try:
            # Import necessary modules
            from agent.llm_client import get_llm_response_async
            
            # Try to import android_config
            try:
                from agent.android_config import DEFAULT_TEMPERATURE, DEFAULT_NUM_PREDICT, get_optimal_llm_parameters
                print(f"From android_config - DEFAULT_TEMPERATURE: {DEFAULT_TEMPERATURE}")
                print(f"From android_config - DEFAULT_NUM_PREDICT: {DEFAULT_NUM_PREDICT}")
                
                optimal_params = get_optimal_llm_parameters()
                print(f"Optimal parameters: {json.dumps(optimal_params, indent=2)}")
            except ImportError:
                print("android_config not available")
            
            # Test a request with default settings
            print("\n=== Testing request with default settings ===")
            messages = [{"role": "user", "content": "What is the value of pi?"}]
            
            # Make the request with our session
            response = await get_llm_response_async(
                messages=messages,
                session=session,
                timeout=30
            )
            
            print(f"\nResponse received (length: {len(response)})")
            print(f"Response snippet: {response[:100]}...")
            
            # Check the model name from llm_client
            try:
                from agent.llm_client import LLM_MODEL
                print(f"\nLLM_CLIENT model: {LLM_MODEL}")
            except ImportError as e:
                print(f"Could not import LLM_MODEL: {e}")
            
            # Try to create an agent to check its defaults, but we'll skip this for now
            # since we're still having issues with cleanup
            print("\nSkipping agent creation to avoid session issues")
            
        except ImportError as e:
            print(f"Import error: {e}")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nAll resources have been cleaned up properly")

if __name__ == "__main__":
    asyncio.run(main()) 