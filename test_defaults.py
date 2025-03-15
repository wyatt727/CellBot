#!/usr/bin/env python3
"""
Test script to verify default token and temperature settings in CellBot
"""

import asyncio
import logging
import os
import sys
import json

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to the path so we can import from agent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def main():
    """Main function to test the defaults"""
    print("=== Testing CellBot Default Settings ===")
    
    # Import necessary modules
    try:
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
        
        # Make the request
        response = await get_llm_response_async(
            messages=messages,
            timeout=30  # Use a shorter timeout for testing
        )
        
        print(f"\nResponse received (length: {len(response)})")
        print(f"Response snippet: {response[:100]}...")
        
        # Check the model name from llm_client
        try:
            from agent.llm_client import LLM_MODEL
            print(f"\nLLM_CLIENT model: {LLM_MODEL}")
        except ImportError as e:
            print(f"Could not import LLM_MODEL: {e}")
        
        # Try to create an agent to check its defaults
        try:
            from agent.agent import MinimalAIAgent
            agent = MinimalAIAgent()
            print("\nAgent created with default settings:")
            print(f"Temperature: {agent.ollama_config.get('temperature')}")
            print(f"Token limit: {agent.ollama_config.get('num_predict')}")
            
            # Close aiohttp session
            if hasattr(agent, 'aiohttp_session'):
                await agent.aiohttp_session.close()
                print("Closed agent's aiohttp session")
        except Exception as e:
            print(f"Error creating agent: {e}")
            
    except ImportError as e:
        print(f"Import error: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 