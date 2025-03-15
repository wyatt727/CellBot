#!/usr/bin/env python3
"""
Timeout Diagnostic Tool for CellBot

This script tests Ollama API response times and timeout behavior
to help diagnose issues with CellBot's timeout handling.
"""

import asyncio
import aiohttp
import time
import logging
import os
import json
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default settings
OLLAMA_API_BASE = os.environ.get("OLLAMA_API_BASE", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "mistral:7b")

async def test_ollama_request(model, timeout, message="Hello, how are you?", thread_count=None):
    """Test a simple request to Ollama API with the specified timeout."""
    logger.info(f"Testing Ollama request with timeout={timeout}s")
    
    # Prepare the request
    options = {}
    if thread_count:
        options["num_thread"] = thread_count
    
    messages = [{"role": "user", "content": message}]
    request_body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options
    }
    
    # Create a session with the specified timeout
    start_time = time.time()
    session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout)
    )
    
    try:
        # Make the request with explicit timeout in the post
        logger.info(f"Sending request to {OLLAMA_API_BASE}/api/chat")
        logger.info(f"Using model: {model}")
        logger.info(f"Using ClientSession timeout: {timeout}s")
        
        async with session.post(
            f"{OLLAMA_API_BASE}/api/chat",
            json=request_body,
            timeout=aiohttp.ClientTimeout(total=timeout)  # Explicit timeout in post request
        ) as response:
            # Check response status
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"Error status {response.status}: {error_text}")
                return False, f"Error: {response.status} - {error_text}"
            
            # Parse the response
            try:
                data = await response.json()
                response_time = time.time() - start_time
                response_message = data.get("message", {}).get("content", "")
                
                logger.info(f"Request completed in {response_time:.2f} seconds")
                return True, {
                    "response_time": response_time,
                    "response_length": len(response_message),
                    "response_start": response_message[:50] + "..." if len(response_message) > 50 else response_message
                }
            except json.JSONDecodeError as e:
                raw_response = await response.text()
                logger.error(f"Error parsing response: {e}")
                logger.error(f"Raw response: {raw_response[:500]}...")
                return False, f"Error parsing response: {e}"
    
    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        logger.error(f"Request timed out after {elapsed:.2f} seconds (timeout was set to {timeout}s)")
        return False, f"Timeout after {elapsed:.2f}s (limit: {timeout}s)"
    
    except aiohttp.ClientError as e:
        logger.error(f"Client error: {e}")
        return False, f"Client error: {e}"
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False, f"Error: {e}"
    
    finally:
        await session.close()

async def run_timeout_tests(model, message, thread_count):
    """Run a series of tests with different timeout values."""
    results = []
    
    # Test the connection first
    logger.info("Testing Ollama server connection...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{OLLAMA_API_BASE}/api/tags") as response:
                if response.status != 200:
                    logger.error(f"Failed to connect to Ollama API: {response.status}")
                    return False
                
                models_data = await response.json()
                available_models = [model["name"] for model in models_data.get("models", [])]
                
                if not available_models:
                    logger.warning("No models found in Ollama")
                else:
                    logger.info(f"Available models: {', '.join(available_models)}")
                
                if model not in available_models:
                    logger.warning(f"Model '{model}' not found in available models")
    except Exception as e:
        logger.error(f"Error checking Ollama server: {e}")
        logger.error(f"Please make sure Ollama is running with 'ollama serve'")
        return False
    
    # Test with different timeouts
    timeouts = [10, 30, 60, 120]
    for timeout in timeouts:
        logger.info(f"\n--- Testing with {timeout}s timeout ---")
        success, result = await test_ollama_request(model, timeout, message, thread_count)
        results.append({
            "timeout": timeout,
            "success": success,
            "result": result
        })
        
        # Add a small delay between tests
        await asyncio.sleep(1)
    
    # Print summary
    print("\n\n=== SUMMARY OF RESULTS ===")
    for result in results:
        status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
        print(f"Timeout {result['timeout']}s: {status}")
        if result["success"]:
            data = result["result"]
            print(f"  Response time: {data['response_time']:.2f}s")
            print(f"  Response length: {data['response_length']} chars")
        else:
            print(f"  Error: {result['result']}")
        print("")
    
    return True

async def main():
    parser = argparse.ArgumentParser(description="Test CellBot timeout behavior")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--message", default="Explain the concept of timeouts in APIs in three sentences.", 
                       help="Message to send to the model")
    parser.add_argument("--threads", type=int, help="Number of threads to use (default: auto)")
    args = parser.parse_args()
    
    print(f"=== CellBot Timeout Diagnostic Tool ===")
    print(f"Testing with model: {args.model}")
    print(f"API endpoint: {OLLAMA_API_BASE}")
    print(f"Thread count: {args.threads if args.threads else 'auto'}")
    print(f"Message: '{args.message}'")
    print("=" * 40 + "\n")
    
    await run_timeout_tests(args.model, args.message, args.threads)

if __name__ == "__main__":
    asyncio.run(main()) 