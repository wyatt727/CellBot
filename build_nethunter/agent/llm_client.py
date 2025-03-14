# agent/llm_client.py
import aiohttp
import logging
import json
import os
import asyncio
from typing import List, Dict, Any, Optional

# Try to import from android_config first, fall back to regular config
try:
    from .android_config import API_URL as LLM_API_BASE, DEFAULT_MODEL as LLM_MODEL
    logging.info("Using Android configuration")
except ImportError:
    # Fallback to environment variables with safe defaults
    LLM_API_BASE = os.environ.get("OLLAMA_API_BASE", "http://127.0.0.1:11434")
    LLM_MODEL = os.environ.get("LLM_MODEL", "mistral:7b")
    logging.info("Using standard configuration")

logger = logging.getLogger(__name__)

async def get_llm_response_async(
    messages: List[Dict[str, str]],
    model: str = LLM_MODEL,
    session: Optional[aiohttp.ClientSession] = None,
    num_thread: int = None,
    num_gpu: int = None,
    timeout: int = 120
) -> str:
    """
    Asynchronously get a response from the LLM API.
    Optimized for mobile with timeout handling and network error recovery.
    
    Args:
        messages: List of message dictionaries (each with 'role' and 'content')
        model: The name of the model to use
        session: Optional aiohttp.ClientSession
        num_thread: Number of CPU threads to use (optional)
        num_gpu: Number of GPU layers to use (optional)
        timeout: Request timeout in seconds
        
    Returns:
        The LLM response text
    """
    own_session = False
    
    # If no session is provided, create one
    if session is None:
        own_session = True
        session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout if timeout else 120)
        )
    
    try:
        # Prepare request body
        options = {}
        if num_thread is not None:
            options["num_thread"] = num_thread
        if num_gpu is not None:
            options["num_gpu"] = num_gpu
            
        # For mobile, add additional options to optimize for low resources
        options["temperature"] = 0.7  # Lower temp = more predictable responses
        options["num_predict"] = 1024  # Limit response length to save resources
            
        request_body = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options
        }
        
        # Set up retry logic
        max_retries = 3
        retry_delay = 2  # Start with 2 seconds delay
        
        for attempt in range(max_retries):
            try:
                async with session.post(
                    f"{LLM_API_BASE}/api/chat",
                    json=request_body,
                    timeout=aiohttp.ClientTimeout(total=timeout if timeout else 120)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        try:
                            error_json = json.loads(error_text)
                            error_message = error_json.get("error", error_text)
                        except json.JSONDecodeError:
                            error_message = error_text
                            
                        logger.error(f"LLM API error (status {response.status}): {error_message}")
                        
                        if response.status == 429:  # Rate limit
                            if attempt < max_retries - 1:
                                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                                logger.info(f"Rate limited. Retrying in {wait_time}s...")
                                await asyncio.sleep(wait_time)
                                continue
                                
                        if response.status == 503 or response.status == 502:  # Service unavailable
                            if attempt < max_retries - 1:
                                wait_time = retry_delay * (2 ** attempt)
                                logger.info(f"Service unavailable. Retrying in {wait_time}s...")
                                await asyncio.sleep(wait_time)
                                continue
                                
                        raise Exception(f"LLM API error: {error_message}")
                    
                    try:
                        data = await response.json()
                        response_message = data.get("message", {}).get("content", "")
                        return response_message
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing LLM response: {e}")
                        raw_response = await response.text()
                        logger.error(f"Raw response: {raw_response[:500]}...")
                        raise Exception(f"Failed to parse LLM response: {e}")
            
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(f"Request timed out. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise TimeoutError(f"LLM request timed out after {timeout}s and {max_retries} retries")
                    
            except aiohttp.ClientError as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(f"Network error: {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise Exception(f"Network error after {max_retries} retries: {e}")
                    
    finally:
        # Close the session if we created it
        if own_session:
            await session.close()
            
    # This code should never be reached due to returns or exceptions above
    raise Exception("Unexpected code path in get_llm_response_async")

async def stream_llm_response(context: list, model: str, session: aiohttp.ClientSession):
    """
    An asynchronous generator that yields response chunks as they are received from the LLM API.
    This allows the caller to update the UI incrementally.
    """
    payload = {
        "model": model,
        "messages": context,
        "stream": True
    }
    url = f"{LLM_API_BASE}/api/chat"
    async with session.post(url, json=payload) as resp:
        resp.raise_for_status()
        async for chunk in resp.content.iter_any():
            yield chunk.decode()
