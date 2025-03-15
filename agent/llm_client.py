# agent/llm_client.py
import aiohttp
import logging
import json
import os
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

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
    temperature: float = None,
    num_predict: int = None,
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
        temperature: Temperature for response sampling (optional)
        num_predict: Maximum number of tokens to generate (optional)
        timeout: Request timeout in seconds
        
    Returns:
        The LLM response text
    """
    own_session = False
    logger.info(f"get_llm_response_async called with timeout: {timeout}")  # Added logging
    
    # If no session is provided, create one
    if session is None:
        own_session = True
        session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout if timeout else 60)  # Default to 60s
        )
        logger.info(f"Created new session with timeout: {timeout if timeout else 60}s")
    else:
        # Log the timeout of the provided session
        if hasattr(session, '_timeout') and session._timeout:
            logger.info(f"Using existing session with timeout: {session._timeout.total}")
        else:
            logger.info("Using existing session with unknown timeout")
    
    try:
        # Prepare request body
        options = {}
        if num_thread is not None:
            options["num_thread"] = num_thread
        if num_gpu is not None:
            options["num_gpu"] = num_gpu
        if temperature is not None:
            options["temperature"] = temperature
        if num_predict is not None:
            options["num_predict"] = num_predict
            
        # For mobile, add default options only if not already set
        if "temperature" not in options:
            options["temperature"] = 0.3  # Lower temp = more deterministic responses
        if "num_predict" not in options:
            options["num_predict"] = 500  # Limit response length to save resources
        if "num_thread" not in options and not num_thread:
            # Use maximum available CPU threads for best performance
            import multiprocessing
            options["num_thread"] = multiprocessing.cpu_count()  # Use all available cores
            
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
                # IMPORTANT: Don't create a new ClientTimeout here, as it overrides the session timeout
                # Instead, use the timeout from the session that was provided or created
                logger.info(f"Sending request (attempt {attempt+1}/{max_retries})")
                request_start = datetime.now()
                
                async with session.post(
                    f"{LLM_API_BASE}/api/chat",
                    json=request_body
                ) as response:
                    request_duration = (datetime.now() - request_start).total_seconds()
                    logger.info(f"Got response in {request_duration:.2f}s with status {response.status}")
                    
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
                                
                        # Handle broken pipe errors (common on mobile devices with limited resources)
                        if response.status == 500 and "broken pipe" in error_message.lower():
                            logger.error(f"Broken pipe error detected, likely due to resource constraints")
                            # Provide more comprehensive error guidance
                            raise Exception(f"LLM API error: {error_message}\n\nThis error can occur for several reasons:\n1. Resource constraints - the model may be using too much memory\n2. Network connectivity issues\n3. Ollama server instability\n\nTry these fixes:\n- Restart Ollama with 'ollama serve' in a separate terminal\n- Use shorter prompts or break down complex questions\n- Close other resource-intensive apps")
                        
                        # Handle general Ollama server issues
                        if response.status == 500 or "ollama" in error_message.lower():
                            if attempt < max_retries - 1:
                                wait_time = retry_delay * (2 ** attempt)
                                logger.warning(f"Ollama server error. Attempting recovery in {wait_time}s...")
                                await asyncio.sleep(wait_time)
                                continue
                        
                        raise Exception(f"LLM API error: {error_message}")
                    
                    try:
                        data = await response.json()
                        response_message = data.get("message", {}).get("content", "")
                        logger.info(f"Successfully extracted response of {len(response_message)} chars")
                        return response_message
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing LLM response: {e}")
                        raw_response = await response.text()
                        logger.error(f"Raw response: {raw_response[:500]}...")
                        raise Exception(f"Failed to parse LLM response: {e}")
            
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(f"Request timed out after attempt {attempt+1}. Retrying in {wait_time}s...")
                    logger.warning(f"Timeout setting was: {timeout}s")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"LLM request timed out. Attempt {attempt+1} of {max_retries}. Timeout setting: {timeout}s")
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
            logger.info("Closed own session")
            
    # This code should never be reached due to returns or exceptions above
    raise Exception("Unexpected code path in get_llm_response_async")

async def stream_llm_response(context: list, model: str, session: Optional[aiohttp.ClientSession] = None):
    """
    An asynchronous generator that yields response chunks as they are received from the LLM API.
    This allows the caller to update the UI incrementally.
    
    Args:
        context: List of message dictionaries
        model: Model name to use
        session: Optional aiohttp.ClientSession. If None, a new session is created and properly closed.
    """
    own_session = False
    
    # Create a session if one wasn't provided
    if session is None:
        own_session = True
        session = aiohttp.ClientSession()
        logger.info("Created new session for streaming")
    
    try:
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
                
    finally:
        # Close the session if we created it
        if own_session and session and not session.closed:
            await session.close()
            logger.info("Closed streaming session")
