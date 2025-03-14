# agent/system_prompt.py
import os
import logging
import aiofiles
import time
import re
from typing import Optional

# Try to import from android_config first, fall back to regular config
try:
    from .android_config import SYSTEM_PROMPT_FILE
    is_android = True
except ImportError:
    try:
        from .config import SYSTEM_PROMPT_FILE
        is_android = False
    except ImportError:
        # Fallback to safe defaults
        import os
        SYSTEM_PROMPT_FILE = os.path.join(os.path.expanduser("~"), "nethunter_cellbot", "system-prompt.txt")
        is_android = False

logger = logging.getLogger(__name__)

# Cache for the system prompt
_system_prompt_cache = {"content": None, "timestamp": 0, "file_mtime": 0}

async def get_system_prompt(user_message: Optional[str] = None) -> str:
    """
    Asynchronously get the system prompt, with caching.
    The prompt is cached to avoid re-reading the file on every request.
    If the file has been modified since the last read, the cache is invalidated.
    
    Args:
        user_message: Optional user message to include in context
    
    Returns:
        The system prompt as a string
    """
    current_time = time.time()
    
    # Check if the system prompt is cached and cache is still valid
    if _system_prompt_cache["content"] is not None:
        try:
            # Check if the file has been modified since the last read
            file_mtime = os.path.getmtime(SYSTEM_PROMPT_FILE)
            if file_mtime <= _system_prompt_cache["file_mtime"]:
                # Cache is still valid
                return _system_prompt_cache["content"]
                
            # File has been modified, invalidate cache
            logger.info("System prompt file has been modified, reloading")
        except FileNotFoundError:
            logger.warning(f"System prompt file not found: {SYSTEM_PROMPT_FILE}")
            return "You are a helpful AI assistant. Answer user questions concisely."
        except Exception as e:
            logger.error(f"Error checking system prompt file modification time: {e}")
            # Use cached version if available
            if _system_prompt_cache["content"] is not None:
                return _system_prompt_cache["content"]
            return "You are a helpful AI assistant. Answer user questions concisely."
    
    # Cache is invalid or not set, load the system prompt
    try:
        async with aiofiles.open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
            prompt = await f.read()
        
        # Update cache
        _system_prompt_cache["content"] = prompt.strip()
        _system_prompt_cache["timestamp"] = current_time
        _system_prompt_cache["file_mtime"] = os.path.getmtime(SYSTEM_PROMPT_FILE)
        
        logger.info("System prompt loaded successfully")
        return _system_prompt_cache["content"]
    
    except FileNotFoundError:
        logger.warning(f"System prompt file not found: {SYSTEM_PROMPT_FILE}")
        # Create default system prompt if on Android
        if is_android:
            # Create a default system prompt for Android
            default_prompt = (
                "You are a concise Kali Nethunter expert running on a rooted OnePlus 12 running OxygenOS 15. "
                "Single code block responses only. No intro or outro. Context-driven. Examples provided.\n\n"
                "Example (sh request):\n"
                "User: create 2 directories named 2 and 3, then list all files in this directory.\n\n"
                "A:\n"
                "```sh\n"
                "mkdir -p 2 3 && ls -la\n"
                "```"
            )
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(SYSTEM_PROMPT_FILE), exist_ok=True)
                # Write default prompt
                async with aiofiles.open(SYSTEM_PROMPT_FILE, "w", encoding="utf-8") as f:
                    await f.write(default_prompt)
                logger.info(f"Created default system prompt file: {SYSTEM_PROMPT_FILE}")
                
                # Update cache
                _system_prompt_cache["content"] = default_prompt
                _system_prompt_cache["timestamp"] = current_time
                _system_prompt_cache["file_mtime"] = os.path.getmtime(SYSTEM_PROMPT_FILE)
                
                return default_prompt
            except Exception as e:
                logger.error(f"Error creating default system prompt file: {e}")
        
        return "You are a helpful AI assistant for Kali NetHunter. Answer user questions concisely."
    
    except Exception as e:
        logger.error(f"Error loading system prompt: {e}")
        return "You are a helpful AI assistant. Answer user questions concisely."
