"""
Session manager utilities for aiohttp ClientSessions.
Handles proper creation and cleanup of sessions to prevent resource leaks.
"""

import aiohttp
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

logger = logging.getLogger(__name__)

@asynccontextmanager
async def managed_session(timeout: Optional[int] = 60):
    """
    Context manager for aiohttp ClientSession that ensures proper cleanup.
    
    Usage:
        async with managed_session(timeout=30) as session:
            async with session.get(url) as response:
                # do something with response
    
    Args:
        timeout: Timeout in seconds or None for no timeout
        
    Yields:
        aiohttp.ClientSession: A properly configured session
    """
    # Create the session with proper timeout config
    timeout_config = None if timeout is None else aiohttp.ClientTimeout(total=timeout)
    session = aiohttp.ClientSession(timeout=timeout_config)
    
    try:
        logger.debug(f"Created managed session with timeout: {timeout}")
        yield session
    finally:
        # Ensure the session is closed properly, even if an exception occurs
        if not session.closed:
            await session.close()
            logger.debug("Managed session closed")

async def ensure_session_closed(session: Optional[aiohttp.ClientSession]) -> None:
    """
    Ensure that a session is properly closed if it exists and isn't already closed.
    This is useful for cleanup operations.
    
    Args:
        session: The session to close or None
    """
    if session is not None and not session.closed:
        try:
            await session.close()
            logger.debug("Session closed successfully")
        except Exception as e:
            logger.error(f"Error closing session: {e}") 