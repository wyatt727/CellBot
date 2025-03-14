# agent/code_executor.py
import os
import asyncio
import logging
from datetime import datetime
import aiofiles
import re
import sys
import tempfile

# Import standard config with fallback to android config
try:
    from .config import GENERATED_CODE_DIR, SAVE_CODE_BLOCKS
except ImportError:
    try:
        from .android_config import GENERATED_CODE_DIR
        SAVE_CODE_BLOCKS = False
    except ImportError:
        # Fallback to safe defaults
        GENERATED_CODE_DIR = os.path.join(os.path.expanduser("~"), "nethunter_cellbot", "generated_code")
        SAVE_CODE_BLOCKS = False
        os.makedirs(GENERATED_CODE_DIR, exist_ok=True)

logger = logging.getLogger(__name__)

def extract_code_blocks(response: str) -> list:
    """
    Extracts code blocks from the LLM response.
    Assumes code blocks are enclosed in triple backticks.
    """
    code_blocks = re.findall(r'```(.*?)\n(.*?)```', response, re.DOTALL)
    return [(lang.strip(), code.strip()) for lang, code in code_blocks]

async def execute_code_async(language: str, code: str, timeout: int = 300) -> (int, str):
    """
    Asynchronously execute a code block.
    Uses aiofiles for file I/O and asyncio.create_subprocess_exec for subprocess management.
    
    Supports both desktop and mobile environments.
    """
    try:
        is_android = 'ANDROID_ROOT' in os.environ or os.path.exists('/system/bin/adb')
        
        # Set up file path
        if SAVE_CODE_BLOCKS:
            os.makedirs(GENERATED_CODE_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = ".py" if language.lower() == "python" else ".sh"
            filename = os.path.join(GENERATED_CODE_DIR, f"generated_{timestamp}{ext}")
            async with aiofiles.open(filename, "w", encoding="utf-8") as f:
                await f.write(code.strip() + "\n")
        else:
            # Create a temporary file that will be automatically cleaned up
            with tempfile.NamedTemporaryFile(mode='w', suffix=".py" if language.lower() == "python" else ".sh", delete=False) as temp_file:
                temp_file.write(code.strip() + "\n")
                filename = temp_file.name
        
        # Set execution permissions on Android
        if is_android and language.lower() != "python":
            os.chmod(filename, 0o755)
        
        # Prepare command based on language and platform
        if language.lower() == "python":
            cmd = [sys.executable, filename]
        elif is_android:
            # On Android/NetHunter, use sh instead of bash which might not be available
            cmd = ["sh", filename]
        else:
            # On desktop systems, prefer bash if available
            cmd = ["bash" if os.path.exists("/bin/bash") else "sh", filename]
        
        # Execute command
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return -1, f"[{language.capitalize()} Error] Execution timed out after {timeout} seconds."
        
        # Handle output
        output = (stdout.decode() + stderr.decode()).strip()
        
        # Clean up temporary file if not saving code blocks
        if not SAVE_CODE_BLOCKS:
            try:
                os.unlink(filename)
            except OSError as e:
                logger.warning(f"Failed to remove temp file: {e}")
                
        return proc.returncode, output
        
    except Exception as e:
        logger.error(f"Error executing {language} code: {str(e)}")
        # Clean up temporary file if not saving code blocks
        if not SAVE_CODE_BLOCKS and 'filename' in locals():
            try:
                os.unlink(filename)
            except OSError:
                pass
        return -1, f"[{language.capitalize()} Error] Exception: {e}"
