# agent/agent.py
import sys
import asyncio
import aiohttp
import subprocess
import logging
import readline
import os
import textwrap
import shutil
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict
# Remove MockConversationDB import
from .system_prompt import get_system_prompt
from .llm_client import get_llm_response_async, LLM_API_BASE
from .code_executor import execute_code_async, extract_code_blocks
from .config import (
    LLM_MODEL, MAX_CONCURRENT_LLM_CALLS, 
    MAX_CONCURRENT_CODE_EXECUTIONS, RESPONSE_TIMEOUT,
    SIMILARITY_THRESHOLD, DEBUG_MODE, SAVE_CODE_BLOCKS
)
import json
import urllib.parse
import html
import re
from concurrent.futures import ThreadPoolExecutor
import hashlib
import functools
import tempfile
import psutil  # For monitoring system resources
import gc  # For garbage collection
import glob

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a search cache
_SEARCH_CACHE = {}  # Query hash -> (results, timestamp)
_SEARCH_CACHE_TTL = 3600  # 1 hour cache TTL
_BACKGROUND_TASKS = set()  # Keep track of background tasks

class CommandHistory:
    """Manages command history with persistence and navigation."""
    def __init__(self, history_file: str = os.path.expanduser("~/.cellbot_history")):
        self.history_file = history_file
        self.current_index = 0
        self.temp_input = ""
        
        # Ensure the directory exists with correct permissions
        try:
            history_dir = os.path.dirname(self.history_file)
            if not os.path.exists(history_dir):
                os.makedirs(history_dir, mode=0o700)  # Only user can read/write
            # If file doesn't exist, create it with correct permissions
            if not os.path.exists(self.history_file):
                with open(self.history_file, 'w') as f:
                    pass
                os.chmod(self.history_file, 0o600)  # Only user can read/write
        except Exception as e:
            logger.warning(f"Failed to setup history file: {e}")
            # Fallback to temporary file in /tmp
            self.history_file = os.path.join('/tmp', '.cellbot_history_' + str(os.getuid()))
        
        self.load_history()
        
        # Set up readline
        readline.set_history_length(1000)
        readline.parse_and_bind('"\\e[A": previous-history')  # Up arrow
        readline.parse_and_bind('"\\e[B": next-history')      # Down arrow
        readline.parse_and_bind('"\\C-r": reverse-search-history')  # Ctrl+R
        readline.parse_and_bind('"\t": complete')  # Tab completion
        
        # Set terminal width for formatting
        self.update_terminal_width()
        
    def update_terminal_width(self):
        """Update the terminal width for better formatting."""
        try:
            columns, _ = shutil.get_terminal_size()
            self.terminal_width = columns if columns > 40 else 80
        except Exception:
            # Default to 80 columns if we can't determine the width
            self.terminal_width = 80
    
    def wrap_text(self, text, initial_indent="", subsequent_indent="  "):
        """Wrap text to fit the terminal width."""
        self.update_terminal_width()  # Refresh terminal width
        wrapper = textwrap.TextWrapper(
            width=self.terminal_width - 2,  # Leave room for borders
            initial_indent=initial_indent,
            subsequent_indent=subsequent_indent,
            break_long_words=True,
            break_on_hyphens=True
        )
        
        # Process text line by line
        result = []
        for line in text.split('\n'):
            if line.strip():  # If not empty
                result.extend(wrapper.wrap(line))
            else:
                result.append('')  # Keep empty lines
        
        return '\n'.join(result)
        
    def load_history(self):
        """Load command history from file."""
        try:
            if os.path.exists(self.history_file):
                readline.read_history_file(self.history_file)
                logger.debug(f"Loaded command history from {self.history_file}")
        except Exception as e:
            logger.warning(f"Failed to load history file: {e}")
    
    def save_history(self):
        """Save command history to file."""
        try:
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            readline.write_history_file(self.history_file)
            logger.debug(f"Saved command history to {self.history_file}")
        except Exception as e:
            logger.warning(f"Failed to save history file: {e}")
    
    def add_command(self, command: str):
        """Add a command to history if it's not empty and different from last command."""
        if command.strip():
            # Only add if different from the last command
            if readline.get_current_history_length() == 0 or command != readline.get_history_item(readline.get_current_history_length()):
                readline.add_history(command)
                self.save_history()

class MinimalAIAgent:
    """
    Minimal AI Agent that uses simple in-memory storage for session data.
    The agent builds context by combining:
      - The system prompt.
      - Recent conversation messages.
      - The current user query.
    This context is then sent to the LLM.
    
    Features:
    - Web search integration for up-to-date information
    - Command history with navigation
    - Model switching at runtime
    - Performance metrics and diagnostics
    - Session-based conversation management
    """
    def __init__(self, model="mistral:7b", timeout=60, max_llm_calls=2, max_code_execs=2, debug_mode=False, save_code=False):
        """Initialize the minimal agent."""
        self.model = model
        self.timeout = timeout
        self.max_llm_calls = max_llm_calls
        self.max_code_execs = max_code_execs
        self.debug_mode = debug_mode
        self.save_code = save_code
        
        # Set up logging
        log_level = logging.DEBUG if debug_mode else logging.INFO
        logging.basicConfig(level=log_level)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        # Initialize command history
        try:
            import readline
            # Configure readline for command history
            # Fix invalid escape sequences
            readline.parse_and_bind('"\\e[A": previous-history')  # Up arrow
            readline.parse_and_bind('"\\e[B": next-history')      # Down arrow
            readline.parse_and_bind('"\\C-r": reverse-search-history')  # Ctrl+R
            self.readline_available = True
        except (ImportError, ModuleNotFoundError):
            self.readline_available = False
            self.logger.warning("readline module not available, command history disabled")
        
        # Simple in-memory storage to replace MockConversationDB
        self.settings = {}
        self.conversation = []
        self.successful_exchanges = []
        self.last_user_query = ""
        
        # Store the timeout value 
        self.default_timeout = timeout
        
        # Setup aiohttp session with proper timeout
        if timeout == 0:
            # 0 means no timeout
            session_timeout = None
            logger.info("Creating aiohttp session with no timeout")
        else:
            session_timeout = timeout
            logger.info(f"Creating aiohttp session with timeout: {timeout}s")
            
        self.aiohttp_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=session_timeout)
        )
        
        # Apply configuration settings
        global DEBUG_MODE, SAVE_CODE_BLOCKS
        DEBUG_MODE = debug_mode
        SAVE_CODE_BLOCKS = save_code
        
        # Optimized semaphores
        self.llm_semaphore = asyncio.Semaphore(max_llm_calls)
        self.code_semaphore = asyncio.Semaphore(max_code_execs)
        self.command_history = CommandHistory()
        self.session_start = datetime.now()
        self.in_comparison_mode = False
        
        # Ollama-specific optimizations
        # Try to import default values from android_config
        try:
            from .android_config import DEFAULT_TEMPERATURE, DEFAULT_NUM_PREDICT, get_optimal_llm_parameters
            # Get optimal parameters based on device capabilities
            optimal_params = get_optimal_llm_parameters()
            default_temp = optimal_params["temperature"]
            default_tokens = optimal_params["num_predict"]
        except (ImportError, AttributeError):
            # Use standard defaults if android_config is not available
            default_temp = 0.7
            default_tokens = 1024

        self.ollama_config = {
            "num_thread": os.cpu_count() or 4,  # Default to CPU count or 4
            "num_gpu": 0,  # Initialize to 0, will be set below
            "timeout": timeout,
            "temperature": default_temp,  # Platform-optimized temperature
            "num_predict": default_tokens  # Platform-optimized token limit
        }
        
        # Auto-detect GPU capabilities
        auto_gpu_layers = self._detect_gpu_capabilities()
        if auto_gpu_layers > 0:
            logger.info(f"Auto-detected GPU capabilities: recommended {auto_gpu_layers} layers")
            
        # Load Ollama config from environment, settings, or auto-detection
        if os.getenv("OLLAMA_NUM_THREAD"):
            self.ollama_config["num_thread"] = int(os.getenv("OLLAMA_NUM_THREAD"))
        elif self.settings.get("ollama_num_thread"):
            try:
                self.ollama_config["num_thread"] = int(self.settings.get("ollama_num_thread"))
            except (ValueError, TypeError):
                pass  # Use default if invalid
                
        if os.getenv("OLLAMA_NUM_GPU"):
            self.ollama_config["num_gpu"] = int(os.getenv("OLLAMA_NUM_GPU"))
        elif self.settings.get("ollama_num_gpu"):
            try:
                self.ollama_config["num_gpu"] = int(self.settings.get("ollama_num_gpu"))
            except (ValueError, TypeError):
                # Use auto-detected GPU if available
                self.ollama_config["num_gpu"] = auto_gpu_layers
        else:
            # No environment or DB setting, use auto-detected GPU
            self.ollama_config["num_gpu"] = auto_gpu_layers
        
        # Load temperature from environment or settings
        if os.getenv("OLLAMA_TEMPERATURE"):
            try:
                self.ollama_config["temperature"] = float(os.getenv("OLLAMA_TEMPERATURE"))
            except (ValueError, TypeError):
                pass  # Use default if invalid
        elif self.settings.get("ollama_temperature"):
            try:
                self.ollama_config["temperature"] = float(self.settings.get("ollama_temperature"))
            except (ValueError, TypeError):
                pass  # Use default if invalid
                
        # Load num_predict from environment or settings
        if os.getenv("OLLAMA_NUM_PREDICT"):
            try:
                self.ollama_config["num_predict"] = int(os.getenv("OLLAMA_NUM_PREDICT"))
            except (ValueError, TypeError):
                pass  # Use default if invalid
        elif self.settings.get("ollama_num_predict"):
            try:
                self.ollama_config["num_predict"] = int(self.settings.get("ollama_num_predict"))
            except (ValueError, TypeError):
                pass  # Use default if invalid
        
        # Performance metrics
        self.perf_metrics = {
            "avg_response_time": 0,
            "total_response_time": 0,
            "total_tokens": 0,
            "requests_count": 0,
            "tokens_per_second": 0,
            "cache_hits": 0,
            "timeouts": 0,
            "success_count": 0
        }
        
        # Track last garbage collection time for memory optimization
        self.last_gc_time = datetime.now()
        self.gc_interval = timedelta(minutes=2)  # Run GC every 2 minutes
        
        # Store default timeout for restoration after /notimeout
        self.default_timeout = timeout
        
        # Initialize command aliases and shortcuts
        self.command_aliases = {
            'h': self.show_help,
            'help': self.show_help,
            'history': self.show_command_history,
            'clear': self.clear_screen,
            'stats': self.show_session_stats,
            'repeat': self.repeat_last_command,
            'search': lambda x: self.web_search(x) if x else "Please provide a search query",
            'web': lambda x: self.web_search(x) if x else "Please provide a search query",
            'perf': self.show_performance_metrics,
            'memory': self.show_memory_stats,
            'model': self.set_model,
            'exit': lambda _: "exit",
            'quit': lambda _: "exit",
            'bye': lambda _: "exit",
            'threads': self.set_threads,
            'gpu': self.set_gpu_layers,
            'temperature': self.set_temperature,
            'temp': self.set_temperature,
            'num_predict': self.set_num_predict,
            'tokens': self.set_num_predict,
            'optimize': self.optimize_for_battery,
            'auto': self.optimize_for_battery,
            'nh': self.execute_nethunter_command,
            'nethunter': self.execute_nethunter_command,
            'netinfo': self.get_network_info,
            'sysinfo': self.get_system_info,
            'copy': self.copy_to_clipboard,
            'paste': self.paste_from_clipboard,
            'battery': self.check_battery_status,
            'timeout': self.set_timeout,
        }
        
        # Mobile optimization settings
        self.low_memory_mode = os.environ.get("CELLBOT_LOW_MEMORY", "false").lower() == "true"
        self.memory_threshold = float(os.environ.get("CELLBOT_MEMORY_THRESHOLD", "85.0"))
        
        # Memory monitoring
        self.memory_usage_history = []
        self.max_history_entries = 10  # Keep last 10 entries
        
    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.conversation.append({"role": role, "content": content})
        
        # Limit conversation length to prevent context overflow
        max_conversation_length = 20  # Adjust as needed
        if len(self.conversation) > max_conversation_length:
            # Remove oldest messages but keep the system message if it exists
            if self.conversation and self.conversation[0]["role"] == "system":
                self.conversation = [self.conversation[0]] + self.conversation[-(max_conversation_length-1):]
            else:
                self.conversation = self.conversation[-max_conversation_length:]

    def _detect_gpu_capabilities(self) -> int:
        """
        Auto-detect GPU capabilities for Ollama.
        Returns recommended number of GPU layers based on available hardware.
        Returns 0 if no GPU is detected or on mobile devices.
        """
        try:
            # For mobile environments, default to 0 GPU layers to avoid overheating
            # First check if we're on a mobile device
            if self._is_mobile_device():
                self.logger.info("Mobile device detected, defaulting to CPU-only")
                return 0
            
            # Check if CUDA is available (NVIDIA GPUs)
            try:
                import subprocess
                # Try to detect NVIDIA GPU with nvidia-smi command
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=2  # Short timeout
                )
                if result.returncode == 0 and result.stdout.strip():
                    self.logger.info(f"NVIDIA GPU detected: {result.stdout.strip()}")
                    # Return conservative number of layers for NVIDIA
                    return 32
            except (FileNotFoundError, subprocess.SubprocessError):
                # nvidia-smi not found or failed
                pass
                
            # Check for Apple Silicon (M1/M2/M3)
            if sys.platform == "darwin" and self._is_apple_silicon():
                self.logger.info("Apple Silicon detected")
                return 32
                
            # Default to 0 if we couldn't detect a compatible GPU
            return 0
        except Exception as e:
            self.logger.warning(f"Error detecting GPU capabilities: {e}")
            return 0
            
    def _is_mobile_device(self) -> bool:
        """
        Detect if running on a mobile device.
        """
        try:
            # Try to import mobile detection from android_config
            try:
                from .android_config import get_device_info
                device_info = get_device_info()
                return device_info.get("is_nethunter", False)
            except (ImportError, AttributeError):
                pass
                
            # Fallback detection methods
            # Check for common NetHunter paths
            nethunter_paths = [
                "/data/data/com.offsec.nethunter/files/home",
                "/data/data/com.termux/files/home"
            ]
            
            for path in nethunter_paths:
                if os.path.exists(path):
                    return True
                    
            # Check for Android-specific paths
            if os.path.exists("/system/build.prop"):
                return True
                
            return False
        except Exception:
            return False
            
    def _is_apple_silicon(self) -> bool:
        """
        Check if running on Apple Silicon (M1/M2/M3).
        """
        try:
            if sys.platform != "darwin":
                return False
                
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True
            )
            return "Apple" in result.stdout
        except Exception:
            return False

    async def show_help(self, args: str = ""):
        """Show help."""
        categories = {
            "Basic Commands": [
                ("/help", "Show this help message"),
                ("/clear", "Clear the screen"),
                ("/exit or /quit", "Exit the program"),
                ("/stats", "Show session statistics"),
                ("/history", "Show command history"),
                ("/!!", "Repeat last command"),
            ],
            "Model Management": [
                ("/model [name]", "Set or show the current model"),
                ("/threads [n]", "Set number of CPU threads"),
                ("/gpu [n]", "Set number of GPU layers"),
                ("/temperature [value]", "Set temperature (0.0-1.0)"),
                ("/tokens [n]", "Set max tokens to generate"),
                ("/timeout [seconds]", "Set response timeout"),
                ("/ollama", "Check Ollama status and restart if needed"),
            ],
            "Enhanced Commands": [
                ("/search [query]", "Search the web for information"),
                ("/s [query]", "Alias for /search"),
                ("/metrics", "Show performance metrics"),
                ("/nocache [prompt]", "Process prompt without using or saving to cache"),
                ("/netinfo [iface]", "Show network information"),
                ("/sysinfo", "Show system information"),
                ("/battery", "Check battery status"),
            ],
            "Mobile Optimization": [
                ("/optimize", "Auto-optimize settings for device"),
                ("/nh [command]", "Run NetHunter command"),
                ("/nethunter [cmd]", "Same as /nh"),
                ("/battery", "Check battery status"),
                ("/memory", "Show memory usage statistics"),
                ("/timeout [seconds]", "Set timeout in seconds"),
            ],
        }
        
        print("\n╭─ CellBot for NetHunter Help ─────────────────────────────")
        for category, commands in categories.items():
            print(f"\n{category}:")
            for cmd, desc in commands:
                print(f"  {cmd} - {desc}")
        print("╰──────────────────────────────────────────────────────────")

    async def show_command_history(self, _: str):
        """Show the command history with timestamps."""
        history = []
        for i in range(1, readline.get_current_history_length() + 1):
            cmd = readline.get_history_item(i)
            history.append(f"│  {i:3d} │ {cmd}")
        
        if not history:
            print("\n╭─ Command History ─── Empty ─────────────────────────────")
            print("╰────────────────────────────────────────────────────────")
            return
            
        width = max(len(line) for line in history) + 2
        print("\n╭─ Command History ─" + "─" * (width - 19))
        for line in history:
            print(line + " " * (width - len(line)))
        print("╰" + "─" * width)

    async def clear_screen(self, _: str = ""):
        """Clear the terminal screen."""
        try:
            if os.name == 'nt':  # Windows
                os.system('cls')
            else:  # Unix-like
                os.system('clear')
            
            # After clearing, show a minimal header
            print(f"""
╭───────────────────────────────────────────────╮
│        CellBot for NetHunter v1.0             │
│                                               │
│        Type /help for commands                │
╰───────────────────────────────────────────────╯
""")
            return None
        except Exception as e:
            return f"❌ Error clearing screen: {str(e)}"

    async def show_session_stats(self, _: str = ""):
        """Show statistics about the current session."""
        try:
            # Calculate session duration
            now = datetime.now()
            session_duration = now - self.session_start
            hours, remainder = divmod(session_duration.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            
            # Get memory usage
            memory = psutil.virtual_memory()
            
            # Get message counts from database
            user_messages = len(self.conversation)
            assistant_messages = len(self.successful_exchanges)
            
            # Display memory and session stats
            print("\n╭─ Session Statistics ──────────────────────────────────")
            print("│")
            print(f"│  ⏱️  Session Duration: {int(hours)}h {int(minutes)}m {int(seconds)}s")
            print(f"│  💬 Messages: {user_messages} queries, {assistant_messages} responses")
            print("│")
            print(f"│  🧠 Memory Usage: {memory.percent:.1f}% ({memory.used / (1024**2):.1f}MB/{memory.total / (1024**2):.1f}MB)")
            print(f"│  💾 Available Memory: {memory.available / (1024**2):.1f}MB")
            print("│")
            
            # Get model information
            print(f"│  🤖 Model: {self.model}")
            print(f"│  🔄 Threads: {self.ollama_config['num_thread']}")
            print(f"│  🖥️  GPU Layers: {self.ollama_config['num_gpu']}")
            print(f"│  🌡️  Temperature: {self.ollama_config.get('temperature', 0.7):.1f}")
            print(f"│  📝 Max Tokens: {self.ollama_config.get('num_predict', 1024)}")
            
            # Get performance metrics
            if hasattr(self, 'perf_metrics'):
                print("│")
                print(f"│  ⚡ Avg Response Time: {self.perf_metrics['avg_response_time']:.2f}s")
                if self.perf_metrics['requests_count'] > 0:
                    print(f"│  📊 Success Rate: {(self.perf_metrics['requests_count'] - self.perf_metrics['timeouts']) / self.perf_metrics['requests_count'] * 100:.1f}%")
            
            print("╰──────────────────────────────────────────────────────────")
            
            return None
        except Exception as e:
            return f"❌ Error retrieving session statistics: {str(e)}"

    async def repeat_last_command(self, _: str = ""):
        """Repeat the last user query."""
        if not self.last_user_query:
            return "No previous query to repeat."
        
        print(f"Repeating: {self.last_user_query}")
        
        # Check if it's a command
        if self.last_user_query.startswith('/'):
            await self.process_command(self.last_user_query)
        else:
            await self.process_query(self.last_user_query)
        
        return None

    def _extract_command_and_args(self, message: str) -> Tuple[Optional[str], str]:
        """Extract command and arguments from a message."""
        if message.startswith('/'):
            parts = message.split(maxsplit=1)
            command = parts[0][1:]  # Remove the leading '/'
            args = parts[1] if len(parts) > 1 else ""
            return command, args
        return None, message

    async def _build_context(self, user_message: str, no_cache: bool = False) -> List[Dict[str, str]]:
        """
        Build the context for the conversation, including:
        1. System prompt
        2. Recent conversation history
        3. Current user message
        
        Args:
            user_message: The current user message
            no_cache: Not used - kept for API compatibility
        
        Returns:
            List of message dictionaries forming the conversation context
        """
        context = []
        
        # 1. Add system prompt
        system_prompt = await get_system_prompt(None)
        context.append({"role": "system", "content": system_prompt})
        
        # 2. Add recent conversation history (most recent 5 messages)
        recent_messages = []
        for msg in self.conversation[-10:]:  # Get the last 10 to filter
            if len(recent_messages) >= 5:
                break
            recent_messages.append({"role": msg["role"], "content": msg["content"]})
        
        context.extend(recent_messages[-5:])  # Add up to 5 most recent messages
        
        # 3. Add current user message
        context.append({"role": "user", "content": user_message})
        return context

    def extract_code_from_response(self, response: str):
        """Extract code blocks from a response without filtering them.
        
        This method intentionally does not filter any code blocks - it simply returns all
        code blocks found in the response. This is because:
        1. Users expect code blocks to be executed when they appear in the response
        2. The system prompt provides examples like 'echo "hello!"' that should be executed
        3. Filtering creates unpredictable behavior where some commands are executed and others aren't
        """
        from .code_executor import extract_code_blocks
        
        # Get code blocks using the original function and return them all without filtering
        return extract_code_blocks(response)

    async def process_code_block(self, language: str, code: str) -> Tuple[int, str]:
        """
        Process a code block with auto-fix attempts.
        If execution fails, the agent sends a fix prompt to the LLM and retries (up to MAX_FIX_ATTEMPTS).
        """
        from .config import MAX_FIX_ATTEMPTS
        attempt = 0
        current_code = code.strip()
        last_error = ""
        while attempt < MAX_FIX_ATTEMPTS:
            ret, output = await execute_code_async(language, current_code)
            if ret == 0:
                return ret, output
            else:
                last_error = output
                fix_prompt = (
                    f"The following {language} code produced an error:\n\n"
                    f"{current_code}\n\n"
                    f"Error Output:\n{output}\n\n"
                    f"Please fix the code. Return only the corrected code in a single code block."
                )
                fix_context = [{"role": "user", "content": fix_prompt}]
                async with self.llm_semaphore:
                    fix_response = await get_llm_response_async(fix_context, self.model, self.aiohttp_session)
                blocks = self.extract_code_from_response(fix_response)
                if blocks:
                    current_code = blocks[0][1]
                    logger.info(f"Auto-fix attempt {attempt+1} applied.")
                else:
                    logger.error("LLM did not return a corrected code block.")
                    break
            attempt += 1
        return ret, last_error

    async def process_code_block_with_semaphore(self, language: str, code: str, idx: int):
        async with self.code_semaphore:
            return await self.process_code_block(language, code)

    async def _process_code_blocks_parallel(self, blocks: List[Tuple[str, str]]) -> List[Tuple[int, str]]:
        """Process code blocks in parallel with optimized concurrency."""
        start_time = datetime.now()
        results = []
        
        if blocks:
            print("\n┌─ Code Execution " + "─" * 50)
            
        for idx, (lang, code) in enumerate(blocks, 1):
            if len(blocks) > 1:
                print(f"\n├─ Block #{idx}")
            print(f"│  {lang}")
            print("│")
            for line in code.strip().split('\n'):
                print(f"│  {line}")
            print("│")
            
            async with self.code_semaphore:
                exec_start = datetime.now()
                ret, output = await self.process_code_block(lang, code)
                exec_time = (datetime.now() - exec_start).total_seconds()
                results.append((ret, output))
                
                if output.strip():
                    print("│")
                    print("│  Result:")
                    for line in output.strip().split('\n'):
                        print(f"│  {line}")
                else:
                    print("│")
                    print("│  No output")
                    
                if DEBUG_MODE:
                    print(f"│  Time: {exec_time:.2f}s")
            print("│")
        
        if blocks:
            if DEBUG_MODE:
                total_time = (datetime.now() - start_time).total_seconds()
                print(f"└─ Total execution time: {total_time:.2f}s")
            else:
                print("└" + "─" * 64)
        
        return results

    async def show_performance_metrics(self, _: str):
        """Show detailed performance metrics."""
        metrics = f"""
╭─ Performance Metrics ──────────────────────────────────────
│
│  LLM Statistics
│  • Total Calls : {self.perf_metrics['requests_count']}
│  • Cache Hits  : {self.perf_metrics['cache_hits']}"""

        if self.perf_metrics['requests_count'] > 0:
            metrics += f"""
│  • Avg Time    : {self.perf_metrics['avg_response_time']:.1f}s
│  • Total Time  : {self.perf_metrics['total_response_time']:.1f}s"""

        metrics += f"""
│  • Timeouts    : {self.perf_metrics['timeouts']}
│
│  Ollama Configuration
│  • CPU Threads : {self.ollama_config['num_thread']}
│  • GPU Layers  : {self.ollama_config['num_gpu']}
│  • Temperature : {self.ollama_config.get('temperature', 0.7):.1f}
│  • Max Tokens  : {self.ollama_config.get('num_predict', 1024)}
│
╰──────────────────────────────────────────────────────────"""
        
        # Return the metrics string instead of printing it
        return metrics

    async def process_message(self, message: str, no_cache: bool = False) -> Tuple[str, bool, bool]:
        """Process a message and return a response.
        
        Args:
            message: The message to process
            no_cache: Whether to skip the cache
            
        Returns:
            Tuple of (response, was_cached, code_executed)
        """
        # Check if it's a /nocache command first
        if message.startswith('/nocache '):
            no_cache = True
            message = message[9:].strip()  # Remove /nocache prefix
            print("ℹ️  Cache disabled for this request")
        
        # Check if it's a command that extends timeout
        use_extended_timeout = False
        original_timeout = self.ollama_config.get("timeout", 60)
        
        if message.startswith('/notimeout '):
            use_extended_timeout = True
            message = message[10:].strip()  # Remove /notimeout prefix
            
            # Store the original timeout for restoration later
            logger.info(f"Disabling timeout (original was {original_timeout}s)")
            
            # Store the old session to close it properly
            old_session = self.aiohttp_session
            
            # Update the timeout in the config
            self.ollama_config["timeout"] = 0  # Disable timeout
            
            # Update aiohttp session timeout
            self.aiohttp_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=None)  # None means no timeout
            )
            logger.info("Created new aiohttp session with no timeout")
            
            # Close the old session asynchronously
            if old_session:
                try:
                    await old_session.close()
                    logger.info("Closed old aiohttp session")
                except Exception as e:
                    logger.warning(f"Error closing old session: {e}")
            
            print("ℹ️  Timeout disabled - will wait indefinitely for response")
        
        # Extract command if message starts with /
        command, args = self._extract_command_and_args(message)
        
        if command:
            # Handle direct command aliases (functions that return values)
            if command in ['search', 'web']:
                search_results = await self.web_search(args)
                return search_results, False, False
            
            # Handle built-in commands
            if command == "history":
                history_response = await self._handle_history_command(command, args)
                if history_response:
                    return history_response, False, False
            
            # Handle model switching
            if command == "model":
                if not args:
                    return f"Current model: {self.model}\nUse /model [model_name] to switch models", False, False
                self.model = args
                return f"Model switched to {self.model}", False, False
            
            # Handle performance command
            if command == "perf":
                await self.show_performance_metrics(args)
                return "Performance metrics displayed above", False, False
        
        try:
            if not no_cache:
                phase = "database_lookup"
                similar_exchanges = self.successful_exchanges
                
                if similar_exchanges:
                    best_match = similar_exchanges[0]
                    similarity = best_match[2]
                    
                    if similarity >= SIMILARITY_THRESHOLD:
                        print("\n╭─ Cache Hit ─────────────────────────────────────────")
                        if similarity == 1.0:
                            print("│  ✓ Exact match found")
                        else:
                            print(f"│  ✓ Similar response found ({similarity:.1%} match)")
                            print("│")
                            print("│  Similar query:")
                            print(f"│  • {best_match[0]}")
                        
                        if DEBUG_MODE:
                            lookup_time = (datetime.now() - start_time).total_seconds()
                            print(f"│  ⏱  Lookup: {lookup_time:.2f}s")
                        
                        print("╰──────────────────────────────────────────────────────")
                        
                        self.perf_metrics["requests_count"] += 1
                        self.perf_metrics["cache_hits"] += 1
                        cached_response = best_match[1]
                        
                        blocks = self.extract_code_from_response(cached_response)
                        if blocks:
                            results = await self._process_code_blocks_parallel(blocks)
                            # Return the full response with was_cached=True and code_executed=True
                            return cached_response, True, True
                        
                        # Return the cached response with code_executed=False
                        return cached_response, True, False
            
            # Begin LLM processing phase
            phase = "llm_processing"
            print("\n╭─ Generating Response ───────────────────────────────────")
            print("│  ⟳ Processing request...")
            
            # Show similar examples even if below threshold
            if not no_cache and similar_exchanges:
                best_match = similar_exchanges[0]
                similarity = best_match[2]
                if similarity >= 0.5:  # Only show if somewhat relevant
                    print("│")
                    print("│  Similar examples found:")
                    print(f"│  • {similarity*100:.1f}%: '{best_match[0]}' → '{best_match[1][:50]}...'")
                    print("│")
                    print("│  ℹ️  Using examples for context but generating new response")
                    print("│     (similarity below cache threshold)")
            
            if DEBUG_MODE:
                context_start = datetime.now()
            
            context = await self._build_context(message, no_cache)
            
            if DEBUG_MODE:
                context_time = (datetime.now() - context_start).total_seconds()
                print(f"│  ⏱  Context: {context_time:.2f}s")
            
            llm_start = datetime.now()
            # We use the LLM semaphore here to limit concurrent API calls
            try:
                async with self.llm_semaphore:
                    response = await get_llm_response_async(
                        context, 
                        self.model, 
                        self.aiohttp_session,
                        temperature=self.ollama_config.get("temperature", 0.7),
                        num_thread=self.ollama_config.get("num_thread", 4),
                        num_gpu=self.ollama_config.get("num_gpu", 0),
                        num_predict=self.ollama_config.get("num_predict", 1024),
                        timeout=self.ollama_config.get("timeout", 60)
                    )
                
                # Add completion to metrics
                self.perf_metrics["requests_count"] += 1
                self.perf_metrics["success_count"] += 1
                
                # Calculate timing data
                llm_time = (datetime.now() - llm_start).total_seconds()
                self.perf_metrics["total_response_time"] += llm_time
                self.perf_metrics["total_tokens"] += llm_time
                self.perf_metrics["avg_response_time"] = (
                    self.perf_metrics["total_response_time"] / 
                    self.perf_metrics["requests_count"]
                )
                
                if llm_time > 10:
                    print(f"│  ⚠  Slow response ({llm_time:.1f}s)")
                elif DEBUG_MODE:
                    print(f"│  ⏱  LLM: {llm_time:.1f}s")
                    
            except asyncio.TimeoutError:
                self.perf_metrics["requests_count"] += 1
                self.perf_metrics["timeouts"] += 1
                raise TimeoutError(f"Response timed out after {self.ollama_config['timeout']}s")
            except Exception as e:
                logger.error(f"LLM response error: {str(e)}")
                print("│")
                print("│  ❌ LLM Response Failed:")
                print(f"│  • Error: {str(e)}")
                if hasattr(e, 'response') and e.response:
                    try:
                        error_json = await e.response.json()
                        if 'error' in error_json:
                            print(f"│  • Details: {error_json['error']}")
                    except:
                        if hasattr(e, 'response'):
                            error_text = await e.response.text()
                            print(f"│  • Details: {error_text[:200]}")
                print("│")
                raise
            
            blocks = self.extract_code_from_response(response)
            
            if not self.in_comparison_mode and not no_cache:
                # Store successful exchange with appropriate similarity
                if not similar_exchanges:
                    # No similar exchange found, use 0.0 similarity
                    self.successful_exchanges.append((message, response, 0.0))
                else:
                    # Use the calculated similarity 
                    self.successful_exchanges.append((message, response, similarity))
                
                if DEBUG_MODE:
                    print("│  ✓ Response cached")
            
            if DEBUG_MODE:
                total_time = (datetime.now() - start_time).total_seconds()
                print(f"│  ⏱  Total: {total_time:.2f}s")
            print("╰──────────────────────────────────────────────────────")
            
            # Execute code blocks immediately for non-cached responses
            if blocks:
                await self._process_code_blocks_parallel(blocks)
                # Return the full text response after executing code blocks
                # We'll mark it as code_executed=True so process_query knows code was already shown
                return response, False, True  # Returns response, was_cached, code_executed
            
            # Return the response with code_executed=False to indicate no code blocks were executed
            return response, False, False  # Returns response, was_cached, code_executed

        except Exception as e:
            total_time = (datetime.now() - start_time).total_seconds()
            print("\n╭─ Error ────────────────────────────────────────────────")
            print(f"│  ❌ {phase}: {str(e)}")
            if DEBUG_MODE:
                print(f"│  ⏱  Time: {total_time:.1f}s")
            print("╰──────────────────────────────────────────────────────")
            logger.error(f"Error in {phase}: {e}")
            return f"❌ Error in {phase}: {str(e)}", False, False

        finally:
            # Restore original timeout if it was changed
            if use_extended_timeout:
                self.ollama_config["timeout"] = original_timeout
                # Restore aiohttp session with default timeout
                await self.aiohttp_session.close()
                self.aiohttp_session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.default_timeout)
                )

    def help_command(self) -> str:
        """Return help text describing available commands."""
        return """Available commands:
/help - Show this help message
/list [search] - List successful exchanges, optionally filtered by search term
/remove <id> - Remove a successful exchange by ID
/update <id> <response> - Update the response for a successful exchange
/compare <prompt> - Compare responses from different models
/nocache <prompt> - Process prompt without using or saving to cache

Examples:
/list python  - List exchanges containing 'python'
/remove 123   - Remove exchange with ID 123
/nocache ls -la  - Run 'ls -la' without cache
"""

    def _is_successful_exchange(self, response: str) -> bool:
        # Implement the logic to determine if a response is a successful exchange
        # This is a placeholder and should be replaced with the actual implementation
        return True

    def list_command(self, search: str) -> str:
        # Implement the logic to list successful exchanges
        # This is a placeholder and should be replaced with the actual implementation
        return "List command not implemented"

    def remove_command(self, id: str) -> str:
        # Implement the logic to remove a successful exchange
        # This is a placeholder and should be replaced with the actual implementation
        return "Remove command not implemented"

    def update_command(self, id: str, response: str) -> str:
        # Implement the logic to update a successful exchange
        # This is a placeholder and should be replaced with the actual implementation
        return "Update command not implemented"

    async def process_success_command(self, command: str):
        """
        Process a /success command for NetHunter environment.
        
        Since the GUI is not available in the mobile terminal, this provides
        a simplified text-based interface for managing successful exchanges.
        """
        print("\n╭─ Success DB Management ───────────────────────────────────")
        print("│")
        print("│  ℹ️  GUI mode is not available in NetHunter terminal.")
        print("│")
        print("│  Available commands:")
        print("│  • /success list [search] - List successful exchanges")
        print("│  • /success stats        - Show statistics")
        print("│")
        print("╰──────────────────────────────────────────────────────────")
        
        # Parse subcommands if provided
        parts = command.split(maxsplit=1)
        subcommand = parts[1].split()[0] if len(parts) > 1 and parts[1].strip() else ""
        search_term = " ".join(parts[1].split()[1:]) if len(parts[1].split()) > 1 else ""
        
        if subcommand == "list":
            # List successful exchanges with optional search filter
            successes = [exchange[0] for exchange in self.successful_exchanges]
            
            if not successes:
                print("\nNo successful exchanges found." + (f" (filter: '{search_term}')" if search_term else ""))
                return
            
            print(f"\nFound {len(successes)} successful exchanges:" + (f" (filter: '{search_term}')" if search_term else ""))
            for i, entry in enumerate(successes[:10], 1):  # Show only first 10
                # Truncate for display
                print(f"{i}. {entry[:50]}..." if len(entry) > 50 else entry)
            
            if len(successes) > 10:
                print(f"... and {len(successes) - 10} more.")
        
        elif subcommand == "stats":
            # Show statistics about successful exchanges
            count = len(self.successful_exchanges)
            print(f"\nTotal successful exchanges: {count}")
            
            # Top keywords if available
            print("Most common keywords in successful exchanges:")
            try:
                words = []
                for exchange in self.successful_exchanges:
                    words.extend(re.findall(r'\b\w{3,}\b', exchange[0].lower()))
                
                from collections import Counter
                word_counts = Counter(words).most_common(5)
                for word, count in word_counts:
                    print(f"• '{word}': {count} occurrences")
            except Exception as e:
                print(f"Could not analyze keywords: {e}")

        else:
            print("\nUnknown subcommand. Use 'list' or 'stats'.")

    async def web_search(self, query: str, num_results: int = 3, fast_mode: bool = True) -> str:
        """
        Web search optimized for NetHunter mobile environment.
        
        Features:
        - Result caching with TTL to reduce network usage
        - Wikipedia search as primary source
        - Short timeouts to prevent hanging on mobile networks
        - Minimal results to conserve bandwidth
        - Graceful degradation for offline operation
        
        Args:
            query: Search query
            num_results: Max results to return (default: 3 for mobile)
            fast_mode: Use speed optimizations
            
        Returns:
            Formatted search results or advisory message
        """
        # Start timing
        start_time = datetime.now()
        
        if not query:
            return "Please provide a search query."
        
        # Check if we have network connectivity
        try:
            # Quick check for connectivity - minimal timeout
            async with self.aiohttp_session.get("https://en.wikipedia.org", 
                                              timeout=aiohttp.ClientTimeout(total=2)) as response:
                if response.status != 200:
                    logger.warning("Network check failed: Wikipedia returned non-200 status")
                    return "⚠️ Network connectivity issue detected. Please check your connection and try again."
        except Exception as e:
            logger.warning(f"Network check failed: {e}")
            return """
╭─ Network Connectivity Issue ─────────────────────────────
│
│  ⚠️  Unable to connect to the internet
│
│  Please check your:
│  • Wi-Fi or mobile data connection
│  • VPN status (if using VPN)
│  • Firewall settings
│
│  For offline operation, you can still use code execution
│  and other local features.
│
╰─────────────────────────────────────────────────────────"""
        
        # Clean and normalize query
        query = query.strip()
        search_hash = hashlib.md5(query.lower().encode()).hexdigest()
        
        # Check cache for recent results
        if search_hash in _SEARCH_CACHE:
            cached_results, timestamp = _SEARCH_CACHE[search_hash]
            if datetime.now() - timestamp < timedelta(seconds=_SEARCH_CACHE_TTL):
                elapsed = (datetime.now() - start_time).total_seconds()
                return f"🚀 Results for '{query}' (cached in {elapsed:.2f}s):\n\n{cached_results}"
        
        print(f"🔍 Searching for: {query}")
        
        try:
            # Only use Wikipedia as it's more reliable
            wiki_results = await self._search_wikipedia(query, num_results)
            
            if not wiki_results:
                return f"""
╭─ No Search Results ─────────────────────────────────────
│
│  No results found for: "{query}"
│
│  Try:
│  • Using different keywords
│  • Being more specific
│  • Checking for typos
│
╰────────────────────────────────────────────────────────"""
            
            # Format results - more compact for mobile
            formatted_results = ""
            for i, result in enumerate(wiki_results, 1):
                title = result.get('title', 'No title')
                link = result.get('link', 'No link')
                snippet = result.get('snippet', 'No description')
                
                formatted_results += f"{i}. {title}\n"
                formatted_results += f"   {snippet}\n"
                formatted_results += f"   🔗 {link}\n\n"
            
            # Cache the results
            _SEARCH_CACHE[search_hash] = (formatted_results, datetime.now())
            
            # Return results with timing info
            elapsed = (datetime.now() - start_time).total_seconds()
            return f"""
╭─ Search Results ({elapsed:.2f}s) ─────────────────────────
│
│  Results for: "{query}"
│
{formatted_results}│
╰────────────────────────────────────────────────────────"""
            
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return f"""
╭─ Search Error ─────────────────────────────────────────
│
│  Unable to search for: "{query}"
│  Error: {str(e)}
│
│  This could be due to:
│  • Network connectivity issues
│  • Search service unavailability
│  • Rate limiting
│
│  Please try again later.
│
╰────────────────────────────────────────────────────────"""

    async def _search_wikipedia(self, query: str, num_results: int = 3) -> List[Dict]:
        """Search Wikipedia using its API with optimized settings for mobile."""
        try:
            encoded_query = urllib.parse.quote(query)
            url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded_query}&format=json&utf8=1&srlimit={num_results}"
            
            # Use a shorter timeout for mobile networks
            async with self.aiohttp_session.get(url, timeout=5) as response:
                if response.status != 200:
                    logger.error(f"Wikipedia API error: {response.status}")
                    return []
                
                data = await response.json()
                results = []
                
                for item in data.get('query', {}).get('search', []):
                    title = html.unescape(item.get('title', ''))
                    snippet = html.unescape(re.sub(r'<.*?>', '', item.get('snippet', '')))
                    url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
                    results.append({
                        'title': title,
                        'link': url,
                        'snippet': snippet
                    })
                return results
                
        except asyncio.TimeoutError:
            logger.error("Wikipedia search timed out")
            raise Exception("Search timed out. Network might be slow or unstable.")
        except Exception as e:
            logger.error(f"Wikipedia search error: {e}")
            return []

    async def _turbo_search(self, query: str, num_results: int = 2) -> str:
        """
        Ultra-fast search optimized for mobile devices.
        Uses Wikipedia with minimal results for speed and bandwidth conservation.
        """
        if not query:
            return "Please provide a search query."
            
        try:
            print(f"⚡ TURBO searching for: {query}")
            # Use even fewer results for turbo mode on mobile
            return await self.web_search(query, num_results=num_results, fast_mode=True)
        except Exception as e:
            logger.error(f"Turbo search error: {e}")
            return f"""
╭─ Turbo Search Error ─────────────────────────────────────
│
│  Unable to perform turbo search for: "{query}"
│  Error: {str(e)}
│
│  Try again later or ask a different question.
│
╰────────────────────────────────────────────────────────"""

    async def set_model(self, model_name: str) -> str:
        """Set the LLM model to use."""
        try:
            models = await self._get_available_models()
            model_name = model_name.strip()

            # If model doesn't exist, offer to pull it
            if model_name not in models:
                result = f"Model '{model_name}' is not currently available locally.\n"
                try:
                    print(f"Attempting to pull model '{model_name}'...")
                    # Get absolute path to ollama if available
                    ollama_path = os.environ.get("OLLAMA_PATH", "ollama")
                    
                    # Try to pull the model using subprocess
                    proc = await asyncio.create_subprocess_exec(
                        ollama_path, "pull", model_name,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    
                    stdout, stderr = await proc.communicate()
                    
                    if proc.returncode == 0:
                        self.model = model_name
                        self.ollama_config["model"] = model_name
                        return f"Successfully pulled and switched to model '{model_name}'."
                    else:
                        error = stderr.decode() if stderr else "Unknown error"
                        return f"Failed to pull model '{model_name}': {error}\n\nAvailable models: {', '.join(models)}"
                        
                except Exception as e:
                    return f"Error pulling model '{model_name}': {str(e)}\n\nAvailable models: {', '.join(models)}"
            
            # Change the model
            self.model = model_name
            self.ollama_config["model"] = model_name
            return f"Switched to model: {model_name}"
            
        except Exception as e:
            return f"Error changing model: {str(e)}"
            
    async def check_ollama_status(self, _: str = "") -> str:
        """Check if Ollama is running properly and attempt to restart if needed."""
        try:
            # Get absolute path to ollama if available
            ollama_path = os.environ.get("OLLAMA_PATH", "ollama")
            
            # Import LLM_API_BASE from llm_client
            from .llm_client import LLM_API_BASE
            
            # Execute ollama list command
            process = await asyncio.create_subprocess_exec(
                ollama_path, "list",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await process.communicate()
            
            # Parse the output
            output = stdout.decode().strip().split("\n")
            
            # Skip the header line and extract model names
            if len(output) > 1:
                models = []
                for line in output[1:]:  # Skip header
                    parts = line.split()
                    if parts:
                        models.append(parts[0])
                return f"✅ Ollama is running correctly.\n\nAvailable models: {', '.join(models)}"
            return "Ollama is not running or no models available."
        except Exception as e:
            # Ollama might not be running
            from .llm_client import LLM_API_BASE
            result = f"❌ Ollama connection error: {str(e)}\n\nAttempting to start Ollama..."
            
            try:
                # Try to start Ollama in the background
                proc = await asyncio.create_subprocess_exec(
                    ollama_path, "serve",
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True  # Detach process from Python
                )
                
                # Wait a bit for Ollama to start
                await asyncio.sleep(3)
                
                # Check if it's now running
                try:
                    async with self.aiohttp_session.get(f"{LLM_API_BASE}/api/tags", timeout=5) as response:
                        if response.status == 200:
                            return f"{result}\n\n✅ Successfully started Ollama!"
                except Exception as new_e:
                    return f"{result}\n\n❌ Failed to connect after starting: {str(new_e)}"
                
                return f"{result}\n\n⚠️ Started Ollama but status is uncertain."
                
            except Exception as start_e:
                return f"{result}\n\n❌ Failed to start Ollama: {str(start_e)}\n\nPlease manually start Ollama with 'ollama serve' in a terminal."

    async def _get_available_models(self) -> List[str]:
        """Get a list of available Ollama models."""
        try:
            # Get absolute path to ollama if available
            ollama_path = os.environ.get("OLLAMA_PATH", "ollama")
            
            # Execute ollama list command
            process = await asyncio.create_subprocess_exec(
                ollama_path, "list",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await process.communicate()
            
            # Parse the output
            output = stdout.decode().strip().split("\n")
            
            # Skip the header line and extract model names
            if len(output) > 1:
                models = []
                for line in output[1:]:  # Skip header
                    parts = line.split()
                    if parts:
                        models.append(parts[0])
                return models
            return []
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return []

    async def set_threads(self, thread_count: str) -> str:
        """Set the number of CPU threads for Ollama.
        
        Args:
            thread_count: String containing the number of threads to use
                          If empty, returns the current thread count
        
        Returns:
            A confirmation message
        """
        # If no argument, return current setting
        if not thread_count.strip():
            return f"Current CPU thread count: {self.ollama_config['num_thread']}"
        
        # Try to parse the thread count
        try:
            new_thread_count = int(thread_count.strip())
            if new_thread_count <= 0:
                return f"Error: Thread count must be positive. Current count: {self.ollama_config['num_thread']}"
                
            # Set the new thread count
            self.ollama_config['num_thread'] = new_thread_count
            
            # Save to settings DB for persistence
            self.settings["ollama_num_thread"] = str(new_thread_count)
            
            return f"CPU thread count set to {new_thread_count}"
        except ValueError:
            return f"Error: '{thread_count}' is not a valid number. Current count: {self.ollama_config['num_thread']}"

    async def set_gpu_layers(self, gpu_layers: str) -> str:
        """Set the number of GPU layers for Ollama.
        
        Args:
            gpu_layers: String containing the number of GPU layers to use
                       If empty, returns the current GPU layer count
        
        Returns:
            A confirmation message
        """
        # If no argument, return current setting
        if not gpu_layers.strip():
            return f"Current GPU layer count: {self.ollama_config.get('num_gpu', 0)}"
        
        # Try to parse the GPU layer count
        try:
            new_gpu_layers = int(gpu_layers.strip())
            if new_gpu_layers < 0:
                return f"Error: GPU layer count must be non-negative. Current count: {self.ollama_config.get('num_gpu', 0)}"
                
            # Set the new GPU layer count
            self.ollama_config['num_gpu'] = new_gpu_layers
            
            # Save to settings DB for persistence
            self.settings["ollama_num_gpu"] = str(new_gpu_layers)
            
            return f"GPU layer count set to {new_gpu_layers}"
        except ValueError:
            return f"Error: '{gpu_layers}' is not a valid number. Current count: {self.ollama_config.get('num_gpu', 0)}"

    async def set_temperature(self, temperature: str) -> str:
        """Set the temperature value for LLM responses.
        
        Args:
            temperature: String containing the temperature value (0.0-1.0)
                         If empty, returns the current temperature
        
        Returns:
            A confirmation message
        """
        # If no argument, return current setting
        if not temperature.strip():
            return f"Current temperature: {self.ollama_config.get('temperature', 0.7)}"
        
        # Try to parse the temperature
        try:
            new_temperature = float(temperature.strip())
            if new_temperature < 0.0 or new_temperature > 1.0:
                return f"Error: Temperature must be between 0.0 and 1.0. Current temperature: {self.ollama_config.get('temperature', 0.7)}"
                
            # Set the new temperature
            self.ollama_config['temperature'] = new_temperature
            
            # Save to settings DB for persistence
            self.settings["ollama_temperature"] = str(new_temperature)
            
            return f"Temperature set to {new_temperature}"
        except ValueError:
            return f"Error: '{temperature}' is not a valid number. Current temperature: {self.ollama_config.get('temperature', 0.7)}"

    async def set_num_predict(self, num_predict: str) -> str:
        """Set the maximum number of tokens to predict."""
        try:
            # Validate input
            if not num_predict.strip():
                return f"Current max tokens: {self.ollama_config.get('num_predict', 1024)}"
                
            num_predict_int = int(num_predict.strip())
            if num_predict_int < 10:
                return "Error: Minimum token count is 10"
            if num_predict_int > 4096:
                return "Error: Maximum token count is 4096"
                
            # Set the value
            self.ollama_config["num_predict"] = num_predict_int
            return f"Maximum tokens to generate set to {num_predict_int}"
        except ValueError:
            return f"Error: Invalid number format. Please provide an integer between 10 and 4096."
            
    async def set_timeout(self, timeout: str) -> str:
        """Set the response timeout in seconds."""
        try:
            # Validate input
            if not timeout.strip():
                return f"Current timeout: {self.ollama_config.get('timeout', 60)} seconds"
                
            timeout_int = int(timeout.strip())
            if timeout_int < 5 and timeout_int != 0:
                return "Error: Minimum timeout is 5 seconds (or 0 for no timeout)"
            if timeout_int > 600:
                return "Warning: Very long timeout (>10 minutes) might lead to network issues"
                
            # Store the old session to close it properly
            old_session = self.aiohttp_session
            
            # Set the value
            self.ollama_config["timeout"] = timeout_int
            logger.info(f"Setting Ollama timeout to: {timeout_int}s")
            
            # Create a new session with the new timeout
            if timeout_int == 0:
                # 0 means no timeout
                self.aiohttp_session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=None)
                )
                logger.info("Created new aiohttp session with no timeout")
                
                # Close the old session asynchronously
                if old_session:
                    try:
                        await old_session.close()
                        logger.info("Closed old aiohttp session")
                    except Exception as e:
                        logger.warning(f"Error closing old session: {e}")
                        
                return "Timeout disabled (will wait indefinitely for responses)"
            else:
                self.aiohttp_session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=timeout_int)
                )
                logger.info(f"Created new aiohttp session with timeout: {timeout_int}s")
                
                # Close the old session asynchronously
                if old_session:
                    try:
                        await old_session.close()
                        logger.info("Closed old aiohttp session")
                    except Exception as e:
                        logger.warning(f"Error closing old session: {e}")
                        
                return f"Response timeout set to {timeout_int} seconds"
        except ValueError:
            return f"Error: Invalid number format. Please provide an integer."
            
    async def process_nocache(self, prompt: str) -> str:
        """Process a prompt without using or saving to the cache."""
        if not prompt.strip():
            return "Please provide a prompt after /nocache"
            
        print(f"Processing without cache: {prompt}")
        
        try:
            # Process the query with no_cache=True to bypass cache
            response, _, code_executed = await self.process_message(prompt, no_cache=True)
            
            # Add to conversation history
            self.add_message("user", prompt)
            self.add_message("assistant", response)
            
            # Return the response
            return response
        except Exception as e:
            return f"Error: {str(e)}"

    def _setup_autocomplete(self):
        """Set up command autocompletion for the agent."""
        readline.set_completer(self._command_completer)
        readline.parse_and_bind("tab: complete")
        
        # Set up known commands for autocompletion
        self.commands = [
            "/search", "/web", "/history", "/model", "/perf", "/notimeout", 
            "/help", "/exit", "/quit", "/bye", "/threads", "/gpu", 
            "/temp", "/temperature", "/tokens", "/num_predict",
            "/optimize", "/auto", "/battery", "/timeout"
        ]
        
        self.history_commands = [
            "search", "clear", "save"
        ]
        
        # Cache common shell commands
        self._update_shell_commands()

    def _update_shell_commands(self):
        """
        Update the list of available shell commands for tab completion.
        Gets common shell commands from PATH.
        """
        try:
            # Start with built-in commands
            self.shell_commands = []
            
            # Add commands from standard directories in PATH
            path_dirs = os.environ.get('PATH', '').split(os.pathsep)
            for dir_path in path_dirs:
                if os.path.exists(dir_path):
                    self.shell_commands.extend([
                        cmd for cmd in os.listdir(dir_path)
                        if os.path.isfile(os.path.join(dir_path, cmd)) and 
                        os.access(os.path.join(dir_path, cmd), os.X_OK)
                    ])
            
            # Remove duplicates and sort
            self.shell_commands = sorted(set(self.shell_commands))
            
        except Exception as e:
            logger.error(f"Error updating shell commands: {e}")
            self.shell_commands = []

    def _command_completer(self, text, state):
        """
        Custom completer function for readline that completes:
        1. Agent commands (starting with /)
        2. Subcommands for known agent commands
        3. Shell commands if not an agent command
        """
        # Check if we're completing an agent command
        if text.startswith("/"):
            options = [cmd for cmd in self.commands if cmd.startswith(text)]
            return options[state] if state < len(options) else None
        
        # Check if we're completing a history subcommand
        if self.current_input and self.current_input.startswith("/history "):
            remaining = self.current_input[9:].lstrip()
            if not " " in remaining:  # No subcommand argument yet
                options = [subcmd for subcmd in self.history_commands if subcmd.startswith(text)]
                return options[state] if state < len(options) else None
            
        # Default to shell command completion
        options = [cmd for cmd in self.shell_commands if cmd.startswith(text)]
        return options[state] if state < len(options) else None

    async def run(self):
        """Run the agent in interactive mode."""
        # Using a try-finally block to ensure cleanup happens
        try:
            # Display welcome banner
            print(self.get_welcome_banner())
            
            # Set up command completion if available
            self._setup_autocomplete()
            
            # Initial memory check for better user experience
            await self.check_memory_usage(display_info=False)
            
            while True:
                # Get user input
                user_input = await self.get_user_input()
                
                # Handle empty input or EOF
                if user_input is None:
                    print("\nExiting...")
                    break
                
                if not user_input.strip():
                    continue
                
                # Save the command in history
                self.command_history.add_command(user_input)
                
                # Exit commands
                if user_input.lower() in ['exit', 'quit', '/exit', '/quit']:
                    print("Exiting...")
                    break
                
                # Check if it's a command (starts with /)
                if user_input.startswith('/'):
                    await self.process_command(user_input)
                else:
                    # Process as a query
                    await self.process_query(user_input)
                    
                # Run garbage collection periodically to manage memory
                gc_now = datetime.now()
                if gc_now - self.last_gc_time > self.gc_interval:
                    self.last_gc_time = gc_now
                    gc.collect()
                    
        except KeyboardInterrupt:
            print("\nOperation cancelled by user. Exiting...")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}", exc_info=True)
            print(f"\n❌ Unexpected error: {e}")
        finally:
            # Always clean up resources
            print("\nCleaning up resources...")
            await self.cleanup()
            print(f"Session ended. Duration: {datetime.now() - self.session_start}")
            self.command_history.save_history()

    async def cleanup(self):
        """Clean up resources before exiting."""
        try:
            # Close aiohttp session properly
            if hasattr(self, 'aiohttp_session') and self.aiohttp_session:
                try:
                    logger.info("Closing aiohttp session...")
                    await self.aiohttp_session.close()
                    logger.info("Aiohttp session closed successfully")
                except Exception as e:
                    logger.error(f"Error closing aiohttp session: {e}")
            
            # Close database connection
            if hasattr(self, 'db') and self.db is not None:
                self.db.close()
                
            # Clear any temporary files
            temp_dir = tempfile.gettempdir()
            pattern = os.path.join(temp_dir, "cellbot_*")
            for path in glob.glob(pattern):
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                except Exception as e:
                    self.logger.warning(f"Failed to remove temporary file {path}: {e}")
            
            # Run final garbage collection
            gc.collect()
            
            # Log session end
            print(f"\nSession ended. Duration: {datetime.now() - self.session_start}")
            
            # Clean up any background tasks
            for task in _BACKGROUND_TASKS:
                if not task.done():
                    task.cancel()
                    
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}", exc_info=True)
