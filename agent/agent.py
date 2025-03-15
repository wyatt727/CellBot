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
# Replace the import of the real ConversationDB with our mock version
from .mock_db import MockConversationDB
from .system_prompt import get_system_prompt
from .llm_client import get_llm_response_async
from .code_executor import execute_code_async
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
    Minimal AI Agent that uses a local SQLite database for conversation history.
    The agent first checks if there is a cached successful exchange whose user prompt is
    very similar (≥ 95%) to the current query. If so, it immediately uses that cached response
    (thus bypassing new LLM text generation), but still processes any code execution.
    Otherwise, it builds context by combining:
      - The system prompt.
      - An example interaction from the success DB (if the best match is at least 80% similar).
      - The current user query.
    This context is then sent to the LLM.
    
    Features:
    - Web search integration for up-to-date information
    - Command history with navigation
    - Model switching at runtime
    - Performance metrics and diagnostics
    - Conversation history management
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
        
        # Use our MockConversationDB instead of the real ConversationDB
        self.db = MockConversationDB()
        self.last_user_query = self.db.get_setting("last_user_query") or ""
        
        # Store the timeout value
        self.default_timeout = timeout
        self.aiohttp_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout)
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
        self.in_comparison_mode = self.db.get_setting("in_comparison_mode") == "true"
        
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
        elif self.db.get_setting("ollama_num_thread"):
            try:
                self.ollama_config["num_thread"] = int(self.db.get_setting("ollama_num_thread"))
            except (ValueError, TypeError):
                pass  # Use default if invalid
                
        if os.getenv("OLLAMA_NUM_GPU"):
            self.ollama_config["num_gpu"] = int(os.getenv("OLLAMA_NUM_GPU"))
        elif self.db.get_setting("ollama_num_gpu"):
            try:
                self.ollama_config["num_gpu"] = int(self.db.get_setting("ollama_num_gpu"))
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
        elif self.db.get_setting("ollama_temperature"):
            try:
                self.ollama_config["temperature"] = float(self.db.get_setting("ollama_temperature"))
            except (ValueError, TypeError):
                pass  # Use default if invalid
                
        # Load num_predict from environment or settings
        if os.getenv("OLLAMA_NUM_PREDICT"):
            try:
                self.ollama_config["num_predict"] = int(os.getenv("OLLAMA_NUM_PREDICT"))
            except (ValueError, TypeError):
                pass  # Use default if invalid
        elif self.db.get_setting("ollama_num_predict"):
            try:
                self.ollama_config["num_predict"] = int(self.db.get_setting("ollama_num_predict"))
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
            "timeouts": 0
        }
        
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
            'success': self.process_success_command,
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
            'battery': self.check_battery_status
        }
        
        # Mobile optimization settings
        self.low_memory_mode = os.environ.get("CELLBOT_LOW_MEMORY", "false").lower() == "true"
        self.memory_threshold = float(os.environ.get("CELLBOT_MEMORY_THRESHOLD", "85.0"))
        self.last_gc_time = datetime.now()
        self.gc_interval = timedelta(minutes=5)  # Run GC every 5 minutes
        
        # Memory monitoring
        self.memory_usage_history = []
        self.max_history_entries = 10  # Keep last 10 entries

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
        """Show help text."""
        help_text = f"""
╭─ CellBot for NetHunter Help ─────────────────────────────

  🔍 WEB INTEGRATION
    /search [query]    - Search the web for information
    /web [query]       - Alias for /search

  📊 MODEL PERFORMANCE
    /model [name]      - Change the AI model
    /threads [num]     - Set CPU threads (default: auto)
    /gpu [layers]      - Set GPU acceleration layers
    /temp [value]      - Set temperature (0.0-1.0)
    /tokens [num]      - Set max tokens to generate
    /optimize          - Auto-optimize settings for device
    /perf              - Show performance metrics
    /memory            - Show memory usage statistics
    {'   /nocache          - Disable caching for next query' if hasattr(self, '_cache') else ''}

  💾 SESSION MANAGEMENT
    /history           - Show recent conversation history
    /save [filename]   - Save conversation to a file
    /success [cmd]     - Mark a command as successfully executed
    /copy [text]       - Copy text to clipboard
    /paste             - Paste from clipboard

  🔧 OPTIONS
    /notimeout         - Disable timeout for next query
    /timeout [seconds] - Set timeout in seconds

  📱 NETHUNTER TOOLS
    /nh [command]      - Run NetHunter command
    /nethunter [cmd]   - Same as /nh
    /netinfo [iface]   - Show network information
    /sysinfo           - Show system information
    /battery           - Check battery status

  🔄 GENERAL COMMANDS
    /repeat            - Repeat last query
    /clear             - Clear the screen
    /stats             - Show session statistics
    /help              - Show this help message
    /exit or /quit     - Exit the session

╰───────────────────────────────────────────────────────────
"""
        print(self.command_history.wrap_text(help_text))
        return None

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
            user_messages = self.db.get_message_count("user")
            assistant_messages = self.db.get_message_count("assistant")
            
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
        1. System prompt (with dynamically included similar examples)
        2. Recent conversation history
        3. Similar successful exchanges (if caching is enabled)
        
        Args:
            user_message: The current user message
            no_cache: If True, skips retrieving similar examples from cache
        
        Returns:
            List of message dictionaries forming the conversation context
        """
        context = []
        
        # 1. Add system prompt with relevant examples
        system_prompt = await get_system_prompt(user_message if not no_cache else None)
        context.append({"role": "system", "content": system_prompt})
        
        # 2. Add recent conversation history
        context.extend(self.db.get_recent_messages())
        
        # 3. Add current user message
        context.append({"role": "user", "content": user_message})
        return context

    def extract_code_from_response(self, response: str):
        """Extract code blocks from a response."""
        from .code_executor import extract_code_blocks
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

    async def process_message(self, message: str, no_cache: bool = False) -> Tuple[str, bool]:
        """
        Process user message and return a response.
        Returns: Tuple[response: str, was_cached: bool]
        """
        start_time = datetime.now()
        phase = "initialization"
        
        # Store for command completion context
        self.current_input = message
        
        # Turbo search mode with ?! prefix (ultra-fast search)
        if message.startswith('?!'):
            search_query = message[2:].strip()
            return await self._turbo_search(search_query), False
        
        # Standard quick search with ? prefix
        elif message.startswith('?'):
            search_query = message[1:].strip()
            search_results = await self.web_search(search_query)
            return search_results, False
        
        # Check for /notimeout flag
        use_extended_timeout = False
        if message.startswith('/notimeout '):
            use_extended_timeout = True
            message = message[10:].strip()  # Remove /notimeout prefix
            original_timeout = self.ollama_config["timeout"]
            self.ollama_config["timeout"] = 0  # Disable timeout
            print("ℹ️  Timeout disabled - will wait indefinitely for response")
            
            # Also update aiohttp session timeout
            self.aiohttp_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=0)  # 0 means no timeout
            )
        
        # Extract command if message starts with /
        command, args = self._extract_command_and_args(message)
        if command:
            # Handle direct command aliases (functions that return values)
            if command in ['search', 'web']:
                search_results = await self.web_search(args)
                return search_results, False
            
            # Handle built-in commands
            if command == "history":
                history_response = await self._handle_history_command(command, args)
                if history_response:
                    return history_response, False
            
            # Handle model switching
            if command == "model":
                if not args:
                    return f"Current model: {self.model}\nUse /model [model_name] to switch models", False
                self.model = args
                return f"Model switched to {self.model}", False
            
            # Handle performance command
            if command == "perf":
                await self.show_performance_metrics(args)
                return "Performance metrics displayed above", False
        
        try:
            if not no_cache:
                phase = "database_lookup"
                similar_exchanges = self.db.find_successful_exchange(message)
                
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
                            # Return an empty string to avoid duplicating the code output
                            return "", True
                        
                        return cached_response, True
            
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
            try:
                async with self.llm_semaphore:
                    response = await get_llm_response_async(
                        context, 
                        self.model, 
                        self.aiohttp_session,
                        num_thread=self.ollama_config["num_thread"],
                        num_gpu=self.ollama_config["num_gpu"],
                        temperature=self.ollama_config.get("temperature"),
                        num_predict=self.ollama_config.get("num_predict"),
                        timeout=self.ollama_config["timeout"]
                    )
                llm_time = (datetime.now() - llm_start).total_seconds()
                
                # Update performance metrics
                self.perf_metrics["requests_count"] += 1
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
            
            if not self.in_comparison_mode and not no_cache and blocks:
                self.db.add_successful_exchange(message, response)
                if DEBUG_MODE:
                    print("│  ✓ Response cached")
            
            if DEBUG_MODE:
                total_time = (datetime.now() - start_time).total_seconds()
                print(f"│  ⏱  Total: {total_time:.2f}s")
            print("╰──────────────────────────────────────────────────────")
            
            # Execute code blocks immediately for non-cached responses
            if blocks:
                await self._process_code_blocks_parallel(blocks)
                # Return an empty string to avoid duplicating code blocks
                return "", False
            
            return response, False

        except Exception as e:
            total_time = (datetime.now() - start_time).total_seconds()
            print("\n╭─ Error ────────────────────────────────────────────────")
            print(f"│  ❌ {phase}: {str(e)}")
            if DEBUG_MODE:
                print(f"│  ⏱  Time: {total_time:.1f}s")
            print("╰──────────────────────────────────────────────────────")
            logger.error(f"Error in {phase}: {e}")
            return f"❌ Error in {phase}: {str(e)}", False

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
            successes = self.db.list_successful_exchanges(search_term) if search_term else self.db.list_successful_exchanges()
            
            if not successes:
                print("\nNo successful exchanges found." + (f" (filter: '{search_term}')" if search_term else ""))
                return
            
            print(f"\nFound {len(successes)} successful exchanges:" + (f" (filter: '{search_term}')" if search_term else ""))
            for i, entry in enumerate(successes[:10], 1):  # Show only first 10
                # Truncate for display
                user_msg = entry[0][:50] + "..." if len(entry[0]) > 50 else entry[0]
                response = entry[1][:50] + "..." if len(entry[1]) > 50 else entry[1]
                print(f"{i}. User: {user_msg}")
                print(f"   Response: {response}\n")
            
            if len(successes) > 10:
                print(f"... and {len(successes) - 10} more.")
        
        elif subcommand == "stats":
            # Show statistics about successful exchanges
            count = len(self.db.list_successful_exchanges())
            print(f"\nTotal successful exchanges: {count}")
            
            # Top keywords if available
            print("Most common keywords in successful exchanges:")
            try:
                words = []
                for entry in self.db.list_successful_exchanges():
                    words.extend(re.findall(r'\b\w{3,}\b', entry[0].lower()))
                
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
        if not model_name.strip():
            return f"Current model: {self.model}"
        
        # Trim whitespace and validate
        new_model = model_name.strip()
        
        # Store the original model in case we need to revert
        original_model = self.model
        
        try:
            # Set the new model
            self.model = new_model
            
            # Build a minimal test context
            test_context = [{"role": "user", "content": "test"}]
            
            # Try a minimal request to validate the model
            try:
                await get_llm_response_async(
                    test_context,
                    self.model,
                    self.aiohttp_session,
                    num_thread=self.ollama_config["num_thread"],
                    num_gpu=self.ollama_config["num_gpu"],
                    temperature=self.ollama_config.get("temperature"),
                    num_predict=self.ollama_config.get("num_predict"),
                    timeout=5  # Short timeout for testing
                )
                
                # If we get here, the model is valid
                return f"Model successfully switched to {self.model}"
            except Exception as request_error:
                error_message = str(request_error).lower()
                
                # If the error contains "model not found", try with :latest suffix
                if "model not found" in error_message and ":" not in new_model:
                    try:
                        model_with_latest = f"{new_model}:latest"
                        self.model = model_with_latest
                        
                        await get_llm_response_async(
                            test_context,
                            self.model,
                            self.aiohttp_session,
                            num_thread=self.ollama_config["num_thread"],
                            num_gpu=self.ollama_config["num_gpu"],
                            temperature=self.ollama_config.get("temperature"),
                            num_predict=self.ollama_config.get("num_predict"),
                            timeout=5  # Short timeout for testing
                        )
                        
                        # If we get here, the model with :latest suffix is valid
                        return f"Model successfully switched to {self.model}"
                    except Exception as latest_error:
                        # Both original name and with :latest suffix failed
                        self.model = original_model
                        raise RuntimeError(f"Model '{new_model}' and '{model_with_latest}' not found.") from latest_error
                else:
                    # Other error, not related to model not found
                    raise request_error
            
        except Exception as e:
            # Revert to the original model
            self.model = original_model
            error_message = str(e)
            
            if "model not found" in error_message.lower():
                # Get available models
                try:
                    available_models = await self._get_available_models()
                    models_text = ", ".join(available_models[:10])
                    if len(available_models) > 10:
                        models_text += f" and {len(available_models) - 10} more"
                    
                    return f"Error: Model '{new_model}' not found. Available models: {models_text}. Current model is still {self.model}."
                except:
                    return f"Error: Model '{new_model}' not found. Current model is still {self.model}."
            else:
                return f"Error switching model: {error_message}. Current model is still {self.model}."
                
    async def _get_available_models(self) -> List[str]:
        """Get a list of available Ollama models."""
        try:
            # Execute ollama list command
            process = await asyncio.create_subprocess_exec(
                "ollama", "list",
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
            self.db.set_setting("ollama_num_thread", str(new_thread_count))
            
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
            self.db.set_setting("ollama_num_gpu", str(new_gpu_layers))
            
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
            self.db.set_setting("ollama_temperature", str(new_temperature))
            
            return f"Temperature set to {new_temperature}"
        except ValueError:
            return f"Error: '{temperature}' is not a valid number. Current temperature: {self.ollama_config.get('temperature', 0.7)}"

    async def set_num_predict(self, num_predict: str) -> str:
        """Set the maximum number of tokens to predict (response length limit).
        
        Args:
            num_predict: String containing the maximum number of tokens
                         If empty, returns the current num_predict value
        
        Returns:
            A confirmation message
        """
        # If no argument, return current setting
        if not num_predict.strip():
            return f"Current max tokens (num_predict): {self.ollama_config.get('num_predict', 1024)}"
        
        # Try to parse the num_predict value
        try:
            new_num_predict = int(num_predict.strip())
            if new_num_predict <= 0:
                return f"Error: Max tokens must be positive. Current value: {self.ollama_config.get('num_predict', 1024)}"
                
            # Set the new num_predict value
            self.ollama_config['num_predict'] = new_num_predict
            
            # Save to settings DB for persistence
            self.db.set_setting("ollama_num_predict", str(new_num_predict))
            
            return f"Max tokens (num_predict) set to {new_num_predict}"
        except ValueError:
            return f"Error: '{num_predict}' is not a valid number. Current value: {self.ollama_config.get('num_predict', 1024)}"

    def _setup_autocomplete(self):
        """Set up command autocompletion for the agent."""
        readline.set_completer(self._command_completer)
        readline.parse_and_bind("tab: complete")
        
        # Set up known commands for autocompletion
        self.commands = [
            "/search", "/web", "/history", "/model", "/perf", "/notimeout", 
            "/help", "/exit", "/quit", "/bye", "/threads", "/gpu", 
            "/temp", "/temperature", "/tokens", "/num_predict",
            "/optimize", "/auto", "/battery"
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
        """Run the agent in a loop."""
        print(self.get_welcome_banner())
        
        # Mobile device memory check on startup
        await self.check_memory_usage(True)  # Initial memory check with display
        
        while True:
            try:
                # Periodic memory check and garbage collection
                if datetime.now() - self.last_gc_time > self.gc_interval:
                    await self.check_memory_usage()
                    self.last_gc_time = datetime.now()
                
                user_query = await self.get_user_input()
                if user_query.lower() in ["exit", "quit", "bye"]:
                    print("Exiting...")
                    await self.clean_up()
                    return
                
                if not user_query.strip():
                    continue

                # Process commands that start with /
                if user_query.startswith("/"):
                    await self.process_command(user_query)
                    continue
                
                # Normal query processing
                await self.process_query(user_query)
                
            except KeyboardInterrupt:
                print("\nInterrupted. Type /exit to quit or press Ctrl+C again to force exit.")
                try:
                    await asyncio.sleep(1)
                except KeyboardInterrupt:
                    print("\nForce exiting...")
                    await self.clean_up()
                    return
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}", exc_info=True)
                print(f"\nAn error occurred: {e}")
                print("Please try again or type /exit to quit.")
    
    async def clean_up(self):
        """Clean up resources before exiting."""
        try:
            # Close database connection
            if hasattr(self, 'db') and self.db is not None:
                self.db.close()
            
            # Close aiohttp session
            if hasattr(self, 'aiohttp_session') and self.aiohttp_session is not None:
                await self.aiohttp_session.close()
            
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
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}", exc_info=True)

    async def check_battery_status(self, args: str = ""):
        """Check and display battery status of the device."""
        try:
            # Try to read battery status from sysfs
            if os.path.exists("/sys/class/power_supply/battery/capacity"):
                with open("/sys/class/power_supply/battery/capacity", "r") as f:
                    battery_level = int(f.read().strip())
                
                # Get charging status if available
                charging_status = "unknown"
                if os.path.exists("/sys/class/power_supply/battery/status"):
                    with open("/sys/class/power_supply/battery/status", "r") as f:
                        charging_status = f.read().strip()
                
                print("\n╭─ Battery Status ───────────────────────────────────────")
                print(f"│  Battery Level: {battery_level}%")
                print(f"│  Charging Status: {charging_status}")
                print("│")
                print("╰──────────────────────────────────────────────────────")
                
                return None
            else:
                return "Battery status not available on this device."
        except Exception as e:
            return f"Error checking battery status: {str(e)}"
    
    async def optimize_for_battery(self, args: str = ""):
        """
        Automatically optimize settings based on current device status.
        Adjusts temperature and token limit based on battery level and available memory.
        """
        try:
            # Try to import optimized parameters function
            try:
                from .android_config import get_optimal_llm_parameters
                optimal_params = get_optimal_llm_parameters()
                
                # Save previous settings for reporting
                prev_temp = self.ollama_config.get("temperature", 0.7)
                prev_tokens = self.ollama_config.get("num_predict", 1024)
                
                # Apply optimized settings
                self.ollama_config["temperature"] = optimal_params["temperature"]
                self.ollama_config["num_predict"] = optimal_params["num_predict"]
                
                # Save to settings DB for persistence
                self.db.set_setting("ollama_temperature", str(optimal_params["temperature"]))
                self.db.set_setting("ollama_num_predict", str(optimal_params["num_predict"]))
                
                print("\n╭─ Auto-Optimization ─────────────────────────────────────")
                print("│")
                print("│  ✅ Settings automatically optimized for current device status")
                print("│")
                print(f"│  🌡️  Temperature: {prev_temp:.1f} → {optimal_params['temperature']:.1f}")
                print(f"│  📝 Max Tokens: {prev_tokens} → {optimal_params['num_predict']}")
                print("│")
                
                # Check battery if available
                if os.path.exists("/sys/class/power_supply/battery/capacity"):
                    with open("/sys/class/power_supply/battery/capacity", "r") as f:
                        battery_level = int(f.read().strip())
                    print(f"│  🔋 Current Battery: {battery_level}%")
                
                # Check memory
                memory = psutil.virtual_memory()
                print(f"│  🧠 Available Memory: {memory.available / (1024**2):.1f}MB")
                print("│")
                print("╰──────────────────────────────────────────────────────")
                
                return None
            except (ImportError, AttributeError):
                # Fall back to basic optimization if android_config not available
                return "Auto-optimization is only available on mobile devices."
                
        except Exception as e:
            return f"Error optimizing settings: {str(e)}"

    async def show_memory_stats(self, _: str = ""):
        """Show memory usage statistics."""
        try:
            import psutil
            
            # Get memory info
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Format memory sizes
            def format_bytes(bytes_value):
                """Format bytes to human readable format."""
                for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                    if bytes_value < 1024 or unit == 'TB':
                        return f"{bytes_value:.1f} {unit}"
                    bytes_value /= 1024
            
            # Track memory usage history
            current_memory_percent = memory.percent
            self.memory_usage_history.append(current_memory_percent)
            
            # Keep history at max length
            if len(self.memory_usage_history) > self.max_history_entries:
                self.memory_usage_history = self.memory_usage_history[-self.max_history_entries:]
            
            # Trend indicators
            trend = "—"  # Default: stable
            if len(self.memory_usage_history) > 1:
                last_two = self.memory_usage_history[-2:]
                diff = last_two[1] - last_two[0]
                if diff > 2:
                    trend = "↑"  # Increasing
                elif diff < -2:
                    trend = "↓"  # Decreasing
            
            print("\n╭─ Memory Statistics ───────────────────────────────────────")
            print("│")
            print(f"│  🧠 RAM: {memory.percent:.1f}% used {trend}")
            print(f"│  • Total: {format_bytes(memory.total)}")
            print(f"│  • Used: {format_bytes(memory.used)}")
            print(f"│  • Available: {format_bytes(memory.available)}")
            
            if hasattr(swap, 'total') and swap.total > 0:
                print("│")
                print(f"│  💾 Swap: {swap.percent:.1f}% used")
                print(f"│  • Total: {format_bytes(swap.total)}")
                print(f"│  • Used: {format_bytes(swap.used)}")
            
            # Process info
            current_process = psutil.Process()
            process_memory = current_process.memory_info()
            
            print("│")
            print(f"│  📊 This Process (CellBot)")
            print(f"│  • RSS: {format_bytes(process_memory.rss)}")
            print(f"│  • VMS: {format_bytes(process_memory.vms)}")
            
            if hasattr(self, 'low_memory_mode'):
                print("│")
                print(f"│  ⚙️ Low Memory Mode: {'Enabled' if self.low_memory_mode else 'Disabled'}")
                print(f"│  • Threshold: {self.memory_threshold:.1f}%")
            
            print("╰──────────────────────────────────────────────────────────")
            
            # Check if memory usage is high and suggest cleanup
            if memory.percent > 90:
                print("\n⚠️  Warning: Memory usage is very high!")
                print("   Consider closing other applications or reducing token limits.")
            
            return None
        except Exception as e:
            return f"Error retrieving memory statistics: {str(e)}"

    async def check_memory_usage(self, display_info=False):
        """
        Check the current memory usage and perform garbage collection if needed.
        This is important for mobile devices with limited resources.
        
        Args:
            display_info: Whether to display memory info to the user
        """
        try:
            import psutil
            import gc
            
            # Get memory info
            memory = psutil.virtual_memory()
            
            # If memory usage is above threshold, trigger garbage collection
            if memory.percent > self.memory_threshold:
                if not self.low_memory_mode:
                    self.logger.info(f"Memory usage high ({memory.percent:.1f}%), entering low memory mode")
                    self.low_memory_mode = True
                
                # Force garbage collection
                gc.collect()
                
                # Clear search cache if it exists
                if '_SEARCH_CACHE' in globals():
                    global _SEARCH_CACHE
                    _SEARCH_CACHE.clear()
                    self.logger.info("Cleared search cache to free memory")
                
                if display_info:
                    print(f"\n⚠️  Memory usage is high ({memory.percent:.1f}%)")
                    print("   Automatic cleanup performed.")
            elif memory.percent < self.memory_threshold - 10 and self.low_memory_mode:
                # Exit low memory mode if memory usage improves significantly
                self.low_memory_mode = False
                self.logger.info(f"Memory usage improved ({memory.percent:.1f}%), exiting low memory mode")
            
            # Add to history
            self.memory_usage_history.append(memory.percent)
            if len(self.memory_usage_history) > self.max_history_entries:
                self.memory_usage_history = self.memory_usage_history[-self.max_history_entries:]
            
            # Display memory info if requested
            if display_info:
                await self.show_memory_stats("")
                
        except Exception as e:
            self.logger.warning(f"Error checking memory: {e}")
            # Don't propagate errors from this utility function

    async def execute_nethunter_command(self, command: str):
        """
        Execute a NetHunter-specific command.
        
        Args:
            command: The NetHunter command to execute
            
        Returns:
            The command output
        """
        if not command.strip():
            return "Please provide a command to execute."
            
        try:
            # Check if we're running in NetHunter environment
            nethunter_paths = [
                "/data/data/com.offsec.nethunter/files/home",
                "/data/data/com.termux/files/home"
            ]
            
            is_nethunter = any(os.path.exists(path) for path in nethunter_paths)
            
            if not is_nethunter:
                return "This command is only available in NetHunter environment."
            
            # Set up the command
            cmd_parts = command.strip().split()
            
            # Add sudo if appropriate
            if cmd_parts and cmd_parts[0] in ["ifconfig", "iwconfig", "aireplay-ng", 
                                             "airmon-ng", "airodump-ng", "aireplay-ng"]:
                cmd_parts = ["sudo"] + cmd_parts
            
            # Execute the command
            try:
                result = subprocess.run(
                    cmd_parts,
                    capture_output=True,
                    text=True,
                    timeout=30  # Set a reasonable timeout
                )
                
                output = result.stdout
                error = result.stderr
                
                if result.returncode != 0 and error:
                    return f"Error executing command: {error}"
                
                return output if output else "Command executed successfully (no output)."
                
            except subprocess.TimeoutExpired:
                return "Command timed out after 30 seconds."
            except subprocess.SubprocessError as e:
                return f"Error executing command: {str(e)}"
            
        except Exception as e:
            return f"Error: {str(e)}"

    async def get_network_info(self, iface: str = ""):
        """
        Display network interface information.
        
        Args:
            iface: Optional specific interface to display
            
        Returns:
            Network information
        """
        try:
            # Check if we're on a system with ifconfig/ip
            is_unix = sys.platform != "win32"
            
            if not is_unix:
                return "Network info is only available on Unix-like systems."
            
            # Determine which command to use
            ip_available = subprocess.run(["which", "ip"], capture_output=True, text=True).returncode == 0
            ifconfig_available = subprocess.run(["which", "ifconfig"], capture_output=True, text=True).returncode == 0
            
            if not (ip_available or ifconfig_available):
                return "Network utilities (ip or ifconfig) not found."
            
            # Function to get interface list
            def get_interfaces():
                if ip_available:
                    result = subprocess.run(["ip", "link", "show"], capture_output=True, text=True)
                    lines = result.stdout.strip().split("\n")
                    interfaces = []
                    for line in lines:
                        if ": " in line:
                            iface_name = line.split(": ")[1].split(":")[0]
                            interfaces.append(iface_name)
                    return interfaces
                elif ifconfig_available:
                    result = subprocess.run(["ifconfig", "-a"], capture_output=True, text=True)
                    lines = result.stdout.strip().split("\n")
                    interfaces = []
                    for line in lines:
                        if line and not line.startswith(" ") and ":" in line:
                            iface_name = line.split(":")[0].split(" ")[0]
                            interfaces.append(iface_name)
                    return interfaces
                return []
            
            # Function to get info for a specific interface
            def get_interface_info(interface):
                if ip_available:
                    link_info = subprocess.run(["ip", "link", "show", interface], capture_output=True, text=True)
                    addr_info = subprocess.run(["ip", "addr", "show", interface], capture_output=True, text=True)
                    return f"{link_info.stdout}\n{addr_info.stdout}".strip()
                elif ifconfig_available:
                    return subprocess.run(["ifconfig", interface], capture_output=True, text=True).stdout.strip()
                return ""
            
            # Get all interfaces if none specified
            interfaces = get_interfaces()
            
            if not interfaces:
                return "No network interfaces found."
            
            # If interface specified, get info for that interface
            if iface:
                if iface not in interfaces:
                    return f"Interface {iface} not found. Available interfaces: {', '.join(interfaces)}"
                
                info = get_interface_info(iface)
                
                return f"""
╭─ Network Interface: {iface} ───────────────────────────────
│
{info}
│
╰──────────────────────────────────────────────────────────
"""
            
            # Otherwise, show a summary of all interfaces
            result = "╭─ Network Interfaces ───────────────────────────────────\n│\n"
            
            for iface in interfaces:
                # Get IP addresses for this interface
                if ip_available:
                    addr_info = subprocess.run(
                        ["ip", "-brief", "addr", "show", iface], 
                        capture_output=True, 
                        text=True
                    ).stdout.strip()
                    
                    if addr_info:
                        result += f"│  • {addr_info}\n"
                elif ifconfig_available:
                    iface_info = subprocess.run(
                        ["ifconfig", iface], 
                        capture_output=True, 
                        text=True
                    ).stdout.strip()
                    
                    # Extract IP address from ifconfig output
                    ip_match = re.search(r"inet\s+(\d+\.\d+\.\d+\.\d+)", iface_info)
                    mac_match = re.search(r"ether\s+([0-9a-f:]+)", iface_info)
                    
                    ip_addr = ip_match.group(1) if ip_match else "No IP"
                    mac_addr = mac_match.group(1) if mac_match else "No MAC"
                    
                    result += f"│  • {iface}: {ip_addr} ({mac_addr})\n"
            
            result += "│\n"
            result += "│  Use '/netinfo [interface]' for detailed information\n"
            result += "╰──────────────────────────────────────────────────────────"
            
            return result
            
        except Exception as e:
            return f"Error retrieving network information: {str(e)}"

    async def get_system_info(self, _: str = ""):
        """
        Display system information.
        
        Returns:
            System information
        """
        try:
            import platform
            import psutil
            
            # Get basic system info
            uname = platform.uname()
            
            # Format memory sizes
            def format_bytes(bytes_value):
                """Format bytes to human readable format."""
                for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                    if bytes_value < 1024 or unit == 'TB':
                        return f"{bytes_value:.1f} {unit}"
                    bytes_value /= 1024
            
            # Get CPU info
            cpu_count = psutil.cpu_count(logical=False)
            cpu_count_logical = psutil.cpu_count(logical=True)
            cpu_percent = psutil.cpu_percent(interval=0.5)
            
            # Get memory info
            memory = psutil.virtual_memory()
            
            # Get disk info
            disk = psutil.disk_usage('/')
            
            # Try to get Android-specific info
            android_info = ""
            try:
                from .android_config import get_device_info
                device_info = get_device_info()
                if device_info.get("is_nethunter", False):
                    android_info = f"""│  📱 Device: {device_info.get('device_model', 'Unknown')}
│  📱 Android: {device_info.get('android_version', 'Unknown')}
│"""
            except ImportError:
                pass
            
            # Format the output
            result = f"""
╭─ System Information ───────────────────────────────────────
│
│  🖥️  System: {uname.system} {uname.release}
│  🏠 Hostname: {uname.node}
│  🔄 Architecture: {uname.machine}
│  🐍 Python: {platform.python_version()}
│
{android_info}│  💻 CPU:
│  • Model: {uname.processor or "Unknown"}
│  • Cores: {cpu_count} physical, {cpu_count_logical} logical
│  • Usage: {cpu_percent}%
│
│  🧠 Memory:
│  • Total: {format_bytes(memory.total)}
│  • Used: {format_bytes(memory.used)} ({memory.percent}%)
│  • Available: {format_bytes(memory.available)}
│
│  💾 Disk:
│  • Total: {format_bytes(disk.total)}
│  • Used: {format_bytes(disk.used)} ({disk.percent}%)
│  • Free: {format_bytes(disk.free)}
│
╰──────────────────────────────────────────────────────────
"""
            return result
            
        except Exception as e:
            return f"Error retrieving system information: {str(e)}"

    async def copy_to_clipboard(self, text: str):
        """
        Copy text to clipboard.
        
        Args:
            text: The text to copy to clipboard
            
        Returns:
            Confirmation message
        """
        if not text.strip():
            return "Please provide text to copy."
            
        try:
            # Check platform
            if sys.platform == "darwin":  # macOS
                try:
                    subprocess.run(["pbcopy"], input=text.encode(), check=True)
                    return "Text copied to clipboard."
                except subprocess.SubprocessError as e:
                    return f"Failed to copy to clipboard: {e}"
                    
            elif sys.platform == "linux":
                # Try different clipboard tools
                clipboard_tools = [
                    ["xclip", "-selection", "clipboard"],
                    ["xsel", "--clipboard", "--input"],
                    ["termux-clipboard-set"]
                ]
                
                for tool in clipboard_tools:
                    try:
                        subprocess.run(tool, input=text.encode(), check=True)
                        return "Text copied to clipboard."
                    except (subprocess.SubprocessError, FileNotFoundError):
                        continue
                
                return "No clipboard tools available. Install xclip or xsel."
                
            elif sys.platform == "win32":  # Windows
                try:
                    subprocess.run(["clip"], input=text.encode(), check=True)
                    return "Text copied to clipboard."
                except subprocess.SubprocessError as e:
                    return f"Failed to copy to clipboard: {e}"
            
            else:
                return f"Clipboard not supported on {sys.platform}"
                
        except Exception as e:
            return f"Error copying to clipboard: {str(e)}"
            
    async def paste_from_clipboard(self, _: str = ""):
        """
        Paste text from clipboard.
        
        Returns:
            The text from clipboard
        """
        try:
            # Check platform
            if sys.platform == "darwin":  # macOS
                try:
                    result = subprocess.run(["pbpaste"], capture_output=True, text=True)
                    if result.returncode == 0:
                        return result.stdout
                    else:
                        return "Failed to paste from clipboard."
                except (subprocess.SubprocessError, FileNotFoundError) as e:
                    return f"Failed to paste from clipboard: {e}"
                    
            elif sys.platform == "linux":
                # Try different clipboard tools
                clipboard_tools = [
                    ["xclip", "-selection", "clipboard", "-o"],
                    ["xsel", "--clipboard", "--output"],
                    ["termux-clipboard-get"]
                ]
                
                for tool in clipboard_tools:
                    try:
                        result = subprocess.run(tool, capture_output=True, text=True)
                        if result.returncode == 0:
                            return result.stdout
                    except (subprocess.SubprocessError, FileNotFoundError):
                        continue
                
                return "No clipboard tools available. Install xclip or xsel."
                
            elif sys.platform == "win32":  # Windows
                # There's no direct way to get clipboard via command-line on Windows
                return "Clipboard paste not supported on Windows in terminal mode."
            
            else:
                return f"Clipboard not supported on {sys.platform}"
                
        except Exception as e:
            return f"Error pasting from clipboard: {str(e)}"

    def get_welcome_banner(self):
        """Return a welcome banner for the agent."""
        try:
            # Try to get device info
            device_info = ""
            try:
                from .android_config import get_device_info
                info = get_device_info()
                if info.get("is_nethunter", False):
                    device_info = f"\n│  📱 Device: {info.get('device_model', 'Unknown')}"
                    if info.get("android_version"):
                        device_info += f" (Android {info.get('android_version')})"
            except (ImportError, AttributeError):
                pass
                
            # Get system info
            import platform
            system_info = f"│  🖥️  System: {platform.system()} {platform.release()}"
            python_info = f"│  🐍 Python: {platform.python_version()}"
            
            # Format the banner
            banner = f"""
╭───────────────────────────────────────────────────────────╮
│                                                           │
│             🤖 CellBot for NetHunter v1.0                 │
│                                                           │
│  Model: {self.model}                                      
{system_info}
{python_info}{device_info}
│                                                           │
│  Type /help for available commands                        │
│                                                           │
╰───────────────────────────────────────────────────────────╯
"""
            return banner
        except Exception as e:
            # Fallback to simple banner if anything fails
            return """
╭───────────────────────────────────────────────╮
│        CellBot for NetHunter v1.0             │
│                                               │
│        Type /help for commands                │
╰───────────────────────────────────────────────╯
"""

    async def get_user_input(self):
        """Get user input with command history support."""
        try:
            # Set up tab completion if not already done
            if not hasattr(self, 'commands'):
                self._setup_autocomplete()
                
            # Display prompt
            prompt = "🤖> "
            
            # Use asyncio to allow for non-blocking input
            loop = asyncio.get_event_loop()
            user_input = await loop.run_in_executor(None, lambda: input(prompt))
            
            # Add to command history if not empty
            if user_input.strip():
                self.command_history.add_command(user_input)
                
            # Store for potential repeat
            if not user_input.startswith('/'):
                self.last_user_query = user_input
                self.db.set_setting("last_user_query", user_input)
                
            return user_input
        except (EOFError, KeyboardInterrupt):
            # Handle Ctrl+D or Ctrl+C gracefully
            print("\nExiting...")
            return "exit"
        except Exception as e:
            self.logger.error(f"Error getting user input: {e}")
            print(f"\nError getting input: {e}")
            return ""

    async def process_command(self, command: str):
        """Process a command that starts with /."""
        if not command.startswith('/'):
            return
            
        # Extract command and arguments
        parts = command.split(maxsplit=1)
        cmd = parts[0][1:]  # Remove the leading '/'
        args = parts[1] if len(parts) > 1 else ""
        
        # Check if it's a known command alias
        if cmd in self.command_aliases:
            handler = self.command_aliases[cmd]
            
            # If it's a callable, call it with args
            if callable(handler):
                try:
                    # Check if it's a coroutine function
                    if asyncio.iscoroutinefunction(handler):
                        result = await handler(args)
                    else:
                        result = handler(args)
                        
                    # If the result is "exit", exit the program
                    if result == "exit":
                        print("Exiting...")
                        await self.clean_up()
                        sys.exit(0)
                        
                    # If there's a result, print it
                    if result:
                        print(result)
                except Exception as e:
                    self.logger.error(f"Error executing command '{cmd}': {e}")
                    print(f"❌ Error executing command: {e}")
            else:
                print(f"Command handler for '{cmd}' is not callable.")
        else:
            print(f"Unknown command: {cmd}")
            print("Type /help for available commands.")

    async def process_query(self, query: str):
        """Process a user query (not a command)."""
        if not query.strip():
            return
            
        try:
            # Process the query
            response, was_cached = await self.process_message(query)
            
            # If there's a response, print it
            if response:
                # Add to conversation history
                self.db.add_message("user", query)
                self.db.add_message("assistant", response)
                
                # Print the response
                print(f"\n{response}")
                
        except asyncio.TimeoutError:
            print("\n⚠️  Response timed out. You can try:")
            print("  • Using /notimeout before your query")
            print("  • Setting a longer timeout with /timeout [seconds]")
            print("  • Simplifying your query")
            print("  • Checking if the LLM server is running properly")
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            print(f"\n❌ Error: {e}")
            
        # Periodically check memory usage
        await self.check_memory_usage()
