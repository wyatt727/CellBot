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
from .db import ConversationDB
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
        
        self.db = ConversationDB()
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
        self.ollama_config = {
            "num_thread": os.cpu_count() or 4,  # Default to CPU count or 4
            "num_gpu": 0,  # Initialize to 0, will be set below
            "timeout": timeout
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
            'nh': self.execute_nethunter_command,
            'nethunter': self.execute_nethunter_command,
            'netinfo': self.get_network_info,
            'sysinfo': self.get_system_info,
            'copy': self.copy_to_clipboard,
            'paste': self.paste_from_clipboard,
            'battery': self.check_battery_status
        }
        
        # Update terminal width for formatting
        self.update_terminal_width()
        
        # Mobile optimization settings
        self.low_memory_mode = os.environ.get("CELLBOT_LOW_MEMORY", "false").lower() == "true"
        self.memory_threshold = float(os.environ.get("CELLBOT_MEMORY_THRESHOLD", "85.0"))
        self.last_gc_time = datetime.now()
        self.gc_interval = timedelta(minutes=5)  # Run GC every 5 minutes
        
        # Memory monitoring
        self.memory_usage_history = []
        self.max_history_entries = 10  # Keep last 10 entries

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
        print(self.wrap_text(help_text))
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

    def _setup_autocomplete(self):
        """Set up command autocompletion for the agent."""
        readline.set_completer(self._command_completer)
        readline.parse_and_bind("tab: complete")
        
        # Set up known commands for autocompletion
        self.commands = [
            "/search", "/web", "/history", "/model", "/perf", "/notimeout", 
            "/help", "/exit", "/quit", "/bye"
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

    async def _handle_history_command(self, command: str, args: str) -> Optional[str]:
        """
        Handle history-related commands.
        
        Commands:
        - /history - Show recent conversation history
        - /history search [query] - Search conversation history
        - /history clear - Clear all conversation history
        - /history save [filename] - Save history to a file
        
        Args:
            command: The command (should be 'history')
            args: Arguments for the history command
            
        Returns:
            Response message or None if command not handled
        """
        if command != "history":
            return None
        
        if not args:
            # Show recent history
            messages = self.db.get_conversation_history(limit=10)
            if not messages:
                return "No conversation history found."
            
            history = "Recent Conversation History:\n\n"
            for i, msg in enumerate(messages, 1):
                role = msg["role"].capitalize()
                content = msg["content"]
                # Truncate long messages
                if len(content) > 100:
                    content = content[:100] + "..."
                history += f"{i}. {role}: {content}\n\n"
            
            return history
        
        # Parse subcommands
        parts = args.split(maxsplit=1)
        subcommand = parts[0].lower() if parts else ""
        subargs = parts[1] if len(parts) > 1 else ""
        
        if subcommand == "search":
            if not subargs:
                return "Please provide a search query."
            
            messages = self.db.search_conversation(subargs)
            if not messages:
                return f"No messages found matching '{subargs}'."
            
            results = f"Search Results for '{subargs}':\n\n"
            for i, msg in enumerate(messages, 1):
                role = msg["role"].capitalize()
                content = msg["content"]
                # Highlight the search term
                content = content.replace(subargs, f"**{subargs}**")
                results += f"{i}. {role}: {content[:100]}...\n\n"
            
            return results
            
        elif subcommand == "clear":
            # Confirm before clearing
            if subargs == "confirm":
                self.db.clear_conversation_history()
                return "Conversation history has been cleared."
            else:
                return "To confirm clearing all conversation history, use '/history clear confirm'"
            
        elif subcommand == "save":
            filename = subargs or f"cellbot_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            try:
                messages = self.db.get_conversation_history(limit=100)
                with open(filename, "w") as f:
                    f.write(f"CellBot for NetHunter Conversation History - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    for msg in messages:
                        role = msg["role"].capitalize()
                        content = msg["content"]
                        f.write(f"{role}: {content}\n\n")
                    
                return f"Conversation history saved to {filename}"
            except Exception as e:
                return f"Error saving conversation history: {str(e)}"
            
        return f"Unknown history subcommand: {subcommand}. Available: search, clear, save"

    def _detect_gpu_capabilities(self) -> int:
        """
        Auto-detect GPU capabilities and return recommended number of GPU layers.
        
        For Android (NetHunter):
        - We'll try to get GPU info from /proc/cpuinfo and /proc/meminfo
        - This is a very basic heuristic and might not be accurate.
        
        Returns:
            int: Recommended number of GPU layers (0 if no GPU detected or info is insufficient)
        """
        try:
            # Basic Android GPU detection (very simplified)
            if sys.platform == 'linux':  # Check if we're on Linux (Android)
                # Check for common GPU vendors in /proc/cpuinfo
                try:
                    with open('/proc/cpuinfo', 'r') as f:
                        cpuinfo = f.read().lower()
                        if 'mali' in cpuinfo:  # Common ARM GPU
                            logger.info("Mali GPU detected, enabling acceleration")
                            return 16  # Example value, adjust as needed
                        elif 'adreno' in cpuinfo:  # Common Qualcomm GPU
                            logger.info("Adreno GPU detected, enabling acceleration")
                            return 16  # Example value
                        # Add more checks for other GPU types as needed
                except Exception as e:
                    logger.warning(f"Could not check GPU info: {e}")
                    pass  # Ignore errors reading /proc/cpuinfo

                # Very basic memory-based heuristic (not reliable)
                try:
                    with open('/proc/meminfo', 'r') as f:
                        meminfo = f.read().lower()
                        # Extract total memory (in KB)
                        total_mem_kb = int(re.search(r'memtotal:\s*(\d+)', meminfo).group(1))
                        total_mem_gb = total_mem_kb / (1024 * 1024)
                        if total_mem_gb > 6:  # If more than 6GB RAM, assume some GPU capability
                            logger.info(f"Device has {total_mem_gb:.1f}GB RAM, enabling modest GPU acceleration")
                            return 8  # Example value
                except Exception as e:
                    logger.warning(f"Could not check memory info: {e}")
                    pass
            
            # For non-Android platforms or if detection fails
            logger.info("No GPU detected or detection failed, using CPU only")
            return 0  # No GPU detected or insufficient information
            
        except Exception as e:
            logger.error(f"Error detecting GPU capabilities: {e}")
            return 0  # Safe fallback to CPU-only mode

    async def set_gpu_layers(self, gpu_count: str) -> str:
        """Set the number of GPU layers for Ollama.
        
        Args:
            gpu_count: String containing the number of GPU layers to use
                       If empty, returns the current GPU layer count
        
        Returns:
            A confirmation message
        """
        # If no argument, return current setting
        if not gpu_count.strip():
            if self.ollama_config['num_gpu'] == 0:
                return "GPU acceleration is currently disabled. Use /gpu [number] to enable it."
            else:
                return f"Current GPU layer count: {self.ollama_config['num_gpu']}"
        
        # Try to parse the GPU count
        try:
            new_gpu_count = int(gpu_count.strip())
            if new_gpu_count < 0:
                return f"Error: GPU layer count must be non-negative. Current count: {self.ollama_config['num_gpu']}"
                
            # Set the new GPU count
            old_count = self.ollama_config['num_gpu']
            self.ollama_config['num_gpu'] = new_gpu_count
            
            # Save to settings DB for persistence
            self.db.set_setting("ollama_num_gpu", str(new_gpu_count))
            
            if new_gpu_count == 0:
                return "GPU acceleration disabled. Running in CPU-only mode."
            elif old_count == 0:
                return f"GPU acceleration enabled with {new_gpu_count} layers."
            else:
                return f"GPU layer count set to {new_gpu_count}"
        except ValueError:
            return f"Error: '{gpu_count}' is not a valid number. Current count: {self.ollama_config['num_gpu']}"

    async def execute_nethunter_command(self, command: str) -> str:
        """
        Execute NetHunter tools and commands directly from CellBot.
        
        Args:
            command: NetHunter command to execute
            
        Returns:
            Command output or error message
        """
        if not command.strip():
            return """
╭─ NetHunter Command ─────────────────────────────────────────
│
│  Please provide a command to execute.
│  Example: /nh ifconfig
│
│  Type /help nh for more information and examples.
│
╰──────────────────────────────────────────────────────────"""
        
        try:
            print(f"\n╭─ Executing NetHunter Command ─────────────────────────────")
            print(f"│  $ {command}")
            print(f"│")
            
            # Execute the command
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Wait for the command to complete with a timeout
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60)
                
                # Process and display the output
                if stdout:
                    stdout_text = stdout.decode('utf-8', errors='replace').strip()
                    # Format the output for better display in terminal
                    formatted_stdout = "\n".join([f"│  {line}" for line in stdout_text.split('\n')])
                    print(formatted_stdout)
                
                if stderr:
                    stderr_text = stderr.decode('utf-8', errors='replace').strip()
                    if stderr_text:
                        print(f"│")
                        print(f"│  Errors/Warnings:")
                        formatted_stderr = "\n".join([f"│  {line}" for line in stderr_text.split('\n')])
                        print(formatted_stderr)
                
                print(f"│")
                print(f"│  Exit Code: {process.returncode}")
                print(f"╰──────────────────────────────────────────────────────────")
                
                # Return a formatted result
                result = f"Command executed with exit code {process.returncode}."
                if process.returncode != 0:
                    result += f"\nErrors occurred during execution."
                
                return result
                
            except asyncio.TimeoutError:
                # Kill the process if it times out
                process.kill()
                print(f"│")
                print(f"│  ⚠️ Command timed out after 60 seconds")
                print(f"╰──────────────────────────────────────────────────────────")
                return "⚠️ Command timed out after 60 seconds. Consider breaking it down into smaller parts."
                
        except Exception as e:
            print(f"│")
            print(f"│  ❌ Error: {str(e)}")
            print(f"╰──────────────────────────────────────────────────────────")
            return f"❌ Error executing command: {str(e)}"

    async def get_network_info(self, interface: str = "") -> str:
        """
        Get detailed network information about the device.
        
        Args:
            interface: Optional specific interface to check
            
        Returns:
            Formatted network information
        """
        try:
            print("\n╭─ Getting Network Information ───────────────────────────")
            
            commands = []
            
            # Base network commands
            if interface:
                print(f"│  Checking interface: {interface}")
                commands = [
                    f"ip addr show {interface}",
                    f"iwconfig {interface} 2>/dev/null || echo 'No wireless extensions'",
                    f"ethtool {interface} 2>/dev/null || echo 'No ethtool information'"
                ]
            else:
                print("│  Checking all network interfaces")
                commands = [
                    "ip addr",
                    "ip route",
                    "cat /proc/net/wireless 2>/dev/null || echo 'No wireless info available'",
                    "netstat -tuln | grep LISTEN"
                ]
            
            results = []
            for cmd in commands:
                print(f"│  Running: {cmd}")
                process = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                
                if stdout:
                    stdout_text = stdout.decode('utf-8', errors='replace').strip()
                    results.append(f"# {cmd}\n{stdout_text}")
            
            # Format and display results
            for result in results:
                print("│")
                lines = result.split('\n')
                for line in lines[:20]:  # Show only first 20 lines
                    print(f"│  {line}")
                if len(lines) > 20:
                    print(f"│  ... ({len(lines) - 20} more lines)")
            
            print("╰──────────────────────────────────────────────────────────")
            
            # Return a summary
            return f"Network information displayed above. {len(commands)} commands executed."
            
        except Exception as e:
            print(f"│  ❌ Error: {str(e)}")
            print("╰──────────────────────────────────────────────────────────")
            return f"❌ Error getting network information: {str(e)}"

    async def get_system_info(self, _: str = "") -> str:
        """
        Get detailed system information about the NetHunter device.
        
        Returns:
            Formatted system information
        """
        try:
            print("\n╭─ Getting System Information ────────────────────────────")
            print("│")
            
            # List of commands to gather system information
            commands = [
                ("CPU Info", "cat /proc/cpuinfo | grep -E 'processor|model name|Hardware'"),
                ("Memory Info", "free -h"),
                ("Storage Info", "df -h"),
                ("Android Version", "getprop ro.build.version.release"),
                ("Device Model", "getprop ro.product.model"),
                ("Kernel Version", "uname -a"),
                ("Battery Info", "cat /sys/class/power_supply/battery/uevent 2>/dev/null || echo 'Battery info not available'"),
                ("Uptime", "uptime")
            ]
            
            for label, cmd in commands:
                print(f"│  {label}:")
                process = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                
                if stdout:
                    stdout_text = stdout.decode('utf-8', errors='replace').strip()
                    lines = stdout_text.split('\n')
                    for line in lines[:5]:  # Show only first 5 lines per command
                        print(f"│    {line}")
                    if len(lines) > 5:
                        print(f"│    ... ({len(lines) - 5} more lines)")
                print("│")
            
            print("╰──────────────────────────────────────────────────────────")
            
            # Return a summary
            return "System information displayed above."
            
        except Exception as e:
            print(f"│  ❌ Error: {str(e)}")
            print("╰──────────────────────────────────────────────────────────")
            return f"❌ Error getting system information: {str(e)}"

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

    async def copy_to_clipboard(self, text: str = "") -> str:
        """
        Copy text to the clipboard.
        
        On mobile, this uses termux-clipboard-set if available.
        Otherwise, it uses a temporary file approach.
        
        Args:
            text: Text to copy. If empty, uses the last message or response.
            
        Returns:
            Confirmation message
        """
        # If no text provided, use last response
        if not text:
            # Try to get the last assistant response from the database
            messages = self.db.get_recent_messages(2)  # Get last 2 messages
            if len(messages) > 0:
                text = messages[-1]["content"] if messages[-1]["role"] == "assistant" else ""
                if not text and len(messages) > 1:
                    # If last message was from user, try the one before
                    text = messages[-2]["content"] if messages[-2]["role"] == "assistant" else ""
            
            if not text:
                return "No text to copy. Please provide text after the command, e.g., /copy hello world"
        
        try:
            # Try termux clipboard command first (for Android)
            process = await asyncio.create_subprocess_shell(
                f"echo '{text}' | termux-clipboard-set",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await process.communicate()
            
            # Check if termux-clipboard-set command was successful
            if process.returncode == 0:
                return f"✓ Copied {len(text)} characters to clipboard"
            
            # If termux not available, try xclip for X11
            process = await asyncio.create_subprocess_shell(
                f"echo '{text}' | xclip -selection clipboard",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await process.communicate()
            
            # Check if xclip command was successful
            if process.returncode == 0:
                return f"✓ Copied {len(text)} characters to clipboard"
            
            # Fallback: write to a temporary file
            temp_file = "/tmp/cellbot_clipboard.txt"
            with open(temp_file, "w") as f:
                f.write(text)
            
            return f"""
✓ Copied {len(text)} characters to a temporary file
Location: {temp_file}

Note: Clipboard access not directly available.
To copy this text, use: cat {temp_file}"""
            
        except Exception as e:
            return f"❌ Error copying to clipboard: {str(e)}"

    async def paste_from_clipboard(self, _: str = "") -> str:
        """
        Paste text from the clipboard.
        
        On mobile, this uses termux-clipboard-get if available.
        
        Returns:
            Text from clipboard or error message
        """
        try:
            # Try termux clipboard command first (for Android)
            process = await asyncio.create_subprocess_shell(
                "termux-clipboard-get",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            # Check if termux-clipboard-get command was successful
            if process.returncode == 0 and stdout:
                text = stdout.decode('utf-8').strip()
                # If the text is long, truncate it in the message
                if len(text) > 200:
                    print(f"📋 Clipboard content (first 200 of {len(text)} characters):")
                    print(text[:200] + "...")
                else:
                    print(f"📋 Clipboard content ({len(text)} characters):")
                    print(text)
                return f"Pasted {len(text)} characters from clipboard"
            
            # If termux not available, try xclip for X11
            process = await asyncio.create_subprocess_shell(
                "xclip -selection clipboard -o",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            # Check if xclip command was successful
            if process.returncode == 0 and stdout:
                text = stdout.decode('utf-8').strip()
                # If the text is long, truncate it in the message
                if len(text) > 200:
                    print(f"📋 Clipboard content (first 200 of {len(text)} characters):")
                    print(text[:200] + "...")
                else:
                    print(f"📋 Clipboard content ({len(text)} characters):")
                    print(text)
                return f"Pasted {len(text)} characters from clipboard"
            
            # Fallback: check if temporary file exists
            temp_file = "/tmp/cellbot_clipboard.txt"
            if os.path.exists(temp_file):
                with open(temp_file, "r") as f:
                    text = f.read()
                # If the text is long, truncate it in the message
                if len(text) > 200:
                    print(f"📋 Clipboard content (first 200 of {len(text)} characters):")
                    print(text[:200] + "...")
                else:
                    print(f"📋 Clipboard content ({len(text)} characters):")
                    print(text)
                return f"Pasted {len(text)} characters from temporary file"
            
            return "❌ No clipboard content available"
            
        except Exception as e:
            return f"❌ Error pasting from clipboard: {str(e)}"

    async def check_battery_status(self, _: str = "") -> str:
        """
        Check battery status on Android device.
        
        Returns:
            Formatted battery status
        """
        try:
            print("\n╭─ Checking Battery Status ──────────────────────────────")
            print("│")
            
            # Try various methods to get battery info
            battery_commands = [
                ("Android Battery", "termux-battery-status 2>/dev/null || echo 'termux-battery-status not available'"),
                ("Battery Uevent", "cat /sys/class/power_supply/battery/uevent 2>/dev/null || echo 'Battery uevent not available'"),
                ("Power Supply", "cat /sys/class/power_supply/*/capacity 2>/dev/null || echo 'Power supply info not available'"),
                ("Battery Status", "dumpsys battery 2>/dev/null || echo 'dumpsys not available'")
            ]
            
            battery_info = {}
            for label, cmd in battery_commands:
                print(f"│  Checking {label}...")
                process = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                
                if stdout:
                    output = stdout.decode('utf-8', errors='replace').strip()
                    print("│")
                    print(f"│  {label}:")
                    
                    # Process different output formats
                    if label == "Android Battery" and "termux-battery-status not available" not in output:
                        try:
                            import json
                            battery_data = json.loads(output)
                            battery_info["percentage"] = battery_data.get("percentage", "Unknown")
                            battery_info["status"] = battery_data.get("status", "Unknown")
                            battery_info["temperature"] = battery_data.get("temperature", "Unknown")
                            print(f"│    Level: {battery_info['percentage']}%")
                            print(f"│    Status: {battery_info['status']}")
                            print(f"│    Temperature: {battery_info['temperature']}°C")
                        except Exception as e:
                            print(f"│    Error parsing JSON: {e}")
                            print(f"│    Raw output: {output[:100]}")
                    elif label == "Battery Uevent" and "not available" not in output:
                        lines = output.split('\n')
                        for line in lines[:10]:  # Show only first 10 lines
                            if "POWER_SUPPLY_CAPACITY=" in line:
                                value = line.split("=")[1]
                                battery_info["percentage"] = value
                                print(f"│    Level: {value}%")
                            elif "POWER_SUPPLY_STATUS=" in line:
                                value = line.split("=")[1]
                                battery_info["status"] = value
                                print(f"│    Status: {value}")
                            elif "POWER_SUPPLY_TEMP=" in line:
                                value = int(line.split("=")[1]) / 10  # Convert to celsius
                                battery_info["temperature"] = value
                                print(f"│    Temperature: {value}°C")
                            else:
                                print(f"│    {line}")
                    else:
                        lines = output.split('\n')
                        for line in lines[:10]:  # Show only first 10 lines
                            print(f"│    {line}")
                        if len(lines) > 10:
                            print(f"│    ... ({len(lines) - 10} more lines)")
                print("│")
            
            # Display a summary if we have the information
            if battery_info:
                print("│  Battery Summary:")
                percentage = battery_info.get("percentage", "Unknown")
                status = battery_info.get("status", "Unknown")
                temperature = battery_info.get("temperature", "Unknown")
                
                print(f"│    Level: {percentage}%")
                print(f"│    Status: {status}")
                if temperature != "Unknown":
                    print(f"│    Temperature: {temperature}°C")
                
                # Provide a user-friendly assessment
                if isinstance(percentage, int) or (isinstance(percentage, str) and percentage.isdigit()):
                    percentage = int(percentage)
                    if percentage < 20:
                        print("│    ⚠️ Battery is critically low. Consider charging soon.")
                    elif percentage < 50:
                        print("│    🔋 Battery is getting low.")
                    else:
                        print("│    ✓ Battery level is good.")
            
            print("╰──────────────────────────────────────────────────────────")
            
            # Return a summary
            if battery_info:
                return f"Battery status: {battery_info.get('percentage', 'Unknown')}% ({battery_info.get('status', 'Unknown')})"
            else:
                return "Battery information not available on this device."
            
        except Exception as e:
            print(f"│  ❌ Error: {str(e)}")
            print("╰──────────────────────────────────────────────────────────")
            return f"❌ Error checking battery status: {str(e)}"

    async def check_memory_usage(self, display=False):
        """
        Check system memory usage and optimize if needed.
        
        Args:
            display: Whether to display memory info to the user
        """
        try:
            # Get memory stats
            memory = psutil.virtual_memory()
            percent_used = memory.percent
            available_mb = memory.available / (1024 * 1024)
            total_mb = memory.total / (1024 * 1024)
            
            # Store in history
            timestamp = datetime.now()
            self.memory_usage_history.append({
                "timestamp": timestamp,
                "percent_used": percent_used,
                "available_mb": available_mb
            })
            
            # Keep history limited
            if len(self.memory_usage_history) > self.max_history_entries:
                self.memory_usage_history.pop(0)
            
            # Log memory usage
            self.logger.info(f"Memory usage: {percent_used:.1f}% used, {available_mb:.1f}MB available")
            
            # Display to user if requested
            if display:
                print(f"\n╭─ System Memory Status ────────────────────────────────")
                print(f"│  Memory usage: {percent_used:.1f}% of {total_mb:.1f}MB total")
                print(f"│  Available: {available_mb:.1f}MB")
                
                # Add recommendations based on available memory
                if available_mb < 200:
                    print(f"│  ⚠️ Very low memory available! Enabling low memory mode.")
                    print(f"│  ℹ️ Consider closing other apps to improve performance.")
                    self.low_memory_mode = True
                elif available_mb < 500:
                    print(f"│  🔸 Limited memory available. Some features may be slower.")
                    print(f"│  ℹ️ Use shorter conversations for better performance.")
                else:
                    print(f"│  ✓ Memory status good")
                print(f"╰──────────────────────────────────────────────────────")
            
            # Check if we're above threshold and need optimization
            if percent_used > self.memory_threshold:
                self.logger.warning(f"Memory usage above threshold ({percent_used:.1f}% > {self.memory_threshold}%)")
                self.low_memory_mode = True
                
                # Take action to reduce memory usage
                gc.collect()  # Force garbage collection
                
                # Trim database cache
                if hasattr(self, 'db') and self.db is not None:
                    # Trim any database caches
                    pass
                
                if display:
                    print(f"\n⚠️ High memory usage detected. Optimizing...")
            
            return percent_used, available_mb
            
        except Exception as e:
            self.logger.error(f"Error checking memory: {e}")
            return None, None

    async def show_memory_stats(self, _: str = "") -> str:
        """
        Display memory usage statistics and history.
        
        Returns:
            Formatted memory stats
        """
        try:
            # Get current memory usage
            current_usage, available_mb = await self.check_memory_usage()
            
            output = ["\n╭─ Memory Usage Statistics ─────────────────────────────"]
            output.append("│")
            output.append(f"│  Current Memory Usage: {current_usage:.1f}%")
            output.append(f"│  Available Memory: {available_mb:.1f}MB")
            output.append(f"│  Low Memory Mode: {'Enabled' if self.low_memory_mode else 'Disabled'}")
            output.append("│")
            
            # Display memory usage history if available
            if self.memory_usage_history:
                output.append("│  Memory Usage History:")
                output.append("│  Timestamp           | Usage % | Available MB")
                output.append("│  -------------------- | ------- | ------------")
                for entry in self.memory_usage_history:
                    timestamp = entry["timestamp"].strftime("%H:%M:%S")
                    percent = entry["percent_used"]
                    avail = entry["available_mb"]
                    output.append(f"│  {timestamp}            | {percent:6.1f}% | {avail:8.1f}MB")
            
            # Add a graph representation if we have enough history
            if len(self.memory_usage_history) >= 3:
                output.append("│")
                output.append("│  Memory Usage Trend:")
                
                # Simple ASCII graph
                max_percent = max(entry["percent_used"] for entry in self.memory_usage_history)
                min_percent = min(entry["percent_used"] for entry in self.memory_usage_history)
                range_percent = max(max_percent - min_percent, 1.0)  # Avoid division by zero
                
                graph_width = 40
                for entry in self.memory_usage_history:
                    percent = entry["percent_used"]
                    normalized = (percent - min_percent) / range_percent
                    bar_length = int(normalized * graph_width)
                    bar = "█" * bar_length
                    output.append(f"│  {percent:5.1f}% |{bar}")
            
            output.append("╰──────────────────────────────────────────────────────")
            
            print("\n".join(output))
            return f"Memory usage: {current_usage:.1f}% (Low memory mode: {'enabled' if self.low_memory_mode else 'disabled'})"
            
        except Exception as e:
            self.logger.error(f"Error showing memory stats: {e}")
            return f"❌ Error showing memory statistics: {str(e)}"

    def get_welcome_banner(self):
        """Return a welcome banner with system information."""
        try:
            # Only show detailed system info in debug mode to save memory
            if not self.debug_mode:
                return f"""
╭───────────────────────────────────────────────╮
│                                               │
│        CellBot for NetHunter v1.0             │
│                                               │
│      Model: {self.model}                     
│      Type /help for commands                   │
│                                               │
╰───────────────────────────────────────────────╯
"""
            
            # Get memory info
            memory = psutil.virtual_memory()
            ram_gb = memory.total / (1024 * 1024 * 1024)
            
            # Detect Android version
            android_version = "Unknown"
            try:
                result = subprocess.run(
                    ["getprop", "ro.build.version.release"], 
                    capture_output=True, 
                    text=True, 
                    timeout=1
                )
                if result.returncode == 0 and result.stdout.strip():
                    android_version = result.stdout.strip()
            except (subprocess.SubprocessError, FileNotFoundError):
                pass
                
            # Detect device model
            device_model = "Unknown"
            try:
                result = subprocess.run(
                    ["getprop", "ro.product.model"], 
                    capture_output=True, 
                    text=True, 
                    timeout=1
                )
                if result.returncode == 0 and result.stdout.strip():
                    device_model = result.stdout.strip()
            except (subprocess.SubprocessError, FileNotFoundError):
                pass
            
            # Get CPU info
            cpu_count = psutil.cpu_count(logical=True)
            cpu_physical = psutil.cpu_count(logical=False) or cpu_count
            
            # Check if running in Termux
            in_termux = os.environ.get("TERMUX_VERSION") is not None
            
            return f"""
╭───────────────────────────────────────────────╮
│                                               │
│        CellBot for NetHunter v1.0             │
│                                               │
│      Model: {self.model}                     
│      Device: {device_model}                   
│      Android: {android_version}                
│      RAM: {ram_gb:.1f}GB                       
│      CPU: {cpu_physical} cores ({cpu_count} threads)
│      Environment: {'Termux' if in_termux else 'NetHunter'}
│                                               │
│      Type /help for commands                   │
│                                               │
╰───────────────────────────────────────────────╯
"""
        except Exception as e:
            # Fallback to simple banner on error
            self.logger.error(f"Error generating welcome banner: {e}")
            return f"""
╭───────────────────────────────────────────────╮
│                                               │
│        CellBot for NetHunter v1.0             │
│                                               │
│      Model: {self.model}                     
│      Type /help for commands                   │
│                                               │
╰───────────────────────────────────────────────╯
"""

    async def get_user_input(self) -> str:
        """Get user input with history support."""
        try:
            # Check if we have readline for better input handling
            if self.readline_available:
                # Use asyncio to avoid blocking the event loop
                user_input = await asyncio.to_thread(input, "\n❯ ")
            else:
                # Fallback approach
                print("\n❯ ", end="", flush=True)
                user_input = await asyncio.to_thread(sys.stdin.readline)
                user_input = user_input.strip()
                
            # Handle special command cases
            if user_input.startswith('/'):
                # If it's a command alias, allow it without the slash
                cmd = user_input[1:].split(' ')[0].lower()
                if cmd in self.command_aliases:
                    return user_input
                    
            # Add to command history if not empty
            if user_input.strip():
                self.command_history.add_command(user_input)
                
            return user_input
            
        except (EOFError, KeyboardInterrupt):
            # Handle Ctrl+D and Ctrl+C gracefully
            print("\nUse /exit to quit")
            return ""
            
    async def process_command(self, user_query: str) -> None:
        """Process commands starting with /."""
        # Extract command and arguments
        parts = user_query[1:].split(maxsplit=1)
        command = parts[0].lower() if parts else ""
        args = parts[1] if len(parts) > 1 else ""
        
        # Check if it's a known command
        if command in self.command_aliases:
            handler = self.command_aliases[command]
            if callable(handler):
                try:
                    # Handle async and sync handlers
                    if asyncio.iscoroutinefunction(handler) or isinstance(handler, functools.partial) and asyncio.iscoroutinefunction(handler.func):
                        # Async handler
                        result = await handler(args)
                    else:
                        # Sync handler (usually a lambda)
                        result = handler(args)
                    
                    # If the command returns a string, print it
                    if result is not None:
                        if result == "exit":
                            print("Exiting...")
                            await self.clean_up()
                            sys.exit(0)
                        print(result)
                except Exception as e:
                    self.logger.error(f"Error executing command '{command}': {e}", exc_info=True)
                    print(f"❌ Error executing command: {str(e)}")
            return
        else:
            # Unknown command
            print(f"❓ Unknown command: /{command}")
            print("Type /help for a list of available commands")
    
    async def process_query(self, user_query: str) -> None:
        """Process a regular user query (not a command)."""
        try:
            # Process special flags
            no_cache = False
            use_extended_timeout = False
            original_timeout = None
            
            # Handle /nocache flag
            if user_query.startswith('nocache '):
                no_cache = True
                user_query = user_query[8:].strip()
                print("ℹ️  Cache disabled for this query")
            
            # Handle /notimeout flag
            if user_query.startswith('notimeout '):
                use_extended_timeout = True
                user_query = user_query[10:].strip()
                original_timeout = self.ollama_config["timeout"]
                self.ollama_config["timeout"] = 0  # Disable timeout
                # Update aiohttp session timeout
                await self.aiohttp_session.close()
                self.aiohttp_session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=0)  # 0 means no timeout
                )
                print("ℹ️  Timeout disabled - will wait indefinitely for response")
                
            try:
                # Store the query in the database
                self.db.set_setting("last_user_query", user_query)
                self.last_user_query = user_query
                
                # Memory check in low memory mode
                if self.low_memory_mode:
                    await self.check_memory_usage()
                
                # Process query with the LLM
                print("\n⏳ Processing query...")
                
                # Show a progress indicator for mobile UX
                progress_task = None
                if not self.debug_mode:
                    progress_task = asyncio.create_task(self._show_progress())
                
                # Get response from LLM
                response, was_cached = await self.process_message(user_query, no_cache=no_cache)
                
                # Cancel progress indicator if active
                if progress_task and not progress_task.done():
                    progress_task.cancel()
                    try:
                        await progress_task
                    except asyncio.CancelledError:
                        pass
                    # Clear the progress line
                    print("\r" + " " * 50 + "\r", end="", flush=True)
                
                # Process response
                if was_cached:
                    print("📋 Using cached response")
                
                # Store in database if not empty
                if not user_query.startswith('/'):
                    self.db.add_message("user", user_query)
                    # Only add non-empty responses to the database
                    if response:
                        self.db.add_message("assistant", response)
                
                # Format and display response
                if response:
                    # Apply text wrapping for better mobile display
                    if hasattr(self, 'wrap_text'):
                        # Use a more subtle indicator for wrapped text
                        print(f"\n{response}")
                    else:
                        print(f"\n{response}")
                
            finally:
                # Restore timeout if it was changed
                if use_extended_timeout and original_timeout is not None:
                    self.ollama_config["timeout"] = original_timeout
                    # Restore aiohttp session with default timeout
                    await self.aiohttp_session.close()
                    self.aiohttp_session = aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=self.default_timeout)
                    )
                    
        except Exception as e:
            self.logger.error(f"Error processing query: {e}", exc_info=True)
            print(f"\n❌ Error: {str(e)}")
            
            # Suggest help for common errors
            if "timed out" in str(e).lower() or "timeout" in str(e).lower():
                print("\nℹ️ Tip: Use '/notimeout' before your query to disable timeouts for slow connections.")
            elif "memory" in str(e).lower():
                print("\nℹ️ Tip: Your device may be low on memory. Try closing other apps or use shorter conversations.")
    
    async def _show_progress(self):
        """Show a simple progress indicator."""
        indicators = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]
        i = 0
        try:
            while True:
                print(f"\r⏳ Processing {indicators[i]} ", end="", flush=True)
                i = (i + 1) % len(indicators)
                await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            # Clear the line when cancelled
            print("\r" + " " * 50 + "\r", end="", flush=True)
            raise

if __name__ == "__main__":
    asyncio.run(MinimalAIAgent(model=LLM_MODEL).run())


