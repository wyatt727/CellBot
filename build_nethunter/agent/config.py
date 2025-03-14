# agent/config.py
import os

# Path to the system prompt file (should be in the project root)
SYSTEM_PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "system-prompt.txt")

# LLM API configuration
OLLAMA_API_BASE = "http://127.0.0.1:11434"  # Base URL for Ollama API

LLM_MODEL = "deepseek-coder-v2"

# Execution parameters
MAX_FIX_ATTEMPTS = 2

# Number of similar examples to include in context
MAX_SIMILAR_EXAMPLES = 1

# Number of context messages to include
CONTEXT_MSG_COUNT = 5

# Performance Configuration
SIMILARITY_THRESHOLD = 0.94  # Threshold for using cached response directly
DEBUG_MODE = False  # Set to True to show detailed timing information
MAX_CONCURRENT_LLM_CALLS = 3
MAX_CONCURRENT_CODE_EXECUTIONS = 3
RESPONSE_TIMEOUT = 120

# Database configuration
DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "conversation.db")

# Directory for generated code files
GENERATED_CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "generated_code")

# New configuration
SAVE_CODE_BLOCKS = False  # Set to True to save code blocks permanently
