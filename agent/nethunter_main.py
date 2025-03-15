# agent/nethunter_main.py
import asyncio
import argparse
import logging
import os
import sys
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Try to import Android config
try:
    from .android_config import (
        get_optimal_thread_count, 
        get_optimal_gpu_layers,
        API_BASE_URL,
        DEFAULT_MODEL,
        MAX_CONCURRENT_LLM_CALLS,
        MAX_CONCURRENT_CODE_EXECS,
        DEFAULT_TIMEOUT,
        DEFAULT_TEMPERATURE,
        DEFAULT_NUM_PREDICT
    )
    logger.info("Android config loaded successfully")
except ImportError:
    logger.warning("Android config not found, using standard config")
    # Fallback to default values if android_config.py is not available
    API_BASE_URL = os.environ.get("CELLBOT_API_URL", "http://127.0.0.1:11434/api")
    DEFAULT_MODEL = os.environ.get("CELLBOT_MODEL", "mistral:7b")
    MAX_CONCURRENT_LLM_CALLS = 2
    MAX_CONCURRENT_CODE_EXECS = 2
    DEFAULT_TIMEOUT = 180
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_NUM_PREDICT = 1024
    
    # Fallback functions
    def get_optimal_thread_count():
        return 2
        
    def get_optimal_gpu_layers():
        return 0

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='NetHunter AI Agent')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        help=f'Model to use (default: {DEFAULT_MODEL})')
    parser.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT,
                        help=f'Response timeout in seconds (default: {DEFAULT_TIMEOUT})')
    parser.add_argument('--max-llm-calls', type=int, default=MAX_CONCURRENT_LLM_CALLS,
                        help=f'Max concurrent LLM API calls (default: {MAX_CONCURRENT_LLM_CALLS})')
    parser.add_argument('--max-code-execs', type=int, default=MAX_CONCURRENT_CODE_EXECS,
                        help=f'Max concurrent code executions (default: {MAX_CONCURRENT_CODE_EXECS})')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('--save-code', action='store_true',
                        help='Save generated code blocks to files')
    parser.add_argument('--threads', type=int, default=None,
                        help='CPU thread count for ollama')
    parser.add_argument('--gpu-layers', type=int, default=None,
                        help='GPU layer count for ollama')
    parser.add_argument('--temperature', type=float, default=None,
                        help=f'Temperature for LLM responses (0.0-1.0, default: {DEFAULT_TEMPERATURE})')
    parser.add_argument('--tokens', type=int, default=None,
                        help=f'Maximum tokens to generate in responses (default: {DEFAULT_NUM_PREDICT})')
    parser.add_argument('--no-db', action='store_true',
                        help='Disable database usage (use in-memory storage)')
                        
    return parser.parse_args()

async def main():
    """Main entry point for the CellBot AI Agent in NetHunter."""
    start_time = datetime.now()
    logger.info(f"CellBot for NetHunter starting at {start_time}")
    
    args = parse_arguments()
    
    try:
        # Import agent dynamically to avoid circular imports
        from .agent import MinimalAIAgent
        
        # Create agent with command line arguments
        agent = MinimalAIAgent(
            model=args.model,
            timeout=args.timeout,
            max_llm_calls=args.max_llm_calls,
            max_code_execs=args.max_code_execs,
            debug_mode=args.debug,
            save_code=args.save_code
        )
        
        # Set thread count and GPU layers if specified
        if args.threads is not None:
            agent.ollama_config["num_thread"] = args.threads
            agent.db.set_setting("ollama_num_thread", str(args.threads))
            
        if args.gpu_layers is not None:
            agent.ollama_config["num_gpu"] = args.gpu_layers
            agent.db.set_setting("ollama_num_gpu", str(args.gpu_layers))
            
        # Set temperature if specified
        if args.temperature is not None:
            if 0.0 <= args.temperature <= 1.0:
                agent.ollama_config["temperature"] = args.temperature
                agent.db.set_setting("ollama_temperature", str(args.temperature))
                logger.info(f"Temperature set to {args.temperature}")
            else:
                logger.warning(f"Temperature value {args.temperature} out of range (0.0-1.0), using default")
                
        # Set max tokens if specified
        if args.tokens is not None:
            if args.tokens > 0:
                agent.ollama_config["num_predict"] = args.tokens
                agent.db.set_setting("ollama_num_predict", str(args.tokens))
                logger.info(f"Max tokens set to {args.tokens}")
            else:
                logger.warning(f"Max tokens value {args.tokens} must be positive, using default")
            
        # Start the agent
        logger.info(f"Agent initialized with model: {args.model}")
        await agent.run()
        
    except KeyboardInterrupt:
        logger.info("CellBot terminated by user")
        print("\n\n🛑 CellBot terminated. Goodbye!")
        
    except Exception as e:
        logger.error(f"Error running CellBot: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        
    finally:
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"CellBot session ended. Duration: {duration}")

if __name__ == "__main__":
    asyncio.run(main()) 