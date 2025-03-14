#!/usr/bin/env python3
"""
CellBot - NetHunter AI Assistant
A command-line AI agent optimized for Kali NetHunter on OnePlus 12.
"""
import asyncio
import os
import sys
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.expanduser("~/nethunter_cellbot/cellbot.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("cellbot")

# Add the script's directory to PATH to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

async def main():
    """Main entry point with enhanced error handling."""
    logger.info("CellBot for NetHunter starting...")
    
    try:
        # Import the main function from the nethunter_main module
        from agent.nethunter_main import main
        await main()
    except KeyboardInterrupt:
        logger.info("CellBot terminated by user")
        print("\n\n🛑 CellBot terminated by user.")
    except ImportError as e:
        logger.error(f"Import error: {e}\n{traceback.format_exc()}")
        print(f"\n❌ Import error: {e}")
        print("This could be due to missing dependencies. Check the log file for details.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}\n{traceback.format_exc()}")
        print(f"\n❌ Fatal error: {e}")
        print("Check the log file for details.")
        sys.exit(1)
    finally:
        logger.info("CellBot session ended")

if __name__ == "__main__":
    # Print a small banner
    print("\n🔒 CellBot for NetHunter - Starting up...\n")
    
    # Run the main async function
    asyncio.run(main()) 