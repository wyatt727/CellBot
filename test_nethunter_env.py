#!/usr/bin/env python3
"""
NetHunter CellBot Environment Test
This script tests the NetHunter environment to ensure it's properly set up for CellBot.
"""
import os
import sys
import logging
import platform
import tempfile
import importlib.util
import subprocess
import asyncio

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test")

def check_python_version():
    """Check if Python version is adequate."""
    version = sys.version_info
    logger.info(f"Python version: {platform.python_version()}")
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        logger.error("❌ Python 3.7+ is required")
        return False
    else:
        logger.info("✅ Python version OK")
        return True

def check_environment():
    """Check basic environment settings."""
    logger.info(f"Platform: {platform.platform()}")
    
    # Check if running on Linux/Android
    if platform.system() != 'Linux':
        logger.warning("⚠️ Not running on Linux - this might not be a NetHunter environment")
    else:
        logger.info("✅ Running on Linux")
    
    # Check user home directory
    home_dir = os.path.expanduser("~")
    logger.info(f"Home directory: {home_dir}")
    
    # Check for NetHunter-specific paths
    nethunter_paths = [
        "/data/data/com.offsec.nethunter/files/home",
        "/data/data/com.termux/files/home"
    ]
    
    is_nethunter = False
    for path in nethunter_paths:
        if os.path.exists(path):
            logger.info(f"✅ Found NetHunter path: {path}")
            is_nethunter = True
    
    if not is_nethunter:
        logger.warning("⚠️ No NetHunter-specific paths found")
    
    return True

def check_dependencies():
    """Check if required dependencies are installed."""
    required_modules = [
        "aiohttp", 
        "aiofiles", 
        "asyncio", 
        "sqlite3",
        "readline"
    ]
    
    all_installed = True
    for module in required_modules:
        try:
            importlib.import_module(module)
            logger.info(f"✅ {module} is installed")
        except ImportError:
            logger.error(f"❌ {module} is NOT installed")
            all_installed = False
    
    return all_installed

def check_file_access():
    """Check if we can create and access files."""
    try:
        # Try to create a temporary file
        with tempfile.NamedTemporaryFile(mode='w+t', delete=False) as temp:
            temp_path = temp.name
            temp.write("CellBot test file")
        
        # Check if we can read it
        with open(temp_path, 'r') as temp:
            content = temp.read()
            if content == "CellBot test file":
                logger.info("✅ File creation and access test passed")
                os.unlink(temp_path)
                return True
            else:
                logger.error("❌ File content verification failed")
    except Exception as e:
        logger.error(f"❌ File access test failed: {e}")
    
    return False

def check_network():
    """Check if we have network connectivity."""
    try:
        # Try to ping Google's DNS
        result = subprocess.run(
            ["ping", "-c", "1", "-W", "2", "8.8.8.8"], 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        )
        if result.returncode == 0:
            logger.info("✅ Network connectivity test passed")
            return True
        else:
            logger.warning("⚠️ Network connectivity test failed - ping to 8.8.8.8 failed")
    except Exception as e:
        logger.error(f"❌ Network test error: {e}")
    
    return False

async def check_ollama():
    """Check if Ollama is installed and running."""
    try:
        # Try to make a basic request to Ollama
        import aiohttp
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get("http://127.0.0.1:11434/api/tags", timeout=2) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [model.get("name") for model in data.get("models", [])]
                        if models:
                            logger.info(f"✅ Ollama is running with models: {', '.join(models)}")
                        else:
                            logger.info("✅ Ollama is running but no models are loaded")
                        return True
                    else:
                        logger.warning(f"⚠️ Ollama responded with status {response.status}")
            except aiohttp.ClientError as e:
                logger.warning(f"⚠️ Ollama not accessible: {e}")
    except Exception as e:
        logger.error(f"❌ Ollama check error: {e}")
    
    return False

def check_project_structure():
    """Check if the project structure is correct."""
    project_dir = os.path.expanduser("~/nethunter_cellbot")
    
    # Check if the project directory exists
    if not os.path.exists(project_dir):
        logger.warning(f"⚠️ Project directory {project_dir} does not exist")
        logger.info("   Run 'mkdir -p ~/nethunter_cellbot' to create it")
        return False
    
    # Check if key files exist
    key_files = [
        "nethunter_cellbot.py",
        "agent/nethunter_main.py",
        "agent/android_config.py",
        "agent/agent.py",
        "agent/db.py",
        "agent/llm_client.py",
        "agent/code_executor.py",
        "system-prompt.txt"
    ]
    
    all_files_exist = True
    for file in key_files:
        file_path = os.path.join(project_dir, file)
        if os.path.exists(file_path):
            logger.info(f"✅ {file} exists")
        else:
            logger.warning(f"⚠️ {file} does not exist")
            all_files_exist = False
    
    return all_files_exist

async def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("CellBot NetHunter Environment Test")
    logger.info("=" * 60)
    
    # Run all checks
    checks = [
        ("Python Version", check_python_version()),
        ("Environment", check_environment()),
        ("Dependencies", check_dependencies()),
        ("File Access", check_file_access()),
        ("Network", check_network()),
        ("Project Structure", check_project_structure()),
        ("Ollama", await check_ollama())
    ]
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    
    all_passed = True
    for name, result in checks:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{name}: {status}")
        all_passed = all_passed and result
    
    logger.info("\nOverall result: " + ("✅ ALL CHECKS PASSED" if all_passed else "❌ SOME CHECKS FAILED"))
    
    # Provide next steps
    logger.info("\n" + "=" * 60)
    logger.info("Next Steps")
    logger.info("=" * 60)
    
    if all_passed:
        logger.info("You're ready to run CellBot!")
        logger.info("Run 'python3 nethunter_cellbot.py' to start")
    else:
        logger.info("Fix the issues above before running CellBot")
        logger.info("Check the README.md file for troubleshooting tips")
    
    logger.info("=" * 60)

if __name__ == "__main__":
    print("\n🔍 Running CellBot NetHunter Environment Test...\n")
    asyncio.run(main()) 