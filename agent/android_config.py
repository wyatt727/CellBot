"""
Android-specific configuration for CellBot running on NetHunter.
This file contains settings optimized for the OnePlus 12 with OxygenOS 15.
"""
import os
import platform
import logging
import subprocess
from typing import Dict, Any, Optional, Tuple

# Setup logging
logger = logging.getLogger(__name__)

# Base paths
HOME_DIR = os.path.expanduser("~")
NETHUNTER_BASE_DIR = os.path.join(HOME_DIR, "nethunter_cellbot")
DB_FILE = os.path.join(NETHUNTER_BASE_DIR, "conversation.db")
SYSTEM_PROMPT_PATH = os.path.join(NETHUNTER_BASE_DIR, "system-prompt.txt")
LOG_PATH = os.path.join(NETHUNTER_BASE_DIR, "cellbot.log")
GENERATED_CODE_DIR = os.path.join(NETHUNTER_BASE_DIR, "generated_code")

# API configuration
API_BASE_URL = os.environ.get("CELLBOT_API_URL", "http://localhost:11434/api")
DEFAULT_MODEL = os.environ.get("CELLBOT_MODEL", "mistral:7b")

# Performance settings
DEFAULT_THREADS = 2  # Conservative default for mobile
DEFAULT_GPU_LAYERS = 0  # Default to CPU-only
MAX_CONCURRENT_LLM_CALLS = 2
MAX_CONCURRENT_CODE_EXECS = 2
DEFAULT_TIMEOUT = 180  # seconds
MAX_RESPONSE_TOKENS = 2048  # Limit token count for mobile
DEFAULT_TEMPERATURE = 0.5  # More deterministic responses for mobile
DEFAULT_NUM_PREDICT = 1536  # Limit token generation for mobile battery savings

# Database settings
DB_CONNECTION_TIMEOUT = 5  # seconds
DB_CONNECTION_POOL_SIZE = 3
DB_MAX_SIMILARITY_RESULTS = 5

# Network settings
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_FACTOR = 1.5
NETWORK_TIMEOUT = 30  # seconds

# Device information cache
_device_info_cache: Optional[Dict[str, Any]] = None

def get_device_info() -> Dict[str, Any]:
    """
    Get information about the device.
    Returns cached information if available.
    """
    global _device_info_cache
    
    if _device_info_cache is not None:
        return _device_info_cache
    
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "is_nethunter": False,
        "device_model": "Unknown",
        "android_version": "Unknown",
        "ram_mb": 0,
        "cpu_cores": 0
    }
    
    # Check if running on NetHunter
    nethunter_paths = [
        "/data/data/com.offsec.nethunter/files/home",
        "/data/data/com.termux/files/home"
    ]
    
    for path in nethunter_paths:
        if os.path.exists(path):
            info["is_nethunter"] = True
            break
    
    # Try to get Android device information
    try:
        # Get device model
        if os.path.exists("/system/build.prop"):
            with open("/system/build.prop", "r") as f:
                for line in f:
                    if "ro.product.model" in line:
                        info["device_model"] = line.split("=")[1].strip()
                    elif "ro.build.version.release" in line:
                        info["android_version"] = line.split("=")[1].strip()
        
        # Get RAM info
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if "MemTotal" in line:
                        # Convert KB to MB
                        info["ram_mb"] = int(line.split()[1]) // 1024
                        break
        except Exception as e:
            logger.warning(f"Failed to get RAM info: {e}")
        
        # Get CPU info
        try:
            with open("/proc/cpuinfo", "r") as f:
                info["cpu_cores"] = f.read().count("processor")
        except Exception as e:
            logger.warning(f"Failed to get CPU info: {e}")
            
        # If we couldn't get CPU cores, try another method
        if info["cpu_cores"] == 0:
            try:
                result = subprocess.run(
                    ["nproc"], 
                    capture_output=True, 
                    text=True, 
                    check=True
                )
                info["cpu_cores"] = int(result.stdout.strip())
            except Exception as e:
                logger.warning(f"Failed to get CPU cores using nproc: {e}")
                # Default to 2 cores if we can't determine
                info["cpu_cores"] = 2
    
    except Exception as e:
        logger.warning(f"Error getting device info: {e}")
    
    # Cache the results
    _device_info_cache = info
    return info

def get_optimal_thread_count() -> int:
    """
    Determine the optimal thread count based on device capabilities.
    For mobile devices, we want to be conservative to avoid overheating.
    """
    device_info = get_device_info()
    cpu_cores = device_info.get("cpu_cores", 0)
    
    if cpu_cores <= 0:
        return DEFAULT_THREADS
    
    # For mobile, use at most half the cores, but at least 1
    if device_info.get("is_nethunter", False):
        return max(1, min(cpu_cores // 2, 4))
    
    # For desktop/server, use more cores
    return max(1, min(cpu_cores - 1, 8))

def get_optimal_gpu_layers() -> int:
    """
    Determine if GPU acceleration should be used.
    For mobile, we're conservative to avoid overheating.
    """
    # For now, default to CPU-only on mobile
    device_info = get_device_info()
    if device_info.get("is_nethunter", False):
        return 0
    
    # For desktop/server, try to use GPU if available
    return 32  # Default for desktop

def get_network_settings() -> Dict[str, Any]:
    """
    Get network-related settings based on device capabilities.
    """
    device_info = get_device_info()
    
    # More conservative settings for mobile
    if device_info.get("is_nethunter", False):
        return {
            "timeout": NETWORK_TIMEOUT,
            "retry_attempts": RETRY_ATTEMPTS,
            "retry_backoff_factor": RETRY_BACKOFF_FACTOR,
            "max_concurrent_requests": 2
        }
    
    # More aggressive settings for desktop/server
    return {
        "timeout": 60,
        "retry_attempts": 5,
        "retry_backoff_factor": 1.2,
        "max_concurrent_requests": 5
    }

def get_llm_settings() -> Dict[str, Any]:
    """
    Get LLM-related settings based on device capabilities.
    """
    device_info = get_device_info()
    
    # Base settings
    settings = {
        "model": DEFAULT_MODEL,
        "timeout": DEFAULT_TIMEOUT,
        "max_tokens": MAX_RESPONSE_TOKENS,
        "temperature": 0.7,
        "top_p": 0.9,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0
    }
    
    # Adjust for mobile
    if device_info.get("is_nethunter", False):
        # More conservative settings for mobile
        settings.update({
            "temperature": DEFAULT_TEMPERATURE,  # More deterministic responses
            "max_tokens": DEFAULT_NUM_PREDICT,  # Shorter responses to save resources
            "timeout": 120       # Shorter timeout
        })
    
    return settings

def get_optimal_llm_parameters() -> Dict[str, Any]:
    """
    Get optimal LLM parameters based on device capabilities.
    Dynamically adjusts temperature and num_predict based on
    available memory, battery status, and device type.
    
    Returns:
        Dict with optimized settings
    """
    device_info = get_device_info()
    
    # Start with default parameters
    params = {
        "temperature": DEFAULT_TEMPERATURE,
        "num_predict": DEFAULT_NUM_PREDICT
    }
    
    # If not on mobile, use higher token limits and more neutral temperature
    if not device_info.get("is_nethunter", False):
        params["temperature"] = 0.7
        params["num_predict"] = 2048
        return params
    
    # For mobile, check available memory and adjust
    ram_mb = device_info.get("ram_mb", 0)
    if ram_mb > 0:
        # If low memory, reduce token limit further
        if ram_mb < 4000:  # Less than 4GB RAM
            params["num_predict"] = 1024
        elif ram_mb > 8000:  # More than 8GB RAM
            params["num_predict"] = 1536
    
    # Try to detect battery status
    try:
        if os.path.exists("/sys/class/power_supply/battery/capacity"):
            with open("/sys/class/power_supply/battery/capacity", "r") as f:
                battery_level = int(f.read().strip())
                
            # If battery is low, use more conservative settings
            if battery_level < 30:
                params["num_predict"] = min(params["num_predict"], 1024)
                params["temperature"] = 0.3  # More deterministic to save compute
    except Exception:
        # If we can't read battery, just use defaults
        pass
    
    return params

def create_directories() -> None:
    """
    Create necessary directories if they don't exist.
    """
    os.makedirs(NETHUNTER_BASE_DIR, exist_ok=True)
    os.makedirs(GENERATED_CODE_DIR, exist_ok=True)
    
    # Create agent directory if it doesn't exist
    agent_dir = os.path.join(NETHUNTER_BASE_DIR, "agent")
    os.makedirs(agent_dir, exist_ok=True)

# Create directories when module is imported
create_directories() 