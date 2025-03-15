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
SYSTEM_PROMPT_PATH = os.path.join(NETHUNTER_BASE_DIR, "system-prompt.txt")
LOG_PATH = os.path.join(NETHUNTER_BASE_DIR, "cellbot.log")
GENERATED_CODE_DIR = os.path.join(NETHUNTER_BASE_DIR, "generated_code")

# API configuration
API_BASE_URL = os.environ.get("CELLBOT_API_URL", "http://localhost:11434/api")
# Select a smaller default model for mobile devices
if platform.system() == "Linux" and (
    "android" in platform.platform().lower() or
    os.path.exists("/system/build.prop") or
    os.path.exists("/data/data/com.termux")
):
    # For Android/NetHunter, use Mistral 7B by default
    DEFAULT_MODEL = os.environ.get("CELLBOT_MODEL", "mistral:7b")
    logger.info("Mobile device detected, using default model: mistral:7b")
else:
    # For desktop/server systems, use the standard model
    DEFAULT_MODEL = os.environ.get("CELLBOT_MODEL", "mistral:7b")

# Performance settings
DEFAULT_THREADS = 8  # Conservative default for mobile
DEFAULT_GPU_LAYERS = 0  # Default to CPU-only
MAX_CONCURRENT_LLM_CALLS = 2
MAX_CONCURRENT_CODE_EXECS = 2
DEFAULT_TIMEOUT = 180  # seconds
MAX_RESPONSE_TOKENS = 500  # Limit token count for mobile
DEFAULT_TEMPERATURE = 0.3  # More deterministic responses for mobile
DEFAULT_NUM_PREDICT = 250  # Limit token generation for mobile battery savings

# Network settings
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_FACTOR = 1.5
NETWORK_TIMEOUT = 700  # seconds

# Context settings
CONTEXT_MSG_COUNT = 5
MAX_SIMILAR_EXAMPLES = 1
SIMILARITY_THRESHOLD = 0.94

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
            
            # macOS alternative for RAM info
            if platform.system() == "Darwin":
                try:
                    result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True)
                    # Convert bytes to MB
                    info["ram_mb"] = int(result.stdout.strip()) // (1024 * 1024)
                except Exception as e:
                    logger.warning(f"Failed to get macOS RAM info: {e}")
        
        # Get CPU info
        try:
            with open("/proc/cpuinfo", "r") as f:
                info["cpu_cores"] = f.read().count("processor")
        except Exception as e:
            logger.warning(f"Failed to get CPU info: {e}")
            
            # macOS alternative for CPU info
            if platform.system() == "Darwin":
                try:
                    result = subprocess.run(["sysctl", "-n", "hw.ncpu"], capture_output=True, text=True, check=True)
                    info["cpu_cores"] = int(result.stdout.strip())
                except Exception as e:
                    logger.warning(f"Failed to get macOS CPU info: {e}")
        
        # If we couldn't get CPU cores, try another method
        if info["cpu_cores"] == 0:
            try:
                # Try using nproc (available on many Unix systems)
                result = subprocess.run(["nproc"], capture_output=True, text=True, check=True)
                info["cpu_cores"] = int(result.stdout.strip())
            except Exception as e:
                logger.warning(f"Failed to get CPU cores using nproc: {e}")
                # Default to 2 cores as a fallback
                info["cpu_cores"] = os.cpu_count() or 2
    
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
    # Base settings
    settings = {
        "model": DEFAULT_MODEL,
        "timeout": DEFAULT_TIMEOUT,
        "max_tokens": MAX_RESPONSE_TOKENS,
        "temperature": DEFAULT_TEMPERATURE,  # Use the 0.3 default
        "top_p": 0.9,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "num_predict": DEFAULT_NUM_PREDICT  # Use the 500 default
    }
    
    # We no longer adjust these settings based on device type
    # All systems use the same defaults now
    
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
        "temperature": DEFAULT_TEMPERATURE,  # Use 0.3 as the default temperature
        "num_predict": DEFAULT_NUM_PREDICT   # Use 500 as the default token limit
    }
    
    # Non-mobile systems always use the default values now
    if not device_info.get("is_nethunter", False):
        return params
    
    # For mobile, check available memory and adjust
    ram_mb = device_info.get("ram_mb", 0)
    if ram_mb > 0:
        # If very low memory, reduce token limit further
        if ram_mb < 2000:  # Less than 2GB RAM
            params["num_predict"] = min(params["num_predict"], 300)
    
    # Try to detect battery status
    try:
        if os.path.exists("/sys/class/power_supply/battery/capacity"):
            with open("/sys/class/power_supply/battery/capacity", "r") as f:
                battery_level = int(f.read().strip())
                
            # If battery is critical, use even more conservative settings
            if battery_level < 15:
                params["num_predict"] = min(params["num_predict"], 300)
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