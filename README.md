# CellBot for NetHunter

A command-line AI assistant optimized for Kali NetHunter running on the OnePlus 12 with OxygenOS 15.

## Features

- **Lightweight & Mobile-Friendly**: Optimized for NetHunter's terminal and mobile resources
- **Code Execution**: Run Python and shell scripts directly in the terminal
- **Conversation History**: Persistent storage of conversations with searchable history
- **Web Search**: Limited web search capabilities when network is available
- **Auto-detection**: Automatically detects device capabilities for optimal performance
- **Mobile Optimizations**: Database connection pooling, caching, and resource-aware processing
- **Virtual Environment**: Automatically creates and uses a Python virtual environment for isolated dependencies

## Installation

### Prerequisites

- Kali NetHunter installed on OnePlus 12
- Python 3.10+ installed (`python3 --version` to check)
- pip3 installed (`pip3 --version` to check)

### Option 1: Automatic Installation

If you have the code on your desktop/laptop, you can use the transfer script:

1. Run the transfer script from your computer:
   ```bash
   ./transfer_to_nethunter.sh
   ```

2. Follow the prompts to enter your NetHunter device's IP address.

### Option 2: Manual Installation

1. Create a directory for CellBot:
   ```bash
   mkdir -p ~/nethunter_cellbot/agent ~/nethunter_cellbot/generated_code
   ```

2. Copy all files to your NetHunter device using any transfer method (USB, ADB, Network share)

3. Use the setup script to create a virtual environment and install dependencies:
   ```bash
   cd ~/nethunter_cellbot
   ./setup.py setup
   ```

## Testing Your Environment

Before running CellBot, you can verify your environment is properly set up:

```bash
./setup.py info
```

This will show:
- Python version being used
- Virtual environment location
- Installed dependencies

For more detailed system checks:
```bash
python3 test_nethunter_env.py
```

## Usage

### Using the Simplified Launcher

The easiest way to run CellBot is with the main launcher script:

```bash
./cellbot.py
```

This automatically:
1. Creates a virtual environment if it doesn't exist
2. Installs required dependencies
3. Runs CellBot using the virtual environment

### Running with Options

To run with specific options:

```bash
./cellbot.py run.py --model mistral:7b --threads 4 --temperature 0.7 --tokens 2048
```

### Advanced Usage with Setup Script

For more control, you can use the setup script directly:

```bash
# Set up the virtual environment
./setup.py setup

# Run CellBot
./setup.py run run.py

# Show environment information
./setup.py info
```

### Manual Usage (Legacy)

If needed, you can still run without the setup script:

```bash
python3 nethunter_cellbot.py
```

Or with options:
```bash
python3 nethunter_cellbot.py --model mistral:7b --threads 4
```

## Command-Line Options

- `--model MODEL`: Specify the LLM model to use (default: mistral:7b)
- `--timeout SECONDS`: Response timeout in seconds (default: 180)
- `--max-llm-calls N`: Maximum concurrent LLM API calls (default: 2)
- `--max-code-execs N`: Maximum concurrent code executions (default: 2)
- `--debug`: Enable debug mode with verbose output
- `--save-code`: Save generated code blocks to files
- `--threads N`: Number of CPU threads for inference
- `--gpu-layers N`: Number of GPU layers to use (if available)
- `--temperature FLOAT`: Temperature for LLM responses (0.0-1.0, default: 0.5 on mobile)
- `--tokens N`: Maximum tokens to generate in responses (default: 1536 on mobile)

## Interactive Commands

Once CellBot is running, you can use these commands:

- `/help`: Display help information
- `/model [name]`: View or change the active model
- `/search [query]`: Search for information
- `/clear`: Clear the screen
- `/threads [N]`: Set CPU thread count
- `/gpu [N]`: Set GPU layer count
- `/temp [value]`: Set temperature (0.0-1.0) for response randomness
- `/tokens [N]`: Set maximum tokens to generate
- `/optimize`: Auto-optimize all settings for current device status
- `/battery`: Check battery status
- `/exit` or `/quit`: Exit CellBot

## NetHunter-Specific Optimizations

CellBot has been optimized for NetHunter environments in several ways:

1. **Resource Management**:
   - Adaptive thread usage based on device capabilities
   - Connection pooling for database operations
   - Caching of frequently accessed data
   - Reduced memory footprint for mobile execution

2. **Network Awareness**:
   - Retry mechanisms with exponential backoff for unstable connections
   - Reduced payload sizes for mobile data efficiency
   - Timeout handling optimized for mobile networks

3. **Storage Efficiency**:
   - Optimized database queries with early termination
   - Similarity search with mobile-friendly algorithms
   - Reduced disk I/O operations

4. **Battery Considerations**:
   - Background operations are limited to preserve battery
   - Efficient resource cleanup when idle
   - Adaptive temperature and token limit based on battery level
   - Auto-detection of optimal settings for device capabilities

5. **LLM Response Optimizations**:
   - Lower temperature settings for more deterministic and efficient responses
   - Reduced token generation limits to save battery and memory
   - Automatic adjustment based on available RAM and battery status

## Troubleshooting

### Common Issues

1. **Python Import Errors**:
   Make sure all dependencies are installed:
   ```bash
   pip3 install -r requirements.txt
   ```

2. **Permission Errors**:
   Ensure the scripts are executable:
   ```bash
   chmod +x nethunter_cellbot.py
   ```

3. **Database Errors**:
   If you encounter database errors, try resetting:
   ```bash
   rm ~/nethunter_cellbot/conversation.db
   ```

4. **Performance Issues**:
   Adjust thread count to match your device:
   ```bash
   python3 nethunter_cellbot.py --threads 2
   ```

5. **Network Connectivity**:
   If you're experiencing network issues:
   ```bash
   # Check if you have internet connectivity
   ping -c 3 8.8.8.8
   
   # If using Ollama locally, check if it's running
   curl http://127.0.0.1:11434/api/tags
   ```

6. **Storage Space**:
   Check available storage:
   ```bash
   df -h ~/nethunter_cellbot
   ```

### Logs

Check the log file for detailed error information:
```bash
cat ~/nethunter_cellbot/cellbot.log
```

You can also enable debug mode for more verbose logging:
```bash
python3 nethunter_cellbot.py --debug
```

## Advanced Configuration

### Custom System Prompt

You can customize the system prompt by editing:
```bash
nano ~/nethunter_cellbot/system-prompt.txt
```

### Android-Specific Settings

Android-specific configurations are in:
```bash
nano ~/nethunter_cellbot/agent/android_config.py
```

## License

This software is provided as-is, without warranty. Use at your own risk. 