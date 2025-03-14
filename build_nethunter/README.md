# CellBot for NetHunter

A command-line AI assistant optimized for Kali NetHunter running on the OnePlus 12 with OxygenOS 15.

## Features

- **Lightweight & Mobile-Friendly**: Optimized for NetHunter's terminal and mobile resources
- **Code Execution**: Run Python and shell scripts directly in the terminal
- **Conversation History**: Persistent storage of conversations with searchable history
- **Web Search**: Limited web search capabilities when network is available
- **Auto-detection**: Automatically detects device capabilities for optimal performance
- **Mobile Optimizations**: Database connection pooling, caching, and resource-aware processing

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

3. Install dependencies:
   ```bash
   cd ~/nethunter_cellbot
   pip3 install -r requirements.txt
   ```

## Testing Your Environment

Before running CellBot, you can verify your environment is properly set up:

```bash
python3 test_nethunter_env.py
```

This script will check:
- Python version compatibility
- NetHunter-specific paths
- Required dependencies
- File system access
- Network connectivity
- Project structure
- Ollama availability (if using local models)

The test will provide a summary of results and next steps based on what it finds.

## Usage

1. Navigate to the CellBot directory:
   ```bash
   cd ~/nethunter_cellbot
   ```

2. Run CellBot:
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

## Interactive Commands

Once CellBot is running, you can use these commands:

- `/help`: Display help information
- `/model [name]`: View or change the active model
- `/search [query]`: Search for information
- `/clear`: Clear the screen
- `/threads [N]`: Set CPU thread count
- `/gpu [N]`: Set GPU layer count
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