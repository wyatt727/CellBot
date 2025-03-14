#!/bin/bash
# CellBot for NetHunter launcher script

CELLBOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$CELLBOT_DIR"

# Set environment variables
export API_URL=${API_URL:-"http://localhost:11434/api"}
export MODEL_NAME=${MODEL_NAME:-"mistral:7b"}
export NUM_THREADS=${NUM_THREADS:-2}
export TIMEOUT=${TIMEOUT:-180}
export DEBUG=${DEBUG:-1}

# Allow explicit override of database path
if [ ! -z "$CELLBOT_DB_PATH" ]; then
    echo "Using custom database path: $CELLBOT_DB_PATH"
    # Ensure directory exists
    DB_DIR=$(dirname "$CELLBOT_DB_PATH")
    mkdir -p "$DB_DIR" 2>/dev/null || true
    
    # Export for Python to use
    export CELLBOT_DB_PATH
fi

echo "Starting CellBot for NetHunter with model $MODEL_NAME..."
echo "API URL: $API_URL"

# Run the main script
python3 agent/nethunter_main.py --threads "$NUM_THREADS" --timeout "$TIMEOUT" --debug "$DEBUG"
