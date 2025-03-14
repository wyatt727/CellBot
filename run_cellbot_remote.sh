#!/bin/bash
# Run CellBot with a remote API

# Set environment variables for the remote API
export CELLBOT_API_URL="http://localhost:11434/api"
export CELLBOT_MODEL="mistral:7b"

# Other options
MODEL="mistral:7b"
THREADS="2"
TIMEOUT="180"
DEBUG="--debug"

# Display options
echo "Starting CellBot with the following options:"
echo "  API URL: $CELLBOT_API_URL"
echo "  Model: $MODEL"
echo "  Threads: $THREADS"
echo "  Timeout: $TIMEOUT"
echo "  Debug mode: Yes"

# Run CellBot
cd ~/nethunter_cellbot
python3 nethunter_cellbot.py --model $MODEL --threads $THREADS --timeout $TIMEOUT $DEBUG 