#!/bin/bash
# Build script for CellBot for NetHunter

set -e  # Exit on error

echo "Building CellBot for NetHunter..."

# Create build directory
BUILD_DIR="build_nethunter"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Copy necessary files
echo "Copying files to build directory..."
cp -r agent "$BUILD_DIR/"
cp -r README.md "$BUILD_DIR/"
cp -r requirements.txt "$BUILD_DIR/"

# Create the chroot-specific run script
cat > "$BUILD_DIR/run_cellbot.sh" << 'EOF'
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
EOF

# Make it executable
chmod +x "$BUILD_DIR/run_cellbot.sh"

# Create a basic README
cat > "$BUILD_DIR/INSTALL.md" << 'EOF'
# CellBot for NetHunter Installation

## Installation Steps

1. Extract the archive to your Kali NetHunter chroot
   ```
   tar -xzf cellbot_nethunter.tar.gz -C /path/to/extract
   ```

2. Install dependencies
   ```
   cd /path/to/extract
   pip install -r requirements.txt
   ```

3. Run CellBot
   ```
   ./run_cellbot.sh
   ```

## Troubleshooting

- If you encounter a disk I/O error with the database, you may need to set the `CELLBOT_DB_PATH` environment variable to a writable location:
  ```
  export CELLBOT_DB_PATH=/tmp/cellbot.db
  ./run_cellbot.sh
  ```
EOF

# Create a tar.gz file
echo "Creating archive..."
ARCHIVE_NAME="cellbot_nethunter_v1.0.tar.gz"
tar -czf "$ARCHIVE_NAME" -C "$BUILD_DIR" .

echo "Build complete: $ARCHIVE_NAME"
echo "Size: $(du -h "$ARCHIVE_NAME" | cut -f1)" 