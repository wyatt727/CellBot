#!/bin/bash
# CellBot Installation Script

# Get the absolute path of the CellBot directory
CELLBOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CELLBOT_SCRIPT="${CELLBOT_DIR}/cellbot.py"

# Ensure the script is executable
chmod +x "${CELLBOT_SCRIPT}"
chmod +x "${CELLBOT_DIR}/setup.py"

# Make sure the virtual environment is set up
echo "Setting up CellBot virtual environment..."
"${CELLBOT_DIR}/setup.py" setup

# Create the symlink in /usr/local/bin or ~/bin if available
if [ -d "/usr/local/bin" ] && [ -w "/usr/local/bin" ]; then
    SYMLINK_PATH="/usr/local/bin/cellbot"
    # Check if symlink already exists
    if [ -e "${SYMLINK_PATH}" ]; then
        echo "Removing existing symlink..."
        rm "${SYMLINK_PATH}"
    fi
    echo "Creating symlink in /usr/local/bin..."
    ln -s "${CELLBOT_SCRIPT}" "${SYMLINK_PATH}"
    echo "CellBot installed! You can now run 'cellbot' from anywhere."
elif [ -d "${HOME}/bin" ]; then
    SYMLINK_PATH="${HOME}/bin/cellbot"
    # Check if symlink already exists
    if [ -e "${SYMLINK_PATH}" ]; then
        echo "Removing existing symlink..."
        rm "${SYMLINK_PATH}"
    fi
    echo "Creating symlink in ~/bin..."
    ln -s "${CELLBOT_SCRIPT}" "${SYMLINK_PATH}"
    echo "CellBot installed! You can now run 'cellbot' from anywhere."
else
    echo "Could not find a suitable directory for the symlink."
    echo "You can still run CellBot with: ${CELLBOT_SCRIPT}"
fi

echo ""
echo "Installation complete!"
echo "To run CellBot: cellbot"
echo "To run with options: cellbot run.py --model mistral:7b"
echo "For more information: cellbot --help" 