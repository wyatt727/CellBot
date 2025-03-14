#!/bin/bash
#
# CellBot for NetHunter - Build Script
# This script packages the CellBot application for deployment on NetHunter devices
#

set -e  # Exit on error

echo "╭───────────────────────────────────────────────╮"
echo "│        CellBot for NetHunter Build            │"
echo "│        Version: 1.0                           │"
echo "╰───────────────────────────────────────────────╯"

# Create build directory
BUILD_DIR="build_nethunter"
PACKAGE_NAME="cellbot_nethunter_v1.0.tar.gz"

echo "[+] Creating build directory..."
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR/agent"
mkdir -p "$BUILD_DIR/scripts"
mkdir -p "$BUILD_DIR/docs"

# Copy core files
echo "[+] Copying core files..."
cp -v agent/agent.py "$BUILD_DIR/agent/"
cp -v agent/db.py "$BUILD_DIR/agent/"
cp -v agent/llm_client.py "$BUILD_DIR/agent/"
cp -v agent/system_prompt.py "$BUILD_DIR/agent/"
cp -v agent/nethunter_main.py "$BUILD_DIR/agent/"
cp -v agent/android_config.py "$BUILD_DIR/agent/"
cp -v agent/__init__.py "$BUILD_DIR/agent/"
cp -v agent/config.py "$BUILD_DIR/agent/"
cp -v agent/code_executor.py "$BUILD_DIR/agent/"

# Copy scripts
echo "[+] Copying scripts..."
cp -v run_cellbot_remote.sh "$BUILD_DIR/scripts/"
cp -v run_in_nethunter.sh "$BUILD_DIR/scripts/"
cp -v install_nethunter.sh "$BUILD_DIR/scripts/"
cp -v test_nethunter_env.py "$BUILD_DIR/scripts/"

# Copy documentation
echo "[+] Copying documentation..."
cp -v README.md "$BUILD_DIR/docs/"
cp -v system-prompt.txt "$BUILD_DIR/docs/"
cp -v requirements.txt "$BUILD_DIR/"

# Create installation scripts
echo "[+] Creating installation scripts..."

cat > "$BUILD_DIR/install.sh" << 'EOF'
#!/bin/bash
#
# CellBot for NetHunter - Installation Script
#

echo "╭───────────────────────────────────────────────╮"
echo "│        CellBot for NetHunter Installer        │"
echo "│        Version: 1.0                           │"
echo "╰───────────────────────────────────────────────╯"

# Determine environment
NETHUNTER_CHROOT="/data/local/nhsystem/kali-arm64"
TERMUX_HOME="/data/data/com.termux/files/home"

if [ -d "$NETHUNTER_CHROOT" ]; then
    echo "[+] NetHunter environment detected"
    INSTALL_DIR="$NETHUNTER_CHROOT/root/nethunter_cellbot"
    ENV_TYPE="nethunter"
elif [ -d "$TERMUX_HOME" ]; then
    echo "[+] Termux environment detected"
    INSTALL_DIR="$TERMUX_HOME/nethunter_cellbot"
    ENV_TYPE="termux"
else
    echo "[+] Standard Linux environment detected"
    INSTALL_DIR="$HOME/nethunter_cellbot"
    ENV_TYPE="standard"
fi

echo "[+] Installing to: $INSTALL_DIR"

# Create installation directory
mkdir -p "$INSTALL_DIR"

# Copy files
echo "[+] Copying files..."
cp -rv agent "$INSTALL_DIR/"
cp -rv scripts "$INSTALL_DIR/"
cp -rv docs "$INSTALL_DIR/"
cp -v requirements.txt "$INSTALL_DIR/"

# Make scripts executable
chmod +x "$INSTALL_DIR/scripts/"*.sh

# Create launch script
cat > "$INSTALL_DIR/cellbot.sh" << 'EOL'
#!/bin/bash
cd "$(dirname "$0")"
export CELLBOT_HOME="$(pwd)"
export PYTHONPATH="$CELLBOT_HOME:$PYTHONPATH"
python3 agent/nethunter_main.py "$@"
EOL

chmod +x "$INSTALL_DIR/cellbot.sh"

# Install dependencies
echo "[+] Installing Python dependencies..."
if [ "$ENV_TYPE" = "termux" ]; then
    # Install Termux packages
    pkg install -y python clang python-dev
    pip install --upgrade pip
    pip install -r "$INSTALL_DIR/requirements.txt"
    
    # Install Termux API if not present
    pkg install -y termux-api
    
    # Create a convenient alias
    echo 'alias cellbot="cd ~/nethunter_cellbot && ./cellbot.sh"' >> ~/.bashrc
    echo 'alias cellbot="cd ~/nethunter_cellbot && ./cellbot.sh"' >> ~/.zshrc 2>/dev/null || true
    
elif [ "$ENV_TYPE" = "nethunter" ]; then
    # Install in NetHunter chroot
    apt update
    apt install -y python3 python3-pip
    pip3 install -r "$INSTALL_DIR/requirements.txt"
    
    # Create a convenient alias
    echo 'alias cellbot="cd ~/nethunter_cellbot && ./cellbot.sh"' >> ~/.bashrc
    echo 'alias cellbot="cd ~/nethunter_cellbot && ./cellbot.sh"' >> ~/.zshrc 2>/dev/null || true
    
else
    # Standard Linux install
    pip3 install -r "$INSTALL_DIR/requirements.txt"
    
    # Create a convenient alias
    echo 'alias cellbot="cd ~/nethunter_cellbot && ./cellbot.sh"' >> ~/.bashrc
    echo 'alias cellbot="cd ~/nethunter_cellbot && ./cellbot.sh"' >> ~/.zshrc 2>/dev/null || true
fi

echo "[+] Installation complete!"
echo "[+] To start CellBot, run: cd $INSTALL_DIR && ./cellbot.sh"
echo "[+] Or simply use the 'cellbot' command after restarting your shell"
EOF

chmod +x "$BUILD_DIR/install.sh"

# Create a README for the package
cat > "$BUILD_DIR/README.txt" << 'EOF'
CellBot for NetHunter v1.0
==========================

An AI assistant optimized for mobile terminals and NetHunter devices.

Installation
-----------
1. Extract this package
2. Run ./install.sh

Key Features
-----------
- Mobile-optimized terminal interface
- NetHunter commands integration
- Clipboard support
- Memory management for mobile devices
- Battery status monitoring
- Text wrapping for small screens

Type /help in CellBot to see all available commands.

Requirements
-----------
- Python 3.8 or higher
- 500MB+ free memory
- Mistral 7B model (recommended)
- Network connectivity for remote API mode

For questions or issues, see the docs/ directory.
EOF

# Package it all up
echo "[+] Creating deployment package..."
tar -czvf "$PACKAGE_NAME" -C "$BUILD_DIR" .

echo "[+] Build complete!"
echo "[+] Package created: $PACKAGE_NAME"
echo "[+] Package size: $(du -h "$PACKAGE_NAME" | cut -f1)"
echo
echo "To deploy:"
echo "1. Transfer $PACKAGE_NAME to your NetHunter device"
echo "2. Extract with: tar -xzvf $PACKAGE_NAME"
echo "3. Install with: ./install.sh" 