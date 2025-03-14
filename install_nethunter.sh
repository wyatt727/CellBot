#!/bin/bash
# CellBot NetHunter Installation Script

# ANSI color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}  CellBot for NetHunter Installation   ${NC}"
echo -e "${BLUE}=======================================${NC}"

# Check if running on Linux/Android
if [[ "$(uname -s)" != "Linux" ]]; then
    echo -e "${RED}Error: This script should be run on a Linux/Android device.${NC}"
    echo -e "${YELLOW}If you're trying to transfer files to NetHunter, use transfer_to_nethunter.sh instead.${NC}"
    exit 1
fi

# Check for Python 3.7+
echo -e "\n${BLUE}Checking Python version...${NC}"
if command -v python3 &>/dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [[ $PYTHON_MAJOR -lt 3 || ($PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 7) ]]; then
        echo -e "${RED}Error: Python 3.7+ is required. Found Python $PYTHON_VERSION${NC}"
        exit 1
    else
        echo -e "${GREEN}Found Python $PYTHON_VERSION - OK${NC}"
    fi
else
    echo -e "${RED}Error: Python 3 not found. Please install Python 3.7+${NC}"
    exit 1
fi

# Check for pip
echo -e "\n${BLUE}Checking pip installation...${NC}"
if command -v pip3 &>/dev/null; then
    echo -e "${GREEN}pip3 is installed - OK${NC}"
else
    echo -e "${YELLOW}Warning: pip3 not found. Attempting to install...${NC}"
    if command -v apt &>/dev/null; then
        sudo apt update && sudo apt install -y python3-pip
    elif command -v pkg &>/dev/null; then
        pkg install python-pip
    else
        echo -e "${RED}Error: Could not install pip. Please install pip manually.${NC}"
        exit 1
    fi
fi

# Create directory structure
echo -e "\n${BLUE}Creating directory structure...${NC}"
CELLBOT_DIR="$HOME/nethunter_cellbot"
mkdir -p "$CELLBOT_DIR/agent" "$CELLBOT_DIR/generated_code"
echo -e "${GREEN}Created directory structure at $CELLBOT_DIR${NC}"

# Copy files if script is run from the source directory
CURRENT_DIR=$(pwd)
if [[ -f "$CURRENT_DIR/nethunter_cellbot.py" && -d "$CURRENT_DIR/agent" ]]; then
    echo -e "\n${BLUE}Copying files to installation directory...${NC}"
    cp -r "$CURRENT_DIR"/* "$CELLBOT_DIR/"
    echo -e "${GREEN}Files copied successfully${NC}"
else
    echo -e "${YELLOW}Warning: Source files not found in current directory.${NC}"
    echo -e "${YELLOW}Please copy the CellBot files to $CELLBOT_DIR manually.${NC}"
fi

# Install dependencies
echo -e "\n${BLUE}Installing dependencies...${NC}"
if [[ -f "$CELLBOT_DIR/requirements.txt" ]]; then
    pip3 install -r "$CELLBOT_DIR/requirements.txt"
    echo -e "${GREEN}Dependencies installed successfully${NC}"
else
    echo -e "${YELLOW}Warning: requirements.txt not found.${NC}"
    echo -e "${YELLOW}Installing basic dependencies...${NC}"
    pip3 install aiohttp aiofiles asyncio
fi

# Set permissions
echo -e "\n${BLUE}Setting file permissions...${NC}"
if [[ -f "$CELLBOT_DIR/nethunter_cellbot.py" ]]; then
    chmod +x "$CELLBOT_DIR/nethunter_cellbot.py"
    echo -e "${GREEN}Permissions set successfully${NC}"
fi

if [[ -f "$CELLBOT_DIR/test_nethunter_env.py" ]]; then
    chmod +x "$CELLBOT_DIR/test_nethunter_env.py"
fi

# Run environment test
echo -e "\n${BLUE}Running environment test...${NC}"
if [[ -f "$CELLBOT_DIR/test_nethunter_env.py" ]]; then
    cd "$CELLBOT_DIR"
    python3 test_nethunter_env.py
else
    echo -e "${YELLOW}Warning: test_nethunter_env.py not found. Skipping environment test.${NC}"
fi

# Final instructions
echo -e "\n${BLUE}=======================================${NC}"
echo -e "${GREEN}Installation completed!${NC}"
echo -e "${BLUE}=======================================${NC}"
echo -e "\nTo run CellBot:"
echo -e "  cd $CELLBOT_DIR"
echo -e "  python3 nethunter_cellbot.py"
echo -e "\nFor more information, see the README.md file."
echo -e "${BLUE}=======================================${NC}" 