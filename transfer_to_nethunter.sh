#!/bin/bash
# CellBot Transfer to NetHunter Script

# ANSI color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}  CellBot Transfer to NetHunter Tool   ${NC}"
echo -e "${BLUE}=======================================${NC}"

# Check if running on Linux/macOS
if [[ "$(uname -s)" != "Linux" && "$(uname -s)" != "Darwin" ]]; then
    echo -e "${RED}Error: This script should be run on a Linux or macOS system.${NC}"
    exit 1
fi

# Check for required tools
echo -e "\n${BLUE}Checking for required tools...${NC}"
MISSING_TOOLS=0

if ! command -v ssh &>/dev/null; then
    echo -e "${RED}Error: ssh not found. Please install OpenSSH.${NC}"
    MISSING_TOOLS=1
fi

if ! command -v scp &>/dev/null; then
    echo -e "${RED}Error: scp not found. Please install OpenSSH.${NC}"
    MISSING_TOOLS=1
fi

if ! command -v rsync &>/dev/null; then
    echo -e "${YELLOW}Warning: rsync not found. Will fall back to scp.${NC}"
fi

if [[ $MISSING_TOOLS -eq 1 ]]; then
    exit 1
fi

# Get NetHunter device IP
echo -e "\n${BLUE}Enter your NetHunter device's IP address:${NC}"
read -p "> " NETHUNTER_IP

if [[ -z "$NETHUNTER_IP" ]]; then
    echo -e "${RED}Error: IP address cannot be empty.${NC}"
    exit 1
fi

# Get SSH port (default: 22)
echo -e "\n${BLUE}Enter SSH port (default: 22):${NC}"
read -p "> " SSH_PORT
SSH_PORT=${SSH_PORT:-22}

# Get username (default: kali)
echo -e "\n${BLUE}Enter username (default: kali):${NC}"
read -p "> " USERNAME
USERNAME=${USERNAME:-kali}

# Check SSH connection
echo -e "\n${BLUE}Testing SSH connection...${NC}"
if ! ssh -p $SSH_PORT -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=accept-new $USERNAME@$NETHUNTER_IP "echo 2>&1" >/dev/null; then
    echo -e "${YELLOW}SSH connection failed. You may need to enter password manually.${NC}"
    
    # Ask if user wants to continue
    echo -e "\n${BLUE}Continue with transfer? (y/n)${NC}"
    read -p "> " CONTINUE
    if [[ "$CONTINUE" != "y" && "$CONTINUE" != "Y" ]]; then
        echo -e "${RED}Transfer aborted.${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}SSH connection successful!${NC}"
fi

# Create remote directory
echo -e "\n${BLUE}Creating directory on NetHunter device...${NC}"
ssh -p $SSH_PORT $USERNAME@$NETHUNTER_IP "mkdir -p ~/nethunter_cellbot/agent ~/nethunter_cellbot/generated_code"

# Transfer files
echo -e "\n${BLUE}Transferring files to NetHunter device...${NC}"
if command -v rsync &>/dev/null; then
    # Use rsync for transfer (more efficient)
    rsync -avz --progress -e "ssh -p $SSH_PORT" \
        --exclude '.git/' \
        --exclude '.vscode/' \
        --exclude '__pycache__/' \
        --exclude '*.pyc' \
        --exclude '.DS_Store' \
        . $USERNAME@$NETHUNTER_IP:~/nethunter_cellbot/
else
    # Fall back to scp
    scp -P $SSH_PORT -r \
        !(*.git|*.vscode|*__pycache__|*.pyc|*.DS_Store) \
        $USERNAME@$NETHUNTER_IP:~/nethunter_cellbot/
fi

# Run installation script on remote device
echo -e "\n${BLUE}Running installation script on NetHunter device...${NC}"
ssh -p $SSH_PORT $USERNAME@$NETHUNTER_IP "cd ~/nethunter_cellbot && chmod +x install_nethunter.sh && ./install_nethunter.sh"

# Final instructions
echo -e "\n${BLUE}=======================================${NC}"
echo -e "${GREEN}Transfer completed!${NC}"
echo -e "${BLUE}=======================================${NC}"
echo -e "\nTo connect to your NetHunter device and run CellBot:"
echo -e "  ssh -p $SSH_PORT $USERNAME@$NETHUNTER_IP"
echo -e "  cd ~/nethunter_cellbot"
echo -e "  python3 nethunter_cellbot.py"
echo -e "\nFor more information, see the README.md file."
echo -e "${BLUE}=======================================${NC}" 