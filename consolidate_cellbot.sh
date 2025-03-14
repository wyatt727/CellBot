#!/bin/bash
# CellBot for NetHunter - Consolidation script
# This script helps consolidate multiple CellBot installations

set -e  # Exit on error

# Text colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Usage function
usage() {
    echo -e "${BLUE}CellBot for NetHunter - Consolidation Script${NC}"
    echo
    echo "This script helps manage multiple CellBot installations"
    echo
    echo -e "Usage: $0 ${YELLOW}[--install-to DIR] [--sync-only] [--help]${NC}"
    echo
    echo "Options:"
    echo "  --install-to DIR   Specify which location to install to"
    echo "                     e.g. /root or /home/kali"
    echo "  --sync-only        Only sync files between locations, don't install"
    echo "  --help             Show this help message"
    echo
}

# Check installations and their differences
check_installations() {
    echo -e "${BLUE}Checking CellBot installations...${NC}"
    
    # Check if installations exist
    ROOT_INSTALL_EXISTS=false
    KALI_INSTALL_EXISTS=false
    
    if [ -d "/root/nethunter_cellbot" ]; then
        ROOT_INSTALL_EXISTS=true
        echo -e "${GREEN}Found installation in /root/nethunter_cellbot${NC}"
    else
        echo -e "${YELLOW}No installation found in /root/nethunter_cellbot${NC}"
    fi
    
    if [ -d "/home/kali/nethunter_cellbot" ]; then
        KALI_INSTALL_EXISTS=true
        echo -e "${GREEN}Found installation in /home/kali/nethunter_cellbot${NC}"
    else
        echo -e "${YELLOW}No installation found in /home/kali/nethunter_cellbot${NC}"
    fi
    
    if [ "$ROOT_INSTALL_EXISTS" = false ] && [ "$KALI_INSTALL_EXISTS" = false ]; then
        echo -e "${RED}Error: No CellBot installations found.${NC}"
        exit 1
    fi
    
    if [ "$ROOT_INSTALL_EXISTS" = true ] && [ "$KALI_INSTALL_EXISTS" = true ]; then
        echo -e "\n${BLUE}Checking for differences between installations...${NC}"
        
        # Compare agent.py files if they exist
        if [ -f "/root/nethunter_cellbot/agent/agent.py" ] && [ -f "/home/kali/nethunter_cellbot/agent/agent.py" ]; then
            if diff -q "/root/nethunter_cellbot/agent/agent.py" "/home/kali/nethunter_cellbot/agent/agent.py" >/dev/null; then
                echo -e "${GREEN}agent.py files are identical${NC}"
            else
                echo -e "${YELLOW}agent.py files are different${NC}"
                diff -u "/root/nethunter_cellbot/agent/agent.py" "/home/kali/nethunter_cellbot/agent/agent.py" | head -n 20
                echo -e "${YELLOW}[Showing first 20 lines of differences...]${NC}"
            fi
        fi
        
        # Compare nethunter_main.py files if they exist
        if [ -f "/root/nethunter_cellbot/agent/nethunter_main.py" ] && [ -f "/home/kali/nethunter_cellbot/agent/nethunter_main.py" ]; then
            if diff -q "/root/nethunter_cellbot/agent/nethunter_main.py" "/home/kali/nethunter_cellbot/agent/nethunter_main.py" >/dev/null; then
                echo -e "${GREEN}nethunter_main.py files are identical${NC}"
            else
                echo -e "${YELLOW}nethunter_main.py files are different${NC}"
                diff -u "/root/nethunter_cellbot/agent/nethunter_main.py" "/home/kali/nethunter_cellbot/agent/nethunter_main.py" | head -n 20
                echo -e "${YELLOW}[Showing first 20 lines of differences...]${NC}"
            fi
        fi
        
        # Check if database files exist in different locations
        echo -e "\n${BLUE}Checking database files...${NC}"
        ROOT_DB=$(find /root -name "conversation.db" 2>/dev/null | grep -v "/.cache/" || true)
        KALI_DB=$(find /home/kali -name "conversation.db" 2>/dev/null | grep -v "/.cache/" || true)
        
        if [ -n "$ROOT_DB" ]; then
            echo -e "${GREEN}Found database in root: $ROOT_DB${NC}"
        else
            echo -e "${YELLOW}No database found in /root${NC}"
        fi
        
        if [ -n "$KALI_DB" ]; then
            echo -e "${GREEN}Found database in /home/kali: $KALI_DB${NC}"
        else
            echo -e "${YELLOW}No database found in /home/kali${NC}"
        fi
    fi
}

# Install to specific location
install_to_location() {
    TARGET_DIR="$1"
    echo -e "${BLUE}Installing CellBot to $TARGET_DIR/nethunter_cellbot${NC}"
    
    # Create target directory if it doesn't exist
    mkdir -p "$TARGET_DIR/nethunter_cellbot"
    
    # Determine which source to use
    if [ -d "/root/nethunter_cellbot" ] && [ -d "/home/kali/nethunter_cellbot" ]; then
        # Both exist, ask user which to use as source
        echo -e "${YELLOW}Found installations in both /root and /home/kali.${NC}"
        echo -e "Which one would you like to use as the source?"
        echo "1) /root/nethunter_cellbot"
        echo "2) /home/kali/nethunter_cellbot"
        read -p "Enter your choice (1/2): " choice
        
        if [ "$choice" = "1" ]; then
            SOURCE_DIR="/root/nethunter_cellbot"
        elif [ "$choice" = "2" ]; then
            SOURCE_DIR="/home/kali/nethunter_cellbot"
        else
            echo -e "${RED}Invalid choice. Exiting.${NC}"
            exit 1
        fi
    elif [ -d "/root/nethunter_cellbot" ]; then
        SOURCE_DIR="/root/nethunter_cellbot"
    elif [ -d "/home/kali/nethunter_cellbot" ]; then
        SOURCE_DIR="/home/kali/nethunter_cellbot"
    else
        echo -e "${RED}Error: No CellBot installations found.${NC}"
        exit 1
    fi
    
    # Copy files
    echo -e "${BLUE}Copying files from $SOURCE_DIR to $TARGET_DIR/nethunter_cellbot...${NC}"
    rsync -avz --delete "$SOURCE_DIR/" "$TARGET_DIR/nethunter_cellbot/"
    
    # Set permissions
    echo -e "${BLUE}Setting permissions...${NC}"
    chmod +x "$TARGET_DIR/nethunter_cellbot/run_cellbot.sh" 2>/dev/null || true
    
    # Create a symlink to make running easier
    if [ ! -f "/usr/local/bin/cellbot" ]; then
        echo -e "${BLUE}Creating symlink in /usr/local/bin...${NC}"
        cat > /usr/local/bin/cellbot << EOF
#!/bin/bash
cd $TARGET_DIR/nethunter_cellbot
./run_cellbot.sh "\$@"
EOF
        chmod +x /usr/local/bin/cellbot
        echo -e "${GREEN}Created symlink: /usr/local/bin/cellbot${NC}"
    fi
    
    # Create configuration file
    echo -e "${BLUE}Creating configuration file...${NC}"
    
    # Check for existing database and use it if found
    DB_PATH=""
    if [ -n "$ROOT_DB" ]; then
        DB_PATH="$ROOT_DB"
    elif [ -n "$KALI_DB" ]; then
        DB_PATH="$KALI_DB"
    fi
    
    # If no DB found, create a default path
    if [ -z "$DB_PATH" ]; then
        DB_PATH="$TARGET_DIR/nethunter_cellbot/data/conversation.db"
    fi
    
    # Create cellbot.conf
    mkdir -p "$TARGET_DIR/nethunter_cellbot/data"
    cat > "$TARGET_DIR/nethunter_cellbot/cellbot.conf" << EOF
# CellBot for NetHunter configuration
# This file is sourced by the run_cellbot.sh script

# API settings
API_URL="http://localhost:11434/api"
MODEL_NAME="mistral:7b"
NUM_THREADS=2
TIMEOUT=180
DEBUG=1

# Database settings
CELLBOT_DB_PATH="$DB_PATH"
EOF
    
    # Update run_cellbot.sh to use the config file
    cat > "$TARGET_DIR/nethunter_cellbot/run_cellbot.sh" << 'EOF'
#!/bin/bash
# CellBot for NetHunter launcher script

CELLBOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$CELLBOT_DIR"

# Load configuration
if [ -f "$CELLBOT_DIR/cellbot.conf" ]; then
    source "$CELLBOT_DIR/cellbot.conf"
fi

# Set environment variables (allow override from command line)
export API_URL=${API_URL:-"http://localhost:11434/api"}
export MODEL_NAME=${MODEL_NAME:-"mistral:7b"}
export NUM_THREADS=${NUM_THREADS:-2}
export TIMEOUT=${TIMEOUT:-180}
export DEBUG=${DEBUG:-1}

# Allow explicit override of database path
if [ ! -z "$CELLBOT_DB_PATH" ]; then
    echo "Using database path: $CELLBOT_DB_PATH"
    # Ensure directory exists
    DB_DIR=$(dirname "$CELLBOT_DB_PATH")
    mkdir -p "$DB_DIR" 2>/dev/null || true
    
    # Export for Python to use
    export CELLBOT_DB_PATH
fi

echo "Starting CellBot for NetHunter with model $MODEL_NAME..."
echo "API URL: $API_URL"

# Run the main script
python3 agent/nethunter_main.py --threads "$NUM_THREADS" --timeout "$TIMEOUT" --debug "$DEBUG" "$@"
EOF
    
    chmod +x "$TARGET_DIR/nethunter_cellbot/run_cellbot.sh"
    
    echo -e "${GREEN}Installation completed successfully!${NC}"
    echo -e "You can now run CellBot using: ${YELLOW}cellbot${NC} or ${YELLOW}$TARGET_DIR/nethunter_cellbot/run_cellbot.sh${NC}"
}

# Synchronize installations
sync_installations() {
    echo -e "${BLUE}Synchronizing CellBot installations...${NC}"
    
    if [ -d "/root/nethunter_cellbot" ] && [ -d "/home/kali/nethunter_cellbot" ]; then
        echo -e "${YELLOW}Found installations in both /root and /home/kali.${NC}"
        echo -e "Which one would you like to use as the source?"
        echo "1) /root/nethunter_cellbot"
        echo "2) /home/kali/nethunter_cellbot"
        read -p "Enter your choice (1/2): " choice
        
        if [ "$choice" = "1" ]; then
            SOURCE_DIR="/root/nethunter_cellbot"
            TARGET_DIR="/home/kali/nethunter_cellbot"
        elif [ "$choice" = "2" ]; then
            SOURCE_DIR="/home/kali/nethunter_cellbot"
            TARGET_DIR="/root/nethunter_cellbot"
        else
            echo -e "${RED}Invalid choice. Exiting.${NC}"
            exit 1
        fi
        
        echo -e "${BLUE}Synchronizing from $SOURCE_DIR to $TARGET_DIR...${NC}"
        rsync -avz --delete "$SOURCE_DIR/" "$TARGET_DIR/"
        echo -e "${GREEN}Synchronization completed successfully!${NC}"
    else
        echo -e "${RED}Error: Both installations must exist for synchronization.${NC}"
        exit 1
    fi
}

# Main function
main() {
    # Parse arguments
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --install-to)
                INSTALL_TO="$2"
                shift 2
                ;;
            --sync-only)
                SYNC_ONLY=true
                shift
                ;;
            --help)
                usage
                exit 0
                ;;
            *)
                echo -e "${RED}Unknown option: $1${NC}"
                usage
                exit 1
                ;;
        esac
    done
    
    # Check installations
    check_installations
    
    # Process based on arguments
    if [ "$SYNC_ONLY" = true ]; then
        sync_installations
    elif [ -n "$INSTALL_TO" ]; then
        install_to_location "$INSTALL_TO"
    else
        echo -e "\n${BLUE}What would you like to do?${NC}"
        echo "1) Install/update to /root"
        echo "2) Install/update to /home/kali"
        echo "3) Synchronize both installations"
        echo "4) Exit"
        read -p "Enter your choice (1-4): " choice
        
        case "$choice" in
            1)
                install_to_location "/root"
                ;;
            2)
                install_to_location "/home/kali"
                ;;
            3)
                sync_installations
                ;;
            4)
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid choice. Exiting.${NC}"
                exit 1
                ;;
        esac
    fi
}

# Run main
main "$@" 