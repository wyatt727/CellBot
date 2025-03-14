# CellBot for NetHunter - Multiple Installations Guide

## The Problem: Multiple CellBot Installations

If you're seeing CellBot installations in both `/root/nethunter_cellbot` and `/home/kali/nethunter_cellbot`, this can cause several issues:

1. **Confusion about which installation to use**: Each installation may have different versions of code.
2. **Database location issues**: Your conversation history might be stored in different databases.
3. **Path conflicts**: CellBot might try to create or access files in one location while you're running from another.
4. **Update problems**: Updates applied to one installation won't affect the other.

## The Solution: The Consolidation Script

I've created a script to help you manage multiple installations. This script can:

- Compare your existing installations to identify differences
- Consolidate your installations into a single preferred location
- Synchronize files between installations
- Set up a centralized configuration for consistent behavior
- Create a simple command-line shortcut for easier usage

## How to Use the Script

### Step 1: Download and make executable

```bash
# Copy the script
cd ~
wget https://raw.githubusercontent.com/yourusername/cellbot-nethunter/main/consolidate_cellbot.sh
# OR (if you already have it in your current directory)
# cp consolidate_cellbot.sh ~

# Make it executable
chmod +x consolidate_cellbot.sh
```

### Step 2: Run the script

```bash
./consolidate_cellbot.sh
```

The script will guide you through the process with interactive prompts:

1. It will first check your existing installations and show differences
2. Then it will offer options to:
   - Install/update to /root
   - Install/update to /home/kali
   - Synchronize both installations
   - Exit

### Step 3: Choose an installation location

The recommended approach is to choose one location (/root or /home/kali) as your primary installation. 

- `/root`: Better for system-wide installation, but requires root access
- `/home/kali`: Better for user-specific installation, accessible without root

### After Installation

After running the script, you'll have:

1. A clean, consolidated installation in your chosen location
2. A configuration file at `<chosen-location>/nethunter_cellbot/cellbot.conf`
3. A command-line shortcut: `cellbot` (so you don't need to remember the path)

## Database Location

The script tries to identify and reuse your existing database. If a database is found, the configuration will point to it. If not, a new default location will be set.

The database path is stored in the `cellbot.conf` file and can be modified if needed:

```
# Edit the configuration
nano <chosen-location>/nethunter_cellbot/cellbot.conf

# Update the database path
CELLBOT_DB_PATH="/your/preferred/path/conversation.db"
```

## Common Issues

### Database Permission Issues

If you encounter database errors even after consolidation:

```bash
# Set a writable location explicitly
export CELLBOT_DB_PATH="/tmp/cellbot.db"
cellbot
```

### Different Code Versions

If you want to ensure you're using the latest code:

```bash
./consolidate_cellbot.sh --sync-only
```

This will synchronize both installations, ensuring they have identical code.

## Going Forward

For future use, remember these tips:

1. Always use the `cellbot` command to launch the application
2. If you install updates, run the consolidation script again to ensure consistency
3. If you need to switch between locations, use the script with the `--install-to` option:
   ```bash
   ./consolidate_cellbot.sh --install-to /root
   # or
   ./consolidate_cellbot.sh --install-to /home/kali
   ```

## Help and Options

For more options, run:

```bash
./consolidate_cellbot.sh --help
``` 