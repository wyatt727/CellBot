#!/bin/bash
# Run a command in the NetHunter chroot

# Get the command to run
CMD="$@"
if [ -z "$CMD" ]; then
  echo "Please provide a command to run in the NetHunter chroot"
  exit 1
fi

# Execute the command in the NetHunter chroot
adb shell "su -c 'chroot /data/local/nhsystem/kali-arm64 /bin/su -l kali -c \"$CMD\"'" 