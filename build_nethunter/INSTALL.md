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
