# agent/db.py
import sqlite3
import logging
import difflib
from typing import Tuple, Optional, List, Dict
import re
from collections import Counter
import os
import time

# Try to import from android_config first, fall back to regular config
try:
    from .android_config import DB_FILE, CONTEXT_MSG_COUNT, MAX_SIMILAR_EXAMPLES
    from .android_config import SIMILARITY_THRESHOLD
except ImportError:
    try:
from .config import (
    DB_FILE, CONTEXT_MSG_COUNT, MAX_SIMILAR_EXAMPLES,
    SIMILARITY_THRESHOLD
)
    except ImportError:
        # Fallback to safe defaults
        import os
        DB_FILE = os.path.join(os.path.expanduser("~"), "nethunter_cellbot", "conversation.db")
        CONTEXT_MSG_COUNT = 5
        MAX_SIMILAR_EXAMPLES = 1
        SIMILARITY_THRESHOLD = 0.94

import json
from datetime import datetime, timedelta
from functools import lru_cache

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set back to INFO level

# Constants for mobile operation
DB_PATH = os.path.expanduser("~/nethunter_cellbot")

# Check for environment variable override
ENV_DB_PATH = os.environ.get("CELLBOT_DB_PATH")
if ENV_DB_PATH:
    logger.info(f"Using database path from environment: {ENV_DB_PATH}")
    DB_FILE = ENV_DB_PATH
    # Set parent directory too
    DB_PATH = os.path.dirname(ENV_DB_PATH)
else:
    # Only redefine DB_FILE if not already set from config imports
    if 'DB_FILE' not in locals() or not DB_FILE:
        DB_FILE = os.path.join(DB_PATH, "conversation.db")

def preprocess_text(text: str) -> str:
    """Preprocess text for better matching by:
    1. Converting to lowercase
    2. Extracting and preserving special patterns
    3. Normalizing whitespace and handling variations
    """
    # Convert to lowercase and normalize whitespace
    text = ' '.join(text.lower().split())
    
    # Quick check for exact matches before heavy processing
    normalized = re.sub(r'[^a-z0-9\s]', ' ', text)
    normalized = ' '.join(normalized.split())
    
    # Store special patterns to preserve
    patterns = {
        'commands': re.findall(r'\b(git|ls|cd|cat|python|pip|npm|yarn|docker|kubectl|whoami|curl|wget)\s+[^\s]+', text),
        'paths': re.findall(r'(?:/[^\s/]+)+/?', text),
        'urls': re.findall(r'https?://[^\s]+', text)
    }
    
    # Handle common variations using a single pass
    variations = {
        'color': ['colour', 'colors', 'colours'],
        'circle': ['circles'],
        'change': ['changes', 'changing', 'modify', 'update'],
        'draw': ['drawing', 'create', 'make'],
        'username': ['user name', 'user-name'],
        'filename': ['file name', 'file-name'],
        'directory': ['dir', 'folder'],
        'delete': ['remove', 'del', 'rm']
    }
    
    # Build variations set efficiently
    words = set(normalized.split())
    variations_to_add = set()
    
    for word in words:
        # Check if word is a key in variations
        if word in variations:
            variations_to_add.update(variations[word])
        # Check if word is a value in any variation list
        for main_word, variants in variations.items():
            if word in variants:
                variations_to_add.add(main_word)
                variations_to_add.update(v for v in variants if v != word)
    
    # Add variations and preserved patterns
    result_parts = [normalized]
    if variations_to_add:
        result_parts.append(' '.join(variations_to_add))
    for pattern_type, matches in patterns.items():
        if matches:
            # Add important patterns with higher weight
            weight = 3 if pattern_type in ['commands', 'paths'] else 2
            result_parts.extend([' '.join(matches)] * weight)
    
    return ' '.join(result_parts)

def compute_similarity(query: str, stored: str) -> float:
    """Calculate a weighted similarity score using multiple metrics with performance tracking."""
    start_time = time.perf_counter()
    timings = {}

    def track_timing(name: str, start: float) -> float:
        duration = time.perf_counter() - start
        timings[name] = duration
        return duration

    # Check for exact match first (case-insensitive)
    if query.lower().strip() == stored.lower().strip():
        track_timing('total', start_time)
        return 1.0

    # Preprocess both texts
    preprocess_start = time.perf_counter()
    query_proc = preprocess_text(query)
    stored_proc = preprocess_text(stored)
    track_timing('preprocessing', preprocess_start)
    
    # 1. Sequence similarity using difflib (25%)
    seq_start = time.perf_counter()
    sequence_sim = difflib.SequenceMatcher(None, query_proc, stored_proc).ratio() * 0.25
    track_timing('sequence_similarity', seq_start)
    
    # 2. Token overlap with position awareness (25%)
    token_start = time.perf_counter()
    query_tokens = query_proc.split()
    stored_tokens = stored_proc.split()
    
    if not query_tokens or not stored_tokens:
        token_sim = 0.0
    else:
        # Calculate token overlap
        common_tokens = set(query_tokens) & set(stored_tokens)
        
        # Consider token positions for matched tokens
        position_scores = []
        for token in common_tokens:
            query_pos = [i for i, t in enumerate(query_tokens) if t == token]
            stored_pos = [i for i, t in enumerate(stored_tokens) if t == token]
            
            # Calculate position similarity for this token
            pos_sim = max(1 - abs(qp/len(query_tokens) - sp/len(stored_tokens)) 
                         for qp in query_pos for sp in stored_pos)
            position_scores.append(pos_sim)
        
        # Combine token overlap with position awareness
        if position_scores:
            token_sim = (len(common_tokens) / max(len(query_tokens), len(stored_tokens)) * 
                        sum(position_scores) / len(position_scores) * 0.25)
        else:
            token_sim = 0.0
    track_timing('token_overlap', token_start)
    
    # 3. Command pattern matching (20%)
    cmd_start = time.perf_counter()
    def extract_commands(text):
        # Extract full commands with arguments
        cmd_pattern = r'\b(git|ls|cd|cat|python|pip|npm|yarn|docker|kubectl|whoami|curl|wget)\s+[^\s]+'
        commands = re.findall(cmd_pattern, text.lower())
        # Also extract just the command names
        cmd_names = re.findall(r'\b(git|ls|cd|cat|python|pip|npm|yarn|docker|kubectl|whoami|curl|wget)\b', text.lower())
        return set(commands), set(cmd_names)
    
    query_cmds, query_cmd_names = extract_commands(query)
    stored_cmds, stored_cmd_names = extract_commands(stored)
    
    if not (query_cmds or query_cmd_names) and not (stored_cmds or stored_cmd_names):
        command_sim = 0.2  # Full score if no commands in either
    elif not (query_cmds or query_cmd_names) or not (stored_cmds or stored_cmd_names):
        command_sim = 0.0  # No match if commands in one but not the other
    else:
        # Weight exact command matches higher than just command name matches
        cmd_match = len(query_cmds & stored_cmds) / max(len(query_cmds | stored_cmds), 1) * 0.15
        name_match = len(query_cmd_names & stored_cmd_names) / max(len(query_cmd_names | stored_cmd_names), 1) * 0.05
        command_sim = cmd_match + name_match
    track_timing('command_matching', cmd_start)
    
    # 4. Semantic similarity using key concept matching (20%)
    sem_start = time.perf_counter()
    concepts = {
        'file_ops': r'\b(file|read|write|open|close|save|delete|remove|copy|move|rename)\b',
        'system': r'\b(system|os|process|service|daemon|run|execute|kill|stop|start)\b',
        'network': r'\b(network|connect|url|http|web|download|upload|server|client)\b',
        'user': r'\b(user|name|login|account|password|auth|sudo|permission)\b',
        'path': r'\b(path|directory|folder|dir|location|root|home|pwd)\b'
    }
    
    def get_concept_matches(text):
        matches = {}
        for concept, pattern in concepts.items():
            matches[concept] = bool(re.search(pattern, text.lower()))
        return matches
    
    query_concepts = get_concept_matches(query)
    stored_concepts = get_concept_matches(stored)
    
    concept_matches = sum(1 for c in concepts if query_concepts[c] == stored_concepts[c])
    semantic_sim = (concept_matches / len(concepts)) * 0.20
    track_timing('semantic_similarity', sem_start)
    
    # 5. Length ratio penalty (10%)
    len_start = time.perf_counter()
    len_ratio = min(len(query_proc), len(stored_proc)) / max(len(query_proc), len(stored_proc))
    length_score = len_ratio * 0.10
    track_timing('length_ratio', len_start)
    
    # Calculate final score
    total_score = sequence_sim + token_sim + command_sim + semantic_sim + length_score
    total_time = track_timing('total', start_time)
    
    # Only log detailed performance metrics at debug level
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"\nPerformance Analysis for similarity calculation:")
        logger.debug(f"{'Component':<20} | {'Time (ms)':<10} | {'% of Total':<10} | {'Score':<10}")
        logger.debug("-" * 55)
        
        for component, duration in timings.items():
            if component != 'total':
                percentage = (duration / total_time) * 100
                score = {
                    'preprocessing': 0,
                    'sequence_similarity': sequence_sim,
                    'token_overlap': token_sim,
                    'command_matching': command_sim,
                    'semantic_similarity': semantic_sim,
                    'length_ratio': length_score
                }.get(component, 0)
                logger.debug(f"{component:<20} | {duration*1000:>9.2f} | {percentage:>9.1f}% | {score:>9.3f}")
        
        logger.debug("-" * 55)
        logger.debug(f"{'Total':<20} | {total_time*1000:>9.2f} | {'100.0':>9}% | {total_score:>9.3f}")
    
    return total_score

# Default conversation examples to pre-populate the DB if it is empty.
DEFAULT_CONVERSATION = [
    {"role": "user", "content": "open google but make the background red"},
    {"role": "assistant", "content": (
        "```python\nwith open(__file__, \"r\") as f:\n    source = f.read()\n"
        "with open(\"red_google.py\", \"w\") as f:\n    f.write(source)\n"
        "import asyncio\nfrom playwright.async_api import async_playwright\n\n"
        "async def run():\n    async with async_playwright() as p:\n        browser = await p.chromium.launch(headless=False)\n"
        "        page = await browser.new_page()\n        await page.goto(\"https://google.com\")\n"
        "        await page.evaluate(\"document.body.style.backgroundColor = 'red'\")\n        await asyncio.sleep(300)\nasyncio.run(run())\n```")},
    {"role": "user", "content": "open my calculator then bring me to facebook"},
    {"role": "assistant", "content": "```sh\nopen -a Calculator && open https://facebook.com\n```"},
    {"role": "user", "content": "whoami"},
    {"role": "assistant", "content": "```sh\nwhoami\n```"}
]

class ConversationDB:
    """
    SQLite database for conversation history and example exchanges.
    
    Optimized for mobile environments with:
    - Connection pooling to reduce overhead
    - Parameterized queries for security
    - Minimal transaction sizes
    - Indexes on frequently queried columns
    """
    
    # Connection pool for reuse
    _connection_pool = {}
    
    def __init__(self, db_file: str = DB_FILE):
        """
        Initialize the ConversationDB.
        
        Args:
            db_file: Path to the SQLite database file
        """
        # Check for a saved database location from previous emergency fallbacks
        db_location_file = os.path.join(os.path.expanduser("~"), ".cellbot_db_location")
        if os.path.exists(db_location_file) and db_file == DB_FILE:
            try:
                with open(db_location_file, "r") as f:
                    saved_path = f.read().strip()
                if os.path.exists(saved_path):
                    logger.warning(f"Using previously saved database location: {saved_path}")
                    db_file = saved_path
            except Exception as e:
                logger.error(f"Error reading saved database location: {e}")
        
        # Try to find the first writable location on Android/NetHunter
        if db_file == DB_FILE:  # Only do this for the default location
            # List of possible locations in priority order
            possible_locations = [
                db_file,  # Original location first
                os.path.join(os.path.expanduser("~"), "cellbot", "conversation.db"),  # Primary location in current user's home dir
                os.path.join(os.path.expanduser("~"), ".cellbot", "conversation.db"),  # Hidden dir in user's home
                os.path.join("/data/local/nhsystem/kali-arm64", "root", "cellbot", "conversation.db"),  # NetHunter chroot location
                os.path.join("/data/local/nhsystem/kali-arm64", "root", ".cellbot", "conversation.db"),  # Hidden dir in NetHunter chroot
                "/data/data/com.offsec.nethunter/files/scripts/cellbot/conversation.db",  # NetHunter scripts directory
                "/data/data/com.offsec.nethunter/cellbot/conversation.db",  # App data directory
                os.path.join("/sdcard", "nethunter", "cellbot", "conversation.db"),  # NetHunter directory on sdcard
                os.path.join("/sdcard", "nethunter_cellbot", "conversation.db"),  # Legacy sdcard location
                "/storage/emulated/0/cellbot/conversation.db",  # Android standard storage
                os.path.join("/tmp", "cellbot.db")  # Last resort: use /tmp in the chroot
            ]
            
            for location in possible_locations:
                try:
                    # Ensure directory exists
                    dir_path = os.path.dirname(location)
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path, exist_ok=True)
                    
                    # Try to touch the file to confirm we can write
                    with open(location, 'a'):
                        pass
                    
                    logger.info(f"Found writable location at: {location}")
                    db_file = location
                    break
                except Exception as loc_err:
                    logger.debug(f"Location {location} not writable: {loc_err}")
        
        self.db_file = db_file
        logger.info(f"Initializing database at: {self.db_file}")
        
        # Ensure the parent directory exists
        try:
            parent_dir = os.path.dirname(self.db_file)
            if not os.path.exists(parent_dir):
                logger.info(f"Creating parent directory: {parent_dir}")
                os.makedirs(parent_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create parent directory: {e}")
            # We'll handle this in _get_connection
            
        # Initialize the connection pool
        if not hasattr(ConversationDB, '_connection_pool'):
            ConversationDB._connection_pool = {}
        
        # Try to establish a connection
        try:
            self.conn = self._get_connection()
            
            # Initialize the database structure
            self._setup_database()
        self._create_tables()
        self._initialize_defaults()
        self._prepare_statements()
        
        # Create settings table if it doesn't exist
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        self.conn.commit()
            logger.info("Database initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    def _get_connection(self):
        """Get a connection from the pool or create a new one."""
        thread_id = id(self)
        if thread_id not in self._connection_pool:
            logger.debug(f"Creating new database connection for thread {thread_id}")
            
            # Ensure the directory exists before attempting to create the database
            db_dir = os.path.dirname(self.db_file)
            try:
                if not os.path.exists(db_dir):
                    logger.info(f"Creating database directory: {db_dir}")
                    os.makedirs(db_dir, exist_ok=True)
                    try:
                        # Set directory permissions for mobile environments
                        os.chmod(db_dir, 0o755)  # rwxr-xr-x
                    except Exception as chmod_err:
                        logger.warning(f"Could not set directory permissions: {chmod_err}")
            except Exception as dir_err:
                logger.error(f"Failed to create database directory {db_dir}: {dir_err}")
                
                # Check if we're in NetHunter environment and might need special handling
                if "/data/local/nhsystem/kali-arm64" in self.db_file:
                    logger.warning("Detected NetHunter chroot environment, checking permissions...")
                    try:
                        # Check if chroot is writable
                        test_path = "/data/local/nhsystem/kali-arm64/root/test_write"
                        with open(test_path, 'w') as f:
                            f.write("test")
                        os.remove(test_path)
                        logger.info("NetHunter chroot is writable")
                        
                        # Try in NetHunter home directory
                        new_db_path = os.path.join(os.path.expanduser("~"), "cellbot.db")
                        logger.warning(f"Attempting to use direct home path: {new_db_path}")
                        self.db_file = new_db_path
                        with open(new_db_path, 'a'):
                            pass
                        return self._get_connection()
                    except Exception as perm_err:
                        logger.error(f"NetHunter permission issue: {perm_err}")
                
                # Try using a fallback location if the primary location fails
                try:
                    # Try /sdcard first (common Android storage location)
                    fallback_dir = "/sdcard/nethunter_cellbot"
                    if not os.path.exists(fallback_dir):
                        os.makedirs(fallback_dir, exist_ok=True)
                    fallback_db = os.path.join(fallback_dir, "conversation.db")
                    logger.warning(f"Attempting to use sdcard location: {fallback_db}")
                    self.db_file = fallback_db
                except Exception as sdcard_err:
                    logger.error(f"Failed to use sdcard location: {sdcard_err}")
                    
                    # Try the NetHunter data directory
                    try:
                        nethunter_dir = "/data/local/nhsystem/kali-arm64/root/.cellbot"
                        if not os.path.exists(nethunter_dir):
                            os.makedirs(nethunter_dir, exist_ok=True)
                        fallback_db = os.path.join(nethunter_dir, "conversation.db")
                        logger.warning(f"Attempting to use NetHunter data directory: {fallback_db}")
                        self.db_file = fallback_db
                    except Exception as nh_err:
                        logger.error(f"Failed to use NetHunter data directory: {nh_err}")
                        
                        # Try Android's app data directory
                        try:
                            app_data_dir = "/data/data/com.offsec.nethunter/files"
                            if not os.path.exists(app_data_dir):
                                os.makedirs(app_data_dir, exist_ok=True)
                            fallback_db = os.path.join(app_data_dir, "cellbot.db")
                            logger.warning(f"Attempting to use app data directory: {fallback_db}")
                            self.db_file = fallback_db
                        except Exception as app_err:
                            logger.error(f"Failed to use app data directory: {app_err}")
                    
                            # Try home directory
                            try:
                                fallback_dir = os.path.join(os.path.expanduser("~"), ".cellbot")
                                os.makedirs(fallback_dir, exist_ok=True)
                                fallback_db = os.path.join(fallback_dir, "conversation.db")
                                logger.warning(f"Attempting to use fallback location: {fallback_db}")
                                self.db_file = fallback_db
                            except Exception as fallback_err:
                                logger.error(f"Failed to create fallback directory: {fallback_err}")
                                
                                # Last resort - use a temporary file
                                try:
                                    import tempfile
                                    temp_dir = tempfile.gettempdir()
                                    self.db_file = os.path.join(temp_dir, "cellbot_temp.db")
                                    logger.warning(f"Using temporary database at: {self.db_file}")
                                except Exception as temp_err:
                                    logger.error(f"Failed to use temp directory: {temp_err}")
                                    
                                    # Absolute last resort - use in-memory database
                                    logger.warning("Using in-memory database as last resort")
                                    self.db_file = ":memory:"
            
            # Create an empty file if it doesn't exist to check permissions
            if self.db_file != ":memory:" and not os.path.exists(self.db_file):
                try:
                    with open(self.db_file, 'a'):
                        pass
                    try:
                        # Set appropriate file permissions
                        os.chmod(self.db_file, 0o644)  # rw-r--r--
                    except Exception as chmod_err:
                        logger.warning(f"Could not set file permissions: {chmod_err}")
                except Exception as touch_err:
                    logger.error(f"Cannot write to database file {self.db_file}: {touch_err}")
                    # Try memory database as absolute last resort
                    logger.warning("Falling back to in-memory database")
                    self.db_file = ":memory:"
            
            # Now try to connect to the database
            try:
                logger.info(f"Connecting to database at: {self.db_file}")
                
                # Use a more conservative approach for Android
                if self.db_file != ":memory:":
                    # Try directly creating the database file first
                    try:
                        with open(self.db_file, 'a+'):
                            pass
                    except Exception as create_err:
                        logger.warning(f"Could not create database file: {create_err}")
                
                # Enable URI connection string for better compatibility
                uri_path = self.db_file
                if self.db_file != ":memory:":
                    uri_path = f"file:{uri_path}?mode=rwc"
                
                # Connect with more conservative settings
                conn = sqlite3.connect(
                    uri_path,
                    uri=True if self.db_file != ":memory:" else False,
                    timeout=20.0,  # Even longer timeout for slow mobile storage
            isolation_level=None  # Autocommit mode
        )
                conn.row_factory = sqlite3.Row
                
                # Enable foreign keys
                conn.execute("PRAGMA foreign_keys = ON")
                
                # Detect if we're in NetHunter environment
                in_nethunter = "/data/local/nhsystem/kali-arm64" in self.db_file or any(
                    path in self.db_file for path in [
                        "/data/data/com.offsec.nethunter",
                        "/sdcard/nethunter",
                        "/tmp/cellbot.db"
                    ]
                )
                
                # Set more conservative SQLite settings for mobile/NetHunter
                if self.db_file != ":memory:":
                    # Use DELETE journaling mode, which is more compatible with Android/NetHunter
                    conn.execute("PRAGMA journal_mode = DELETE")
                    
                    if in_nethunter:
                        # Ultra-conservative settings for NetHunter
                        logger.info("Using NetHunter-optimized SQLite settings")
                        conn.execute("PRAGMA synchronous = OFF")  # Less durable but much faster in NetHunter
                        conn.execute("PRAGMA temp_store = MEMORY")
                        conn.execute("PRAGMA cache_size = 100")  # Very small cache for limited memory
                        conn.execute("PRAGMA mmap_size = 0")  # Disable mmap in NetHunter
                        conn.execute("PRAGMA page_size = 1024")  # Smaller pages for faster commits
                    else:
                        # Standard conservative settings for other environments
                        conn.execute("PRAGMA synchronous = NORMAL")
                        conn.execute("PRAGMA temp_store = MEMORY")
                        conn.execute("PRAGMA cache_size = 500")
                        conn.execute("PRAGMA mmap_size = 0")
                
                self._connection_pool[thread_id] = conn
                logger.info(f"Successfully connected to database at {self.db_file}")
                
            except sqlite3.Error as e:
                logger.error(f"SQLite error: {e}")
                
                # Handle specific SQLite errors
                if "disk I/O error" in str(e) or "unable to open database file" in str(e):
                    logger.error("Critical disk I/O error, trying to diagnose...")
                    
                    # Check if we're in NetHunter and the filesystem is read-only
                    try:
                        if "/data/local/nhsystem/kali-arm64" in self.db_file:
                            # Try to create a file in the same directory to test permissions
                            dir_path = os.path.dirname(self.db_file)
                            test_file = os.path.join(dir_path, ".test_write")
                            try:
                                with open(test_file, 'w') as f:
                                    f.write("test")
                                os.remove(test_file)
                                logger.info("Directory is writable, but SQLite still failed. Possible SQLite issue.")
                            except Exception:
                                logger.warning("Directory appears to be read-only in NetHunter chroot")
                                # Try to use /tmp which should be writable in most chroots
                                tmp_db = "/tmp/cellbot.db"
                                logger.warning(f"Trying /tmp location: {tmp_db}")
                                self.db_file = tmp_db
                                return self._get_connection()
                    except Exception as test_err:
                        logger.error(f"Error testing filesystem: {test_err}")
                    
                    try:
                        # Last resort: in-memory database
                        conn = sqlite3.connect(":memory:")
                        conn.row_factory = sqlite3.Row
                        self.db_file = ":memory:"
                        self._connection_pool[thread_id] = conn
                        logger.warning("Using in-memory database - data will be lost when app closes")
                        return conn
                    except Exception as memory_err:
                        logger.critical(f"Failed to create in-memory database: {memory_err}")
                        raise RuntimeError(f"Cannot create any database (even in-memory): {memory_err}")
                        
                elif "database disk image is malformed" in str(e):
                    logger.warning("Database appears corrupted, creating new file")
                    try:
                        # Backup the corrupted file if possible
                        if os.path.exists(self.db_file) and self.db_file != ":memory:":
                            backup_path = f"{self.db_file}.bak.{int(time.time())}"
                            try:
                                os.rename(self.db_file, backup_path)
                                logger.info(f"Backed up corrupted database to {backup_path}")
                            except Exception as rename_err:
                                logger.error(f"Failed to back up corrupted database: {rename_err}")
                        
                        # Create new database file
                        if self.db_file != ":memory:":
                            with open(self.db_file, 'w'):  # Truncate file
                                pass
                        
                        # Connect to new empty database
                        conn = sqlite3.connect(self.db_file)
                        conn.row_factory = sqlite3.Row
                        # Use more conservative settings
                        if self.db_file != ":memory:":
                            conn.execute("PRAGMA journal_mode = DELETE")
                            conn.execute("PRAGMA synchronous = NORMAL")
                        self._connection_pool[thread_id] = conn
                        
                        return conn
                    except Exception as recreate_err:
                        logger.error(f"Failed to recreate database: {recreate_err}")
                        
                        # Last resort: in-memory database
                        try:
                            conn = sqlite3.connect(":memory:")
                            conn.row_factory = sqlite3.Row
                            self.db_file = ":memory:"
                            self._connection_pool[thread_id] = conn
                            logger.warning("Using in-memory database - data will be lost when app closes")
                            return conn
                        except Exception as memory_err:
                            logger.critical(f"Failed to create in-memory database: {memory_err}")
                            raise RuntimeError(f"Cannot create any database: {memory_err}")
                else:
                    raise
                
        return self._connection_pool[thread_id]

    def _setup_database(self):
        """Setup the database with necessary tables and indexes."""
        try:
            cur = self.conn.cursor()
            # Create main conversation table.
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversation (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Create successful_exchanges table.
            cur.execute("""
                CREATE TABLE IF NOT EXISTS successful_exchanges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_prompt TEXT NOT NULL,
                    assistant_response TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Create model_comparisons table with enhanced timing metrics
            cur.execute("""
                CREATE TABLE IF NOT EXISTS model_comparisons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt TEXT NOT NULL,
                    model TEXT NOT NULL,
                    response TEXT,
                    total_time REAL,
                    generation_time REAL,
                    execution_time REAL,
                    tokens_per_second REAL,
                    code_results TEXT,
                    error TEXT,
                    token_count INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Create settings table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            # Create composite index on successful_exchanges.
            cur.execute("CREATE INDEX IF NOT EXISTS idx_user_prompt_timestamp ON successful_exchanges(user_prompt, timestamp)")
            # Create index on model_comparisons
            cur.execute("CREATE INDEX IF NOT EXISTS idx_model_comparisons ON model_comparisons(prompt, model, timestamp)")
            self.conn.commit()
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error setting up database: {e}")
            if "database is locked" in str(e):
                logger.warning("Database appears to be locked. This might be due to another process accessing it.")
            raise

    def _create_tables(self):
        cur = self.conn.cursor()
        # Create main conversation table.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversation (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Create successful_exchanges table.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS successful_exchanges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_prompt TEXT NOT NULL,
                assistant_response TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Create model_comparisons table with enhanced timing metrics
        cur.execute("""
            CREATE TABLE IF NOT EXISTS model_comparisons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                model TEXT NOT NULL,
                response TEXT,
                total_time REAL,
                generation_time REAL,
                execution_time REAL,
                tokens_per_second REAL,
                code_results TEXT,
                error TEXT,
                token_count INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Create composite index on successful_exchanges.
        cur.execute("CREATE INDEX IF NOT EXISTS idx_user_prompt_timestamp ON successful_exchanges(user_prompt, timestamp)")
        # Create index on model_comparisons
        cur.execute("CREATE INDEX IF NOT EXISTS idx_model_comparisons ON model_comparisons(prompt, model, timestamp)")
        self.conn.commit()

    def _initialize_defaults(self):
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM conversation")
        count = cur.fetchone()[0]
        if count == 0:
            logger.info("Conversation DB empty. Inserting default examples.")
            messages = [(msg["role"], msg["content"]) for msg in DEFAULT_CONVERSATION]
            cur.executemany("INSERT INTO conversation (role, content) VALUES (?, ?)", messages)
            self.conn.commit()

    def _prepare_statements(self):
        """Pre-compile commonly used SQL statements using cursor objects."""
        # Create cursors for prepared statements
        self.prepared_statements = {
            'add_message': {
                'cursor': self.conn.cursor(),
                'sql': "INSERT INTO conversation (role, content) VALUES (?, ?)"
            },
            'get_recent': {
                'cursor': self.conn.cursor(),
                'sql': "SELECT role, content FROM conversation WHERE role IN ('user','assistant') ORDER BY timestamp DESC LIMIT ?"
            },
            'add_exchange': {
                'cursor': self.conn.cursor(),
                'sql': "INSERT INTO successful_exchanges (user_prompt, assistant_response) VALUES (?, ?)"
            }
        }

    def add_message(self, role: str, content: str):
        """Optimized message addition using prepared statement."""
        try:
            stmt = self.prepared_statements['add_message']
            stmt['cursor'].execute(stmt['sql'], (role, content))
        except sqlite3.Error as e:
            logger.error(f"Database error adding message: {e}")
            # Fallback to new cursor if the prepared one fails
            cursor = self.conn.cursor()
            cursor.execute("INSERT INTO conversation (role, content) VALUES (?, ?)", (role, content))

    def get_recent_messages(self, limit: int = CONTEXT_MSG_COUNT) -> List[Dict[str, str]]:
        """Optimized recent message retrieval using prepared statement."""
        try:
            stmt = self.prepared_statements['get_recent']
            stmt['cursor'].execute(stmt['sql'], (limit,))
            rows = stmt['cursor'].fetchall()
            return [{"role": role, "content": content} for role, content in reversed(rows)]
        except sqlite3.Error as e:
            logger.error(f"Database error getting recent messages: {e}")
            return []

    def add_successful_exchange(self, user_prompt: str, assistant_response: str) -> bool:
        """Optimized successful exchange addition with duplicate check."""
        try:
            # Check for duplicates using indexed columns
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT EXISTS(SELECT 1 FROM successful_exchanges WHERE user_prompt = ? AND assistant_response = ?)",
                (user_prompt, assistant_response)
            )
            count = cursor.fetchone()[0]
            
            if not count:
                stmt = self.prepared_statements['add_exchange']
                stmt['cursor'].execute(stmt['sql'], (user_prompt, assistant_response))
                # Clear the cache when new data is added
                self.find_successful_exchange.cache_clear()
                return True
            return False
        except sqlite3.Error as e:
            logger.error(f"Database error adding successful exchange: {e}")
            return False

    @lru_cache(maxsize=1000)
    def find_successful_exchange(self, user_prompt: str, threshold: float = SIMILARITY_THRESHOLD) -> List[Tuple[str, str, float]]:
        """
        Find similar successful exchanges in the database.
        Returns a list of tuples (prompt, response, similarity_score) sorted by similarity.
        The results are not filtered by threshold - that's handled by the caller.
        """
        start_time = time.perf_counter()
        timings = {}
        
        def track_timing(name: str, start: float) -> float:
            duration = time.perf_counter() - start
            timings[name] = duration
            return duration
        
        # Database query - search ALL exchanges
        query_start = time.perf_counter()
        cur = self.conn.execute(
            "SELECT user_prompt, assistant_response FROM successful_exchanges"
        )
        rows = cur.fetchall()
        track_timing('database_query', query_start)
        
        if not rows:
            logger.debug(f"No exchanges found in database for prompt: {user_prompt}")
            return []
            
        # Calculate similarity scores
        scoring_start = time.perf_counter()
        matches = [(stored_prompt, stored_response, compute_similarity(user_prompt, stored_prompt))
                  for stored_prompt, stored_response in rows]
        track_timing('similarity_scoring', scoring_start)
        
        # Sort matches by similarity
        sort_start = time.perf_counter()
        sorted_matches = sorted(matches, key=lambda x: x[2], reverse=True)
        
        # Get top matches up to MAX_SIMILAR_EXAMPLES
        filtered_matches = sorted_matches[:MAX_SIMILAR_EXAMPLES] if MAX_SIMILAR_EXAMPLES > 0 else []
        track_timing('sort_and_filter', sort_start)
        
        # Log performance metrics at debug level
        if logger.isEnabledFor(logging.DEBUG):
            total_time = track_timing('total', start_time)
            if filtered_matches:
                logger.debug(f"Found {len(filtered_matches)} similar examples in {total_time*1000:.2f}ms")
                logger.debug(f"Best match similarity: {filtered_matches[0][2]:.2%}")
            else:
                logger.debug(f"No similar examples found (checked {len(rows)} exchanges in {total_time*1000:.2f}ms)")
            
        return filtered_matches

    def list_successful_exchanges(self, search_term: str = "", offset: int = 0, limit: int = 100) -> List[Dict]:
        """
        List successful exchanges with pagination and optimized search.
        
        Args:
            search_term: Optional search term to filter results
            offset: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of exchange dictionaries
        """
        cur = self.conn.cursor()
        try:
            if search_term:
                # Use LIKE query with index optimization
                search_pattern = f"%{search_term}%"
                cur.execute("""
                    SELECT id, user_prompt, assistant_response, timestamp
                    FROM successful_exchanges
                    WHERE user_prompt LIKE ? OR assistant_response LIKE ?
                    ORDER BY timestamp DESC
                    LIMIT ? OFFSET ?
                """, (search_pattern, search_pattern, limit, offset))
            else:
                cur.execute("""
                    SELECT id, user_prompt, assistant_response, timestamp
                    FROM successful_exchanges
                    ORDER BY timestamp DESC
                    LIMIT ? OFFSET ?
                """, (limit, offset))
            
            rows = cur.fetchall()
            return [{"id": r[0], "user_prompt": r[1], "assistant_response": r[2], "timestamp": r[3]} for r in rows]
            
        except sqlite3.Error as e:
            logger.error(f"Database error in list_successful_exchanges: {e}")
            return []

    def get_total_exchanges_count(self, search_term: str = "") -> int:
        """
        Get total count of exchanges, optionally filtered by search term.
        
        Args:
            search_term: Optional search term to filter results
            
        Returns:
            Total number of matching exchanges
        """
        cur = self.conn.cursor()
        try:
            if search_term:
                search_pattern = f"%{search_term}%"
                cur.execute("""
                    SELECT COUNT(*)
                    FROM successful_exchanges
                    WHERE user_prompt LIKE ? OR assistant_response LIKE ?
                """, (search_pattern, search_pattern))
            else:
                cur.execute("SELECT COUNT(*) FROM successful_exchanges")
            
            return cur.fetchone()[0]
            
        except sqlite3.Error as e:
            logger.error(f"Database error in get_total_exchanges_count: {e}")
            return 0

    def batch_update_exchanges(self, exchange_ids: List[int], new_response: str) -> bool:
        """
        Update multiple exchanges with the same response.
        
        Args:
            exchange_ids: List of exchange IDs to update
            new_response: New response text to set
            
        Returns:
            bool: True if all updates were successful
        """
        cur = self.conn.cursor()
        try:
            # Start a transaction
            cur.execute("BEGIN TRANSACTION")
            
            # Update all exchanges
            cur.executemany(
                """
                UPDATE successful_exchanges 
                SET assistant_response = ?, timestamp = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                [(new_response, exchange_id) for exchange_id in exchange_ids]
            )
            
            # Commit the transaction
            cur.execute("COMMIT")
            
            # Clear the cache
            self.find_successful_exchange.cache_clear()
            
            logger.info(f"Batch updated {len(exchange_ids)} exchanges")
            return True
            
        except sqlite3.Error as e:
            # Rollback on error
            cur.execute("ROLLBACK")
            logger.error(f"Database error in batch_update_exchanges: {e}")
            return False

    def remove_successful_exchange(self, exchange_id: int) -> bool:
        """
        Remove a single exchange by ID.
        
        Args:
            exchange_id: ID of the exchange to remove
            
        Returns:
            bool: True if deletion was successful
        """
        cur = self.conn.cursor()
        try:
            # Start a transaction
            cur.execute("BEGIN TRANSACTION")
            
            # Verify the exchange exists
            cur.execute("SELECT EXISTS(SELECT 1 FROM successful_exchanges WHERE id = ?)", (exchange_id,))
            if not cur.fetchone()[0]:
                logger.error(f"Exchange ID {exchange_id} not found")
                return False
            
            # Delete the exchange
            cur.execute("DELETE FROM successful_exchanges WHERE id = ?", (exchange_id,))
            
            # Commit the transaction
            cur.execute("COMMIT")
            
            # Clear the cache
            self.find_successful_exchange.cache_clear()
            
            logger.info(f"Successfully deleted exchange {exchange_id}")
            return True
            
        except sqlite3.Error as e:
            # Rollback on error
            cur.execute("ROLLBACK")
            logger.error(f"Database error removing exchange {exchange_id}: {e}")
            return False

    def batch_delete_exchanges(self, exchange_ids: List[int]) -> Tuple[bool, List[int]]:
        """
        Delete multiple exchanges at once.
        
        Args:
            exchange_ids: List of exchange IDs to delete
            
        Returns:
            Tuple[bool, List[int]]: (overall success, list of successfully deleted IDs)
        """
        cur = self.conn.cursor()
        successful_deletes = []
        
        try:
            # Start a transaction
            cur.execute("BEGIN TRANSACTION")
            
            # Delete exchanges one by one to track success
            for exchange_id in exchange_ids:
                try:
                    cur.execute("DELETE FROM successful_exchanges WHERE id = ?", (exchange_id,))
                    if cur.rowcount > 0:
                        successful_deletes.append(exchange_id)
                except sqlite3.Error as e:
                    logger.error(f"Error deleting exchange {exchange_id}: {e}")
            
            # Commit the transaction if any deletions were successful
            if successful_deletes:
                cur.execute("COMMIT")
                # Clear the cache
                self.find_successful_exchange.cache_clear()
                logger.info(f"Successfully deleted {len(successful_deletes)} exchanges")
            else:
                cur.execute("ROLLBACK")
                logger.warning("No exchanges were deleted")
            
            return bool(successful_deletes), successful_deletes
            
        except sqlite3.Error as e:
            # Rollback on error
            cur.execute("ROLLBACK")
            logger.error(f"Database error in batch_delete_exchanges: {e}")
            return False, []

    def add_comparison_result(self, prompt: str, result: dict):
        """
        Adds a model comparison result to the database with enhanced timing metrics.
        
        Args:
            prompt: The prompt that was tested
            result: Dictionary containing the model's response and metrics
        """
        try:
            cur = self.conn.cursor()
            
            # Convert code_results to JSON string
            code_results_json = json.dumps(result.get('code_results', []))
            timing = result.get('timing', {})
            
            cur.execute("""
                INSERT INTO model_comparisons 
                (prompt, model, response, total_time, generation_time, execution_time, 
                tokens_per_second, code_results, error, token_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prompt,
                result['model'],
                result.get('response', ''),
                timing.get('total_time'),
                timing.get('generation_time'),
                timing.get('execution_time'),
                timing.get('generation_tokens_per_second'),
                code_results_json,
                result.get('error'),
                result.get('token_count')
            ))
            
            self.conn.commit()
            logger.info(f"Stored comparison result for model {result['model']}")
            
        except Exception as e:
            logger.error(f"Failed to store comparison result: {e}")
            raise

    def get_comparison_results(self, prompt: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """
        Retrieves model comparison results from the database.
        
        Args:
            prompt: Optional prompt to filter by
            limit: Maximum number of results to return
            
        Returns:
            List of comparison result dictionaries
        """
        try:
            cur = self.conn.cursor()
            
            if prompt:
                cur.execute("""
                    SELECT prompt, model, response, total_time, generation_time, execution_time,
                           tokens_per_second, code_results, error, token_count, timestamp
                    FROM model_comparisons
                    WHERE prompt = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (prompt, limit))
            else:
                cur.execute("""
                    SELECT prompt, model, response, total_time, generation_time, execution_time,
                           tokens_per_second, code_results, error, token_count, timestamp
                    FROM model_comparisons
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,))
                
            rows = cur.fetchall()
            results = []
            
            for row in rows:
                try:
                    code_results = json.loads(row[7]) if row[7] else []
                except json.JSONDecodeError:
                    code_results = []
                    logger.warning(f"Failed to decode code_results JSON for comparison ID {row[0]}")
                
                results.append({
                    "prompt": row[0],
                    "model": row[1],
                    "response": row[2],
                    "timing": {
                        "total_time": row[3],
                        "generation_time": row[4],
                        "execution_time": row[5],
                        "generation_tokens_per_second": row[6]
                    },
                    "code_results": code_results,
                    "error": row[8],
                    "token_count": row[9],
                    "timestamp": row[10]
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve comparison results: {e}")
            return []

    def cleanup_malformed_responses(self):
        """Clean up malformed responses in the database by removing any text outside of code blocks."""
        cur = self.conn.cursor()
        try:
            # Start a transaction
            cur.execute("BEGIN TRANSACTION")
            
            # Get all responses
            cur.execute("SELECT id, assistant_response FROM successful_exchanges")
            rows = cur.fetchall()
            
            # Pattern to match complete code blocks
            code_block_pattern = r"```[a-z]*\n[\s\S]*?```"
            
            updates = []
            for row_id, response in rows:
                # Find all code blocks
                code_blocks = re.findall(code_block_pattern, response)
                if code_blocks:
                    # Join multiple code blocks with newlines if there are any
                    cleaned_response = "\n".join(code_blocks)
                    if cleaned_response != response:
                        updates.append((cleaned_response, row_id))
            
            if updates:
                # Perform the updates
                cur.executemany(
                    "UPDATE successful_exchanges SET assistant_response = ? WHERE id = ?",
                    updates
                )
                logger.info(f"Cleaned up {len(updates)} malformed responses")
                
                # Clear the cache since we modified entries
                self.find_successful_exchange.cache_clear()
                
            # Commit the transaction
            cur.execute("COMMIT")
            return len(updates)
            
        except sqlite3.Error as e:
            # Rollback on error
            cur.execute("ROLLBACK")
            logger.error(f"Database error cleaning up responses: {e}")
            return 0

    def close(self):
        """
        Close all database connections in the pool.
        This method should be called when shutting down the application.
        """
        try:
            for thread_id, conn in list(self._connection_pool.items()):
                try:
                    conn.close()
                    del self._connection_pool[thread_id]
                except Exception as close_error:
                    logger.warning(f"Error closing connection for thread {thread_id}: {close_error}")
            logger.debug("All database connections closed")
                except Exception as e:
            logger.error(f"Error in close(): {e}")

    def find_similar_exchange(self, query: str, threshold: float = None) -> List[Tuple[str, str, float]]:
        """
        Find similar successful exchanges based on normalized query similarity.
        Optimized for mobile with:
        - Limit on the number of exchanges to search
        - Early termination if exact match found
        - Preprocessing to reduce computation
        
        Args:
            query: The user query to find similar exchanges for
            threshold: Optional similarity threshold (defaults to SIMILARITY_THRESHOLD from config)
            
        Returns:
            List of (user_prompt, assistant_response, similarity) tuples, sorted by similarity
        """
        if not threshold:
            threshold = SIMILARITY_THRESHOLD
            
        query = query.strip().lower()
        
        # Quick check for exact match first
        try:
            cursor = self.conn.execute(
                "SELECT user_prompt, assistant_response FROM successful_exchanges WHERE LOWER(user_prompt) = ? LIMIT 1",
                (query,)
            )
            exact_match = cursor.fetchone()
            if exact_match:
                # Return exact match with similarity 1.0
                return [(exact_match[0], exact_match[1], 1.0)]
        except Exception as e:
            logger.error(f"Error checking for exact match: {e}")
        
        # Limit the number of exchanges to compare to avoid performance issues on mobile
        try:
            cursor = self.conn.execute(
                "SELECT user_prompt, assistant_response FROM successful_exchanges ORDER BY timestamp DESC LIMIT 100"
            )
            successful_exchanges = cursor.fetchall()
        except Exception as e:
            logger.error(f"Error fetching successful exchanges: {e}")
            return []
            
        # If no successful exchanges, return empty list
        if not successful_exchanges:
            return []
            
        # Preprocess the query
        query_normalized = preprocess_text(query)
        
        # Compute similarities efficiently - only for the first 100 exchanges
        similarities = []
        for exchange in successful_exchanges:
            user_prompt, assistant_response = exchange
            prompt_normalized = preprocess_text(user_prompt.strip().lower())
            
            # Compute similarity
            similarity = compute_similarity(query_normalized, prompt_normalized)
            
            # Only keep if above minimum threshold - reduces memory usage
            min_consider = 0.5  # Only consider exchanges with at least 50% similarity
            if similarity >= min_consider:
                similarities.append((user_prompt, assistant_response, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[2], reverse=True)
        
        # Return top MAX_SIMILAR_EXAMPLES or fewer if not enough
        return similarities[:MAX_SIMILAR_EXAMPLES]

    def update_successful_exchange(self, exchange_id: int, new_response: str, new_prompt: str = None) -> bool:
        """
        Update a successful exchange with new response and optionally new prompt.
        
        Args:
            exchange_id: ID of the exchange to update
            new_response: New response text
            new_prompt: Optional new prompt text
            
        Returns:
            bool: True if update was successful
        """
        cur = self.conn.cursor()
        try:
            # Start a transaction
            cur.execute("BEGIN TRANSACTION")
            
            # Verify the exchange exists
            cur.execute("SELECT EXISTS(SELECT 1 FROM successful_exchanges WHERE id = ?)", (exchange_id,))
            if not cur.fetchone()[0]:
                logger.error(f"Exchange ID {exchange_id} not found")
                return False
            
            # Update the exchange
            if new_prompt is not None:
                cur.execute("""
                    UPDATE successful_exchanges 
                    SET assistant_response = ?, user_prompt = ?, timestamp = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (new_response, new_prompt, exchange_id))
            else:
                cur.execute("""
                    UPDATE successful_exchanges 
                    SET assistant_response = ?, timestamp = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (new_response, exchange_id))
            
            # Commit the transaction
            cur.execute("COMMIT")
            
            # Clear the cache since we modified an entry
            self.find_successful_exchange.cache_clear()
            
            logger.info(f"Successfully updated exchange {exchange_id}")
            return True
            
        except sqlite3.Error as e:
            # Rollback on error
            cur.execute("ROLLBACK")
            logger.error(f"Database error updating exchange {exchange_id}: {e}")
            return False

    def get_setting(self, key: str) -> Optional[str]:
        """Get a setting value by key."""
        cur = self.conn.execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = cur.fetchone()
        return row[0] if row else None

    def set_setting(self, key: str, value: str) -> None:
        """Set a setting value by key."""
        self.conn.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
            (key, value)
        )
        self.conn.commit()

    def get_conversation_history(self, limit: int = 100) -> List[Dict[str, str]]:
        """
        Get conversation history with a specified limit.
        
        Args:
            limit: Maximum number of messages to return (default 100)
            
        Returns:
            List of conversation messages with role and content
        """
        messages = []
        try:
            query = """
                SELECT role, content, timestamp 
                FROM conversation 
                WHERE role IN ('user', 'assistant') 
                ORDER BY timestamp DESC
                LIMIT ?
            """
            
            cursor = self.conn.execute(query, (limit,))
            rows = cursor.fetchall()
            
            # Convert to dictionary format and reverse to chronological order
            for role, content, timestamp in reversed(rows):
                messages.append({
                    "role": role,
                    "content": content,
                    "timestamp": timestamp
                })
                
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e}")
            
        return messages
        
    def search_conversation(self, search_term: str, limit: int = 20) -> List[Dict[str, str]]:
        """
        Search conversation history for a specific term.
        
        Args:
            search_term: Term to search for
            limit: Maximum number of results to return (default 20)
            
        Returns:
            List of matching conversation messages
        """
        if not search_term:
            return []
            
        results = []
        try:
            query = """
                SELECT role, content, timestamp 
                FROM conversation 
                WHERE role IN ('user', 'assistant') 
                AND content LIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
            """
            
            # Use SQL LIKE for basic pattern matching
            pattern = f"%{search_term}%"
            cursor = self.conn.execute(query, (pattern, limit))
            rows = cursor.fetchall()
            
            for role, content, timestamp in rows:
                results.append({
                    "role": role,
                    "content": content,
                    "timestamp": timestamp
                })
                
        except Exception as e:
            logger.error(f"Error searching conversation history: {e}")
            
        return results
        
    def clear_conversation_history(self) -> bool:
        """
        Clear all conversation history.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Keep any system messages but delete user and assistant messages
            query = """
                DELETE FROM conversation 
                WHERE role IN ('user', 'assistant')
            """
            self.conn.execute(query)
            return True
        except Exception as e:
            logger.error(f"Error clearing conversation history: {e}")
            return False
