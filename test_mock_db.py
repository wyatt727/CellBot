#!/usr/bin/env python3
# test_mock_db.py - Test the MockConversationDB implementation

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import directly from the file to avoid circular imports
from agent.mock_db import MockConversationDB

def test_mock_db():
    """Test the basic functionality of MockConversationDB."""
    print("Testing MockConversationDB...")
    
    # Initialize the mock database
    db = MockConversationDB()
    print("✓ MockConversationDB initialized successfully")
    
    # Test adding and retrieving messages
    db.add_message("user", "Hello, world!")
    db.add_message("assistant", "Hi there!")
    
    history = db.get_history()
    print(f"✓ Added and retrieved {len(history)} messages")
    
    # Test settings
    db.set_setting("test_key", "test_value")
    value = db.get_setting("test_key")
    print(f"✓ Set and retrieved setting: {value}")
    
    # Test message count
    count = db.get_message_count("user")
    print(f"✓ Message count for 'user': {count}")
    
    print("All tests passed!")

if __name__ == "__main__":
    test_mock_db() 