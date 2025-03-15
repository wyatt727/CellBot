# agent/mock_db.py
import logging
from typing import Tuple, Optional, List, Dict, Any
import time
import os

# Setup logging
logger = logging.getLogger(__name__)

class MockConversationDB:
    """
    In-memory mock database for conversation history that mimics the ConversationDB interface
    but doesn't use an actual SQLite database.
    """
    
    def __init__(self, db_file: str = None):
        """
        Initialize the MockConversationDB.
        
        Args:
            db_file: Ignored, only kept for API compatibility
        """
        logger.info("Using MockConversationDB (no database)")
        self.conversation = []
        self.successful_exchanges = []
        self.settings = {}
        self.model_comparisons = []
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.conversation.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
    
    def get_history(self, limit: int = 100) -> List[Dict[str, str]]:
        """Get conversation history."""
        return [{"role": msg["role"], "content": msg["content"]} 
                for msg in self.conversation[-limit:]]
    
    def get_recent_messages(self, limit: int = 5) -> List[Dict[str, str]]:
        """Get recent conversation messages."""
        messages = []
        for msg in self.conversation[-limit*2:]:
            if msg["role"] in ["user", "assistant"]:
                messages.append({"role": msg["role"], "content": msg["content"]})
        return messages[-limit:]
    
    def find_similar_exchange(self, user_prompt: str, threshold: float = 0.9) -> Tuple[Optional[Dict], float]:
        """Find a similar exchange in the successful exchanges."""
        # Always return None to indicate no match found
        return None, 0.0
    
    def add_successful_exchange(self, user_prompt: str, assistant_response: str):
        """Add a successful exchange."""
        # Just store in memory, don't write to database
        self.successful_exchanges.append({
            "user_prompt": user_prompt,
            "assistant_response": assistant_response,
            "timestamp": time.time()
        })
    
    def list_successful_exchanges(self, search_term: str = "", offset: int = 0, limit: int = 100) -> List[Dict]:
        """List successful exchanges."""
        if not search_term:
            return self.successful_exchanges[offset:offset+limit]
        
        # Simple substring search
        results = []
        for exchange in self.successful_exchanges:
            if (search_term.lower() in exchange["user_prompt"].lower() or 
                search_term.lower() in exchange["assistant_response"].lower()):
                results.append(exchange)
        
        return results[offset:offset+limit]
    
    def remove_successful_exchange(self, exchange_id: int) -> bool:
        """Remove a successful exchange."""
        # Since we don't have real IDs, just return success
        return True
    
    def add_comparison_result(self, prompt: str, model: str, response: str,
                             total_time: float, generation_time: float = 0,
                             execution_time: float = 0, tokens_per_second: float = 0,
                             code_results: str = "", error: str = "", token_count: int = 0):
        """Add a model comparison result."""
        self.model_comparisons.append({
            "prompt": prompt,
            "model": model,
            "response": response,
            "total_time": total_time,
            "generation_time": generation_time,
            "execution_time": execution_time,
            "tokens_per_second": tokens_per_second,
            "code_results": code_results,
            "error": error,
            "token_count": token_count,
            "timestamp": time.time()
        })
    
    def get_comparison_results(self, prompt: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get model comparison results."""
        if prompt:
            results = [r for r in self.model_comparisons if r["prompt"] == prompt]
            return results[:limit]
        return self.model_comparisons[:limit]
    
    def get_setting(self, key: str) -> Optional[str]:
        """Get a setting value."""
        return self.settings.get(key)
    
    def set_setting(self, key: str, value: str) -> None:
        """Set a setting value."""
        self.settings[key] = value
    
    def get_message_count(self, role: str) -> int:
        """Get count of messages by role."""
        return sum(1 for msg in self.conversation if msg["role"] == role)
    
    def get_conversation_history(self, limit: int = 100) -> List[Dict[str, str]]:
        """Get conversation history."""
        messages = []
        for msg in self.conversation[-limit:]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": msg.get("timestamp", time.time())
            })
        return messages
    
    def search_conversation(self, search_term: str, limit: int = 20) -> List[Dict[str, str]]:
        """Search conversation history."""
        if not search_term:
            return []
        
        results = []
        for msg in self.conversation:
            if search_term.lower() in msg["content"].lower():
                results.append({
                    "role": msg["role"],
                    "content": msg["content"],
                    "timestamp": msg.get("timestamp", time.time())
                })
                if len(results) >= limit:
                    break
        
        return results
    
    def clear_conversation_history(self) -> bool:
        """Clear conversation history."""
        self.conversation = []
        return True
    
    def close(self):
        """Close the database connection (no-op for mock)."""
        pass 