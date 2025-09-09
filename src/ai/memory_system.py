#!/usr/bin/env python3
"""
Ultra AI Memory System
Persistent memory storage for personal information and conversation context
"""

import json
import os
import logging
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
import threading
from datetime import datetime

class UltraAIMemory:
    """Memory system for Ultra AI to store and recall personal information."""
    
    def __init__(self):
        self.memory_dir = Path("/storage/emulated/0/AI_Models/.ultra_ai/memory")
        self.memory_file = self.memory_dir / "ultra_ai_memory.json"
        self.conversation_file = self.memory_dir / "conversations.json"
        self.lock = threading.Lock()
        
        # Ensure memory directory exists
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize memory structure
        self.memory = self._load_memory()
        self.conversations = self._load_conversations()
        
        logging.info("🧠 Ultra AI Memory System initialized")
    
    def _load_memory(self) -> Dict[str, Any]:
        """Load memory from file or create new structure."""
        try:
            if self.memory_file.exists():
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load memory: {e}")
        
        # Default memory structure
        return {
            "personal_info": {
                "user_name": None,
                "location": None,
                "preferences": {},
                "interests": [],
                "important_dates": {}
            },
            "facts_learned": {},
            "user_context": {
                "conversation_style": "friendly",
                "technical_level": "intermediate",
                "language": "english"
            },
            "relationships": {},
            "created_at": time.time(),
            "last_updated": time.time()
        }
    
    def _load_conversations(self) -> List[Dict]:
        """Load conversation history."""
        try:
            if self.conversation_file.exists():
                with open(self.conversation_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load conversations: {e}")
        
        return []
    
    def _save_memory(self):
        """Save memory to file."""
        try:
            self.memory["last_updated"] = time.time()
            with open(self.memory_file, 'w') as f:
                json.dump(self.memory, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save memory: {e}")
    
    def _save_conversations(self):
        """Save conversations to file."""
        try:
            # Keep only last 100 conversations to prevent file from getting too large
            if len(self.conversations) > 100:
                self.conversations = self.conversations[-100:]
                
            with open(self.conversation_file, 'w') as f:
                json.dump(self.conversations, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save conversations: {e}")
    
    def store_personal_info(self, key: str, value: Any, category: str = "personal_info") -> bool:
        """Store personal information about the user."""
        with self.lock:
            try:
                if category not in self.memory:
                    self.memory[category] = {}
                
                if category == "personal_info":
                    self.memory["personal_info"][key] = value
                else:
                    self.memory[category][key] = value
                
                self._save_memory()
                logging.info(f"💾 Stored {category}.{key}: {value}")
                return True
            except Exception as e:
                logging.error(f"Failed to store personal info: {e}")
                return False
    
    def get_personal_info(self, key: str, category: str = "personal_info") -> Optional[Any]:
        """Retrieve personal information."""
        with self.lock:
            try:
                if category in self.memory and key in self.memory[category]:
                    return self.memory[category][key]
                return None
            except Exception as e:
                logging.error(f"Failed to get personal info: {e}")
                return None
    
    def remember_fact(self, fact: str, context: str = "", importance: int = 5) -> bool:
        """Remember a new fact about the user or conversation."""
        with self.lock:
            try:
                fact_key = f"fact_{int(time.time())}"
                self.memory["facts_learned"][fact_key] = {
                    "fact": fact,
                    "context": context,
                    "importance": importance,
                    "timestamp": time.time(),
                    "date": datetime.now().isoformat()
                }
                self._save_memory()
                logging.info(f"🧠 Remembered fact: {fact}")
                return True
            except Exception as e:
                logging.error(f"Failed to remember fact: {e}")
                return False
    
    def recall_facts(self, query: str = "", limit: int = 10) -> List[Dict]:
        """Recall facts, optionally filtered by query."""
        with self.lock:
            try:
                facts = []
                for fact_key, fact_data in self.memory["facts_learned"].items():
                    if not query or query.lower() in fact_data["fact"].lower():
                        facts.append({
                            "key": fact_key,
                            **fact_data
                        })
                
                # Sort by importance and recency
                facts.sort(key=lambda x: (x["importance"], x["timestamp"]), reverse=True)
                return facts[:limit]
            except Exception as e:
                logging.error(f"Failed to recall facts: {e}")
                return []
    
    def add_conversation(self, user_message: str, ai_response: str, model: str = "unknown") -> bool:
        """Add a conversation to history."""
        with self.lock:
            try:
                conversation = {
                    "timestamp": time.time(),
                    "date": datetime.now().isoformat(),
                    "user_message": user_message,
                    "ai_response": ai_response,
                    "model": model,
                    "session_id": f"session_{int(time.time() // 3600)}"  # Hour-based sessions
                }
                
                self.conversations.append(conversation)
                self._save_conversations()
                return True
            except Exception as e:
                logging.error(f"Failed to add conversation: {e}")
                return False
    
    def get_recent_conversations(self, limit: int = 5) -> List[Dict]:
        """Get recent conversation history."""
        with self.lock:
            return self.conversations[-limit:] if self.conversations else []
    
    def analyze_user_preferences(self) -> Dict[str, Any]:
        """Analyze stored conversations to understand user preferences."""
        with self.lock:
            try:
                analysis = {
                    "conversation_count": len(self.conversations),
                    "common_topics": [],
                    "communication_style": "friendly",
                    "technical_level": "intermediate",
                    "active_hours": [],
                    "preferred_models": {}
                }
                
                if self.conversations:
                    # Analyze recent conversations
                    recent = self.conversations[-20:]  # Last 20 conversations
                    
                    # Count model usage
                    model_counts = {}
                    for conv in recent:
                        model = conv.get("model", "unknown")
                        model_counts[model] = model_counts.get(model, 0) + 1
                    
                    analysis["preferred_models"] = model_counts
                    
                    # Analyze conversation times
                    hours = [datetime.fromisoformat(conv["date"]).hour for conv in recent if "date" in conv]
                    if hours:
                        # Find most common hour ranges
                        hour_counts = {}
                        for hour in hours:
                            hour_range = f"{hour:02d}:00-{(hour+1)%24:02d}:00"
                            hour_counts[hour_range] = hour_counts.get(hour_range, 0) + 1
                        
                        analysis["active_hours"] = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                
                return analysis
            except Exception as e:
                logging.error(f"Failed to analyze preferences: {e}")
                return {"error": str(e)}
    
    def get_context_for_ai(self) -> Dict[str, Any]:
        """Get relevant context information for AI responses."""
        with self.lock:
            try:
                context = {
                    "user_name": self.get_personal_info("user_name"),
                    "location": self.get_personal_info("location"),
                    "preferences": self.memory.get("personal_info", {}).get("preferences", {}),
                    "recent_facts": self.recall_facts(limit=5),
                    "conversation_style": self.memory.get("user_context", {}).get("conversation_style", "friendly"),
                    "technical_level": self.memory.get("user_context", {}).get("technical_level", "intermediate"),
                    "recent_conversations": len(self.conversations),
                    "important_context": []
                }
                
                # Add important recent facts
                for fact in context["recent_facts"]:
                    if fact["importance"] >= 7:
                        context["important_context"].append(fact["fact"])
                
                return context
            except Exception as e:
                logging.error(f"Failed to get AI context: {e}")
                return {}
    
    def process_message_for_memory(self, message: str) -> Dict[str, Any]:
        """Process a user message to extract information worth remembering."""
        message_lower = message.lower()
        extracted_info = {
            "personal_info": {},
            "facts": [],
            "preferences": {}
        }
        
        # Detect name mentions
        if "my name is" in message_lower or "i'm " in message_lower or "i am " in message_lower:
            # Simple name extraction
            for phrase in ["my name is", "i'm", "i am"]:
                if phrase in message_lower:
                    after_phrase = message_lower.split(phrase, 1)[1].strip()
                    potential_name = after_phrase.split()[0] if after_phrase.split() else ""
                    if potential_name and len(potential_name) > 1:
                        extracted_info["personal_info"]["user_name"] = potential_name.capitalize()
                        break
        
        # Detect location mentions
        location_phrases = ["i live in", "i'm from", "i am from", "my location is", "i'm located in"]
        for phrase in location_phrases:
            if phrase in message_lower:
                after_phrase = message_lower.split(phrase, 1)[1].strip()
                # Take next few words as potential location
                location_words = after_phrase.split()[:3]
                if location_words:
                    location = " ".join(location_words).rstrip(".,!?").title()
                    extracted_info["personal_info"]["location"] = location
                    break
        
        # Detect preferences
        preference_phrases = ["i like", "i love", "i prefer", "i enjoy", "i hate", "i don't like"]
        for phrase in preference_phrases:
            if phrase in message_lower:
                after_phrase = message_lower.split(phrase, 1)[1].strip()
                if after_phrase:
                    preference = after_phrase.split('.')[0].strip()  # Until first period
                    sentiment = "positive" if phrase in ["i like", "i love", "i prefer", "i enjoy"] else "negative"
                    extracted_info["preferences"][preference] = sentiment
                    break
        
        # Detect general facts worth remembering
        fact_indicators = ["i have", "i work", "i study", "i am a", "i'm a", "my job", "my work", "my hobby"]
        for indicator in fact_indicators:
            if indicator in message_lower:
                # This could be a fact worth remembering
                extracted_info["facts"].append(message.strip())
                break
        
        return extracted_info
    
    def auto_store_from_message(self, message: str) -> Dict[str, Any]:
        """Automatically extract and store information from a user message."""
        extracted = self.process_message_for_memory(message)
        results = {
            "stored_personal_info": [],
            "stored_preferences": [],
            "stored_facts": []
        }
        
        # Store personal info
        for key, value in extracted["personal_info"].items():
            if self.store_personal_info(key, value):
                results["stored_personal_info"].append({key: value})
        
        # Store preferences
        for preference, sentiment in extracted["preferences"].items():
            if self.store_personal_info(preference, sentiment, "preferences"):
                results["stored_preferences"].append({preference: sentiment})
        
        # Store facts
        for fact in extracted["facts"]:
            if self.remember_fact(fact, context="auto-extracted from conversation"):
                results["stored_facts"].append(fact)
        
        return results
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memory."""
        with self.lock:
            return {
                "total_facts": len(self.memory.get("facts_learned", {})),
                "total_conversations": len(self.conversations),
                "personal_info_items": len([v for v in self.memory.get("personal_info", {}).values() if v is not None]),
                "preferences_stored": len(self.memory.get("preferences", {})),
                "memory_created": datetime.fromtimestamp(self.memory.get("created_at", 0)).isoformat() if self.memory.get("created_at") else None,
                "last_updated": datetime.fromtimestamp(self.memory.get("last_updated", 0)).isoformat() if self.memory.get("last_updated") else None,
                "user_name": self.get_personal_info("user_name"),
                "user_location": self.get_personal_info("location")
            }

# Global memory system instance
ultra_ai_memory = UltraAIMemory()

if __name__ == "__main__":
    # Test the memory system
    print("🧠 Ultra AI Memory System Test")
    print("=" * 50)
    
    # Test storing personal info
    print("\n📝 Testing personal info storage...")
    ultra_ai_memory.store_personal_info("user_name", "Alex")
    ultra_ai_memory.store_personal_info("location", "San Francisco")
    
    # Test remembering facts
    print("\n🧠 Testing fact storage...")
    ultra_ai_memory.remember_fact("User prefers Python programming", "coding discussion", 8)
    ultra_ai_memory.remember_fact("User owns a Tesla Model 3", "car conversation", 6)
    
    # Test conversation storage
    print("\n💬 Testing conversation storage...")
    ultra_ai_memory.add_conversation("Hello, I'm Alex!", "Nice to meet you Alex!", "qwen2")
    
    # Test auto-extraction
    print("\n🔍 Testing auto-extraction...")
    test_message = "Hi, my name is Sarah and I live in New York. I love machine learning and I work as a data scientist."
    results = ultra_ai_memory.auto_store_from_message(test_message)
    print(f"Auto-extracted: {results}")
    
    # Test retrieval
    print("\n📊 Testing retrieval...")
    context = ultra_ai_memory.get_context_for_ai()
    print(f"AI Context: {json.dumps(context, indent=2)}")
    
    # Test stats
    print("\n📈 Memory Statistics:")
    stats = ultra_ai_memory.get_memory_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")