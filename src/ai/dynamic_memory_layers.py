#!/usr/bin/env python3
"""
Ultra AI Dynamic Memory Layers
Multi-layer storage system for dynamic context injection into GGUF models
Built atop Termux using flat files (JSON/SQLite) and lightweight databases
"""

import asyncio
import json
import logging
import sqlite3
import pickle
import os
import time
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import queue
from collections import defaultdict
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class StorageLayer(Enum):
    """Storage layer types for different data persistence needs."""
    VOLATILE = "volatile"      # In-memory (session-based)
    SESSION = "session"        # JSON files (user session)
    PERSISTENT = "persistent"  # SQLite (long-term memory)
    ARCHIVE = "archive"        # Compressed storage (historical)
    CONTEXT = "context"        # Dynamic context injection

@dataclass
class MemoryEntry:
    """Structured memory entry for consistent storage across layers."""
    id: str
    content: str
    content_type: str
    layer: StorageLayer
    importance: int  # 1-10 scale
    access_count: int
    created_at: float
    updated_at: float
    expires_at: Optional[float]
    tags: List[str]
    metadata: Dict[str, Any]
    embedding_hash: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['layer'] = data['layer'].value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        # Make a copy to avoid modifying the original dict
        data_copy = data.copy()
        data_copy['layer'] = StorageLayer(data_copy['layer'])
        return cls(**data_copy)

class VolatileMemoryLayer:
    """In-memory storage for immediate, session-based context."""
    
    def __init__(self, max_entries=1000):
        self.memory: Dict[str, MemoryEntry] = {}
        self.max_entries = max_entries
        self.access_times = {}
        self.lock = threading.RLock()
        
        logging.info("🧠 Volatile Memory Layer initialized")
    
    def store(self, content: str, content_type: str = "text", 
             importance: int = 5, tags: List[str] = None,
             metadata: Dict[str, Any] = None, expires_in_seconds: int = 3600) -> str:
        """Store entry in volatile memory."""
        with self.lock:
            entry_id = str(uuid.uuid4())
            now = time.time()
            
            entry = MemoryEntry(
                id=entry_id,
                content=content,
                content_type=content_type,
                layer=StorageLayer.VOLATILE,
                importance=importance,
                access_count=0,
                created_at=now,
                updated_at=now,
                expires_at=now + expires_in_seconds if expires_in_seconds else None,
                tags=tags or [],
                metadata=metadata or {},
                embedding_hash=None
            )
            
            # Clean up if at capacity
            if len(self.memory) >= self.max_entries:
                self._cleanup_expired()
                if len(self.memory) >= self.max_entries:
                    self._evict_least_important()
            
            self.memory[entry_id] = entry
            self.access_times[entry_id] = now
            
            return entry_id
    
    def retrieve(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve entry from volatile memory."""
        with self.lock:
            if entry_id in self.memory:
                entry = self.memory[entry_id]
                
                # Check expiration
                if entry.expires_at and time.time() > entry.expires_at:
                    del self.memory[entry_id]
                    if entry_id in self.access_times:
                        del self.access_times[entry_id]
                    return None
                
                # Update access
                entry.access_count += 1
                self.access_times[entry_id] = time.time()
                
                return entry
        
        return None
    
    def search(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """Simple text search in volatile memory."""
        with self.lock:
            results = []
            query_lower = query.lower()
            
            for entry in self.memory.values():
                if (query_lower in entry.content.lower() or 
                    any(query_lower in tag.lower() for tag in entry.tags)):
                    
                    # Check expiration
                    if entry.expires_at and time.time() > entry.expires_at:
                        continue
                    
                    results.append(entry)
            
            # Sort by importance and recency
            results.sort(key=lambda x: (x.importance, x.updated_at), reverse=True)
            return results[:limit]
    
    def _cleanup_expired(self):
        """Remove expired entries."""
        now = time.time()
        expired_ids = []
        
        for entry_id, entry in self.memory.items():
            if entry.expires_at and now > entry.expires_at:
                expired_ids.append(entry_id)
        
        for entry_id in expired_ids:
            del self.memory[entry_id]
            if entry_id in self.access_times:
                del self.access_times[entry_id]
    
    def _evict_least_important(self):
        """Evict least important entries to make room."""
        if not self.memory:
            return
        
        # Sort by importance and access time
        entries_by_priority = sorted(
            self.memory.items(),
            key=lambda x: (x[1].importance, self.access_times.get(x[0], 0))
        )
        
        # Remove bottom 10%
        to_remove = max(1, len(entries_by_priority) // 10)
        for entry_id, _ in entries_by_priority[:to_remove]:
            del self.memory[entry_id]
            if entry_id in self.access_times:
                del self.access_times[entry_id]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get volatile memory statistics."""
        with self.lock:
            return {
                "total_entries": len(self.memory),
                "max_entries": self.max_entries,
                "memory_usage_mb": sys.getsizeof(self.memory) / (1024 * 1024),
                "content_types": list(set(entry.content_type for entry in self.memory.values())),
                "average_importance": sum(entry.importance for entry in self.memory.values()) / len(self.memory) if self.memory else 0
            }

class SessionMemoryLayer:
    """JSON file-based storage for user session persistence."""
    
    def __init__(self, session_dir="/storage/emulated/0/AI_Models/.ultra_ai/sessions"):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.current_session_id = None
        self.session_file = None
        self.session_data = {}
        self.lock = threading.RLock()
        
        logging.info("📝 Session Memory Layer initialized")
    
    def start_session(self, session_id: str = None) -> str:
        """Start a new session or resume existing one."""
        with self.lock:
            if session_id is None:
                session_id = f"session_{int(time.time())}_{str(uuid.uuid4())[:8]}"
            
            self.current_session_id = session_id
            self.session_file = self.session_dir / f"{session_id}.json"
            
            # Load existing session data
            if self.session_file.exists():
                try:
                    with open(self.session_file, 'r') as f:
                        self.session_data = json.load(f)
                except Exception as e:
                    logging.warning(f"Failed to load session {session_id}: {e}")
                    self.session_data = {"entries": {}, "metadata": {}}
            else:
                self.session_data = {
                    "entries": {},
                    "metadata": {
                        "created_at": time.time(),
                        "session_id": session_id,
                        "entry_count": 0
                    }
                }
            
            return session_id
    
    def store(self, content: str, content_type: str = "text",
             importance: int = 5, tags: List[str] = None,
             metadata: Dict[str, Any] = None) -> str:
        """Store entry in current session."""
        with self.lock:
            if not self.current_session_id:
                self.start_session()
            
            entry_id = str(uuid.uuid4())
            now = time.time()
            
            entry = MemoryEntry(
                id=entry_id,
                content=content,
                content_type=content_type,
                layer=StorageLayer.SESSION,
                importance=importance,
                access_count=0,
                created_at=now,
                updated_at=now,
                expires_at=None,
                tags=tags or [],
                metadata=metadata or {},
                embedding_hash=None
            )
            
            self.session_data["entries"][entry_id] = entry.to_dict()
            self.session_data["metadata"]["entry_count"] = len(self.session_data["entries"])
            self.session_data["metadata"]["updated_at"] = now
            
            # Save to file
            self._save_session()
            
            return entry_id
    
    def retrieve(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve entry from current session."""
        with self.lock:
            if entry_id in self.session_data.get("entries", {}):
                entry_dict = self.session_data["entries"][entry_id]
                entry = MemoryEntry.from_dict(entry_dict)
                
                # Update access count
                entry.access_count += 1
                self.session_data["entries"][entry_id] = entry.to_dict()
                self._save_session()
                
                return entry
        
        return None
    
    def search(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """Search entries in current session."""
        with self.lock:
            results = []
            query_lower = query.lower()
            
            for entry_dict in self.session_data.get("entries", {}).values():
                entry = MemoryEntry.from_dict(entry_dict)
                
                if (query_lower in entry.content.lower() or 
                    any(query_lower in tag.lower() for tag in entry.tags)):
                    results.append(entry)
            
            results.sort(key=lambda x: (x.importance, x.updated_at), reverse=True)
            return results[:limit]
    
    def _save_session(self):
        """Save session data to file using atomic write."""
        try:
            # Write to temporary file first for atomic operation
            temp_file = self.session_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(self.session_data, f, indent=2)
            
            # Atomic rename - either succeeds completely or not at all
            temp_file.replace(self.session_file)
            
        except Exception as e:
            logging.error(f"Failed to save session: {e}")
            # Clean up temp file if it exists
            temp_file = self.session_file.with_suffix('.tmp')
            if temp_file.exists():
                temp_file.unlink()
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all available sessions."""
        sessions = []
        for session_file in self.session_dir.glob("session_*.json"):
            try:
                with open(session_file, 'r') as f:
                    data = json.load(f)
                    sessions.append({
                        "session_id": data["metadata"]["session_id"],
                        "created_at": data["metadata"]["created_at"],
                        "entry_count": data["metadata"]["entry_count"],
                        "file_size": session_file.stat().st_size
                    })
            except Exception as e:
                logging.warning(f"Failed to read session {session_file}: {e}")
        
        return sorted(sessions, key=lambda x: x["created_at"], reverse=True)

class PersistentMemoryLayer:
    """SQLite-based storage for long-term memory persistence."""
    
    def __init__(self, db_path="/storage/emulated/0/AI_Models/.ultra_ai/memory/persistent.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self.lock = threading.RLock()
        
        self._init_database()
        logging.info("💾 Persistent Memory Layer initialized")
    
    def _init_database(self):
        """Initialize SQLite database."""
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS persistent_memory (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            content_type TEXT DEFAULT 'text',
            importance INTEGER DEFAULT 5,
            access_count INTEGER DEFAULT 0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            expires_at REAL,
            tags TEXT,
            metadata TEXT,
            embedding_hash TEXT,
            full_text_search TEXT
        );
        
        CREATE INDEX IF NOT EXISTS idx_persistent_content ON persistent_memory(content);
        CREATE INDEX IF NOT EXISTS idx_persistent_importance ON persistent_memory(importance);
        CREATE INDEX IF NOT EXISTS idx_persistent_created ON persistent_memory(created_at);
        CREATE INDEX IF NOT EXISTS idx_persistent_fts ON persistent_memory(full_text_search);
        
        CREATE TABLE IF NOT EXISTS memory_relationships (
            id TEXT PRIMARY KEY,
            from_entry_id TEXT,
            to_entry_id TEXT,
            relationship_type TEXT,
            strength REAL DEFAULT 1.0,
            created_at REAL NOT NULL,
            FOREIGN KEY (from_entry_id) REFERENCES persistent_memory (id),
            FOREIGN KEY (to_entry_id) REFERENCES persistent_memory (id)
        );
        """)
        
        self.conn.commit()
    
    def store(self, content: str, content_type: str = "text",
             importance: int = 5, tags: List[str] = None,
             metadata: Dict[str, Any] = None, expires_at: float = None) -> str:
        """Store entry in persistent memory."""
        with self.lock:
            entry_id = str(uuid.uuid4())
            now = time.time()
            
            # Create searchable text
            full_text_search = f"{content} {' '.join(tags or [])} {content_type}"
            
            # Create embedding hash for deduplication
            embedding_hash = hashlib.md5(content.encode()).hexdigest()
            
            self.conn.execute("""
                INSERT INTO persistent_memory 
                (id, content, content_type, importance, access_count, created_at, 
                 updated_at, expires_at, tags, metadata, embedding_hash, full_text_search)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry_id, content, content_type, importance, 0, now, now,
                expires_at, json.dumps(tags or []), json.dumps(metadata or {}),
                embedding_hash, full_text_search
            ))
            
            self.conn.commit()
            return entry_id
    
    def retrieve(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve entry from persistent memory."""
        with self.lock:
            cursor = self.conn.execute(
                "SELECT * FROM persistent_memory WHERE id = ?", (entry_id,)
            )
            row = cursor.fetchone()
            
            if row:
                # Update access count
                self.conn.execute(
                    "UPDATE persistent_memory SET access_count = access_count + 1 WHERE id = ?",
                    (entry_id,)
                )
                self.conn.commit()
                
                return MemoryEntry(
                    id=row['id'],
                    content=row['content'],
                    content_type=row['content_type'],
                    layer=StorageLayer.PERSISTENT,
                    importance=row['importance'],
                    access_count=row['access_count'] + 1,
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    expires_at=row['expires_at'],
                    tags=json.loads(row['tags']),
                    metadata=json.loads(row['metadata']),
                    embedding_hash=row['embedding_hash']
                )
        
        return None
    
    def search(self, query: str, limit: int = 20) -> List[MemoryEntry]:
        """Search persistent memory with full-text search."""
        with self.lock:
            results = []
            
            # Full-text search with ranking
            cursor = self.conn.execute("""
                SELECT *, 
                       (CASE WHEN full_text_search LIKE ? THEN importance * 2 ELSE importance END) as relevance_score
                FROM persistent_memory 
                WHERE full_text_search LIKE ? 
                   OR content LIKE ?
                ORDER BY relevance_score DESC, access_count DESC, created_at DESC
                LIMIT ?
            """, (f"%{query}%", f"%{query}%", f"%{query}%", limit))
            
            for row in cursor.fetchall():
                entry = MemoryEntry(
                    id=row['id'],
                    content=row['content'],
                    content_type=row['content_type'],
                    layer=StorageLayer.PERSISTENT,
                    importance=row['importance'],
                    access_count=row['access_count'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    expires_at=row['expires_at'],
                    tags=json.loads(row['tags']),
                    metadata=json.loads(row['metadata']),
                    embedding_hash=row['embedding_hash']
                )
                results.append(entry)
            
            return results
    
    def cleanup_expired(self):
        """Remove expired entries."""
        with self.lock:
            now = time.time()
            self.conn.execute(
                "DELETE FROM persistent_memory WHERE expires_at IS NOT NULL AND expires_at < ?",
                (now,)
            )
            self.conn.commit()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get persistent memory statistics."""
        with self.lock:
            cursor = self.conn.execute("SELECT COUNT(*) as count FROM persistent_memory")
            total = cursor.fetchone()['count']
            
            cursor = self.conn.execute("""
                SELECT content_type, COUNT(*) as count 
                FROM persistent_memory 
                GROUP BY content_type
            """)
            content_types = {row['content_type']: row['count'] for row in cursor.fetchall()}
            
            return {
                "total_entries": total,
                "content_types": content_types,
                "db_size_mb": self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0
            }

class DynamicContextInjector:
    """Dynamic context injection system for GGUF model prompts."""
    
    def __init__(self, volatile_layer: VolatileMemoryLayer,
                 session_layer: SessionMemoryLayer,
                 persistent_layer: PersistentMemoryLayer):
        self.volatile = volatile_layer
        self.session = session_layer
        self.persistent = persistent_layer
        
        # Context configuration
        self.max_context_tokens = 800  # Reserve tokens for context
        self.context_priorities = {
            StorageLayer.VOLATILE: 1.0,    # Highest priority - immediate context
            StorageLayer.SESSION: 0.8,     # High priority - session context
            StorageLayer.PERSISTENT: 0.6   # Medium priority - long-term memory
        }
        
        logging.info("🎯 Dynamic Context Injector initialized")
    
    async def build_context_for_prompt(self, user_prompt: str, 
                                     user_info: Dict[str, Any] = None,
                                     max_tokens: int = None) -> Dict[str, Any]:
        """Build dynamic context for injection into GGUF model prompt."""
        if max_tokens is None:
            max_tokens = self.max_context_tokens
        
        context = {
            "user_context": "",
            "relevant_memories": "",
            "conversation_history": "",
            "metadata": {
                "sources": [],
                "total_memories": 0,
                "context_tokens": 0
            }
        }
        
        # 1. Extract user information context
        if user_info:
            user_context_parts = []
            if user_info.get("name"):
                user_context_parts.append(f"User's name: {user_info['name']}")
            if user_info.get("location"):
                user_context_parts.append(f"User's location: {user_info['location']}")
            if user_info.get("preferences"):
                prefs = ", ".join(f"{k}: {v}" for k, v in user_info["preferences"].items())
                user_context_parts.append(f"User preferences: {prefs}")
            
            context["user_context"] = ". ".join(user_context_parts)
        
        # 2. Search for relevant memories across all layers
        relevant_memories = []
        
        # Search volatile memory (immediate context)
        volatile_results = self.volatile.search(user_prompt, limit=5)
        for memory in volatile_results:
            relevant_memories.append((memory, self.context_priorities[StorageLayer.VOLATILE]))
        
        # Search session memory (current session context)
        session_results = self.session.search(user_prompt, limit=8)
        for memory in session_results:
            relevant_memories.append((memory, self.context_priorities[StorageLayer.SESSION]))
        
        # Search persistent memory (long-term context)
        persistent_results = self.persistent.search(user_prompt, limit=10)
        for memory in persistent_results:
            relevant_memories.append((memory, self.context_priorities[StorageLayer.PERSISTENT]))
        
        # 3. Sort and prioritize memories
        relevant_memories.sort(key=lambda x: (x[1], x[0].importance, x[0].access_count), reverse=True)
        
        # 4. Build context string within token limit
        context_parts = []
        token_count = 0
        used_memories = 0
        
        # Estimate tokens (rough approximation: 4 chars per token)
        def estimate_tokens(text: str) -> int:
            return len(text) // 4
        
        # Add user context first
        if context["user_context"]:
            tokens = estimate_tokens(context["user_context"])
            if token_count + tokens < max_tokens:
                token_count += tokens
        
        # Add relevant memories
        for memory, priority in relevant_memories:
            memory_text = f"[{memory.content_type}] {memory.content}"
            if memory.tags:
                memory_text += f" (tags: {', '.join(memory.tags)})"
            
            tokens = estimate_tokens(memory_text)
            if token_count + tokens < max_tokens:
                context_parts.append(memory_text)
                token_count += tokens
                used_memories += 1
                context["metadata"]["sources"].append({
                    "layer": memory.layer.value,
                    "importance": memory.importance,
                    "content_type": memory.content_type
                })
            else:
                break  # Stop if we exceed token limit
        
        context["relevant_memories"] = "\n".join(context_parts)
        context["metadata"]["total_memories"] = used_memories
        context["metadata"]["context_tokens"] = token_count
        
        return context
    
    def inject_context_into_prompt(self, original_prompt: str, 
                                 context: Dict[str, Any],
                                 model_personality: str = "") -> str:
        """Inject context into the model prompt without modifying the static model."""
        
        # Build enhanced prompt with dynamic context
        enhanced_parts = []
        
        # Model personality and identity
        if model_personality:
            enhanced_parts.append(f"You are {model_personality}")
        
        # User context
        if context["user_context"]:
            enhanced_parts.append(f"User Context: {context['user_context']}")
        
        # Relevant memories
        if context["relevant_memories"]:
            enhanced_parts.append(f"Relevant Information from Memory:\n{context['relevant_memories']}")
        
        # Instructions for using context
        enhanced_parts.append(
            "Use the above context to provide personalized, informed responses. "
            "Reference specific memories when relevant, but don't mention the storage system itself."
        )
        
        # Original user prompt
        enhanced_parts.append(f"User Question: {original_prompt}")
        
        # Response instruction
        enhanced_parts.append("Response:")
        
        return "\n\n".join(enhanced_parts)
    
    async def store_interaction(self, user_prompt: str, ai_response: str,
                              model_used: str, context_used: Dict[str, Any]):
        """Store the interaction across appropriate memory layers."""
        
        # Store in volatile memory for immediate context
        self.volatile.store(
            content=f"User: {user_prompt}\nAssistant: {ai_response}",
            content_type="conversation",
            importance=3,
            tags=["conversation", model_used],
            metadata={
                "model": model_used,
                "context_tokens": context_used["metadata"]["context_tokens"],
                "memories_used": context_used["metadata"]["total_memories"]
            },
            expires_in_seconds=1800  # 30 minutes
        )
        
        # Store in session memory for session context
        self.session.store(
            content=f"Q: {user_prompt}\nA: {ai_response}",
            content_type="conversation",
            importance=4,
            tags=["conversation", model_used, "session"],
            metadata={
                "model": model_used,
                "timestamp": time.time()
            }
        )
        
        # Store important interactions in persistent memory
        if len(user_prompt) > 20 or any(keyword in user_prompt.lower() 
                                       for keyword in ["remember", "important", "name", "preference"]):
            self.persistent.store(
                content=f"User said: {user_prompt}\nI responded: {ai_response}",
                content_type="important_conversation",
                importance=6,
                tags=["important", "conversation", model_used],
                metadata={
                    "model": model_used,
                    "session_id": self.session.current_session_id,
                    "context_quality": context_used["metadata"]["total_memories"]
                }
            )

class UltraMemoryManager:
    """Unified memory management system coordinating all storage layers."""
    
    def __init__(self):
        # Initialize all storage layers
        self.volatile = VolatileMemoryLayer(max_entries=1000)
        self.session = SessionMemoryLayer()
        self.persistent = PersistentMemoryLayer()
        self.context_injector = DynamicContextInjector(
            self.volatile, self.session, self.persistent
        )
        
        # Background maintenance
        self.maintenance_thread = None
        self.running = False
        
        logging.info("🎛️ Ultra Memory Manager initialized with all storage layers")
    
    def start_background_maintenance(self):
        """Start background maintenance tasks."""
        self.running = True
        self.maintenance_thread = threading.Thread(target=self._maintenance_loop, daemon=True)
        self.maintenance_thread.start()
        logging.info("🔄 Background memory maintenance started")
    
    def _maintenance_loop(self):
        """Background maintenance tasks."""
        while self.running:
            try:
                # Cleanup expired entries every 5 minutes
                self.persistent.cleanup_expired()
                
                # Sleep for 5 minutes
                time.sleep(300)
                
            except Exception as e:
                logging.error(f"Memory maintenance error: {e}")
                time.sleep(60)  # Wait 1 minute on error
    
    def stop_maintenance(self):
        """Stop background maintenance."""
        self.running = False
        if self.maintenance_thread and self.maintenance_thread.is_alive():
            self.maintenance_thread.join(timeout=2)
    
    async def enhance_model_prompt(self, user_prompt: str, model_personality: str,
                                 user_info: Dict[str, Any] = None) -> Tuple[str, Dict[str, Any]]:
        """Enhance model prompt with dynamic context injection."""
        
        # Build context from all storage layers
        context = await self.context_injector.build_context_for_prompt(
            user_prompt, user_info
        )
        
        # Inject context into prompt
        enhanced_prompt = self.context_injector.inject_context_into_prompt(
            user_prompt, context, model_personality
        )
        
        return enhanced_prompt, context
    
    async def store_conversation(self, user_prompt: str, ai_response: str,
                               model_used: str, context_used: Dict[str, Any]):
        """Store conversation across appropriate memory layers."""
        await self.context_injector.store_interaction(
            user_prompt, ai_response, model_used, context_used
        )
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get statistics from all memory layers."""
        return {
            "volatile": self.volatile.get_stats(),
            "session": {
                "current_session": self.session.current_session_id,
                "sessions_available": len(self.session.list_sessions())
            },
            "persistent": self.persistent.get_stats(),
            "system": {
                "maintenance_running": self.running,
                "total_layers": 3,
                "context_injector_ready": self.context_injector is not None
            }
        }

# Global memory manager instance
ultra_memory_manager = UltraMemoryManager()

if __name__ == "__main__":
    # Test the multi-layer storage system
    async def test_storage_layers():
        print("🗄️ Ultra AI Dynamic Memory Layers Test")
        print("=" * 60)
        
        # Start a session
        session_id = ultra_memory_manager.session.start_session()
        print(f"📝 Started session: {session_id}")
        
        # Start background maintenance
        ultra_memory_manager.start_background_maintenance()
        
        # Test storing across layers
        print("\n💾 Testing storage across layers...")
        
        # Store in volatile (temporary)
        volatile_id = ultra_memory_manager.volatile.store(
            "The user prefers dark mode themes",
            content_type="preference",
            importance=7,
            tags=["ui", "preference"]
        )
        print(f"  Volatile: {volatile_id[:8]}...")
        
        # Store in session (current session)
        session_id = ultra_memory_manager.session.store(
            "User is learning Python programming",
            content_type="fact",
            importance=6,
            tags=["learning", "programming"]
        )
        print(f"  Session: {session_id[:8]}...")
        
        # Store in persistent (long-term)
        persistent_id = ultra_memory_manager.persistent.store(
            "User's name is Alex and lives in San Francisco",
            content_type="personal",
            importance=9,
            tags=["personal", "identity"]
        )
        print(f"  Persistent: {persistent_id[:8]}...")
        
        # Test context injection
        print("\n🎯 Testing dynamic context injection...")
        
        user_info = {
            "name": "Alex",
            "location": "San Francisco",
            "preferences": {"theme": "dark", "language": "python"}
        }
        
        test_prompt = "Help me with a Python coding project"
        enhanced_prompt, context = await ultra_memory_manager.enhance_model_prompt(
            test_prompt, "Ultra AI Programming Expert", user_info
        )
        
        print(f"  Original prompt: {test_prompt}")
        print(f"  Enhanced prompt length: {len(enhanced_prompt)} chars")
        print(f"  Context memories used: {context['metadata']['total_memories']}")
        print(f"  Context tokens: {context['metadata']['context_tokens']}")
        
        # Store the interaction
        await ultra_memory_manager.store_conversation(
            test_prompt, "I'll help you with your Python project!", 
            "codellama", context
        )
        
        # Show comprehensive stats
        print("\n📊 Comprehensive Memory Statistics:")
        stats = ultra_memory_manager.get_comprehensive_stats()
        for layer, layer_stats in stats.items():
            print(f"  {layer.upper()}: {layer_stats}")
        
        print("\n✅ Multi-layer storage system test complete!")
        
        # Stop maintenance
        ultra_memory_manager.stop_maintenance()
    
    asyncio.run(test_storage_layers())