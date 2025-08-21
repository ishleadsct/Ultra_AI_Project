"""
Ultra AI Project - Memory Manager

Manages system memory, conversation context, embeddings, and persistent storage
across agents and user sessions. Provides intelligent memory retrieval and cleanup.
"""

import asyncio
import json
import pickle
import hashlib
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from pathlib import Path
import sqlite3
from collections import defaultdict, OrderedDict

from ..utils.logger import get_logger
from ..utils.helpers import generate_memory_id, sanitize_string

logger = get_logger(__name__)

@dataclass
class MemoryItem:
    """Individual memory item."""
    id: str
    type: str  # conversation, document, embedding, cache, etc.
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    importance: float = 1.0  # 0.0 to 10.0 scale
    expires_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "access_count": self.access_count,
            "importance": self.importance,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "tags": self.tags
        }

@dataclass
class ConversationContext:
    """Conversation context and history."""
    conversation_id: str
    user_id: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    context_summary: str = ""
    last_updated: datetime = field(default_factory=datetime.now)
    active_agents: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EmbeddingResult:
    """Embedding search result."""
    item_id: str
    content: str
    similarity: float
    metadata: Dict[str, Any]

class LRUCache:
    """LRU Cache implementation for memory items."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: Any):
        """Put item in cache."""
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            self.cache[key] = value
            if len(self.cache) > self.max_size:
                # Remove least recently used
                self.cache.popitem(last=False)
    
    def remove(self, key: str):
        """Remove item from cache."""
        self.cache.pop(key, None)

    def clear(self):
        """Clear cache."""
        self.cache.clear()
    
    def size(self) -> int:
        """Get cache size."""
        return len(self.cache)

class MemoryManager:
    """Central memory management system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_memory_items = config.get("max_memory_mb", 2048) * 1024  # Convert to KB
        self.cleanup_interval = config.get("cleanup_interval", 3600)  # 1 hour
        self.conversation_limit = config.get("conversation_memory_limit", 100)
        
        # Storage
        self.memory_items: Dict[str, MemoryItem] = {}
        self.conversations: Dict[str, ConversationContext] = {}
        self.embeddings: Dict[str, List[float]] = {}
        
        # Caches
        self.cache = LRUCache(max_size=config.get("cache_size", 1000))
        self.query_cache = LRUCache(max_size=500)
        
        # Indexes for fast retrieval
        self.type_index: Dict[str, List[str]] = defaultdict(list)
        self.tag_index: Dict[str, List[str]] = defaultdict(list)
        self.user_index: Dict[str, List[str]] = defaultdict(list)
        
        # Statistics
        self.stats = {
            "total_items": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "queries_processed": 0,
            "cleanup_runs": 0,
            "memory_usage_mb": 0.0
        }
        
        # Database connection
        self.db_path = Path("runtime/memory/memory.db")
        self.db_connection: Optional[sqlite3.Connection] = None
        
        # Tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        self.running = False
        
        logger.info("MemoryManager initialized")
    
    async def initialize(self):
        """Initialize memory manager."""
        try:
            logger.info("Initializing MemoryManager...")
            
            # Create directories
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Initialize database
            await self._init_database()
            
            # Load existing memory items
            await self._load_memory_items()
            
            # Start cleanup task
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            self.running = True
            
            logger.info(f"MemoryManager initialized with {len(self.memory_items)} items")
            
        except Exception as e:
            logger.error(f"Failed to initialize MemoryManager: {e}")
            raise
    
    async def _init_database(self):
        """Initialize SQLite database."""
        try:
            self.db_connection = sqlite3.connect(str(self.db_path))
            
            # Create tables
            cursor = self.db_connection.cursor()
            
            # Memory items table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_items (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    accessed_at TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    importance REAL DEFAULT 1.0,
                    expires_at TEXT,
                    tags TEXT
                )
            """)
            
            # Conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    messages TEXT NOT NULL,
                    context_summary TEXT,
                    last_updated TEXT NOT NULL,
                    active_agents TEXT,
                    metadata TEXT
                )
            """)
            
            # Embeddings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    item_id TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    content_hash TEXT NOT NULL
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_type ON memory_items(type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_created ON memory_items(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id)")
            
            self.db_connection.commit()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

async def _load_memory_items(self):
        """Load existing memory items from database."""
        try:
            if not self.db_connection:
                return
            
            cursor = self.db_connection.cursor()
            
            # Load memory items
            cursor.execute("SELECT * FROM memory_items")
            rows = cursor.fetchall()
            
            for row in rows:
                try:
                    item = MemoryItem(
                        id=row[0],
                        type=row[1],
                        content=json.loads(row[2]),
                        metadata=json.loads(row[3]) if row[3] else {},
                        created_at=datetime.fromisoformat(row[4]),
                        accessed_at=datetime.fromisoformat(row[5]),
                        access_count=row[6],
                        importance=row[7],
                        expires_at=datetime.fromisoformat(row[8]) if row[8] else None,
                        tags=json.loads(row[9]) if row[9] else []
                    )
                    
                    self.memory_items[item.id] = item
                    self._update_indexes(item)
                    
                except Exception as e:
                    logger.error(f"Failed to load memory item {row[0]}: {e}")
            
            # Load conversations
            cursor.execute("SELECT * FROM conversations")
            rows = cursor.fetchall()
            
            for row in rows:
                try:
                    context = ConversationContext(
                        conversation_id=row[0],
                        user_id=row[1],
                        messages=json.loads(row[2]),
                        context_summary=row[3] or "",
                        last_updated=datetime.fromisoformat(row[4]),
                        active_agents=json.loads(row[5]) if row[5] else [],
                        metadata=json.loads(row[6]) if row[6] else {}
                    )
                    
                    self.conversations[context.conversation_id] = context
                    
                except Exception as e:
                    logger.error(f"Failed to load conversation {row[0]}: {e}")
            
            self._update_stats()
            logger.info(f"Loaded {len(self.memory_items)} memory items and {len(self.conversations)} conversations")
            
        except Exception as e:
            logger.error(f"Failed to load memory items: {e}")
    
    def _update_indexes(self, item: MemoryItem):
        """Update search indexes for item."""
        # Type index
        if item.id not in self.type_index[item.type]:
            self.type_index[item.type].append(item.id)
        
        # Tag index
        for tag in item.tags:
            if item.id not in self.tag_index[tag]:
                self.tag_index[tag].append(item.id)
        
        # User index (if user_id in metadata)
        user_id = item.metadata.get("user_id")
        if user_id:
            if item.id not in self.user_index[user_id]:
                self.user_index[user_id].append(item.id)
    
    def _remove_from_indexes(self, item: MemoryItem):
        """Remove item from search indexes."""
        # Type index
        if item.id in self.type_index[item.type]:
            self.type_index[item.type].remove(item.id)
        
        # Tag index
        for tag in item.tags:
            if item.id in self.tag_index[tag]:
                self.tag_index[tag].remove(item.id)
        
        # User index
        user_id = item.metadata.get("user_id")
        if user_id and item.id in self.user_index[user_id]:
            self.user_index[user_id].remove(item.id)
    
    async def store_memory(self, 
                          memory_type: str,
                          content: Any,
                          metadata: Optional[Dict[str, Any]] = None,
                          importance: float = 1.0,
                          expires_in: Optional[float] = None,
                          tags: Optional[List[str]] = None) -> str:
        """Store a new memory item."""
        
        memory_id = generate_memory_id()
        expires_at = None
        if expires_in:
            expires_at = datetime.now() + timedelta(seconds=expires_in)
        
        item = MemoryItem(
            id=memory_id,
            type=memory_type,
            content=content,
            metadata=metadata or {},
            importance=importance,
            expires_at=expires_at,
            tags=tags or []
        )
        
        # Store in memory
        self.memory_items[memory_id] = item
        self._update_indexes(item)
        
        # Store in database
        await self._save_memory_item(item)
        
        # Update cache
        self.cache.put(memory_id, item)
        
        # Update stats
        self.stats["total_items"] += 1
        self._update_stats()
        
        logger.debug(f"Stored memory item {memory_id} (type: {memory_type})")
        return memory_id
    
    async def _save_memory_item(self, item: MemoryItem):
        """Save memory item to database."""
        try:
            if not self.db_connection:
                return
            
            cursor = self.db_connection.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO memory_items 
                (id, type, content, metadata, created_at, accessed_at, access_count, 
                 importance, expires_at, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                item.id,
                item.type,
                json.dumps(item.content),
                json.dumps(item.metadata),
                item.created_at.isoformat(),
                item.accessed_at.isoformat(),
                item.access_count,
                item.importance,
                item.expires_at.isoformat() if item.expires_at else None,
                json.dumps(item.tags)
            ))
            
            self.db_connection.commit()

except Exception as e:
            logger.error(f"Failed to save memory item {item.id}: {e}")
    
    async def retrieve_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve a specific memory item."""
        # Check cache first
        cached_item = self.cache.get(memory_id)
        if cached_item:
            self.stats["cache_hits"] += 1
            cached_item.accessed_at = datetime.now()
            cached_item.access_count += 1
            return cached_item
        
        self.stats["cache_misses"] += 1
        
        # Check memory
        if memory_id in self.memory_items:
            item = self.memory_items[memory_id]
            item.accessed_at = datetime.now()
            item.access_count += 1
            
            # Update cache
            self.cache.put(memory_id, item)
            
            # Update database
            await self._save_memory_item(item)
            
            return item
        
        return None
    
    async def search_memory(self, 
                           query: Optional[str] = None,
                           memory_type: Optional[str] = None,
                           tags: Optional[List[str]] = None,
                           user_id: Optional[str] = None,
                           limit: int = 50) -> List[MemoryItem]:
        """Search memory items."""
        
        # Create cache key
        cache_key = f"search_{hash((query, memory_type, tuple(tags or []), user_id, limit))}"
        cached_result = self.query_cache.get(cache_key)
        if cached_result:
            return cached_result
        
        candidates = set(self.memory_items.keys())
        
        # Filter by type
        if memory_type:
            candidates &= set(self.type_index.get(memory_type, []))
        
        # Filter by tags
        if tags:
            for tag in tags:
                candidates &= set(self.tag_index.get(tag, []))
        
        # Filter by user
        if user_id:
            candidates &= set(self.user_index.get(user_id, []))
        
        # Get items and filter
        results = []
        for item_id in candidates:
            if item_id not in self.memory_items:
                continue
                
            item = self.memory_items[item_id]
            
            # Check expiration
            if item.expires_at and datetime.now() > item.expires_at:
                continue
            
            # Text search if query provided
            if query:
                content_str = str(item.content).lower()
                if query.lower() not in content_str:
                    continue
            
            results.append(item)
        
        # Sort by importance and recency
        results.sort(key=lambda x: (x.importance, x.accessed_at), reverse=True)
        
        # Limit results
        results = results[:limit]
        
        # Cache results
        self.query_cache.put(cache_key, results)
        
        self.stats["queries_processed"] += 1
        return results
    
    async def store_conversation(self, conversation_id: str, user_id: str, 
                                message: Dict[str, Any]) -> bool:
        """Store a conversation message."""
        try:
            if conversation_id not in self.conversations:
                self.conversations[conversation_id] = ConversationContext(
                    conversation_id=conversation_id,
                    user_id=user_id
                )
            
            context = self.conversations[conversation_id]
            context.messages.append(message)
            context.last_updated = datetime.now()
            
            # Limit conversation length
            if len(context.messages) > self.conversation_limit:
                # Keep first few and last many messages
                keep_first = 10
                keep_last = self.conversation_limit - keep_first
                context.messages = (context.messages[:keep_first] + 
                                  context.messages[-keep_last:])
            
            # Save to database
            await self._save_conversation(context)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store conversation message: {e}")
            return False
    
    async def _save_conversation(self, context: ConversationContext):
        """Save conversation to database."""
        try:
            if not self.db_connection:
                return
            
            cursor = self.db_connection.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO conversations 
                (conversation_id, user_id, messages, context_summary, last_updated, 
                 active_agents, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                context.conversation_id,
                context.user_id,
                json.dumps(context.messages),
                context.context_summary,
                context.last_updated.isoformat(),
                json.dumps(context.active_agents),
                json.dumps(context.metadata)
            ))
            
            self.db_connection.commit()
            
        except Exception as e:
            logger.error(f"Failed to save conversation {context.conversation_id}: {e}")
    
    async def get_conversation(self, conversation_id: str) -> Optional[ConversationContext]:
        """Get conversation context."""
        return self.conversations.get(conversation_id)
    
    async def store_task_result(self, task_result):
        """Store task result in memory."""
        await self.store_memory(
            memory_type="task_result",
            content={
                "task_id": task_result.task_id,
                "status": task_result.status.value,
                "result": task_result.result,
                "error": task_result.error,
                "execution_time": task_result.execution_time
            },
            metadata={
                "timestamp": task_result.timestamp.isoformat(),
                "task_type": "result"
            },
            importance=3.0,
            expires_in=7 * 24 * 3600,  # 7 days
            tags=["task", "result"]
        )
    
    async def _cleanup_loop(self):
        """Periodic cleanup of expired and low-importance items."""
        while self.running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_expired_items()
                await self._cleanup_low_importance_items()
                self.stats["cleanup_runs"] += 1
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    async def _cleanup_expired_items(self):
        """Remove expired memory items."""
        now = datetime.now()
        expired_items = []
        
        for item_id, item in self.memory_items.items():
            if item.expires_at and now > item.expires_at:
                expired_items.append(item_id)
        
        for item_id in expired_items:
            await self.delete_memory(item_id)
        
        if expired_items:
            logger.info(f"Cleaned up {len(expired_items)} expired memory items")
    
    async def _cleanup_low_importance_items(self):
        """Remove low-importance items if memory usage is high."""
        # Check memory usage
        current_usage = await self._calculate_memory_usage()
        if current_usage < self.max_memory_items * 0.8:  # 80% threshold
            return
        
        # Sort by importance and age
        items_by_priority = sorted(
            self.memory_items.values(),
            key=lambda x: (x.importance, x.accessed_at)
        )
        
        # Remove lowest priority items
        items_to_remove = len(items_by_priority) // 10  # Remove 10%
        for item in items_by_priority[:items_to_remove]:
            await self.delete_memory(item.id)
        
        logger.info(f"Cleaned up {items_to_remove} low-importance memory items")
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory item."""
        try:
            if memory_id not in self.memory_items:
                return False
            
            item = self.memory_items[memory_id]
            
            # Remove from indexes
            self._remove_from_indexes(item)
            
            # Remove from memory
            del self.memory_items[memory_id]
            
            # Remove from cache
            self.cache.remove(memory_id)
            
            # Remove from database
            if self.db_connection:
                cursor = self.db_connection.cursor()
                cursor.execute("DELETE FROM memory_items WHERE id = ?", (memory_id,))
                self.db_connection.commit()
            
            self.stats["total_items"] -= 1
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete memory item {memory_id}: {e}")
            return False
    
    async def _calculate_memory_usage(self) -> int:
        """Calculate current memory usage in items."""
        return len(self.memory_items)
    
    def _update_stats(self):
        """Update memory statistics."""
        self.stats["memory_usage_mb"] = len(self.memory_items) * 0.001  # Rough estimate
    
    async def get_usage_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        return {
            **self.stats,
            "total_conversations": len(self.conversations),
            "cache_size": self.cache.size(),
            "query_cache_size": self.query_cache.size(),
            "index_sizes": {
                "type": sum(len(items) for items in self.type_index.values()),
                "tag": sum(len(items) for items in self.tag_index.values()),
                "user": sum(len(items) for items in self.user_index.values())
            }
        }
    
    async def health_check(self) -> bool:
        """Perform health check."""
        try:
            # Check database connection
            if self.db_connection:
                cursor = self.db_connection.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
            
            # Check memory usage
            usage = await self._calculate_memory_usage()
            if usage > self.max_memory_items:
                logger.warning(f"Memory usage ({usage}) exceeds limit ({self.max_memory_items})")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"MemoryManager health check failed: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown memory manager."""
        logger.info("Shutting down MemoryManager...")
        self.running = False
        
        # Cancel cleanup task
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        # Close database connection
        if self.db_connection:
            self.db_connection.close()
        
        logger.info("MemoryManager shutdown complete")
