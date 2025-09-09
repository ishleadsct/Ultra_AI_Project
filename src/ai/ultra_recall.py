#!/usr/bin/env python3
"""
Ultra Recall - LocalRecall-inspired REST API and Knowledge Base System
Persistent memory infrastructure for Ultra AI models on Android/Termux
"""

import asyncio
import json
import logging
import time
import sqlite3
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading
import sys
import re

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from .storage_ai import storage_ai
    from .memory_system import ultra_ai_memory
    storage_integration = True
except ImportError:
    storage_integration = False

class UltraRecallKnowledgeBase:
    """LocalRecall-inspired knowledge base with vector similarity and semantic search."""
    
    def __init__(self):
        self.db_path = Path("/storage/emulated/0/AI_Models/.ultra_ai/memory/ultra_recall.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize SQLite database
        self._init_database()
        
        # Memory configuration
        self.max_memory_entries = 10000
        self.retention_days = 365
        self.similarity_threshold = 0.3
        
        # Embedding cache (simple text similarity for Android compatibility)
        self.embedding_cache = {}
        self.cache_size_limit = 1000
        
        logging.info("🗄️ Ultra Recall Knowledge Base initialized")
    
    def _init_database(self):
        """Initialize the SQLite database with required tables."""
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Enable dict-like access
        
        # Create tables
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS knowledge_entries (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            content_type TEXT DEFAULT 'text',
            source TEXT,
            context TEXT,
            metadata TEXT,
            embeddings TEXT,
            importance INTEGER DEFAULT 5,
            access_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP,
            tags TEXT
        );
        
        CREATE TABLE IF NOT EXISTS memory_sessions (
            session_id TEXT PRIMARY KEY,
            agent_id TEXT,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ended_at TIMESTAMP,
            metadata TEXT
        );
        
        CREATE TABLE IF NOT EXISTS memory_links (
            id TEXT PRIMARY KEY,
            from_entry_id TEXT,
            to_entry_id TEXT,
            relation_type TEXT,
            strength REAL DEFAULT 1.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (from_entry_id) REFERENCES knowledge_entries (id),
            FOREIGN KEY (to_entry_id) REFERENCES knowledge_entries (id)
        );
        
        CREATE TABLE IF NOT EXISTS conversation_history (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            user_message TEXT,
            ai_response TEXT,
            model_used TEXT,
            context_used TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT
        );
        
        CREATE INDEX IF NOT EXISTS idx_knowledge_content ON knowledge_entries(content);
        CREATE INDEX IF NOT EXISTS idx_knowledge_created ON knowledge_entries(created_at);
        CREATE INDEX IF NOT EXISTS idx_knowledge_importance ON knowledge_entries(importance);
        CREATE INDEX IF NOT EXISTS idx_conversation_session ON conversation_history(session_id);
        CREATE INDEX IF NOT EXISTS idx_conversation_timestamp ON conversation_history(timestamp);
        """)
        
        self.conn.commit()
        logging.info("✅ Ultra Recall database initialized")
    
    def _generate_text_embedding(self, text: str) -> List[float]:
        """Generate simple text embedding using character frequency and word patterns."""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        # Clean and normalize text
        clean_text = re.sub(r'[^\w\s]', '', text.lower())
        words = clean_text.split()
        
        # Create simple embedding based on:
        # 1. Character frequency distribution
        # 2. Word length distribution  
        # 3. Common word patterns
        
        embedding = [0.0] * 128  # 128-dimensional embedding
        
        if not words:
            return embedding
        
        # Character frequency features (first 26 dimensions)
        char_freq = {}
        for char in clean_text:
            if char.isalpha():
                char_freq[char] = char_freq.get(char, 0) + 1
        
        total_chars = sum(char_freq.values()) or 1
        for i, char in enumerate('abcdefghijklmnopqrstuvwxyz'):
            if i < 26:
                embedding[i] = char_freq.get(char, 0) / total_chars
        
        # Word length distribution (next 10 dimensions)
        word_lengths = [len(word) for word in words]
        avg_length = sum(word_lengths) / len(word_lengths)
        for i in range(10):
            target_length = i + 1
            count = sum(1 for length in word_lengths if length == target_length)
            embedding[26 + i] = count / len(words)
        
        # Common word patterns (next 20 dimensions)
        common_patterns = [
            'the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with',
            'for', 'as', 'was', 'on', 'are', 'you', 'this', 'be', 'at', 'have'
        ]
        for i, pattern in enumerate(common_patterns):
            if i < 20:
                count = sum(1 for word in words if pattern in word)
                embedding[36 + i] = count / len(words)
        
        # Text statistics (remaining dimensions)
        embedding[56] = len(words) / 100.0  # Normalized word count
        embedding[57] = avg_length / 10.0   # Normalized avg word length
        embedding[58] = len(set(words)) / len(words)  # Vocabulary diversity
        
        # Additional semantic features
        question_words = ['what', 'where', 'when', 'why', 'how', 'who']
        embedding[59] = sum(1 for word in words if word in question_words) / len(words)
        
        # Cache the embedding
        if len(self.embedding_cache) >= self.cache_size_limit:
            # Remove oldest entries
            oldest_keys = list(self.embedding_cache.keys())[:100]
            for key in oldest_keys:
                del self.embedding_cache[key]
        
        self.embedding_cache[text] = embedding
        return embedding
    
    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        if len(embedding1) != len(embedding2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        magnitude1 = sum(a * a for a in embedding1) ** 0.5
        magnitude2 = sum(b * b for b in embedding2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    async def store_knowledge(self, content: str, content_type: str = "text", 
                            source: str = None, context: str = None, 
                            importance: int = 5, tags: List[str] = None,
                            metadata: Dict[str, Any] = None) -> str:
        """Store knowledge entry in the database."""
        try:
            entry_id = str(uuid.uuid4())
            
            # Generate embedding
            embedding = self._generate_text_embedding(content)
            
            # Prepare data
            tags_str = json.dumps(tags or [])
            metadata_str = json.dumps(metadata or {})
            embedding_str = json.dumps(embedding)
            
            # Store in database
            self.conn.execute("""
                INSERT INTO knowledge_entries 
                (id, content, content_type, source, context, metadata, embeddings, 
                 importance, tags, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """, (entry_id, content, content_type, source, context, metadata_str, 
                  embedding_str, importance, tags_str))
            
            self.conn.commit()
            
            logging.info(f"📝 Stored knowledge entry: {entry_id}")
            return entry_id
            
        except Exception as e:
            logging.error(f"Knowledge storage error: {e}")
            return None
    
    async def search_knowledge(self, query: str, limit: int = 10, 
                             similarity_threshold: float = None) -> List[Dict[str, Any]]:
        """Search knowledge base using semantic similarity."""
        try:
            if similarity_threshold is None:
                similarity_threshold = self.similarity_threshold
            
            # Generate query embedding
            query_embedding = self._generate_text_embedding(query)
            
            # Get all entries
            cursor = self.conn.execute("""
                SELECT id, content, content_type, source, context, metadata, 
                       embeddings, importance, access_count, created_at, tags
                FROM knowledge_entries 
                WHERE expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP
                ORDER BY importance DESC, created_at DESC
                LIMIT 1000
            """)
            
            entries = cursor.fetchall()
            
            # Calculate similarities
            results = []
            for entry in entries:
                try:
                    stored_embedding = json.loads(entry['embeddings'])
                    similarity = self._calculate_similarity(query_embedding, stored_embedding)
                    
                    if similarity >= similarity_threshold:
                        # Update access count
                        self.conn.execute(
                            "UPDATE knowledge_entries SET access_count = access_count + 1 WHERE id = ?",
                            (entry['id'],)
                        )
                        
                        results.append({
                            'id': entry['id'],
                            'content': entry['content'],
                            'content_type': entry['content_type'],
                            'source': entry['source'],
                            'context': entry['context'],
                            'metadata': json.loads(entry['metadata'] or '{}'),
                            'importance': entry['importance'],
                            'similarity': similarity,
                            'access_count': entry['access_count'],
                            'created_at': entry['created_at'],
                            'tags': json.loads(entry['tags'] or '[]')
                        })
                except Exception as e:
                    logging.warning(f"Error processing entry {entry['id']}: {e}")
                    continue
            
            # Sort by similarity and limit
            results.sort(key=lambda x: (x['similarity'], x['importance']), reverse=True)
            results = results[:limit]
            
            self.conn.commit()
            
            logging.info(f"🔍 Found {len(results)} relevant knowledge entries")
            return results
            
        except Exception as e:
            logging.error(f"Knowledge search error: {e}")
            return []
    
    async def store_conversation(self, user_message: str, ai_response: str,
                               model_used: str, session_id: str = None,
                               context_used: str = None, metadata: Dict[str, Any] = None) -> str:
        """Store conversation in the database."""
        try:
            conv_id = str(uuid.uuid4())
            if session_id is None:
                session_id = f"session_{int(time.time())}"
            
            metadata_str = json.dumps(metadata or {})
            
            self.conn.execute("""
                INSERT INTO conversation_history 
                (id, session_id, user_message, ai_response, model_used, 
                 context_used, metadata, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (conv_id, session_id, user_message, ai_response, 
                  model_used, context_used, metadata_str))
            
            self.conn.commit()
            
            # Also store as knowledge for future reference
            combined_content = f"User: {user_message}\nAssistant: {ai_response}"
            await self.store_knowledge(
                content=combined_content,
                content_type="conversation",
                source=f"session_{session_id}",
                context=f"Model: {model_used}",
                importance=3,
                tags=["conversation", model_used],
                metadata={
                    "session_id": session_id,
                    "model": model_used,
                    "conversation_id": conv_id
                }
            )
            
            logging.info(f"💬 Stored conversation: {conv_id}")
            return conv_id
            
        except Exception as e:
            logging.error(f"Conversation storage error: {e}")
            return None
    
    async def get_conversation_context(self, session_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent conversation context for a session."""
        try:
            cursor = self.conn.execute("""
                SELECT id, user_message, ai_response, model_used, context_used, 
                       timestamp, metadata
                FROM conversation_history 
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (session_id, limit))
            
            conversations = []
            for row in cursor.fetchall():
                conversations.append({
                    'id': row['id'],
                    'user_message': row['user_message'],
                    'ai_response': row['ai_response'],
                    'model_used': row['model_used'],
                    'context_used': row['context_used'],
                    'timestamp': row['timestamp'],
                    'metadata': json.loads(row['metadata'] or '{}')
                })
            
            return list(reversed(conversations))  # Return in chronological order
            
        except Exception as e:
            logging.error(f"Context retrieval error: {e}")
            return []
    
    async def create_memory_link(self, from_entry_id: str, to_entry_id: str,
                               relation_type: str = "related", strength: float = 1.0) -> str:
        """Create a link between two knowledge entries."""
        try:
            link_id = str(uuid.uuid4())
            
            self.conn.execute("""
                INSERT INTO memory_links (id, from_entry_id, to_entry_id, relation_type, strength)
                VALUES (?, ?, ?, ?, ?)
            """, (link_id, from_entry_id, to_entry_id, relation_type, strength))
            
            self.conn.commit()
            
            logging.info(f"🔗 Created memory link: {from_entry_id} -> {to_entry_id}")
            return link_id
            
        except Exception as e:
            logging.error(f"Memory link creation error: {e}")
            return None
    
    async def get_related_knowledge(self, entry_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get knowledge entries related to a specific entry."""
        try:
            cursor = self.conn.execute("""
                SELECT ke.id, ke.content, ke.content_type, ke.importance, 
                       ml.relation_type, ml.strength
                FROM memory_links ml
                JOIN knowledge_entries ke ON (ml.to_entry_id = ke.id)
                WHERE ml.from_entry_id = ?
                ORDER BY ml.strength DESC, ke.importance DESC
                LIMIT ?
            """, (entry_id, limit))
            
            related = []
            for row in cursor.fetchall():
                related.append({
                    'id': row['id'],
                    'content': row['content'],
                    'content_type': row['content_type'],
                    'importance': row['importance'],
                    'relation_type': row['relation_type'],
                    'strength': row['strength']
                })
            
            return related
            
        except Exception as e:
            logging.error(f"Related knowledge retrieval error: {e}")
            return []
    
    async def cleanup_expired_entries(self):
        """Clean up expired and low-importance entries."""
        try:
            # Remove expired entries
            self.conn.execute("DELETE FROM knowledge_entries WHERE expires_at < CURRENT_TIMESTAMP")
            
            # Remove old low-importance entries if over limit
            cursor = self.conn.execute("SELECT COUNT(*) as count FROM knowledge_entries")
            count = cursor.fetchone()['count']
            
            if count > self.max_memory_entries:
                entries_to_remove = count - self.max_memory_entries
                self.conn.execute("""
                    DELETE FROM knowledge_entries 
                    WHERE id IN (
                        SELECT id FROM knowledge_entries 
                        ORDER BY importance ASC, access_count ASC, created_at ASC
                        LIMIT ?
                    )
                """, (entries_to_remove,))
            
            # Clean up orphaned links
            self.conn.execute("""
                DELETE FROM memory_links 
                WHERE from_entry_id NOT IN (SELECT id FROM knowledge_entries)
                   OR to_entry_id NOT IN (SELECT id FROM knowledge_entries)
            """)
            
            self.conn.commit()
            logging.info("🧹 Knowledge base cleanup completed")
            
        except Exception as e:
            logging.error(f"Cleanup error: {e}")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        try:
            stats = {}
            
            # Knowledge entries count
            cursor = self.conn.execute("SELECT COUNT(*) as count FROM knowledge_entries")
            stats['total_knowledge_entries'] = cursor.fetchone()['count']
            
            # Conversations count
            cursor = self.conn.execute("SELECT COUNT(*) as count FROM conversation_history")
            stats['total_conversations'] = cursor.fetchone()['count']
            
            # Memory links count
            cursor = self.conn.execute("SELECT COUNT(*) as count FROM memory_links")
            stats['total_memory_links'] = cursor.fetchone()['count']
            
            # Top content types
            cursor = self.conn.execute("""
                SELECT content_type, COUNT(*) as count 
                FROM knowledge_entries 
                GROUP BY content_type 
                ORDER BY count DESC
            """)
            stats['content_types'] = {row['content_type']: row['count'] for row in cursor.fetchall()}
            
            # Recent activity
            cursor = self.conn.execute("""
                SELECT COUNT(*) as count 
                FROM knowledge_entries 
                WHERE created_at > datetime('now', '-24 hours')
            """)
            stats['entries_last_24h'] = cursor.fetchone()['count']
            
            # Database size
            stats['database_size_mb'] = self.db_path.stat().st_size / (1024 * 1024)
            
            return stats
            
        except Exception as e:
            logging.error(f"Statistics error: {e}")
            return {}

class UltraRecallAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for Ultra Recall REST API."""
    
    def __init__(self, *args, knowledge_base=None, **kwargs):
        self.knowledge_base = knowledge_base
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests."""
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        params = parse_qs(parsed_url.query)
        
        if path == '/api/knowledge/search':
            self._handle_search_knowledge(params)
        elif path == '/api/conversation/context':
            self._handle_get_context(params)
        elif path == '/api/knowledge/related':
            self._handle_get_related(params)
        elif path == '/api/stats':
            self._handle_get_stats()
        elif path == '/api/health':
            self._handle_health_check()
        else:
            self._send_404()
    
    def do_POST(self):
        """Handle POST requests."""
        path = self.path
        
        if path == '/api/knowledge/store':
            self._handle_store_knowledge()
        elif path == '/api/conversation/store':
            self._handle_store_conversation()
        elif path == '/api/memory/link':
            self._handle_create_link()
        elif path == '/api/cleanup':
            self._handle_cleanup()
        else:
            self._send_404()
    
    def _get_post_data(self):
        """Get POST data as JSON."""
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        return json.loads(post_data.decode())
    
    def _send_json_response(self, data, status_code=200):
        """Send JSON response."""
        response = json.dumps(data).encode()
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(response))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(response)
    
    def _send_404(self):
        """Send 404 response."""
        self.send_response(404)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"error": "Not found"}).encode())
    
    def _handle_search_knowledge(self, params):
        """Handle knowledge search request."""
        try:
            query = params.get('q', [''])[0]
            limit = int(params.get('limit', [10])[0])
            threshold = float(params.get('threshold', [0.3])[0])
            
            if not query:
                self._send_json_response({"error": "Query parameter 'q' is required"}, 400)
                return
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(
                    self.knowledge_base.search_knowledge(query, limit, threshold)
                )
                self._send_json_response({
                    "success": True,
                    "query": query,
                    "results": results,
                    "count": len(results)
                })
            finally:
                loop.close()
                
        except Exception as e:
            self._send_json_response({"error": str(e)}, 500)
    
    def _handle_store_knowledge(self):
        """Handle knowledge storage request."""
        try:
            data = self._get_post_data()
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                entry_id = loop.run_until_complete(
                    self.knowledge_base.store_knowledge(**data)
                )
                
                if entry_id:
                    self._send_json_response({
                        "success": True,
                        "entry_id": entry_id,
                        "message": "Knowledge stored successfully"
                    })
                else:
                    self._send_json_response({"error": "Failed to store knowledge"}, 500)
            finally:
                loop.close()
                
        except Exception as e:
            self._send_json_response({"error": str(e)}, 400)
    
    def _handle_store_conversation(self):
        """Handle conversation storage request."""
        try:
            data = self._get_post_data()
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                conv_id = loop.run_until_complete(
                    self.knowledge_base.store_conversation(**data)
                )
                
                if conv_id:
                    self._send_json_response({
                        "success": True,
                        "conversation_id": conv_id,
                        "message": "Conversation stored successfully"
                    })
                else:
                    self._send_json_response({"error": "Failed to store conversation"}, 500)
            finally:
                loop.close()
                
        except Exception as e:
            self._send_json_response({"error": str(e)}, 400)
    
    def _handle_get_context(self, params):
        """Handle conversation context request."""
        try:
            session_id = params.get('session_id', [''])[0]
            limit = int(params.get('limit', [5])[0])
            
            if not session_id:
                self._send_json_response({"error": "session_id parameter is required"}, 400)
                return
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                context = loop.run_until_complete(
                    self.knowledge_base.get_conversation_context(session_id, limit)
                )
                self._send_json_response({
                    "success": True,
                    "session_id": session_id,
                    "context": context,
                    "count": len(context)
                })
            finally:
                loop.close()
                
        except Exception as e:
            self._send_json_response({"error": str(e)}, 500)
    
    def _handle_get_related(self, params):
        """Handle related knowledge request."""
        try:
            entry_id = params.get('entry_id', [''])[0]
            limit = int(params.get('limit', [5])[0])
            
            if not entry_id:
                self._send_json_response({"error": "entry_id parameter is required"}, 400)
                return
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                related = loop.run_until_complete(
                    self.knowledge_base.get_related_knowledge(entry_id, limit)
                )
                self._send_json_response({
                    "success": True,
                    "entry_id": entry_id,
                    "related": related,
                    "count": len(related)
                })
            finally:
                loop.close()
                
        except Exception as e:
            self._send_json_response({"error": str(e)}, 500)
    
    def _handle_get_stats(self):
        """Handle statistics request."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                stats = loop.run_until_complete(self.knowledge_base.get_statistics())
                self._send_json_response({
                    "success": True,
                    "statistics": stats
                })
            finally:
                loop.close()
                
        except Exception as e:
            self._send_json_response({"error": str(e)}, 500)
    
    def _handle_health_check(self):
        """Handle health check request."""
        self._send_json_response({
            "success": True,
            "service": "Ultra Recall API",
            "status": "healthy",
            "timestamp": datetime.now().isoformat()
        })
    
    def _handle_create_link(self):
        """Handle memory link creation request."""
        try:
            data = self._get_post_data()
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                link_id = loop.run_until_complete(
                    self.knowledge_base.create_memory_link(**data)
                )
                
                if link_id:
                    self._send_json_response({
                        "success": True,
                        "link_id": link_id,
                        "message": "Memory link created successfully"
                    })
                else:
                    self._send_json_response({"error": "Failed to create memory link"}, 500)
            finally:
                loop.close()
                
        except Exception as e:
            self._send_json_response({"error": str(e)}, 400)
    
    def _handle_cleanup(self):
        """Handle cleanup request."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.knowledge_base.cleanup_expired_entries())
                self._send_json_response({
                    "success": True,
                    "message": "Cleanup completed successfully"
                })
            finally:
                loop.close()
                
        except Exception as e:
            self._send_json_response({"error": str(e)}, 500)
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass

class UltraRecallServer:
    """Ultra Recall REST API server."""
    
    def __init__(self, port=5555, host='127.0.0.1'):
        self.port = port
        self.host = host
        self.knowledge_base = UltraRecallKnowledgeBase()
        self.server = None
        self.server_thread = None
    
    def start_server(self):
        """Start the Ultra Recall API server."""
        def handler(*args, **kwargs):
            return UltraRecallAPIHandler(*args, knowledge_base=self.knowledge_base, **kwargs)
        
        self.server = HTTPServer((self.host, self.port), handler)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        
        print(f"🚀 Ultra Recall API Server started on http://{self.host}:{self.port}")
        print("📚 LocalRecall-inspired Knowledge Base & Memory Infrastructure")
        print("=" * 60)
        print("🔗 API Endpoints:")
        print(f"   GET  /api/health - Health check")
        print(f"   GET  /api/stats - Knowledge base statistics")
        print(f"   GET  /api/knowledge/search?q=query - Search knowledge")
        print(f"   POST /api/knowledge/store - Store knowledge entry")
        print(f"   GET  /api/conversation/context?session_id=xxx - Get conversation context")
        print(f"   POST /api/conversation/store - Store conversation")
        print(f"   POST /api/memory/link - Create memory links")
        print(f"   GET  /api/knowledge/related?entry_id=xxx - Get related knowledge")
        print(f"   POST /api/cleanup - Cleanup expired entries")
        print("=" * 60)
        
        return True
    
    def stop_server(self):
        """Stop the Ultra Recall API server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=2)
        
        print("🛑 Ultra Recall API Server stopped")

# Global Ultra Recall instance
ultra_recall_server = UltraRecallServer()

# Convenience functions for integration with Ultra AI
async def store_knowledge_entry(content: str, **kwargs) -> str:
    """Store knowledge entry using Ultra Recall."""
    return await ultra_recall_server.knowledge_base.store_knowledge(content, **kwargs)

async def search_knowledge_base(query: str, **kwargs) -> List[Dict[str, Any]]:
    """Search knowledge base using Ultra Recall."""
    return await ultra_recall_server.knowledge_base.search_knowledge(query, **kwargs)

async def store_ai_conversation(user_msg: str, ai_response: str, model: str, **kwargs) -> str:
    """Store AI conversation using Ultra Recall."""
    return await ultra_recall_server.knowledge_base.store_conversation(
        user_msg, ai_response, model, **kwargs
    )

async def get_conversation_memory(session_id: str, **kwargs) -> List[Dict[str, Any]]:
    """Get conversation memory using Ultra Recall."""
    return await ultra_recall_server.knowledge_base.get_conversation_context(session_id, **kwargs)

if __name__ == "__main__":
    # Test Ultra Recall system
    async def test_ultra_recall():
        print("🗄️ Ultra Recall Knowledge Base Test")
        print("=" * 50)
        
        kb = UltraRecallKnowledgeBase()
        
        # Test knowledge storage
        print("\n📝 Testing knowledge storage...")
        
        knowledge_entries = [
            ("Python is a programming language", "fact", "programming"),
            ("The user prefers dark mode interfaces", "preference", "ui"),
            ("Machine learning models can be trained on Android", "technical", "ai"),
            ("SQLite is embedded in this knowledge base", "system", "database")
        ]
        
        entry_ids = []
        for content, content_type, context in knowledge_entries:
            entry_id = await kb.store_knowledge(
                content=content,
                content_type=content_type,
                context=context,
                importance=5,
                tags=[content_type, "test"]
            )
            entry_ids.append(entry_id)
            print(f"  Stored: {entry_id[:8]}... - {content[:30]}...")
        
        # Test knowledge search
        print("\n🔍 Testing knowledge search...")
        
        queries = ["programming", "user interface", "machine learning", "database"]
        for query in queries:
            results = await kb.search_knowledge(query, limit=3)
            print(f"  Query: '{query}' -> {len(results)} results")
            for result in results[:2]:
                print(f"    - {result['content'][:40]}... (similarity: {result['similarity']:.3f})")
        
        # Test conversation storage
        print("\n💬 Testing conversation storage...")
        
        conv_id = await kb.store_conversation(
            user_message="Hello, what can you help me with?",
            ai_response="I'm Ultra AI! I can help with programming, analysis, and creative tasks.",
            model_used="qwen2",
            session_id="test_session_1"
        )
        print(f"  Stored conversation: {conv_id}")
        
        # Test conversation context
        print("\n📚 Testing conversation context...")
        context = await kb.get_conversation_context("test_session_1", limit=5)
        print(f"  Retrieved {len(context)} conversation entries")
        
        # Test memory links
        print("\n🔗 Testing memory links...")
        if len(entry_ids) >= 2:
            link_id = await kb.create_memory_link(
                entry_ids[0], entry_ids[1], 
                relation_type="related_to", 
                strength=0.8
            )
            print(f"  Created memory link: {link_id}")
        
        # Test statistics
        print("\n📊 Testing statistics...")
        stats = await kb.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Test cleanup
        print("\n🧹 Testing cleanup...")
        await kb.cleanup_expired_entries()
        print("  Cleanup completed")
        
    # Run test
    asyncio.run(test_ultra_recall())
    
    # Start API server for manual testing
    print("\n🚀 Starting Ultra Recall API Server...")
    server = UltraRecallServer(port=5555)
    server.start_server()
    
    try:
        input("\nPress Enter to stop the server...")
    except KeyboardInterrupt:
        pass
    finally:
        server.stop_server()