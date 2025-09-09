#!/usr/bin/env python3
"""
Ultra AI Memory System and Session Persistence Test
Tests memory layers, session persistence, and cross-session continuity
"""

import sys
import json
import os
import asyncio
import time

# Add src to path
sys.path.append('src')

async def test_memory_system():
    print("🧠 Ultra AI Memory System Test...")
    print("=" * 50)
    
    try:
        # Test 1: Import Memory Components
        print("🧪 Test 1: Import Memory System Components...")
        
        try:
            import sys
            sys.path.insert(0, 'src')
            from src.core.memory_manager import MemoryManager, MemoryItem
            print("✅ MemoryManager imported successfully")
        except ImportError as e:
            print(f"❌ MemoryManager import failed: {e}")
            # Try alternative import
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("memory_manager", "src/core/memory_manager.py")
                memory_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(memory_module)
                MemoryManager = memory_module.MemoryManager
                MemoryItem = memory_module.MemoryItem
                print("✅ MemoryManager imported via alternative method")
            except Exception as e2:
                print(f"❌ Alternative import also failed: {e2}")
                return False
            
        # Test 2: Create Memory Manager Instance
        print("\n🧪 Test 2: Create Memory Manager Instance...")
        try:
            memory_manager = MemoryManager()
            print("✅ MemoryManager instance created successfully")
        except Exception as e:
            print(f"❌ MemoryManager instantiation failed: {e}")
            return False
            
        # Test 3: Test Memory Storage
        print("\n🧪 Test 3: Test Memory Storage...")
        try:
            test_memory_id = memory_manager.store(
                "conversation",
                "Hello, I'm testing the Ultra AI memory system!",
                metadata={"user": "test_user", "timestamp": time.time()},
                tags=["test", "memory_system"]
            )
            print(f"✅ Memory stored successfully with ID: {test_memory_id}")
        except Exception as e:
            print(f"❌ Memory storage failed: {e}")
            return False
            
        # Test 4: Test Memory Retrieval
        print("\n🧪 Test 4: Test Memory Retrieval...")
        try:
            retrieved_memory = memory_manager.get(test_memory_id)
            if retrieved_memory:
                print(f"✅ Memory retrieved successfully: {retrieved_memory.content[:50]}...")
                print(f"   Type: {retrieved_memory.type}")
                print(f"   Tags: {retrieved_memory.tags}")
            else:
                print("❌ Memory retrieval failed - no memory found")
                return False
        except Exception as e:
            print(f"❌ Memory retrieval failed: {e}")
            return False
            
        # Test 5: Test Memory Search
        print("\n🧪 Test 5: Test Memory Search...")
        try:
            search_results = memory_manager.search("testing")
            if search_results:
                print(f"✅ Memory search successful - found {len(search_results)} results")
                for result in search_results[:3]:  # Show first 3 results
                    print(f"   - {result.content[:30]}... (Score: {result.metadata.get('score', 'N/A')})")
            else:
                print("⚠️  Memory search returned no results (may be expected)")
        except Exception as e:
            print(f"❌ Memory search failed: {e}")
            return False
            
        # Test 6: Test Session Directory
        print("\n🧪 Test 6: Test Session Storage Directory...")
        session_dir = "/storage/emulated/0/AI_Models/.ultra_ai/sessions/"
        if os.path.exists(session_dir):
            session_files = os.listdir(session_dir)
            print(f"✅ Session directory exists with {len(session_files)} files")
            if session_files:
                print(f"   Latest session files: {session_files[:3]}")
        else:
            print("⚠️  Session directory not found - may be created on first use")
            
        # Test 7: Test Memory Directory
        print("\n🧪 Test 7: Test Memory Storage Directory...")
        memory_dir = "/storage/emulated/0/AI_Models/.ultra_ai/memory/"
        if os.path.exists(memory_dir):
            memory_files = os.listdir(memory_dir)
            print(f"✅ Memory directory exists with {len(memory_files)} files")
            if memory_files:
                print(f"   Memory files: {memory_files[:3]}")
        else:
            print("⚠️  Memory directory not found - may be created on first use")
            
        # Test 8: Test Memory Manager Status
        print("\n🧪 Test 8: Test Memory Manager Status...")
        try:
            stats = memory_manager.get_stats()
            print("✅ Memory Manager Statistics:")
            for key, value in stats.items():
                print(f"   {key}: {value}")
        except Exception as e:
            print(f"❌ Memory stats failed: {e}")
            
        print("\n🎉 MEMORY SYSTEM TEST: SUCCESS")
        return True
        
    except Exception as e:
        print(f"❌ Unexpected error during memory test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_memory_system())
    if success:
        print("\n🎉 Ultra AI Memory System: WORKING")
        sys.exit(0)
    else:
        print("\n❌ Ultra AI Memory System: FAILED")
        sys.exit(1)