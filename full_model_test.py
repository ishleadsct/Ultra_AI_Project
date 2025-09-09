#!/usr/bin/env python3
"""
Full Ultra AI Model Loading and Response Test
"""

import sys
import json
import os
import asyncio
import time

# Add src to path
sys.path.append('src')

async def full_model_test():
    print("🚀 Full Ultra AI Model Loading and Response Test...")
    print("=" * 60)
    
    try:
        from ai.gguf_ai import GGUFAIManager
        
        # Create manager
        ai_manager = GGUFAIManager()
        models = ai_manager.get_available_models()
        
        # Use smallest model for testing
        test_model = "qwen2"  # 940MB model
        print(f"🤖 Testing with: {test_model} ({models[test_model]['description']})")
        print("⏳ Loading model... (this may take 1-2 minutes)")
        
        # Load model with timeout handling
        start_time = time.time()
        load_result = await ai_manager.load_model(test_model)
        load_time = time.time() - start_time
        
        if load_result['success']:
            print(f"✅ Model loaded successfully in {load_time:.1f} seconds")
            print(f"📊 Model: {load_result.get('model', test_model)}")
            print(f"📝 Description: {load_result.get('description', 'N/A')}")
            
            # Test response generation
            print("\n🧪 Testing response generation...")
            test_prompt = "Hello! Please introduce yourself as Ultra AI and tell me what you can do."
            
            response_start = time.time()
            response_result = await ai_manager.generate_response(
                test_prompt,
                max_tokens=150
            )
            response_time = time.time() - response_start
            
            if response_result['success']:
                print(f"✅ Response generated in {response_time:.1f} seconds")
                print(f"📝 Prompt: {test_prompt}")
                print(f"🤖 Response: {response_result['response']}")
                print(f"📊 Tokens: {response_result.get('tokens_used', 'N/A')}")
                
                # Test model status
                status = ai_manager.get_model_status()
                print(f"\n📊 Model Status:")
                print(f"   Current Model: {status.get('current_model', 'N/A')}")
                print(f"   Loaded: {status.get('is_loaded', False)}")
                print(f"   Memory Usage: {status.get('memory_usage_mb', 'N/A')} MB")
                
                print("\n🎉 FULL MODEL LOADING TEST: SUCCESS")
                return True
            else:
                print(f"❌ Response generation failed: {response_result.get('error', 'Unknown error')}")
                return False
                
        else:
            print(f"❌ Model loading failed: {load_result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚨 This test will take 2-3 minutes to complete due to model loading time")
    print("⚡ Starting test...")
    
    success = asyncio.run(full_model_test())
    
    if success:
        print("\n🎉 Ultra AI Model Loading & Response: WORKING")
        sys.exit(0)
    else:
        print("\n❌ Ultra AI Model Loading & Response: FAILED")
        sys.exit(1)