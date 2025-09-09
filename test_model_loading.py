#!/usr/bin/env python3

import sys
import json
import os
import asyncio

# Add src to path
sys.path.append('src')

def test_model_loading():
    print("🚀 Testing Ultra AI Model Loading...")
    print("=" * 50)
    
    # Load models configuration
    try:
        with open('models.json', 'r') as f:
            models = json.load(f)
        print(f"✅ Found {len(models)} models in configuration")
    except Exception as e:
        print(f"❌ Failed to load models.json: {e}")
        return False
    
    # Test model file existence
    available_models = []
    for model in models:
        model_path = model['path']
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"✅ {model['name']}: {size_mb:.1f}MB - {model['display_name']}")
            available_models.append(model)
        else:
            print(f"❌ {model['name']}: Model file not found at {model_path}")
    
    if not available_models:
        print("❌ No models available for loading!")
        return False
    
    # Test loading a small model
    print("\n🧪 Testing Model Loading...")
    smallest_model = min(available_models, key=lambda x: x['size_gb'])
    print(f"Testing with smallest model: {smallest_model['name']} ({smallest_model['size_gb']}GB)")
    
    try:
        # Import GGUF AI
        from ai.gguf_ai import GGUFAIManager
        
        # Test model initialization
        model_path = smallest_model['path']
        print(f"Initializing model from: {model_path}")
        
        # Create AI manager
        ai_manager = GGUFAIManager()
        
        # Load model (async method, only takes model name)
        import asyncio
        result = asyncio.run(ai_manager.load_model(smallest_model['name']))
        success = result.get('success', False)
        
        if success:
            print("✅ Model loaded successfully!")
            
            # Test generation
            print("🧪 Testing response generation...")
            response_result = asyncio.run(ai_manager.generate_response(
                "Hello, I'm testing Ultra AI. Please introduce yourself briefly.",
                max_tokens=100
            ))
            response = response_result.get('response', '') if response_result.get('success') else None
            
            if response:
                print(f"✅ Response generated: {response[:100]}...")
                print("🎉 MODEL LOADING TEST PASSED!")
                return True
            else:
                print("❌ Failed to generate response")
                return False
        else:
            print("❌ Failed to load model")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Trying alternative import...")
        try:
            import src.ai.gguf_ai as gguf_ai
            print("✅ Alternative import successful")
            return True
        except Exception as e2:
            print(f"❌ Alternative import failed: {e2}")
            return False
    except Exception as e:
        print(f"❌ Error during model loading test: {e}")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("\n🎉 Ultra AI Model Loading: WORKING")
        sys.exit(0)
    else:
        print("\n❌ Ultra AI Model Loading: FAILED")
        sys.exit(1)