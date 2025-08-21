#!/usr/bin/env python3
"""Test Ultra AI with zero external dependencies"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_basic_imports():
    """Test that basic modules import"""
    try:
        from utils.logger import get_logger
        from utils.helpers import sanitize_string
        from tools.base_tool import BaseTool
        print("✓ Basic imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

async def test_simple_tools():
    """Test simple tools"""
    try:
        from tools.simple_tools import SimpleCodeExecutor, SimpleMessageFormatter
        
        # Test code executor
        executor = SimpleCodeExecutor()
        result = await executor.execute(code="print('Hello World!')\nx = 5 + 3\nprint(f'5 + 3 = {x}')")
        
        print("SimpleCodeExecutor:")
        print(f"  Success: {result.success}")
        if result.success:
            print(f"  Output: {repr(result.data['output'])}")
        
        # Test message formatter
        formatter = SimpleMessageFormatter()
        result2 = await formatter.execute(text="hello world", format_type="title")
        
        print("SimpleMessageFormatter:")
        print(f"  Success: {result2.success}")
        if result2.success:
            print(f"  Formatted: {result2.data['formatted_text']}")
        
        return result.success and result2.success
        
    except Exception as e:
        print(f"✗ Simple tools test failed: {e}")
        return False

def test_file_structure():
    """Test that all required files exist"""
    required_files = [
        "src/__init__.py",
        "src/utils/__init__.py", 
        "src/utils/logger.py",
        "src/utils/helpers.py",
        "src/tools/__init__.py",
        "src/tools/base_tool.py"
    ]
    
    missing = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing.append(file_path)
    
    if missing:
        print(f"✗ Missing files: {missing}")
        return False
    else:
        print("✓ All required files present")
        return True

async def main():
    print("=== Ultra AI Simple Test (No External Dependencies) ===\n")
    
    tests = [
        ("File Structure", test_file_structure),
        ("Basic Imports", test_basic_imports),
        ("Simple Tools", test_simple_tools)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"Running {name} test...")
        if asyncio.iscoroutinefunction(test_func):
            success = await test_func()
        else:
            success = test_func()
        results.append((name, success))
        print(f"{'✓' if success else '✗'} {name}: {'PASS' if success else 'FAIL'}\n")
    
    # Summary
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"=== Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("\nSuccess! Your Ultra AI project is working with basic functionality.")
        print("\nNext steps:")
        print("1. Try individual tools manually")
        print("2. Add external dependencies only as needed")
        print("3. Test specific features you want to use")
    else:
        print("\nSome tests failed. Check the errors above.")

if __name__ == "__main__":
    asyncio.run(main())
