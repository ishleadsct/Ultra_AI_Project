#!/usr/bin/env python3
"""Ultra AI Test - Step by step debugging"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_step_by_step():
    print("=== Step by Step Test ===\n")
    
    # Step 1: Test utils
    print("1. Testing utils...")
    try:
        from utils.logger import get_logger
        logger = get_logger("test")
        print("   ✓ Logger imported")
        
        from utils.helpers import sanitize_string, current_timestamp
        test_str = sanitize_string("Test String!")
        print(f"   ✓ Helpers working: '{test_str}'")
        
    except Exception as e:
        print(f"   ✗ Utils failed: {e}")
        return False
    
    # Step 2: Test base tool
    print("2. Testing base tool...")
    try:
        from tools.base_tool import BaseTool, ToolCategory, ParameterType
        print("   ✓ Base tool imported")
        
    except Exception as e:
        print(f"   ✗ Base tool failed: {e}")
        return False
    
    # Step 3: Test simple tools (syntax check)
    print("3. Testing simple tools syntax...")
    try:
        import ast
        with open("src/tools/simple_tools.py", "r") as f:
            code = f.read()
        ast.parse(code)
        print("   ✓ Syntax is valid")
        
    except SyntaxError as e:
        print(f"   ✗ Syntax error: {e}")
        return False
    except Exception as e:
        print(f"   ✗ File error: {e}")
        return False
    
    # Step 4: Import simple tools
    print("4. Testing simple tools import...")
    try:
        from tools.simple_tools import SimpleCodeExecutor
        executor = SimpleCodeExecutor()
        print(f"   ✓ Created executor: {executor.name}")
        
    except Exception as e:
        print(f"   ✗ Simple tools import failed: {e}")
        return False
    
    print("\n✓ All steps passed! Basic structure is working.")
    return True

if __name__ == "__main__":
    success = test_step_by_step()
    if not success:
        print("\n⚠️  Some issues found - check the errors above")
    else:
        print("\n🎉 Ready for basic functionality testing!")
