#!/usr/bin/env python3
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

async def main():
    print("=== Ultra AI Final Test ===")
    
    try:
        # Test imports
        from utils.logger import get_logger
        from utils.helpers import sanitize_string, current_timestamp
        from tools.base_tool import BaseTool, ToolCategory, ParameterType, ToolResult, ToolParameter
        
        print("All imports successful")
        
        # Test functionality
        logger = get_logger("test")
        logger.info("Logger working")
        
        clean_text = sanitize_string("Test String!")
        timestamp = current_timestamp()
        
        print(f"Helpers working: '{clean_text}', {timestamp}")
        
        # Create a simple test tool
        class TestTool(BaseTool):
            @property
            def name(self):
                return "test_tool"
            
            @property
            def description(self):
                return "A simple test tool"
            
            @property
            def category(self):
                return ToolCategory.CODE
            
            @property
            def parameters(self):
                return [
                    ToolParameter(
                        name="message",
                        type=ParameterType.STRING,
                        description="Test message"
                    )
                ]
            
            async def execute(self, **kwargs):
                message = kwargs.get("message", "Hello World!")
                return ToolResult(
                    success=True,
                    data={"message": message, "processed": True}
                )
        
        # Test the tool
        tool = TestTool()
        result = await tool.execute(message="Ultra AI is working!")
        
        print(f"Tool test: {result.success}")
        print(f"Tool result: {result.data}")
        
        print("\nSUCCESS: Ultra AI basic framework is working!")
        print("\nNext steps:")
        print("1. Create custom tools by extending BaseTool")
        print("2. Add more complex functionality as needed")
        print("3. Install additional dependencies for specific features")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(main())
