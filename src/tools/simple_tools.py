"""
Simple tools that work without external dependencies
"""

import asyncio
import json
import subprocess
import tempfile
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import sys

# Add the parent directory to sys.path to fix import issues
sys.path.append(str(Path(__file__).parent.parent))

from tools.base_tool import BaseTool, ToolCategory, ToolParameter, ParameterType, ToolResult
from utils.logger import get_logger

logger = get_logger(__name__)

class SimpleCodeExecutor(BaseTool):
    """Execute Python code without external dependencies."""
    
    @property
    def name(self) -> str:
        return "simple_code_executor"
    
    @property
    def description(self) -> str:
        return "Execute simple Python code safely"
    
    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="code",
                type=ParameterType.STRING,
                description="Python code to execute",
                required=True
            )
        ]
    
    async def execute(self, **kwargs) -> ToolResult:
        try:
            code = kwargs.get("code")
            
            # Simple safe execution
            import io
            import contextlib
            
            output_buffer = io.StringIO()
            
            with contextlib.redirect_stdout(output_buffer):
                # Restricted execution
                allowed_globals = {
                    "__builtins__": {
                        "print": print,
                        "len": len,
                        "str": str,
                        "int": int,
                        "float": float,
                        "list": list,
                        "dict": dict,
                        "range": range,
                        "sum": sum,
                        "max": max,
                        "min": min
                    }
                }
                
                exec(code, allowed_globals)
            
            output = output_buffer.getvalue()
            
            return ToolResult(
                success=True,
                data={"output": output, "code": code}
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e)
            )

class SimpleMessageFormatter(BaseTool):
    """Format messages without external dependencies."""
    
    @property
    def name(self) -> str:
        return "simple_message_formatter"
    
    @property
    def description(self) -> str:
        return "Format text messages for different platforms"
    
    @property
    def category(self) -> ToolCategory:
        return ToolCategory.COMMUNICATION
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="text",
                type=ParameterType.STRING,
                description="Text to format",
                required=True
            ),
            ToolParameter(
                name="format_type",
                type=ParameterType.STRING,
                description="Format type",
                required=False,
                default="plain"
            )
        ]
    
    async def execute(self, **kwargs) -> ToolResult:
        try:
            # Support both 'text' and 'message' parameters for flexibility
            text = kwargs.get("text") or kwargs.get("message")
            if not text:
                return ToolResult(
                    success=False,
                    error="No text or message provided"
                )
                
            format_type = kwargs.get("format_type", "plain")
            prefix = kwargs.get("prefix", "")
            suffix = kwargs.get("suffix", "")
            
            # Apply formatting
            if format_type == "uppercase":
                formatted = text.upper()
            elif format_type == "lowercase":
                formatted = text.lower()
            elif format_type == "title":
                formatted = text.title()
            elif format_type == "bold":
                formatted = f"**{text}**"
            elif format_type == "italic":
                formatted = f"*{text}*"
            elif format_type == "code":
                formatted = f"`{text}`"
            else:
                formatted = text
            
            # Add prefix and suffix
            if prefix or suffix:
                formatted = f"{prefix}{formatted}{suffix}"
            
            return ToolResult(
                success=True,
                data={
                    "formatted_text": formatted, 
                    "original": text,
                    "format_type": format_type,
                    "prefix": prefix,
                    "suffix": suffix
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e)
            )
