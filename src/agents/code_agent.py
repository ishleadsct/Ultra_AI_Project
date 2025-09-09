"""
Ultra AI Project - Code Agent

Specialized agent for programming and software development tasks including
code generation, review, debugging, testing, and technical documentation.
"""

import asyncio
import re
import ast
import subprocess
import tempfile
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
import json

from .base_agent import BaseAgent, AgentConfig, Task, TaskStatus, AgentCapability
from ..utils.logger import get_logger
from ..utils.helpers import sanitize_string

logger = get_logger(__name__)

class CodeExecutionResult:
    """Result of code execution."""
    
    def __init__(self, success: bool, output: str = "", error: str = "", 
                 execution_time: float = 0.0, exit_code: int = 0):
        self.success = success
        self.output = output
        self.error = error
        self.execution_time = execution_time
        self.exit_code = exit_code

class CodeAnalysis:
    """Code analysis result."""
    
    def __init__(self):
        self.syntax_valid = True
        self.complexity_score = 0
        self.lines_of_code = 0
        self.functions_count = 0
        self.classes_count = 0
        self.issues = []
        self.suggestions = []
        self.dependencies = []

class CodeAgent(BaseAgent):
    """Specialized agent for code-related tasks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        # Set up code agent configuration
        if config is None:
            config = {}
        
        agent_config = AgentConfig(
            name=config.get("name", "code_agent"),
            agent_type="code",
            max_concurrent_tasks=config.get("max_concurrent_tasks", 3),
            timeout=config.get("timeout", 600.0),  # 10 minutes for code tasks
            memory_limit=config.get("memory_limit", 1000),
            capabilities=[
                AgentCapability.CODE_GENERATION.value,
                AgentCapability.CODE_REVIEW.value,
                AgentCapability.DEBUGGING.value,
                AgentCapability.TESTING.value,
                AgentCapability.TEXT_GENERATION.value
            ],
            preferred_models=config.get("preferred_models", ["gpt-4", "claude-3-sonnet"]),
            enable_memory=config.get("enable_memory", True),
            custom_settings=config.get("custom_settings", {})
        )
        
        super().__init__(agent_config, **kwargs)
        
        # Code-specific configuration
        self.supported_languages = {
            "python", "javascript", "typescript", "java", "cpp", "c", 
            "csharp", "go", "rust", "php", "ruby", "swift", "kotlin",
            "html", "css", "sql", "bash", "powershell"
        }
        
        self.execution_enabled = config.get("enable_execution", False)
        self.max_execution_time = config.get("max_execution_time", 30.0)
        self.sandbox_path = Path(config.get("sandbox_path", "runtime/agents/code_sandbox"))
        
        # Code templates and patterns
        self.code_templates = self._load_code_templates()
        self.best_practices = self._load_best_practices()
        
        # Create sandbox directory
        self.sandbox_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("CodeAgent initialized")
    
    async def _agent_initialize(self):
        """Code agent specific initialization."""
        try:
            # Initialize code analysis tools
            await self._initialize_code_tools()
            
            # Load programming knowledge
            await self._load_programming_knowledge()
            
            logger.info("CodeAgent initialization complete")
            
        except Exception as e:
            logger.error(f"CodeAgent initialization failed: {e}")
            raise
    
    async def _initialize_code_tools(self):
        """Initialize code analysis and execution tools."""
        try:
            # Check for available code analysis tools
            self.available_tools = {
                "python": await self._check_python_available(),
                "node": await self._check_node_available(),
                "pylint": await self._check_pylint_available(),
                "black": await self._check_black_available()
            }
            
            logger.debug(f"Available code tools: {self.available_tools}")
            
        except Exception as e:
            logger.error(f"Failed to initialize code tools: {e}")
    
    async def _check_python_available(self) -> bool:
        """Check if Python is available."""
        try:
            result = subprocess.run(["python", "--version"], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except:
            return False
    
    async def _check_node_available(self) -> bool:
        """Check if Node.js is available."""
        try:
            result = subprocess.run(["node", "--version"], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except:
            return False
    
    async def _check_pylint_available(self) -> bool:
        """Check if pylint is available."""
        try:
            result = subprocess.run(["pylint", "--version"], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except:
            return False
    
    async def _check_black_available(self) -> bool:
        """Check if black formatter is available."""
        try:
            result = subprocess.run(["black", "--version"], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except:
            return False
    
    async def _load_programming_knowledge(self):
        """Load programming knowledge into memory."""
        try:
            # Store common programming patterns and best practices
            await self.store_memory(
                content=self.best_practices,
                memory_type="best_practices",
                importance=3.0,
                tags=["programming", "best_practices"]
            )
            
            # Store code templates
            await self.store_memory(
                content=self.code_templates,
                memory_type="templates",
                importance=2.5,
                tags=["programming", "templates"]
            )
            
        except Exception as e:
            logger.error(f"Failed to load programming knowledge: {e}")
    
    def _load_code_templates(self) -> Dict[str, str]:
        """Load code templates for common patterns."""
        return {
            "python_function": '''def {function_name}({parameters}):
    """
    {docstring}
    """
    {body}
    return {return_value}''',
            
            "python_class": '''class {class_name}:
    """
    {docstring}
    """
    
    def __init__(self{init_parameters}):
        {init_body}
    
    {methods}''',
            
            "javascript_function": '''function {function_name}({parameters}) {{
    /**
     * {docstring}
     */
    {body}
    return {return_value};
}}''',
            
            "test_function": '''def test_{test_name}():
    """Test {description}."""
    # Arrange
    {arrange}
    
    # Act
    {act}
    
    # Assert
    {assert_statements}''',
        }
    
    def _load_best_practices(self) -> Dict[str, List[str]]:
        """Load programming best practices."""
        return {
            "python": [
                "Follow PEP 8 style guidelines",
                "Use meaningful variable and function names",
                "Write docstrings for functions and classes",
                "Handle exceptions appropriately",
                "Use type hints where applicable",
                "Keep functions small and focused",
                "Use list comprehensions when appropriate",
                "Avoid global variables",
                "Use context managers for resource management"
            ],
            "javascript": [
                "Use const and let instead of var",
                "Use meaningful variable and function names",
                "Handle errors with try-catch blocks",
                "Use async/await for asynchronous operations",
                "Validate input parameters",
                "Use strict mode",
                "Avoid global scope pollution",
                "Use proper error handling",
                "Document complex logic with comments"
            ],
            "general": [
                "Write clean, readable code",
                "Use version control effectively",
                "Write comprehensive tests",
                "Document your code",
                "Follow the DRY principle",
                "Use meaningful commit messages",
                "Review code before deployment",
                "Optimize for readability first, performance second"
            ]
        }
    
    async def _execute_task(self, task: Task) -> Any:
        """Execute a code-related task."""
        task_type = task.task_type
        data = task.data
        
        try:
            if task_type == "generate_code":
                return await self._generate_code(data)
            elif task_type == "review_code":
                return await self._review_code(data)
            elif task_type == "debug_code":
                return await self._debug_code(data)
            elif task_type == "test_code":
                return await self._test_code(data)
            elif task_type == "execute_code":
                return await self._execute_code(data)
            elif task_type == "analyze_code":
                return await self._analyze_code(data)
            elif task_type == "format_code":
                return await self._format_code(data)
            elif task_type == "explain_code":
                return await self._explain_code(data)
            elif task_type == "optimize_code":
                return await self._optimize_code(data)
            elif task_type == "generate_tests":
                return await self._generate_tests(data)
            elif task_type == "documentation":
                return await self._generate_documentation(data)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
                
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            raise
    
    async def _generate_code(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code based on requirements."""
        try:
            requirements = data.get("requirements", "")
            language = data.get("language", "python").lower()
            style = data.get("style", "clean")
            include_tests = data.get("include_tests", False)
            include_docs = data.get("include_docs", True)
            
            if not requirements:
                raise ValueError("Requirements are required for code generation")
            
            if language not in self.supported_languages:
                raise ValueError(f"Unsupported language: {language}")
            
            # Retrieve relevant templates and best practices
            templates = await self.retrieve_memory(memory_type="templates", limit=5)
            best_practices = await self.retrieve_memory(memory_type="best_practices", limit=10)
            
            # Build prompt for code generation
            prompt = self._build_code_generation_prompt(
                requirements, language, style, include_tests, include_docs,
                templates, best_practices
            )
            
            # Generate code using LLM
            if not self.model_manager:
                raise ValueError("Model manager not available")
            
            response = await self.model_manager.generate_completion(
                prompt=prompt,
                temperature=0.1,  # Low temperature for more deterministic code
                max_tokens=2048
            )
            
            if not response.success:
                raise ValueError(f"Code generation failed: {response.error}")
            
            generated_code = response.content
            
            # Extract code blocks from response
            code_blocks = self._extract_code_blocks(generated_code)
            main_code = code_blocks.get("main", generated_code)
            test_code = code_blocks.get("test", "")
            
            # Analyze generated code
            analysis = await self._analyze_code_content(main_code, language)
            
            result = {
                "code": main_code,
                "language": language,
                "analysis": analysis.__dict__,
                "requirements": requirements,
                "style": style
            }
            
            if test_code:
                result["tests"] = test_code
            
            # Store in memory for future reference
            await self.store_memory(
                content=result,
                memory_type="generated_code",
                importance=2.0,
                tags=["code_generation", language, style]
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            raise
    
    def _build_code_generation_prompt(self, requirements: str, language: str, 
                                    style: str, include_tests: bool, include_docs: bool,
                                    templates: List, best_practices: List) -> str:
        """Build prompt for code generation."""
        prompt = f"""Generate {language} code that meets the following requirements:

Requirements:
{requirements}

Style Guidelines:
- Follow {style} coding style
- Use best practices for {language}
"""
        
        if best_practices:
            practices = [bp.content for bp in best_practices if language in str(bp.content)]
            if practices:
                prompt += f"\nBest Practices to Follow:\n"
                for practice in practices[:3]:  # Limit to top 3
                    if isinstance(practice, dict) and language in practice:
                        prompt += f"- {', '.join(practice[language][:5])}\n"
        
        if include_docs:
            prompt += "\n- Include comprehensive documentation and docstrings"
        
        if include_tests:
            prompt += "\n- Include unit tests"
        
        prompt += f"""

Please provide:
1. Clean, well-structured {language} code
2. Proper error handling
3. Clear variable and function names
4. Comments for complex logic
"""
        
        if include_tests:
            prompt += "5. Unit tests in a separate code block marked as 'test'\n"
        
        prompt += "\nFormat your response with code blocks using ```{language} markers."
        
        return prompt
    
    def _extract_code_blocks(self, content: str) -> Dict[str, str]:
        """Extract code blocks from markdown formatted text."""
        import re
        
        # Pattern to match code blocks
        pattern = r'```(\w+)?\n(.*?)\n```'
        matches = re.findall(pattern, content, re.DOTALL)
        
        code_blocks = {}
        main_code = ""
        
        for i, (lang, code) in enumerate(matches):
            code = code.strip()
            if 'test' in code.lower() or 'Test' in code:
                code_blocks['test'] = code
            elif i == 0 or not main_code:  # First block or no main code yet
                main_code = code
                code_blocks['main'] = code
        
        # If no code blocks found, treat entire content as code
        if not code_blocks:
            code_blocks['main'] = content.strip()
        
        return code_blocks
    
    async def _review_code(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Review code for quality, bugs, and improvements."""
        try:
            code = data.get("code", "")
            language = data.get("language", "python").lower()
            focus_areas = data.get("focus_areas", ["quality", "security", "performance"])
            
            if not code:
                raise ValueError("Code is required for review")
            
            # Analyze code structure
            analysis = await self._analyze_code_content(code, language)
            
            # Build review prompt
            prompt = f"""Review the following {language} code for:
{', '.join(focus_areas)}

Code:
```{language}
{code}
```

Please provide:
- Overall code quality assessment (1-10 scale)
- Specific issues found (bugs, security vulnerabilities, performance problems)
- Suggestions for improvement
- Best practices recommendations
- Code complexity assessment

Be thorough but constructive in your feedback."""

            # Generate review using LLM
            if not self.model_manager:
                raise ValueError("Model manager not available")
            
            response = await self.model_manager.generate_completion(
                prompt=prompt,
                temperature=0.2,
                max_tokens=1500
            )
            
            if not response.success:
                raise ValueError(f"Code review failed: {response.error}")
            
            review_text = response.content
            
            # Parse review into structured format
            review_result = self._parse_review_response(review_text)
            
            result = {
                "code": code,
                "language": language,
                "analysis": analysis.__dict__,
                "review": review_result,
                "focus_areas": focus_areas,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store review in memory
            await self.store_memory(
                content=result,
                memory_type="code_review",
                importance=2.5,
                tags=["code_review", language] + focus_areas
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Code review failed: {e}")
            raise

    def _parse_review_response(self, review_text: str) -> Dict[str, Any]:
        """Parse LLM review response into structured format."""
        # This is a simplified parser - could be enhanced with more sophisticated NLP
        lines = review_text.split('\n')
        
        review = {
            "quality_score": 7,  # Default
            "issues": [],
            "suggestions": [],
            "summary": "",
        }
    
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Look for quality score
        if 'quality' in line.lower() and any(char.isdigit() for char in line):
            import re
            scores = re.findall(r'(\d+(?:\.\d+)?)', line)
            if scores:
                try:
                    review["quality_score"] = float(scores[0])
                except:
                    pass
        
        # Identify sections
        if 'issues' in line.lower() or 'problems' in line.lower():
            current_section = 'issues'
        elif 'suggestions' in line.lower() or 'improvements' in line.lower():
            current_section = 'suggestions'
        elif line.startswith('- ') or line.startswith('* '):
            # List item
            item = line[2:].strip()
            if current_section and item:
                review[current_section].append(item)
        elif not current_section and len(line) > 20:
            # Likely part of summary
            review["summary"] += line + " "
    
    review["summary"] = review["summary"].strip()
    return review

async def _debug_code(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """Debug code to find and fix issues."""
    try:
        code = data.get("code", "")
        language = data.get("language", "python").lower()
        error_message = data.get("error_message", "")
        context = data.get("context", "")
        
        if not code:
            raise ValueError("Code is required for debugging")
        
        # Analyze code for syntax errors
        analysis = await self._analyze_code_content(code, language)
        
        # Build debugging prompt
        prompt = f"""Debug the following {language} code:Code:{code}"""if error_message:
            prompt += f"\nError Message:\n{error_message}\n"
        
        if context:
            prompt += f"\nContext:\n{context}\n"
        
        prompt += """Please:

Identify the issue(s) in the codeExplain why the error is occurringProvide a corrected version of the codeSuggest preventive measures for similar issuesFormat the corrected code in a code block."""# Generate debugging response
        if not self.model_manager:
            raise ValueError("Model manager not available")
        
        response = await self.model_manager.generate_completion(
            prompt=prompt,
            temperature=0.1,
            max_tokens=1500
        )
        
        if not response.success:
            raise ValueError(f"Debugging failed: {response.error}")
        
        debug_response = response.content
        
        # Extract corrected code
        code_blocks = self._extract_code_blocks(debug_response)
        corrected_code = code_blocks.get("main", "")
        
        result = {
            "original_code": code,
            "corrected_code": corrected_code,
            "language": language,
            "error_message": error_message,
            "analysis": analysis.__dict__,
            "debug_explanation": debug_response,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store debugging session
        await self.store_memory(
            content=result,
            memory_type="debug_session",
            importance=3.0,
            tags=["debugging", language, "error_fix"]
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Code debugging failed: {e}")
        raise

async def _test_code(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """Test code functionality."""
    try:
        code = data.get("code", "")
        language = data.get("language", "python").lower()
        test_cases = data.get("test_cases", [])
        
        if not code:
            raise ValueError("Code is required for testing")
        
        results = {
            "code": code,
            "language": language,
            "test_results": [],
            "overall_success": True,
            "execution_summary": {}
        }
        
        if self.execution_enabled and language == "python":
            # Execute Python code tests
            execution_result = await self._execute_python_code(code, test_cases)
            results["test_results"] = execution_result
            results["overall_success"] = execution_result.success
        else:
            # Static analysis only
            analysis = await self._analyze_code_content(code, language)
            results["static_analysis"] = analysis.__dict__
            results["execution_summary"] = {
                "status": "static_analysis_only",
                "reason": "Code execution disabled or language not supported"
            }
        
        return results
        
    except Exception as e:
        logger.error(f"Code testing failed: {e}")
        raise

async def _execute_code(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute code safely in sandbox."""
    try:
        code = data.get("code", "")
        language = data.get("language", "python").lower()
        inputs = data.get("inputs", [])
        
        if not code:
            raise ValueError("Code is required for execution")
        
        if not self.execution_enabled:
            return {
                "success": False,
                "error": "Code execution is disabled",
                "output": ""
            }
        
        if language == "python":
            result = await self._execute_python_code(code, inputs)
        elif language == "javascript":
            result = await self._execute_javascript_code(code, inputs)
        else:
            return {
                "success": False,
                "error": f"Execution not supported for language: {language}",
                "output": ""
            }
        
        return result.__dict__
        
    except Exception as e:
        logger.error(f"Code execution failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "output": ""
        }

async def _execute_python_code(self, code: str, inputs: List[Any] = None) -> CodeExecutionResult:
    """Execute Python code safely."""
    try:
        import time
        start_time = time.time()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_file:
            tmp_file.write(code)
            tmp_file.flush()
            
            try:
                # Execute with timeout
                process = subprocess.Popen(
                    ["python", tmp_file.name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=self.sandbox_path
                )
                
                try:
                    stdout, stderr = process.communicate(timeout=self.max_execution_time)
                    execution_time = time.time() - start_time
                    
                    return CodeExecutionResult(
                        success=process.returncode == 0,
                        output=stdout,
                        error=stderr,
                        execution_time=execution_time,
                        exit_code=process.returncode
                    )
                    
                except subprocess.TimeoutExpired:
                    process.kill()
                    return CodeExecutionResult(
                        success=False,
                        error=f"Execution timed out after {self.max_execution_time}s",
                        execution_time=self.max_execution_time
                    )
                    
            finally:
                # Clean up
                Path(tmp_file.name).unlink(missing_ok=True)
                
    except Exception as e:
        return CodeExecutionResult(
            success=False,
            error=f"Execution failed: {str(e)}"
        )

async def _execute_javascript_code(self, code: str, inputs: List[Any] = None) -> CodeExecutionResult:
    """Execute JavaScript code safely."""
    try:
        import time
        start_time = time.time()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as tmp_file:
            tmp_file.write(code)
            tmp_file.flush()
            
            try:
                # Execute with Node.js
                process = subprocess.Popen(
                    ["node", tmp_file.name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=self.sandbox_path
                )
                
                try:
                    stdout, stderr = process.communicate(timeout=self.max_execution_time)
                    execution_time = time.time() - start_time
                    
                    return CodeExecutionResult(
                        success=process.returncode == 0,
                        output=stdout,
                        error=stderr,
                        execution_time=execution_time,
                        exit_code=process.returncode
                    )
                    
                except subprocess.TimeoutExpired:
                    process.kill()
                    return CodeExecutionResult(
                        success=False,
                        error=f"Execution timed out after {self.max_execution_time}s",
                        execution_time=self.max_execution_time
                    )
                    
            finally:
                # Clean up
                Path(tmp_file.name).unlink(missing_ok=True)
                
    except Exception as e:
        return CodeExecutionResult(
            success=False,
            error=f"Execution failed: {str(e)}"
        )

async def _analyze_code(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze code structure and complexity."""
    try:
        code = data.get("code", "")
        language = data.get("language", "python").lower()
        
        if not code:
            raise ValueError("Code is required for analysis")
        
        analysis = await self._analyze_code_content(code, language)
        
        return {
            "code": code,
            "language": language,
            "analysis": analysis.__dict__,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Code analysis failed: {e}")
        raise

async def _analyze_code_content(self, code: str, language: str) -> CodeAnalysis:
    """Perform detailed code analysis."""
    analysis = CodeAnalysis()
    
    try:
        if language == "python":
            analysis = await self._analyze_python_code(code)
        elif language in ["javascript", "typescript"]:
            analysis = await self._analyze_javascript_code(code)
        else:
            # Basic analysis for other languages
            analysis.lines_of_code = len([line for line in code.split('\n') if line.strip()])
            analysis.syntax_valid = True  # Assume valid for unsupported languages
            
    except Exception as e:
        analysis.syntax_valid = False
        analysis.issues.append(f"Analysis failed: {str(e)}")
    
    return analysis

async def _analyze_python_code(self, code: str) -> CodeAnalysis:
    """Analyze Python code using AST."""
    analysis = CodeAnalysis()
    
    try:
        # Parse AST
        tree = ast.parse(code)
        analysis.syntax_valid = True
        
        # Count various elements
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                analysis.functions_count += 1
            elif isinstance(node, ast.ClassDef):
                analysis.classes_count += 1
        
        # Count lines of code
        lines = code.split('\n')
        analysis.lines_of_code = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
        
        # Simple complexity calculation (number of decision points)
        complexity = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        analysis.complexity_score = complexity
        
        # Basic issue detection
        if analysis.functions_count == 0 and analysis.lines_of_code > 20:
            analysis.suggestions.append("Consider breaking code into functions")
        
        if analysis.lines_of_code > 100:
            analysis.suggestions.append("Consider splitting into multiple files")
        
    except SyntaxError as e:
        analysis.syntax_valid = False
        analysis.issues.append(f"Syntax error: {str(e)}")
    except Exception as e:
        analysis.issues.append(f"Analysis error: {str(e)}")
    
    return analysis

async def _analyze_javascript_code(self, code: str) -> CodeAnalysis:
    """Analyze JavaScript code (basic analysis)."""
    analysis = CodeAnalysis()
    
    try:
        lines = code.split('\n')
        analysis.lines_of_code = len([line for line in lines if line.strip() and not line.strip().startswith('//')])
        
        # Count functions (simple regex)
        import re
        function_pattern = r'function\s+\w+|const\s+\w+\s*=\s*\([^)]*\)\s*=>'
        analysis.functions_count = len(re.findall(function_pattern, code))
        
        # Count classes
        class_pattern = r'class\s+\w+'
        analysis.classes_count = len(re.findall(class_pattern, code))
        
        # Basic syntax check (very simple)
        brace_count = code.count('{') - code.count('}')
        paren_count = code.count('(') - code.count(')')
        
        if brace_count != 0:
            analysis.issues.append("Mismatched braces")
            analysis.syntax_valid = False

if paren_count != 0:
                analysis.issues.append("Mismatched parentheses")
                analysis.syntax_valid = False
            
            # Simple complexity estimation
            complexity_keywords = ['if', 'else', 'for', 'while', 'switch', 'case', 'try', 'catch']
            complexity = sum(code.lower().count(keyword) for keyword in complexity_keywords)
            analysis.complexity_score = complexity
            
        except Exception as e:
            analysis.issues.append(f"Analysis error: {str(e)}")
        
        return analysis
    
    async def _format_code(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format code according to language standards."""
        try:
            code = data.get("code", "")
            language = data.get("language", "python").lower()
            
            if not code:
                raise ValueError("Code is required for formatting")
            
            formatted_code = code  # Default to original
            
            if language == "python" and self.available_tools.get("black"):
                formatted_code = await self._format_python_with_black(code)
            else:
                # Use LLM for formatting
                formatted_code = await self._format_code_with_llm(code, language)
            
            return {
                "original_code": code,
                "formatted_code": formatted_code,
                "language": language,
                "formatter": "black" if language == "python" and self.available_tools.get("black") else "llm"
            }
            
        except Exception as e:
            logger.error(f"Code formatting failed: {e}")
            raise
    
    async def _format_python_with_black(self, code: str) -> str:
        """Format Python code using black formatter."""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_file:
                tmp_file.write(code)
                tmp_file.flush()
                
                try:
                    result = subprocess.run(
                        ["black", "--code", code],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    
                    if result.returncode == 0:
                        return result.stdout
                    else:
                        logger.warning(f"Black formatting failed: {result.stderr}")
                        return code
                        
                finally:
                    Path(tmp_file.name).unlink(missing_ok=True)
                    
        except Exception as e:
            logger.warning(f"Black formatting error: {e}")
            return code
    
    async def _format_code_with_llm(self, code: str, language: str) -> str:
        """Format code using LLM."""
        try:
            prompt = f"""Format the following {language} code according to standard conventions:

```{language}
{code}Please provide only the formatted code without explanations."""if not self.model_manager:
            return code
        
        response = await self.model_manager.generate_completion(
            prompt=prompt,
            temperature=0.0,
            max_tokens=len(code) + 500
        )
        
        if response.success:
            formatted = self._extract_code_blocks(response.content)
            return formatted.get("main", code)
        
        return code
        
    except Exception as e:
        logger.error(f"LLM formatting failed: {e}")
        return code

async def _explain_code(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """Explain what the code does."""
    try:
        code = data.get("code", "")
        language = data.get("language", "python").lower()
        detail_level = data.get("detail_level", "medium")  # low, medium, high
        
        if not code:
            raise ValueError("Code is required for explanation")
        
        # Build explanation prompt
        prompt = f"""Explain the following {language} code:{code}Detail level: {detail_level}Please provide:Overall purpose of the codeStep-by-step breakdown of what it doesKey algorithms or patterns usedInput and output descriptionAny notable features or potential issuesMake the explanation clear and appropriate for the detail level requested."""if not self.model_manager:
            raise ValueError("Model manager not available")
        
        response = await self.model_manager.generate_completion(
            prompt=prompt,
            temperature=0.2,
            max_tokens=1500
        )
        
        if not response.success:
            raise ValueError(f"Code explanation failed: {response.error}")
        
        explanation = response.content
        
        # Analyze code for additional context
        analysis = await self._analyze_code_content(code, language)
        
        return {
            "code": code,
            "language": language,
            "explanation": explanation,
            "detail_level": detail_level,
            "analysis": analysis.__dict__,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Code explanation failed: {e}")
        raise

async def _optimize_code(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """Optimize code for performance and efficiency."""
    try:
        code = data.get("code", "")
        language = data.get("language", "python").lower()
        optimization_focus = data.get("focus", ["performance", "readability"])
        
        if not code:
            raise ValueError("Code is required for optimization")
        
        # Analyze original code
        original_analysis = await self._analyze_code_content(code, language)
        
        # Build optimization prompt
        prompt = f"""Optimize the following {language} code for {', '.join(optimization_focus)}:{code}Please provide:Optimized version of the codeExplanation of optimizations madePerformance impact estimationAny trade-offs consideredFocus areas: {', '.join(optimization_focus)}"""if not self.model_manager:
            raise ValueError("Model manager not available")
        
        response = await self.model_manager.generate_completion(
            prompt=prompt,
            temperature=0.1,
            max_tokens=2000
        )
        
        if not response.success:
            raise ValueError(f"Code optimization failed: {response.error}")
        
        optimization_response = response.content
        
        # Extract optimized code
        code_blocks = self._extract_code_blocks(optimization_response)
        optimized_code = code_blocks.get("main", "")
        
        # Analyze optimized code
        optimized_analysis = None
        if optimized_code:
            optimized_analysis = await self._analyze_code_content(optimized_code, language)
        
        result = {
            "original_code": code,
            "optimized_code": optimized_code,
            "language": language,
            "optimization_focus": optimization_focus,
            "optimization_explanation": optimization_response,
            "original_analysis": original_analysis.__dict__,
            "optimized_analysis": optimized_analysis.__dict__ if optimized_analysis else None,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store optimization in memory
        await self.store_memory(
            content=result,
            memory_type="code_optimization",
            importance=2.5,
            tags=["optimization", language] + optimization_focus
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Code optimization failed: {e}")
        raise

async def _generate_tests(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate unit tests for the given code."""
    try:
        code = data.get("code", "")
        language = data.get("language", "python").lower()
        test_framework = data.get("test_framework", "unittest" if language == "python" else "jest")
        coverage_level = data.get("coverage_level", "comprehensive")
        
        if not code:
            raise ValueError("Code is required for test generation")
        
        # Analyze code to understand structure
        analysis = await self._analyze_code_content(code, language)
        
        # Build test generation prompt
        prompt = f"""Generate {coverage_level} unit tests for the following {language} code using {test_framework}:{code}Please provide:Test cases for normal functionalityEdge case testsError condition testsMock/stub usage where appropriateClear test names and documentationTest framework: {test_framework} Coverage level: {coverage_level}Format the tests in a code block."""if not self.model_manager:
            raise ValueError("Model manager not available")
        
        response = await self.model_manager.generate_completion(
            prompt=prompt,
            temperature=0.1,
            max_tokens=2000
        )
        
        if not response.success:
            raise ValueError(f"Test generation failed: {response.error}")
        
        test_response = response.content
        
        # Extract test code
        code_blocks = self._extract_code_blocks(test_response)
        test_code = code_blocks.get("main", test_response)
        
        result = {
            "original_code": code,
            "test_code": test_code,
            "language": language,
            "test_framework": test_framework,
            "coverage_level": coverage_level,
            "analysis": analysis.__dict__,
            "test_explanation": test_response,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store in memory
        await self.store_memory(
            content=result,
            memory_type="generated_tests",
            importance=2.0,
            tags=["test_generation", language, test_framework]
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Test generation failed: {e}")
        raise

async def _generate_documentation(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate documentation for the code."""
    try:
        code = data.get("code", "")
        language = data.get("language", "python").lower()
        doc_type = data.get("doc_type", "api")  # api, readme, inline
        include_examples = data.get("include_examples", True)
        
        if not code:
            raise ValueError("Code is required for documentation generation")
        
        # Analyze code structure
        analysis = await self._analyze_code_content(code, language)
        
        # Build documentation prompt
        prompt = f"""Generate {doc_type} documentation for the following {language} code:{code}Documentation type: {doc_type} Include examples: {include_examples}Please provide:Clear description of functionalityParameter descriptionsReturn value descriptionsUsage examples (if requested)Any important notes or warningsFormat appropriately for {doc_type} documentation."""if not self.model_manager:
            raise ValueError("Model manager not available")
        
        response = await self.model_manager.generate_completion(
            prompt=prompt,
            temperature=0.2,
            max_tokens=1500
        )
        
        if not response.success:
            raise ValueError(f"Documentation generation failed: {response.error}")
        
        documentation = response.content
        
        result = {
            "code": code,
            "documentation": documentation,
            "language": language,
            "doc_type": doc_type,
            "include_examples": include_examples,
            "analysis": analysis.__dict__,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store in memory
        await self.store_memory(
            content=result,
            memory_type="code_documentation",
            importance=2.0,
            tags=["documentation", language, doc_type]
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Documentation generation failed: {e}")
        raise

async def _agent_maintenance(self):
    """Code agent specific maintenance tasks."""
    try:
        # Clean up sandbox directory
        await self._cleanup_sandbox()
        
        # Update programming knowledge
        if len(self.agent_memory) > self.config.memory_limit * 0.8:
            await self._compress_programming_knowledge()
        
    except Exception as e:
        logger.error(f"Code agent maintenance failed: {e}")

async def _cleanup_sandbox(self):
    """Clean up temporary files in sandbox."""
    try:
        # Remove files older than 1 hour
        cutoff_time = datetime.now().timestamp() - 3600
        
        for file_path in self.sandbox_path.glob("*"):
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                try:
                    file_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to remove {file_path}: {e}")
                    
    except Exception as e:
        logger.error(f"Sandbox cleanup failed: {e}")

async def _compress_programming_knowledge(self):
    """Compress and consolidate programming knowledge in memory."""
    try:
        # Retrieve coding patterns and consolidate similar ones
        patterns = await self.retrieve_memory(memory_type="generated_code", limit=50)
        
        if len(patterns) > 20:
            # Keep only the most important/recent ones
            patterns.sort(key=lambda x: (x.importance, x.accessed_at), reverse=True)
            
            # Remove less important patterns
            for pattern in patterns[20:]:
                if pattern.memory_id in self.agent_memory:
                    del self.agent_memory[pattern.memory_id]
                    
            logger.debug("Compressed programming knowledge in memory")
            
    except Exception as e:
        logger.error(f"Knowledge compression failed: {e}")

async def _agent_shutdown(self):
    """Code agent specific shutdown tasks."""
    try:
        # Clean up all sandbox files
        for file_path in self.sandbox_path.glob("*"):
            if file_path.is_file():
                try:
                    file_path.unlink()
                except Exception:
                    pass
                    
        logger.info("Code agent shutdown complete")
        
    except Exception as e:
        logger.error(f"Code agent shutdown error: {e}")EOF
