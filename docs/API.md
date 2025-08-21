# API Documentation

## Overview

The Ultra AI Project provides a comprehensive REST API and WebSocket interface for interacting with AI agents and system components. This documentation covers all available endpoints, request/response formats, authentication, and usage examples.

## Base URLhttp://localhost:8000/api/v1## Authentication

### API Key Authentication

Include your API key in the request headers:

```http
Authorization: Bearer your-api-key-hereJWT Token AuthenticationFor web applications, use JWT tokens:Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...Getting API Keys# Generate a new API key
curl -X POST http://localhost:8000/api/v1/auth/api-keys \
  -H "Content-Type: application/json" \
  -d '{"name": "my-app", "permissions": ["tasks:create", "tasks:read"]}'Core EndpointsHealth CheckCheck system status and health.GET /healthResponse:{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": 3600,
  "components": {
    "database": "healthy",
    "ai_models": "healthy",
    "agents": "healthy"
  }
}System InformationGet system configuration and capabilities.GET /system/infoResponse:{
  "version": "1.0.0",
  "agents": ["code", "research", "analysis", "creative"],
  "models": {
    "openai": ["gpt-4", "gpt-3.5-turbo"],
    "anthropic": ["claude-3-opus", "claude-3-sonnet"]
  },
  "features": ["websocket", "streaming", "file_upload"]
}Task ManagementCreate TaskSubmit a new task to an AI agent.POST /tasksRequest Body:{
  "agent": "code",
  "task": "Create a Python function to calculate fibonacci numbers",
  "parameters": {
    "language": "python",
    "style": "recursive",
    "include_tests": true
  },
  "priority": "normal",
  "timeout": 300
}Response:{
  "task_id": "task_12345",
  "status": "queued",
  "created_at": "2025-08-21T10:30:00Z",
  "estimated_completion": "2025-08-21T10:35:00Z"
}Get Task StatusCheck the status of a specific task.GET /tasks/{task_id}Response:{
  "task_id": "task_12345",
  "status": "completed",
  "agent": "code",
  "created_at": "2025-08-21T10:30:00Z",
  "completed_at": "2025-08-21T10:32:15Z",
  "result": {
    "success": true,
    "output": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "metadata": {
      "execution_time": 2.5,
      "tokens_used": 150,
      "model": "gpt-4"
    }
  }
}List TasksGet a list of tasks with optional filtering.GET /tasks?status=completed&agent=code&limit=10&offset=0Query Parameters:status: Filter by task status (queued, running, completed, failed)agent: Filter by agent typelimit: Number of results to return (default: 20, max: 100)offset: Number of results to skip (default: 0)sort: Sort order (created_at, completed_at, priority)Response:{
  "tasks": [
    {
      "task_id": "task_12345",
      "status": "completed",
      "agent": "code",
      "task": "Create a Python function...",
      "created_at": "2025-08-21T10:30:00Z",
      "completed_at": "2025-08-21T10:32:15Z"
    }
  ],
  "total": 45,
  "limit": 10,
  "offset": 0
}Cancel TaskCancel a queued or running task.DELETE /tasks/{task_id}Response:{
  "task_id": "task_12345",
  "status": "cancelled",
  "message": "Task cancelled successfully"
}Agent ManagementList AgentsGet information about available agents.GET /agentsResponse:{
  "agents": [
    {
      "name": "code",
      "description": "AI agent for code generation and analysis",
      "capabilities": ["generation", "review", "optimization"],
      "status": "active",
      "version": "1.0.0"
    },
    {
      "name": "research",
      "description": "AI agent for research and information gathering",
      "capabilities": ["web_search", "analysis", "summarization"],
      "status": "active",
      "version": "1.0.0"
    }
  ]
}Get Agent DetailsGet detailed information about a specific agent.GET /agents/{agent_name}Response:{
  "name": "code",
  "description": "AI agent for code generation and analysis",
  "capabilities": ["generation", "review", "optimization", "debugging"],
  "parameters": {
    "language": {
      "type": "string",
      "required": false,
      "default": "python",
      "options": ["python", "javascript", "java", "cpp", "go"]
    },
    "style": {
      "type": "string",
      "required": false,
      "default": "clean",
      "options": ["clean", "compact", "documented"]
    }
  },
  "status": "active",
  "version": "1.0.0",
  "stats": {
    "tasks_completed": 1250,
    "average_response_time": 2.3,
    "success_rate": 0.97
  }
}Agent ConfigurationUpdate agent configuration (admin only).PUT /agents/{agent_name}/configRequest Body:{
  "model": "gpt-4",
  "max_tokens": 2000,
  "temperature": 0.7,
  "timeout": 300
}File ManagementUpload FileUpload a file for processing by agents.POST /files
Content-Type: multipart/form-dataForm Data:file: The file to uploaddescription: Optional file descriptiontags: Optional comma-separated tagsResponse:{
  "file_id": "file_67890",
  "filename": "data.csv",
  "size": 1024000,
  "mime_type": "text/csv",
  "uploaded_at": "2025-08-21T10:30:00Z",
  "url": "/api/v1/files/file_67890"
}Get FileRetrieve an uploaded file.GET /files/{file_id}List FilesGet a list of uploaded files.GET /files?limit=10&offset=0Response:{
  "files": [
    {
      "file_id": "file_67890",
      "filename": "data.csv",
      "size": 1024000,
      "mime_type": "text/csv",
      "uploaded_at": "2025-08-21T10:30:00Z"
    }
  ],
  "total": 15,
  "limit": 10,
  "offset": 0
}Delete FileDelete an uploaded file.DELETE /files/{file_id}WebSocket APIConnectionConnect to the WebSocket endpoint for real-time communication:const ws = new WebSocket('ws://localhost:8000/ws');AuthenticationSend authentication message after connection:ws.send(JSON.stringify({
  type: 'auth',
  token: 'your-jwt-token-here'
}));Task StreamingSubmit a task with streaming response:ws.send(JSON.stringify({
  type: 'task',
  agent: 'code',
  task: 'Create a web scraper',
  stream: true
}));Message TypesTask Update{
  "type": "task_update",
  "task_id": "task_12345",
  "status": "running",
  "progress": 0.3
}Streaming Response{
  "type": "stream",
  "task_id": "task_12345",
  "chunk": "def scrape_website(url):\n    import requests\n"
}Task Complete{
  "type": "task_complete",
  "task_id": "task_12345",
  "result": {
    "success": true,
    "output": "Complete code here..."
  }
}Error{
  "type": "error",
  "task_id": "task_12345",
  "error": "Rate limit exceeded",
  "code": "RATE_LIMIT"
}Configuration ManagementGet ConfigurationRetrieve system configuration (admin only).GET /configUpdate ConfigurationUpdate system configuration (admin only).PUT /configRequest Body:{
  "system": {
    "max_workers": 8,
    "timeout": 600
  },
  "models": {
    "openai": {
      "default_model": "gpt-4",
      "api_key": "sk-..."
    }
  }
}Analytics and MonitoringUsage StatisticsGet system usage statistics.GET /analytics/usage?period=7dResponse:{
  "period": "7d",
  "total_tasks": 1250,
  "successful_tasks": 1213,
  "failed_tasks": 37,
  "average_response_time": 2.3,
  "by_agent": {
    "code": 450,
    "research": 320,
    "analysis": 280,
    "creative": 200
  },
  "by_day": [
    {"date": "2025-08-15", "tasks": 180},
    {"date": "2025-08-16", "tasks": 195}
  ]
}Performance MetricsGet detailed performance metrics.GET /analytics/performanceResponse:{
  "response_times": {
    "p50": 1.2,
    "p95": 4.8,
    "p99": 8.1
  },
  "error_rates": {
    "total": 0.03,
    "by_agent": {
      "code": 0.02,
      "research": 0.04
    }
  },
  "resource_usage": {
    "cpu": 0.45,
    "memory": 0.62,
    "disk": 0.23
  }
}Error HandlingError Response FormatAll errors follow a consistent format:{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid agent name provided",
    "details": {
      "field": "agent",
      "provided": "invalid_agent",
      "valid_options": ["code", "research", "analysis", "creative"]
    },
    "timestamp": "2025-08-21T10:30:00Z"
  }
}Common Error CodesCodeHTTP StatusDescriptionVALIDATION_ERROR400Request validation failedUNAUTHORIZED401Authentication requiredFORBIDDEN403Insufficient permissionsNOT_FOUND404Resource not foundRATE_LIMIT429Rate limit exceededAGENT_UNAVAILABLE503Agent temporarily unavailableINTERNAL_ERROR500Internal system errorRate LimitingDefault LimitsAPI Requests: 1000 requests per hour per API keyTask Submissions: 100 tasks per hour per userFile Uploads: 50 uploads per hour per userWebSocket Connections: 10 concurrent connections per userRate Limit HeadersAll responses include rate limit information:X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 845
X-RateLimit-Reset: 1692615600SDK and Client LibrariesPython SDKfrom ultra_ai import Client

client = Client(api_key="your-api-key")

# Submit a task
task = client.tasks.create(
    agent="code",
    task="Create a REST API",
    parameters={"framework": "fastapi"}
)

# Wait for completion
result = client.tasks.wait(task.id)
print(result.output)JavaScript SDKimport { UltraAI } from 'ultra-ai-js';

const client = new UltraAI({ apiKey: 'your-api-key' });

// Submit a task
const task = await client.tasks.create({
  agent: 'research',
  task: 'Research latest AI trends',
  parameters: { sources: 'academic' }
});

// Stream results
client.tasks.stream(task.id, (chunk) => {
  console.log(chunk);
});ExamplesCode Generation Examplecurl -X POST http://localhost:8000/api/v1/tasks \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "code",
    "task": "Create a Python class for a binary search tree",
    "parameters": {
      "language": "python",
      "include_tests": true,
      "style": "documented"
    }
  }'Research Task Examplecurl -X POST http://localhost:8000/api/v1/tasks \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "research",
    "task": "Research the latest developments in quantum computing",
    "parameters": {
      "sources": ["academic", "news"],
      "date_range": "2025",
      "max_sources": 10
    }
  }'File Analysis Example# Upload file
curl -X POST http://localhost:8000/api/v1/files \
  -H "Authorization: Bearer your-api-key" \
  -F "file=@data.csv" \
  -F "description=Sales data for analysis"

# Analyze file
curl -X POST http://localhost:8000/api/v1/tasks \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "analysis",
    "task": "Analyze sales trends in the uploaded data",
    "parameters": {
      "file_id": "file_67890",
      "analysis_type": "trend",
      "generate_charts": true
    }
  }'VersioningThe API uses semantic versioning. The current version is v1. All endpoints are prefixed with /api/v1/.Version Historyv1.0.0: Initial releasev1.1.0: Added file upload supportv1.2.0: Added WebSocket streamingBackward CompatibilityWe maintain backward compatibility within major versions. Deprecated features will be announced at least 6 months before removal.SupportDocumentation: docs.ultra-ai-project.comGitHub Issues: github.com/ultra-ai/issuesAPI Status: status.ultra-ai-project.com
