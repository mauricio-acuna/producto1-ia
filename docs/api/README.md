# Documentación de Referencia de API

> **Framework de Agentes IA de Nivel Empresarial**  
> Documentación completa de API siguiendo estándares de OpenAI y Anthropic

---

## Tabla de Contenidos

- [Inicio Rápido](#inicio-rápido)
- [Autenticación](#autenticación)
- [Conceptos Centrales](#conceptos-centrales)
- [Gestión de Agentes](#gestión-de-agentes)
- [Registro de Herramientas](#registro-de-herramientas)
- [Orquestación de Flujos](#orquestación-de-flujos)
- [RAG y Conocimiento](#rag-y-conocimiento)
- [Seguridad y Cumplimiento](#seguridad-y-cumplimiento)
- [Monitoreo y Observabilidad](#monitoreo-y-observabilidad)
- [Manejo de Errores](#manejo-de-errores)
- [Límites de Tasa](#límites-de-tasa)
- [SDKs & Libraries](#sdks--libraries)
- [Migration Guide](#migration-guide)

---

## Quick Start

### Installation

```bash
# Python SDK
pip install ai-agent-framework

# Node.js SDK  
npm install @ai-agent/framework

# Go SDK
go get github.com/ai-agent/framework-go
```

### Basic Usage

```python
from ai_agent_framework import AgentFramework, AgentConfig

# Initialize framework
framework = AgentFramework(
    api_key="your-api-key",
    base_url="https://api.ai-agent-framework.com/v1"
)

# Create agent
agent = framework.create_agent(
    config=AgentConfig(
        name="document-analyzer",
        type="analysis",
        capabilities=["document_processing", "data_extraction"]
    )
)

# Execute workflow
result = await agent.execute_workflow(
    workflow_id="doc-analysis-v1",
    input_data={"document_url": "https://example.com/doc.pdf"}
)

print(f"Analysis complete: {result.summary}")
```

### Authentication

All API requests require authentication using API keys:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     https://api.ai-agent-framework.com/v1/agents
```

---

## Core Concepts

### Agent Types

| Type | Description | Use Cases | Capabilities |
|------|-------------|-----------|--------------|
| `analysis` | Data analysis and insights | Data science, reporting | Statistical analysis, visualization |
| `integration` | API and system integration | Enterprise connectivity | REST/GraphQL calls, data transformation |
| `knowledge` | Knowledge management | Q&A systems, research | RAG, citation, summarization |
| `workflow` | Process automation | Business workflows | Multi-step orchestration |
| `security` | Security and compliance | Audit, monitoring | Threat detection, compliance checks |

### Execution Patterns

#### PEC Pattern (Plan-Execute-Critique)
```python
class PECAgent:
    async def execute_pec_workflow(self, task: Task) -> Result:
        # Plan Phase
        plan = await self.plan(task)
        
        # Execute Phase  
        execution_result = await self.execute(plan)
        
        # Critique Phase
        critique = await self.critique(execution_result, task)
        
        return Result(
            output=execution_result.output,
            critique=critique,
            confidence=critique.confidence_score
        )
```

#### Multi-Agent Coordination
```python
class MultiAgentOrchestrator:
    async def coordinate_agents(self, workflow: Workflow) -> WorkflowResult:
        # Parallel execution
        agent_tasks = [
            self.agents['analyzer'].execute(workflow.analysis_phase),
            self.agents['integrator'].execute(workflow.integration_phase)
        ]
        
        results = await asyncio.gather(*agent_tasks)
        return self.synthesize_results(results)
```

---

## Agent Management

### Create Agent

**POST** `/v1/agents`

Create a new AI agent with specified configuration.

#### Request

```json
{
  "name": "document-processor",
  "type": "analysis", 
  "description": "Processes and analyzes documents",
  "capabilities": [
    "document_parsing",
    "text_extraction", 
    "sentiment_analysis"
  ],
  "config": {
    "max_document_size": "10MB",
    "supported_formats": ["pdf", "docx", "txt"],
    "output_format": "structured_json"
  },
  "security_policy": {
    "data_classification": "internal",
    "retention_days": 90,
    "encryption_required": true
  }
}
```

#### Response

```json
{
  "id": "agent_1a2b3c4d5e6f",
  "name": "document-processor",
  "type": "analysis",
  "status": "active",
  "created_at": "2025-08-17T10:30:00Z",
  "endpoints": {
    "execute": "/v1/agents/agent_1a2b3c4d5e6f/execute",
    "status": "/v1/agents/agent_1a2b3c4d5e6f/status",
    "logs": "/v1/agents/agent_1a2b3c4d5e6f/logs"
  },
  "capabilities": [
    "document_parsing",
    "text_extraction",
    "sentiment_analysis"
  ]
}
```

#### Code Examples

<details>
<summary>Python</summary>

```python
import asyncio
from ai_agent_framework import AgentFramework, AgentConfig, SecurityPolicy

async def create_document_processor():
    framework = AgentFramework(api_key="your-api-key")
    
    agent = await framework.create_agent(
        config=AgentConfig(
            name="document-processor",
            type="analysis",
            capabilities=[
                "document_parsing",
                "text_extraction", 
                "sentiment_analysis"
            ],
            config={
                "max_document_size": "10MB",
                "supported_formats": ["pdf", "docx", "txt"]
            },
            security_policy=SecurityPolicy(
                data_classification="internal",
                retention_days=90,
                encryption_required=True
            )
        )
    )
    
    print(f"Agent created: {agent.id}")
    return agent

# Usage
agent = asyncio.run(create_document_processor())
```
</details>

<details>
<summary>Node.js</summary>

```javascript
const { AgentFramework, AgentConfig } = require('@ai-agent/framework');

async function createDocumentProcessor() {
  const framework = new AgentFramework({
    apiKey: process.env.AI_AGENT_API_KEY
  });
  
  const agent = await framework.createAgent({
    name: 'document-processor',
    type: 'analysis',
    capabilities: [
      'document_parsing',
      'text_extraction',
      'sentiment_analysis'
    ],
    config: {
      maxDocumentSize: '10MB',
      supportedFormats: ['pdf', 'docx', 'txt']
    },
    securityPolicy: {
      dataClassification: 'internal',
      retentionDays: 90,
      encryptionRequired: true
    }
  });
  
  console.log(`Agent created: ${agent.id}`);
  return agent;
}

// Usage
createDocumentProcessor().then(agent => {
  console.log('Agent ready for processing');
});
```
</details>

<details>
<summary>cURL</summary>

```bash
curl -X POST https://api.ai-agent-framework.com/v1/agents \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "document-processor",
    "type": "analysis",
    "capabilities": [
      "document_parsing",
      "text_extraction",
      "sentiment_analysis"
    ],
    "config": {
      "max_document_size": "10MB",
      "supported_formats": ["pdf", "docx", "txt"]
    },
    "security_policy": {
      "data_classification": "internal",
      "retention_days": 90,
      "encryption_required": true
    }
  }'
```
</details>

### List Agents

**GET** `/v1/agents`

Retrieve all agents for your organization.

#### Query Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `type` | string | Filter by agent type | `null` |
| `status` | string | Filter by status (`active`, `inactive`, `error`) | `active` |
| `limit` | integer | Number of results (1-100) | `20` |
| `cursor` | string | Pagination cursor | `null` |

#### Response

```json
{
  "agents": [
    {
      "id": "agent_1a2b3c4d5e6f",
      "name": "document-processor", 
      "type": "analysis",
      "status": "active",
      "created_at": "2025-08-17T10:30:00Z",
      "last_used": "2025-08-17T14:25:33Z",
      "usage_stats": {
        "total_executions": 1247,
        "success_rate": 0.987,
        "avg_execution_time": 2.34
      }
    }
  ],
  "pagination": {
    "has_more": true,
    "next_cursor": "cursor_xyz789"
  }
}
```

### Execute Agent Workflow

**POST** `/v1/agents/{agent_id}/execute`

Execute a workflow using the specified agent.

#### Request

```json
{
  "workflow_id": "document-analysis-v1",
  "input_data": {
    "document_url": "https://example.com/contract.pdf",
    "analysis_type": "legal_review",
    "metadata": {
      "priority": "high",
      "deadline": "2025-08-18T09:00:00Z"
    }
  },
  "options": {
    "async": true,
    "webhook_url": "https://your-app.com/webhooks/agent-complete",
    "timeout_seconds": 300
  }
}
```

#### Response (Synchronous)

```json
{
  "execution_id": "exec_9x8y7z6w5v4u",
  "status": "completed",
  "result": {
    "analysis": {
      "document_type": "legal_contract",
      "key_terms": [
        {
          "term": "termination_clause",
          "location": "Section 12.3",
          "risk_level": "medium"
        }
      ],
      "compliance_score": 0.92,
      "recommendations": [
        "Review intellectual property clauses",
        "Clarify force majeure provisions"
      ]
    },
    "citations": [
      {
        "source": "contract.pdf#section_12:lines_45-52",
        "content": "Either party may terminate this agreement...",
        "confidence": 0.95
      }
    ]
  },
  "metrics": {
    "execution_time": 4.23,
    "tokens_used": 1247,
    "cost": 0.032
  }
}
```

#### Response (Asynchronous)

```json
{
  "execution_id": "exec_9x8y7z6w5v4u", 
  "status": "processing",
  "started_at": "2025-08-17T15:30:00Z",
  "estimated_completion": "2025-08-17T15:35:00Z",
  "status_url": "/v1/executions/exec_9x8y7z6w5v4u"
}
```

---

## Tool Registry

### Register Tool

**POST** `/v1/tools`

Register a new tool for use by agents.

#### Request

```json
{
  "name": "pdf_extractor",
  "description": "Extract text and metadata from PDF documents",
  "version": "1.2.0",
  "specification": {
    "type": "function",
    "function": {
      "name": "extract_pdf_content",
      "description": "Extract text content from PDF file",
      "parameters": {
        "type": "object",
        "properties": {
          "file_url": {
            "type": "string",
            "description": "URL to PDF file"
          },
          "extract_images": {
            "type": "boolean", 
            "description": "Whether to extract images",
            "default": false
          },
          "pages": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Specific pages to extract (optional)"
          }
        },
        "required": ["file_url"]
      }
    }
  },
  "access_policy": {
    "agent_types": ["analysis", "knowledge"],
    "security_clearance": "internal",
    "rate_limit": {
      "requests_per_minute": 60,
      "concurrent_requests": 10
    }
  },
  "implementation": {
    "type": "webhook",
    "endpoint": "https://your-service.com/tools/pdf-extractor",
    "authentication": {
      "type": "bearer_token",
      "token": "tool_token_xyz"
    }
  }
}
```

### List Available Tools

**GET** `/v1/tools`

#### Response

```json
{
  "tools": [
    {
      "id": "tool_pdf_extractor_v1",
      "name": "pdf_extractor",
      "version": "1.2.0",
      "description": "Extract text and metadata from PDF documents",
      "capabilities": ["text_extraction", "metadata_extraction"],
      "usage_stats": {
        "total_calls": 15420,
        "success_rate": 0.993,
        "avg_response_time": 1.87
      },
      "access_policy": {
        "agent_types": ["analysis", "knowledge"],
        "rate_limit": {
          "requests_per_minute": 60
        }
      }
    }
  ]
}
```

---

## Workflow Orchestration

### Create Workflow

**POST** `/v1/workflows`

Define a multi-step workflow for agent execution.

#### Request

```json
{
  "name": "legal-document-review",
  "description": "Comprehensive legal document analysis workflow",
  "version": "1.0.0",
  "steps": [
    {
      "id": "extract_content",
      "type": "tool_execution",
      "tool": "pdf_extractor",
      "input_mapping": {
        "file_url": "$.input.document_url"
      },
      "timeout": 30
    },
    {
      "id": "analyze_content", 
      "type": "agent_execution",
      "agent_type": "analysis",
      "input_mapping": {
        "text": "$.steps.extract_content.result.text",
        "analysis_type": "$.input.analysis_type"
      },
      "depends_on": ["extract_content"]
    },
    {
      "id": "compliance_check",
      "type": "agent_execution", 
      "agent_type": "security",
      "input_mapping": {
        "document_content": "$.steps.analyze_content.result",
        "compliance_framework": "$.input.compliance_framework"
      },
      "depends_on": ["analyze_content"]
    },
    {
      "id": "generate_report",
      "type": "synthesis",
      "input_mapping": {
        "analysis": "$.steps.analyze_content.result",
        "compliance": "$.steps.compliance_check.result"
      },
      "depends_on": ["analyze_content", "compliance_check"]
    }
  ],
  "error_handling": {
    "retry_policy": {
      "max_retries": 3,
      "backoff_strategy": "exponential"
    },
    "fallback_strategy": "partial_results"
  }
}
```

### Execute Workflow

**POST** `/v1/workflows/{workflow_id}/execute`

#### Request

```json
{
  "input": {
    "document_url": "https://contracts.example.com/partnership-agreement.pdf",
    "analysis_type": "risk_assessment",
    "compliance_framework": "SOX"
  },
  "options": {
    "async": true,
    "webhook_url": "https://your-app.com/workflow-complete"
  }
}
```

#### Response

```json
{
  "execution_id": "wf_exec_abc123def456",
  "workflow_id": "legal-document-review",
  "status": "processing",
  "current_step": "extract_content",
  "progress": 0.25,
  "started_at": "2025-08-17T16:00:00Z",
  "estimated_completion": "2025-08-17T16:10:00Z"
}
```

---

## RAG & Knowledge

### Upload Knowledge Base

**POST** `/v1/knowledge/upload`

Upload documents to the knowledge base for RAG functionality.

#### Request (Multipart)

```bash
curl -X POST https://api.ai-agent-framework.com/v1/knowledge/upload \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "file=@company-policies.pdf" \
  -F "metadata={\"category\":\"policies\",\"classification\":\"internal\"}"
```

#### Response

```json
{
  "document_id": "doc_pol_abc123",
  "status": "processing",
  "metadata": {
    "filename": "company-policies.pdf",
    "size": 2457600,
    "category": "policies",
    "classification": "internal"
  },
  "processing_status": {
    "chunking": "pending",
    "embedding": "pending", 
    "indexing": "pending"
  }
}
```

### Query Knowledge Base

**POST** `/v1/knowledge/query`

Perform RAG query against the knowledge base.

#### Request

```json
{
  "query": "What is the company policy on remote work?",
  "filters": {
    "category": ["policies", "hr"],
    "classification": "internal"
  },
  "options": {
    "max_results": 5,
    "include_citations": true,
    "similarity_threshold": 0.75
  }
}
```

#### Response

```json
{
  "query": "What is the company policy on remote work?",
  "results": [
    {
      "content": "Employees may work remotely up to 3 days per week with manager approval...",
      "score": 0.92,
      "citation": {
        "document_id": "doc_pol_abc123",
        "source": "company-policies.pdf#section_4:lines_23-31",
        "title": "Remote Work Policy"
      },
      "metadata": {
        "category": "policies",
        "last_updated": "2025-07-15T00:00:00Z"
      }
    }
  ],
  "total_results": 3,
  "processing_time": 0.234
}
```

---

## Security & Compliance

### Audit Logs

**GET** `/v1/audit/logs`

Retrieve audit logs for compliance and security monitoring.

#### Query Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `start_date` | string | ISO 8601 start date |
| `end_date` | string | ISO 8601 end date |
| `event_type` | string | Filter by event type |
| `agent_id` | string | Filter by agent |
| `user_id` | string | Filter by user |

#### Response

```json
{
  "logs": [
    {
      "id": "log_audit_xyz789",
      "timestamp": "2025-08-17T15:30:45Z",
      "event_type": "agent_execution",
      "agent_id": "agent_1a2b3c4d5e6f",
      "user_id": "user_john_doe",
      "action": "execute_workflow",
      "resource": "legal-document-review",
      "ip_address": "203.0.113.42",
      "user_agent": "Python SDK 1.2.3",
      "result": "success",
      "metadata": {
        "execution_id": "exec_9x8y7z6w5v4u",
        "duration": 4.23,
        "tokens_used": 1247
      }
    }
  ],
  "pagination": {
    "total": 15420,
    "page": 1,
    "per_page": 100,
    "has_more": true
  }
}
```

### Security Scan

**POST** `/v1/security/scan`

Perform security scan on agent configuration or workflow.

#### Request

```json
{
  "target_type": "agent",
  "target_id": "agent_1a2b3c4d5e6f",
  "scan_type": "vulnerability_assessment",
  "options": {
    "include_dependencies": true,
    "check_compliance": ["SOC2", "GDPR"]
  }
}
```

#### Response

```json
{
  "scan_id": "scan_sec_def456",
  "status": "completed",
  "started_at": "2025-08-17T16:00:00Z",
  "completed_at": "2025-08-17T16:02:15Z",
  "results": {
    "overall_score": 8.7,
    "vulnerabilities": [
      {
        "severity": "medium",
        "category": "access_control",
        "description": "Tool access policy could be more restrictive",
        "recommendation": "Implement role-based access for sensitive tools"
      }
    ],
    "compliance": {
      "SOC2": {
        "status": "compliant",
        "score": 9.2
      },
      "GDPR": {
        "status": "needs_attention",
        "score": 7.8,
        "issues": ["Data retention policy not configured"]
      }
    }
  }
}
```

---

## Monitoring & Observability

### Metrics

**GET** `/v1/metrics`

Retrieve performance and usage metrics.

#### Query Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `metric` | string | Metric name (`executions`, `latency`, `errors`, `costs`) |
| `period` | string | Time period (`1h`, `24h`, `7d`, `30d`) |
| `granularity` | string | Data granularity (`minute`, `hour`, `day`) |

#### Response

```json
{
  "metric": "executions",
  "period": "24h",
  "granularity": "hour",
  "data": [
    {
      "timestamp": "2025-08-17T00:00:00Z",
      "value": 45,
      "metadata": {
        "success_rate": 0.978,
        "avg_latency": 2.34
      }
    }
  ],
  "summary": {
    "total": 1247,
    "average": 52,
    "peak": 127,
    "success_rate": 0.987
  }
}
```

### Health Check

**GET** `/v1/health`

System health status.

#### Response

```json
{
  "status": "healthy",
  "timestamp": "2025-08-17T16:30:00Z",
  "services": {
    "agent_orchestrator": {
      "status": "healthy",
      "response_time": 23,
      "uptime": 99.99
    },
    "tool_registry": {
      "status": "healthy", 
      "response_time": 12,
      "uptime": 99.98
    },
    "knowledge_base": {
      "status": "degraded",
      "response_time": 156,
      "uptime": 99.95,
      "message": "High query volume, performance may be impacted"
    }
  },
  "region": "us-east-1",
  "version": "1.4.2"
}
```

---

## Error Handling

### Error Response Format

All API errors follow a consistent format:

```json
{
  "error": {
    "type": "validation_error",
    "code": "INVALID_AGENT_CONFIG",
    "message": "Agent configuration contains invalid capabilities",
    "details": {
      "field": "capabilities",
      "invalid_values": ["non_existent_capability"],
      "valid_options": ["document_parsing", "text_extraction", "sentiment_analysis"]
    },
    "request_id": "req_abc123def456"
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|------------|-------------|
| `INVALID_API_KEY` | 401 | API key is invalid or expired |
| `INSUFFICIENT_PERMISSIONS` | 403 | User lacks required permissions |
| `AGENT_NOT_FOUND` | 404 | Specified agent does not exist |
| `RATE_LIMIT_EXCEEDED` | 429 | Rate limit exceeded |
| `VALIDATION_ERROR` | 400 | Request validation failed |
| `INTERNAL_SERVER_ERROR` | 500 | Internal server error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |

### Retry Guidelines

```python
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def execute_with_retry(agent, workflow_data):
    try:
        return await agent.execute_workflow(workflow_data)
    except RateLimitError:
        # Rate limit errors should not be retried immediately
        raise
    except ServiceUnavailableError:
        # Service unavailable can be retried
        raise
    except ValidationError:
        # Validation errors should not be retried
        raise
```

---

## Rate Limits

### Default Limits

| Endpoint | Rate Limit | Burst Limit |
|----------|------------|-------------|
| `/v1/agents` | 100 req/min | 200 req/min |
| `/v1/agents/{id}/execute` | 1000 req/min | 1500 req/min |
| `/v1/workflows/{id}/execute` | 500 req/min | 750 req/min |
| `/v1/knowledge/query` | 2000 req/min | 3000 req/min |
| `/v1/tools` | 200 req/min | 400 req/min |

### Rate Limit Headers

All responses include rate limit information:

```http
HTTP/1.1 200 OK
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1629123456
X-RateLimit-Burst: 1500
```

### Handling Rate Limits

```python
import time
from ai_agent_framework.exceptions import RateLimitError

async def handle_rate_limited_request(agent, data):
    try:
        return await agent.execute_workflow(data)
    except RateLimitError as e:
        # Wait for reset time
        reset_time = int(e.response.headers.get('X-RateLimit-Reset', 0))
        wait_time = max(0, reset_time - int(time.time()))
        
        if wait_time > 0:
            await asyncio.sleep(wait_time)
            return await agent.execute_workflow(data)
        else:
            raise
```

---

## SDKs & Libraries

### Python SDK

```python
# Installation
pip install ai-agent-framework

# Basic usage
from ai_agent_framework import AgentFramework, AsyncAgentFramework

# Synchronous client
framework = AgentFramework(api_key="your-api-key")
agent = framework.create_agent(config)

# Asynchronous client (recommended)
async_framework = AsyncAgentFramework(api_key="your-api-key")
agent = await async_framework.create_agent(config)
```

### Node.js SDK

```javascript
// Installation  
npm install @ai-agent/framework

// Usage
const { AgentFramework } = require('@ai-agent/framework');

const framework = new AgentFramework({
  apiKey: process.env.AI_AGENT_API_KEY,
  baseURL: 'https://api.ai-agent-framework.com/v1'
});

const agent = await framework.createAgent(config);
```

### Go SDK

```go
// Installation
go get github.com/ai-agent/framework-go

// Usage
package main

import (
    "github.com/ai-agent/framework-go"
)

func main() {
    client := aiagent.NewClient("your-api-key")
    
    agent, err := client.CreateAgent(context.Background(), &aiagent.AgentConfig{
        Name: "document-processor",
        Type: "analysis",
    })
    
    if err != nil {
        log.Fatal(err)
    }
}
```

---

## Migration Guide

### Upgrading from v1.3 to v1.4

#### Breaking Changes

1. **Agent Configuration Schema**
   ```python
   # Old (v1.3)
   agent = framework.create_agent(
       name="test-agent",
       capabilities=["analysis"]
   )
   
   # New (v1.4)
   agent = framework.create_agent(
       config=AgentConfig(
           name="test-agent",
           type="analysis",  # New required field
           capabilities=["document_analysis"]  # Updated capability names
       )
   )
   ```

2. **Workflow Execution Response**
   ```python
   # Old response format
   {
       "result": {...},
       "execution_time": 4.23
   }
   
   # New response format  
   {
       "execution_id": "exec_123",
       "result": {...},
       "metrics": {
           "execution_time": 4.23,
           "tokens_used": 1247,
           "cost": 0.032
       }
   }
   ```

#### Migration Script

```python
from ai_agent_framework.migration import migrate_v13_to_v14

# Automatically migrate existing agents
migration_result = migrate_v13_to_v14(
    api_key="your-api-key",
    dry_run=True  # Preview changes
)

print(f"Agents to migrate: {len(migration_result.agents)}")
for agent in migration_result.agents:
    print(f"- {agent.name}: {agent.migration_actions}")

# Apply migration
if input("Proceed with migration? (y/n): ") == 'y':
    migrate_v13_to_v14(api_key="your-api-key", dry_run=False)
```

---

## Support & Resources

### Documentation
- [Developer Guide](https://docs.ai-agent-framework.com/guide)
- [Tutorials](https://docs.ai-agent-framework.com/tutorials)  
- [Best Practices](https://docs.ai-agent-framework.com/best-practices)

### Community
- [GitHub Repository](https://github.com/ai-agent/framework)
- [Discord Community](https://discord.gg/ai-agent-framework)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/ai-agent-framework)

### Enterprise Support
- Email: enterprise-support@ai-agent-framework.com
- SLA: 99.9% uptime guarantee
- Response Time: < 4 hours for critical issues

---

*Last updated: August 17, 2025*  
*API Version: 1.4.2*
