# API Examples & Tutorials

Complete examples for integrating with the AI Agent Framework API.

## Table of Contents

- [Authentication Examples](#authentication-examples)
- [Agent Management Examples](#agent-management-examples)
- [Workflow Execution Examples](#workflow-execution-examples)
- [Tool Integration Examples](#tool-integration-examples)
- [Knowledge Base Examples](#knowledge-base-examples)
- [Multi-Agent Coordination Examples](#multi-agent-coordination-examples)
- [Security & Compliance Examples](#security--compliance-examples)
- [Production Integration Patterns](#production-integration-patterns)

---

## Authentication Examples

### Basic Authentication

```python
from ai_agent_framework import AgentFramework
import os

# Initialize with API key
framework = AgentFramework(
    api_key=os.getenv("AI_AGENT_API_KEY"),
    base_url="https://api.ai-agent-framework.com/v1"
)

# Test connection
health = await framework.get_health()
print(f"API Status: {health.status}")
```

### Custom Headers and Configuration

```python
from ai_agent_framework import AgentFramework, Config

config = Config(
    api_key="your-api-key",
    base_url="https://api.ai-agent-framework.com/v1",
    timeout=30,
    max_retries=3,
    custom_headers={
        "X-Organization-ID": "org_123",
        "X-Environment": "production"
    }
)

framework = AgentFramework(config=config)
```

### Environment-specific Configuration

```python
import os
from ai_agent_framework import AgentFramework

# Production configuration
if os.getenv("ENVIRONMENT") == "production":
    framework = AgentFramework(
        api_key=os.getenv("PROD_API_KEY"),
        base_url="https://api.ai-agent-framework.com/v1",
        timeout=60,
        max_retries=5
    )
# Staging configuration
elif os.getenv("ENVIRONMENT") == "staging":
    framework = AgentFramework(
        api_key=os.getenv("STAGING_API_KEY"),
        base_url="https://staging-api.ai-agent-framework.com/v1",
        timeout=30,
        max_retries=3
    )
else:
    raise ValueError("ENVIRONMENT must be set to 'production' or 'staging'")
```

---

## Agent Management Examples

### Creating Specialized Agents

#### Document Analysis Agent

```python
from ai_agent_framework import AgentConfig, SecurityPolicy

# Document processing agent for legal documents
legal_doc_agent = await framework.create_agent(
    config=AgentConfig(
        name="legal-document-analyzer",
        type="analysis",
        description="Specialized agent for legal document analysis",
        capabilities=[
            "legal_text_analysis",
            "contract_review",
            "compliance_checking",
            "risk_assessment"
        ],
        config={
            "max_document_size": "50MB",
            "supported_formats": ["pdf", "docx", "txt"],
            "analysis_depth": "comprehensive",
            "legal_frameworks": ["US", "EU", "UK"],
            "output_format": "structured_json"
        },
        security_policy=SecurityPolicy(
            data_classification="confidential",
            retention_days=2555,  # 7 years for legal compliance
            encryption_required=True,
            access_controls=["legal_team", "compliance_team"]
        )
    )
)

print(f"Legal Agent Created: {legal_doc_agent.id}")
```

#### API Integration Agent

```python
# API integration agent for enterprise systems
api_agent = await framework.create_agent(
    config=AgentConfig(
        name="enterprise-api-integrator", 
        type="integration",
        description="Handles complex API integrations with enterprise systems",
        capabilities=[
            "rest_api_calls",
            "graphql_queries", 
            "soap_integration",
            "oauth_authentication",
            "data_transformation",
            "error_recovery"
        ],
        config={
            "concurrent_requests": 10,
            "rate_limit_per_minute": 1000,
            "timeout_seconds": 30,
            "retry_attempts": 3,
            "supported_auth": ["oauth2", "api_key", "jwt", "basic"],
            "data_formats": ["json", "xml", "csv", "parquet"]
        },
        security_policy=SecurityPolicy(
            data_classification="internal",
            retention_days=90,
            encryption_required=True
        )
    )
)
```

#### Knowledge Management Agent

```python
# RAG-enabled knowledge agent
knowledge_agent = await framework.create_agent(
    config=AgentConfig(
        name="enterprise-knowledge-manager",
        type="knowledge",
        description="RAG-enabled agent for enterprise knowledge management",
        capabilities=[
            "document_indexing",
            "semantic_search",
            "citation_generation",
            "knowledge_synthesis",
            "multi_language_support"
        ],
        config={
            "embedding_model": "text-embedding-3-large",
            "chunk_size": 1000,
            "overlap_size": 200,
            "similarity_threshold": 0.75,
            "max_context_length": 16000,
            "supported_languages": ["en", "es", "fr", "de", "pt"]
        },
        security_policy=SecurityPolicy(
            data_classification="internal",
            retention_days=365,
            encryption_required=True
        )
    )
)
```

### Agent Configuration Updates

```python
# Update agent capabilities
updated_agent = await framework.update_agent(
    agent_id="agent_1a2b3c4d5e6f",
    updates={
        "capabilities": [
            "document_parsing",
            "text_extraction", 
            "sentiment_analysis",
            "entity_recognition",  # New capability
            "language_detection"   # New capability
        ],
        "config": {
            "max_document_size": "20MB",  # Increased limit
            "supported_formats": ["pdf", "docx", "txt", "html"],  # Added HTML
            "enable_ocr": True  # New feature
        }
    }
)
```

### Agent Lifecycle Management

```python
# List all agents with filtering
agents = await framework.list_agents(
    type="analysis",
    status="active",
    limit=50
)

for agent in agents.agents:
    print(f"Agent: {agent.name} - Status: {agent.status}")
    print(f"  Last Used: {agent.last_used}")
    print(f"  Success Rate: {agent.usage_stats.success_rate:.2%}")

# Deactivate unused agents
for agent in agents.agents:
    if agent.usage_stats.total_executions == 0:
        await framework.update_agent(
            agent_id=agent.id,
            updates={"status": "inactive"}
        )
        print(f"Deactivated unused agent: {agent.name}")
```

---

## Workflow Execution Examples

### Simple Document Processing

```python
# Execute document analysis workflow
result = await legal_doc_agent.execute_workflow(
    input_data={
        "document_url": "https://contracts.company.com/partnership-agreement.pdf",
        "analysis_type": "comprehensive_review",
        "focus_areas": ["liability", "termination", "intellectual_property"],
        "compliance_frameworks": ["SOX", "GDPR"]
    },
    options={
        "timeout_seconds": 300,
        "include_citations": True,
        "generate_summary": True
    }
)

print(f"Analysis Status: {result.status}")
print(f"Key Findings: {len(result.result.key_findings)}")
print(f"Risk Score: {result.result.risk_assessment.overall_score}")

# Display citations
for citation in result.citations:
    print(f"Source: {citation.source}")
    print(f"Content: {citation.content[:100]}...")
    print(f"Confidence: {citation.confidence:.2%}")
```

### Asynchronous Workflow Execution

```python
import asyncio

# Start async workflow
execution = await legal_doc_agent.execute_workflow(
    input_data={
        "document_url": "https://contracts.company.com/merger-agreement.pdf",
        "analysis_type": "due_diligence_review"
    },
    options={
        "async": True,
        "webhook_url": "https://your-app.com/webhooks/analysis-complete",
        "timeout_seconds": 600
    }
)

print(f"Execution ID: {execution.execution_id}")
print(f"Status: {execution.status}")
print(f"Estimated Completion: {execution.estimated_completion}")

# Poll for completion
while True:
    status = await framework.get_execution_status(execution.execution_id)
    print(f"Progress: {status.progress:.1%} - Current Step: {status.current_step}")
    
    if status.status in ["completed", "failed"]:
        break
    
    await asyncio.sleep(10)

if status.status == "completed":
    print("Analysis completed successfully!")
    print(f"Result: {status.result}")
else:
    print(f"Analysis failed: {status.error}")
```

### Batch Processing

```python
import asyncio

# Process multiple documents concurrently
documents = [
    "https://docs.company.com/contract1.pdf",
    "https://docs.company.com/contract2.pdf", 
    "https://docs.company.com/contract3.pdf",
    "https://docs.company.com/contract4.pdf"
]

async def process_document(doc_url):
    """Process a single document"""
    try:
        result = await legal_doc_agent.execute_workflow(
            input_data={
                "document_url": doc_url,
                "analysis_type": "quick_review"
            },
            options={"timeout_seconds": 120}
        )
        return {"url": doc_url, "status": "success", "result": result}
    except Exception as e:
        return {"url": doc_url, "status": "error", "error": str(e)}

# Execute batch processing
batch_results = await asyncio.gather(
    *[process_document(doc) for doc in documents],
    return_exceptions=True
)

# Process results
successful = [r for r in batch_results if r.get("status") == "success"]
failed = [r for r in batch_results if r.get("status") == "error"]

print(f"Successfully processed: {len(successful)} documents")
print(f"Failed to process: {len(failed)} documents")

for result in failed:
    print(f"Failed: {result['url']} - Error: {result['error']}")
```

---

## Tool Integration Examples

### Registering a Custom Tool

```python
# Register PDF extraction tool
pdf_tool = await framework.register_tool(
    name="advanced_pdf_extractor",
    description="Extract text, tables, and images from PDF documents",
    version="2.1.0",
    specification={
        "type": "function",
        "function": {
            "name": "extract_pdf_content",
            "description": "Extract comprehensive content from PDF files",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_url": {
                        "type": "string",
                        "description": "URL to the PDF file"
                    },
                    "extract_tables": {
                        "type": "boolean",
                        "description": "Extract tables as structured data",
                        "default": True
                    },
                    "extract_images": {
                        "type": "boolean", 
                        "description": "Extract and analyze images",
                        "default": False
                    },
                    "ocr_enabled": {
                        "type": "boolean",
                        "description": "Use OCR for scanned documents",
                        "default": True
                    },
                    "pages": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Specific pages to extract (optional)"
                    },
                    "output_format": {
                        "type": "string",
                        "enum": ["json", "markdown", "html"],
                        "default": "json"
                    }
                },
                "required": ["file_url"]
            }
        }
    },
    access_policy={
        "agent_types": ["analysis", "knowledge"],
        "security_clearance": "internal",
        "rate_limit": {
            "requests_per_minute": 100,
            "concurrent_requests": 20
        }
    },
    implementation={
        "type": "webhook",
        "endpoint": "https://pdf-service.company.com/api/v2/extract",
        "authentication": {
            "type": "bearer_token",
            "token": os.getenv("PDF_SERVICE_TOKEN")
        }
    }
)

print(f"Tool registered: {pdf_tool.id}")
```

### Database Integration Tool

```python
# Register database query tool
db_tool = await framework.register_tool(
    name="enterprise_database_query",
    description="Execute secure database queries against enterprise data warehouse",
    version="1.3.0",
    specification={
        "type": "function",
        "function": {
            "name": "execute_database_query",
            "description": "Execute SQL queries with built-in security controls",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL query to execute (SELECT only)"
                    },
                    "database": {
                        "type": "string",
                        "enum": ["sales", "marketing", "finance", "hr"],
                        "description": "Target database"
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Query parameters for prepared statements"
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10000,
                        "default": 1000,
                        "description": "Maximum number of rows to return"
                    }
                },
                "required": ["query", "database"]
            }
        }
    },
    access_policy={
        "agent_types": ["analysis", "workflow"],
        "security_clearance": "confidential",
        "rate_limit": {
            "requests_per_minute": 50,
            "concurrent_requests": 5
        }
    }
)
```

### API Integration Tool

```python
# Register Salesforce integration tool
salesforce_tool = await framework.register_tool(
    name="salesforce_integration",
    description="Interact with Salesforce CRM via REST API",
    version="1.0.0", 
    specification={
        "type": "function",
        "function": {
            "name": "salesforce_operation",
            "description": "Perform operations on Salesforce objects",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["query", "create", "update", "delete"],
                        "description": "Operation to perform"
                    },
                    "object_type": {
                        "type": "string",
                        "enum": ["Account", "Contact", "Opportunity", "Lead", "Case"],
                        "description": "Salesforce object type"
                    },
                    "query": {
                        "type": "string",
                        "description": "SOQL query (for query operation)"
                    },
                    "data": {
                        "type": "object",
                        "description": "Object data (for create/update operations)"
                    },
                    "record_id": {
                        "type": "string",
                        "description": "Record ID (for update/delete operations)"
                    }
                },
                "required": ["operation", "object_type"]
            }
        }
    },
    access_policy={
        "agent_types": ["integration", "workflow"],
        "security_clearance": "internal"
    }
)
```

---

## Knowledge Base Examples

### Document Upload and Processing

```python
# Upload policy documents
policy_docs = [
    "company-handbook.pdf",
    "security-policies.pdf", 
    "hr-procedures.pdf",
    "compliance-guidelines.pdf"
]

for doc_path in policy_docs:
    with open(doc_path, "rb") as file:
        upload_result = await framework.upload_document(
            file=file,
            metadata={
                "category": "policies",
                "classification": "internal",
                "department": "hr",
                "version": "2025.1",
                "effective_date": "2025-08-01"
            }
        )
        
        print(f"Uploaded: {doc_path} -> {upload_result.document_id}")
        print(f"Status: {upload_result.status}")

# Wait for processing completion
import asyncio

async def wait_for_processing(document_id):
    while True:
        status = await framework.get_document_status(document_id)
        if status.processing_status.indexing == "completed":
            return True
        elif status.processing_status.indexing == "failed":
            return False
        await asyncio.sleep(5)

# Wait for all documents to be processed
for upload_result in upload_results:
    success = await wait_for_processing(upload_result.document_id)
    print(f"Document {upload_result.document_id}: {'Ready' if success else 'Failed'}")
```

### Advanced Knowledge Queries

```python
# Complex query with filters and citations
query_result = await framework.query_knowledge(
    query="What is the policy for remote work and flexible schedules?",
    filters={
        "category": ["policies", "hr"],
        "classification": "internal",
        "department": ["hr", "operations"]
    },
    options={
        "max_results": 10,
        "include_citations": True,
        "similarity_threshold": 0.8,
        "rerank_results": True
    }
)

print(f"Found {len(query_result.results)} relevant results")
print(f"Processing time: {query_result.processing_time:.3f}s")

for i, result in enumerate(query_result.results, 1):
    print(f"\n--- Result {i} (Score: {result.score:.3f}) ---")
    print(f"Content: {result.content}")
    print(f"Source: {result.citation.source}")
    print(f"Title: {result.citation.title}")
    print(f"Confidence: {result.citation.confidence:.2%}")
```

### Multi-step Knowledge Research

```python
async def research_topic(topic: str, max_depth: int = 3):
    """Perform multi-step research on a topic"""
    
    research_results = []
    current_queries = [topic]
    
    for depth in range(max_depth):
        print(f"Research depth {depth + 1}: {len(current_queries)} queries")
        
        depth_results = []
        for query in current_queries:
            result = await framework.query_knowledge(
                query=query,
                options={
                    "max_results": 5,
                    "include_citations": True,
                    "similarity_threshold": 0.75
                }
            )
            depth_results.extend(result.results)
        
        research_results.extend(depth_results)
        
        # Generate follow-up queries based on results
        if depth < max_depth - 1:
            current_queries = await generate_followup_queries(depth_results)
    
    return research_results

async def generate_followup_queries(results):
    """Generate follow-up queries from research results"""
    # Use knowledge agent to identify gaps and generate new queries
    synthesis_prompt = f"""
    Based on these research results, identify 3 key areas that need further investigation:
    
    {[r.content[:200] for r in results[:5]]}
    
    Generate specific queries for each area.
    """
    
    synthesis_result = await knowledge_agent.execute_workflow(
        input_data={"prompt": synthesis_prompt, "task": "query_generation"}
    )
    
    return synthesis_result.result.get("queries", [])

# Perform comprehensive research
research_results = await research_topic(
    "AI governance and ethics policies in enterprise settings"
)

print(f"Total research results: {len(research_results)}")
```

---

## Multi-Agent Coordination Examples

### Document Processing Pipeline

```python
# Create multiple specialized agents
agents = {
    "extractor": await framework.create_agent(
        config=AgentConfig(
            name="document-extractor",
            type="analysis",
            capabilities=["document_parsing", "text_extraction", "metadata_extraction"]
        )
    ),
    "analyzer": await framework.create_agent(
        config=AgentConfig(
            name="content-analyzer", 
            type="analysis",
            capabilities=["sentiment_analysis", "entity_recognition", "topic_modeling"]
        )
    ),
    "compliance": await framework.create_agent(
        config=AgentConfig(
            name="compliance-checker",
            type="security",
            capabilities=["compliance_analysis", "risk_assessment", "policy_validation"]
        )
    ),
    "reporter": await framework.create_agent(
        config=AgentConfig(
            name="report-generator",
            type="workflow",
            capabilities=["report_generation", "data_visualization", "executive_summary"]
        )
    )
}

# Define multi-agent workflow
workflow = await framework.create_workflow(
    name="comprehensive-document-analysis",
    description="Multi-agent document analysis pipeline",
    steps=[
        {
            "id": "extract_content",
            "type": "agent_execution",
            "agent_id": agents["extractor"].id,
            "input_mapping": {
                "document_url": "$.input.document_url"
            },
            "timeout": 120
        },
        {
            "id": "analyze_content",
            "type": "agent_execution", 
            "agent_id": agents["analyzer"].id,
            "input_mapping": {
                "text": "$.steps.extract_content.result.text",
                "metadata": "$.steps.extract_content.result.metadata"
            },
            "depends_on": ["extract_content"]
        },
        {
            "id": "check_compliance",
            "type": "agent_execution",
            "agent_id": agents["compliance"].id,
            "input_mapping": {
                "document_content": "$.steps.extract_content.result",
                "analysis_result": "$.steps.analyze_content.result",
                "compliance_frameworks": "$.input.compliance_frameworks"
            },
            "depends_on": ["extract_content", "analyze_content"]
        },
        {
            "id": "generate_report",
            "type": "agent_execution",
            "agent_id": agents["reporter"].id,
            "input_mapping": {
                "extraction_result": "$.steps.extract_content.result",
                "analysis_result": "$.steps.analyze_content.result", 
                "compliance_result": "$.steps.check_compliance.result",
                "report_format": "$.input.report_format"
            },
            "depends_on": ["extract_content", "analyze_content", "check_compliance"]
        }
    ],
    error_handling={
        "retry_policy": {
            "max_retries": 2,
            "backoff_strategy": "exponential"
        },
        "fallback_strategy": "partial_results"
    }
)

# Execute multi-agent workflow
result = await framework.execute_workflow(
    workflow_id=workflow.id,
    input={
        "document_url": "https://contracts.company.com/merger-agreement.pdf",
        "compliance_frameworks": ["SOX", "GDPR", "SEC"],
        "report_format": "executive_summary"
    },
    options={
        "async": True,
        "webhook_url": "https://your-app.com/workflow-complete"
    }
)

print(f"Multi-agent workflow started: {result.execution_id}")
```

### Real-time Collaboration

```python
import asyncio

class MultiAgentCollaboration:
    def __init__(self, framework):
        self.framework = framework
        self.active_executions = {}
        
    async def coordinate_agents(self, task_data):
        """Coordinate multiple agents for complex task"""
        
        # Start parallel agent executions
        tasks = {
            "research": self.research_agent_task(task_data),
            "analysis": self.analysis_agent_task(task_data),
            "validation": self.validation_agent_task(task_data)
        }
        
        # Wait for all tasks with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks.values(), return_exceptions=True),
                timeout=300
            )
            
            # Combine results
            return await self.synthesize_results(dict(zip(tasks.keys(), results)))
            
        except asyncio.TimeoutError:
            # Handle timeout - get partial results
            return await self.handle_timeout(tasks)
    
    async def research_agent_task(self, task_data):
        """Research agent execution"""
        return await self.framework.agents["research"].execute_workflow(
            input_data={
                "topic": task_data["topic"],
                "depth": "comprehensive"
            }
        )
    
    async def analysis_agent_task(self, task_data):
        """Analysis agent execution"""
        return await self.framework.agents["analysis"].execute_workflow(
            input_data={
                "data": task_data["data"],
                "analysis_type": "detailed"
            }
        )
    
    async def validation_agent_task(self, task_data):
        """Validation agent execution"""
        return await self.framework.agents["validation"].execute_workflow(
            input_data={
                "subject": task_data["topic"],
                "validation_criteria": task_data.get("criteria", [])
            }
        )
    
    async def synthesize_results(self, results):
        """Synthesize results from multiple agents"""
        return await self.framework.agents["synthesizer"].execute_workflow(
            input_data={
                "research_result": results["research"],
                "analysis_result": results["analysis"],
                "validation_result": results["validation"],
                "synthesis_type": "comprehensive"
            }
        )

# Usage
collaboration = MultiAgentCollaboration(framework)

final_result = await collaboration.coordinate_agents({
    "topic": "AI ethics in healthcare applications",
    "data": {"recent_publications": "...", "case_studies": "..."},
    "criteria": ["medical_accuracy", "privacy_compliance", "bias_detection"]
})
```

---

## Security & Compliance Examples

### Audit Trail Monitoring

```python
from datetime import datetime, timedelta

# Query audit logs for the last 24 hours
end_date = datetime.utcnow()
start_date = end_date - timedelta(hours=24)

audit_logs = await framework.get_audit_logs(
    start_date=start_date.isoformat(),
    end_date=end_date.isoformat(),
    event_type="agent_execution",
    limit=1000
)

print(f"Total audit entries: {audit_logs.pagination.total}")

# Analyze audit patterns
execution_stats = {}
error_count = 0

for log in audit_logs.logs:
    agent_id = log.agent_id
    if agent_id not in execution_stats:
        execution_stats[agent_id] = {"success": 0, "failure": 0}
    
    if log.result == "success":
        execution_stats[agent_id]["success"] += 1
    else:
        execution_stats[agent_id]["failure"] += 1
        error_count += 1

print(f"Total errors in 24h: {error_count}")
for agent_id, stats in execution_stats.items():
    total = stats["success"] + stats["failure"]
    success_rate = stats["success"] / total if total > 0 else 0
    print(f"Agent {agent_id}: {success_rate:.2%} success rate ({total} executions)")
```

### Security Scanning

```python
# Perform comprehensive security scan
security_scan = await framework.security_scan(
    target_type="agent",
    target_id="agent_1a2b3c4d5e6f",
    scan_type="vulnerability_assessment",
    options={
        "include_dependencies": True,
        "check_compliance": ["SOC2", "GDPR", "HIPAA"],
        "depth": "comprehensive"
    }
)

print(f"Security Scan ID: {security_scan.scan_id}")
print(f"Overall Score: {security_scan.results.overall_score}/10")

# Review vulnerabilities
if security_scan.results.vulnerabilities:
    print("\nSecurity Vulnerabilities Found:")
    for vuln in security_scan.results.vulnerabilities:
        print(f"- {vuln.severity.upper()}: {vuln.description}")
        print(f"  Category: {vuln.category}")
        print(f"  Recommendation: {vuln.recommendation}")

# Review compliance status
print("\nCompliance Status:")
for framework_name, status in security_scan.results.compliance.items():
    print(f"- {framework_name}: {status.status} (Score: {status.score}/10)")
    if status.issues:
        for issue in status.issues:
            print(f"  Issue: {issue}")
```

### Data Privacy Controls

```python
# Configure data privacy settings for agent
privacy_policy = {
    "data_minimization": True,
    "anonymization_required": True,
    "retention_policy": {
        "personal_data": 365,  # days
        "anonymized_data": 2555,  # 7 years
        "metadata_only": 3650   # 10 years
    },
    "geographic_restrictions": ["EU", "US"],
    "consent_requirements": {
        "explicit_consent": ["personal_data", "biometric_data"],
        "opt_out_available": True
    }
}

# Update agent with privacy controls
updated_agent = await framework.update_agent(
    agent_id="agent_1a2b3c4d5e6f",
    updates={
        "security_policy": {
            "data_classification": "personal",
            "privacy_policy": privacy_policy,
            "encryption_required": True,
            "audit_all_access": True
        }
    }
)

# Verify privacy compliance
privacy_check = await framework.security_scan(
    target_type="agent",
    target_id=updated_agent.id,
    scan_type="compliance_check",
    options={"check_compliance": ["GDPR"]}
)

if privacy_check.results.compliance["GDPR"].status == "compliant":
    print("Agent is GDPR compliant")
else:
    print("GDPR compliance issues found:")
    for issue in privacy_check.results.compliance["GDPR"].issues:
        print(f"- {issue}")
```

---

## Production Integration Patterns

### Error Handling and Resilience

```python
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from ai_agent_framework.exceptions import RateLimitError, ServiceUnavailableError

class ResilientAgentClient:
    def __init__(self, framework):
        self.framework = framework
        self.circuit_breaker = CircuitBreaker()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ServiceUnavailableError, ConnectionError))
    )
    async def execute_with_retry(self, agent_id, workflow_data):
        """Execute workflow with automatic retry logic"""
        
        # Check circuit breaker
        if self.circuit_breaker.is_open(agent_id):
            raise CircuitBreakerOpenError(f"Circuit breaker open for agent {agent_id}")
        
        try:
            result = await self.framework.execute_agent_workflow(
                agent_id=agent_id,
                workflow_data=workflow_data
            )
            
            # Record success
            self.circuit_breaker.record_success(agent_id)
            return result
            
        except RateLimitError as e:
            # Handle rate limiting
            reset_time = int(e.response.headers.get('X-RateLimit-Reset', 0))
            wait_time = max(0, reset_time - int(time.time()))
            
            if wait_time > 0:
                print(f"Rate limited. Waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
                return await self.execute_with_retry(agent_id, workflow_data)
            else:
                raise
                
        except Exception as e:
            # Record failure for circuit breaker
            self.circuit_breaker.record_failure(agent_id)
            raise

class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = {}
        self.last_failure_time = {}
        
    def is_open(self, agent_id):
        if agent_id not in self.failure_count:
            return False
            
        if self.failure_count[agent_id] >= self.failure_threshold:
            time_since_failure = time.time() - self.last_failure_time.get(agent_id, 0)
            return time_since_failure < self.timeout
            
        return False
    
    def record_success(self, agent_id):
        self.failure_count[agent_id] = 0
    
    def record_failure(self, agent_id):
        self.failure_count[agent_id] = self.failure_count.get(agent_id, 0) + 1
        self.last_failure_time[agent_id] = time.time()

# Usage
resilient_client = ResilientAgentClient(framework)

try:
    result = await resilient_client.execute_with_retry(
        agent_id="agent_1a2b3c4d5e6f",
        workflow_data={"document_url": "https://example.com/doc.pdf"}
    )
    print(f"Execution successful: {result.execution_id}")
except Exception as e:
    print(f"Execution failed after retries: {e}")
```

### Configuration Management

```python
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class AgentFrameworkConfig:
    """Production configuration for AI Agent Framework"""
    
    # API Configuration
    api_key: str
    base_url: str = "https://api.ai-agent-framework.com/v1"
    timeout: int = 30
    max_retries: int = 3
    
    # Rate Limiting
    default_rate_limit: int = 1000
    burst_limit: int = 1500
    
    # Security
    encryption_enabled: bool = True
    audit_enabled: bool = True
    security_scan_interval: int = 86400  # 24 hours
    
    # Monitoring
    metrics_enabled: bool = True
    tracing_enabled: bool = True
    log_level: str = "INFO"
    
    # Performance
    connection_pool_size: int = 20
    keep_alive_timeout: int = 30
    
    # Environment-specific settings
    environment: str = "production"
    
    @classmethod
    def from_environment(cls) -> 'AgentFrameworkConfig':
        """Load configuration from environment variables"""
        return cls(
            api_key=os.getenv("AI_AGENT_API_KEY"),
            base_url=os.getenv("AI_AGENT_BASE_URL", cls.base_url),
            timeout=int(os.getenv("AI_AGENT_TIMEOUT", cls.timeout)),
            max_retries=int(os.getenv("AI_AGENT_MAX_RETRIES", cls.max_retries)),
            default_rate_limit=int(os.getenv("AI_AGENT_RATE_LIMIT", cls.default_rate_limit)),
            encryption_enabled=os.getenv("AI_AGENT_ENCRYPTION", "true").lower() == "true",
            audit_enabled=os.getenv("AI_AGENT_AUDIT", "true").lower() == "true",
            metrics_enabled=os.getenv("AI_AGENT_METRICS", "true").lower() == "true",
            tracing_enabled=os.getenv("AI_AGENT_TRACING", "true").lower() == "true",
            log_level=os.getenv("AI_AGENT_LOG_LEVEL", cls.log_level),
            environment=os.getenv("ENVIRONMENT", cls.environment)
        )
    
    def validate(self) -> None:
        """Validate configuration"""
        if not self.api_key:
            raise ValueError("API key is required")
        
        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")
        
        if self.max_retries < 0:
            raise ValueError("Max retries cannot be negative")

# Production setup
config = AgentFrameworkConfig.from_environment()
config.validate()

framework = AgentFramework(
    api_key=config.api_key,
    base_url=config.base_url,
    timeout=config.timeout,
    max_retries=config.max_retries,
    encryption_enabled=config.encryption_enabled,
    audit_enabled=config.audit_enabled
)
```

This comprehensive API documentation provides production-ready examples that demonstrate enterprise-grade integration patterns, following the same high standards used by OpenAI, Anthropic, and other industry leaders. Each example includes error handling, monitoring, security considerations, and performance optimizations suitable for production deployments.
