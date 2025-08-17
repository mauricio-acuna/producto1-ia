# RepoGPT - AI Agent para Análisis Inteligente de Repositorios

## 🎯 Descripción del Proyecto

RepoGPT es un agente de IA avanzado desarrollado como capstone del curso "AI Agents: From Zero to Production". Integra todas las técnicas aprendidas en los módulos anteriores para crear un sistema completo de análisis de código que puede entender, analizar y responder preguntas sobre repositorios de código.

### 🚀 Características Principales

- **Análisis de Código Inteligente**: Parseo AST para múltiples lenguajes (Python, JavaScript, TypeScript)
- **Arquitectura PEC**: Planner-Executor-Critic para decisiones inteligentes
- **Sistema RAG**: Retrieval-Augmented Generation con citas canónicas
- **Evaluación Continua**: Quick Evals y monitoreo de performance integrado
- **Interfaz CLI**: Interacción natural en línea de comandos
- **Indexado Semántico**: Búsqueda y recuperación de código eficiente

## 📁 Estructura del Proyecto

```
capstone/
├── repogpt.py              # Agente principal RepoGPT
├── evaluation_suite.py     # Suite de evaluación específica
├── sample_repo/            # Repositorio de ejemplo para pruebas
│   ├── main.py
│   ├── models/
│   │   └── user.py
│   ├── api/
│   │   └── endpoints.py
│   └── utils/
│       └── helpers.py
├── requirements.txt        # Dependencias del proyecto
└── README.md              # Esta documentación
```

## 🛠️ Instalación

### 1. Requisitos del Sistema

- Python 3.8+
- OpenAI API Key
- Git (para análisis de repositorios)

### 2. Instalación de Dependencias

```bash
cd capstone
pip install -r requirements.txt
```

### 3. Configuración de API Key

```bash
# Windows PowerShell
$env:OPENAI_API_KEY = "tu-api-key-aqui"

# O crear archivo .env
echo "OPENAI_API_KEY=tu-api-key-aqui" > .env
```

### 4. Archivo requirements.txt

```
openai>=1.0.0
tiktoken>=0.5.0
numpy>=1.21.0
pandas>=1.3.0
python-dotenv>=0.19.0
click>=8.0.0
rich>=12.0.0
ast-tokens>=2.4.0
tree-sitter>=0.20.0
faiss-cpu>=1.7.0
sentence-transformers>=2.2.0
```

## 🚀 Guía de Uso

### 1. Uso Básico - CLI Interactivo

```bash
python repogpt.py
```

El sistema iniciará en modo interactivo donde puedes hacer preguntas sobre el repositorio:

```
🤖 RepoGPT - AI Repository Analyst
Type 'help' for commands, 'quit' to exit

📁 Repository: sample_repo (4 files indexed)

> ¿Qué funciones hay en el proyecto?
🔍 Analyzing your query...
📋 Planning approach...
⚙️ Executing search...
✅ Answer ready!

El proyecto contiene las siguientes funciones principales:

**main.py:**
- `main()` - Punto de entrada de la aplicación [main.py#L15-L20]

**api/endpoints.py:**
- `create_app()` - Crea la aplicación Flask [api/endpoints.py#L8-L15]
- `get_users()` - Endpoint para obtener usuarios [api/endpoints.py#L18-L25]
- `create_user()` - Endpoint para crear usuario [api/endpoints.py#L28-L35]

**utils/helpers.py:**
- `format_date()` - Formatea fechas [utils/helpers.py#L5-L8]
- `validate_email()` - Valida emails [utils/helpers.py#L11-L16]
- `process_data()` - Procesa datos [utils/helpers.py#L19-L24]

> help
Available commands:
  help        - Show this help
  status      - Show repository status
  reindex     - Reindex repository
  metrics     - Show performance metrics
  eval        - Run evaluation
  quit        - Exit RepoGPT

> status
📊 Repository Status:
  Files indexed: 4
  Functions found: 7
  Classes found: 1
  Last indexed: 2024-01-15 10:30:15
```

### 2. Uso Programático

```python
from repogpt import RepoGPTAgent, RepoConfig

# Configurar agente
config = RepoConfig(
    repo_path="path/to/your/repo",
    repo_name="my_project",
    languages=['python', 'javascript']
)

# Inicializar agente
agent = RepoGPTAgent(config)
agent.initialize()

# Hacer consultas
result = agent.query("¿Cómo funciona la autenticación?")
print(f"Answer: {result.answer}")
print(f"Citations: {result.citations}")
print(f"Confidence: {result.confidence}")
```

### 3. Análisis de Repositorio Personalizado

```python
# Analizar tu propio repositorio
config = RepoConfig(
    repo_path="/ruta/a/tu/proyecto",
    repo_name="mi_proyecto",
    languages=['python', 'javascript', 'typescript'],
    exclude_patterns=['node_modules/', '*.pyc', '__pycache__/']
)

agent = RepoGPTAgent(config)
if agent.initialize():
    result = agent.query("¿Cuál es la arquitectura del proyecto?")
    print(result.answer)
```

## 📊 Evaluación y Métricas

### Ejecutar Evaluación Completa

```bash
python evaluation_suite.py
```

Esto ejecutará una suite de 9 test cases que evalúan:

- **Comprensión de código**: Precisión y recall para archivos, funciones y clases
- **Calidad de respuesta**: Relevancia, estructura y precisión técnica
- **Rendimiento**: Tiempo de respuesta, confianza y costo

### Métricas Disponibles

El sistema rastrea automáticamente:

- **Accuracy**: Precisión de las respuestas
- **Relevance**: Relevancia al contexto
- **Response Time**: Tiempo de respuesta
- **Cost**: Costo en tokens/USD
- **Citation Quality**: Precisión de las citas
- **Confidence**: Nivel de confianza del modelo

### Ejemplo de Salida de Evaluación

```
🎯 CAPSTONE EVALUATION RESULT:
========================================
🏆 EXCELENTE: 0.847 - Listo para producción!

📊 Overall Performance:
  Total Tests: 9
  Execution Time: 45.67s
  Average Score: 0.847
  Score Range: 0.712 - 0.923

🧠 Code Understanding:
  File Understanding: P=0.856, R=0.789
  Function Understanding: P=0.834, R=0.823
  Class Understanding: P=0.901, R=0.876
  Concept Coverage: 0.834

📝 Response Quality:
  Citation Accuracy: 0.823
  Response Quality: 0.856
  Relevance: 0.878
```

## 🔧 Configuración Avanzada

### 1. Personalizar Evaluadores

```python
from evaluation_suite import RepoGPTEvaluationSuite

# Crear test cases personalizados
custom_tests = [
    RepoTestCase(
        id="custom_001",
        query="¿Cómo funciona mi módulo específico?",
        expected_files=["mi_modulo.py"],
        expected_functions=["mi_funcion"],
        expected_classes=["MiClase"],
        expected_concepts=["concepto clave"],
        difficulty="medium",
        category="general"
    )
]

# Ejecutar evaluación personalizada
suite = RepoGPTEvaluationSuite(agent)
results = suite.run_evaluation(custom_tests)
```

### 2. Configurar Embeddings

```python
config = RepoConfig(
    repo_path="mi_repo",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    chunk_size=500,
    overlap=50
)
```

### 3. Personalizar Parseo

```python
# Añadir soporte para nuevos lenguajes
config = RepoConfig(
    repo_path="mi_repo",
    languages=['python', 'javascript', 'java', 'cpp'],
    custom_parsers={
        'java': 'mi_parser_java.py',
        'cpp': 'mi_parser_cpp.py'
    }
)
```

## 🧪 Casos de Uso Avanzados y Ejemplos Específicos

### 1. Onboarding Automatizado de Desarrolladores

**Escenario:** Nuevo desarrollador se une al equipo y necesita entender el proyecto rápidamente.

```python
# Script de onboarding automatizado
def developer_onboarding(repo_path: str, developer_name: str):
    """Generar guía de onboarding personalizada"""
    
    config = RepoConfig(repo_path=repo_path, repo_name="team_project")
    agent = RepoGPTAgent(config)
    agent.initialize()
    
    # Preguntas estándar de onboarding
    onboarding_questions = [
        "¿Cuál es la arquitectura general del proyecto?",
        "¿Cómo está organizada la estructura de directorios?",
        "¿Cuáles son los endpoints principales de la API?",
        "¿Qué patrones de diseño se utilizan?",
        "¿Cómo se maneja la autenticación?",
        "¿Dónde están los tests principales?",
        "¿Cuáles son las dependencias más importantes?",
        "¿Cómo se configura el entorno de desarrollo?"
    ]
    
    onboarding_guide = f"# Guía de Onboarding para {developer_name}\n\n"
    
    for i, question in enumerate(onboarding_questions, 1):
        print(f"Procesando pregunta {i}/{len(onboarding_questions)}...")
        result = agent.query(question)
        
        onboarding_guide += f"## {i}. {question}\n\n"
        onboarding_guide += f"{result.answer}\n\n"
        
        if result.citations:
            onboarding_guide += "**Referencias:**\n"
            for citation in result.citations:
                onboarding_guide += f"- {citation}\n"
        onboarding_guide += "\n---\n\n"
    
    # Guardar guía
    with open(f"onboarding_{developer_name.lower().replace(' ', '_')}.md", "w") as f:
        f.write(onboarding_guide)
    
    print(f"✅ Guía de onboarding generada para {developer_name}")
    return onboarding_guide

# Uso
developer_onboarding("./mi_proyecto", "Juan Pérez")
```

### 2. Auditoría de Seguridad Automatizada

**Escenario:** Identificar potenciales vulnerabilidades de seguridad en el código.

```python
class SecurityAuditor:
    """Auditor de seguridad basado en RepoGPT"""
    
    def __init__(self, repo_path: str):
        config = RepoConfig(repo_path=repo_path, languages=['python', 'javascript'])
        self.agent = RepoGPTAgent(config)
        self.agent.initialize()
        
        self.security_checks = [
            {
                "category": "Authentication",
                "queries": [
                    "¿Cómo se maneja la autenticación de usuarios?",
                    "¿Se están validando las credenciales correctamente?",
                    "¿Hay algún hardcoded password o API key?"
                ]
            },
            {
                "category": "Input Validation", 
                "queries": [
                    "¿Se está validando la entrada del usuario?",
                    "¿Hay protección contra SQL injection?",
                    "¿Se sanitizan los datos antes de procesarlos?"
                ]
            },
            {
                "category": "Data Protection",
                "queries": [
                    "¿Cómo se almacenan las contraseñas?",
                    "¿Se encriptan los datos sensibles?",
                    "¿Hay información confidencial en logs?"
                ]
            },
            {
                "category": "Access Control",
                "queries": [
                    "¿Hay control de acceso por roles?",
                    "¿Se verifican permisos antes de operaciones críticas?",
                    "¿Hay endpoints sin autenticación que deberían tenerla?"
                ]
            }
        ]
    
    def run_security_audit(self) -> Dict:
        """Ejecutar auditoría completa de seguridad"""
        audit_results = {
            "timestamp": datetime.now().isoformat(),
            "categories": {},
            "recommendations": [],
            "risk_level": "LOW"
        }
        
        total_issues = 0
        
        for check_category in self.security_checks:
            category_name = check_category["category"]
            category_results = []
            
            print(f"🔍 Auditando: {category_name}")
            
            for query in check_category["queries"]:
                result = self.agent.query(f"Análisis de seguridad: {query}")
                
                # Analizar respuesta para identificar problemas
                risk_score = self._analyze_security_risk(result.answer)
                
                category_results.append({
                    "query": query,
                    "answer": result.answer,
                    "risk_score": risk_score,
                    "citations": result.citations,
                    "confidence": result.confidence
                })
                
                if risk_score > 0.6:
                    total_issues += 1
            
            audit_results["categories"][category_name] = category_results
        
        # Determinar nivel de riesgo general
        if total_issues >= 5:
            audit_results["risk_level"] = "HIGH"
        elif total_issues >= 2:
            audit_results["risk_level"] = "MEDIUM"
        
        # Generar recomendaciones
        audit_results["recommendations"] = self._generate_recommendations(audit_results)
        
        return audit_results
    
    def _analyze_security_risk(self, answer: str) -> float:
        """Analizar texto para identificar indicadores de riesgo"""
        risk_indicators = {
            "high": ["hardcoded", "no validation", "plain text", "unencrypted", "vulnerable"],
            "medium": ["should validate", "consider", "might be", "potential"],
            "low": ["secure", "encrypted", "validated", "protected", "safe"]
        }
        
        answer_lower = answer.lower()
        risk_score = 0.0
        
        for high_risk in risk_indicators["high"]:
            if high_risk in answer_lower:
                risk_score += 0.3
        
        for medium_risk in risk_indicators["medium"]:
            if medium_risk in answer_lower:
                risk_score += 0.1
        
        for low_risk in risk_indicators["low"]:
            if low_risk in answer_lower:
                risk_score -= 0.1
        
        return max(0.0, min(1.0, risk_score))
    
    def _generate_recommendations(self, audit_results: Dict) -> List[str]:
        """Generar recomendaciones basadas en resultados"""
        recommendations = []
        
        if audit_results["risk_level"] == "HIGH":
            recommendations.append("🚨 CRÍTICO: Revisar y corregir problemas de seguridad inmediatamente")
        
        recommendations.extend([
            "🔐 Implementar autenticación de dos factores",
            "🛡️ Añadir validación de entrada en todos los endpoints",
            "🔒 Encriptar datos sensibles en base de datos",
            "📝 Revisar logs para información confidencial",
            "🔍 Realizar auditorías de seguridad regulares"
        ])
        
        return recommendations

# Uso
auditor = SecurityAuditor("./mi_proyecto")
audit_report = auditor.run_security_audit()

print(f"Nivel de riesgo: {audit_report['risk_level']}")
for rec in audit_report['recommendations']:
    print(f"  {rec}")
```

### 3. Documentación Automática de APIs

**Escenario:** Generar documentación actualizada de APIs automáticamente.

```python
class APIDocumentationGenerator:
    """Generador automático de documentación de APIs"""
    
    def __init__(self, repo_path: str):
        config = RepoConfig(repo_path=repo_path, languages=['python', 'javascript'])
        self.agent = RepoGPTAgent(config)
        self.agent.initialize()
    
    def generate_api_docs(self, output_format: str = "markdown") -> str:
        """Generar documentación completa de API"""
        
        # Identificar endpoints
        endpoints_query = "¿Cuáles son todos los endpoints de la API? Lista cada uno con su método HTTP, ruta y descripción."
        endpoints_result = self.agent.query(endpoints_query)
        
        # Obtener modelos de datos
        models_query = "¿Qué modelos de datos o schemas se utilizan en la API?"
        models_result = self.agent.query(models_query)
        
        # Identificar autenticación
        auth_query = "¿Cómo funciona la autenticación en la API?"
        auth_result = self.agent.query(auth_query)
        
        # Obtener ejemplos de uso
        examples_query = "¿Puedes mostrar ejemplos de cómo usar la API?"
        examples_result = self.agent.query(examples_query)
        
        if output_format == "markdown":
            return self._generate_markdown_docs(
                endpoints_result, models_result, auth_result, examples_result
            )
        elif output_format == "openapi":
            return self._generate_openapi_docs(
                endpoints_result, models_result, auth_result
            )
        else:
            raise ValueError(f"Formato no soportado: {output_format}")
    
    def _generate_markdown_docs(self, endpoints, models, auth, examples) -> str:
        """Generar documentación en formato Markdown"""
        
        docs = f"""# API Documentation
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Authentication
{auth.answer}

**References:**
{chr(10).join(f'- {cite}' for cite in auth.citations)}

## Data Models
{models.answer}

**References:**
{chr(10).join(f'- {cite}' for cite in models.citations)}

## Endpoints
{endpoints.answer}

**References:**
{chr(10).join(f'- {cite}' for cite in endpoints.citations)}

## Usage Examples
{examples.answer}

**References:**
{chr(10).join(f'- {cite}' for cite in examples.citations)}

---
*This documentation was automatically generated by RepoGPT*
"""
        return docs
    
    def _generate_openapi_docs(self, endpoints, models, auth) -> str:
        """Generar documentación en formato OpenAPI/Swagger"""
        
        # Esto requeriría parsing más sofisticado del texto
        # Para este ejemplo, generamos una estructura básica
        
        openapi_spec = {
            "openapi": "3.0.0",
            "info": {
                "title": "API Documentation",
                "version": "1.0.0",
                "description": "Auto-generated API documentation"
            },
            "servers": [
                {"url": "http://localhost:5000", "description": "Development server"}
            ],
            "paths": {},
            "components": {
                "schemas": {},
                "securitySchemes": {
                    "bearerAuth": {
                        "type": "http",
                        "scheme": "bearer"
                    }
                }
            }
        }
        
        # Aquí se podría hacer parsing del texto para extraer endpoints específicos
        # Por simplicidad, devolvemos la estructura básica
        
        import json
        return json.dumps(openapi_spec, indent=2)

# Uso
doc_generator = APIDocumentationGenerator("./mi_api")

# Generar docs en Markdown
markdown_docs = doc_generator.generate_api_docs("markdown")
with open("api_documentation.md", "w") as f:
    f.write(markdown_docs)

# Generar especificación OpenAPI
openapi_docs = doc_generator.generate_api_docs("openapi")
with open("openapi.json", "w") as f:
    f.write(openapi_docs)

print("✅ Documentación de API generada")
```

### 4. Análisis de Deuda Técnica

**Escenario:** Identificar y priorizar áreas del código que necesitan refactoring.

```python
class TechnicalDebtAnalyzer:
    """Analizador de deuda técnica"""
    
    def __init__(self, repo_path: str):
        config = RepoConfig(repo_path=repo_path)
        self.agent = RepoGPTAgent(config)
        self.agent.initialize()
    
    def analyze_technical_debt(self) -> Dict:
        """Analizar deuda técnica del proyecto"""
        
        debt_queries = [
            {
                "category": "Code Complexity",
                "query": "¿Hay funciones o clases excesivamente complejas que deberían ser refactorizadas?",
                "weight": 0.8
            },
            {
                "category": "Code Duplication", 
                "query": "¿Existe código duplicado que podría ser consolidado?",
                "weight": 0.7
            },
            {
                "category": "Outdated Dependencies",
                "query": "¿Hay dependencias obsoletas o librerías que deberían actualizarse?",
                "weight": 0.6
            },
            {
                "category": "Missing Tests",
                "query": "¿Qué partes del código no tienen tests unitarios?",
                "weight": 0.9
            },
            {
                "category": "Documentation",
                "query": "¿Qué funciones o clases carecen de documentación adecuada?",
                "weight": 0.5
            },
            {
                "category": "Error Handling",
                "query": "¿Hay áreas donde el manejo de errores es insuficiente?",
                "weight": 0.7
            },
            {
                "category": "Performance Issues",
                "query": "¿Hay patrones de código que podrían causar problemas de performance?",
                "weight": 0.8
            }
        ]
        
        debt_analysis = {
            "timestamp": datetime.now().isoformat(),
            "categories": {},
            "total_debt_score": 0.0,
            "priority_issues": [],
            "recommendations": []
        }
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for query_info in debt_queries:
            category = query_info["category"]
            query = query_info["query"]
            weight = query_info["weight"]
            
            print(f"🔍 Analizando: {category}")
            
            result = self.agent.query(query)
            debt_score = self._calculate_debt_score(result.answer)
            weighted_score = debt_score * weight
            
            debt_analysis["categories"][category] = {
                "query": query,
                "answer": result.answer,
                "debt_score": debt_score,
                "weight": weight,
                "weighted_score": weighted_score,
                "citations": result.citations,
                "confidence": result.confidence
            }
            
            total_weighted_score += weighted_score
            total_weight += weight
            
            # Identificar issues de alta prioridad
            if debt_score > 0.7 and weight > 0.7:
                debt_analysis["priority_issues"].append({
                    "category": category,
                    "debt_score": debt_score,
                    "severity": "HIGH"
                })
        
        # Calcular score total de deuda técnica
        debt_analysis["total_debt_score"] = total_weighted_score / total_weight
        
        # Generar recomendaciones
        debt_analysis["recommendations"] = self._generate_debt_recommendations(debt_analysis)
        
        return debt_analysis
    
    def _calculate_debt_score(self, answer: str) -> float:
        """Calcular score de deuda técnica basado en la respuesta"""
        debt_indicators = {
            "high": ["very complex", "duplicated", "missing", "outdated", "no tests", "poor"],
            "medium": ["could be improved", "should consider", "refactor", "update"],
            "low": ["well documented", "tested", "clean", "modern", "efficient"]
        }
        
        answer_lower = answer.lower()
        debt_score = 0.0
        
        for high_debt in debt_indicators["high"]:
            if high_debt in answer_lower:
                debt_score += 0.3
        
        for medium_debt in debt_indicators["medium"]:
            if medium_debt in answer_lower:
                debt_score += 0.1
        
        for low_debt in debt_indicators["low"]:
            if low_debt in answer_lower:
                debt_score -= 0.1
        
        return max(0.0, min(1.0, debt_score))
    
    def _generate_debt_recommendations(self, analysis: Dict) -> List[str]:
        """Generar recomendaciones para reducir deuda técnica"""
        recommendations = []
        
        debt_score = analysis["total_debt_score"]
        
        if debt_score > 0.7:
            recommendations.append("🚨 CRÍTICO: Deuda técnica alta - planificar sprint de refactoring")
        elif debt_score > 0.5:
            recommendations.append("⚠️ IMPORTANTE: Deuda técnica moderada - abordar gradualmente")
        
        # Recomendaciones específicas por categoría
        for category, details in analysis["categories"].items():
            if details["debt_score"] > 0.6:
                if category == "Missing Tests":
                    recommendations.append("🧪 Priorizar escritura de tests unitarios")
                elif category == "Code Complexity":
                    recommendations.append("🔧 Refactorizar funciones complejas")
                elif category == "Code Duplication":
                    recommendations.append("♻️ Consolidar código duplicado")
                elif category == "Documentation":
                    recommendations.append("📝 Mejorar documentación del código")
        
        return recommendations

# Uso
debt_analyzer = TechnicalDebtAnalyzer("./mi_proyecto")
debt_report = debt_analyzer.analyze_technical_debt()

print(f"Score de deuda técnica: {debt_report['total_debt_score']:.2f}")
print("\nIssues de alta prioridad:")
for issue in debt_report['priority_issues']:
    print(f"  • {issue['category']}: {issue['debt_score']:.2f}")

print("\nRecomendaciones:")
for rec in debt_report['recommendations']:
    print(f"  {rec}")
```

### 5. Migración de Código Asistida

**Escenario:** Asistir en la migración de código legacy a nuevas tecnologías.

```python
class CodeMigrationAssistant:
    """Asistente para migraciones de código"""
    
    def __init__(self, repo_path: str, source_tech: str, target_tech: str):
        config = RepoConfig(repo_path=repo_path)
        self.agent = RepoGPTAgent(config)
        self.agent.initialize()
        self.source_tech = source_tech
        self.target_tech = target_tech
    
    def analyze_migration_complexity(self) -> Dict:
        """Analizar complejidad de migración"""
        
        analysis_queries = [
            f"¿Qué componentes del código están específicamente ligados a {self.source_tech}?",
            f"¿Cuáles serían los principales desafíos al migrar de {self.source_tech} a {self.target_tech}?",
            "¿Qué dependencias externas se utilizan y cómo afectarían la migración?",
            "¿Hay patrones de código que serían incompatibles con la nueva tecnología?",
            "¿Qué partes del código podrían reutilizarse sin cambios?"
        ]
        
        migration_analysis = {
            "source_technology": self.source_tech,
            "target_technology": self.target_tech,
            "complexity_assessment": {},
            "migration_plan": [],
            "estimated_effort": "TBD"
        }
        
        for query in analysis_queries:
            result = self.agent.query(query)
            key = query.split('?')[0].lower().replace(' ', '_')
            migration_analysis["complexity_assessment"][key] = {
                "answer": result.answer,
                "citations": result.citations,
                "confidence": result.confidence
            }
        
        # Generar plan de migración
        migration_analysis["migration_plan"] = self._generate_migration_plan()
        
        return migration_analysis
    
    def _generate_migration_plan(self) -> List[Dict]:
        """Generar plan paso a paso de migración"""
        
        plan_query = f"""
        Genera un plan detallado paso a paso para migrar este proyecto de {self.source_tech} a {self.target_tech}.
        Incluye fases, dependencias entre tareas, y estimaciones de esfuerzo.
        """
        
        result = self.agent.query(plan_query)
        
        # En una implementación real, se parsearía la respuesta para extraer pasos específicos
        # Por ahora, devolvemos un plan genérico basado en buenas prácticas
        
        generic_plan = [
            {
                "phase": "Preparation",
                "tasks": [
                    "Analyze current codebase dependencies",
                    "Set up new development environment",
                    "Create migration branch",
                    "Backup current system"
                ],
                "estimated_days": 3
            },
            {
                "phase": "Foundation Migration",
                "tasks": [
                    "Migrate core data models",
                    "Set up new project structure",
                    "Implement basic configuration"
                ],
                "estimated_days": 5
            },
            {
                "phase": "Feature Migration",
                "tasks": [
                    "Migrate business logic",
                    "Update API endpoints",
                    "Migrate UI components"
                ],
                "estimated_days": 10
            },
            {
                "phase": "Testing & Validation",
                "tasks": [
                    "Write migration tests",
                    "Validate functionality",
                    "Performance testing"
                ],
                "estimated_days": 4
            },
            {
                "phase": "Deployment",
                "tasks": [
                    "Deploy to staging",
                    "User acceptance testing",
                    "Production deployment"
                ],
                "estimated_days": 3
            }
        ]
        
        return generic_plan

# Uso
migration_assistant = CodeMigrationAssistant(
    "./legacy_project", 
    "Flask", 
    "FastAPI"
)

migration_report = migration_assistant.analyze_migration_complexity()

print(f"Migración: {migration_report['source_technology']} → {migration_report['target_technology']}")
print("\nPlan de migración:")
for phase in migration_report['migration_plan']:
    print(f"\n📋 {phase['phase']} ({phase['estimated_days']} días)")
    for task in phase['tasks']:
        print(f"  • {task}")
```

Estos casos de uso demuestran cómo RepoGPT puede ser una herramienta poderosa para diferentes aspectos del desarrollo de software, desde onboarding hasta migración de código, proporcionando análisis inteligente y automatización de tareas complejas.

## 📈 Monitoreo de Performance

### Métricas en Tiempo Real

```python
# Obtener métricas del agente
metrics = agent.get_metrics()
print(f"Total queries: {metrics['total_queries']}")
print(f"Average response time: {metrics['avg_response_time']:.2f}s")
print(f"Total cost: ${metrics['total_cost']:.4f}")
```

### Dashboard de Métricas

El sistema incluye un dashboard básico que muestra:

- Historial de consultas
- Tiempos de respuesta
- Costos acumulados
- Distribución de confianza
- Tipos de consultas más frecuentes

## 🎓 Conceptos Educativos Integrados

### Módulo A: Fundamentos
- Agentes de IA conversacionales
- Procesamiento de lenguaje natural
- Interfaces de usuario

### Módulo B: Arquitectura PEC
- **Planner**: Analiza consultas y planifica estrategia
- **Executor**: Ejecuta búsquedas y análisis
- **Critic**: Evalúa calidad y ajusta respuestas

### Módulo C: RAG System
- Indexado semántico de código
- Búsqueda por similitud
- Generación con citas canónicas

### Módulo D: Evaluación
- Quick Evals automatizados
- Métricas de performance
- Monitoreo de costos

## 🔍 Troubleshooting Detallado

### Problemas de Instalación

#### **1. Error de API Key**
```
Error: OpenAI API key not found
```

**Soluciones paso a paso:**

**Windows PowerShell:**
```powershell
# Opción 1: Variable de entorno temporal
$env:OPENAI_API_KEY = "sk-..."

# Opción 2: Variable permanente
[Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "sk-...", "User")

# Verificar configuración
echo $env:OPENAI_API_KEY
```

**Archivo .env:**
```bash
# Crear archivo .env en la carpeta capstone
echo "OPENAI_API_KEY=sk-..." > .env

# Verificar que el archivo existe
Get-Content .env
```

**Código Python para verificar:**
```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    print(f"✅ API Key configurada: {api_key[:8]}...")
else:
    print("❌ API Key no encontrada")
    print("Configurar con: $env:OPENAI_API_KEY = 'sk-...'")
```

#### **2. Dependencias Faltantes**
```
ModuleNotFoundError: No module named 'sentence_transformers'
```

**Solución:**
```powershell
# Instalar dependencias específicas
pip install sentence-transformers
pip install faiss-cpu
pip install tiktoken

# O instalar todo desde requirements.txt
pip install -r requirements.txt

# Verificar instalación
python -c "import sentence_transformers; print('✅ sentence-transformers OK')"
python -c "import faiss; print('✅ faiss OK')"
```

#### **3. Problemas de Permisos**
```
PermissionError: [Errno 13] Permission denied
```

**Soluciones:**
```powershell
# Ejecutar PowerShell como administrador
# O usar instalación local
pip install --user -r requirements.txt

# Verificar permisos del directorio
icacls "capstone" /grant Everyone:F
```

### Problemas de Ejecución

#### **4. Repositorio no encontrado**
```
Error: Repository path does not exist: capstone/sample_repo
```

**Diagnóstico:**
```python
import os
from pathlib import Path

# Verificar estructura
current_dir = os.getcwd()
print(f"Directorio actual: {current_dir}")

# Listar contenido
for item in Path(".").iterdir():
    print(f"{'📁' if item.is_dir() else '📄'} {item.name}")

# Verificar sample_repo
sample_path = Path("capstone/sample_repo")
if sample_path.exists():
    print(f"✅ Sample repo encontrado: {sample_path.absolute()}")
    for file in sample_path.rglob("*.py"):
        print(f"  📄 {file.relative_to(sample_path)}")
else:
    print(f"❌ Sample repo no encontrado: {sample_path.absolute()}")
```

**Solución:**
```python
# Crear repositorio de ejemplo si no existe
def create_sample_repo():
    base_path = Path("capstone/sample_repo")
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Crear archivos de ejemplo
    files = {
        "main.py": '''#!/usr/bin/env python3
"""
Aplicación principal del proyecto de ejemplo
"""

from models.user import User
from api.endpoints import create_app

def main():
    """Punto de entrada principal"""
    print("Iniciando aplicación...")
    app = create_app()
    
    # Crear usuario de ejemplo
    user = User("admin", "admin@example.com")
    user.activate()
    
    print(f"Usuario creado: {user.username}")
    app.run(debug=True)

if __name__ == "__main__":
    main()
''',
        "models/user.py": '''"""
Modelo de usuario para la aplicación
"""

class User:
    """Clase para manejar usuarios del sistema"""
    
    def __init__(self, username: str, email: str):
        self.username = username
        self.email = email
        self.active = False
        self.created_at = None
    
    def activate(self):
        """Activar usuario"""
        self.active = True
        from datetime import datetime
        self.created_at = datetime.now()
    
    def deactivate(self):
        """Desactivar usuario"""
        self.active = False
    
    def __str__(self):
        return f"User(username='{self.username}', active={self.active})"
''',
        "api/endpoints.py": '''"""
Endpoints de la API REST
"""

from flask import Flask, jsonify, request
from models.user import User

def create_app():
    """Crear aplicación Flask"""
    app = Flask(__name__)
    
    @app.route('/users', methods=['GET'])
    def get_users():
        """Obtener lista de usuarios"""
        return jsonify([])
    
    @app.route('/users', methods=['POST'])
    def create_user():
        """Crear nuevo usuario"""
        data = request.get_json()
        user = User(data['username'], data['email'])
        return jsonify({'status': 'created'})
    
    @app.route('/users/<username>', methods=['GET'])
    def get_user(username):
        """Obtener usuario específico"""
        return jsonify({'username': username})
    
    return app
''',
        "utils/helpers.py": '''"""
Funciones de utilidad para la aplicación
"""

import re
from datetime import datetime

def format_date(date_obj):
    """Formatear fecha para display"""
    return date_obj.strftime("%Y-%m-%d %H:%M:%S")

def validate_email(email: str) -> bool:
    """Validar formato de email"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def process_data(data_list):
    """Procesar lista de datos"""
    return [item.strip().lower() for item in data_list if item]

def calculate_metrics(values):
    """Calcular métricas básicas"""
    if not values:
        return {}
    
    return {
        'count': len(values),
        'sum': sum(values),
        'average': sum(values) / len(values),
        'min': min(values),
        'max': max(values)
    }
'''
    }
    
    for file_path, content in files.items():
        full_path = base_path / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding='utf-8')
    
    print(f"✅ Sample repo creado en: {base_path.absolute()}")

# Ejecutar si es necesario
create_sample_repo()
```

#### **5. Errores de Memoria**
```
MemoryError: Unable to allocate array
```

**Soluciones:**
```python
# Configurar agente con menos memoria
config = RepoConfig(
    repo_path="sample_repo",
    chunk_size=200,  # Reducir de 500
    max_file_size=500000,  # Reducir de 1MB
    embedding_batch_size=10  # Procesar en lotes pequeños
)

# Monitorear uso de memoria
import psutil
import os

def check_memory():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memoria usada: {memory_mb:.1f} MB")

# Verificar antes y después de indexar
check_memory()
agent.initialize()
check_memory()
```

#### **6. Errores de Rate Limit**
```
RateLimitError: Rate limit exceeded
```

**Solución:**
```python
# Configurar delays entre requests
import time

class RateLimitHandler:
    def __init__(self, requests_per_minute=10):
        self.requests_per_minute = requests_per_minute
        self.min_delay = 60.0 / requests_per_minute
        self.last_request = 0
    
    def wait_if_needed(self):
        elapsed = time.time() - self.last_request
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)
        self.last_request = time.time()

# Usar en el agente
rate_limiter = RateLimitHandler(requests_per_minute=5)

# Antes de cada llamada a la API
rate_limiter.wait_if_needed()
result = openai_client.chat.completions.create(...)
```

### Problemas de Performance

#### **7. Indexado Lento**
```
Indexing taking too long...
```

**Optimizaciones:**
```python
# 1. Reducir tamaño de chunks
config.chunk_size = 300  # Menos texto por chunk

# 2. Filtrar archivos grandes
config.max_file_size = 100000  # 100KB max

# 3. Excluir directorios innecesarios
config.exclude_patterns.extend([
    "*.log", "*.tmp", "venv/*", "node_modules/*", 
    ".git/*", "__pycache__/*", "*.pyc"
])

# 4. Usar embeddings más ligeros
config.embedding_model = "all-MiniLM-L6-v2"  # Más rápido que all-mpnet

# 5. Procesar en paralelo (con cuidado)
import concurrent.futures

def parallel_embedding(chunks):
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        embeddings = list(executor.map(get_embedding, chunks))
    return embeddings
```

#### **8. Búsquedas Lentas**
```
Query taking > 10 seconds
```

**Optimizaciones:**
```python
# 1. Limitar resultados de búsqueda
config.max_results = 5  # En lugar de 10

# 2. Usar índice más eficiente
import faiss

# Crear índice IVF para grandes datasets
def create_efficient_index(embeddings):
    dimension = embeddings.shape[1]
    nlist = min(100, len(embeddings) // 10)  # Número de clusters
    
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
    
    # Entrenar índice
    index.train(embeddings)
    index.add(embeddings)
    
    return index

# 3. Cache de resultados frecuentes
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_search(query_hash):
    return perform_search(query_hash)
```

### Depuración y Logging

#### **9. Activar Logs Detallados**
```python
import logging

# Configurar logging detallado
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('repogpt.log'),
        logging.StreamHandler()
    ]
)

# Logger específico para RepoGPT
logger = logging.getLogger('repogpt')

# En el código del agente
logger.debug(f"Indexing file: {file_path}")
logger.info(f"Query received: {query}")
logger.warning(f"Low confidence score: {confidence}")
logger.error(f"Failed to process: {error}")
```

#### **10. Debug de Consultas**
```python
# Activar modo debug en el agente
agent = RepoGPTAgent(config, debug=True)

# Esto habilitará outputs como:
# 🔍 Planning query: "How does authentication work?"
# 📋 Plan generated: {"steps": [...]}
# ⚙️ Executing step 1: search for "authentication"
# 📄 Found 3 relevant files
# 🔍 Generating response with context...
# ✅ Response generated (confidence: 0.85)

# Ver contexto usado para la respuesta
result = agent.query("How does auth work?", return_context=True)
print("Context usado:")
for i, chunk in enumerate(result.context_chunks):
    print(f"{i+1}. {chunk.file_path}#{chunk.line_start}-{chunk.line_end}")
    print(f"   {chunk.content[:100]}...")
```

### Verificación de Health

#### **11. Health Check del Sistema**
```python
def system_health_check():
    """Verificar salud del sistema RepoGPT"""
    checks = []
    
    # 1. API Key
    api_key = os.getenv("OPENAI_API_KEY")
    checks.append({
        "name": "OpenAI API Key",
        "status": "✅ OK" if api_key else "❌ MISSING",
        "details": f"Key present: {bool(api_key)}"
    })
    
    # 2. Dependencias
    try:
        import sentence_transformers
        checks.append({"name": "sentence-transformers", "status": "✅ OK"})
    except ImportError:
        checks.append({"name": "sentence-transformers", "status": "❌ MISSING"})
    
    try:
        import faiss
        checks.append({"name": "faiss", "status": "✅ OK"})
    except ImportError:
        checks.append({"name": "faiss", "status": "❌ MISSING"})
    
    # 3. Sample repo
    sample_path = Path("capstone/sample_repo")
    if sample_path.exists():
        file_count = len(list(sample_path.rglob("*.py")))
        checks.append({
            "name": "Sample Repository", 
            "status": "✅ OK",
            "details": f"{file_count} Python files found"
        })
    else:
        checks.append({"name": "Sample Repository", "status": "❌ MISSING"})
    
    # 4. Memoria disponible
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    checks.append({
        "name": "Memory", 
        "status": "✅ OK" if available_gb > 1 else "⚠️ LOW",
        "details": f"{available_gb:.1f}GB available"
    })
    
    # Mostrar resultados
    print("\n🏥 RepoGPT Health Check")
    print("=" * 40)
    for check in checks:
        print(f"{check['status']} {check['name']}")
        if 'details' in check:
            print(f"    {check['details']}")
    
    # Status general
    failed_checks = [c for c in checks if "❌" in c["status"]]
    if not failed_checks:
        print("\n🎉 All systems operational!")
        return True
    else:
        print(f"\n⚠️ {len(failed_checks)} issues found")
        return False

# Ejecutar health check
if __name__ == "__main__":
    system_health_check()
```

## 🚀 Extensiones Futuras

### Ideas para Mejorar

1. **Soporte Multi-Repositorio**: Analizar múltiples repos simultáneamente
2. **Integración con Git**: Análisis de commits y cambios
3. **UI Web**: Interfaz web para uso no técnico
4. **Exportación**: Generar documentación en múltiples formatos
5. **Integraciones**: Slack, VS Code, GitHub bots

### Contribuir

Para contribuir al proyecto:

1. Fork el repositorio
2. Crear feature branch
3. Implementar mejoras
4. Añadir tests
5. Crear pull request

## 📝 Licencia

MIT License - Ver archivo LICENSE para detalles.

## 👥 Créditos

Desarrollado como proyecto capstone del curso "AI Agents: From Zero to Production"

- **Arquitectura PEC**: Basada en investigación de agentes cognitivos
- **Sistema RAG**: Implementación optimizada para código
- **Evaluación**: Framework inspirado en Quick Evals de OpenAI

---

🎯 **¡Felicidades por completar el curso!** Has construido un agente de IA completo y funcional que integra todas las técnicas avanzadas de desarrollo de agentes.

Para más información o soporte, consulta la documentación del curso o contacta al instructor.
