# Módulo B: Primer Mini-Agente
## Patrón Planner→Executor→Critic • Tool Calling Seguro • Manejo de Errores

---

### 🎯 Objetivos del Módulo

Al finalizar este módulo serás capaz de:
- **Implementar** un agente siguiendo el patrón Planner-Executor-Critic (PEC)
- **Integrar** tool calling seguro con validación de herramientas
- **Manejar** errores y flujos de recuperación en agentes
- **Evaluar** resultados y tomar decisiones de retry/abort
- **Crear** un agente autónomo que puede completar tareas complejas

**⏱️ Duración estimada:** 3-4 horas  
**🔧 Prerrequisitos:** Módulo A completado, conceptos de APIs, JSON Schema

---

## 1. Arquitectura del Patrón PEC

### 1.1 ¿Qué es el Patrón Planner-Executor-Critic?

El patrón **PEC** divide la responsabilidad del agente en tres componentes especializados:

```
┌─────────────────────────────────────────────────────────┐
│                    AGENTE PEC                           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   PLANNER   │───▶│  EXECUTOR   │───▶│   CRITIC    │  │
│  │             │    │             │    │             │  │
│  │ • Analiza   │    │ • Ejecuta   │    │ • Evalúa    │  │
│  │ • Planifica │    │ • Herramient│    │ • Decide    │  │
│  │ • Estructura│    │ • Acciones  │    │ • Retry/Ok  │  │
│  └─────────────┘    └─────────────┘    └─────────────┘  │
│         │                    │                    │     │
│         └──────── FEEDBACK LOOP ──────────────────┘     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 1.2 Responsabilidades de Cada Componente

**🧠 PLANNER (Planificador)**
- Analizar el objetivo del usuario
- Descomponer en pasos ejecutables
- Seleccionar herramientas apropiadas
- Generar plan estructurado en JSON

**⚡ EXECUTOR (Ejecutor)**
- Ejecutar cada paso del plan
- Llamar a herramientas externas
- Manejar parámetros y validaciones
- Capturar resultados y errores

**👨‍⚖️ CRITIC (Crítico)**
- Evaluar si los resultados son satisfactorios
- Detectar errores o resultados incompletos
- Decidir si continuar, reintentar o abortar
- Proporcionar feedback para mejoras

### 1.3 Ventajas del Patrón PEC

✅ **Modularidad**: Cada componente tiene responsabilidad clara  
✅ **Testabilidad**: Se puede probar cada parte independientemente  
✅ **Debugging**: Fácil identificar dónde ocurren problemas  
✅ **Escalabilidad**: Se pueden mejorar componentes por separado  
✅ **Robustez**: Manejo de errores distribuido y especializado  

---

## 2. Diseño del Mini-Agente

### 2.1 Especificación del Agente

Nuestro mini-agente será capaz de:
- 🔍 **Buscar información** en documentos
- 🧮 **Realizar cálculos** matemáticos
- 📄 **Formatear respuestas** en diferentes estilos
- ⏰ **Obtener fecha/hora** actual
- 🌐 **Consultar APIs** externas (simuladas)

### 2.2 Schema del Plan de Ejecución

```json
{
  "plan_id": "plan_20250816_143022",
  "goal": "Buscar información sobre Python y calcular estadísticas",
  "steps": [
    {
      "step_id": 1,
      "action": "search",
      "tool": "search_documents",
      "parameters": {
        "query": "Python programming language",
        "max_results": 3
      },
      "expected_output": "lista de documentos relevantes",
      "validation_criteria": ["al menos 1 resultado", "relevancia > 0.7"]
    },
    {
      "step_id": 2,
      "action": "calculate",
      "tool": "math_calculator", 
      "parameters": {
        "expression": "len(results) * 100 / total_docs"
      },
      "expected_output": "porcentaje de relevancia",
      "validation_criteria": ["resultado numérico", "0 <= valor <= 100"]
    }
  ],
  "success_criteria": [
    "información recuperada exitosamente",
    "cálculos completados sin errores",
    "respuesta formateada correctamente"
  ],
  "max_retries": 3,
  "timeout_seconds": 30
}
```

### 2.3 Herramientas Disponibles

| Herramienta | Descripción | Parámetros | Salida |
|-------------|-------------|------------|---------|
| `search_documents` | Buscar en base de conocimiento | `query`, `max_results` | Lista de documentos |
| `math_calculator` | Calcular expresiones matemáticas | `expression` | Resultado numérico |
| `format_response` | Formatear texto en diferentes estilos | `content`, `format` | Texto formateado |
| `get_current_time` | Obtener fecha y hora actual | `timezone` | Timestamp |
| `weather_api` | Consultar clima (simulado) | `location` | Datos meteorológicos |

---

## 3. Implementación del Planner

### 3.1 Clase BasePlanner

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Any
from datetime import datetime
import json

class BasePlanner(ABC):
    """Clase base para planificadores de agentes"""
    
    def __init__(self, available_tools: List[str]):
        self.available_tools = set(available_tools)
        self.plan_counter = 0
    
    @abstractmethod
    def create_plan(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Crear plan de ejecución para el objetivo dado"""
        pass
    
    def validate_plan(self, plan: Dict[str, Any]) -> tuple[bool, str]:
        """Validar que el plan sea ejecutable"""
        # Verificar estructura básica
        required_fields = ['plan_id', 'goal', 'steps']
        for field in required_fields:
            if field not in plan:
                return False, f"Campo requerido faltante: {field}"
        
        # Verificar que los pasos usen herramientas disponibles
        for step in plan['steps']:
            tool = step.get('tool')
            if tool not in self.available_tools:
                return False, f"Herramienta no disponible: {tool}"
        
        return True, "Plan válido"
    
    def generate_plan_id(self) -> str:
        """Generar ID único para el plan"""
        self.plan_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"plan_{timestamp}_{self.plan_counter:03d}"
```

### 3.2 SimplePlanner Implementación

```python
class SimplePlanner(BasePlanner):
    """Planificador simple que descompone objetivos en pasos"""
    
    def __init__(self, available_tools: List[str]):
        super().__init__(available_tools)
        self.planning_templates = {
            'search_and_calculate': self._search_calculate_template,
            'information_retrieval': self._info_retrieval_template,
            'data_processing': self._data_processing_template
        }
    
    def create_plan(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Crear plan basado en análisis del objetivo"""
        
        # Analizar tipo de tarea
        task_type = self._analyze_goal(goal)
        
        # Seleccionar template apropiado
        if task_type in self.planning_templates:
            plan = self.planning_templates[task_type](goal, context)
        else:
            plan = self._default_template(goal, context)
        
        # Validar antes de retornar
        is_valid, message = self.validate_plan(plan)
        if not is_valid:
            raise ValueError(f"Plan inválido generado: {message}")
        
        return plan
    
    def _analyze_goal(self, goal: str) -> str:
        """Analizar el objetivo para determinar tipo de tarea"""
        goal_lower = goal.lower()
        
        if any(word in goal_lower for word in ['buscar', 'encontrar', 'información']):
            if any(word in goal_lower for word in ['calcular', 'contar', 'estadística']):
                return 'search_and_calculate'
            return 'information_retrieval'
        elif any(word in goal_lower for word in ['procesar', 'transformar', 'formatear']):
            return 'data_processing'
        else:
            return 'general'
```

---

## 4. Implementación del Executor

### 4.1 Clase BaseExecutor

```python
class BaseExecutor(ABC):
    """Clase base para ejecutores de planes"""
    
    def __init__(self, tool_registry: 'ToolRegistry'):
        self.tool_registry = tool_registry
        self.execution_log = []
    
    @abstractmethod
    def execute_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecutar plan completo"""
        pass
    
    def execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecutar un paso individual"""
        start_time = datetime.now()
        
        try:
            # Validar paso
            if not self._validate_step(step):
                raise ValueError(f"Paso inválido: {step}")
            
            # Ejecutar herramienta
            tool_name = step['tool']
            parameters = step.get('parameters', {})
            
            result = self.tool_registry.execute_tool(tool_name, parameters)
            
            # Registrar ejecución
            execution_record = {
                'step_id': step['step_id'],
                'tool': tool_name,
                'parameters': parameters,
                'result': result,
                'status': 'success',
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'timestamp': datetime.now().isoformat()
            }
            
            self.execution_log.append(execution_record)
            return execution_record
            
        except Exception as e:
            # Registrar error
            error_record = {
                'step_id': step['step_id'],
                'tool': step.get('tool', 'unknown'),
                'parameters': step.get('parameters', {}),
                'error': str(e),
                'status': 'error',
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'timestamp': datetime.now().isoformat()
            }
            
            self.execution_log.append(error_record)
            return error_record
```

---

## 5. Implementación del Critic

### 5.1 Clase BaseCritic

```python
class BaseCritic(ABC):
    """Clase base para críticos evaluadores"""
    
    def __init__(self, success_threshold: float = 0.8):
        self.success_threshold = success_threshold
        self.evaluation_history = []
    
    @abstractmethod
    def evaluate_execution(self, plan: Dict[str, Any], 
                          execution_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluar resultados de ejecución"""
        pass
    
    def evaluate_step(self, step: Dict[str, Any], 
                     result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluar resultado de un paso individual"""
        
        if result['status'] == 'error':
            return {
                'step_id': step['step_id'],
                'success': False,
                'score': 0.0,
                'issues': [f"Error en ejecución: {result.get('error', 'Unknown error')}"],
                'recommendation': 'retry'
            }
        
        # Validar criterios de éxito del paso
        validation_criteria = step.get('validation_criteria', [])
        passed_validations = 0
        issues = []
        
        for criterion in validation_criteria:
            passed, issue = self._validate_criterion(criterion, result)
            if passed:
                passed_validations += 1
            else:
                issues.append(issue)
        
        # Calcular score
        if validation_criteria:
            score = passed_validations / len(validation_criteria)
        else:
            score = 1.0 if result['status'] == 'success' else 0.0
        
        return {
            'step_id': step['step_id'],
            'success': score >= self.success_threshold,
            'score': score,
            'issues': issues,
            'recommendation': 'continue' if score >= self.success_threshold else 'retry'
        }
```

---

## 6. Laboratorio Práctico

### 6.1 Ejercicio 1: Implementar Planner Básico ✅

**Objetivo:** Crear un planificador que descomponga objetivos simples

**Archivo:** `labs/module-b/basic_planner.py`

**Estado**: ✅ **COMPLETADO** - SimplePlanner funcional con análisis de objetivos y generación de planes estructurados

```python
# ✅ IMPLEMENTADO: SimplePlanner con capacidad de:
# 1. ✅ Analizar objetivos de usuario
# 2. ✅ Generar planes estructurados con dataclasses
# 3. ✅ Validar herramientas disponibles
# 4. ✅ Manejar diferentes tipos de tareas con templates
```

### 6.2 Ejercicio 2: Tool Registry Seguro ✅

**Objetivo:** Implementar registro de herramientas con validación

**Archivo:** `labs/module-b/tool_registry.py`

**Estado**: ✅ **COMPLETADO** - ToolRegistry con circuit breakers, rate limiting y validación avanzada

```python
# ✅ IMPLEMENTADO: ToolRegistry que:
# 1. ✅ Registra herramientas de forma segura con validación
# 2. ✅ Valida parámetros con schemas y regex patterns
# 3. ✅ Maneja timeouts y límites con circuit breakers
# 4. ✅ Registra todas las llamadas para auditoría y métricas
```

### 6.3 Ejercicio 3: Agente PEC Completo ✅

**Objetivo:** Integrar todos los componentes en un agente funcional

**Archivo:** `labs/module-b/pec_agent.py`

**Estado**: ✅ **COMPLETADO** - PECAgent funcional con executor robusto y critic inteligente

```python
# ✅ IMPLEMENTADO: PECAgent que:
# 1. ✅ Combina Planner + Executor + Critic en flujo completo
# 2. ✅ Maneja ejecución step-by-step con retry logic
# 3. ✅ Implementa manejo avanzado de errores y recuperación
# 4. ✅ Genera respuestas estructuradas con evaluación de calidad
```

---

## 7. Casos de Uso Prácticos

### 7.1 Caso de Uso: "Analizar Documentación de Python"

**Entrada del Usuario:**
```
"Busca información sobre decoradores en Python, 
cuenta cuántos ejemplos encuentras y formatea 
un resumen ejecutivo"
```

**Plan Generado:**
```json
{
  "plan_id": "plan_20250816_143521_001",
  "goal": "Analizar documentación de Python sobre decoradores",
  "steps": [
    {
      "step_id": 1,
      "action": "search",
      "tool": "search_documents",
      "parameters": {
        "query": "Python decoradores ejemplos",
        "max_results": 5
      },
      "expected_output": "documentos sobre decoradores",
      "validation_criteria": ["al menos 2 resultados"]
    },
    {
      "step_id": 2,
      "action": "calculate",
      "tool": "math_calculator",
      "parameters": {
        "expression": "count_examples_in_results"
      },
      "expected_output": "número de ejemplos",
      "validation_criteria": ["resultado numérico >= 0"]
    },
    {
      "step_id": 3,
      "action": "format",
      "tool": "format_response",
      "parameters": {
        "content": "resumen_decoradores",
        "format": "executive_summary"
      },
      "expected_output": "resumen ejecutivo",
      "validation_criteria": ["longitud 100-300 palabras"]
    }
  ],
  "success_criteria": [
    "información recuperada",
    "ejemplos contabilizados",
    "resumen generado"
  ]
}
```

---

## 8. Patrones Avanzados de Manejo de Errores

### 8.1 Clasificación de Errores en Agentes

Los agentes pueden enfrentar diferentes tipos de errores que requieren estrategias específicas:

#### Errores Recuperables
```python
class RecoverableErrors:
    """Errores que el agente puede manejar automáticamente"""
    
    TIMEOUT_ERRORS = [
        "RequestTimeout",
        "APITimeout", 
        "NetworkTimeout"
    ]
    
    RATE_LIMIT_ERRORS = [
        "RateLimitExceeded",
        "TooManyRequests",
        "QuotaExceeded"
    ]
    
    TEMPORARY_FAILURES = [
        "ServiceUnavailable",
        "TemporaryServerError",
        "DatabaseLocked"
    ]

    @staticmethod
    def get_retry_strategy(error_type: str) -> Dict:
        """Obtener estrategia de retry según tipo de error"""
        strategies = {
            "timeout": {"max_retries": 3, "backoff": "exponential", "delay": 1},
            "rate_limit": {"max_retries": 5, "backoff": "linear", "delay": 60},
            "temporary": {"max_retries": 2, "backoff": "fixed", "delay": 5}
        }
        
        for category, error_list in {
            "timeout": RecoverableErrors.TIMEOUT_ERRORS,
            "rate_limit": RecoverableErrors.RATE_LIMIT_ERRORS,
            "temporary": RecoverableErrors.TEMPORARY_FAILURES
        }.items():
            if error_type in error_list:
                return strategies[category]
        
        return {"max_retries": 1, "backoff": "none", "delay": 0}
```

#### Errores No Recuperables
```python
class NonRecoverableErrors:
    """Errores que requieren intervención humana o cambio de plan"""
    
    AUTHENTICATION_ERRORS = [
        "InvalidCredentials",
        "AuthenticationFailed",
        "PermissionDenied"
    ]
    
    VALIDATION_ERRORS = [
        "InvalidInput",
        "SchemaValidationFailed",
        "DataFormatError"
    ]
    
    LOGIC_ERRORS = [
        "DivisionByZero",
        "IndexOutOfRange",
        "NullPointerException"
    ]
```

### 8.2 Estrategias de Recuperación Inteligente

#### Retry con Backoff Exponencial
```python
import time
import random
from typing import Callable, Any

class SmartRetryHandler:
    """Manejador inteligente de reintentos"""
    
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.attempt_count = 0
    
    def exponential_backoff(self, attempt: int, base_delay: float = 1.0) -> float:
        """Calcular delay con backoff exponencial + jitter"""
        delay = base_delay * (2 ** attempt)
        jitter = random.uniform(0, 0.1) * delay  # Añadir jitter para evitar thundering herd
        return delay + jitter
    
    def should_retry(self, error: Exception) -> bool:
        """Decidir si vale la pena reintentar"""
        error_name = type(error).__name__
        
        # No reintentar errores de autenticación o validación
        if error_name in NonRecoverableErrors.AUTHENTICATION_ERRORS + NonRecoverableErrors.VALIDATION_ERRORS:
            return False
        
        # Reintentar errores temporales y de red
        if error_name in RecoverableErrors.TIMEOUT_ERRORS + RecoverableErrors.TEMPORARY_FAILURES:
            return self.attempt_count < self.max_retries
        
        return False
    
    def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Ejecutar función con retry inteligente"""
        self.attempt_count = 0
        last_error = None
        
        while self.attempt_count <= self.max_retries:
            try:
                result = func(*args, **kwargs)
                return result
                
            except Exception as e:
                last_error = e
                self.attempt_count += 1
                
                if not self.should_retry(e):
                    raise e
                
                if self.attempt_count <= self.max_retries:
                    delay = self.exponential_backoff(self.attempt_count)
                    print(f"⚠️ Attempt {self.attempt_count} failed: {e}")
                    print(f"🔄 Retrying in {delay:.2f}s...")
                    time.sleep(delay)
        
        raise last_error
```

### 8.3 Estrategias de Fallback

```python
class FallbackStrategy:
    """Estrategias de fallback cuando fallan las herramientas principales"""
    
    def __init__(self):
        self.fallback_chains = {
            "search": [
                self.primary_search,
                self.backup_search, 
                self.cached_search,
                self.default_response
            ],
            "calculate": [
                self.primary_calculator,
                self.simple_calculator,
                self.manual_calculation
            ]
        }
    
    def execute_with_fallback(self, tool_type: str, *args, **kwargs):
        """Ejecutar con cadena de fallbacks"""
        fallback_chain = self.fallback_chains.get(tool_type, [])
        
        for i, fallback_func in enumerate(fallback_chain):
            try:
                result = fallback_func(*args, **kwargs)
                
                if i > 0:  # Si usamos fallback, loggear
                    print(f"⚠️ Used fallback #{i} for {tool_type}")
                
                return result
                
            except Exception as e:
                if i == len(fallback_chain) - 1:  # Último fallback
                    raise Exception(f"All fallbacks failed for {tool_type}: {e}")
                continue
    
    # Implementaciones de ejemplo
    def primary_search(self, query: str) -> str:
        # Búsqueda principal (puede fallar)
        raise Exception("Primary search API down")
    
    def backup_search(self, query: str) -> str:
        # Búsqueda backup
        return f"Backup search results for: {query}"
    
    def cached_search(self, query: str) -> str:
        # Resultados en caché
        return f"Cached results for: {query}"
    
    def default_response(self, query: str) -> str:
        # Respuesta por defecto
        return "Unable to perform search at this time. Please try again later."
```

### 8.4 Circuit Breaker Pattern Avanzado

```python
from enum import Enum
from dataclasses import dataclass
import time

class CircuitState(Enum):
    CLOSED = "closed"      # Funcionamiento normal
    OPEN = "open"          # Circuito abierto, falla rápida
    HALF_OPEN = "half_open"  # Probando si el servicio se recuperó

@dataclass
class CircuitConfig:
    failure_threshold: int = 5      # Fallos para abrir circuito
    success_threshold: int = 3      # Éxitos para cerrar circuito  
    timeout: float = 60.0          # Tiempo antes de probar HALF_OPEN

class AdvancedCircuitBreaker:
    """Circuit Breaker avanzado con ventana deslizante"""
    
    def __init__(self, config: CircuitConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.call_history = []  # Lista de (timestamp, success/failure)
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.last_failure_time = 0
    
    def call(self, func, *args, **kwargs):
        """Ejecutar función con circuit breaker avanzado"""
        current_time = time.time()
        
        # Si está OPEN, verificar si es tiempo de probar
        if self.state == CircuitState.OPEN:
            if current_time - self.last_failure_time > self.config.timeout:
                self.state = CircuitState.HALF_OPEN
                print("🔄 Circuit breaker moved to HALF_OPEN")
            else:
                raise Exception(f"Circuit breaker is OPEN. Try again in {self.config.timeout - (current_time - self.last_failure_time):.1f}s")
        
        try:
            result = func(*args, **kwargs)
            self._on_success(current_time)
            return result
            
        except Exception as e:
            self._on_failure(current_time)
            raise e
    
    def _on_success(self, timestamp: float):
        """Manejar llamada exitosa"""
        self.call_history.append((timestamp, True))
        self.consecutive_failures = 0
        self.consecutive_successes += 1
        
        # Si estamos en HALF_OPEN y tenemos suficientes éxitos, cerrar circuito
        if self.state == CircuitState.HALF_OPEN and self.consecutive_successes >= self.config.success_threshold:
            self.state = CircuitState.CLOSED
            print("✅ Circuit breaker CLOSED - Service recovered")
    
    def _on_failure(self, timestamp: float):
        """Manejar llamada fallida"""
        self.call_history.append((timestamp, False))
        self.consecutive_successes = 0
        self.consecutive_failures += 1
        self.last_failure_time = timestamp
        
        # Verificar si debemos abrir el circuito
        if self.consecutive_failures >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            print(f"❌ Circuit breaker OPENED after {self.consecutive_failures} failures")
        
        # Si estamos en HALF_OPEN y falla, volver a OPEN
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            print("❌ Circuit breaker back to OPEN - Service still failing")
```

## 8bis. Patrones de Manejo de Errores (Básicos)

### 8bis.1 Estrategias de Recuperación

```python
class ErrorRecoveryStrategy:
    """Estrategias para manejar errores en agentes"""
    
    RETRY_STRATEGIES = {
        'tool_timeout': 'retry_with_longer_timeout',
        'invalid_parameters': 'retry_with_corrected_params',
        'resource_unavailable': 'retry_with_alternative_tool',
        'partial_failure': 'continue_with_partial_results',
        'critical_error': 'abort_and_report'
    }
    
    def handle_error(self, error_type: str, context: Dict) -> str:
        """Determinar estrategia de recuperación"""
        return self.RETRY_STRATEGIES.get(error_type, 'abort_and_report')
```

### 8bis.2 Circuit Breaker Pattern (Básico)

```python
class CircuitBreaker:
    """Implementación de circuit breaker para herramientas"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        """Ejecutar función con circuit breaker"""
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.timeout:
                self.state = 'HALF_OPEN'
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise e
```

---

## 9. Métricas y Evaluación

### 9.1 Métricas del Agente

- **Tasa de éxito de planes**: % de planes ejecutados exitosamente
- **Tiempo promedio de ejecución**: Latencia por plan
- **Uso de herramientas**: Frecuencia y eficiencia
- **Tasa de retry**: % de pasos que requieren reintento
- **Precisión del Critic**: % de evaluaciones correctas

### 9.2 Dashboard de Métricas

```python
def generate_agent_metrics(execution_logs: List[Dict]) -> Dict:
    """Generar métricas del agente"""
    total_plans = len(execution_logs)
    successful_plans = sum(1 for log in execution_logs if log['status'] == 'success')
    
    return {
        'success_rate': successful_plans / total_plans if total_plans > 0 else 0,
        'avg_execution_time': calculate_avg_time(execution_logs),
        'most_used_tools': get_tool_usage_stats(execution_logs),
        'error_patterns': analyze_error_patterns(execution_logs)
    }
```

---

## 10. Quiz de Evaluación

### Pregunta 1
¿Cuál es la responsabilidad principal del componente Critic en el patrón PEC?

- [ ] a) Generar planes de ejecución
- [ ] b) Ejecutar herramientas externas
- [x] c) Evaluar resultados y decidir próximos pasos
- [ ] d) Validar parámetros de entrada

### Pregunta 2
¿Qué ventaja principal ofrece el patrón PEC sobre un enfoque monolítico?

- [ ] a) Mayor velocidad de ejecución
- [x] b) Mejor modularidad y mantenibilidad
- [ ] c) Menor uso de memoria
- [ ] d) Compatibilidad con más APIs

### Pregunta 3
¿Cuándo debe un agente implementar circuit breaker pattern?

- [ ] a) Siempre, en todas las herramientas
- [ ] b) Nunca, es innecesario
- [x] c) Para herramientas externas propensas a fallos
- [ ] d) Solo en entornos de producción

---

## 11. Recursos Adicionales

### 11.1 Plantillas Descargables
- 📄 [`pec-agent-template.py`](../../templates/pec-agent-template.py) - Template base de agente
- 📄 [`tool-registry-config.json`](../../templates/tool-registry-config.json) - Configuración de herramientas
- 📄 [`error-handling-patterns.py`](../../templates/error-handling-patterns.py) - Patrones de manejo de errores

### 11.2 Lecturas Recomendadas
- [ReAct: Reasoning and Acting with Language Models](https://arxiv.org/abs/2210.03629)
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)
- [Circuit Breaker Pattern - Martin Fowler](https://martinfowler.com/bliki/CircuitBreaker.html)

---

## ✅ Siguiente Paso

Una vez completado este módulo, estarás listo para **Módulo C: RAG Básico con Citas**, donde implementarás recuperación de información y generación aumentada.

**🎯 Meta del próximo módulo:** Crear un sistema RAG que pueda buscar, recuperar y citar información de forma confiable.

---

*¿Tienes dudas sobre el patrón PEC? Revisa los [ejemplos completos](../examples/) o consulta el [foro de estudiantes](../community/).*
