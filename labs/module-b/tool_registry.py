"""
Laboratorio Módulo B: Tool Registry Seguro
Sistema de registro y ejecución de herramientas con validación y seguridad
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
import json
import time
import threading
from dataclasses import dataclass, asdict
from enum import Enum
import re


class ToolStatus(Enum):
    """Estados posibles de una herramienta"""
    AVAILABLE = "available"
    DISABLED = "disabled"
    ERROR = "error"
    MAINTENANCE = "maintenance"


@dataclass
class ToolCall:
    """Registro de una llamada a herramienta"""
    tool_name: str
    parameters: Dict[str, Any]
    result: Any
    status: str
    execution_time: float
    timestamp: str
    error_message: Optional[str] = None


@dataclass
class ToolDefinition:
    """Definición de una herramienta"""
    name: str
    description: str
    parameters_schema: Dict[str, Any]
    function: Callable
    timeout_seconds: int = 30
    max_retries: int = 3
    status: ToolStatus = ToolStatus.AVAILABLE
    rate_limit_per_minute: int = 60


class CircuitBreaker:
    """
    Implementación de Circuit Breaker pattern para herramientas
    
    Estados:
    - CLOSED: Funcionamiento normal
    - OPEN: Bloqueado por errores
    - HALF_OPEN: Probando recuperación
    """
    
    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'
        self.success_count = 0
        self.lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        TODO: Ejecutar función con circuit breaker
        
        Debe manejar los tres estados:
        - CLOSED: Ejecutar normalmente
        - OPEN: Rechazar llamadas hasta timeout
        - HALF_OPEN: Permitir llamada de prueba
        """
        with self.lock:
            current_time = time.time()
            
            if self.state == 'OPEN':
                if self.last_failure_time and (current_time - self.last_failure_time) > self.timeout_seconds:
                    self.state = 'HALF_OPEN'
                    self.success_count = 0
                else:
                    raise Exception(f"Circuit breaker is OPEN. Tool temporarily unavailable.")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
                
            except Exception as e:
                self._on_failure(current_time)
                raise e
    
    def _on_success(self):
        """Manejar ejecución exitosa"""
        if self.state == 'HALF_OPEN':
            self.success_count += 1
            if self.success_count >= 2:  # Necesitamos al menos 2 éxitos para cerrar
                self.state = 'CLOSED'
                self.failure_count = 0
        elif self.state == 'CLOSED':
            self.failure_count = max(0, self.failure_count - 1)
    
    def _on_failure(self, failure_time: float):
        """Manejar fallo en ejecución"""
        self.failure_count += 1
        self.last_failure_time = failure_time
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
        elif self.state == 'HALF_OPEN':
            self.state = 'OPEN'


class RateLimiter:
    """Limitador de velocidad para herramientas"""
    
    def __init__(self, max_calls_per_minute: int):
        self.max_calls_per_minute = max_calls_per_minute
        self.call_times = []
        self.lock = threading.Lock()
    
    def is_allowed(self) -> bool:
        """
        TODO: Verificar si se permite una nueva llamada
        
        Debe mantener ventana deslizante de 1 minuto
        """
        with self.lock:
            current_time = time.time()
            cutoff_time = current_time - 60  # 1 minuto atrás
            
            # Remover llamadas antiguas
            self.call_times = [t for t in self.call_times if t > cutoff_time]
            
            # Verificar si se puede hacer nueva llamada
            if len(self.call_times) < self.max_calls_per_minute:
                self.call_times.append(current_time)
                return True
            
            return False


class ParameterValidator:
    """Validador de parámetros de herramientas"""
    
    @staticmethod
    def validate_parameters(parameters: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, str]:
        """
        TODO: Validar parámetros contra schema definido
        
        Schema format:
        {
            "param_name": {
                "type": "string|number|boolean|array|object",
                "required": bool,
                "min_length": int,
                "max_length": int,
                "pattern": "regex",
                "min_value": number,
                "max_value": number
            }
        }
        """
        errors = []
        
        # Verificar parámetros requeridos
        for param_name, param_config in schema.items():
            is_required = param_config.get('required', False)
            
            if is_required and param_name not in parameters:
                errors.append(f"Parámetro requerido faltante: {param_name}")
                continue
            
            if param_name not in parameters:
                continue
            
            value = parameters[param_name]
            param_type = param_config.get('type', 'string')
            
            # Validar tipo
            type_valid, type_error = ParameterValidator._validate_type(value, param_type)
            if not type_valid:
                errors.append(f"Parámetro {param_name}: {type_error}")
                continue
            
            # Validaciones específicas por tipo
            if param_type == 'string':
                string_valid, string_error = ParameterValidator._validate_string(value, param_config)
                if not string_valid:
                    errors.append(f"Parámetro {param_name}: {string_error}")
            
            elif param_type == 'number':
                number_valid, number_error = ParameterValidator._validate_number(value, param_config)
                if not number_valid:
                    errors.append(f"Parámetro {param_name}: {number_error}")
        
        if errors:
            return False, "; ".join(errors)
        
        return True, "Parámetros válidos"
    
    @staticmethod
    def _validate_type(value: Any, expected_type: str) -> Tuple[bool, str]:
        """Validar tipo de datos"""
        type_map = {
            'string': str,
            'number': (int, float),
            'boolean': bool,
            'array': list,
            'object': dict
        }
        
        expected_python_type = type_map.get(expected_type)
        if expected_python_type is None:
            return False, f"Tipo no soportado: {expected_type}"
        
        if not isinstance(value, expected_python_type):
            return False, f"Se esperaba {expected_type}, se recibió {type(value).__name__}"
        
        return True, ""
    
    @staticmethod
    def _validate_string(value: str, config: Dict[str, Any]) -> Tuple[bool, str]:
        """Validar parámetros de string"""
        min_length = config.get('min_length')
        max_length = config.get('max_length')
        pattern = config.get('pattern')
        
        if min_length is not None and len(value) < min_length:
            return False, f"Longitud mínima: {min_length}, actual: {len(value)}"
        
        if max_length is not None and len(value) > max_length:
            return False, f"Longitud máxima: {max_length}, actual: {len(value)}"
        
        if pattern and not re.match(pattern, value):
            return False, f"No coincide con patrón requerido: {pattern}"
        
        return True, ""
    
    @staticmethod
    def _validate_number(value: float, config: Dict[str, Any]) -> Tuple[bool, str]:
        """Validar parámetros numéricos"""
        min_value = config.get('min_value')
        max_value = config.get('max_value')
        
        if min_value is not None and value < min_value:
            return False, f"Valor mínimo: {min_value}, actual: {value}"
        
        if max_value is not None and value > max_value:
            return False, f"Valor máximo: {max_value}, actual: {value}"
        
        return True, ""


class ToolRegistry:
    """
    Registro central de herramientas con validación, rate limiting y circuit breakers
    
    TODO: Implementar sistema completo de gestión de herramientas
    """
    
    def __init__(self):
        self.tools: Dict[str, ToolDefinition] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.call_history: List[ToolCall] = []
        self.validator = ParameterValidator()
    
    def register_tool(self, tool_def: ToolDefinition) -> bool:
        """
        TODO: Registrar nueva herramienta en el registry
        
        Debe:
        1. Validar definición de herramienta
        2. Crear circuit breaker y rate limiter
        3. Agregar al registro
        """
        # Validar definición
        if not tool_def.name or not tool_def.function:
            raise ValueError("Herramienta debe tener nombre y función")
        
        if tool_def.name in self.tools:
            raise ValueError(f"Herramienta {tool_def.name} ya está registrada")
        
        # Registrar herramienta
        self.tools[tool_def.name] = tool_def
        
        # Crear circuit breaker
        self.circuit_breakers[tool_def.name] = CircuitBreaker(
            failure_threshold=5,
            timeout_seconds=60
        )
        
        # Crear rate limiter
        self.rate_limiters[tool_def.name] = RateLimiter(
            tool_def.rate_limit_per_minute
        )
        
        return True
    
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """
        TODO: Ejecutar herramienta con todas las validaciones
        
        Proceso:
        1. Verificar que herramienta existe y está disponible
        2. Validar parámetros
        3. Verificar rate limiting
        4. Ejecutar con circuit breaker
        5. Registrar llamada
        """
        start_time = time.time()
        
        try:
            # Verificar herramienta existe
            if tool_name not in self.tools:
                raise ValueError(f"Herramienta no encontrada: {tool_name}")
            
            tool_def = self.tools[tool_name]
            
            # Verificar estado de herramienta
            if tool_def.status != ToolStatus.AVAILABLE:
                raise ValueError(f"Herramienta {tool_name} no está disponible: {tool_def.status.value}")
            
            # Validar parámetros
            is_valid, validation_message = self.validator.validate_parameters(
                parameters, tool_def.parameters_schema
            )
            if not is_valid:
                raise ValueError(f"Parámetros inválidos: {validation_message}")
            
            # Verificar rate limiting
            rate_limiter = self.rate_limiters[tool_name]
            if not rate_limiter.is_allowed():
                raise ValueError(f"Rate limit excedido para {tool_name}")
            
            # Ejecutar con circuit breaker
            circuit_breaker = self.circuit_breakers[tool_name]
            result = circuit_breaker.call(tool_def.function, **parameters)
            
            # Registrar llamada exitosa
            execution_time = time.time() - start_time
            call_record = ToolCall(
                tool_name=tool_name,
                parameters=parameters,
                result=result,
                status="success",
                execution_time=execution_time,
                timestamp=datetime.now().isoformat()
            )
            
            self.call_history.append(call_record)
            return result
            
        except Exception as e:
            # Registrar llamada fallida
            execution_time = time.time() - start_time
            call_record = ToolCall(
                tool_name=tool_name,
                parameters=parameters,
                result=None,
                status="error",
                execution_time=execution_time,
                timestamp=datetime.now().isoformat(),
                error_message=str(e)
            )
            
            self.call_history.append(call_record)
            raise e
    
    def get_tool_status(self, tool_name: str) -> Dict[str, Any]:
        """Obtener estado actual de una herramienta"""
        if tool_name not in self.tools:
            return {"error": "Herramienta no encontrada"}
        
        tool_def = self.tools[tool_name]
        circuit_breaker = self.circuit_breakers[tool_name]
        
        # Estadísticas de uso
        tool_calls = [call for call in self.call_history if call.tool_name == tool_name]
        success_calls = [call for call in tool_calls if call.status == "success"]
        
        return {
            "name": tool_name,
            "status": tool_def.status.value,
            "circuit_breaker_state": circuit_breaker.state,
            "total_calls": len(tool_calls),
            "successful_calls": len(success_calls),
            "success_rate": len(success_calls) / len(tool_calls) if tool_calls else 0,
            "average_execution_time": sum(call.execution_time for call in success_calls) / len(success_calls) if success_calls else 0
        }
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas generales del registry"""
        total_calls = len(self.call_history)
        successful_calls = len([call for call in self.call_history if call.status == "success"])
        
        # Herramientas más usadas
        tool_usage = {}
        for call in self.call_history:
            tool_usage[call.tool_name] = tool_usage.get(call.tool_name, 0) + 1
        
        most_used_tool = max(tool_usage.items(), key=lambda x: x[1]) if tool_usage else None
        
        return {
            "total_tools": len(self.tools),
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "success_rate": successful_calls / total_calls if total_calls > 0 else 0,
            "most_used_tool": most_used_tool[0] if most_used_tool else None,
            "tool_usage": tool_usage
        }


# Implementaciones de herramientas de ejemplo
def search_documents_impl(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Implementación simulada de búsqueda de documentos"""
    # Simular búsqueda en dataset
    time.sleep(0.1)  # Simular latencia
    
    mock_results = [
        {"id": "doc1", "title": "Python Basics", "content": "Python is a programming language...", "score": 0.95},
        {"id": "doc2", "title": "Advanced Python", "content": "Advanced concepts in Python...", "score": 0.87},
        {"id": "doc3", "title": "Python Libraries", "content": "Popular Python libraries...", "score": 0.82},
    ]
    
    # Filtrar por query (simplificado)
    if "python" in query.lower():
        return mock_results[:max_results]
    else:
        return mock_results[:1]  # Resultado menos relevante


def math_calculator_impl(expression: str) -> float:
    """Implementación segura de calculadora matemática"""
    # Validar expresión solo contiene caracteres seguros
    safe_chars = set('0123456789+-*/.()')
    if not all(c in safe_chars or c.isspace() for c in expression):
        raise ValueError("Expresión contiene caracteres no permitidos")
    
    try:
        # Evaluar de forma segura (en producción usar parser matemático)
        result = eval(expression)
        if isinstance(result, (int, float)):
            return float(result)
        else:
            raise ValueError("El resultado debe ser numérico")
    except Exception as e:
        raise ValueError(f"Error evaluando expresión: {str(e)}")


def format_response_impl(content: str, format: str = "summary") -> str:
    """Implementación de formateador de respuestas"""
    if format == "summary":
        return f"📋 Resumen:\n\n{content}\n\n---\nGenerado automáticamente"
    elif format == "json":
        return json.dumps({"content": content, "generated_at": datetime.now().isoformat()})
    elif format == "markdown":
        return f"# Respuesta\n\n{content}\n\n*Generado el {datetime.now().strftime('%Y-%m-%d %H:%M')}*"
    else:
        return content


def get_current_time_impl(timezone: str = "UTC") -> str:
    """Implementación de obtención de hora actual"""
    return datetime.now().isoformat()


def test_tool_registry():
    """Función de prueba para el ToolRegistry"""
    print("=== PRUEBAS DEL TOOL REGISTRY ===\n")
    
    registry = ToolRegistry()
    
    # Registrar herramientas de prueba
    tools_to_register = [
        ToolDefinition(
            name="search_documents",
            description="Buscar documentos en base de conocimiento",
            parameters_schema={
                "query": {"type": "string", "required": True, "min_length": 1, "max_length": 200},
                "max_results": {"type": "number", "required": False, "min_value": 1, "max_value": 20}
            },
            function=search_documents_impl,
            rate_limit_per_minute=30
        ),
        ToolDefinition(
            name="math_calculator",
            description="Realizar cálculos matemáticos",
            parameters_schema={
                "expression": {"type": "string", "required": True, "pattern": r"^[0-9+\-*/.() ]+$"}
            },
            function=math_calculator_impl,
            rate_limit_per_minute=60
        ),
        ToolDefinition(
            name="format_response", 
            description="Formatear respuestas en diferentes estilos",
            parameters_schema={
                "content": {"type": "string", "required": True, "min_length": 1},
                "format": {"type": "string", "required": False}
            },
            function=format_response_impl,
            rate_limit_per_minute=100
        )
    ]
    
    # Registrar herramientas
    print("--- Registrando Herramientas ---")
    for tool_def in tools_to_register:
        try:
            registry.register_tool(tool_def)
            print(f"✅ {tool_def.name} registrada exitosamente")
        except Exception as e:
            print(f"❌ Error registrando {tool_def.name}: {str(e)}")
    
    # Casos de prueba
    test_cases = [
        {
            "tool": "search_documents",
            "params": {"query": "python programming", "max_results": 3},
            "should_succeed": True
        },
        {
            "tool": "math_calculator",
            "params": {"expression": "2 + 2 * 3"},
            "should_succeed": True
        },
        {
            "tool": "format_response",
            "params": {"content": "Test content", "format": "markdown"},
            "should_succeed": True
        },
        {
            "tool": "search_documents",
            "params": {"query": ""},  # Query vacío - debe fallar
            "should_succeed": False
        },
        {
            "tool": "math_calculator",
            "params": {"expression": "eval('malicious')"},  # Expresión insegura
            "should_succeed": False
        },
        {
            "tool": "nonexistent_tool",
            "params": {},
            "should_succeed": False
        }
    ]
    
    print("\n--- Ejecutando Casos de Prueba ---")
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nCaso {i}: {test_case['tool']}")
        print(f"Parámetros: {test_case['params']}")
        
        try:
            result = registry.execute_tool(test_case['tool'], test_case['params'])
            success = True
            print(f"✅ Resultado: {str(result)[:100]}{'...' if len(str(result)) > 100 else ''}")
        except Exception as e:
            success = False
            print(f"❌ Error: {str(e)}")
        
        expected = test_case['should_succeed']
        if success == expected:
            print(f"✅ Comportamiento esperado")
        else:
            print(f"❌ Comportamiento inesperado (esperaba {'éxito' if expected else 'fallo'})")
    
    # Mostrar estadísticas
    print("\n--- Estadísticas del Registry ---")
    stats = registry.get_registry_stats()
    print(f"Herramientas registradas: {stats['total_tools']}")
    print(f"Llamadas totales: {stats['total_calls']}")
    print(f"Tasa de éxito: {stats['success_rate']:.2%}")
    print(f"Herramienta más usada: {stats['most_used_tool']}")
    
    # Estado individual de herramientas
    print("\n--- Estado de Herramientas ---")
    for tool_name in registry.tools.keys():
        status = registry.get_tool_status(tool_name)
        print(f"{tool_name}:")
        print(f"  Estado: {status['status']}")
        print(f"  Circuit Breaker: {status['circuit_breaker_state']}")
        print(f"  Llamadas: {status['total_calls']} (éxito: {status['successful_calls']})")
        print(f"  Tasa éxito: {status['success_rate']:.2%}")
        print(f"  Tiempo promedio: {status['average_execution_time']:.3f}s")


if __name__ == "__main__":
    test_tool_registry()
    
    print("\n=== EJERCICIOS ADICIONALES ===")
    print("1. Implementa herramientas adicionales (weather_api, file_processor)")
    print("2. Agrega persistencia del historial de llamadas")
    print("3. Implementa alertas automáticas por fallos frecuentes")
    print("4. Crea dashboard web para monitoreo en tiempo real")
    print("5. Agrega métricas de performance y optimización automática")
