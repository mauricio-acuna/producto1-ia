"""
Laboratorio Módulo A: Validación de Seguridad
Ejercicio para implementar filtros y validadores siguiendo safety.min.yaml
"""

import re
import yaml
import time
from typing import Tuple, List, Dict, Any
from datetime import datetime, timedelta
import hashlib


class SecurityValidator:
    """Validador de seguridad para entradas de usuario"""
    
    def __init__(self, config_path: str = None):
        """
        Inicializar validador con configuración de seguridad
        
        Args:
            config_path: Ruta al archivo safety.min.yaml
        """
        self.max_length = 1000
        self.allowed_characters = "alphanumeric_extended"
        self.blocked_patterns = [
            r'eval\(',
            r'exec\(',
            r'__import__',
            r'subprocess',
            r'os\.',
            r'system\(',
            r'shell=True'
        ]
        
        # Cargar configuración si existe
        if config_path:
            self.load_config(config_path)
        
        # Rate limiting simple
        self.request_history = {}
        self.max_requests_per_minute = 10
        
    def load_config(self, config_path: str):
        """Cargar configuración desde archivo YAML"""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                
            # Cargar configuración de validación de entrada
            if 'input_validation' in config:
                input_config = config['input_validation']
                self.max_length = input_config.get('max_length', self.max_length)
                self.allowed_characters = input_config.get('allowed_characters', self.allowed_characters)
                self.blocked_patterns = input_config.get('blocked_patterns', self.blocked_patterns)
            
            # Cargar configuración de límites
            if 'resource_limits' in config:
                limits = config['resource_limits']
                self.max_requests_per_minute = limits.get('max_requests_per_minute', self.max_requests_per_minute)
                
        except Exception as e:
            print(f"Warning: No se pudo cargar configuración: {e}")
    
    def validate_user_input(self, input_text: str, user_id: str = "anonymous") -> Tuple[bool, str]:
        """
        TODO: Validar entrada del usuario según reglas de seguridad
        
        Args:
            input_text: Texto de entrada del usuario
            user_id: Identificador del usuario (para rate limiting)
            
        Returns:
            Tupla (es_válida, mensaje_detalle)
        """
        
        # 1. Verificar rate limiting
        if not self._check_rate_limit(user_id):
            return False, "Demasiadas solicitudes. Espera un momento antes de intentar de nuevo."
        
        # 2. Verificar longitud
        if len(input_text) > self.max_length:
            return False, f"Entrada demasiado larga. Máximo {self.max_length} caracteres."
        
        # 3. Verificar caracteres permitidos
        if not self._check_allowed_characters(input_text):
            return False, "La entrada contiene caracteres no permitidos."
        
        # 4. Verificar patrones bloqueados
        blocked_pattern = self._check_blocked_patterns(input_text)
        if blocked_pattern:
            return False, f"Patrón de seguridad detectado: {blocked_pattern}"
        
        # 5. Verificar inyección de código
        if self._detect_code_injection(input_text):
            return False, "Posible intento de inyección de código detectado."
        
        # 6. Verificar información sensible
        if self._detect_sensitive_info(input_text):
            return False, "La entrada puede contener información sensible."
        
        return True, "Entrada válida y segura."
    
    def _check_rate_limit(self, user_id: str) -> bool:
        """Verificar límites de velocidad de solicitudes"""
        now = datetime.now()
        
        # Limpiar historial antiguo
        cutoff = now - timedelta(minutes=1)
        if user_id in self.request_history:
            self.request_history[user_id] = [
                timestamp for timestamp in self.request_history[user_id]
                if timestamp > cutoff
            ]
        
        # Verificar límite
        user_requests = self.request_history.get(user_id, [])
        if len(user_requests) >= self.max_requests_per_minute:
            return False
        
        # Registrar nueva solicitud
        if user_id not in self.request_history:
            self.request_history[user_id] = []
        self.request_history[user_id].append(now)
        
        return True
    
    def _check_allowed_characters(self, text: str) -> bool:
        """Verificar que solo se usen caracteres permitidos"""
        if self.allowed_characters == "alphanumeric_extended":
            # Permitir letras, números, espacios y puntuación básica
            allowed_pattern = r'^[a-zA-Z0-9\s\.,;:!?\-_áéíóúñüÁÉÍÓÚÑÜ]+$'
            return bool(re.match(allowed_pattern, text))
        
        return True  # Por defecto, permitir todo si no está especificado
    
    def _check_blocked_patterns(self, text: str) -> str:
        """Verificar patrones bloqueados y retornar el primero encontrado"""
        for pattern in self.blocked_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return pattern
        return ""
    
    def _detect_code_injection(self, text: str) -> bool:
        """Detectar posibles intentos de inyección de código"""
        # Patrones comunes de inyección
        injection_patterns = [
            r'<script',
            r'javascript:',
            r'onload=',
            r'onerror=',
            r'src=.*\.js',
            r'\{\{.*\}\}',  # Template injection
            r'\$\{.*\}',    # Expression injection
            r'`.*`',        # Template literals
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _detect_sensitive_info(self, text: str) -> bool:
        """Detectar información potencialmente sensible"""
        # Patrones para detectar información sensible
        sensitive_patterns = [
            r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Números de tarjeta
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Emails
            r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b',  # SSN format
            r'password\s*[:=]\s*\S+',  # Passwords
            r'api[_-]?key\s*[:=]\s*\S+',  # API keys
            r'token\s*[:=]\s*\S+',  # Tokens
        ]
        
        for pattern in sensitive_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False


class ToolRegistry:
    """Registro seguro de herramientas disponibles"""
    
    def __init__(self):
        self.allowed_tools = {
            'search_documents',
            'get_weather',
            'calculate',
            'format_response',
            'get_time',
            'translate_text'
        }
        
        self.blocked_tools = {
            'execute_code',
            'file_system_access',
            'network_request',
            'database_write',
            'system_command'
        }
        
        self.tool_usage_log = []
    
    def is_tool_allowed(self, tool_name: str) -> bool:
        """
        TODO: Verificar si una herramienta está permitida
        
        Args:
            tool_name: Nombre de la herramienta
            
        Returns:
            True si está permitida, False en caso contrario
        """
        # Verificar que esté en la lista permitida y no en la bloqueada
        is_allowed = (tool_name in self.allowed_tools and 
                     tool_name not in self.blocked_tools)
        
        # Registrar intento de uso
        self.tool_usage_log.append({
            'tool': tool_name,
            'allowed': is_allowed,
            'timestamp': datetime.now().isoformat()
        })
        
        return is_allowed
    
    def validate_tool_parameters(self, tool_name: str, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """Validar parámetros de herramientas"""
        if not self.is_tool_allowed(tool_name):
            return False, f"Herramienta '{tool_name}' no está permitida"
        
        # Validaciones específicas por herramienta
        validators = {
            'search_documents': self._validate_search_params,
            'get_weather': self._validate_weather_params,
            'calculate': self._validate_calculate_params,
        }
        
        if tool_name in validators:
            return validators[tool_name](parameters)
        
        return True, "Parámetros válidos"
    
    def _validate_search_params(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Validar parámetros de búsqueda"""
        if 'query' not in params:
            return False, "Parámetro 'query' requerido"
        
        query = params['query']
        if len(query) > 200:
            return False, "Query demasiado larga"
        
        if not isinstance(query, str):
            return False, "Query debe ser string"
        
        return True, "Parámetros de búsqueda válidos"
    
    def _validate_weather_params(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Validar parámetros del clima"""
        if 'location' not in params:
            return False, "Parámetro 'location' requerido"
        
        location = params['location']
        if not isinstance(location, str) or len(location.strip()) == 0:
            return False, "Location debe ser un string no vacío"
        
        return True, "Parámetros de clima válidos"
    
    def _validate_calculate_params(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Validar parámetros de cálculo"""
        if 'expression' not in params:
            return False, "Parámetro 'expression' requerido"
        
        expression = params['expression']
        
        # Solo permitir caracteres seguros para matemáticas
        safe_chars = set('0123456789+-*/.() ')
        if not all(c in safe_chars for c in expression):
            return False, "Expresión contiene caracteres no permitidos"
        
        return True, "Parámetros de cálculo válidos"


def test_security_validation():
    """Función de prueba para validar diferentes casos de seguridad"""
    print("=== PRUEBAS DE VALIDACIÓN DE SEGURIDAD ===\n")
    
    validator = SecurityValidator()
    tool_registry = ToolRegistry()
    
    # Casos de prueba para validación de entrada
    test_inputs = [
        ("Hola, ¿cómo estás?", True, "Entrada normal y segura"),
        ("eval('malicious code')", False, "Intento de ejecución de código"),
        ("Mi email es usuario@ejemplo.com", False, "Contiene información sensible"),
        ("x" * 1500, False, "Entrada demasiado larga"),
        ("Busca información sobre Python", True, "Consulta legítima"),
        ("<script>alert('xss')</script>", False, "Intento de XSS"),
        ("subprocess.call(['rm', '-rf', '/'])", False, "Comando del sistema"),
        ("¿Cuál es la capital de España?", True, "Pregunta normal"),
    ]
    
    print("--- Validación de Entradas ---")
    for i, (input_text, expected_valid, description) in enumerate(test_inputs, 1):
        is_valid, message = validator.validate_user_input(input_text, f"user_{i}")
        status = "✅ VÁLIDA" if is_valid else "❌ INVÁLIDA"
        result = "✅ CORRECTO" if is_valid == expected_valid else "❌ ERROR"
        
        print(f"{i}. {description}")
        print(f"   Entrada: '{input_text[:50]}{'...' if len(input_text) > 50 else ''}'")
        print(f"   Resultado: {status} - {message}")
        print(f"   Evaluación: {result}\n")
    
    # Casos de prueba para herramientas
    print("--- Validación de Herramientas ---")
    tool_tests = [
        ("search_documents", {"query": "machine learning"}, True),
        ("execute_code", {"code": "print('hello')"}, False),
        ("get_weather", {"location": "Madrid"}, True),
        ("file_system_access", {"path": "/etc/passwd"}, False),
        ("calculate", {"expression": "2 + 2"}, True),
        ("calculate", {"expression": "eval('malicious')"}, False),
    ]
    
    for tool_name, params, expected_allowed in tool_tests:
        is_allowed = tool_registry.is_tool_allowed(tool_name)
        params_valid, params_message = tool_registry.validate_tool_parameters(tool_name, params)
        
        overall_valid = is_allowed and params_valid
        status = "✅ PERMITIDA" if overall_valid else "❌ BLOQUEADA"
        result = "✅ CORRECTO" if (overall_valid == expected_allowed) else "❌ ERROR"
        
        print(f"Herramienta: {tool_name}")
        print(f"Parámetros: {params}")
        print(f"Resultado: {status}")
        print(f"Detalle: {params_message if not is_allowed else params_message}")
        print(f"Evaluación: {result}\n")


def interactive_security_test():
    """Herramienta interactiva para probar validaciones"""
    print("=== PROBADOR INTERACTIVO DE SEGURIDAD ===\n")
    
    validator = SecurityValidator()
    tool_registry = ToolRegistry()
    
    while True:
        print("Opciones:")
        print("1. Probar validación de entrada")
        print("2. Probar herramienta y parámetros")
        print("3. Ver log de herramientas")
        print("4. Probar rate limiting")
        print("5. Salir")
        
        choice = input("\nElige una opción (1-5): ").strip()
        
        if choice == "1":
            text = input("Introduce el texto a validar: ")
            user_id = input("ID de usuario (opcional): ") or "test_user"
            
            is_valid, message = validator.validate_user_input(text, user_id)
            status = "✅ VÁLIDA" if is_valid else "❌ INVÁLIDA"
            print(f"\nResultado: {status}")
            print(f"Mensaje: {message}")
        
        elif choice == "2":
            tool_name = input("Nombre de la herramienta: ")
            print("Introduce parámetros (formato: clave=valor, uno por línea, línea vacía para terminar):")
            
            params = {}
            while True:
                line = input().strip()
                if not line:
                    break
                if '=' in line:
                    key, value = line.split('=', 1)
                    params[key.strip()] = value.strip()
            
            is_allowed = tool_registry.is_tool_allowed(tool_name)
            params_valid, params_message = tool_registry.validate_tool_parameters(tool_name, params)
            
            print(f"\nHerramienta permitida: {'✅ SÍ' if is_allowed else '❌ NO'}")
            print(f"Parámetros válidos: {'✅ SÍ' if params_valid else '❌ NO'}")
            print(f"Mensaje: {params_message}")
        
        elif choice == "3":
            print("\n--- Log de Uso de Herramientas ---")
            for entry in tool_registry.tool_usage_log[-10:]:  # Últimas 10 entradas
                status = "✅" if entry['allowed'] else "❌"
                print(f"{entry['timestamp']}: {status} {entry['tool']}")
        
        elif choice == "4":
            user_id = input("ID de usuario para probar rate limiting: ")
            attempts = int(input("Número de intentos a simular: ") or "12")
            
            print(f"\nSimulando {attempts} solicitudes...")
            for i in range(attempts):
                is_valid, message = validator.validate_user_input(f"test {i}", user_id)
                if not is_valid and "solicitudes" in message:
                    print(f"Solicitud {i+1}: ❌ Rate limit alcanzado")
                    break
                else:
                    print(f"Solicitud {i+1}: ✅ Permitida")
        
        elif choice == "5":
            break
        
        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    # TODO: Ejecuta las pruebas y experimenta con el validador
    test_security_validation()
    
    print("\n¿Quieres probar el validador interactivo? (y/n): ", end="")
    if input().lower().startswith('y'):
        interactive_security_test()
    
    print("\n=== EJERCICIOS ADICIONALES ===")
    print("1. Implementa logging persistente de eventos de seguridad")
    print("2. Agrega detección de patrones de ataque más sofisticados")
    print("3. Crea un sistema de alertas para administradores")
    print("4. Implementa whitelist/blacklist de usuarios")
    print("5. Agrega métricas de seguridad y reportes")
