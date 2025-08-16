"""
Laboratorio Módulo A: JSON Estructurado
Ejercicio para generar y validar salidas estructuradas según schema
"""

import json
import jsonschema
from typing import Dict, Any, Tuple, List
from datetime import datetime

# Schema para respuestas de agente
AGENT_RESPONSE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["search", "create", "update", "delete", "analyze", "respond", "calculate"]
        },
        "parameters": {
            "type": "object",
            "description": "Parámetros específicos para la acción"
        },
        "reasoning": {
            "type": "string",
            "description": "Justificación de por qué se eligió esta acción",
            "minLength": 10
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Nivel de confianza (0-1)"
        },
        "metadata": {
            "type": "object",
            "properties": {
                "timestamp": {"type": "string"},
                "source": {"type": "string"},
                "version": {"type": "string"}
            }
        }
    },
    "required": ["action", "parameters", "reasoning"]
}

# Schema para planes de ejecución
EXECUTION_PLAN_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "plan_id": {"type": "string"},
        "goal": {"type": "string", "minLength": 5},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step_id": {"type": "integer"},
                    "action": {"type": "string"},
                    "parameters": {"type": "object"},
                    "expected_output": {"type": "string"}
                },
                "required": ["step_id", "action", "parameters"]
            },
            "minItems": 1
        },
        "estimated_duration": {"type": "number", "minimum": 0}
    },
    "required": ["plan_id", "goal", "steps"]
}


class StructuredOutputGenerator:
    """Generador de salidas estructuradas para agentes"""
    
    def __init__(self):
        self.validator = StructuredOutputValidator()
    
    def generate_agent_response(self, action: str, params: Dict[str, Any], 
                              reasoning: str = "", confidence: float = 0.8) -> Dict[str, Any]:
        """
        TODO: Generar respuesta estructurada válida según el schema
        
        Args:
            action: Acción a realizar
            params: Parámetros para la acción
            reasoning: Justificación de la acción
            confidence: Nivel de confianza (0-1)
            
        Returns:
            Respuesta estructurada válida
        """
        response = {
            "action": action,
            "parameters": params,
            "reasoning": reasoning or f"Ejecutando {action} con los parámetros proporcionados",
            "confidence": max(0.0, min(1.0, confidence)),  # Clamp entre 0 y 1
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "source": "StructuredOutputGenerator",
                "version": "1.0.0"
            }
        }
        
        return response
    
    def generate_execution_plan(self, goal: str, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        TODO: Generar plan de ejecución estructurado
        
        Args:
            goal: Objetivo del plan
            actions: Lista de acciones a realizar
            
        Returns:
            Plan de ejecución estructurado
        """
        plan = {
            "plan_id": f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "goal": goal,
            "steps": [],
            "estimated_duration": 0
        }
        
        for i, action in enumerate(actions, 1):
            step = {
                "step_id": i,
                "action": action.get("action", "unknown"),
                "parameters": action.get("parameters", {}),
                "expected_output": action.get("expected_output", "resultado de la acción")
            }
            plan["steps"].append(step)
        
        # Estimar duración (2 segundos por paso base)
        plan["estimated_duration"] = len(actions) * 2.0
        
        return plan


class StructuredOutputValidator:
    """Validador de salidas estructuradas"""
    
    def validate_response(self, response: Dict[str, Any]) -> Tuple[bool, str]:
        """
        TODO: Validar respuesta de agente contra schema
        
        Args:
            response: Respuesta a validar
            
        Returns:
            Tupla (es_válida, mensaje_error)
        """
        try:
            jsonschema.validate(response, AGENT_RESPONSE_SCHEMA)
            return True, "Respuesta válida"
        except jsonschema.ValidationError as e:
            return False, f"Error de validación: {e.message}"
        except Exception as e:
            return False, f"Error inesperado: {str(e)}"
    
    def validate_plan(self, plan: Dict[str, Any]) -> Tuple[bool, str]:
        """
        TODO: Validar plan de ejecución contra schema
        
        Args:
            plan: Plan a validar
            
        Returns:
            Tupla (es_válido, mensaje_error)
        """
        try:
            jsonschema.validate(plan, EXECUTION_PLAN_SCHEMA)
            return True, "Plan válido"
        except jsonschema.ValidationError as e:
            return False, f"Error de validación: {e.message}"
        except Exception as e:
            return False, f"Error inesperado: {str(e)}"
    
    def validate_json_syntax(self, json_string: str) -> Tuple[bool, str, Dict]:
        """Validar sintaxis JSON básica"""
        try:
            parsed = json.loads(json_string)
            return True, "JSON válido", parsed
        except json.JSONDecodeError as e:
            return False, f"JSON inválido: {e.msg} en línea {e.lineno}", {}


def test_structured_outputs():
    """Función de prueba para validar diferentes casos"""
    print("=== PRUEBAS DE SALIDAS ESTRUCTURADAS ===\n")
    
    generator = StructuredOutputGenerator()
    validator = StructuredOutputValidator()
    
    # Caso 1: Respuesta válida
    print("--- Caso 1: Respuesta Válida ---")
    response = generator.generate_agent_response(
        action="search",
        params={"query": "machine learning", "max_results": 5},
        reasoning="El usuario solicita información sobre machine learning",
        confidence=0.9
    )
    
    print("Respuesta generada:")
    print(json.dumps(response, indent=2, ensure_ascii=False))
    
    is_valid, message = validator.validate_response(response)
    print(f"Validación: {'✅ VÁLIDA' if is_valid else '❌ INVÁLIDA'} - {message}\n")
    
    # Caso 2: Plan de ejecución
    print("--- Caso 2: Plan de Ejecución ---")
    actions = [
        {
            "action": "search",
            "parameters": {"query": "clima Madrid"},
            "expected_output": "información meteorológica"
        },
        {
            "action": "create",
            "parameters": {"type": "reminder", "message": "Revisar clima"},
            "expected_output": "recordatorio creado"
        }
    ]
    
    plan = generator.generate_execution_plan(
        goal="Obtener información del clima y crear recordatorio",
        actions=actions
    )
    
    print("Plan generado:")
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    
    is_valid, message = validator.validate_plan(plan)
    print(f"Validación: {'✅ VÁLIDO' if is_valid else '❌ INVÁLIDO'} - {message}\n")
    
    # Caso 3: Respuesta inválida (para testing)
    print("--- Caso 3: Respuesta Inválida ---")
    invalid_response = {
        "action": "invalid_action",  # No está en el enum
        "parameters": {},
        # "reasoning" falta (requerido)
        "confidence": 1.5  # Fuera del rango 0-1
    }
    
    print("Respuesta inválida:")
    print(json.dumps(invalid_response, indent=2, ensure_ascii=False))
    
    is_valid, message = validator.validate_response(invalid_response)
    print(f"Validación: {'✅ VÁLIDA' if is_valid else '❌ INVÁLIDA'} - {message}\n")


def interactive_generator():
    """Generador interactivo para experimentar"""
    print("=== GENERADOR INTERACTIVO ===\n")
    
    generator = StructuredOutputGenerator()
    validator = StructuredOutputValidator()
    
    while True:
        print("Opciones:")
        print("1. Generar respuesta de agente")
        print("2. Generar plan de ejecución")
        print("3. Validar JSON personalizado")
        print("4. Salir")
        
        choice = input("\nElige una opción (1-4): ").strip()
        
        if choice == "1":
            action = input("Acción (search/create/update/delete/analyze/respond/calculate): ")
            query = input("Query/parámetro principal: ")
            reasoning = input("Razonamiento: ")
            
            try:
                confidence = float(input("Confianza (0-1): ") or "0.8")
            except ValueError:
                confidence = 0.8
            
            params = {"query": query} if query else {}
            
            response = generator.generate_agent_response(action, params, reasoning, confidence)
            print("\nRespuesta generada:")
            print(json.dumps(response, indent=2, ensure_ascii=False))
            
            is_valid, message = validator.validate_response(response)
            print(f"\nValidación: {'✅ VÁLIDA' if is_valid else '❌ INVÁLIDA'} - {message}")
        
        elif choice == "2":
            goal = input("Objetivo del plan: ")
            num_steps = int(input("Número de pasos: ") or "2")
            
            actions = []
            for i in range(num_steps):
                print(f"\nPaso {i+1}:")
                action = input("  Acción: ")
                param_key = input("  Parámetro clave: ")
                param_value = input("  Valor del parámetro: ")
                
                actions.append({
                    "action": action,
                    "parameters": {param_key: param_value} if param_key else {},
                    "expected_output": f"resultado del paso {i+1}"
                })
            
            plan = generator.generate_execution_plan(goal, actions)
            print("\nPlan generado:")
            print(json.dumps(plan, indent=2, ensure_ascii=False))
            
            is_valid, message = validator.validate_plan(plan)
            print(f"\nValidación: {'✅ VÁLIDO' if is_valid else '❌ INVÁLIDO'} - {message}")
        
        elif choice == "3":
            print("Introduce tu JSON (termina con línea vacía):")
            lines = []
            while True:
                line = input()
                if line.strip() == "":
                    break
                lines.append(line)
            
            json_string = "\n".join(lines)
            is_valid, message, parsed = validator.validate_json_syntax(json_string)
            
            print(f"\nValidación sintáctica: {'✅ VÁLIDA' if is_valid else '❌ INVÁLIDA'} - {message}")
            
            if is_valid:
                # Intentar validar contra schemas conocidos
                response_valid, response_msg = validator.validate_response(parsed)
                plan_valid, plan_msg = validator.validate_plan(parsed)
                
                if response_valid:
                    print("✅ Es una respuesta de agente válida")
                elif plan_valid:
                    print("✅ Es un plan de ejecución válido")
                else:
                    print("ℹ️ JSON válido pero no coincide con schemas conocidos")
        
        elif choice == "4":
            break
        
        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    # TODO: Ejecuta las pruebas y experimenta con el generador
    test_structured_outputs()
    
    print("\n¿Quieres probar el generador interactivo? (y/n): ", end="")
    if input().lower().startswith('y'):
        interactive_generator()
    
    print("\n=== EJERCICIOS ADICIONALES ===")
    print("1. Crea schemas para otros tipos de datos (usuarios, tareas, etc.)")
    print("2. Implementa validación de campos específicos (emails, URLs, fechas)")
    print("3. Agrega generación automática de documentación desde schemas")
    print("4. Crea tests unitarios para todos los casos de validación")
