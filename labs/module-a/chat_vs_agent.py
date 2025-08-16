"""
Laboratorio Módulo A: Chat vs Agente
Ejercicio práctico para comparar implementaciones básicas
"""

import json
import time
from typing import Dict, List, Any
from datetime import datetime

class SimpleChat:
    """Implementación básica de chatbot tradicional"""
    
    def __init__(self):
        self.responses = {
            "hola": "¡Hola! ¿En qué puedo ayudarte?",
            "clima": "No puedo acceder a información del clima en tiempo real.",
            "hora": f"La hora actual es {datetime.now().strftime('%H:%M')}",
            "ayuda": "Puedo responder preguntas básicas sobre algunos temas."
        }
    
    def respond(self, message: str) -> str:
        """
        TODO: Implementar lógica de respuesta directa
        
        Args:
            message: Mensaje del usuario
            
        Returns:
            Respuesta del chatbot
        """
        message_lower = message.lower()
        
        # Buscar palabras clave
        for keyword, response in self.responses.items():
            if keyword in message_lower:
                return response
        
        return "No entiendo tu pregunta. Puedes preguntar sobre: hola, clima, hora, ayuda"


class SimpleAgent:
    """Implementación básica de agente con ciclo planificar-ejecutar-evaluar"""
    
    def __init__(self):
        self.tools = {
            "get_time": self._get_current_time,
            "search_web": self._mock_web_search,
            "calculate": self._basic_calculator,
            "create_reminder": self._create_reminder
        }
        self.memory = []
    
    def process(self, goal: str) -> Dict[str, Any]:
        """
        TODO: Implementar ciclo planificar-ejecutar-evaluar
        
        Args:
            goal: Objetivo del usuario
            
        Returns:
            Resultado estructurado del procesamiento
        """
        result = {
            "goal": goal,
            "plan": None,
            "execution": [],
            "evaluation": None,
            "final_response": None,
            "timestamp": datetime.now().isoformat()
        }
        
        # FASE 1: PLANIFICAR
        plan = self._plan(goal)
        result["plan"] = plan
        
        # FASE 2: EJECUTAR
        execution_results = []
        for step in plan["steps"]:
            step_result = self._execute_step(step)
            execution_results.append(step_result)
        
        result["execution"] = execution_results
        
        # FASE 3: EVALUAR
        evaluation = self._evaluate(goal, execution_results)
        result["evaluation"] = evaluation
        
        # GENERAR RESPUESTA FINAL
        result["final_response"] = self._generate_response(execution_results, evaluation)
        
        # Guardar en memoria
        self.memory.append(result)
        
        return result
    
    def _plan(self, goal: str) -> Dict[str, Any]:
        """Genera un plan para alcanzar el objetivo"""
        plan = {
            "goal": goal,
            "steps": [],
            "estimated_duration": 0
        }
        
        goal_lower = goal.lower()
        
        # Análisis simple del objetivo para generar pasos
        if "clima" in goal_lower:
            plan["steps"].append({
                "action": "search_web",
                "parameters": {"query": "clima tiempo actual"},
                "expected_output": "información meteorológica"
            })
        
        if "hora" in goal_lower or "tiempo" in goal_lower:
            plan["steps"].append({
                "action": "get_time",
                "parameters": {},
                "expected_output": "hora actual"
            })
        
        if "recordatorio" in goal_lower or "reminder" in goal_lower:
            plan["steps"].append({
                "action": "create_reminder",
                "parameters": {"message": goal},
                "expected_output": "confirmación de recordatorio"
            })
        
        if any(op in goal_lower for op in ["+", "-", "*", "/", "calcular", "suma"]):
            plan["steps"].append({
                "action": "calculate",
                "parameters": {"expression": goal},
                "expected_output": "resultado del cálculo"
            })
        
        # Si no hay pasos específicos, agregar búsqueda general
        if not plan["steps"]:
            plan["steps"].append({
                "action": "search_web",
                "parameters": {"query": goal},
                "expected_output": "información relevante"
            })
        
        plan["estimated_duration"] = len(plan["steps"]) * 2  # 2 segundos por paso
        
        return plan
    
    def _execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta un paso individual del plan"""
        action = step["action"]
        parameters = step["parameters"]
        
        step_result = {
            "action": action,
            "parameters": parameters,
            "success": False,
            "output": None,
            "error": None,
            "duration": 0
        }
        
        start_time = time.time()
        
        try:
            if action in self.tools:
                output = self.tools[action](**parameters)
                step_result["output"] = output
                step_result["success"] = True
            else:
                step_result["error"] = f"Herramienta '{action}' no disponible"
        
        except Exception as e:
            step_result["error"] = str(e)
        
        step_result["duration"] = time.time() - start_time
        
        return step_result
    
    def _evaluate(self, goal: str, execution_results: List[Dict]) -> Dict[str, Any]:
        """Evalúa si se cumplió el objetivo"""
        total_steps = len(execution_results)
        successful_steps = sum(1 for result in execution_results if result["success"])
        
        evaluation = {
            "goal_achieved": successful_steps == total_steps,
            "success_rate": successful_steps / total_steps if total_steps > 0 else 0,
            "total_duration": sum(result["duration"] for result in execution_results),
            "recommendations": []
        }
        
        # Generar recomendaciones
        if evaluation["success_rate"] < 1.0:
            evaluation["recommendations"].append("Revisar herramientas fallidas")
        
        if evaluation["total_duration"] > 10:
            evaluation["recommendations"].append("Optimizar tiempo de ejecución")
        
        return evaluation
    
    def _generate_response(self, execution_results: List[Dict], evaluation: Dict) -> str:
        """Genera respuesta final basada en los resultados"""
        if not evaluation["goal_achieved"]:
            return "No pude completar todas las tareas solicitadas. Por favor revisa los errores."
        
        # Compilar resultados exitosos
        outputs = []
        for result in execution_results:
            if result["success"] and result["output"]:
                outputs.append(str(result["output"]))
        
        if outputs:
            return f"Completado exitosamente. Resultados: {' | '.join(outputs)}"
        else:
            return "Tareas completadas sin resultados específicos."
    
    # Herramientas mock para simulación
    def _get_current_time(self) -> str:
        return datetime.now().strftime("Son las %H:%M del %d/%m/%Y")
    
    def _mock_web_search(self, query: str) -> str:
        # Simulación de búsqueda web
        return f"Resultados simulados para: '{query}' - Información encontrada satisfactoriamente"
    
    def _basic_calculator(self, expression: str) -> str:
        try:
            # Solo operaciones básicas seguras
            allowed_chars = set('0123456789+-*/.() ')
            if all(c in allowed_chars for c in expression):
                result = eval(expression)  # ⚠️ Solo para demo - nunca usar eval en producción
                return f"Resultado: {result}"
            else:
                return "Error: Expresión contiene caracteres no permitidos"
        except:
            return "Error: No se pudo calcular la expresión"
    
    def _create_reminder(self, message: str) -> str:
        # Simulación de creación de recordatorio
        return f"Recordatorio creado: '{message}' para mañana a las 09:00"


def compare_systems():
    """Función para comparar ambos sistemas con ejemplos"""
    print("=== COMPARACIÓN: CHAT vs AGENTE ===\n")
    
    chat = SimpleChat()
    agent = SimpleAgent()
    
    test_cases = [
        "¿Qué hora es?",
        "Busca información sobre el clima y crea un recordatorio",
        "Calcula 25 + 17 * 2",
        "Hola, ¿cómo estás?"
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"--- Caso de Prueba {i}: '{test_case}' ---")
        
        # Respuesta del Chat
        print("🤖 CHAT:")
        chat_response = chat.respond(test_case)
        print(f"   {chat_response}")
        
        # Respuesta del Agente
        print("🧠 AGENTE:")
        agent_result = agent.process(test_case)
        print(f"   Plan: {len(agent_result['plan']['steps'])} pasos")
        print(f"   Éxito: {agent_result['evaluation']['success_rate']:.0%}")
        print(f"   Respuesta: {agent_result['final_response']}")
        
        print()


if __name__ == "__main__":
    # TODO: Ejecuta la comparación y analiza las diferencias
    compare_systems()
    
    print("\n=== EJERCICIOS ===")
    print("1. Modifica SimpleChat para manejar más casos")
    print("2. Agrega nuevas herramientas al SimpleAgent")
    print("3. Mejora la lógica de planificación del agente")
    print("4. Compara rendimiento y capacidades de ambos sistemas")
