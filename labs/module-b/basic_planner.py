"""
Laboratorio Módulo B: Planner Básico
Implementación del componente Planner del patrón PEC
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import re
from dataclasses import dataclass, asdict


@dataclass
class PlanStep:
    """Representa un paso individual en un plan de ejecución"""
    step_id: int
    action: str
    tool: str
    parameters: Dict[str, Any]
    expected_output: str
    validation_criteria: List[str]
    max_retries: int = 3
    timeout_seconds: int = 30


@dataclass
class ExecutionPlan:
    """Representa un plan completo de ejecución"""
    plan_id: str
    goal: str
    steps: List[PlanStep]
    success_criteria: List[str]
    max_total_retries: int = 3
    timeout_seconds: int = 300
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir plan a diccionario JSON serializable"""
        return {
            'plan_id': self.plan_id,
            'goal': self.goal,
            'steps': [asdict(step) for step in self.steps],
            'success_criteria': self.success_criteria,
            'max_total_retries': self.max_total_retries,
            'timeout_seconds': self.timeout_seconds,
            'created_at': self.created_at
        }


class BasePlanner(ABC):
    """Clase base abstracta para planificadores de agentes"""
    
    def __init__(self, available_tools: List[str]):
        self.available_tools = set(available_tools)
        self.plan_counter = 0
        self.planning_history = []
    
    @abstractmethod
    def create_plan(self, goal: str, context: Dict[str, Any] = None) -> ExecutionPlan:
        """
        Crear plan de ejecución para el objetivo dado
        
        Args:
            goal: Objetivo que debe cumplir el agente
            context: Contexto adicional para la planificación
            
        Returns:
            ExecutionPlan con los pasos a ejecutar
        """
        pass
    
    def validate_plan(self, plan: ExecutionPlan) -> Tuple[bool, str]:
        """
        Validar que el plan sea ejecutable
        
        Args:
            plan: Plan a validar
            
        Returns:
            Tupla (es_válido, mensaje_detalle)
        """
        # Verificar que el plan tenga pasos
        if not plan.steps:
            return False, "El plan no tiene pasos definidos"
        
        # Verificar que todos los pasos usen herramientas disponibles
        for step in plan.steps:
            if step.tool not in self.available_tools:
                return False, f"Herramienta no disponible: {step.tool}"
        
        # Verificar que los IDs de pasos sean secuenciales
        expected_ids = list(range(1, len(plan.steps) + 1))
        actual_ids = [step.step_id for step in plan.steps]
        if actual_ids != expected_ids:
            return False, "Los IDs de pasos deben ser secuenciales empezando en 1"
        
        # Verificar que haya criterios de éxito
        if not plan.success_criteria:
            return False, "El plan debe tener criterios de éxito definidos"
        
        return True, "Plan válido"
    
    def generate_plan_id(self) -> str:
        """Generar ID único para el plan"""
        self.plan_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"plan_{timestamp}_{self.plan_counter:03d}"


class SimplePlanner(BasePlanner):
    """
    Planificador simple que descompone objetivos en pasos ejecutables
    
    TODO: Completar implementación con:
    1. Análisis de objetivos por patrones de texto
    2. Templates de planes para diferentes tipos de tareas
    3. Generación de pasos con validaciones apropiadas
    """
    
    def __init__(self, available_tools: List[str]):
        super().__init__(available_tools)
        
        # Mapeo de patrones de texto a tipos de tarea
        self.task_patterns = {
            'search_and_calculate': [
                r'buscar.*calcular',
                r'encontrar.*contar',
                r'información.*estadística',
                r'datos.*análisis'
            ],
            'information_retrieval': [
                r'buscar.*información',
                r'encontrar.*sobre',
                r'consultar.*acerca',
                r'obtener.*datos'
            ],
            'data_processing': [
                r'procesar.*datos',
                r'transformar.*formato',
                r'convertir.*en',
                r'formatear.*como'
            ],
            'calculation': [
                r'calcular.*',
                r'resolver.*ecuación',
                r'operación.*matemática'
            ]
        }
        
        # Templates de planes por tipo de tarea
        self.plan_templates = {
            'search_and_calculate': self._create_search_calculate_plan,
            'information_retrieval': self._create_info_retrieval_plan,
            'data_processing': self._create_data_processing_plan,
            'calculation': self._create_calculation_plan
        }
    
    def create_plan(self, goal: str, context: Dict[str, Any] = None) -> ExecutionPlan:
        """
        TODO: Implementar creación de plan basada en análisis del objetivo
        
        El proceso debe:
        1. Analizar el tipo de tarea del objetivo
        2. Seleccionar template apropiado
        3. Generar pasos específicos
        4. Validar el plan resultante
        """
        if context is None:
            context = {}
        
        # Analizar tipo de tarea
        task_type = self._analyze_goal_type(goal)
        
        # Generar plan usando template apropiado
        if task_type in self.plan_templates:
            plan = self.plan_templates[task_type](goal, context)
        else:
            plan = self._create_generic_plan(goal, context)
        
        # Validar plan antes de retornar
        is_valid, message = self.validate_plan(plan)
        if not is_valid:
            raise ValueError(f"Plan inválido generado: {message}")
        
        # Registrar en historial
        self.planning_history.append({
            'goal': goal,
            'task_type': task_type,
            'plan_id': plan.plan_id,
            'created_at': plan.created_at
        })
        
        return plan
    
    def _analyze_goal_type(self, goal: str) -> str:
        """
        TODO: Analizar el objetivo para determinar tipo de tarea
        
        Debe revisar patrones regex para clasificar la tarea
        """
        goal_lower = goal.lower()
        
        for task_type, patterns in self.task_patterns.items():
            for pattern in patterns:
                if re.search(pattern, goal_lower):
                    return task_type
        
        return 'generic'
    
    def _create_search_calculate_plan(self, goal: str, context: Dict[str, Any]) -> ExecutionPlan:
        """
        TODO: Crear plan para tareas que requieren búsqueda + cálculo
        
        Template típico:
        1. Buscar información relevante
        2. Extraer datos numéricos
        3. Realizar cálculos
        4. Formatear resultados
        """
        plan_id = self.generate_plan_id()
        
        # Extraer términos de búsqueda del objetivo
        search_terms = self._extract_search_terms(goal)
        
        steps = [
            PlanStep(
                step_id=1,
                action="search",
                tool="search_documents",
                parameters={
                    "query": search_terms,
                    "max_results": context.get("max_results", 5)
                },
                expected_output="lista de documentos relevantes",
                validation_criteria=[
                    "al menos 1 resultado encontrado",
                    "relevancia promedio > 0.5"
                ]
            ),
            PlanStep(
                step_id=2,
                action="calculate",
                tool="math_calculator",
                parameters={
                    "expression": context.get("calculation", "count(results)")
                },
                expected_output="resultado numérico",
                validation_criteria=[
                    "resultado es número válido",
                    "resultado >= 0"
                ]
            ),
            PlanStep(
                step_id=3,
                action="format",
                tool="format_response",
                parameters={
                    "content": "{{search_results}} + {{calculation_result}}",
                    "format": context.get("output_format", "summary")
                },
                expected_output="respuesta formateada",
                validation_criteria=[
                    "longitud entre 100-500 palabras",
                    "incluye datos y cálculos"
                ]
            )
        ]
        
        return ExecutionPlan(
            plan_id=plan_id,
            goal=goal,
            steps=steps,
            success_criteria=[
                "información relevante encontrada",
                "cálculos completados correctamente",
                "respuesta bien formateada"
            ]
        )
    
    def _create_info_retrieval_plan(self, goal: str, context: Dict[str, Any]) -> ExecutionPlan:
        """
        TODO: Crear plan para recuperación simple de información
        """
        plan_id = self.generate_plan_id()
        search_terms = self._extract_search_terms(goal)
        
        steps = [
            PlanStep(
                step_id=1,
                action="search",
                tool="search_documents",
                parameters={
                    "query": search_terms,
                    "max_results": context.get("max_results", 3)
                },
                expected_output="documentos relevantes",
                validation_criteria=[
                    "al menos 1 resultado encontrado"
                ]
            ),
            PlanStep(
                step_id=2,
                action="format",
                tool="format_response",
                parameters={
                    "content": "{{search_results}}",
                    "format": context.get("output_format", "informative")
                },
                expected_output="información formateada",
                validation_criteria=[
                    "información completa y relevante"
                ]
            )
        ]
        
        return ExecutionPlan(
            plan_id=plan_id,
            goal=goal,
            steps=steps,
            success_criteria=[
                "información relevante recuperada",
                "respuesta clara y completa"
            ]
        )
    
    def _create_data_processing_plan(self, goal: str, context: Dict[str, Any]) -> ExecutionPlan:
        """
        TODO: Crear plan para procesamiento de datos
        """
        plan_id = self.generate_plan_id()
        
        steps = [
            PlanStep(
                step_id=1,
                action="format",
                tool="format_response",
                parameters={
                    "content": context.get("input_data", ""),
                    "format": context.get("target_format", "structured")
                },
                expected_output="datos procesados",
                validation_criteria=[
                    "formato correcto aplicado",
                    "datos preservados"
                ]
            )
        ]
        
        return ExecutionPlan(
            plan_id=plan_id,
            goal=goal,
            steps=steps,
            success_criteria=[
                "datos procesados correctamente"
            ]
        )
    
    def _create_calculation_plan(self, goal: str, context: Dict[str, Any]) -> ExecutionPlan:
        """
        TODO: Crear plan para cálculos matemáticos
        """
        plan_id = self.generate_plan_id()
        
        # Extraer expresión matemática del objetivo
        expression = self._extract_math_expression(goal, context)
        
        steps = [
            PlanStep(
                step_id=1,
                action="calculate",
                tool="math_calculator",
                parameters={
                    "expression": expression
                },
                expected_output="resultado numérico",
                validation_criteria=[
                    "resultado es número válido"
                ]
            ),
            PlanStep(
                step_id=2,
                action="format",
                tool="format_response",
                parameters={
                    "content": f"Resultado: {{calculation_result}}",
                    "format": "mathematical"
                },
                expected_output="resultado formateado",
                validation_criteria=[
                    "incluye explicación del cálculo"
                ]
            )
        ]
        
        return ExecutionPlan(
            plan_id=plan_id,
            goal=goal,
            steps=steps,
            success_criteria=[
                "cálculo ejecutado correctamente",
                "resultado presentado claramente"
            ]
        )
    
    def _create_generic_plan(self, goal: str, context: Dict[str, Any]) -> ExecutionPlan:
        """
        TODO: Crear plan genérico para objetivos no clasificados
        """
        plan_id = self.generate_plan_id()
        
        steps = [
            PlanStep(
                step_id=1,
                action="analyze",
                tool="format_response",
                parameters={
                    "content": f"Analizando objetivo: {goal}",
                    "format": "analysis"
                },
                expected_output="análisis del objetivo",
                validation_criteria=[
                    "objetivo analizado"
                ]
            )
        ]
        
        return ExecutionPlan(
            plan_id=plan_id,
            goal=goal,
            steps=steps,
            success_criteria=[
                "objetivo procesado"
            ]
        )
    
    def _extract_search_terms(self, goal: str) -> str:
        """Extraer términos de búsqueda del objetivo"""
        # Remover palabras de acción comunes
        stop_words = {'buscar', 'encontrar', 'sobre', 'acerca', 'de', 'la', 'el', 'en', 'y', 'o'}
        words = goal.lower().split()
        search_words = [w for w in words if w not in stop_words and len(w) > 2]
        return ' '.join(search_words[:5])  # Máximo 5 palabras clave
    
    def _extract_math_expression(self, goal: str, context: Dict[str, Any]) -> str:
        """Extraer expresión matemática del objetivo o contexto"""
        if 'expression' in context:
            return context['expression']
        
        # Buscar patrones matemáticos en el objetivo
        math_patterns = [
            r'(\d+\s*[+\-*/]\s*\d+)',
            r'(\d+\s*\^\s*\d+)',
            r'sqrt\(\d+\)',
            r'log\(\d+\)'
        ]
        
        for pattern in math_patterns:
            match = re.search(pattern, goal)
            if match:
                return match.group(1)
        
        return "1 + 1"  # Expresión por defecto


def test_simple_planner():
    """Función de prueba para el SimplePlanner"""
    print("=== PRUEBAS DEL SIMPLE PLANNER ===\n")
    
    # Herramientas disponibles simuladas
    available_tools = [
        'search_documents',
        'math_calculator', 
        'format_response',
        'get_current_time'
    ]
    
    planner = SimplePlanner(available_tools)
    
    # Casos de prueba
    test_cases = [
        {
            "goal": "Buscar información sobre Python y calcular cuántos ejemplos hay",
            "context": {"max_results": 5},
            "expected_type": "search_and_calculate"
        },
        {
            "goal": "Encontrar información sobre machine learning",
            "context": {},
            "expected_type": "information_retrieval"
        },
        {
            "goal": "Calcular 25 * 4 + 10",
            "context": {"expression": "25 * 4 + 10"},
            "expected_type": "calculation"
        },
        {
            "goal": "Formatear los datos en formato JSON",
            "context": {"input_data": "test data", "target_format": "json"},
            "expected_type": "data_processing"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"--- Caso {i}: {test_case['expected_type']} ---")
        print(f"Objetivo: {test_case['goal']}")
        
        try:
            plan = planner.create_plan(test_case['goal'], test_case['context'])
            
            print(f"Plan ID: {plan.plan_id}")
            print(f"Pasos: {len(plan.steps)}")
            
            for step in plan.steps:
                print(f"  {step.step_id}. {step.action} con {step.tool}")
                print(f"     Parámetros: {step.parameters}")
                print(f"     Salida esperada: {step.expected_output}")
            
            print(f"Criterios de éxito: {plan.success_criteria}")
            
            # Validar plan
            is_valid, message = planner.validate_plan(plan)
            print(f"Validación: {'✅ VÁLIDO' if is_valid else '❌ INVÁLIDO'} - {message}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print()
    
    # Mostrar historial de planificación
    print("--- Historial de Planificación ---")
    for entry in planner.planning_history:
        print(f"Plan {entry['plan_id']}: {entry['task_type']} - {entry['goal'][:50]}...")


def interactive_planner():
    """Herramienta interactiva para probar el planner"""
    print("=== PLANNER INTERACTIVO ===\n")
    
    available_tools = [
        'search_documents',
        'math_calculator',
        'format_response',
        'get_current_time',
        'weather_api'
    ]
    
    planner = SimplePlanner(available_tools)
    
    print("Herramientas disponibles:")
    for tool in available_tools:
        print(f"  - {tool}")
    print()
    
    while True:
        print("Opciones:")
        print("1. Crear plan para un objetivo")
        print("2. Ver historial de planes")
        print("3. Validar plan existente")
        print("4. Salir")
        
        choice = input("\nElige una opción (1-4): ").strip()
        
        if choice == "1":
            goal = input("Describe el objetivo: ")
            
            # Recopilar contexto opcional
            print("\nContexto opcional (presiona Enter para valores por defecto):")
            max_results = input("Máximo resultados de búsqueda (5): ") or "5"
            output_format = input("Formato de salida (summary): ") or "summary"
            
            context = {
                "max_results": int(max_results),
                "output_format": output_format
            }
            
            try:
                plan = planner.create_plan(goal, context)
                
                print(f"\n✅ Plan creado: {plan.plan_id}")
                print(f"Objetivo: {plan.goal}")
                print(f"Pasos ({len(plan.steps)}):")
                
                for step in plan.steps:
                    print(f"  {step.step_id}. {step.action.upper()}")
                    print(f"     Herramienta: {step.tool}")
                    print(f"     Parámetros: {step.parameters}")
                    print(f"     Resultado esperado: {step.expected_output}")
                    print(f"     Validaciones: {step.validation_criteria}")
                    print()
                
                print("Criterios de éxito:")
                for criterion in plan.success_criteria:
                    print(f"  - {criterion}")
                
            except Exception as e:
                print(f"❌ Error creando plan: {str(e)}")
        
        elif choice == "2":
            if not planner.planning_history:
                print("No hay planes en el historial.")
            else:
                print("\nHistorial de planes:")
                for i, entry in enumerate(planner.planning_history, 1):
                    print(f"{i}. {entry['plan_id']} ({entry['task_type']})")
                    print(f"   Objetivo: {entry['goal'][:60]}...")
                    print(f"   Creado: {entry['created_at']}")
                    print()
        
        elif choice == "3":
            # Para este ejercicio, solo mostrar cómo sería la validación
            print("Función de validación disponible en planner.validate_plan()")
            print("Verifica: herramientas disponibles, IDs secuenciales, criterios de éxito")
        
        elif choice == "4":
            break
        
        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    # TODO: Ejecuta las pruebas y experimenta con el planner
    test_simple_planner()
    
    print("\n¿Quieres probar el planner interactivo? (y/n): ", end="")
    if input().lower().startswith('y'):
        interactive_planner()
    
    print("\n=== EJERCICIOS ADICIONALES ===")
    print("1. Agrega nuevos tipos de tareas y sus patrones")
    print("2. Implementa validación más sofisticada de parámetros")
    print("3. Crea templates más específicos por dominio")
    print("4. Agrega estimación de tiempo y recursos por paso")
    print("5. Implementa planificación condicional (if/then/else)")
