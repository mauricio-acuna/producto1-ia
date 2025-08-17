"""
Laboratorio Módulo B: Agente PEC Completo
Integración de Planner + Executor + Critic en un agente funcional
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import time
from dataclasses import dataclass, asdict
from enum import Enum

# Importar componentes de labs anteriores
from basic_planner import SimplePlanner, ExecutionPlan, PlanStep
from tool_registry import ToolRegistry, ToolDefinition, ToolCall


class ExecutionStatus(Enum):
    """Estados de ejecución de un plan"""
    PENDING = "pending"
    RUNNING = "running" 
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class ExecutionResult:
    """Resultado de ejecución de un plan"""
    plan_id: str
    status: ExecutionStatus
    completed_steps: int
    total_steps: int
    execution_time: float
    final_result: Any
    step_results: List[Dict[str, Any]]
    error_message: Optional[str] = None
    critic_evaluation: Optional[Dict[str, Any]] = None


class PlanExecutor:
    """
    Ejecutor de planes con manejo de errores y retry logic
    
    TODO: Implementar ejecución robusta de planes
    """
    
    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry
        self.execution_history: List[ExecutionResult] = []
    
    def execute_plan(self, plan: ExecutionPlan) -> ExecutionResult:
        """
        TODO: Ejecutar plan completo con manejo de errores
        
        Proceso:
        1. Validar plan antes de ejecutar
        2. Ejecutar pasos secuencialmente
        3. Manejar errores y retries
        4. Recopilar resultados
        5. Generar resultado final
        """
        start_time = time.time()
        step_results = []
        completed_steps = 0
        
        try:
            print(f"🚀 Iniciando ejecución del plan: {plan.plan_id}")
            print(f"📋 Objetivo: {plan.goal}")
            print(f"📝 Pasos a ejecutar: {len(plan.steps)}")
            
            # Ejecutar cada paso
            for step in plan.steps:
                print(f"\n--- Ejecutando Paso {step.step_id}: {step.action} ---")
                
                step_result = self._execute_step_with_retry(step)
                step_results.append(step_result)
                
                if step_result['status'] == 'success':
                    completed_steps += 1
                    print(f"✅ Paso {step.step_id} completado exitosamente")
                else:
                    print(f"❌ Paso {step.step_id} falló: {step_result.get('error', 'Error desconocido')}")
                    
                    # Decidir si continuar o abortar
                    if self._should_abort_execution(step, step_result, plan):
                        print(f"🛑 Abortando ejecución después del paso {step.step_id}")
                        break
            
            # Determinar estado final
            execution_time = time.time() - start_time
            
            if completed_steps == len(plan.steps):
                status = ExecutionStatus.SUCCESS
                final_result = self._compile_final_result(step_results)
                error_message = None
            elif completed_steps > 0:
                status = ExecutionStatus.PARTIAL_SUCCESS
                final_result = self._compile_partial_result(step_results)
                error_message = f"Solo {completed_steps}/{len(plan.steps)} pasos completados"
            else:
                status = ExecutionStatus.FAILED
                final_result = None
                error_message = "Ningún paso se completó exitosamente"
            
            result = ExecutionResult(
                plan_id=plan.plan_id,
                status=status,
                completed_steps=completed_steps,
                total_steps=len(plan.steps),
                execution_time=execution_time,
                final_result=final_result,
                step_results=step_results,
                error_message=error_message
            )
            
            self.execution_history.append(result)
            
            print(f"\n🏁 Ejecución completada en {execution_time:.2f}s")
            print(f"📊 Estado: {status.value}")
            print(f"✅ Pasos completados: {completed_steps}/{len(plan.steps)}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_result = ExecutionResult(
                plan_id=plan.plan_id,
                status=ExecutionStatus.FAILED,
                completed_steps=completed_steps,
                total_steps=len(plan.steps),
                execution_time=execution_time,
                final_result=None,
                step_results=step_results,
                error_message=str(e)
            )
            
            self.execution_history.append(error_result)
            print(f"💥 Error crítico en ejecución: {str(e)}")
            
            return error_result
    
    def _execute_step_with_retry(self, step: PlanStep) -> Dict[str, Any]:
        """
        TODO: Ejecutar paso individual con lógica de retry
        """
        last_error = None
        
        for attempt in range(step.max_retries + 1):
            try:
                if attempt > 0:
                    print(f"🔄 Reintento {attempt}/{step.max_retries}")
                    time.sleep(min(attempt * 2, 10))  # Backoff exponencial limitado
                
                start_time = time.time()
                
                # Ejecutar herramienta
                result = self.tool_registry.execute_tool(step.tool, step.parameters)
                execution_time = time.time() - start_time
                
                return {
                    'step_id': step.step_id,
                    'status': 'success',
                    'result': result,
                    'execution_time': execution_time,
                    'attempts': attempt + 1,
                    'timestamp': datetime.now().isoformat()
                }
                
            except Exception as e:
                last_error = e
                execution_time = time.time() - start_time
                print(f"⚠️ Intento {attempt + 1} falló: {str(e)}")
                
                if attempt == step.max_retries:
                    return {
                        'step_id': step.step_id,
                        'status': 'error',
                        'result': None,
                        'error': str(e),
                        'execution_time': execution_time,
                        'attempts': attempt + 1,
                        'timestamp': datetime.now().isoformat()
                    }
    
    def _should_abort_execution(self, step: PlanStep, step_result: Dict[str, Any], plan: ExecutionPlan) -> bool:
        """Decidir si abortar ejecución basado en el fallo del paso"""
        # Por ahora, política simple: abortar si el paso falló y es crítico
        # En una implementación más sofisticada, esto podría basarse en
        # la importancia del paso, tipo de error, etc.
        return step_result['status'] != 'success'
    
    def _compile_final_result(self, step_results: List[Dict[str, Any]]) -> Any:
        """Compilar resultado final exitoso"""
        # Extraer resultados de cada paso
        results = {}
        for step_result in step_results:
            if step_result['status'] == 'success':
                results[f"step_{step_result['step_id']}"] = step_result['result']
        
        return {
            'success': True,
            'step_results': results,
            'summary': self._generate_execution_summary(step_results)
        }
    
    def _compile_partial_result(self, step_results: List[Dict[str, Any]]) -> Any:
        """Compilar resultado parcial"""
        successful_results = {}
        failed_steps = []
        
        for step_result in step_results:
            if step_result['status'] == 'success':
                successful_results[f"step_{step_result['step_id']}"] = step_result['result']
            else:
                failed_steps.append({
                    'step_id': step_result['step_id'],
                    'error': step_result.get('error', 'Unknown error')
                })
        
        return {
            'success': False,
            'partial_results': successful_results,
            'failed_steps': failed_steps,
            'summary': f"Ejecución parcial: {len(successful_results)} pasos exitosos"
        }
    
    def _generate_execution_summary(self, step_results: List[Dict[str, Any]]) -> str:
        """Generar resumen de ejecución"""
        total_time = sum(sr.get('execution_time', 0) for sr in step_results)
        successful_steps = len([sr for sr in step_results if sr['status'] == 'success'])
        
        return f"Ejecutados {successful_steps}/{len(step_results)} pasos en {total_time:.2f}s"


class ResultCritic:
    """
    Crítico que evalúa resultados de ejecución y determina próximos pasos
    
    TODO: Implementar evaluación inteligente de resultados
    """
    
    def __init__(self, success_threshold: float = 0.8):
        self.success_threshold = success_threshold
        self.evaluation_history: List[Dict[str, Any]] = []
    
    def evaluate_execution(self, plan: ExecutionPlan, execution_result: ExecutionResult) -> Dict[str, Any]:
        """
        TODO: Evaluar resultado de ejecución completa
        
        Evaluación debe considerar:
        1. Criterios de éxito del plan
        2. Calidad de resultados por paso
        3. Eficiencia de ejecución
        4. Completitud de objetivos
        """
        evaluation_start = time.time()
        
        # Evaluar completitud
        completeness_score = execution_result.completed_steps / execution_result.total_steps
        
        # Evaluar criterios de éxito del plan
        success_criteria_met = self._evaluate_success_criteria(plan, execution_result)
        
        # Evaluar calidad de resultados individuales
        step_quality_scores = []
        for step_result in execution_result.step_results:
            if step_result['status'] == 'success':
                quality_score = self._evaluate_step_quality(step_result)
                step_quality_scores.append(quality_score)
        
        average_quality = sum(step_quality_scores) / len(step_quality_scores) if step_quality_scores else 0
        
        # Evaluar eficiencia
        efficiency_score = self._evaluate_efficiency(execution_result)
        
        # Calcular score final
        final_score = (
            completeness_score * 0.4 +
            success_criteria_met * 0.3 +
            average_quality * 0.2 +
            efficiency_score * 0.1
        )
        
        # Determinar recomendación
        recommendation = self._generate_recommendation(final_score, execution_result)
        
        evaluation = {
            'plan_id': plan.plan_id,
            'overall_score': final_score,
            'completeness_score': completeness_score,
            'success_criteria_score': success_criteria_met,
            'average_quality_score': average_quality,
            'efficiency_score': efficiency_score,
            'recommendation': recommendation,
            'issues_found': self._identify_issues(execution_result),
            'suggestions': self._generate_suggestions(execution_result),
            'evaluation_time': time.time() - evaluation_start,
            'timestamp': datetime.now().isoformat()
        }
        
        self.evaluation_history.append(evaluation)
        
        print(f"\n🧠 EVALUACIÓN DEL CRÍTICO")
        print(f"📊 Score general: {final_score:.2f}/1.0")
        print(f"✅ Completitud: {completeness_score:.2f}")
        print(f"🎯 Criterios cumplidos: {success_criteria_met:.2f}")
        print(f"⭐ Calidad promedio: {average_quality:.2f}")
        print(f"⚡ Eficiencia: {efficiency_score:.2f}")
        print(f"💡 Recomendación: {recommendation}")
        
        if evaluation['issues_found']:
            print(f"⚠️ Problemas detectados:")
            for issue in evaluation['issues_found']:
                print(f"  - {issue}")
        
        if evaluation['suggestions']:
            print(f"💭 Sugerencias:")
            for suggestion in evaluation['suggestions']:
                print(f"  - {suggestion}")
        
        return evaluation
    
    def _evaluate_success_criteria(self, plan: ExecutionPlan, execution_result: ExecutionResult) -> float:
        """Evaluar si se cumplieron los criterios de éxito del plan"""
        if not plan.success_criteria:
            return 1.0
        
        # Implementación simplificada - en la práctica sería más sofisticada
        if execution_result.status == ExecutionStatus.SUCCESS:
            return 1.0
        elif execution_result.status == ExecutionStatus.PARTIAL_SUCCESS:
            return 0.6
        else:
            return 0.0
    
    def _evaluate_step_quality(self, step_result: Dict[str, Any]) -> float:
        """Evaluar calidad del resultado de un paso"""
        if step_result['status'] != 'success':
            return 0.0
        
        # Factores de calidad
        quality_score = 1.0
        
        # Penalizar si requirió muchos reintentos
        attempts = step_result.get('attempts', 1)
        if attempts > 1:
            quality_score *= (1.0 - (attempts - 1) * 0.1)
        
        # Bonificar ejecución rápida
        execution_time = step_result.get('execution_time', 0)
        if execution_time < 1.0:  # Menos de 1 segundo
            quality_score *= 1.1
        elif execution_time > 10.0:  # Más de 10 segundos
            quality_score *= 0.9
        
        return max(0.0, min(1.0, quality_score))
    
    def _evaluate_efficiency(self, execution_result: ExecutionResult) -> float:
        """Evaluar eficiencia de la ejecución"""
        # Factores de eficiencia
        efficiency_score = 1.0
        
        # Tiempo total de ejecución
        if execution_result.execution_time < 5.0:
            efficiency_score = 1.0
        elif execution_result.execution_time < 30.0:
            efficiency_score = 0.8
        else:
            efficiency_score = 0.6
        
        # Número de reintentos totales
        total_attempts = sum(sr.get('attempts', 1) for sr in execution_result.step_results)
        expected_attempts = len(execution_result.step_results)
        
        if total_attempts == expected_attempts:
            pass  # Perfecto, sin penalización
        else:
            retry_penalty = (total_attempts - expected_attempts) * 0.1
            efficiency_score *= (1.0 - retry_penalty)
        
        return max(0.0, min(1.0, efficiency_score))
    
    def _generate_recommendation(self, score: float, execution_result: ExecutionResult) -> str:
        """Generar recomendación basada en el score"""
        if score >= 0.9:
            return "EXCELLENT - Ejecución exitosa y eficiente"
        elif score >= 0.7:
            return "GOOD - Ejecución satisfactoria con mejoras menores"
        elif score >= 0.5:
            return "ACCEPTABLE - Ejecución parcial, revisar pasos fallidos"
        elif score >= 0.3:
            return "POOR - Múltiples problemas, considerar replaneación"
        else:
            return "FAILED - Ejecución fallida, replaneación necesaria"
    
    def _identify_issues(self, execution_result: ExecutionResult) -> List[str]:
        """Identificar problemas específicos en la ejecución"""
        issues = []
        
        # Problemas de completitud
        if execution_result.completed_steps < execution_result.total_steps:
            failed_steps = execution_result.total_steps - execution_result.completed_steps
            issues.append(f"{failed_steps} pasos fallaron en la ejecución")
        
        # Problemas de tiempo
        if execution_result.execution_time > 60:
            issues.append("Ejecución tomó más de 1 minuto")
        
        # Problemas de reintentos
        total_attempts = sum(sr.get('attempts', 1) for sr in execution_result.step_results)
        if total_attempts > len(execution_result.step_results) * 1.5:
            issues.append("Demasiados reintentos requeridos")
        
        return issues
    
    def _generate_suggestions(self, execution_result: ExecutionResult) -> List[str]:
        """Generar sugerencias de mejora"""
        suggestions = []
        
        if execution_result.status == ExecutionStatus.FAILED:
            suggestions.append("Verificar disponibilidad de herramientas")
            suggestions.append("Revisar parámetros de entrada")
        
        if execution_result.execution_time > 30:
            suggestions.append("Considerar paralelización de pasos independientes")
            suggestions.append("Optimizar parámetros de herramientas")
        
        failed_steps = [sr for sr in execution_result.step_results if sr['status'] != 'success']
        if failed_steps:
            suggestions.append("Agregar validación previa para pasos críticos")
            suggestions.append("Implementar pasos de fallback alternativos")
        
        return suggestions


class PECAgent:
    """
    Agente completo implementando el patrón Planner-Executor-Critic
    
    TODO: Integrar todos los componentes en un agente funcional
    """
    
    def __init__(self, available_tools: List[str]):
        # Inicializar componentes
        self.tool_registry = ToolRegistry()
        self.planner = SimplePlanner(available_tools)
        self.executor = PlanExecutor(self.tool_registry)
        self.critic = ResultCritic()
        
        # Registrar herramientas básicas
        self._register_default_tools()
        
        # Historial del agente
        self.conversation_history = []
        self.agent_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'total_execution_time': 0,
            'average_score': 0
        }
    
    def process_request(self, user_goal: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        TODO: Procesar solicitud completa del usuario usando patrón PEC
        
        Flujo:
        1. PLANNER: Crear plan para el objetivo
        2. EXECUTOR: Ejecutar plan step by step
        3. CRITIC: Evaluar resultados y calidad
        4. Generar respuesta final al usuario
        """
        start_time = time.time()
        
        try:
            print(f"\n🤖 AGENTE PEC - Procesando solicitud")
            print(f"👤 Usuario: {user_goal}")
            
            # FASE 1: PLANIFICACIÓN
            print(f"\n🧠 FASE 1: PLANIFICACIÓN")
            plan = self.planner.create_plan(user_goal, context or {})
            
            # FASE 2: EJECUCIÓN  
            print(f"\n⚡ FASE 2: EJECUCIÓN")
            execution_result = self.executor.execute_plan(plan)
            
            # FASE 3: CRÍTICA Y EVALUACIÓN
            print(f"\n🧠 FASE 3: EVALUACIÓN")
            evaluation = self.critic.evaluate_execution(plan, execution_result)
            
            # Generar respuesta final
            response = self._generate_user_response(user_goal, execution_result, evaluation)
            
            # Actualizar métricas
            total_time = time.time() - start_time
            self._update_metrics(execution_result, evaluation, total_time)
            
            # Registrar en historial
            conversation_entry = {
                'user_goal': user_goal,
                'plan_id': plan.plan_id,
                'execution_result': asdict(execution_result),
                'evaluation': evaluation,
                'response': response,
                'timestamp': datetime.now().isoformat(),
                'total_time': total_time
            }
            
            self.conversation_history.append(conversation_entry)
            
            print(f"\n✨ RESPUESTA FINAL AL USUARIO:")
            print(f"{response['message']}")
            
            return response
            
        except Exception as e:
            error_response = {
                'success': False,
                'message': f"Lo siento, ocurrió un error al procesar tu solicitud: {str(e)}",
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
            self._update_metrics(None, None, time.time() - start_time)
            return error_response
    
    def _register_default_tools(self):
        """Registrar herramientas por defecto"""
        # Importar implementaciones de herramientas del lab anterior
        from tool_registry import (
            search_documents_impl, 
            math_calculator_impl, 
            format_response_impl, 
            get_current_time_impl
        )
        
        tools = [
            ToolDefinition(
                name="search_documents",
                description="Buscar documentos en base de conocimiento",
                parameters_schema={
                    "query": {"type": "string", "required": True, "min_length": 1},
                    "max_results": {"type": "number", "required": False, "min_value": 1, "max_value": 10}
                },
                function=search_documents_impl
            ),
            ToolDefinition(
                name="math_calculator",
                description="Realizar cálculos matemáticos seguros",
                parameters_schema={
                    "expression": {"type": "string", "required": True, "pattern": r"^[0-9+\-*/.() ]+$"}
                },
                function=math_calculator_impl
            ),
            ToolDefinition(
                name="format_response",
                description="Formatear respuestas en diferentes estilos",
                parameters_schema={
                    "content": {"type": "string", "required": True},
                    "format": {"type": "string", "required": False}
                },
                function=format_response_impl
            ),
            ToolDefinition(
                name="get_current_time",
                description="Obtener fecha y hora actual",
                parameters_schema={
                    "timezone": {"type": "string", "required": False}
                },
                function=get_current_time_impl
            )
        ]
        
        for tool in tools:
            self.tool_registry.register_tool(tool)
    
    def _generate_user_response(self, user_goal: str, execution_result: ExecutionResult, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Generar respuesta final para el usuario"""
        
        if execution_result.status == ExecutionStatus.SUCCESS:
            # Ejecución exitosa
            message = self._format_successful_response(execution_result.final_result)
            
        elif execution_result.status == ExecutionStatus.PARTIAL_SUCCESS:
            # Ejecución parcial
            message = self._format_partial_response(execution_result.final_result)
            
        else:
            # Ejecución fallida
            message = f"No pude completar tu solicitud debido a: {execution_result.error_message}"
        
        return {
            'success': execution_result.status in [ExecutionStatus.SUCCESS, ExecutionStatus.PARTIAL_SUCCESS],
            'message': message,
            'execution_details': {
                'plan_id': execution_result.plan_id,
                'steps_completed': f"{execution_result.completed_steps}/{execution_result.total_steps}",
                'execution_time': f"{execution_result.execution_time:.2f}s",
                'quality_score': f"{evaluation['overall_score']:.2f}/1.0"
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def _format_successful_response(self, final_result: Dict[str, Any]) -> str:
        """Formatear respuesta exitosa"""
        if 'summary' in final_result:
            return f"✅ Completado exitosamente. {final_result['summary']}"
        else:
            return "✅ Tu solicitud se completó exitosamente."
    
    def _format_partial_response(self, final_result: Dict[str, Any]) -> str:
        """Formatear respuesta parcial"""
        if 'summary' in final_result:
            return f"⚠️ Completado parcialmente. {final_result['summary']}"
        else:
            return "⚠️ Pude completar parte de tu solicitud, pero algunos pasos fallaron."
    
    def _update_metrics(self, execution_result: Optional[ExecutionResult], evaluation: Optional[Dict[str, Any]], total_time: float):
        """Actualizar métricas del agente"""
        self.agent_metrics['total_requests'] += 1
        self.agent_metrics['total_execution_time'] += total_time
        
        if execution_result and execution_result.status in [ExecutionStatus.SUCCESS, ExecutionStatus.PARTIAL_SUCCESS]:
            self.agent_metrics['successful_requests'] += 1
        
        if evaluation:
            # Actualizar promedio de score
            current_avg = self.agent_metrics['average_score']
            total_requests = self.agent_metrics['total_requests']
            new_score = evaluation['overall_score']
            
            self.agent_metrics['average_score'] = (current_avg * (total_requests - 1) + new_score) / total_requests
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del agente"""
        success_rate = (self.agent_metrics['successful_requests'] / 
                       self.agent_metrics['total_requests']) if self.agent_metrics['total_requests'] > 0 else 0
        
        avg_time = (self.agent_metrics['total_execution_time'] / 
                   self.agent_metrics['total_requests']) if self.agent_metrics['total_requests'] > 0 else 0
        
        return {
            **self.agent_metrics,
            'success_rate': success_rate,
            'average_execution_time': avg_time,
            'tool_registry_stats': self.tool_registry.get_registry_stats()
        }


def test_pec_agent():
    """Función de prueba para el agente PEC completo"""
    print("=== PRUEBAS DEL AGENTE PEC COMPLETO ===\n")
    
    # Crear agente
    available_tools = ['search_documents', 'math_calculator', 'format_response', 'get_current_time']
    agent = PECAgent(available_tools)
    
    # Casos de prueba
    test_cases = [
        {
            "goal": "Busca información sobre Python y cuenta cuántos resultados encuentras",
            "context": {"max_results": 3}
        },
        {
            "goal": "Calcula 15 * 3 + 8 y formatea el resultado",
            "context": {"output_format": "markdown"}
        },
        {
            "goal": "Obtén la hora actual y formátea un saludo personalizado",
            "context": {"output_format": "summary"}
        },
        {
            "goal": "Busca información sobre machine learning y procesa los datos",
            "context": {"max_results": 2}
        }
    ]
    
    print("Ejecutando casos de prueba...\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"{'='*60}")
        print(f"CASO DE PRUEBA {i}")
        print(f"{'='*60}")
        
        response = agent.process_request(test_case['goal'], test_case['context'])
        
        print(f"\n📊 RESULTADO:")
        print(f"Éxito: {'✅' if response['success'] else '❌'}")
        if 'execution_details' in response:
            details = response['execution_details']
            print(f"Pasos: {details['steps_completed']}")
            print(f"Tiempo: {details['execution_time']}")
            print(f"Calidad: {details['quality_score']}")
        
        print(f"\n{'='*60}\n")
        
        # Pausa entre casos
        time.sleep(1)
    
    # Mostrar estadísticas finales
    print("📈 ESTADÍSTICAS FINALES DEL AGENTE")
    print("="*50)
    stats = agent.get_agent_stats()
    
    print(f"Solicitudes totales: {stats['total_requests']}")
    print(f"Solicitudes exitosas: {stats['successful_requests']}")
    print(f"Tasa de éxito: {stats['success_rate']:.2%}")
    print(f"Score promedio: {stats['average_score']:.2f}")
    print(f"Tiempo promedio: {stats['average_execution_time']:.2f}s")
    
    tool_stats = stats['tool_registry_stats']
    print(f"\nHerramientas más usadas:")
    for tool, count in tool_stats['tool_usage'].items():
        print(f"  - {tool}: {count} llamadas")


def interactive_pec_agent():
    """Demostración interactiva del agente PEC"""
    print("=== AGENTE PEC INTERACTIVO ===\n")
    print("Bienvenido al agente PEC (Planner-Executor-Critic)")
    print("Puedes pedirme que busque información, haga cálculos, o combine tareas.\n")
    
    agent = PECAgent(['search_documents', 'math_calculator', 'format_response', 'get_current_time'])
    
    while True:
        print("Opciones:")
        print("1. Hacer una solicitud al agente")
        print("2. Ver estadísticas del agente")
        print("3. Ver historial de conversaciones")
        print("4. Salir")
        
        choice = input("\nElige una opción (1-4): ").strip()
        
        if choice == "1":
            goal = input("\n¿Qué te gustaría que haga? ")
            
            print("\nContexto opcional:")
            max_results = input("Máximo resultados de búsqueda (3): ") or "3"
            output_format = input("Formato de salida (summary/markdown/json): ") or "summary"
            
            context = {
                "max_results": int(max_results),
                "output_format": output_format
            }
            
            print(f"\n🚀 Procesando tu solicitud...")
            response = agent.process_request(goal, context)
            
            print(f"\n📋 RESUMEN:")
            print(f"Estado: {'✅ Éxito' if response['success'] else '❌ Error'}")
            
            if 'execution_details' in response:
                details = response['execution_details']
                print(f"Pasos ejecutados: {details['steps_completed']}")
                print(f"Tiempo total: {details['execution_time']}")
                print(f"Puntuación de calidad: {details['quality_score']}")
        
        elif choice == "2":
            stats = agent.get_agent_stats()
            print(f"\n📊 ESTADÍSTICAS DEL AGENTE:")
            print(f"Solicitudes procesadas: {stats['total_requests']}")
            print(f"Tasa de éxito: {stats['success_rate']:.1%}")
            print(f"Score promedio de calidad: {stats['average_score']:.2f}")
            print(f"Tiempo promedio por solicitud: {stats['average_execution_time']:.2f}s")
        
        elif choice == "3":
            if not agent.conversation_history:
                print("\nNo hay conversaciones en el historial.")
            else:
                print(f"\n📜 HISTORIAL ({len(agent.conversation_history)} entradas):")
                for i, entry in enumerate(agent.conversation_history[-5:], 1):  # Últimas 5
                    print(f"{i}. {entry['user_goal'][:60]}...")
                    print(f"   Resultado: {'✅' if entry['response']['success'] else '❌'}")
                    print(f"   Tiempo: {entry['total_time']:.2f}s")
                    print()
        
        elif choice == "4":
            print("\n👋 ¡Hasta luego!")
            break
        
        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    # Ejecutar pruebas automatizadas
    test_pec_agent()
    
    # Demostración interactiva
    print("\n¿Quieres probar el agente interactivo? (y/n): ", end="")
    if input().lower().startswith('y'):
        interactive_pec_agent()
    
    print("\n=== EJERCICIOS ADICIONALES ===")
    print("1. Implementa planificación condicional (if-then-else)")
    print("2. Agrega capacidad de aprendizaje del agente")
    print("3. Implementa ejecución paralela de pasos independientes")
    print("4. Crea sistema de plugins para herramientas")
    print("5. Agrega interfaz web para interacción visual")
