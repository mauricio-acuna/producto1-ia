"""
Evaluation Suite para RepoGPT - Sistema de Evaluación del Capstone
==================================================================

Esta suite evalúa el agente RepoGPT usando los sistemas de evaluación
desarrollados en el Módulo D, adaptados específicamente para análisis de código.

Autor: Sistema de IA Educativo
Módulo: E - Capstone Project
"""

import time
import json
import statistics
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

# Importar el agente RepoGPT
from repogpt import RepoGPTAgent, RepoConfig, QueryResult

# Importar evaluadores del Módulo D (simulados)
# En implementación real, estos serían imports directos
from abc import ABC, abstractmethod
import re

# =============================================================================
# 1. TEST CASES ESPECÍFICOS PARA REPOSITORIOS
# =============================================================================

@dataclass
class RepoTestCase:
    """Caso de prueba específico para evaluación de repositorios"""
    id: str
    query: str
    expected_files: List[str]  # Archivos que deberían aparecer en la respuesta
    expected_functions: List[str]  # Funciones que deberían mencionarse
    expected_classes: List[str]  # Clases que deberían mencionarse
    expected_concepts: List[str]  # Conceptos clave que deberían aparecer
    difficulty: str  # "easy", "medium", "hard"
    category: str  # "function", "class", "structure", "general"
    
    def __post_init__(self):
        if not self.expected_files:
            self.expected_files = []
        if not self.expected_functions:
            self.expected_functions = []
        if not self.expected_classes:
            self.expected_classes = []
        if not self.expected_concepts:
            self.expected_concepts = []

@dataclass
class RepoEvaluationResult:
    """Resultado de evaluación específico para repositorios"""
    test_case_id: str
    query: str
    answer: str
    citations: List[str]
    
    # Métricas específicas
    file_precision: float
    file_recall: float
    function_precision: float
    function_recall: float
    class_precision: float
    class_recall: float
    concept_coverage: float
    
    # Métricas generales
    citation_accuracy: float
    response_quality: float
    relevance_score: float
    
    # Métricas de performance
    response_time: float
    confidence: float
    cost_usd: float
    
    # Score general
    overall_score: float
    
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

# =============================================================================
# 2. EVALUADORES ESPECÍFICOS PARA CÓDIGO
# =============================================================================

class CodeUnderstandingEvaluator:
    """Evaluador de comprensión de código"""
    
    def __init__(self):
        pass
    
    def evaluate_file_mentions(self, answer: str, expected_files: List[str]) -> tuple[float, float]:
        """Evaluar precisión y recall de archivos mencionados"""
        
        # Extraer archivos mencionados en la respuesta
        mentioned_files = set()
        
        # Buscar patrones de archivos (.py, .js, etc.)
        file_patterns = [
            r'(\w+\.py)',
            r'(\w+\.js)',
            r'(\w+\.ts)',
            r'(\w+\.java)',
            r'(\w+/\w+\.py)',
            r'(models/\w+)',
            r'(api/\w+)',
            r'(utils/\w+)'
        ]
        
        for pattern in file_patterns:
            matches = re.findall(pattern, answer, re.IGNORECASE)
            mentioned_files.update(matches)
        
        # También buscar nombres de archivos explícitos
        for expected_file in expected_files:
            if expected_file.lower() in answer.lower():
                mentioned_files.add(expected_file)
        
        # Calcular precisión y recall
        expected_set = set(expected_files)
        
        if not mentioned_files and not expected_set:
            return 1.0, 1.0
        
        precision = len(mentioned_files & expected_set) / len(mentioned_files) if mentioned_files else 0.0
        recall = len(mentioned_files & expected_set) / len(expected_set) if expected_set else 0.0
        
        return precision, recall
    
    def evaluate_function_mentions(self, answer: str, expected_functions: List[str]) -> tuple[float, float]:
        """Evaluar precisión y recall de funciones mencionadas"""
        
        mentioned_functions = set()
        
        # Buscar patrones de funciones
        function_patterns = [
            r'función\s+(\w+)',
            r'function\s+(\w+)',
            r'def\s+(\w+)',
            r'(\w+)\s*\(',
            r'método\s+(\w+)',
            r'method\s+(\w+)'
        ]
        
        for pattern in function_patterns:
            matches = re.findall(pattern, answer, re.IGNORECASE)
            mentioned_functions.update(matches)
        
        # Buscar nombres de funciones explícitos
        for expected_func in expected_functions:
            if expected_func.lower() in answer.lower():
                mentioned_functions.add(expected_func)
        
        expected_set = set(expected_functions)
        
        if not mentioned_functions and not expected_set:
            return 1.0, 1.0
        
        precision = len(mentioned_functions & expected_set) / len(mentioned_functions) if mentioned_functions else 0.0
        recall = len(mentioned_functions & expected_set) / len(expected_set) if expected_set else 0.0
        
        return precision, recall
    
    def evaluate_class_mentions(self, answer: str, expected_classes: List[str]) -> tuple[float, float]:
        """Evaluar precisión y recall de clases mencionadas"""
        
        mentioned_classes = set()
        
        # Buscar patrones de clases
        class_patterns = [
            r'clase\s+(\w+)',
            r'class\s+(\w+)',
            r'(\w+)\s*\(.*\):',  # Python class definition
            r'class\s+(\w+)\s*{',  # JavaScript/Java class
        ]
        
        for pattern in class_patterns:
            matches = re.findall(pattern, answer, re.IGNORECASE)
            mentioned_classes.update(matches)
        
        # Buscar nombres de clases explícitos
        for expected_class in expected_classes:
            if expected_class.lower() in answer.lower():
                mentioned_classes.add(expected_class)
        
        expected_set = set(expected_classes)
        
        if not mentioned_classes and not expected_set:
            return 1.0, 1.0
        
        precision = len(mentioned_classes & expected_set) / len(mentioned_classes) if mentioned_classes else 0.0
        recall = len(mentioned_classes & expected_set) / len(expected_set) if expected_set else 0.0
        
        return precision, recall
    
    def evaluate_concept_coverage(self, answer: str, expected_concepts: List[str]) -> float:
        """Evaluar cobertura de conceptos clave"""
        
        if not expected_concepts:
            return 1.0
        
        answer_lower = answer.lower()
        covered_concepts = 0
        
        for concept in expected_concepts:
            if concept.lower() in answer_lower:
                covered_concepts += 1
        
        return covered_concepts / len(expected_concepts)
    
    def evaluate_citation_accuracy(self, citations: List[str], expected_files: List[str]) -> float:
        """Evaluar precisión de las citas"""
        
        if not citations:
            return 0.0 if expected_files else 1.0
        
        accurate_citations = 0
        citation_pattern = r'\[([^\[\]]+)#'
        
        for citation in citations:
            match = re.search(citation_pattern, citation)
            if match:
                cited_file = match.group(1)
                # Verificar si el archivo citado está en los archivos esperados
                if any(expected_file in cited_file or cited_file in expected_file 
                      for expected_file in expected_files):
                    accurate_citations += 1
        
        return accurate_citations / len(citations)

class ResponseQualityEvaluator:
    """Evaluador de calidad de respuesta"""
    
    def evaluate_response_quality(self, answer: str, query: str) -> Dict[str, float]:
        """Evaluar calidad general de la respuesta"""
        
        scores = {}
        
        # Length appropriateness
        answer_length = len(answer)
        if answer_length < 50:
            scores['length'] = 0.3
        elif answer_length < 100:
            scores['length'] = 0.6
        elif answer_length < 300:
            scores['length'] = 1.0
        elif answer_length < 500:
            scores['length'] = 0.9
        else:
            scores['length'] = 0.7  # Muy largo puede ser malo
        
        # Informativeness (basado en variedad de palabras)
        words = answer.lower().split()
        unique_words = set(words)
        scores['informativeness'] = min(1.0, len(unique_words) / max(len(words), 1) * 2)
        
        # Structure (buscar indicadores de buena estructura)
        structure_indicators = [
            ':', '.', ',', ';',  # Puntuación
            'función', 'clase', 'archivo',  # Términos técnicos
            'implementa', 'define', 'utiliza'  # Verbos explicativos
        ]
        
        structure_score = 0
        for indicator in structure_indicators:
            if indicator in answer.lower():
                structure_score += 1
        
        scores['structure'] = min(1.0, structure_score / len(structure_indicators))
        
        # Relevance to query
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        overlap = len(query_words & answer_words)
        scores['relevance'] = min(1.0, overlap / max(len(query_words), 1))
        
        # Technical accuracy (buscar términos técnicos apropiados)
        technical_terms = [
            'función', 'method', 'class', 'variable', 'import', 'module',
            'api', 'endpoint', 'database', 'model', 'controller'
        ]
        
        tech_score = 0
        for term in technical_terms:
            if term in answer.lower():
                tech_score += 1
        
        scores['technical_accuracy'] = min(1.0, tech_score / 5)  # Normalize to max 5 terms
        
        return scores

# =============================================================================
# 3. SUITE DE EVALUACIÓN PRINCIPAL
# =============================================================================

class RepoGPTEvaluationSuite:
    """Suite completa de evaluación para RepoGPT"""
    
    def __init__(self, agent: RepoGPTAgent):
        self.agent = agent
        self.code_evaluator = CodeUnderstandingEvaluator()
        self.quality_evaluator = ResponseQualityEvaluator()
        
        # Test cases predefinidos
        self.test_cases = self._create_test_cases()
    
    def _create_test_cases(self) -> List[RepoTestCase]:
        """Crear casos de prueba específicos para el repositorio de ejemplo"""
        
        return [
            # Test cases fáciles
            RepoTestCase(
                id="easy_001",
                query="¿Qué funciones hay en el proyecto?",
                expected_files=["main.py", "utils/helpers.py", "api/endpoints.py"],
                expected_functions=["main", "format_date", "validate_email", "create_app"],
                expected_classes=[],
                expected_concepts=["función", "aplicación", "utilidad"],
                difficulty="easy",
                category="function"
            ),
            
            RepoTestCase(
                id="easy_002",
                query="¿Qué clases están definidas?",
                expected_files=["models/user.py"],
                expected_functions=[],
                expected_classes=["User"],
                expected_concepts=["clase", "modelo", "usuario"],
                difficulty="easy",
                category="class"
            ),
            
            RepoTestCase(
                id="easy_003",
                query="¿Qué hace el archivo main.py?",
                expected_files=["main.py"],
                expected_functions=["main"],
                expected_classes=["User"],
                expected_concepts=["aplicación", "punto de entrada", "inicio"],
                difficulty="easy",
                category="structure"
            ),
            
            # Test cases medianos
            RepoTestCase(
                id="medium_001",
                query="¿Cómo funciona la gestión de usuarios?",
                expected_files=["models/user.py", "api/endpoints.py"],
                expected_functions=["create_user", "get_users", "activate", "deactivate"],
                expected_classes=["User"],
                expected_concepts=["usuario", "gestión", "API", "endpoint"],
                difficulty="medium",
                category="general"
            ),
            
            RepoTestCase(
                id="medium_002",
                query="¿Qué utilidades están disponibles en el proyecto?",
                expected_files=["utils/helpers.py"],
                expected_functions=["format_date", "validate_email", "process_data", "calculate_metrics"],
                expected_classes=[],
                expected_concepts=["utilidad", "formato", "validación", "procesamiento"],
                difficulty="medium",
                category="function"
            ),
            
            RepoTestCase(
                id="medium_003",
                query="¿Cómo está estructurado el API?",
                expected_files=["api/endpoints.py"],
                expected_functions=["create_app", "get_users", "create_user", "get_user"],
                expected_classes=[],
                expected_concepts=["API", "endpoint", "Flask", "HTTP"],
                difficulty="medium",
                category="structure"
            ),
            
            # Test cases difíciles
            RepoTestCase(
                id="hard_001",
                query="¿Cuáles son las dependencias y cómo se relacionan los módulos?",
                expected_files=["main.py", "models/user.py", "utils/helpers.py", "api/endpoints.py"],
                expected_functions=["main", "create_app"],
                expected_classes=["User", "Flask"],
                expected_concepts=["dependencia", "importación", "módulo", "relación"],
                difficulty="hard",
                category="structure"
            ),
            
            RepoTestCase(
                id="hard_002",
                query="¿Qué patrones de diseño se utilizan en el proyecto?",
                expected_files=["api/endpoints.py", "models/user.py"],
                expected_functions=["create_app"],
                expected_classes=["User"],
                expected_concepts=["patrón", "diseño", "arquitectura", "estructura"],
                difficulty="hard",
                category="general"
            ),
            
            RepoTestCase(
                id="hard_003",
                query="¿Cómo se podría mejorar la arquitectura del proyecto?",
                expected_files=["main.py", "models/user.py", "api/endpoints.py"],
                expected_functions=["main", "create_app"],
                expected_classes=["User"],
                expected_concepts=["mejora", "arquitectura", "optimización", "refactoring"],
                difficulty="hard",
                category="general"
            )
        ]
    
    def run_evaluation(self, test_cases: Optional[List[RepoTestCase]] = None) -> Dict[str, Any]:
        """Ejecutar evaluación completa"""
        
        if test_cases is None:
            test_cases = self.test_cases
        
        print(f"🚀 Running RepoGPT Evaluation Suite")
        print(f"📊 Evaluating {len(test_cases)} test cases...")
        
        results = []
        start_time = time.time()
        
        for i, test_case in enumerate(test_cases):
            print(f"\n📝 Test Case {i+1}/{len(test_cases)}: {test_case.id}")
            print(f"   Query: {test_case.query}")
            
            # Ejecutar consulta
            query_result = self.agent.query(test_case.query)
            
            # Evaluar resultado
            evaluation_result = self._evaluate_test_case(test_case, query_result)
            results.append(evaluation_result)
            
            print(f"   ✅ Score: {evaluation_result.overall_score:.3f}")
        
        total_time = time.time() - start_time
        
        # Generar resumen
        summary = self._generate_summary(results, total_time)
        
        return {
            'summary': summary,
            'detailed_results': results,
            'test_cases': test_cases
        }
    
    def _evaluate_test_case(self, test_case: RepoTestCase, query_result: QueryResult) -> RepoEvaluationResult:
        """Evaluar un caso de prueba específico"""
        
        # Evaluar comprensión de código
        file_precision, file_recall = self.code_evaluator.evaluate_file_mentions(
            query_result.answer, test_case.expected_files
        )
        
        function_precision, function_recall = self.code_evaluator.evaluate_function_mentions(
            query_result.answer, test_case.expected_functions
        )
        
        class_precision, class_recall = self.code_evaluator.evaluate_class_mentions(
            query_result.answer, test_case.expected_classes
        )
        
        concept_coverage = self.code_evaluator.evaluate_concept_coverage(
            query_result.answer, test_case.expected_concepts
        )
        
        citation_accuracy = self.code_evaluator.evaluate_citation_accuracy(
            query_result.citations, test_case.expected_files
        )
        
        # Evaluar calidad de respuesta
        quality_scores = self.quality_evaluator.evaluate_response_quality(
            query_result.answer, test_case.query
        )
        
        # Calcular score general
        overall_score = self._calculate_overall_score(
            file_precision, file_recall,
            function_precision, function_recall,
            class_precision, class_recall,
            concept_coverage, citation_accuracy,
            quality_scores, test_case.difficulty
        )
        
        return RepoEvaluationResult(
            test_case_id=test_case.id,
            query=test_case.query,
            answer=query_result.answer,
            citations=query_result.citations,
            file_precision=file_precision,
            file_recall=file_recall,
            function_precision=function_precision,
            function_recall=function_recall,
            class_precision=class_precision,
            class_recall=class_recall,
            concept_coverage=concept_coverage,
            citation_accuracy=citation_accuracy,
            response_quality=statistics.mean(quality_scores.values()),
            relevance_score=quality_scores.get('relevance', 0.0),
            response_time=query_result.response_time,
            confidence=query_result.confidence,
            cost_usd=query_result.cost_usd,
            overall_score=overall_score
        )
    
    def _calculate_overall_score(self, 
                               file_precision: float, file_recall: float,
                               function_precision: float, function_recall: float,
                               class_precision: float, class_recall: float,
                               concept_coverage: float, citation_accuracy: float,
                               quality_scores: Dict[str, float], difficulty: str) -> float:
        """Calcular score general ponderado"""
        
        # F1 scores para precision/recall
        file_f1 = 2 * file_precision * file_recall / (file_precision + file_recall) if (file_precision + file_recall) > 0 else 0
        function_f1 = 2 * function_precision * function_recall / (function_precision + function_recall) if (function_precision + function_recall) > 0 else 0
        class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall) if (class_precision + class_recall) > 0 else 0
        
        # Ponderaciones por componente
        weights = {
            'file_understanding': 0.20,
            'function_understanding': 0.25,
            'class_understanding': 0.15,
            'concept_coverage': 0.15,
            'citation_accuracy': 0.10,
            'response_quality': 0.15
        }
        
        score = (
            file_f1 * weights['file_understanding'] +
            function_f1 * weights['function_understanding'] +
            class_f1 * weights['class_understanding'] +
            concept_coverage * weights['concept_coverage'] +
            citation_accuracy * weights['citation_accuracy'] +
            statistics.mean(quality_scores.values()) * weights['response_quality']
        )
        
        # Ajuste por dificultad
        difficulty_multipliers = {
            'easy': 1.0,
            'medium': 1.1,
            'hard': 1.2
        }
        
        return min(1.0, score * difficulty_multipliers.get(difficulty, 1.0))
    
    def _generate_summary(self, results: List[RepoEvaluationResult], total_time: float) -> Dict[str, Any]:
        """Generar resumen de la evaluación"""
        
        if not results:
            return {}
        
        # Métricas generales
        overall_scores = [r.overall_score for r in results]
        
        summary = {
            'total_tests': len(results),
            'execution_time': total_time,
            'avg_execution_time_per_test': total_time / len(results),
            
            # Scores generales
            'overall_score': {
                'mean': statistics.mean(overall_scores),
                'median': statistics.median(overall_scores),
                'min': min(overall_scores),
                'max': max(overall_scores),
                'std': statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0
            },
            
            # Métricas específicas
            'code_understanding': {
                'avg_file_precision': statistics.mean([r.file_precision for r in results]),
                'avg_file_recall': statistics.mean([r.file_recall for r in results]),
                'avg_function_precision': statistics.mean([r.function_precision for r in results]),
                'avg_function_recall': statistics.mean([r.function_recall for r in results]),
                'avg_class_precision': statistics.mean([r.class_precision for r in results]),
                'avg_class_recall': statistics.mean([r.class_recall for r in results]),
                'avg_concept_coverage': statistics.mean([r.concept_coverage for r in results])
            },
            
            # Métricas de calidad
            'quality_metrics': {
                'avg_citation_accuracy': statistics.mean([r.citation_accuracy for r in results]),
                'avg_response_quality': statistics.mean([r.response_quality for r in results]),
                'avg_relevance': statistics.mean([r.relevance_score for r in results])
            },
            
            # Métricas de performance
            'performance_metrics': {
                'avg_response_time': statistics.mean([r.response_time for r in results]),
                'avg_confidence': statistics.mean([r.confidence for r in results]),
                'total_cost': sum([r.cost_usd for r in results]),
                'avg_cost_per_query': statistics.mean([r.cost_usd for r in results])
            }
        }
        
        # Análisis por dificultad
        difficulty_analysis = {}
        test_cases_by_difficulty = {}
        
        for result in results:
            # Encontrar el test case correspondiente
            test_case = next((tc for tc in self.test_cases if tc.id == result.test_case_id), None)
            if test_case:
                difficulty = test_case.difficulty
                if difficulty not in test_cases_by_difficulty:
                    test_cases_by_difficulty[difficulty] = []
                test_cases_by_difficulty[difficulty].append(result)
        
        for difficulty, difficulty_results in test_cases_by_difficulty.items():
            difficulty_scores = [r.overall_score for r in difficulty_results]
            difficulty_analysis[difficulty] = {
                'count': len(difficulty_results),
                'avg_score': statistics.mean(difficulty_scores),
                'pass_rate': sum(1 for score in difficulty_scores if score >= 0.7) / len(difficulty_scores)
            }
        
        summary['difficulty_analysis'] = difficulty_analysis
        
        return summary
    
    def export_results(self, results: Dict[str, Any], filename: str = "evaluation_results.json"):
        """Exportar resultados a archivo JSON"""
        
        # Hacer serializable
        serializable_results = self._make_serializable(results)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Evaluation results exported to {filename}")
    
    def _make_serializable(self, obj):
        """Convertir objetos a formato serializable"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (RepoTestCase, RepoEvaluationResult)):
            return asdict(obj)
        else:
            return obj
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generar reporte de evaluación legible"""
        
        summary = results['summary']
        
        report = f"""
RepoGPT Evaluation Report
========================

📊 Overall Performance:
  Total Tests: {summary['total_tests']}
  Execution Time: {summary['execution_time']:.2f}s
  Average Score: {summary['overall_score']['mean']:.3f}
  Score Range: {summary['overall_score']['min']:.3f} - {summary['overall_score']['max']:.3f}

🧠 Code Understanding:
  File Understanding: P={summary['code_understanding']['avg_file_precision']:.3f}, R={summary['code_understanding']['avg_file_recall']:.3f}
  Function Understanding: P={summary['code_understanding']['avg_function_precision']:.3f}, R={summary['code_understanding']['avg_function_recall']:.3f}
  Class Understanding: P={summary['code_understanding']['avg_class_precision']:.3f}, R={summary['code_understanding']['avg_class_recall']:.3f}
  Concept Coverage: {summary['code_understanding']['avg_concept_coverage']:.3f}

📝 Response Quality:
  Citation Accuracy: {summary['quality_metrics']['avg_citation_accuracy']:.3f}
  Response Quality: {summary['quality_metrics']['avg_response_quality']:.3f}
  Relevance: {summary['quality_metrics']['avg_relevance']:.3f}

⚡ Performance:
  Avg Response Time: {summary['performance_metrics']['avg_response_time']:.3f}s
  Avg Confidence: {summary['performance_metrics']['avg_confidence']:.3f}
  Total Cost: ${summary['performance_metrics']['total_cost']:.4f}
  Cost per Query: ${summary['performance_metrics']['avg_cost_per_query']:.4f}

📈 Difficulty Analysis:
"""
        
        for difficulty, analysis in summary.get('difficulty_analysis', {}).items():
            report += f"  {difficulty.title()}: {analysis['avg_score']:.3f} avg, {analysis['pass_rate']:.1%} pass rate ({analysis['count']} tests)\n"
        
        # Añadir detalles de tests fallidos
        failed_tests = [r for r in results['detailed_results'] if r.overall_score < 0.7]
        if failed_tests:
            report += f"\n⚠️ Failed Tests ({len(failed_tests)}):\n"
            for test in failed_tests:
                report += f"  {test.test_case_id}: {test.overall_score:.3f} - {test.query[:60]}...\n"
        
        return report

# =============================================================================
# 4. FUNCIÓN PRINCIPAL PARA EVALUACIÓN
# =============================================================================

def run_capstone_evaluation():
    """Ejecutar evaluación completa del capstone"""
    
    print("🎯 RepoGPT Capstone Evaluation")
    print("=" * 40)
    
    # Configurar agente
    config = RepoConfig(
        repo_path="capstone/sample_repo",
        repo_name="sample_repo",
        languages=['python']
    )
    
    # Inicializar agente
    agent = RepoGPTAgent(config)
    if not agent.initialize():
        print("❌ Failed to initialize agent")
        return
    
    # Crear suite de evaluación
    evaluation_suite = RepoGPTEvaluationSuite(agent)
    
    # Ejecutar evaluación
    results = evaluation_suite.run_evaluation()
    
    # Generar y mostrar reporte
    report = evaluation_suite.generate_report(results)
    print("\n" + "="*60)
    print(report)
    
    # Exportar resultados
    evaluation_suite.export_results(results)
    
    # Evaluación final
    overall_score = results['summary']['overall_score']['mean']
    
    print(f"\n🎯 CAPSTONE EVALUATION RESULT:")
    print(f"{'='*40}")
    
    if overall_score >= 0.85:
        print(f"🏆 EXCELENTE: {overall_score:.3f} - Listo para producción!")
    elif overall_score >= 0.70:
        print(f"✅ APROBADO: {overall_score:.3f} - Funcional con mejoras menores")
    elif overall_score >= 0.50:
        print(f"⚠️ NECESITA MEJORAS: {overall_score:.3f} - Requiere refinamiento")
    else:
        print(f"❌ NO APROBADO: {overall_score:.3f} - Requiere trabajo adicional")
    
    return results

if __name__ == "__main__":
    results = run_capstone_evaluation()
