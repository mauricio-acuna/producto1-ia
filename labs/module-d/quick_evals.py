"""
Laboratorio 1: Quick Evals - Sistema de Evaluación Rápida
=========================================================

Este laboratorio implementa un sistema completo de evaluaciones rápidas (quick evals)
para sistemas de IA, incluyendo evaluaciones básicas, gates de calidad y reporting.

Autor: Sistema de IA Educativo
Módulo: D - Métricas y Evaluación
"""

import time
import json
import re
import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from abc import ABC, abstractmethod
import statistics

# =============================================================================
# 1. CLASES BASE Y ENUMS
# =============================================================================

class EvalResult(Enum):
    """Resultados posibles de una evaluación"""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    ERROR = "error"

class EvalCategory(Enum):
    """Categorías de evaluaciones"""
    CONTENT = "content"
    QUALITY = "quality"
    SAFETY = "safety"
    PERFORMANCE = "performance"
    BUSINESS = "business"

@dataclass
class QuickEvalResult:
    """Resultado de una evaluación rápida"""
    eval_name: str
    category: EvalCategory
    result: EvalResult
    score: float  # 0.0 - 1.0
    message: str
    details: Dict[str, Any]
    execution_time: float
    cost_usd: float = 0.0
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class TestCase:
    """Caso de prueba para evaluaciones"""
    id: str
    input_data: str
    expected_output: Optional[str] = None
    actual_output: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

# =============================================================================
# 2. CLASE BASE PARA EVALUACIONES
# =============================================================================

class QuickEval(ABC):
    """Clase base para evaluaciones rápidas"""
    
    def __init__(self, 
                 name: str, 
                 category: EvalCategory,
                 threshold: float = 0.7,
                 cost_per_eval: float = 0.0001):
        self.name = name
        self.category = category
        self.threshold = threshold
        self.cost_per_eval = cost_per_eval
    
    @abstractmethod
    def _evaluate_impl(self, test_case: TestCase) -> tuple[float, str, Dict[str, Any]]:
        """Implementación específica de la evaluación
        
        Returns:
            tuple: (score, message, details)
        """
        pass
    
    def evaluate(self, test_case: TestCase) -> QuickEvalResult:
        """Evaluar un caso de prueba"""
        start_time = time.time()
        
        try:
            score, message, details = self._evaluate_impl(test_case)
            result = self._determine_result(score)
            execution_time = time.time() - start_time
            
            return QuickEvalResult(
                eval_name=self.name,
                category=self.category,
                result=result,
                score=score,
                message=message,
                details=details,
                execution_time=execution_time,
                cost_usd=self.cost_per_eval
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return QuickEvalResult(
                eval_name=self.name,
                category=self.category,
                result=EvalResult.ERROR,
                score=0.0,
                message=f"Evaluation failed: {str(e)}",
                details={"error": str(e), "error_type": type(e).__name__},
                execution_time=execution_time,
                cost_usd=0.0
            )
    
    def _determine_result(self, score: float) -> EvalResult:
        """Determinar resultado basado en threshold"""
        if score >= self.threshold:
            return EvalResult.PASS
        elif score >= self.threshold * 0.8:
            return EvalResult.WARNING
        else:
            return EvalResult.FAIL

# =============================================================================
# 3. EVALUACIONES ESPECÍFICAS
# =============================================================================

class LengthEval(QuickEval):
    """Evaluación de longitud de respuesta"""
    
    def __init__(self, min_length: int = 10, max_length: int = 1000):
        super().__init__("length_check", EvalCategory.CONTENT)
        self.min_length = min_length
        self.max_length = max_length
    
    def _evaluate_impl(self, test_case: TestCase) -> tuple[float, str, Dict[str, Any]]:
        output = test_case.actual_output or ""
        length = len(output)
        
        if self.min_length <= length <= self.max_length:
            score = 1.0
            message = f"Length {length} within range"
        elif length == 0:
            score = 0.0
            message = "Empty output"
        elif length < self.min_length:
            score = length / self.min_length
            message = f"Output too short: {length} < {self.min_length}"
        else:  # length > max_length
            score = max(0.0, 1.0 - (length - self.max_length) / self.max_length)
            message = f"Output too long: {length} > {self.max_length}"
        
        details = {
            "length": length,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "within_range": self.min_length <= length <= self.max_length
        }
        
        return score, message, details

class ForbiddenContentEval(QuickEval):
    """Evaluación de contenido prohibido"""
    
    def __init__(self, forbidden_terms: List[str], case_sensitive: bool = False):
        super().__init__("forbidden_content", EvalCategory.SAFETY)
        self.forbidden_terms = forbidden_terms
        self.case_sensitive = case_sensitive
        
        if not case_sensitive:
            self.forbidden_terms = [term.lower() for term in forbidden_terms]
    
    def _evaluate_impl(self, test_case: TestCase) -> tuple[float, str, Dict[str, Any]]:
        output = test_case.actual_output or ""
        
        if not self.case_sensitive:
            check_text = output.lower()
        else:
            check_text = output
        
        found_terms = []
        for term in self.forbidden_terms:
            if term in check_text:
                found_terms.append(term)
        
        if not found_terms:
            score = 1.0
            message = "No forbidden content detected"
        else:
            # Score decreases with number of forbidden terms
            score = max(0.0, 1.0 - len(found_terms) / len(self.forbidden_terms))
            message = f"Found forbidden terms: {found_terms}"
        
        details = {
            "forbidden_terms_found": found_terms,
            "total_forbidden_count": len(found_terms),
            "searched_terms": self.forbidden_terms
        }
        
        return score, message, details

class CitationEval(QuickEval):
    """Evaluación de presencia de citas canónicas"""
    
    def __init__(self, require_citations: bool = True):
        super().__init__("citation_check", EvalCategory.QUALITY)
        self.require_citations = require_citations
    
    def _evaluate_impl(self, test_case: TestCase) -> tuple[float, str, Dict[str, Any]]:
        output = test_case.actual_output or ""
        
        # Patrón para citas canónicas: [documento#L1-L5]
        citation_pattern = r'\[([^\[\]]+)#L(\d+)-L(\d+)\]'
        citations = re.findall(citation_pattern, output)
        
        valid_citations = []
        for doc, start_line, end_line in citations:
            start_line = int(start_line)
            end_line = int(end_line)
            
            if start_line <= end_line and start_line > 0:
                valid_citations.append({
                    "document": doc,
                    "start_line": start_line,
                    "end_line": end_line,
                    "span": end_line - start_line + 1
                })
        
        total_citations = len(citations)
        valid_citation_count = len(valid_citations)
        
        if self.require_citations:
            if valid_citation_count == 0:
                score = 0.0
                message = "No valid citations found"
            else:
                score = 1.0
                message = f"Found {valid_citation_count} valid citations"
        else:
            # Si no se requieren citas, siempre pasa
            score = 1.0
            message = f"Citations optional. Found {valid_citation_count}"
        
        details = {
            "total_citations": total_citations,
            "valid_citations": valid_citation_count,
            "citation_details": valid_citations,
            "citation_ratio": valid_citation_count / total_citations if total_citations > 0 else 0
        }
        
        return score, message, details

class LanguageQualityEval(QuickEval):
    """Evaluación básica de calidad de lenguaje"""
    
    def __init__(self):
        super().__init__("language_quality", EvalCategory.QUALITY)
    
    def _evaluate_impl(self, test_case: TestCase) -> tuple[float, str, Dict[str, Any]]:
        output = test_case.actual_output or ""
        
        if not output.strip():
            return 0.0, "Empty output", {"word_count": 0}
        
        # Métricas básicas
        words = output.split()
        sentences = re.split(r'[.!?]+', output)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Calcular métricas
        word_count = len(words)
        sentence_count = len(sentences)
        avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
        
        # Detectar problemas básicos
        issues = []
        score_deductions = 0
        
        # Muy pocas palabras
        if word_count < 5:
            issues.append("Too few words")
            score_deductions += 0.3
        
        # Oraciones muy largas o muy cortas
        if avg_words_per_sentence > 30:
            issues.append("Sentences too long")
            score_deductions += 0.2
        elif avg_words_per_sentence < 3 and sentence_count > 1:
            issues.append("Sentences too short")
            score_deductions += 0.2
        
        # Repetición excesiva
        unique_words = len(set(word.lower() for word in words))
        repetition_ratio = unique_words / word_count if word_count > 0 else 0
        if repetition_ratio < 0.5 and word_count > 10:
            issues.append("High word repetition")
            score_deductions += 0.3
        
        # Score final
        score = max(0.0, 1.0 - score_deductions)
        
        if score >= 0.8:
            message = "Good language quality"
        elif issues:
            message = f"Language issues: {', '.join(issues)}"
        else:
            message = "Average language quality"
        
        details = {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_words_per_sentence": avg_words_per_sentence,
            "unique_word_ratio": repetition_ratio,
            "issues": issues,
            "score_deductions": score_deductions
        }
        
        return score, message, details

class RelevanceEval(QuickEval):
    """Evaluación de relevancia usando overlap simple"""
    
    def __init__(self):
        super().__init__("relevance_check", EvalCategory.QUALITY)
    
    def _evaluate_impl(self, test_case: TestCase) -> tuple[float, str, Dict[str, Any]]:
        input_text = test_case.input_data or ""
        output_text = test_case.actual_output or ""
        
        if not input_text or not output_text:
            return 0.0, "Missing input or output", {}
        
        # Tokenizar y normalizar
        input_words = set(word.lower().strip() for word in re.findall(r'\w+', input_text))
        output_words = set(word.lower().strip() for word in re.findall(r'\w+', output_text))
        
        # Calcular overlap
        common_words = input_words & output_words
        
        if not input_words:
            return 0.0, "No words in input", {}
        
        relevance_score = len(common_words) / len(input_words)
        
        # Ajustar score basado en longitud de output
        if len(output_words) > 0:
            coverage_score = len(common_words) / len(output_words)
            # Score combinado favorece tanto relevancia como cobertura
            score = (relevance_score + coverage_score) / 2
        else:
            score = 0.0
        
        if score >= 0.3:
            message = f"Good relevance (score: {score:.3f})"
        else:
            message = f"Low relevance (score: {score:.3f})"
        
        details = {
            "input_words": len(input_words),
            "output_words": len(output_words),
            "common_words": len(common_words),
            "relevance_score": relevance_score,
            "coverage_score": coverage_score if len(output_words) > 0 else 0,
            "common_word_list": list(common_words)[:10]  # Limit for display
        }
        
        return score, message, details

# =============================================================================
# 4. QUALITY GATES SYSTEM
# =============================================================================

class QualityGate:
    """Sistema de quality gates con múltiples evaluaciones"""
    
    def __init__(self, 
                 name: str, 
                 evaluations: List[QuickEval],
                 parallel: bool = True,
                 pass_rate_threshold: float = 0.8,
                 max_failures: int = 0):
        self.name = name
        self.evaluations = evaluations
        self.parallel = parallel
        self.pass_rate_threshold = pass_rate_threshold
        self.max_failures = max_failures
    
    def run_gate(self, test_cases: List[TestCase]) -> Dict[str, Any]:
        """Ejecutar quality gate con múltiples casos de prueba"""
        start_time = time.time()
        
        print(f"🚀 Running Quality Gate: {self.name}")
        print(f"📊 Test cases: {len(test_cases)}, Evaluations: {len(self.evaluations)}")
        
        if self.parallel:
            all_results = self._run_parallel(test_cases)
        else:
            all_results = self._run_sequential(test_cases)
        
        # Agregar resultados por evaluación
        eval_summary = self._summarize_by_evaluation(all_results)
        
        # Métricas agregadas
        total_evaluations = len(all_results)
        passed = sum(1 for r in all_results if r.result == EvalResult.PASS)
        failed = sum(1 for r in all_results if r.result == EvalResult.FAIL)
        warnings = sum(1 for r in all_results if r.result == EvalResult.WARNING)
        errors = sum(1 for r in all_results if r.result == EvalResult.ERROR)
        
        pass_rate = passed / total_evaluations if total_evaluations > 0 else 0
        
        # Determinar si el gate pasa
        gate_passed = (pass_rate >= self.pass_rate_threshold and 
                      failed <= self.max_failures and
                      errors == 0)
        
        execution_time = time.time() - start_time
        total_cost = sum(r.cost_usd for r in all_results)
        
        gate_result = {
            "gate_name": self.name,
            "gate_passed": gate_passed,
            "pass_rate": pass_rate,
            "total_evaluations": total_evaluations,
            "results_summary": {
                "passed": passed,
                "failed": failed,
                "warnings": warnings,
                "errors": errors
            },
            "execution_time": execution_time,
            "total_cost_usd": total_cost,
            "evaluation_summary": eval_summary,
            "detailed_results": all_results[:20]  # Limit detailed results
        }
        
        # Print summary
        self._print_gate_summary(gate_result)
        
        return gate_result
    
    def _run_parallel(self, test_cases: List[TestCase]) -> List[QuickEvalResult]:
        """Ejecutar evaluaciones en paralelo"""
        results = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for test_case in test_cases:
                for evaluation in self.evaluations:
                    future = executor.submit(evaluation.evaluate, test_case)
                    futures.append(future)
            
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=30)
                    results.append(result)
                except Exception as e:
                    error_result = QuickEvalResult(
                        eval_name="error",
                        category=EvalCategory.PERFORMANCE,
                        result=EvalResult.ERROR,
                        score=0.0,
                        message=f"Parallel execution failed: {str(e)}",
                        details={"error": str(e)},
                        execution_time=0.0
                    )
                    results.append(error_result)
        
        return results
    
    def _run_sequential(self, test_cases: List[TestCase]) -> List[QuickEvalResult]:
        """Ejecutar evaluaciones secuencialmente"""
        results = []
        
        for i, test_case in enumerate(test_cases):
            print(f"  📝 Processing test case {i+1}/{len(test_cases)}")
            
            for evaluation in self.evaluations:
                result = evaluation.evaluate(test_case)
                results.append(result)
        
        return results
    
    def _summarize_by_evaluation(self, results: List[QuickEvalResult]) -> Dict[str, Any]:
        """Resumir resultados por tipo de evaluación"""
        eval_groups = {}
        
        for result in results:
            if result.eval_name not in eval_groups:
                eval_groups[result.eval_name] = []
            eval_groups[result.eval_name].append(result)
        
        summary = {}
        for eval_name, eval_results in eval_groups.items():
            total = len(eval_results)
            passed = sum(1 for r in eval_results if r.result == EvalResult.PASS)
            failed = sum(1 for r in eval_results if r.result == EvalResult.FAIL)
            avg_score = statistics.mean(r.score for r in eval_results)
            avg_time = statistics.mean(r.execution_time for r in eval_results)
            
            summary[eval_name] = {
                "total": total,
                "passed": passed,
                "failed": failed,
                "pass_rate": passed / total if total > 0 else 0,
                "avg_score": avg_score,
                "avg_execution_time": avg_time
            }
        
        return summary
    
    def _print_gate_summary(self, gate_result: Dict[str, Any]):
        """Imprimir resumen del quality gate"""
        print(f"\n📋 Quality Gate Results: {gate_result['gate_name']}")
        print(f"{'='*50}")
        
        if gate_result['gate_passed']:
            print("✅ GATE PASSED")
        else:
            print("❌ GATE FAILED")
        
        print(f"\n📊 Overall Metrics:")
        print(f"  Pass Rate: {gate_result['pass_rate']:.1%}")
        print(f"  Total Evaluations: {gate_result['total_evaluations']}")
        print(f"  Execution Time: {gate_result['execution_time']:.3f}s")
        print(f"  Total Cost: ${gate_result['total_cost_usd']:.4f}")
        
        print(f"\n🎯 Results Breakdown:")
        summary = gate_result['results_summary']
        print(f"  ✅ Passed: {summary['passed']}")
        print(f"  ❌ Failed: {summary['failed']}")
        print(f"  ⚠️  Warnings: {summary['warnings']}")
        print(f"  🚫 Errors: {summary['errors']}")
        
        print(f"\n📈 By Evaluation Type:")
        for eval_name, eval_summary in gate_result['evaluation_summary'].items():
            print(f"  {eval_name}: {eval_summary['pass_rate']:.1%} pass rate "
                  f"({eval_summary['passed']}/{eval_summary['total']})")

# =============================================================================
# 5. SUITE DE EVALUACIONES
# =============================================================================

class QuickEvalSuite:
    """Suite completa de evaluaciones rápidas"""
    
    def __init__(self):
        self.evaluations = {
            "content": [
                LengthEval(min_length=20, max_length=500),
                LanguageQualityEval(),
                RelevanceEval()
            ],
            "safety": [
                ForbiddenContentEval([
                    "violencia", "hate", "spam", "offensive", 
                    "illegal", "harmful", "dangerous"
                ])
            ],
            "quality": [
                CitationEval(require_citations=True)
            ]
        }
        
        self.gates = {
            "basic": QualityGate(
                "Basic Quality Gate",
                self.evaluations["content"] + self.evaluations["safety"],
                pass_rate_threshold=0.8
            ),
            "rag": QualityGate(
                "RAG Quality Gate", 
                self.evaluations["content"] + self.evaluations["quality"],
                pass_rate_threshold=0.9
            ),
            "production": QualityGate(
                "Production Ready Gate",
                self.evaluations["content"] + self.evaluations["safety"] + self.evaluations["quality"],
                pass_rate_threshold=0.95,
                max_failures=0
            )
        }
    
    def run_suite(self, 
                  test_cases: List[TestCase], 
                  gate_name: str = "basic") -> Dict[str, Any]:
        """Ejecutar suite de evaluaciones"""
        
        if gate_name not in self.gates:
            raise ValueError(f"Gate {gate_name} not found. Available: {list(self.gates.keys())}")
        
        gate = self.gates[gate_name]
        return gate.run_gate(test_cases)
    
    def create_custom_gate(self, 
                          name: str,
                          evaluation_categories: List[str],
                          pass_rate_threshold: float = 0.8) -> QualityGate:
        """Crear gate personalizado"""
        
        evaluations = []
        for category in evaluation_categories:
            if category in self.evaluations:
                evaluations.extend(self.evaluations[category])
        
        return QualityGate(name, evaluations, pass_rate_threshold=pass_rate_threshold)

# =============================================================================
# 6. REPORTES Y EXPORTACIÓN
# =============================================================================

class EvalReporter:
    """Generador de reportes de evaluación"""
    
    @staticmethod
    def export_results_json(results: Dict[str, Any], filename: str):
        """Exportar resultados a JSON"""
        
        # Convertir objetos complejos a dict
        serializable_results = EvalReporter._make_serializable(results)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Results exported to {filename}")
    
    @staticmethod
    def _make_serializable(obj):
        """Convertir objetos a formato serializable"""
        if isinstance(obj, dict):
            return {k: EvalReporter._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [EvalReporter._make_serializable(item) for item in obj]
        elif isinstance(obj, QuickEvalResult):
            return asdict(obj)
        elif isinstance(obj, (EvalResult, EvalCategory)):
            return obj.value
        elif isinstance(obj, TestCase):
            return asdict(obj)
        else:
            return obj
    
    @staticmethod
    def generate_summary_report(results: Dict[str, Any]) -> str:
        """Generar reporte de resumen"""
        
        report = f"""
Quick Evals Summary Report
=========================

Gate: {results['gate_name']}
Status: {'✅ PASSED' if results['gate_passed'] else '❌ FAILED'}
Pass Rate: {results['pass_rate']:.1%}

Execution Details:
- Total Evaluations: {results['total_evaluations']}
- Execution Time: {results['execution_time']:.3f}s
- Total Cost: ${results['total_cost_usd']:.4f}

Results Breakdown:
- Passed: {results['results_summary']['passed']}
- Failed: {results['results_summary']['failed']}
- Warnings: {results['results_summary']['warnings']}
- Errors: {results['results_summary']['errors']}

Evaluation Performance:
"""
        
        for eval_name, eval_summary in results['evaluation_summary'].items():
            report += f"- {eval_name}: {eval_summary['pass_rate']:.1%} pass rate "
            report += f"(avg score: {eval_summary['avg_score']:.3f})\n"
        
        return report

# =============================================================================
# 7. FUNCIÓN PRINCIPAL Y TESTING
# =============================================================================

def create_sample_test_cases() -> List[TestCase]:
    """Crear casos de prueba de ejemplo"""
    
    test_cases = [
        TestCase(
            id="test_001",
            input_data="¿Qué es machine learning?",
            actual_output="Machine learning es una rama de la inteligencia artificial que permite a las máquinas aprender patrones de los datos sin ser programadas explícitamente para cada tarea específica."
        ),
        TestCase(
            id="test_002", 
            input_data="Explica qué es RAG",
            actual_output="RAG (Retrieval-Augmented Generation) es una técnica que combina recuperación de información con generación de texto. Según [documento_ai.txt#L15-L18], permite generar respuestas más precisas y verificables."
        ),
        TestCase(
            id="test_003",
            input_data="¿Cómo funciona una red neuronal?",
            actual_output="Sí."  # Respuesta muy corta para testing
        ),
        TestCase(
            id="test_004",
            input_data="Ventajas del deep learning",
            actual_output="El deep learning tiene múltiples ventajas: puede procesar grandes volúmenes de datos, detectar patrones complejos, y automatizar tareas que antes requerían programación manual. Sus aplicaciones incluyen visión por computadora, procesamiento de lenguaje natural y reconocimiento de voz [ml_guide.pdf#L42-L47]."
        ),
        TestCase(
            id="test_005",
            input_data="¿Qué es Python?",
            actual_output=""  # Output vacío para testing
        )
    ]
    
    return test_cases

def demo_quick_evals():
    """Demostración completa del sistema de quick evals"""
    
    print("🚀 Quick Evals - Sistema de Evaluación Rápida")
    print("=" * 60)
    
    # Crear casos de prueba
    test_cases = create_sample_test_cases()
    print(f"📝 Created {len(test_cases)} test cases")
    
    # Crear suite de evaluaciones
    eval_suite = QuickEvalSuite()
    
    # Ejecutar diferentes gates
    gates_to_test = ["basic", "rag", "production"]
    
    all_results = {}
    
    for gate_name in gates_to_test:
        print(f"\n{'='*60}")
        print(f"Testing Gate: {gate_name.upper()}")
        print(f"{'='*60}")
        
        results = eval_suite.run_suite(test_cases, gate_name)
        all_results[gate_name] = results
        
        # Exportar resultados
        filename = f"eval_results_{gate_name}.json"
        EvalReporter.export_results_json(results, filename)
    
    # Generar reporte comparativo
    print(f"\n{'='*60}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'='*60}")
    
    for gate_name, results in all_results.items():
        status = "✅ PASS" if results['gate_passed'] else "❌ FAIL"
        print(f"{gate_name.upper()}: {status} "
              f"(Pass Rate: {results['pass_rate']:.1%}, "
              f"Time: {results['execution_time']:.3f}s)")
    
    return all_results

def test_individual_evaluations():
    """Test de evaluaciones individuales"""
    
    print("\n🧪 Testing Individual Evaluations")
    print("=" * 40)
    
    # Crear caso de prueba
    test_case = TestCase(
        id="individual_test",
        input_data="¿Qué es inteligencia artificial?",
        actual_output="La inteligencia artificial es una rama de la ciencia computacional que busca crear sistemas capaces de realizar tareas que normalmente requieren inteligencia humana [ai_basics.pdf#L10-L15]."
    )
    
    # Test cada evaluación
    evaluations = [
        LengthEval(min_length=20, max_length=200),
        ForbiddenContentEval(["spam", "hate"]),
        CitationEval(),
        LanguageQualityEval(),
        RelevanceEval()
    ]
    
    for evaluation in evaluations:
        result = evaluation.evaluate(test_case)
        
        status_icon = "✅" if result.result == EvalResult.PASS else "❌" if result.result == EvalResult.FAIL else "⚠️"
        print(f"{status_icon} {result.eval_name}: {result.score:.3f} - {result.message}")
        print(f"   Details: {result.details}")
        print()

if __name__ == "__main__":
    # Ejecutar demostración completa
    demo_results = demo_quick_evals()
    
    # Test evaluaciones individuales
    test_individual_evaluations()
    
    print("\n🎉 Quick Evals Demo Complete!")
    print("\nKey Features Demonstrated:")
    print("✅ Multiple evaluation types (content, safety, quality)")
    print("✅ Quality gates with configurable thresholds")
    print("✅ Parallel execution for performance")
    print("✅ Detailed reporting and JSON export")
    print("✅ Error handling and edge cases")
    print("\n📊 Ready for integration with CI/CD pipelines!")
