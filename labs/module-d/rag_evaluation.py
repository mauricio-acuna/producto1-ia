"""
Laboratorio 3: RAG Evaluation Pipeline - Sistema Completo de Evaluación para RAG
===============================================================================

Este laboratorio implementa un pipeline completo de evaluación para sistemas RAG,
incluyendo métricas de retrieval, generación, y verificación de calidad end-to-end.

Autor: Sistema de IA Educativo
Módulo: D - Métricas y Evaluación
"""

import re
import json
import math
import time
import statistics
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict, Counter
from enum import Enum
import uuid

# =============================================================================
# 1. MODELOS DE DATOS
# =============================================================================

class RAGComponent(Enum):
    """Componentes del sistema RAG"""
    RETRIEVAL = "retrieval"
    RANKING = "ranking"
    GENERATION = "generation"
    CITATION = "citation"
    END_TO_END = "end_to_end"

@dataclass
class RAGTestCase:
    """Caso de prueba para evaluación RAG"""
    id: str
    query: str
    expected_answer: Optional[str] = None
    relevant_docs: List[str] = None  # IDs de documentos relevantes
    retrieved_docs: List[str] = None  # IDs de documentos recuperados
    generated_answer: Optional[str] = None
    citations: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.relevant_docs is None:
            self.relevant_docs = []
        if self.retrieved_docs is None:
            self.retrieved_docs = []
        if self.citations is None:
            self.citations = []
        if self.metadata is None:
            self.metadata = {}

@dataclass
class RetrievalMetrics:
    """Métricas de evaluación para retrieval"""
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    f1_at_k: Dict[int, float]
    mean_reciprocal_rank: float
    ndcg_at_k: Dict[int, float]
    hit_rate: float
    total_relevant: int
    total_retrieved: int

@dataclass
class GenerationMetrics:
    """Métricas de evaluación para generación"""
    bleu_score: float
    rouge_1: Dict[str, float]
    rouge_2: Dict[str, float]
    rouge_l: Dict[str, float]
    semantic_similarity: float
    fluency_score: float
    relevance_score: float
    factual_consistency: float

@dataclass
class CitationMetrics:
    """Métricas de evaluación para citas"""
    citation_precision: float
    citation_recall: float
    citation_f1: float
    citation_accuracy: float
    citation_coverage: float
    avg_citations_per_response: float
    valid_citation_ratio: float

@dataclass
class RAGEvaluationResult:
    """Resultado completo de evaluación RAG"""
    test_case_id: str
    component: RAGComponent
    retrieval_metrics: Optional[RetrievalMetrics] = None
    generation_metrics: Optional[GenerationMetrics] = None
    citation_metrics: Optional[CitationMetrics] = None
    overall_score: float = 0.0
    execution_time: float = 0.0
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

# =============================================================================
# 2. EVALUADOR DE RETRIEVAL
# =============================================================================

class RetrievalEvaluator:
    """Evaluador para componente de retrieval"""
    
    def __init__(self, k_values: List[int] = [1, 3, 5, 10]):
        self.k_values = k_values
    
    def evaluate(self, 
                test_case: RAGTestCase,
                relevance_scores: Optional[Dict[str, float]] = None) -> RetrievalMetrics:
        """Evaluar métricas de retrieval"""
        
        retrieved = test_case.retrieved_docs
        relevant = set(test_case.relevant_docs)
        
        if not retrieved:
            return self._empty_metrics()
        
        # Calcular métricas para diferentes valores de K
        precision_at_k = {}
        recall_at_k = {}
        f1_at_k = {}
        ndcg_at_k = {}
        
        for k in self.k_values:
            precision_at_k[k] = self._precision_at_k(retrieved, relevant, k)
            recall_at_k[k] = self._recall_at_k(retrieved, relevant, k)
            f1_at_k[k] = self._f1_score(precision_at_k[k], recall_at_k[k])
            
            if relevance_scores:
                ndcg_at_k[k] = self._ndcg_at_k(retrieved, relevance_scores, k)
            else:
                ndcg_at_k[k] = 0.0
        
        # Otras métricas
        mrr = self._mean_reciprocal_rank(retrieved, relevant)
        hit_rate = self._hit_rate(retrieved, relevant)
        
        return RetrievalMetrics(
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            f1_at_k=f1_at_k,
            mean_reciprocal_rank=mrr,
            ndcg_at_k=ndcg_at_k,
            hit_rate=hit_rate,
            total_relevant=len(relevant),
            total_retrieved=len(retrieved)
        )
    
    def _precision_at_k(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        """Precision@K"""
        if k <= 0 or not retrieved:
            return 0.0
        
        top_k = retrieved[:k]
        relevant_in_top_k = len([doc for doc in top_k if doc in relevant])
        return relevant_in_top_k / len(top_k)
    
    def _recall_at_k(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        """Recall@K"""
        if k <= 0 or not relevant:
            return 0.0
        
        top_k = retrieved[:k]
        relevant_in_top_k = len([doc for doc in top_k if doc in relevant])
        return relevant_in_top_k / len(relevant)
    
    def _f1_score(self, precision: float, recall: float) -> float:
        """F1 Score"""
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    
    def _mean_reciprocal_rank(self, retrieved: List[str], relevant: Set[str]) -> float:
        """Mean Reciprocal Rank"""
        for i, doc in enumerate(retrieved, 1):
            if doc in relevant:
                return 1.0 / i
        return 0.0
    
    def _ndcg_at_k(self, retrieved: List[str], relevance_scores: Dict[str, float], k: int) -> float:
        """Normalized Discounted Cumulative Gain@K"""
        if k <= 0:
            return 0.0
        
        def dcg(scores: List[float]) -> float:
            return sum(score / math.log2(i + 2) for i, score in enumerate(scores))
        
        # DCG real
        actual_scores = [relevance_scores.get(doc, 0.0) for doc in retrieved[:k]]
        actual_dcg = dcg(actual_scores)
        
        # DCG ideal
        ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
        ideal_dcg = dcg(ideal_scores)
        
        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
    
    def _hit_rate(self, retrieved: List[str], relevant: Set[str]) -> float:
        """Hit Rate (si hay al menos un documento relevante recuperado)"""
        return 1.0 if any(doc in relevant for doc in retrieved) else 0.0
    
    def _empty_metrics(self) -> RetrievalMetrics:
        """Métricas vacías"""
        return RetrievalMetrics(
            precision_at_k={k: 0.0 for k in self.k_values},
            recall_at_k={k: 0.0 for k in self.k_values},
            f1_at_k={k: 0.0 for k in self.k_values},
            mean_reciprocal_rank=0.0,
            ndcg_at_k={k: 0.0 for k in self.k_values},
            hit_rate=0.0,
            total_relevant=0,
            total_retrieved=0
        )

# =============================================================================
# 3. EVALUADOR DE GENERACIÓN
# =============================================================================

class GenerationEvaluator:
    """Evaluador para componente de generación"""
    
    def __init__(self):
        pass
    
    def evaluate(self, test_case: RAGTestCase) -> GenerationMetrics:
        """Evaluar métricas de generación"""
        
        if not test_case.generated_answer or not test_case.expected_answer:
            return self._empty_metrics()
        
        generated = test_case.generated_answer
        expected = test_case.expected_answer
        
        # BLEU score
        bleu = self._bleu_score(expected, generated)
        
        # ROUGE scores
        rouge_1 = self._rouge_n(expected, generated, n=1)
        rouge_2 = self._rouge_n(expected, generated, n=2)
        rouge_l = self._rouge_l(expected, generated)
        
        # Semantic similarity (simplified)
        semantic_sim = self._semantic_similarity(expected, generated)
        
        # Quality scores
        fluency = self._fluency_score(generated)
        relevance = self._relevance_score(test_case.query, generated)
        
        # Factual consistency with sources
        factual_consistency = self._factual_consistency(generated, test_case.retrieved_docs)
        
        return GenerationMetrics(
            bleu_score=bleu,
            rouge_1=rouge_1,
            rouge_2=rouge_2,
            rouge_l=rouge_l,
            semantic_similarity=semantic_sim,
            fluency_score=fluency,
            relevance_score=relevance,
            factual_consistency=factual_consistency
        )
    
    def _bleu_score(self, reference: str, candidate: str, n: int = 1) -> float:
        """BLEU score simplificado (n-grama)"""
        ref_tokens = self._tokenize(reference.lower())
        cand_tokens = self._tokenize(candidate.lower())
        
        if not cand_tokens:
            return 0.0
        
        # N-gramas
        ref_ngrams = self._get_ngrams(ref_tokens, n)
        cand_ngrams = self._get_ngrams(cand_tokens, n)
        
        if not cand_ngrams:
            return 0.0
        
        matches = sum(min(ref_ngrams[ngram], cand_ngrams[ngram]) 
                     for ngram in cand_ngrams if ngram in ref_ngrams)
        
        return matches / len(cand_tokens)
    
    def _rouge_n(self, reference: str, candidate: str, n: int = 1) -> Dict[str, float]:
        """ROUGE-N score"""
        ref_tokens = self._tokenize(reference.lower())
        cand_tokens = self._tokenize(candidate.lower())
        
        ref_ngrams = self._get_ngrams(ref_tokens, n)
        cand_ngrams = self._get_ngrams(cand_tokens, n)
        
        if not ref_ngrams and not cand_ngrams:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        
        if not cand_ngrams:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        matches = sum(min(ref_ngrams[ngram], cand_ngrams[ngram]) 
                     for ngram in cand_ngrams if ngram in ref_ngrams)
        
        precision = matches / sum(cand_ngrams.values()) if cand_ngrams else 0.0
        recall = matches / sum(ref_ngrams.values()) if ref_ngrams else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {"precision": precision, "recall": recall, "f1": f1}
    
    def _rouge_l(self, reference: str, candidate: str) -> Dict[str, float]:
        """ROUGE-L (Longest Common Subsequence)"""
        ref_tokens = self._tokenize(reference.lower())
        cand_tokens = self._tokenize(candidate.lower())
        
        lcs_length = self._lcs_length(ref_tokens, cand_tokens)
        
        if not ref_tokens and not cand_tokens:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        
        precision = lcs_length / len(cand_tokens) if cand_tokens else 0.0
        recall = lcs_length / len(ref_tokens) if ref_tokens else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {"precision": precision, "recall": recall, "f1": f1}
    
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Similitud semántica simplificada usando overlap de palabras"""
        words1 = set(self._tokenize(text1.lower()))
        words2 = set(self._tokenize(text2.lower()))
        
        if not words1 and not words2:
            return 1.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _fluency_score(self, text: str) -> float:
        """Score de fluidez básico"""
        if not text.strip():
            return 0.0
        
        # Métricas básicas de fluidez
        words = self._tokenize(text)
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        if not words or not sentences:
            return 0.0
        
        # Promedios
        avg_words_per_sentence = len(words) / len(sentences)
        unique_word_ratio = len(set(words)) / len(words)
        
        # Score basado en heurísticas
        score = 1.0
        
        # Penalizar oraciones muy largas o muy cortas
        if avg_words_per_sentence > 25 or avg_words_per_sentence < 3:
            score -= 0.3
        
        # Penalizar baja diversidad léxica
        if unique_word_ratio < 0.5:
            score -= 0.3
        
        # Penalizar texto muy corto
        if len(words) < 5:
            score -= 0.4
        
        return max(0.0, score)
    
    def _relevance_score(self, query: str, answer: str) -> float:
        """Score de relevancia entre query y respuesta"""
        query_words = set(self._tokenize(query.lower()))
        answer_words = set(self._tokenize(answer.lower()))
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words & answer_words)
        return overlap / len(query_words)
    
    def _factual_consistency(self, answer: str, source_docs: List[str]) -> float:
        """Verificar consistencia factual básica con fuentes"""
        if not source_docs or not answer.strip():
            return 0.0
        
        answer_words = set(self._tokenize(answer.lower()))
        
        # Verificar overlap con fuentes
        total_overlap = 0
        total_source_words = 0
        
        for doc_id in source_docs[:3]:  # Considerar solo primeras 3 fuentes
            # En implementación real, aquí accederíamos al contenido del documento
            # Por ahora simulamos contenido basado en el ID
            doc_content = f"documento sobre {doc_id} con información relevante"
            doc_words = set(self._tokenize(doc_content.lower()))
            
            overlap = len(answer_words & doc_words)
            total_overlap += overlap
            total_source_words += len(doc_words)
        
        if total_source_words == 0:
            return 0.0
        
        return min(1.0, total_overlap / len(answer_words)) if answer_words else 0.0
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenización simple"""
        return re.findall(r'\w+', text)
    
    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Generar n-gramas"""
        ngrams = Counter()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams[ngram] += 1
        return ngrams
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Longitud de la subsecuencia común más larga"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _empty_metrics(self) -> GenerationMetrics:
        """Métricas vacías"""
        return GenerationMetrics(
            bleu_score=0.0,
            rouge_1={"precision": 0.0, "recall": 0.0, "f1": 0.0},
            rouge_2={"precision": 0.0, "recall": 0.0, "f1": 0.0},
            rouge_l={"precision": 0.0, "recall": 0.0, "f1": 0.0},
            semantic_similarity=0.0,
            fluency_score=0.0,
            relevance_score=0.0,
            factual_consistency=0.0
        )

# =============================================================================
# 4. EVALUADOR DE CITAS
# =============================================================================

class CitationEvaluator:
    """Evaluador para sistema de citas"""
    
    def __init__(self):
        self.citation_pattern = r'\[([^\[\]]+)#L(\d+)-L(\d+)\]'
    
    def evaluate(self, test_case: RAGTestCase) -> CitationMetrics:
        """Evaluar métricas de citas"""
        
        if not test_case.generated_answer:
            return self._empty_metrics()
        
        # Extraer citas del texto generado
        found_citations = self._extract_citations(test_case.generated_answer)
        expected_citations = set(test_case.citations) if test_case.citations else set()
        
        # Métricas básicas
        precision = self._citation_precision(found_citations, expected_citations)
        recall = self._citation_recall(found_citations, expected_citations)
        f1 = self._f1_score(precision, recall)
        
        # Accuracy de formato de citas
        accuracy = self._citation_accuracy(found_citations)
        
        # Coverage: cuántos documentos recuperados tienen citas
        coverage = self._citation_coverage(found_citations, test_case.retrieved_docs)
        
        # Estadísticas adicionales
        avg_citations = len(found_citations)
        valid_ratio = sum(1 for c in found_citations if self._is_valid_citation(c)) / len(found_citations) if found_citations else 0.0
        
        return CitationMetrics(
            citation_precision=precision,
            citation_recall=recall,
            citation_f1=f1,
            citation_accuracy=accuracy,
            citation_coverage=coverage,
            avg_citations_per_response=avg_citations,
            valid_citation_ratio=valid_ratio
        )
    
    def _extract_citations(self, text: str) -> List[Dict[str, Any]]:
        """Extraer citas del texto"""
        citations = []
        
        for match in re.finditer(self.citation_pattern, text):
            doc_id = match.group(1)
            start_line = int(match.group(2))
            end_line = int(match.group(3))
            
            citations.append({
                "document": doc_id,
                "start_line": start_line,
                "end_line": end_line,
                "full_citation": match.group(0),
                "valid": self._is_valid_citation_format(start_line, end_line)
            })
        
        return citations
    
    def _citation_precision(self, found_citations: List[Dict[str, Any]], expected_citations: Set[str]) -> float:
        """Precisión de citas"""
        if not found_citations:
            return 0.0
        
        correct_citations = 0
        for citation in found_citations:
            if citation["full_citation"] in expected_citations:
                correct_citations += 1
        
        return correct_citations / len(found_citations)
    
    def _citation_recall(self, found_citations: List[Dict[str, Any]], expected_citations: Set[str]) -> float:
        """Recall de citas"""
        if not expected_citations:
            return 1.0 if not found_citations else 0.0
        
        found_citation_strs = {c["full_citation"] for c in found_citations}
        correct_citations = len(found_citation_strs & expected_citations)
        
        return correct_citations / len(expected_citations)
    
    def _citation_accuracy(self, citations: List[Dict[str, Any]]) -> float:
        """Accuracy del formato de citas"""
        if not citations:
            return 0.0
        
        valid_citations = sum(1 for c in citations if c["valid"])
        return valid_citations / len(citations)
    
    def _citation_coverage(self, citations: List[Dict[str, Any]], retrieved_docs: List[str]) -> float:
        """Coverage: proporción de documentos recuperados que tienen citas"""
        if not retrieved_docs:
            return 0.0
        
        cited_docs = {c["document"] for c in citations}
        covered_docs = len(cited_docs & set(retrieved_docs))
        
        return covered_docs / len(retrieved_docs)
    
    def _is_valid_citation(self, citation: Dict[str, Any]) -> bool:
        """Verificar si una cita es válida"""
        return citation.get("valid", False)
    
    def _is_valid_citation_format(self, start_line: int, end_line: int) -> bool:
        """Verificar formato válido de cita"""
        return start_line > 0 and end_line >= start_line
    
    def _f1_score(self, precision: float, recall: float) -> float:
        """F1 Score"""
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    
    def _empty_metrics(self) -> CitationMetrics:
        """Métricas vacías"""
        return CitationMetrics(
            citation_precision=0.0,
            citation_recall=0.0,
            citation_f1=0.0,
            citation_accuracy=0.0,
            citation_coverage=0.0,
            avg_citations_per_response=0.0,
            valid_citation_ratio=0.0
        )

# =============================================================================
# 5. EVALUADOR RAG COMPLETO
# =============================================================================

class RAGEvaluator:
    """Evaluador completo para sistemas RAG"""
    
    def __init__(self, k_values: List[int] = [1, 3, 5, 10]):
        self.retrieval_evaluator = RetrievalEvaluator(k_values)
        self.generation_evaluator = GenerationEvaluator()
        self.citation_evaluator = CitationEvaluator()
    
    def evaluate_test_case(self, 
                          test_case: RAGTestCase,
                          components: List[RAGComponent] = None,
                          relevance_scores: Optional[Dict[str, float]] = None) -> List[RAGEvaluationResult]:
        """Evaluar un caso de prueba completo"""
        
        if components is None:
            components = [RAGComponent.RETRIEVAL, RAGComponent.GENERATION, RAGComponent.CITATION]
        
        results = []
        
        for component in components:
            start_time = time.time()
            
            if component == RAGComponent.RETRIEVAL:
                metrics = self.retrieval_evaluator.evaluate(test_case, relevance_scores)
                overall_score = self._calculate_retrieval_score(metrics)
                
                result = RAGEvaluationResult(
                    test_case_id=test_case.id,
                    component=component,
                    retrieval_metrics=metrics,
                    overall_score=overall_score,
                    execution_time=time.time() - start_time
                )
                
            elif component == RAGComponent.GENERATION:
                metrics = self.generation_evaluator.evaluate(test_case)
                overall_score = self._calculate_generation_score(metrics)
                
                result = RAGEvaluationResult(
                    test_case_id=test_case.id,
                    component=component,
                    generation_metrics=metrics,
                    overall_score=overall_score,
                    execution_time=time.time() - start_time
                )
                
            elif component == RAGComponent.CITATION:
                metrics = self.citation_evaluator.evaluate(test_case)
                overall_score = self._calculate_citation_score(metrics)
                
                result = RAGEvaluationResult(
                    test_case_id=test_case.id,
                    component=component,
                    citation_metrics=metrics,
                    overall_score=overall_score,
                    execution_time=time.time() - start_time
                )
                
            else:
                # End-to-end evaluation
                retrieval_metrics = self.retrieval_evaluator.evaluate(test_case, relevance_scores)
                generation_metrics = self.generation_evaluator.evaluate(test_case)
                citation_metrics = self.citation_evaluator.evaluate(test_case)
                
                overall_score = self._calculate_end_to_end_score(
                    retrieval_metrics, generation_metrics, citation_metrics
                )
                
                result = RAGEvaluationResult(
                    test_case_id=test_case.id,
                    component=component,
                    retrieval_metrics=retrieval_metrics,
                    generation_metrics=generation_metrics,
                    citation_metrics=citation_metrics,
                    overall_score=overall_score,
                    execution_time=time.time() - start_time
                )
            
            results.append(result)
        
        return results
    
    def evaluate_dataset(self, 
                        test_cases: List[RAGTestCase],
                        components: List[RAGComponent] = None) -> Dict[str, Any]:
        """Evaluar un dataset completo"""
        
        print(f"🚀 Evaluating RAG system with {len(test_cases)} test cases...")
        
        all_results = []
        component_scores = defaultdict(list)
        
        for i, test_case in enumerate(test_cases):
            print(f"  📝 Processing test case {i+1}/{len(test_cases)}: {test_case.id}")
            
            case_results = self.evaluate_test_case(test_case, components)
            all_results.extend(case_results)
            
            for result in case_results:
                component_scores[result.component].append(result.overall_score)
        
        # Calcular métricas agregadas
        aggregated_metrics = self._aggregate_results(all_results)
        
        # Summary por componente
        component_summary = {}
        for component, scores in component_scores.items():
            component_summary[component.value] = {
                "avg_score": statistics.mean(scores),
                "median_score": statistics.median(scores),
                "min_score": min(scores),
                "max_score": max(scores),
                "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                "count": len(scores)
            }
        
        return {
            "dataset_summary": {
                "total_test_cases": len(test_cases),
                "total_evaluations": len(all_results),
                "avg_execution_time": statistics.mean([r.execution_time for r in all_results]),
                "total_execution_time": sum(r.execution_time for r in all_results)
            },
            "component_summary": component_summary,
            "aggregated_metrics": aggregated_metrics,
            "detailed_results": all_results[:20]  # Limitar resultados detallados
        }
    
    def _calculate_retrieval_score(self, metrics: RetrievalMetrics) -> float:
        """Calcular score agregado para retrieval"""
        if not metrics.precision_at_k:
            return 0.0
        
        # Weighted average of key metrics
        p_at_5 = metrics.precision_at_k.get(5, 0.0)
        r_at_5 = metrics.recall_at_k.get(5, 0.0)
        mrr = metrics.mean_reciprocal_rank
        hit_rate = metrics.hit_rate
        
        return (p_at_5 * 0.3 + r_at_5 * 0.3 + mrr * 0.25 + hit_rate * 0.15)
    
    def _calculate_generation_score(self, metrics: GenerationMetrics) -> float:
        """Calcular score agregado para generación"""
        rouge_1_f1 = metrics.rouge_1.get("f1", 0.0)
        semantic_sim = metrics.semantic_similarity
        fluency = metrics.fluency_score
        relevance = metrics.relevance_score
        factual = metrics.factual_consistency
        
        return (rouge_1_f1 * 0.25 + semantic_sim * 0.2 + fluency * 0.2 + 
                relevance * 0.2 + factual * 0.15)
    
    def _calculate_citation_score(self, metrics: CitationMetrics) -> float:
        """Calcular score agregado para citas"""
        precision = metrics.citation_precision
        recall = metrics.citation_recall
        accuracy = metrics.citation_accuracy
        coverage = metrics.citation_coverage
        
        return (precision * 0.3 + recall * 0.25 + accuracy * 0.25 + coverage * 0.2)
    
    def _calculate_end_to_end_score(self, 
                                   retrieval: RetrievalMetrics,
                                   generation: GenerationMetrics,
                                   citation: CitationMetrics) -> float:
        """Calcular score end-to-end"""
        retrieval_score = self._calculate_retrieval_score(retrieval)
        generation_score = self._calculate_generation_score(generation)
        citation_score = self._calculate_citation_score(citation)
        
        # Weighted combination
        return (retrieval_score * 0.4 + generation_score * 0.4 + citation_score * 0.2)
    
    def _aggregate_results(self, results: List[RAGEvaluationResult]) -> Dict[str, Any]:
        """Agregar resultados por componente"""
        aggregated = {}
        
        # Group by component
        component_groups = defaultdict(list)
        for result in results:
            component_groups[result.component].append(result)
        
        for component, component_results in component_groups.items():
            comp_name = component.value
            
            if component == RAGComponent.RETRIEVAL:
                aggregated[comp_name] = self._aggregate_retrieval_metrics(component_results)
            elif component == RAGComponent.GENERATION:
                aggregated[comp_name] = self._aggregate_generation_metrics(component_results)
            elif component == RAGComponent.CITATION:
                aggregated[comp_name] = self._aggregate_citation_metrics(component_results)
        
        return aggregated
    
    def _aggregate_retrieval_metrics(self, results: List[RAGEvaluationResult]) -> Dict[str, Any]:
        """Agregar métricas de retrieval"""
        metrics = [r.retrieval_metrics for r in results if r.retrieval_metrics]
        
        if not metrics:
            return {}
        
        # Average precision@k, recall@k, etc.
        k_values = list(metrics[0].precision_at_k.keys())
        
        avg_precision_at_k = {}
        avg_recall_at_k = {}
        avg_f1_at_k = {}
        
        for k in k_values:
            precisions = [m.precision_at_k[k] for m in metrics]
            recalls = [m.recall_at_k[k] for m in metrics]
            f1s = [m.f1_at_k[k] for m in metrics]
            
            avg_precision_at_k[k] = statistics.mean(precisions)
            avg_recall_at_k[k] = statistics.mean(recalls)
            avg_f1_at_k[k] = statistics.mean(f1s)
        
        return {
            "avg_precision_at_k": avg_precision_at_k,
            "avg_recall_at_k": avg_recall_at_k,
            "avg_f1_at_k": avg_f1_at_k,
            "avg_mrr": statistics.mean([m.mean_reciprocal_rank for m in metrics]),
            "avg_hit_rate": statistics.mean([m.hit_rate for m in metrics])
        }
    
    def _aggregate_generation_metrics(self, results: List[RAGEvaluationResult]) -> Dict[str, Any]:
        """Agregar métricas de generación"""
        metrics = [r.generation_metrics for r in results if r.generation_metrics]
        
        if not metrics:
            return {}
        
        return {
            "avg_bleu": statistics.mean([m.bleu_score for m in metrics]),
            "avg_rouge_1_f1": statistics.mean([m.rouge_1["f1"] for m in metrics]),
            "avg_rouge_2_f1": statistics.mean([m.rouge_2["f1"] for m in metrics]),
            "avg_semantic_similarity": statistics.mean([m.semantic_similarity for m in metrics]),
            "avg_fluency": statistics.mean([m.fluency_score for m in metrics]),
            "avg_relevance": statistics.mean([m.relevance_score for m in metrics]),
            "avg_factual_consistency": statistics.mean([m.factual_consistency for m in metrics])
        }
    
    def _aggregate_citation_metrics(self, results: List[RAGEvaluationResult]) -> Dict[str, Any]:
        """Agregar métricas de citas"""
        metrics = [r.citation_metrics for r in results if r.citation_metrics]
        
        if not metrics:
            return {}
        
        return {
            "avg_citation_precision": statistics.mean([m.citation_precision for m in metrics]),
            "avg_citation_recall": statistics.mean([m.citation_recall for m in metrics]),
            "avg_citation_f1": statistics.mean([m.citation_f1 for m in metrics]),
            "avg_citation_accuracy": statistics.mean([m.citation_accuracy for m in metrics]),
            "avg_citation_coverage": statistics.mean([m.citation_coverage for m in metrics]),
            "avg_citations_per_response": statistics.mean([m.avg_citations_per_response for m in metrics])
        }

# =============================================================================
# 6. GENERADOR DE REPORTES
# =============================================================================

class RAGReportGenerator:
    """Generador de reportes para evaluación RAG"""
    
    @staticmethod
    def generate_summary_report(evaluation_results: Dict[str, Any]) -> str:
        """Generar reporte de resumen"""
        
        report = """
RAG Evaluation Summary Report
============================

Dataset Overview:
"""
        
        dataset_summary = evaluation_results["dataset_summary"]
        report += f"- Total Test Cases: {dataset_summary['total_test_cases']}\n"
        report += f"- Total Evaluations: {dataset_summary['total_evaluations']}\n"
        report += f"- Total Execution Time: {dataset_summary['total_execution_time']:.3f}s\n"
        report += f"- Average Time per Evaluation: {dataset_summary['avg_execution_time']:.3f}s\n"
        
        report += "\nComponent Performance:\n"
        report += "=" * 22 + "\n"
        
        component_summary = evaluation_results["component_summary"]
        for component, metrics in component_summary.items():
            report += f"\n{component.upper()}:\n"
            report += f"  Average Score: {metrics['avg_score']:.3f}\n"
            report += f"  Median Score: {metrics['median_score']:.3f}\n"
            report += f"  Score Range: {metrics['min_score']:.3f} - {metrics['max_score']:.3f}\n"
            report += f"  Standard Deviation: {metrics['std_dev']:.3f}\n"
            report += f"  Evaluations: {metrics['count']}\n"
        
        # Detailed metrics if available
        aggregated = evaluation_results.get("aggregated_metrics", {})
        
        if "retrieval" in aggregated:
            report += "\nDetailed Retrieval Metrics:\n"
            ret_metrics = aggregated["retrieval"]
            for k, precision in ret_metrics.get("avg_precision_at_k", {}).items():
                recall = ret_metrics["avg_recall_at_k"][k]
                f1 = ret_metrics["avg_f1_at_k"][k]
                report += f"  P@{k}: {precision:.3f}, R@{k}: {recall:.3f}, F1@{k}: {f1:.3f}\n"
            report += f"  MRR: {ret_metrics.get('avg_mrr', 0):.3f}\n"
            report += f"  Hit Rate: {ret_metrics.get('avg_hit_rate', 0):.3f}\n"
        
        if "generation" in aggregated:
            report += "\nDetailed Generation Metrics:\n"
            gen_metrics = aggregated["generation"]
            report += f"  BLEU Score: {gen_metrics.get('avg_bleu', 0):.3f}\n"
            report += f"  ROUGE-1 F1: {gen_metrics.get('avg_rouge_1_f1', 0):.3f}\n"
            report += f"  Semantic Similarity: {gen_metrics.get('avg_semantic_similarity', 0):.3f}\n"
            report += f"  Fluency: {gen_metrics.get('avg_fluency', 0):.3f}\n"
            report += f"  Relevance: {gen_metrics.get('avg_relevance', 0):.3f}\n"
            report += f"  Factual Consistency: {gen_metrics.get('avg_factual_consistency', 0):.3f}\n"
        
        if "citation" in aggregated:
            report += "\nDetailed Citation Metrics:\n"
            cit_metrics = aggregated["citation"]
            report += f"  Citation Precision: {cit_metrics.get('avg_citation_precision', 0):.3f}\n"
            report += f"  Citation Recall: {cit_metrics.get('avg_citation_recall', 0):.3f}\n"
            report += f"  Citation F1: {cit_metrics.get('avg_citation_f1', 0):.3f}\n"
            report += f"  Citation Accuracy: {cit_metrics.get('avg_citation_accuracy', 0):.3f}\n"
            report += f"  Citation Coverage: {cit_metrics.get('avg_citation_coverage', 0):.3f}\n"
        
        return report
    
    @staticmethod
    def export_results_json(results: Dict[str, Any], filename: str):
        """Exportar resultados a JSON"""
        
        # Hacer serializable
        serializable_results = RAGReportGenerator._make_serializable(results)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"📄 RAG evaluation results exported to {filename}")
    
    @staticmethod
    def _make_serializable(obj):
        """Convertir objetos a formato serializable"""
        if isinstance(obj, dict):
            return {k: RAGReportGenerator._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [RAGReportGenerator._make_serializable(item) for item in obj]
        elif isinstance(obj, (RAGTestCase, RAGEvaluationResult, RetrievalMetrics, 
                             GenerationMetrics, CitationMetrics)):
            return asdict(obj)
        elif isinstance(obj, (RAGComponent)):
            return obj.value
        else:
            return obj

# =============================================================================
# 7. FUNCIÓN PRINCIPAL Y DEMO
# =============================================================================

def create_sample_rag_test_cases() -> List[RAGTestCase]:
    """Crear casos de prueba de ejemplo para RAG"""
    
    test_cases = [
        RAGTestCase(
            id="rag_test_001",
            query="¿Qué es machine learning?",
            expected_answer="Machine learning es una rama de la inteligencia artificial que permite a las máquinas aprender de datos sin programación explícita.",
            relevant_docs=["ml_intro.pdf", "ai_fundamentals.txt"],
            retrieved_docs=["ml_intro.pdf", "ai_fundamentals.txt", "statistics.pdf"],
            generated_answer="Machine learning es una técnica de IA que permite a los sistemas aprender automáticamente de los datos [ml_intro.pdf#L12-L15]. Es fundamental en muchas aplicaciones modernas [ai_fundamentals.txt#L3-L7].",
            citations=["[ml_intro.pdf#L12-L15]", "[ai_fundamentals.txt#L3-L7]"]
        ),
        
        RAGTestCase(
            id="rag_test_002",
            query="Explica qué es RAG en IA",
            expected_answer="RAG (Retrieval-Augmented Generation) combina recuperación de información con generación de texto para crear respuestas más precisas y verificables.",
            relevant_docs=["rag_paper.pdf", "nlp_techniques.txt"],
            retrieved_docs=["rag_paper.pdf", "nlp_techniques.txt", "transformer_models.pdf"],
            generated_answer="RAG es una arquitectura que combina retrieval y generation para mejorar la calidad de las respuestas [rag_paper.pdf#L1-L5]. Permite generar texto basado en información recuperada de fuentes externas.",
            citations=["[rag_paper.pdf#L1-L5]"]
        ),
        
        RAGTestCase(
            id="rag_test_003",
            query="¿Cómo funcionan las redes neuronales?",
            expected_answer="Las redes neuronales son modelos computacionales inspirados en el cerebro humano que procesan información a través de capas de neuronas interconectadas.",
            relevant_docs=["neural_networks.pdf", "deep_learning.txt"],
            retrieved_docs=["neural_networks.pdf", "machine_learning.txt", "deep_learning.txt"],
            generated_answer="Las redes neuronales procesan información usando múltiples capas de neuronas artificiales que aprenden patrones complejos en los datos.",
            citations=[]  # No citations for testing
        ),
        
        RAGTestCase(
            id="rag_test_004",
            query="Ventajas del deep learning",
            expected_answer="El deep learning ofrece capacidad de aprender representaciones automáticamente, manejo de datos no estructurados y alta precisión en tareas complejas.",
            relevant_docs=["deep_learning_advantages.pdf"],
            retrieved_docs=["deep_learning_advantages.pdf", "neural_networks.pdf"],
            generated_answer="El deep learning permite procesamiento automático de características, manejo de grandes volúmenes de datos y alta precisión en reconocimiento de patrones [deep_learning_advantages.pdf#L8-L12]. También es efectivo para datos no estructurados [neural_networks.pdf#L20-L25].",
            citations=["[deep_learning_advantages.pdf#L8-L12]", "[neural_networks.pdf#L20-L25]"]
        ),
        
        RAGTestCase(
            id="rag_test_005",
            query="¿Qué son los transformers?",
            expected_answer="Los transformers son una arquitectura de red neuronal basada en mecanismos de atención que ha revolucionado el procesamiento de lenguaje natural.",
            relevant_docs=["transformer_paper.pdf", "attention_mechanisms.txt"],
            retrieved_docs=["transformer_paper.pdf", "attention_mechanisms.txt", "bert_model.pdf"],
            generated_answer="Los transformers son modelos basados en self-attention que procesan secuencias de manera paralela [transformer_paper.pdf#L15-L20]. Utilizan mecanismos de atención para capturar dependencias de largo alcance [attention_mechanisms.txt#L5-L10].",
            citations=["[transformer_paper.pdf#L15-L20]", "[attention_mechanisms.txt#L5-L10]"]
        )
    ]
    
    return test_cases

def demo_rag_evaluation():
    """Demostración completa del sistema de evaluación RAG"""
    
    print("📈 RAG Evaluation Pipeline - Demo")
    print("=" * 50)
    
    # Crear casos de prueba
    test_cases = create_sample_rag_test_cases()
    print(f"📝 Created {len(test_cases)} RAG test cases")
    
    # Crear evaluador
    evaluator = RAGEvaluator(k_values=[1, 3, 5])
    
    # Evaluar dataset completo
    print(f"\n🚀 Running comprehensive RAG evaluation...")
    
    components = [RAGComponent.RETRIEVAL, RAGComponent.GENERATION, 
                 RAGComponent.CITATION, RAGComponent.END_TO_END]
    
    results = evaluator.evaluate_dataset(test_cases, components)
    
    # Generar y mostrar reporte
    print(f"\n" + "="*60)
    report = RAGReportGenerator.generate_summary_report(results)
    print(report)
    
    # Exportar resultados
    RAGReportGenerator.export_results_json(results, "rag_evaluation_results.json")
    
    return results

def demo_individual_components():
    """Demo de componentes individuales de evaluación"""
    
    print("\n🧪 Testing Individual RAG Components")
    print("=" * 40)
    
    # Crear caso de prueba
    test_case = RAGTestCase(
        id="individual_test",
        query="¿Qué es inteligencia artificial?",
        expected_answer="La inteligencia artificial es una rama de la ciencia computacional que busca crear sistemas capaces de realizar tareas que requieren inteligencia humana.",
        relevant_docs=["ai_intro.pdf", "machine_learning.txt"],
        retrieved_docs=["ai_intro.pdf", "machine_learning.txt", "statistics.pdf"],
        generated_answer="La inteligencia artificial (IA) es una disciplina que desarrolla sistemas capaces de realizar tareas cognitivas [ai_intro.pdf#L5-L8]. Incluye subcampos como machine learning y procesamiento de lenguaje natural [machine_learning.txt#L12-L15].",
        citations=["[ai_intro.pdf#L5-L8]", "[machine_learning.txt#L12-L15]"]
    )
    
    # Test retrieval
    print("\n🔍 Retrieval Evaluation:")
    retrieval_eval = RetrievalEvaluator()
    retrieval_metrics = retrieval_eval.evaluate(test_case)
    
    print(f"  Precision@3: {retrieval_metrics.precision_at_k[3]:.3f}")
    print(f"  Recall@3: {retrieval_metrics.recall_at_k[3]:.3f}")
    print(f"  MRR: {retrieval_metrics.mean_reciprocal_rank:.3f}")
    print(f"  Hit Rate: {retrieval_metrics.hit_rate:.3f}")
    
    # Test generation
    print("\n📝 Generation Evaluation:")
    generation_eval = GenerationEvaluator()
    generation_metrics = generation_eval.evaluate(test_case)
    
    print(f"  BLEU Score: {generation_metrics.bleu_score:.3f}")
    print(f"  ROUGE-1 F1: {generation_metrics.rouge_1['f1']:.3f}")
    print(f"  Semantic Similarity: {generation_metrics.semantic_similarity:.3f}")
    print(f"  Fluency: {generation_metrics.fluency_score:.3f}")
    print(f"  Relevance: {generation_metrics.relevance_score:.3f}")
    
    # Test citations
    print("\n📎 Citation Evaluation:")
    citation_eval = CitationEvaluator()
    citation_metrics = citation_eval.evaluate(test_case)
    
    print(f"  Citation Precision: {citation_metrics.citation_precision:.3f}")
    print(f"  Citation Recall: {citation_metrics.citation_recall:.3f}")
    print(f"  Citation F1: {citation_metrics.citation_f1:.3f}")
    print(f"  Citation Accuracy: {citation_metrics.citation_accuracy:.3f}")
    print(f"  Citation Coverage: {citation_metrics.citation_coverage:.3f}")

if __name__ == "__main__":
    # Ejecutar demo completo
    print("🚀 Starting RAG Evaluation Pipeline Demo\n")
    
    # Demo componentes individuales
    demo_individual_components()
    
    # Demo evaluación completa
    results = demo_rag_evaluation()
    
    print("\n🎉 RAG Evaluation Pipeline Demo Complete!")
    print("\nKey Features Demonstrated:")
    print("✅ Comprehensive retrieval metrics (Precision@K, Recall@K, NDCG, MRR)")
    print("✅ Generation quality evaluation (BLEU, ROUGE, semantic similarity)")
    print("✅ Citation system validation with canonical format")
    print("✅ End-to-end RAG pipeline evaluation")
    print("✅ Automated report generation and JSON export")
    print("✅ Component-wise and aggregated performance analysis")
    print("\n📊 Ready for production RAG system evaluation!")
