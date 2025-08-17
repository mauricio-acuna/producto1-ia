# Módulo D: Métricas y Evaluación

## 🎯 Objetivos de Aprendizaje

Al completar este módulo, serás capaz de:

1. **Implementar quick evals** para sistemas de IA en producción
2. **Medir coste de tokens y latencia** de manera sistemática
3. **Crear gates de evaluación** para control de calidad
4. **Evaluar sistemas RAG** con métricas especializadas
5. **Implementar monitoring continuo** de performance

---

## 📊 1. Introducción a la Evaluación de IA

### 1.1 ¿Por qué Evaluar Sistemas de IA?

La evaluación es crítica porque los sistemas de IA:

- **No son determinísticos**: Mismo input puede dar outputs diferentes
- **Dependen de datos**: Calidad de entrada afecta calidad de salida
- **Evolucionan continuamente**: Modelos se actualizan, datos cambian
- **Impactan negocio**: Errores pueden ser costosos

### 1.2 Tipos de Evaluación

| Tipo | Cuándo | Objetivo | Ejemplo |
|------|--------|----------|---------|
| **Offline** | Desarrollo | Validar antes de deployment | Test sets, benchmarks |
| **Online** | Producción | Monitorear performance real | A/B tests, user feedback |
| **Human** | Casos críticos | Validar calidad subjetiva | Human ratings, expert review |
| **Automated** | Continuo | Detectar degradación | Automated metrics, alerts |

### 1.3 Pyramid de Evaluación

```
        🔺 Human Evaluation
       🔺🔺 Online Metrics  
      🔺🔺🔺 Offline Benchmarks
     🔺🔺🔺🔺 Unit Tests & Quick Evals
    🔺🔺🔺🔺🔺 System Integration Tests
```

**Base amplia**: Tests automatizados rápidos y económicos  
**Cima estrecha**: Evaluación humana costosa pero precisa

---

## ⚡ 2. Quick Evals: Evaluación Rápida

### 2.1 Concepto de Quick Evals

**Quick Evals** son evaluaciones automatizadas que:
- Se ejecutan en < 30 segundos
- Cuestan < $0.01 por evaluación
- Detectan 80% de problemas comunes
- Se pueden ejecutar en CI/CD

### 2.2 Diseño de Quick Evals

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum

class EvalResult(Enum):
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"

@dataclass
class QuickEvalResult:
    """Resultado de una evaluación rápida"""
    eval_name: str
    result: EvalResult
    score: float  # 0.0 - 1.0
    message: str
    details: Dict[str, Any]
    execution_time: float
    cost_usd: float = 0.0

class QuickEval:
    """Base class para evaluaciones rápidas"""
    
    def __init__(self, name: str, threshold: float = 0.7):
        self.name = name
        self.threshold = threshold
    
    def evaluate(self, input_data: Any, output_data: Any) -> QuickEvalResult:
        """Evaluar input/output y retornar resultado"""
        raise NotImplementedError
    
    def _determine_result(self, score: float) -> EvalResult:
        """Determinar resultado basado en threshold"""
        if score >= self.threshold:
            return EvalResult.PASS
        elif score >= self.threshold * 0.8:
            return EvalResult.WARNING
        else:
            return EvalResult.FAIL
```

### 2.3 Quick Evals Específicos para Diferentes Casos de Uso

#### 2.3.1 Evaluaciones de Contenido

**Length Check - Control de Longitud:**
```python
class LengthEval(QuickEval):
    """Evaluar longitud de respuesta apropiada"""
    
    def __init__(self, min_length: int = 10, max_length: int = 1000):
        super().__init__("length_check")
        self.min_length = min_length
        self.max_length = max_length
    
    def evaluate(self, input_data: str, output_data: str) -> QuickEvalResult:
        start_time = time.time()
        length = len(output_data)
        
        # Scoring gradual en lugar de binario
        if length < self.min_length:
            score = max(0.0, length / self.min_length)
            result = EvalResult.FAIL if score < 0.5 else EvalResult.WARNING
            message = f"Respuesta muy corta: {length} chars (mín: {self.min_length})"
        elif length > self.max_length:
            score = max(0.0, 1.0 - (length - self.max_length) / self.max_length)
            result = EvalResult.WARNING if score > 0.7 else EvalResult.FAIL
            message = f"Respuesta muy larga: {length} chars (máx: {self.max_length})"
        else:
            score = 1.0
            result = EvalResult.PASS
            message = f"Longitud apropiada: {length} chars"
        
        return QuickEvalResult(
            eval_name=self.name,
            result=result,
            score=score,
            message=message,
            details={
                "length": length, 
                "min": self.min_length, 
                "max": self.max_length,
                "words": len(output_data.split()),
                "sentences": output_data.count('.') + output_data.count('!') + output_data.count('?')
            },
            execution_time=time.time() - start_time
        )
```

**Forbidden Content - Detección de Contenido Prohibido:**
```python
class ForbiddenContentEval(QuickEval):
    """Detectar contenido prohibido o inapropiado"""
    
    def __init__(self, forbidden_terms: List[str], severity_weights: Dict[str, float] = None):
        super().__init__("forbidden_content")
        self.forbidden_terms = [term.lower() for term in forbidden_terms]
        self.severity_weights = severity_weights or {}
    
    def evaluate(self, input_data: str, output_data: str) -> QuickEvalResult:
        start_time = time.time()
        output_lower = output_data.lower()
        found_terms = []
        total_severity = 0.0
        
        for term in self.forbidden_terms:
            if term in output_lower:
                count = output_lower.count(term)
                weight = self.severity_weights.get(term, 1.0)
                severity = count * weight
                
                found_terms.append({
                    "term": term,
                    "count": count,
                    "weight": weight,
                    "severity": severity
                })
                total_severity += severity
        
        if not found_terms:
            score = 1.0
            result = EvalResult.PASS
            message = "No forbidden content detected"
        else:
            # Score basado en severidad total
            score = max(0.0, 1.0 - total_severity / 10.0)  # Normalizar a escala
            result = EvalResult.FAIL if score < 0.3 else EvalResult.WARNING
            message = f"Found {len(found_terms)} forbidden terms (severity: {total_severity:.2f})"
        
        return QuickEvalResult(
            eval_name=self.name,
            result=result,
            score=score,
            message=message,
            details={
                "found_terms": found_terms,
                "total_severity": total_severity,
                "unique_violations": len(found_terms)
            },
            execution_time=time.time() - start_time
        )

# Ejemplo de uso
forbidden_evaluator = ForbiddenContentEval(
    forbidden_terms=["password", "credit card", "ssn", "hack"],
    severity_weights={"password": 2.0, "hack": 3.0, "credit card": 5.0}
)
```

#### 2.3.2 Evaluaciones de Calidad Técnica

**JSON Validity - Validación de Formato JSON:**
```python
import json
import jsonschema

class JSONValidityEval(QuickEval):
    """Evaluar validez y estructura de JSON"""
    
    def __init__(self, required_schema: Dict = None, required_fields: List[str] = None):
        super().__init__("json_validity")
        self.required_schema = required_schema
        self.required_fields = required_fields or []
    
    def evaluate(self, input_data: str, output_data: str) -> QuickEvalResult:
        start_time = time.time()
        details = {}
        
        try:
            # 1. Verificar que es JSON válido
            parsed_json = json.loads(output_data)
            details["valid_json"] = True
            
            # 2. Verificar campos requeridos
            missing_fields = []
            for field in self.required_fields:
                if field not in parsed_json:
                    missing_fields.append(field)
            
            details["missing_fields"] = missing_fields
            details["has_all_required_fields"] = len(missing_fields) == 0
            
            # 3. Verificar schema si se proporciona
            schema_valid = True
            schema_errors = []
            if self.required_schema:
                try:
                    jsonschema.validate(parsed_json, self.required_schema)
                    details["schema_valid"] = True
                except jsonschema.ValidationError as e:
                    schema_valid = False
                    schema_errors.append(str(e))
                    details["schema_valid"] = False
                    details["schema_errors"] = schema_errors
            
            # Calcular score
            score = 0.6  # Base por JSON válido
            if len(missing_fields) == 0:
                score += 0.2
            if schema_valid:
                score += 0.2
            
            result = EvalResult.PASS if score >= 0.8 else EvalResult.WARNING
            message = f"JSON válido. Campos faltantes: {len(missing_fields)}"
            
        except json.JSONDecodeError as e:
            score = 0.0
            result = EvalResult.FAIL
            message = f"JSON inválido: {str(e)}"
            details = {"valid_json": False, "error": str(e)}
        
        return QuickEvalResult(
            eval_name=self.name,
            result=result,
            score=score,
            message=message,
            details=details,
            execution_time=time.time() - start_time
        )
```

**Relevance Check - Evaluación de Relevancia:**
```python
class RelevanceEval(QuickEval):
    """Evaluar relevancia de la respuesta a la pregunta"""
    
    def __init__(self, similarity_threshold: float = 0.3):
        super().__init__("relevance_check")
        self.threshold = similarity_threshold
    
    def _calculate_word_overlap(self, query: str, response: str) -> float:
        """Calcular overlap de palabras clave"""
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        # Filtrar palabras comunes (stop words básicos)
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        query_words -= stop_words
        response_words -= stop_words
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words & response_words)
        return overlap / len(query_words)
    
    def _check_question_answered(self, query: str, response: str) -> bool:
        """Verificar si la respuesta aborda la pregunta"""
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Buscar patrones de preguntas y respuestas correspondientes
        question_patterns = {
            "what": ["is", "are", "means", "definition"],
            "how": ["steps", "process", "way", "method"],
            "why": ["because", "reason", "due to", "since"],
            "when": ["date", "time", "year", "day"],
            "where": ["location", "place", "address", "at"]
        }
        
        for q_word, answer_indicators in question_patterns.items():
            if q_word in query_lower:
                if any(indicator in response_lower for indicator in answer_indicators):
                    return True
        
        return False
    
    def evaluate(self, input_data: str, output_data: str) -> QuickEvalResult:
        start_time = time.time()
        
        # Calcular métricas de relevancia
        word_overlap = self._calculate_word_overlap(input_data, output_data)
        question_answered = self._check_question_answered(input_data, output_data)
        
        # Score combinado
        score = word_overlap * 0.6 + (0.4 if question_answered else 0.0)
        
        if score >= 0.7:
            result = EvalResult.PASS
            message = f"Alta relevancia (score: {score:.2f})"
        elif score >= 0.4:
            result = EvalResult.WARNING
            message = f"Relevancia moderada (score: {score:.2f})"
        else:
            result = EvalResult.FAIL
            message = f"Baja relevancia (score: {score:.2f})"
        
        return QuickEvalResult(
            eval_name=self.name,
            result=result,
            score=score,
            message=message,
            details={
                "word_overlap": word_overlap,
                "question_answered": question_answered,
                "query_words": len(input_data.split()),
                "response_words": len(output_data.split())
            },
            execution_time=time.time() - start_time
        )
```

#### 2.3.3 Evaluaciones de Performance

**Response Time Check:**
```python
class ResponseTimeEval(QuickEval):
    """Evaluar tiempo de respuesta"""
    
    def __init__(self, max_time: float = 5.0, warning_time: float = 3.0):
        super().__init__("response_time")
        self.max_time = max_time
        self.warning_time = warning_time
    
    def evaluate(self, input_data: Any, output_data: Any, response_time: float = None) -> QuickEvalResult:
        if response_time is None:
            return QuickEvalResult(
                eval_name=self.name,
                result=EvalResult.FAIL,
                score=0.0,
                message="Response time not provided",
                details={},
                execution_time=0.001
            )
        
        if response_time <= self.warning_time:
            score = 1.0
            result = EvalResult.PASS
            message = f"Excelente tiempo de respuesta: {response_time:.2f}s"
        elif response_time <= self.max_time:
            score = 1.0 - (response_time - self.warning_time) / (self.max_time - self.warning_time)
            result = EvalResult.WARNING
            message = f"Tiempo de respuesta aceptable: {response_time:.2f}s"
        else:
            score = 0.0
            result = EvalResult.FAIL
            message = f"Tiempo de respuesta excesivo: {response_time:.2f}s"
        
        return QuickEvalResult(
            eval_name=self.name,
            result=result,
            score=score,
            message=message,
            details={
                "response_time": response_time,
                "max_time": self.max_time,
                "warning_time": self.warning_time,
                "performance_category": "excellent" if response_time <= 1.0 else 
                                      "good" if response_time <= 2.0 else
                                      "acceptable" if response_time <= self.max_time else "poor"
            },
            execution_time=0.001
        )
```
        for term in self.forbidden_terms:
            if term in output_lower:
                found_terms.append(term)
        
        if not found_terms:
            score = 1.0
            result = EvalResult.PASS
            message = "No forbidden content detected"
        else:
            score = 0.0
            result = EvalResult.FAIL
            message = f"Found forbidden terms: {found_terms}"
        
        return QuickEvalResult(
            eval_name=self.name,
            result=result,
            score=score,
            message=message,
            details={"forbidden_terms_found": found_terms},
            execution_time=0.002
        )
```

#### 2.3.3 Citation Check
```python
class CitationEval(QuickEval):
    """Verificar presencia de citas en respuestas RAG"""
    
    def evaluate(self, input_data: str, output_data: str) -> QuickEvalResult:
        import re
        
        # Buscar patrones de cita canónica
        citation_pattern = r'\[.*?#L\d+-L\d+\]'
        citations = re.findall(citation_pattern, output_data)
        
        if citations:
            score = 1.0
            result = EvalResult.PASS
            message = f"Found {len(citations)} citations"
        else:
            score = 0.0
            result = EvalResult.FAIL
            message = "No citations found in RAG response"
        
        return QuickEvalResult(
            eval_name=self.name,
            result=result,
            score=score,
            message=message,
            details={"citations_found": len(citations), "citations": citations},
            execution_time=0.005
        )
```

---

## 💰 3. Medición de Coste y Latencia

### 3.1 Cost Tracking

```python
from dataclasses import dataclass
import time
from typing import Optional

@dataclass
class CostMetrics:
    """Métricas de coste para llamadas a LLM"""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_input_usd: float
    cost_output_usd: float
    total_cost_usd: float
    model_name: str
    
    def cost_per_1k_tokens(self) -> float:
        """Coste por 1000 tokens"""
        return (self.total_cost_usd / self.total_tokens) * 1000 if self.total_tokens > 0 else 0

class CostCalculator:
    """Calculadora de costes para diferentes modelos"""
    
    # Precios por 1K tokens (Agosto 2025)
    MODEL_PRICING = {
        "gpt-4o": {"input": 0.0025, "output": 0.010},
        "gpt-4o-mini": {"input": 0.000150, "output": 0.000600},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "gemini-1.5-flash": {"input": 0.00007, "output": 0.00021},
        "gemini-1.5-pro": {"input": 0.00125, "output": 0.00375}
    }
    
    def calculate_cost(self, 
                      model_name: str,
                      input_tokens: int,
                      output_tokens: int) -> CostMetrics:
        """Calcular coste de una llamada"""
        
        if model_name not in self.MODEL_PRICING:
            raise ValueError(f"Modelo {model_name} no soportado")
        
        pricing = self.MODEL_PRICING[model_name]
        
        cost_input = (input_tokens / 1000) * pricing["input"]
        cost_output = (output_tokens / 1000) * pricing["output"]
        total_cost = cost_input + cost_output
        total_tokens = input_tokens + output_tokens
        
        return CostMetrics(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_input_usd=cost_input,
            cost_output_usd=cost_output,
            total_cost_usd=total_cost,
            model_name=model_name
        )
    
    def estimate_monthly_cost(self, 
                            daily_requests: int,
                            avg_input_tokens: int,
                            avg_output_tokens: int,
                            model_name: str) -> Dict[str, float]:
        """Estimar coste mensual"""
        
        single_cost = self.calculate_cost(model_name, avg_input_tokens, avg_output_tokens)
        
        daily_cost = daily_requests * single_cost.total_cost_usd
        monthly_cost = daily_cost * 30
        
        return {
            "cost_per_request": single_cost.total_cost_usd,
            "daily_cost": daily_cost,
            "monthly_cost": monthly_cost,
            "requests_per_dollar": 1.0 / single_cost.total_cost_usd if single_cost.total_cost_usd > 0 else 0
        }
```

### 3.2 Latency Measurement

```python
@dataclass
class LatencyMetrics:
    """Métricas de latencia"""
    total_time: float
    ttfb: float  # Time to First Byte
    tokens_per_second: float
    p50_latency: float
    p95_latency: float
    p99_latency: float

class LatencyTracker:
    """Tracker de latencia con percentiles"""
    
    def __init__(self):
        self.latencies: List[float] = []
        self.ttfb_times: List[float] = []
    
    def start_request(self) -> float:
        """Iniciar medición de request"""
        return time.time()
    
    def end_request(self, start_time: float, tokens_generated: int = 0) -> float:
        """Finalizar medición y retornar latencia"""
        end_time = time.time()
        latency = end_time - start_time
        
        self.latencies.append(latency)
        return latency
    
    def record_ttfb(self, ttfb: float):
        """Registrar Time to First Byte"""
        self.ttfb_times.append(ttfb)
    
    def get_metrics(self) -> LatencyMetrics:
        """Obtener métricas agregadas"""
        if not self.latencies:
            return LatencyMetrics(0, 0, 0, 0, 0, 0)
        
        latencies_sorted = sorted(self.latencies)
        n = len(latencies_sorted)
        
        return LatencyMetrics(
            total_time=sum(self.latencies),
            ttfb=sum(self.ttfb_times) / len(self.ttfb_times) if self.ttfb_times else 0,
            tokens_per_second=0,  # Requiere más data
            p50_latency=latencies_sorted[int(n * 0.5)],
            p95_latency=latencies_sorted[int(n * 0.95)],
            p99_latency=latencies_sorted[int(n * 0.99)]
        )
```

---

## 🚪 4. Gates de Evaluación

### 4.1 Concepto de Quality Gates

**Quality Gates** son checkpoints automáticos que:
- Bloquean deployments con calidad insuficiente
- Ejecutan múltiples evaluaciones en paralelo
- Proporcionan feedback inmediato
- Mantienen logs para auditoría

### 4.2 Implementación de Quality Gates

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

class QualityGate:
    """Gate de calidad con múltiples evaluaciones"""
    
    def __init__(self, name: str, evaluations: List[QuickEval], parallel: bool = True):
        self.name = name
        self.evaluations = evaluations
        self.parallel = parallel
    
    def run_gate(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Ejecutar gate de calidad"""
        start_time = time.time()
        all_results = []
        
        if self.parallel:
            results = self._run_parallel(test_cases)
        else:
            results = self._run_sequential(test_cases)
        
        # Agregar resultados
        all_results.extend(results)
        
        # Calcular métricas agregadas
        total_evaluations = len(all_results)
        passed = sum(1 for r in all_results if r.result == EvalResult.PASS)
        failed = sum(1 for r in all_results if r.result == EvalResult.FAIL)
        warnings = sum(1 for r in all_results if r.result == EvalResult.WARNING)
        
        pass_rate = passed / total_evaluations if total_evaluations > 0 else 0
        
        # Determinar si el gate pasa
        gate_passed = pass_rate >= 0.8 and failed == 0
        
        return {
            "gate_name": self.name,
            "gate_passed": gate_passed,
            "pass_rate": pass_rate,
            "total_evaluations": total_evaluations,
            "passed": passed,
            "failed": failed,
            "warnings": warnings,
            "execution_time": time.time() - start_time,
            "results": all_results
        }
    
    def _run_parallel(self, test_cases: List[Dict[str, Any]]) -> List[QuickEvalResult]:
        """Ejecutar evaluaciones en paralelo"""
        results = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for test_case in test_cases:
                for evaluation in self.evaluations:
                    future = executor.submit(
                        evaluation.evaluate,
                        test_case.get("input", ""),
                        test_case.get("output", "")
                    )
                    futures.append(future)
            
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=30)
                    results.append(result)
                except Exception as e:
                    # Crear resultado de error
                    error_result = QuickEvalResult(
                        eval_name="error",
                        result=EvalResult.FAIL,
                        score=0.0,
                        message=f"Evaluation failed: {str(e)}",
                        details={"error": str(e)},
                        execution_time=0.0
                    )
                    results.append(error_result)
        
        return results
    
    def _run_sequential(self, test_cases: List[Dict[str, Any]]) -> List[QuickEvalResult]:
        """Ejecutar evaluaciones secuencialmente"""
        results = []
        
        for test_case in test_cases:
            for evaluation in self.evaluations:
                try:
                    result = evaluation.evaluate(
                        test_case.get("input", ""),
                        test_case.get("output", "")
                    )
                    results.append(result)
                except Exception as e:
                    error_result = QuickEvalResult(
                        eval_name=evaluation.name,
                        result=EvalResult.FAIL,
                        score=0.0,
                        message=f"Evaluation failed: {str(e)}",
                        details={"error": str(e)},
                        execution_time=0.0
                    )
                    results.append(error_result)
        
        return results
```

---

## 📈 5. Métricas Específicas para RAG

### 5.1 Métricas de Retrieval

```python
class RetrievalMetrics:
    """Métricas para componente de retrieval"""
    
    @staticmethod
    def precision_at_k(retrieved_docs: List[str], 
                      relevant_docs: List[str], 
                      k: int) -> float:
        """Precision@K: Proporción de documentos relevantes en top-K"""
        top_k = retrieved_docs[:k]
        relevant_in_top_k = len(set(top_k) & set(relevant_docs))
        return relevant_in_top_k / k if k > 0 else 0.0
    
    @staticmethod
    def recall_at_k(retrieved_docs: List[str],
                   relevant_docs: List[str],
                   k: int) -> float:
        """Recall@K: Proporción del total relevante recuperado en top-K"""
        top_k = retrieved_docs[:k]
        relevant_in_top_k = len(set(top_k) & set(relevant_docs))
        return relevant_in_top_k / len(relevant_docs) if relevant_docs else 0.0
    
    @staticmethod
    def mean_reciprocal_rank(retrieved_docs: List[str],
                           relevant_docs: List[str]) -> float:
        """MRR: Promedio del ranking recíproco del primer documento relevante"""
        for i, doc in enumerate(retrieved_docs, 1):
            if doc in relevant_docs:
                return 1.0 / i
        return 0.0
    
    @staticmethod
    def ndcg_at_k(retrieved_docs: List[str],
                  relevance_scores: Dict[str, float],
                  k: int) -> float:
        """NDCG@K: Normalized Discounted Cumulative Gain"""
        def dcg(scores: List[float]) -> float:
            return sum(score / math.log2(i + 2) for i, score in enumerate(scores))
        
        # DCG real
        actual_scores = [relevance_scores.get(doc, 0.0) for doc in retrieved_docs[:k]]
        actual_dcg = dcg(actual_scores)
        
        # DCG ideal
        ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
        ideal_dcg = dcg(ideal_scores)
        
        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
```

### 5.2 Métricas de Generation

```python
class GenerationMetrics:
    """Métricas para componente de generación"""
    
    @staticmethod
    def bleu_score(reference: str, candidate: str) -> float:
        """BLEU score simplificado (unigram)"""
        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()
        
        if not cand_tokens:
            return 0.0
        
        matches = sum(1 for token in cand_tokens if token in ref_tokens)
        return matches / len(cand_tokens)
    
    @staticmethod
    def rouge_1(reference: str, candidate: str) -> Dict[str, float]:
        """ROUGE-1 (unigram overlap)"""
        ref_tokens = set(reference.lower().split())
        cand_tokens = set(candidate.lower().split())
        
        if not ref_tokens and not cand_tokens:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        
        if not cand_tokens:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        overlap = len(ref_tokens & cand_tokens)
        
        precision = overlap / len(cand_tokens)
        recall = overlap / len(ref_tokens) if ref_tokens else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {"precision": precision, "recall": recall, "f1": f1}
    
    @staticmethod
    def factual_consistency(response: str, sources: List[str]) -> float:
        """Verificar consistencia factual básica"""
        # Implementación simplificada
        response_lower = response.lower()
        
        # Buscar claims específicos en las fuentes
        verified_claims = 0
        total_claims = 1  # Simplificado
        
        for source in sources:
            source_lower = source.lower()
            # Verificar overlap de entidades/conceptos clave
            common_words = set(response_lower.split()) & set(source_lower.split())
            if len(common_words) > 3:  # Threshold arbitrario
                verified_claims += 1
        
        return min(verified_claims / total_claims, 1.0)
```

---

## 🔧 6. Laboratorio Práctico

### 6.1 Ejercicio 1: Quick Evals Básicos ⏳

**Objetivo:** Implementar evaluaciones rápidas para sistemas de IA

**Archivo:** `labs/module-d/quick_evals.py`

```python
# TODO: Implementar QuickEvalSuite que:
# 1. Ejecute múltiples evaluaciones en paralelo
# 2. Soporte diferentes tipos de checks
# 3. Genere reportes detallados
# 4. Se integre con CI/CD
```

### 6.2 Ejercicio 2: Cost & Latency Monitoring ⏳

**Objetivo:** Sistema de monitoreo de coste y performance

**Archivo:** `labs/module-d/cost_monitoring.py`

```python
# TODO: Crear CostMonitor que:
# 1. Rastree costes por modelo y operación
# 2. Mida latencia con percentiles
# 3. Genere alertas por umbrales
# 4. Exporte métricas a dashboards
```

### 6.3 Ejercicio 3: RAG Evaluation Pipeline ⏳

**Objetivo:** Pipeline completo de evaluación para sistemas RAG

**Archivo:** `labs/module-d/rag_evaluation.py`

```python
# TODO: Implementar RAGEvaluator que:
# 1. Evalúe retrieval con Precision@K, NDCG
# 2. Mida calidad de generación
# 3. Verifique citas y consistencia factual
# 4. Genere reportes comparativos
```

---

## 📊 7. Casos de Uso Prácticos

### 7.1 CI/CD Integration

```python
# Ejemplo de integración con CI/CD
def run_pre_deployment_eval():
    """Evaluación antes de deployment"""
    
    # Cargar test cases
    test_cases = load_test_cases("data/eval_cases.json")
    
    # Configurar evaluaciones
    evals = [
        LengthEval(min_length=10, max_length=500),
        ForbiddenContentEval(["violencia", "hate", "spam"]),
        CitationEval()
    ]
    
    # Ejecutar quality gate
    gate = QualityGate("pre_deployment", evals)
    results = gate.run_gate(test_cases)
    
    # Decidir si continuar deployment
    if not results["gate_passed"]:
        print("❌ Quality gate failed - blocking deployment")
        exit(1)
    else:
        print("✅ Quality gate passed - proceeding with deployment")
```

### 7.2 A/B Testing Evaluation

```python
def compare_model_versions():
    """Comparar diferentes versiones de modelos"""
    
    models = ["gpt-4o-mini", "gpt-4o"]
    results = {}
    
    for model in models:
        cost_calc = CostCalculator()
        latency_tracker = LatencyTracker()
        
        # Ejecutar evaluaciones
        model_results = run_evaluation_suite(model)
        
        results[model] = {
            "accuracy": model_results["accuracy"],
            "cost_per_request": cost_calc.calculate_cost(model, 100, 50).total_cost_usd,
            "avg_latency": latency_tracker.get_metrics().p50_latency
        }
    
    # Analizar trade-offs
    print("📊 Model Comparison:")
    for model, metrics in results.items():
        print(f"{model}: Accuracy={metrics['accuracy']:.3f}, "
              f"Cost=${metrics['cost_per_request']:.4f}, "
              f"Latency={metrics['avg_latency']:.3f}s")
```

---

## 🎯 8. Métricas de Negocio

### 8.1 KPIs para Sistemas de IA

| Categoría | Métrica | Descripción | Target |
|-----------|---------|-------------|---------|
| **Calidad** | Accuracy | % respuestas correctas | >85% |
| **Performance** | P95 Latency | Latencia percentil 95 | <2s |
| **Coste** | Cost per Query | Coste promedio por consulta | <$0.02 |
| **Usabilidad** | User Satisfaction | Rating promedio usuarios | >4.2/5 |
| **Confiabilidad** | Uptime | Disponibilidad del servicio | >99.9% |

### 8.2 Alerting & Monitoring

```python
class AlertingSystem:
    """Sistema de alertas para métricas"""
    
    def __init__(self):
        self.thresholds = {
            "accuracy_drop": 0.05,  # 5% drop triggers alert
            "cost_spike": 2.0,      # 2x cost increase
            "latency_spike": 2.0,   # 2x latency increase
            "error_rate": 0.01      # 1% error rate
        }
    
    def check_metrics(self, current_metrics: Dict[str, float], 
                     baseline_metrics: Dict[str, float]):
        """Verificar si métricas requieren alertas"""
        alerts = []
        
        # Check accuracy drop
        if "accuracy" in current_metrics and "accuracy" in baseline_metrics:
            drop = baseline_metrics["accuracy"] - current_metrics["accuracy"]
            if drop > self.thresholds["accuracy_drop"]:
                alerts.append(f"Accuracy dropped by {drop:.3f}")
        
        # Check cost spike
        if "cost" in current_metrics and "cost" in baseline_metrics:
            ratio = current_metrics["cost"] / baseline_metrics["cost"]
            if ratio > self.thresholds["cost_spike"]:
                alerts.append(f"Cost increased by {ratio:.2f}x")
        
        return alerts
```

---

## 📈 9. Dashboard y Reporting

### 9.1 Métricas Dashboard

```python
def generate_dashboard_data():
    """Generar datos para dashboard de métricas"""
    
    return {
        "overview": {
            "total_requests": 10543,
            "success_rate": 0.987,
            "avg_cost_per_request": 0.0023,
            "avg_latency": 1.234
        },
        "quality_metrics": {
            "accuracy": 0.923,
            "citation_rate": 0.891,
            "user_satisfaction": 4.3
        },
        "performance_trends": {
            "latency_7d": [1.2, 1.3, 1.1, 1.4, 1.2, 1.3, 1.2],
            "cost_7d": [0.002, 0.0023, 0.0021, 0.0024, 0.0022, 0.0023, 0.0023]
        },
        "alerts": [
            {"type": "warning", "message": "Latency spike detected", "timestamp": "2025-08-17T10:30:00Z"}
        ]
    }
```

---

## ✅ 10. Resumen del Módulo

Al completar este módulo habrás implementado:

1. **Quick Evals automatizados** para validación continua
2. **Sistema de tracking** de coste y latencia
3. **Quality Gates** para control de deployments
4. **Métricas especializadas** para sistemas RAG
5. **Pipeline de evaluación** end-to-end

**Tiempo estimado:** 2-3 horas  
**Prerrequisitos:** Módulos A, B y C completados  
**Siguiente:** Módulo E - Capstone Project

---

## 🔗 Recursos Adicionales

- [Evaluation in Production](https://eugeneyan.com/writing/evals/)
- [LLM Evaluation Metrics](https://docs.anthropic.com/claude/docs/evaluating-claude)
- [Cost Optimization for LLMs](https://platform.openai.com/docs/guides/production-best-practices)
- [A/B Testing for ML](https://netflixtechblog.com/its-all-a-bout-testing-the-netflix-experimentation-platform-4e1ca458c15)
