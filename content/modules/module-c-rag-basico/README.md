# Módulo C: RAG Básico con Citas

## 🎯 Objetivos de Aprendizaje

Al completar este módulo, serás capaz de:

1. **Implementar RAG básico** con algoritmos BM25 y TF-IDF
2. **Generar citas canónicas** en formato `uri#Lx-Ly`
3. **Aplicar técnicas de re-ranking** usando Maximal Marginal Relevance (MMR)
4. **Integrar RAG con agentes** del módulo anterior
5. **Manejar documentos estructurados** y metadatos

---

## 📚 1. Introducción a RAG

### 1.1 ¿Qué es Retrieval-Augmented Generation?

RAG es un patrón arquitectónico que combina:

- **Retrieval**: Búsqueda en base de conocimiento
- **Augmentation**: Enriquecimiento del contexto  
- **Generation**: Generación de respuestas con LLM

```
Usuario → Consulta → [RETRIEVAL] → Documentos → [AUGMENTATION] → Contexto → [GENERATION] → Respuesta
```

### 1.2 Ventajas del RAG

| Ventaja | Descripción | Ejemplo |
|---------|-------------|---------|
| **Actualización** | Información siempre actualizada | Documentación técnica |
| **Precisión** | Respuestas basadas en fuentes | Citación de documentos |
| **Control** | Fuentes conocidas y verificables | Políticas empresariales |
| **Eficiencia** | Sin re-entrenamiento de modelos | Cambios en productos |

### 1.3 Arquitectura RAG Básica

```python
class BasicRAG:
    def __init__(self):
        self.document_store = DocumentStore()      # Almacén de documentos
        self.retriever = Retriever()              # Motor de búsqueda
        self.ranker = Ranker()                    # Re-ranking por relevancia
        self.generator = Generator()              # LLM para respuestas
    
    def query(self, question: str) -> str:
        # 1. RETRIEVAL: Buscar documentos relevantes
        candidates = self.retriever.search(question)
        
        # 2. RANKING: Ordenar por relevancia
        ranked_docs = self.ranker.rank(question, candidates)
        
        # 3. AUGMENTATION: Preparar contexto
        context = self._build_context(ranked_docs)
        
        # 4. GENERATION: Generar respuesta
        response = self.generator.generate(question, context)
        
        return response
```

---

## 🔍 2. Algoritmos de Búsqueda

### 2.1 TF-IDF (Term Frequency - Inverse Document Frequency)

**Concepto:** Mide la importancia de un término en un documento relativo a una colección.

**Fórmula:**
```
TF-IDF(t,d) = TF(t,d) × IDF(t)
TF(t,d) = (frecuencia de t en d) / (total términos en d)
IDF(t) = log(N / df(t))
```

**Implementación básica:**
```python
import math
from collections import Counter
from typing import List, Dict

class TFIDFRetriever:
    def __init__(self, documents: List[str]):
        self.documents = documents
        self.vocab = self._build_vocabulary()
        self.idf_scores = self._compute_idf()
        self.doc_vectors = self._compute_doc_vectors()
    
    def _build_vocabulary(self) -> set:
        """Construir vocabulario único"""
        vocab = set()
        for doc in self.documents:
            vocab.update(doc.lower().split())
        return vocab
    
    def _compute_idf(self) -> Dict[str, float]:
        """Calcular IDF para cada término"""
        idf_scores = {}
        N = len(self.documents)
        
        for term in self.vocab:
            df = sum(1 for doc in self.documents if term in doc.lower())
            idf_scores[term] = math.log(N / max(df, 1))
        
        return idf_scores
    
    def search(self, query: str, top_k: int = 5) -> List[tuple]:
        """Buscar documentos más relevantes"""
        query_vector = self._compute_query_vector(query)
        scores = []
        
        for i, doc_vector in enumerate(self.doc_vectors):
            score = self._cosine_similarity(query_vector, doc_vector)
            scores.append((i, score))
        
        # Ordenar por score descendente
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
```

### 2.2 BM25 (Best Matching 25)

**Concepto:** Mejora de TF-IDF que considera longitud del documento y saturación de términos.

**Fórmula:**
```
BM25(q,d) = Σ IDF(qi) × (tf(qi,d) × (k1 + 1)) / (tf(qi,d) + k1 × (1 - b + b × |d|/avgdl))
```

**Parámetros:**
- `k1`: Controla saturación de términos (típicamente 1.2-2.0)
- `b`: Controla impacto de longitud de documento (típicamente 0.75)

**Implementación:**
```python
class BM25Retriever:
    def __init__(self, documents: List[str], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.avgdl = sum(len(doc.split()) for doc in documents) / len(documents)
        self.doc_freqs = []
        self.idf = {}
        self._initialize()
    
    def _initialize(self):
        """Inicializar frecuencias e IDF"""
        df = {}  # document frequency
        
        for doc in self.documents:
            tokens = doc.lower().split()
            freq = Counter(tokens)
            self.doc_freqs.append(freq)
            
            for token in freq.keys():
                df[token] = df.get(token, 0) + 1
        
        # Calcular IDF
        N = len(self.documents)
        for token, freq in df.items():
            self.idf[token] = math.log((N - freq + 0.5) / (freq + 0.5))
    
    def search(self, query: str, top_k: int = 5) -> List[tuple]:
        """Buscar con BM25"""
        query_tokens = query.lower().split()
        scores = []
        
        for i, doc_freq in enumerate(self.doc_freqs):
            score = 0
            doc_len = sum(doc_freq.values())
            
            for token in query_tokens:
                if token in doc_freq:
                    tf = doc_freq[token]
                    idf = self.idf.get(token, 0)
                    
                    # BM25 score para este término
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                    score += idf * (numerator / denominator)
            
            scores.append((i, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
```

---

## 📖 3. Sistema de Citas Canónicas

### 3.1 Formato de Citas

**Estructura:** `uri#Lx-Ly`
- `uri`: Identificador único del documento
- `Lx`: Línea de inicio
- `Ly`: Línea de fin

**Ejemplos:**
```
doc://manual-python.md#L45-L52
https://docs.python.org/tutorial#L120-L125
file://policies/security.pdf#L8-L15
```

### 3.2 Generación de Citas

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Citation:
    """Cita canónica con metadatos"""
    uri: str
    start_line: int
    end_line: int
    content: str
    relevance_score: float
    metadata: Dict[str, Any]
    
    def to_canonical(self) -> str:
        """Generar cita en formato canónico"""
        return f"{self.uri}#L{self.start_line}-L{self.end_line}"
    
    def __str__(self) -> str:
        return f"[{self.to_canonical()}] {self.content[:100]}..."

class CitationGenerator:
    """Generador de citas canónicas"""
    
    def __init__(self, line_context: int = 2):
        self.line_context = line_context
    
    def generate_citation(self, 
                         document_uri: str,
                         content: str,
                         match_start: int,
                         match_end: int,
                         score: float,
                         metadata: Dict = None) -> Citation:
        """
        Generar cita con contexto de líneas
        
        Args:
            document_uri: URI del documento
            content: Contenido completo del documento
            match_start: Posición inicial del match
            match_end: Posición final del match  
            score: Score de relevancia
            metadata: Metadatos adicionales
        """
        lines = content.split('\n')
        
        # Encontrar líneas del match
        current_pos = 0
        start_line = 1
        end_line = 1
        
        for i, line in enumerate(lines):
            line_end = current_pos + len(line) + 1  # +1 por \n
            
            if current_pos <= match_start < line_end:
                start_line = i + 1
            
            if current_pos <= match_end <= line_end:
                end_line = i + 1
                break
                
            current_pos = line_end
        
        # Agregar contexto
        context_start = max(1, start_line - self.line_context)
        context_end = min(len(lines), end_line + self.line_context)
        
        # Extraer contenido citado
        cited_content = '\n'.join(lines[context_start-1:context_end])
        
        return Citation(
            uri=document_uri,
            start_line=context_start,
            end_line=context_end,
            content=cited_content,
            relevance_score=score,
            metadata=metadata or {}
        )
```

### 3.3 Validación de Citas

```python
import re

class CitationValidator:
    """Validador de citas canónicas"""
    
    CITATION_PATTERN = re.compile(r'^([^#]+)#L(\d+)-L(\d+)$')
    
    @classmethod
    def validate_format(cls, citation: str) -> tuple[bool, str]:
        """Validar formato de cita canónica"""
        match = cls.CITATION_PATTERN.match(citation)
        
        if not match:
            return False, "Formato inválido. Debe ser: uri#Lx-Ly"
        
        uri, start_line, end_line = match.groups()
        start_line, end_line = int(start_line), int(end_line)
        
        if start_line > end_line:
            return False, "Línea inicial no puede ser mayor que línea final"
        
        if start_line < 1:
            return False, "Números de línea deben ser >= 1"
        
        return True, "Formato válido"
    
    @classmethod
    def parse_citation(cls, citation: str) -> Optional[tuple]:
        """Parsear cita canónica"""
        match = cls.CITATION_PATTERN.match(citation)
        if match:
            uri, start_line, end_line = match.groups()
            return uri, int(start_line), int(end_line)
        return None
```

---

## 🔄 4. Re-ranking con MMR

### 4.1 Maximal Marginal Relevance

**Objetivo:** Balancear relevancia con diversidad para evitar documentos redundantes.

**Fórmula:**
```
MMR = λ × Sim(q, d) - (1-λ) × max Sim(d, d')
```
- `λ`: Parámetro de balance (0-1)
- `Sim(q, d)`: Similaridad consulta-documento
- `Sim(d, d')`: Similaridad entre documentos ya seleccionados

### 4.2 Implementación MMR

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MMRRanker:
    """Re-ranker usando Maximal Marginal Relevance"""
    
    def __init__(self, lambda_param: float = 0.7):
        self.lambda_param = lambda_param
        self.vectorizer = TfidfVectorizer(stop_words='english')
    
    def rank(self, 
             query: str, 
             documents: List[str], 
             top_k: int = 5) -> List[int]:
        """
        Re-rankear documentos usando MMR
        
        Args:
            query: Consulta del usuario
            documents: Lista de documentos candidatos
            top_k: Número de documentos a retornar
            
        Returns:
            Lista de índices de documentos rankeados
        """
        if not documents:
            return []
        
        # Vectorizar consulta y documentos
        all_texts = [query] + documents
        vectors = self.vectorizer.fit_transform(all_texts)
        
        query_vector = vectors[0]
        doc_vectors = vectors[1:]
        
        # Calcular similaridades consulta-documento
        query_similarities = cosine_similarity(query_vector, doc_vectors)[0]
        
        # MMR iterativo
        selected_indices = []
        remaining_indices = list(range(len(documents)))
        
        for _ in range(min(top_k, len(documents))):
            best_score = -float('inf')
            best_idx = None
            
            for idx in remaining_indices:
                # Relevancia con la consulta
                relevance = query_similarities[idx]
                
                # Máxima similaridad con documentos ya seleccionados
                if selected_indices:
                    selected_vectors = doc_vectors[selected_indices]
                    current_vector = doc_vectors[idx:idx+1]
                    max_similarity = cosine_similarity(current_vector, selected_vectors).max()
                else:
                    max_similarity = 0
                
                # Score MMR
                mmr_score = (self.lambda_param * relevance - 
                           (1 - self.lambda_param) * max_similarity)
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
        
        return selected_indices
```

---

## 🗃️ 5. Almacén de Documentos

### 5.1 Estructura de Documentos

```python
@dataclass
class Document:
    """Documento con metadatos"""
    id: str
    uri: str
    title: str
    content: str
    author: Optional[str] = None
    created_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_lines(self) -> List[str]:
        """Obtener líneas del documento"""
        return self.content.split('\n')
    
    def extract_snippet(self, start_line: int, end_line: int) -> str:
        """Extraer snippet por líneas"""
        lines = self.get_lines()
        snippet_lines = lines[start_line-1:end_line]
        return '\n'.join(snippet_lines)

class DocumentStore:
    """Almacén de documentos con índices"""
    
    def __init__(self):
        self.documents: Dict[str, Document] = {}
        self.uri_index: Dict[str, str] = {}  # uri -> doc_id
        self.tag_index: Dict[str, List[str]] = {}  # tag -> [doc_ids]
    
    def add_document(self, document: Document) -> None:
        """Agregar documento al almacén"""
        self.documents[document.id] = document
        self.uri_index[document.uri] = document.id
        
        # Indexar por tags
        for tag in document.tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = []
            self.tag_index[tag].append(document.id)
    
    def get_by_id(self, doc_id: str) -> Optional[Document]:
        """Obtener documento por ID"""
        return self.documents.get(doc_id)
    
    def get_by_uri(self, uri: str) -> Optional[Document]:
        """Obtener documento por URI"""
        doc_id = self.uri_index.get(uri)
        return self.documents.get(doc_id) if doc_id else None
    
    def search_by_tags(self, tags: List[str]) -> List[Document]:
        """Buscar documentos por tags"""
        doc_ids = set()
        for tag in tags:
            if tag in self.tag_index:
                doc_ids.update(self.tag_index[tag])
        
        return [self.documents[doc_id] for doc_id in doc_ids]
    
    def get_all_documents(self) -> List[Document]:
        """Obtener todos los documentos"""
        return list(self.documents.values())
```

---

## 🔧 6. Laboratorio Práctico

### 6.1 Ejercicio 1: Implementar Retriever Básico ✅

**Objetivo:** Crear retriever con TF-IDF y BM25

**Archivo:** `labs/module-c/basic_retriever.py`

**Estado**: ✅ **COMPLETADO** - TFIDFRetriever y BM25Retriever funcionales con preprocessor avanzado

```python
# ✅ IMPLEMENTADO: BasicRetriever que:
# 1. ✅ Soporta TF-IDF y BM25 con parámetros configurables
# 2. ✅ Maneja documentos estructurados con metadatos
# 3. ✅ Genera scores de relevancia precisos
# 4. ✅ Permite configuración de parámetros (k1, b para BM25)
```

### 6.2 Ejercicio 2: Sistema de Citas ✅

**Objetivo:** Implementar generación y validación de citas canónicas

**Archivo:** `labs/module-c/citation_system.py`

**Estado**: ✅ **COMPLETADO** - Sistema completo de citas con validación estricta y generación automática

```python
# ✅ IMPLEMENTADO: CitationSystem que:
# 1. ✅ Genera citas en formato uri#Lx-Ly con contexto
# 2. ✅ Valida formato con regex patterns estrictos
# 3. ✅ Maneja contexto de líneas automáticamente
# 4. ✅ Extrae snippets citados con validación de consistencia
```

### 6.3 Ejercicio 3: RAG Completo ✅

**Objetivo:** Integrar retrieval, ranking y generación

**Archivo:** `labs/module-c/complete_rag.py`

**Estado**: ✅ **COMPLETADO** - Pipeline RAG completo con MMR, generación inteligente e integración con agente PEC

```python
# ✅ IMPLEMENTADO: CompleteRAG que:
# 1. ✅ Combina retriever + MMR ranker + response generator
# 2. ✅ Usa MMR para re-ranking y diversidad de resultados
# 3. ✅ Genera respuestas con citas canónicas automáticas
# 4. ✅ Integra con agente PEC del módulo B como herramienta
```

---

## 📊 7. Casos de Uso Reales

### 7.1 Documentación Técnica

```python
# Ejemplo: RAG para documentación de API
docs = [
    "La función authenticate() valida credenciales de usuario...",
    "El endpoint /api/users retorna lista de usuarios activos...",
    "Para autorización usa headers Authorization: Bearer <token>..."
]

retriever = BM25Retriever(docs)
results = retriever.search("cómo autenticar usuario", top_k=2)

# Resultado: [(0, 0.85), (2, 0.72)]
# Con citas: doc://api-docs.md#L15-L18, doc://api-docs.md#L45-L47
```

### 7.2 Base de Conocimiento Empresarial

```python
# Ejemplo: Políticas de empresa
policy_docs = [
    "Las vacaciones deben solicitarse con 15 días de anticipación...",
    "El trabajo remoto requiere aprobación del supervisor directo...",
    "Los gastos de viaje deben incluir recibos originales..."
]

# Con MMR para diversidad de respuestas
ranker = MMRRanker(lambda_param=0.8)
ranked_results = ranker.rank("política de trabajo remoto", policy_docs)
```

### 7.3 Soporte Técnico

```python
# Ejemplo: Troubleshooting automático
trouble_docs = [
    "Error 404: Verificar URL y permisos de acceso...",
    "Conexión lenta: Revisar configuración de red...", 
    "Login fallido: Confirmar credenciales y estado de cuenta..."
]

# Pipeline completo RAG
rag_system = CompleteRAG()
response = rag_system.query("no puedo hacer login")
# Respuesta incluye citas a documentos relevantes
```

---

## 🎯 8. Métricas de Evaluación

### 8.1 Métricas de Retrieval

| Métrica | Descripción | Fórmula |
|---------|-------------|---------|
| **Precision@k** | % relevantes en top-k | relevantes_recuperados / k |
| **Recall@k** | % del total relevante recuperado | relevantes_recuperados / total_relevantes |
| **MRR** | Mean Reciprocal Rank | 1/N × Σ(1/rank_primer_relevante) |
| **NDCG@k** | Normalized Discounted Cumulative Gain | DCG@k / IDCG@k |

### 8.2 Métricas de Generación

| Métrica | Descripción | Uso |
|---------|-------------|-----|
| **BLEU** | Overlap de n-gramas con referencia | Traducción automática |
| **ROUGE** | Overlap de tokens con referencia | Resumen automático |
| **BERTScore** | Similaridad semántica con embeddings | Evaluación general |
| **Factualidad** | % claims verificables en fuentes | RAG específico |

### 8.3 Implementación de Evaluación

```python
class RAGEvaluator:
    """Evaluador específico para sistemas RAG"""
    
    def evaluate_retrieval(self, 
                          queries: List[str],
                          true_relevant: List[List[str]],
                          retrieved: List[List[str]]) -> Dict[str, float]:
        """Evaluar calidad del retrieval"""
        metrics = {}
        
        # Precision@k promedio
        precisions = []
        for relevant, retr in zip(true_relevant, retrieved):
            relevant_set = set(relevant)
            retrieved_set = set(retr)
            precision = len(relevant_set & retrieved_set) / len(retrieved_set)
            precisions.append(precision)
        
        metrics['precision_at_k'] = sum(precisions) / len(precisions)
        
        return metrics
    
    def evaluate_citations(self, 
                          generated_citations: List[str],
                          source_documents: List[str]) -> Dict[str, float]:
        """Evaluar calidad de las citas"""
        valid_citations = 0
        total_citations = len(generated_citations)
        
        for citation in generated_citations:
            if self._is_citation_valid(citation, source_documents):
                valid_citations += 1
        
        return {
            'citation_accuracy': valid_citations / total_citations,
            'total_citations': total_citations
        }
```

---

## 🚀 9. Optimizaciones Avanzadas

### 9.1 Cache de Embeddings

```python
import pickle
from functools import lru_cache

class CachedRetriever:
    """Retriever con cache de embeddings"""
    
    def __init__(self, cache_file: str = "embeddings.pkl"):
        self.cache_file = cache_file
        self.embedding_cache = self._load_cache()
    
    @lru_cache(maxsize=1000)
    def get_embedding(self, text: str) -> np.ndarray:
        """Obtener embedding con cache"""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        # Generar embedding (simulado)
        embedding = np.random.random(384)  # Simular embedding
        self.embedding_cache[text] = embedding
        return embedding
    
    def save_cache(self):
        """Guardar cache en disco"""
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.embedding_cache, f)
```

### 9.2 Índices Aproximados

```python
import faiss
import numpy as np

class FaissRetriever:
    """Retriever usando FAISS para búsqueda aproximada"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product
        self.doc_ids = []
    
    def add_documents(self, embeddings: np.ndarray, doc_ids: List[str]):
        """Agregar documentos al índice"""
        self.index.add(embeddings.astype('float32'))
        self.doc_ids.extend(doc_ids)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[tuple]:
        """Búsqueda aproximada con FAISS"""
        scores, indices = self.index.search(
            query_embedding.astype('float32').reshape(1, -1), 
            top_k
        )
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.doc_ids):
                results.append((self.doc_ids[idx], float(score)))
        
        return results
```

---

## 📈 10. Próximos Pasos

### 10.1 Conexión con Módulo D

El Módulo C establece la base para evaluación en el Módulo D:
- Métricas de retrieval (Precision@k, Recall@k)
- Evaluación de calidad de citas
- Benchmarks de latencia y throughput

### 10.2 Preparación para Capstone

Los componentes RAG serán integrados en el proyecto final:
- Agente PEC (Módulo B) + RAG (Módulo C)
- Sistema completo de Q&A con citas
- Dashboard de evaluación (Módulo D)

---

## ✅ Resumen del Módulo

Al completar este módulo habrás implementado:

1. **Retriever básico** con TF-IDF y BM25
2. **Sistema de citas canónicas** con formato `uri#Lx-Ly`
3. **Re-ranking con MMR** para diversidad de resultados
4. **Almacén de documentos** con índices y metadatos
5. **Pipeline RAG completo** integrado con agentes

**Tiempo estimado:** 3-4 horas  
**Prerrequisitos:** Módulos A y B completados  
**Siguiente:** Módulo D - Métricas y Evaluación
