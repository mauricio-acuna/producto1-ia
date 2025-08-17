"""
Laboratorio Módulo C: Retriever Básico
Implementación de algoritmos TF-IDF y BM25 para búsqueda en documentos
"""

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import json


@dataclass
class Document:
    """Documento con metadatos para el sistema RAG"""
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
        """Extraer snippet por líneas (1-indexed)"""
        lines = self.get_lines()
        snippet_lines = lines[start_line-1:end_line]
        return '\n'.join(snippet_lines)
    
    def word_count(self) -> int:
        """Contar palabras en el documento"""
        return len(self.content.split())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para serialización"""
        return {
            'id': self.id,
            'uri': self.uri,
            'title': self.title,
            'content': self.content,
            'author': self.author,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'tags': self.tags,
            'metadata': self.metadata
        }


class TextPreprocessor:
    """
    Preprocesador de texto para normalización y tokenización
    
    TODO: Implementar preprocesamiento robusto
    """
    
    def __init__(self, 
                 lowercase: bool = True,
                 remove_punctuation: bool = True,
                 min_word_length: int = 2,
                 stop_words: Optional[List[str]] = None):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.min_word_length = min_word_length
        self.stop_words = set(stop_words or self._default_stop_words())
    
    def _default_stop_words(self) -> List[str]:
        """Stop words básicas en español e inglés"""
        return [
            # Español
            'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 
            'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para', 'al',
            'del', 'los', 'las', 'una', 'pero', 'sus', 'le', 'ha', 'me', 'si',
            'sin', 'sobre', 'este', 'ya', 'entre', 'cuando', 'todo', 'esta',
            'ser', 'son', 'dos', 'también', 'fue', 'había', 'era', 'muy',
            # Inglés  
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her',
            'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there',
            'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get',
            'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no',
            'just', 'him', 'know', 'take', 'people', 'into', 'year', 'your',
            'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then',
            'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
            'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first',
            'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these',
            'give', 'day', 'most', 'us'
        ]
    
    def tokenize(self, text: str) -> List[str]:
        """
        TODO: Tokenizar texto con preprocesamiento
        
        Pasos:
        1. Convertir a minúsculas si está habilitado
        2. Remover puntuación si está habilitado
        3. Dividir en tokens
        4. Filtrar por longitud mínima
        5. Remover stop words
        """
        if not text:
            return []
        
        # Paso 1: Convertir a minúsculas
        if self.lowercase:
            text = text.lower()
        
        # Paso 2: Remover puntuación y caracteres especiales
        if self.remove_punctuation:
            # Mantener solo letras, números y espacios
            text = re.sub(r'[^\w\s]', ' ', text)
        
        # Paso 3: Dividir en tokens
        tokens = text.split()
        
        # Paso 4: Filtrar por longitud mínima
        tokens = [token for token in tokens if len(token) >= self.min_word_length]
        
        # Paso 5: Remover stop words
        tokens = [token for token in tokens if token not in self.stop_words]
        
        return tokens
    
    def preprocess_documents(self, documents: List[Document]) -> List[List[str]]:
        """Preprocesar lista de documentos"""
        return [self.tokenize(doc.content) for doc in documents]


class TFIDFRetriever:
    """
    Retriever usando algoritmo TF-IDF (Term Frequency - Inverse Document Frequency)
    
    TODO: Implementar TF-IDF completo
    """
    
    def __init__(self, documents: List[Document], preprocessor: Optional[TextPreprocessor] = None):
        self.documents = documents
        self.preprocessor = preprocessor or TextPreprocessor()
        
        # Índices
        self.vocabulary: set = set()
        self.term_frequencies: List[Dict[str, int]] = []
        self.document_frequencies: Dict[str, int] = {}
        self.idf_scores: Dict[str, float] = {}
        self.doc_vectors: List[Dict[str, float]] = []
        
        # Estadísticas
        self.total_documents = len(documents)
        self.avg_doc_length = 0
        
        # Inicializar
        self._build_index()
    
    def _build_index(self):
        """
        TODO: Construir índice TF-IDF
        
        Pasos:
        1. Tokenizar todos los documentos
        2. Construir vocabulario
        3. Calcular frecuencias de términos (TF)
        4. Calcular frecuencias de documento (DF)
        5. Calcular puntuaciones IDF
        6. Generar vectores de documentos
        """
        print("🔨 Construyendo índice TF-IDF...")
        
        # Paso 1: Tokenizar documentos
        tokenized_docs = []
        total_words = 0
        
        for doc in self.documents:
            tokens = self.preprocessor.tokenize(doc.content)
            tokenized_docs.append(tokens)
            total_words += len(tokens)
        
        # Calcular longitud promedio de documento
        self.avg_doc_length = total_words / len(self.documents) if self.documents else 0
        
        # Paso 2: Construir vocabulario
        for tokens in tokenized_docs:
            self.vocabulary.update(tokens)
        
        print(f"📚 Vocabulario construido: {len(self.vocabulary)} términos únicos")
        
        # Paso 3: Calcular frecuencias de términos (TF)
        for tokens in tokenized_docs:
            tf_dict = Counter(tokens)
            self.term_frequencies.append(tf_dict)
        
        # Paso 4: Calcular frecuencias de documento (DF)
        for tokens in tokenized_docs:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.document_frequencies[token] = self.document_frequencies.get(token, 0) + 1
        
        # Paso 5: Calcular puntuaciones IDF
        for term in self.vocabulary:
            df = self.document_frequencies[term]
            # IDF = log(N / df) donde N = total documentos
            self.idf_scores[term] = math.log(self.total_documents / df)
        
        # Paso 6: Generar vectores de documentos TF-IDF
        for tf_dict in self.term_frequencies:
            doc_vector = {}
            # Normalización por longitud del documento
            doc_length = sum(tf_dict.values())
            
            for term, tf in tf_dict.items():
                # TF normalizado
                normalized_tf = tf / doc_length if doc_length > 0 else 0
                # TF-IDF = TF * IDF
                tfidf_score = normalized_tf * self.idf_scores[term]
                doc_vector[term] = tfidf_score
            
            self.doc_vectors.append(doc_vector)
        
        print(f"✅ Índice TF-IDF completado: {len(self.doc_vectors)} vectores de documentos")
    
    def _compute_query_vector(self, query: str) -> Dict[str, float]:
        """Calcular vector TF-IDF para una consulta"""
        query_tokens = self.preprocessor.tokenize(query)
        query_tf = Counter(query_tokens)
        query_length = len(query_tokens)
        
        query_vector = {}
        for term, tf in query_tf.items():
            if term in self.vocabulary:  # Solo términos conocidos
                normalized_tf = tf / query_length if query_length > 0 else 0
                tfidf_score = normalized_tf * self.idf_scores[term]
                query_vector[term] = tfidf_score
        
        return query_vector
    
    def _cosine_similarity(self, vector1: Dict[str, float], vector2: Dict[str, float]) -> float:
        """Calcular similaridad coseno entre dos vectores"""
        # Términos en común
        common_terms = set(vector1.keys()) & set(vector2.keys())
        
        if not common_terms:
            return 0.0
        
        # Producto punto
        dot_product = sum(vector1[term] * vector2[term] for term in common_terms)
        
        # Magnitudes
        magnitude1 = math.sqrt(sum(score ** 2 for score in vector1.values()))
        magnitude2 = math.sqrt(sum(score ** 2 for score in vector2.values()))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        TODO: Buscar documentos más relevantes usando TF-IDF
        
        Proceso:
        1. Calcular vector de consulta
        2. Calcular similaridad con cada documento
        3. Ordenar por relevancia
        4. Retornar top-k resultados
        """
        if not query.strip():
            return []
        
        print(f"🔍 Buscando: '{query}'")
        
        # Paso 1: Calcular vector de consulta
        query_vector = self._compute_query_vector(query)
        
        if not query_vector:
            print("⚠️ No se encontraron términos de consulta en el vocabulario")
            return []
        
        # Paso 2: Calcular similaridad con cada documento
        scores = []
        for i, doc_vector in enumerate(self.doc_vectors):
            similarity = self._cosine_similarity(query_vector, doc_vector)
            scores.append((i, similarity))
        
        # Paso 3: Ordenar por relevancia (descendente)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Paso 4: Retornar top-k resultados
        results = []
        for i, (doc_idx, score) in enumerate(scores[:top_k]):
            if score > 0:  # Solo incluir documentos con score positivo
                results.append((self.documents[doc_idx], score))
                print(f"  {i+1}. {self.documents[doc_idx].title[:50]}... (score: {score:.4f})")
        
        return results
    
    def get_term_statistics(self, term: str) -> Dict[str, Any]:
        """Obtener estadísticas de un término"""
        if term not in self.vocabulary:
            return {"error": "Término no encontrado en vocabulario"}
        
        df = self.document_frequencies.get(term, 0)
        idf = self.idf_scores.get(term, 0)
        
        return {
            "term": term,
            "document_frequency": df,
            "idf_score": idf,
            "appears_in_percent": (df / self.total_documents) * 100
        }


class BM25Retriever:
    """
    Retriever usando algoritmo BM25 (Best Matching 25)
    Mejora de TF-IDF que considera longitud de documento y saturación de términos
    
    TODO: Implementar BM25 completo
    """
    
    def __init__(self, 
                 documents: List[Document], 
                 k1: float = 1.5, 
                 b: float = 0.75,
                 preprocessor: Optional[TextPreprocessor] = None):
        self.documents = documents
        self.k1 = k1  # Controla saturación de términos (típicamente 1.2-2.0)
        self.b = b    # Controla impacto de longitud de documento (típicamente 0.75)
        self.preprocessor = preprocessor or TextPreprocessor()
        
        # Índices
        self.vocabulary: set = set()
        self.doc_frequencies: List[Counter] = []
        self.document_frequencies: Dict[str, int] = {}
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0
        self.idf_scores: Dict[str, float] = {}
        
        # Estadísticas
        self.total_documents = len(documents)
        
        # Inicializar
        self._build_index()
    
    def _build_index(self):
        """
        TODO: Construir índice BM25
        
        Pasos:
        1. Tokenizar documentos y calcular longitudes
        2. Construir vocabulario y frecuencias
        3. Calcular longitud promedio de documento
        4. Calcular IDF modificado para BM25
        """
        print("🔨 Construyendo índice BM25...")
        
        # Paso 1: Tokenizar documentos
        tokenized_docs = []
        for doc in self.documents:
            tokens = self.preprocessor.tokenize(doc.content)
            tokenized_docs.append(tokens)
            
            # Frecuencias de términos en este documento
            freq_counter = Counter(tokens)
            self.doc_frequencies.append(freq_counter)
            
            # Longitud del documento
            self.doc_lengths.append(len(tokens))
        
        # Paso 2: Construir vocabulario
        for tokens in tokenized_docs:
            self.vocabulary.update(tokens)
        
        # Paso 3: Calcular longitud promedio
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        
        # Paso 4: Calcular document frequencies
        for tokens in tokenized_docs:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.document_frequencies[token] = self.document_frequencies.get(token, 0) + 1
        
        # Paso 5: Calcular IDF para BM25
        for term in self.vocabulary:
            df = self.document_frequencies[term]
            # IDF para BM25: log((N - df + 0.5) / (df + 0.5))
            idf = math.log((self.total_documents - df + 0.5) / (df + 0.5))
            self.idf_scores[term] = max(0, idf)  # Evitar IDF negativos
        
        print(f"📚 Vocabulario BM25: {len(self.vocabulary)} términos únicos")
        print(f"📏 Longitud promedio de documento: {self.avg_doc_length:.1f} tokens")
        print(f"✅ Índice BM25 completado")
    
    def _compute_bm25_score(self, query_tokens: List[str], doc_idx: int) -> float:
        """Calcular score BM25 para un documento específico"""
        score = 0.0
        doc_freq = self.doc_frequencies[doc_idx]
        doc_length = self.doc_lengths[doc_idx]
        
        for token in query_tokens:
            if token not in self.vocabulary:
                continue
            
            # Frecuencia del término en el documento
            tf = doc_freq.get(token, 0)
            if tf == 0:
                continue
            
            # IDF del término
            idf = self.idf_scores[token]
            
            # Componente de normalización por longitud
            normalization = 1 - self.b + self.b * (doc_length / self.avg_doc_length)
            
            # Score BM25 para este término
            # BM25 = IDF * (tf * (k1 + 1)) / (tf + k1 * normalization)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * normalization
            
            term_score = idf * (numerator / denominator)
            score += term_score
        
        return score
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        TODO: Buscar documentos usando BM25
        
        Proceso:
        1. Tokenizar consulta
        2. Calcular score BM25 para cada documento
        3. Ordenar por relevancia
        4. Retornar top-k resultados
        """
        if not query.strip():
            return []
        
        print(f"🔍 BM25 Búsqueda: '{query}'")
        
        # Paso 1: Tokenizar consulta
        query_tokens = self.preprocessor.tokenize(query)
        
        if not query_tokens:
            print("⚠️ No se encontraron términos válidos en la consulta")
            return []
        
        print(f"🔤 Términos de consulta: {query_tokens}")
        
        # Paso 2: Calcular scores BM25
        scores = []
        for i in range(len(self.documents)):
            bm25_score = self._compute_bm25_score(query_tokens, i)
            scores.append((i, bm25_score))
        
        # Paso 3: Ordenar por relevancia
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Paso 4: Retornar top-k resultados
        results = []
        for i, (doc_idx, score) in enumerate(scores[:top_k]):
            if score > 0:
                results.append((self.documents[doc_idx], score))
                print(f"  {i+1}. {self.documents[doc_idx].title[:50]}... (BM25: {score:.4f})")
        
        return results
    
    def explain_score(self, query: str, doc_idx: int) -> Dict[str, Any]:
        """Explicar cómo se calculó el score BM25 para un documento"""
        query_tokens = self.preprocessor.tokenize(query)
        doc_freq = self.doc_frequencies[doc_idx]
        doc_length = self.doc_lengths[doc_idx]
        
        explanation = {
            "document_id": self.documents[doc_idx].id,
            "document_title": self.documents[doc_idx].title,
            "query_tokens": query_tokens,
            "document_length": doc_length,
            "avg_document_length": self.avg_doc_length,
            "k1": self.k1,
            "b": self.b,
            "term_scores": [],
            "total_score": 0
        }
        
        total_score = 0
        for token in query_tokens:
            if token in self.vocabulary:
                tf = doc_freq.get(token, 0)
                idf = self.idf_scores[token]
                df = self.document_frequencies[token]
                
                normalization = 1 - self.b + self.b * (doc_length / self.avg_doc_length)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * normalization
                term_score = idf * (numerator / denominator)
                
                total_score += term_score
                
                explanation["term_scores"].append({
                    "term": token,
                    "tf": tf,
                    "df": df,
                    "idf": idf,
                    "normalization": normalization,
                    "term_score": term_score
                })
        
        explanation["total_score"] = total_score
        return explanation


def create_sample_documents() -> List[Document]:
    """Crear documentos de muestra para pruebas"""
    sample_docs = [
        {
            "id": "doc_001",
            "uri": "docs://python-tutorial.md",
            "title": "Tutorial de Python - Introducción",
            "content": """
Python es un lenguaje de programación interpretado de alto nivel. 
Es conocido por su sintaxis clara y legible, lo que lo hace ideal para principiantes.
Python soporta múltiples paradigmas de programación incluyendo programación orientada a objetos,
programación funcional y programación procedural.

Las características principales de Python incluyen:
- Sintaxis simple y clara
- Tipado dinámico
- Gestión automática de memoria
- Amplia biblioteca estándar
- Gran ecosistema de paquetes externos
            """,
            "author": "Tutorial Team",
            "tags": ["python", "programming", "tutorial", "beginner"]
        },
        {
            "id": "doc_002", 
            "uri": "docs://machine-learning-intro.md",
            "title": "Introducción al Machine Learning",
            "content": """
Machine Learning (ML) es una rama de la inteligencia artificial que permite a los sistemas
aprender y mejorar automáticamente a partir de la experiencia sin ser programados explícitamente.

Los principales tipos de machine learning son:
1. Aprendizaje supervisado: usa datos etiquetados para entrenar modelos
2. Aprendizaje no supervisado: encuentra patrones en datos sin etiquetas  
3. Aprendizaje por refuerzo: aprende mediante interacción con el entorno

Python es uno de los lenguajes más populares para machine learning debido a
bibliotecas como scikit-learn, TensorFlow y PyTorch.
            """,
            "author": "AI Research Team",
            "tags": ["machine-learning", "ai", "python", "data-science"]
        },
        {
            "id": "doc_003",
            "uri": "docs://web-development.md", 
            "title": "Desarrollo Web con Python",
            "content": """
Python ofrece varios frameworks para desarrollo web, siendo Django y Flask los más populares.

Django es un framework full-stack que incluye:
- ORM (Object-Relational Mapping)
- Sistema de autenticación
- Panel de administración
- Sistema de templates

Flask es más minimalista y flexible:
- Microframework ligero
- Mayor control sobre componentes
- Ideal para APIs REST
- Fácil de aprender para principiantes

Ambos frameworks permiten crear aplicaciones web robustas y escalables.
            """,
            "author": "Web Dev Team",
            "tags": ["python", "web-development", "django", "flask", "api"]
        },
        {
            "id": "doc_004",
            "uri": "docs://data-analysis.md",
            "title": "Análisis de Datos con Python", 
            "content": """
Python es excelente para análisis de datos gracias a bibliotecas especializadas.

Las principales bibliotecas para análisis de datos son:
- Pandas: manipulación y análisis de datos estructurados
- NumPy: computación numérica y arrays multidimensionales
- Matplotlib: visualización de datos y gráficos
- Seaborn: visualizaciones estadísticas avanzadas
- SciPy: algoritmos científicos y estadísticos

El flujo típico de análisis incluye:
1. Carga y limpieza de datos
2. Exploración y visualización  
3. Análisis estadístico
4. Modelado predictivo
5. Comunicación de resultados
            """,
            "author": "Data Science Team",
            "tags": ["python", "data-analysis", "pandas", "numpy", "visualization"]
        },
        {
            "id": "doc_005",
            "uri": "docs://api-design.md",
            "title": "Diseño de APIs REST",
            "content": """
Una API REST (Representational State Transfer) es un estilo arquitectónico para servicios web.

Los principios REST incluyen:
- Arquitectura cliente-servidor
- Stateless (sin estado)
- Cacheable
- Interface uniforme
- Sistema de capas

Métodos HTTP principales:
- GET: obtener recursos
- POST: crear nuevos recursos  
- PUT: actualizar recursos existentes
- DELETE: eliminar recursos
- PATCH: actualización parcial

El diseño de URLs debe ser intuitivo y seguir convenciones REST.
Python con Flask o FastAPI facilita la creación de APIs REST.
            """,
            "author": "API Design Team", 
            "tags": ["api", "rest", "web-services", "http", "python"]
        }
    ]
    
    documents = []
    for doc_data in sample_docs:
        doc = Document(
            id=doc_data["id"],
            uri=doc_data["uri"],
            title=doc_data["title"], 
            content=doc_data["content"].strip(),
            author=doc_data["author"],
            created_at=datetime.now(),
            tags=doc_data["tags"]
        )
        documents.append(doc)
    
    return documents


def test_retrievers():
    """Función de prueba para comparar TF-IDF y BM25"""
    print("=== PRUEBAS DE RETRIEVERS ===\n")
    
    # Crear documentos de muestra
    documents = create_sample_documents()
    print(f"📚 Documentos cargados: {len(documents)}")
    for doc in documents:
        print(f"  - {doc.title} ({doc.word_count()} palabras)")
    
    # Crear retrievers
    print(f"\n🔧 Inicializando retrievers...")
    tfidf_retriever = TFIDFRetriever(documents)
    bm25_retriever = BM25Retriever(documents)
    
    # Casos de prueba
    test_queries = [
        "tutorial python principiantes",
        "machine learning algoritmos",
        "desarrollo web django flask",
        "análisis datos pandas numpy", 
        "API REST diseño"
    ]
    
    print(f"\n🧪 Ejecutando pruebas comparativas...\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"{'='*60}")
        print(f"CONSULTA {i}: {query}")
        print(f"{'='*60}")
        
        # TF-IDF
        print(f"\n📊 RESULTADOS TF-IDF:")
        tfidf_results = tfidf_retriever.search(query, top_k=3)
        
        # BM25
        print(f"\n📊 RESULTADOS BM25:")
        bm25_results = bm25_retriever.search(query, top_k=3)
        
        # Comparar resultados
        print(f"\n🔍 COMPARACIÓN:")
        print(f"TF-IDF encontró {len(tfidf_results)} documentos relevantes")
        print(f"BM25 encontró {len(bm25_results)} documentos relevantes")
        
        if tfidf_results and bm25_results:
            tfidf_top = tfidf_results[0][0].id if tfidf_results else None
            bm25_top = bm25_results[0][0].id if bm25_results else None
            
            if tfidf_top == bm25_top:
                print(f"✅ Ambos algoritmos coinciden en el documento más relevante: {tfidf_top}")
            else:
                print(f"🔄 Diferentes documentos más relevantes:")
                print(f"  TF-IDF: {tfidf_top}")
                print(f"  BM25: {bm25_top}")
        
        print(f"\n")
    
    # Estadísticas de términos
    print(f"{'='*60}")
    print(f"ESTADÍSTICAS DE TÉRMINOS")
    print(f"{'='*60}")
    
    interesting_terms = ["python", "machine", "learning", "api", "datos"]
    
    for term in interesting_terms:
        tfidf_stats = tfidf_retriever.get_term_statistics(term)
        print(f"\n🔤 Término: '{term}'")
        if "error" not in tfidf_stats:
            print(f"  Frecuencia en documentos: {tfidf_stats['document_frequency']}")
            print(f"  Score IDF: {tfidf_stats['idf_score']:.4f}")
            print(f"  Aparece en {tfidf_stats['appears_in_percent']:.1f}% de los documentos")


def interactive_retriever_demo():
    """Demostración interactiva de los retrievers"""
    print("=== DEMO INTERACTIVO DE RETRIEVERS ===\n")
    print("Prueba los algoritmos TF-IDF y BM25 con tus propias consultas.")
    
    # Cargar documentos
    documents = create_sample_documents()
    tfidf_retriever = TFIDFRetriever(documents)
    bm25_retriever = BM25Retriever(documents)
    
    print(f"\n📚 Base de conocimiento cargada: {len(documents)} documentos")
    print("Temas disponibles: Python, Machine Learning, Desarrollo Web, APIs, Análisis de Datos\n")
    
    while True:
        print("Opciones:")
        print("1. Buscar con ambos algoritmos")
        print("2. Explicar score BM25 de un resultado")
        print("3. Ver estadísticas de un término")
        print("4. Listar todos los documentos")
        print("5. Salir")
        
        choice = input("\nElige una opción (1-5): ").strip()
        
        if choice == "1":
            query = input("\n🔍 Ingresa tu consulta: ").strip()
            if query:
                print(f"\nBuscando: '{query}'\n")
                
                # TF-IDF
                print("📊 RESULTADOS TF-IDF:")
                tfidf_results = tfidf_retriever.search(query, top_k=5)
                
                print("\n📊 RESULTADOS BM25:")  
                bm25_results = bm25_retriever.search(query, top_k=5)
                
                # Mostrar contenido del mejor resultado
                if bm25_results:
                    best_doc = bm25_results[0][0]
                    print(f"\n📄 CONTENIDO DEL MEJOR RESULTADO:")
                    print(f"Título: {best_doc.title}")
                    print(f"Contenido (primeras 200 chars): {best_doc.content[:200]}...")
        
        elif choice == "2":
            query = input("\n🔍 Consulta para explicar: ").strip()
            doc_num = input("Número de documento (1-5): ").strip()
            
            try:
                doc_idx = int(doc_num) - 1
                if 0 <= doc_idx < len(documents):
                    explanation = bm25_retriever.explain_score(query, doc_idx)
                    
                    print(f"\n🧮 EXPLICACIÓN DEL SCORE BM25:")
                    print(f"Documento: {explanation['document_title']}")
                    print(f"Score total: {explanation['total_score']:.4f}")
                    print(f"Parámetros: k1={explanation['k1']}, b={explanation['b']}")
                    print(f"Longitud doc: {explanation['document_length']}, promedio: {explanation['avg_document_length']:.1f}")
                    
                    print(f"\n📊 Scores por término:")
                    for term_info in explanation['term_scores']:
                        print(f"  '{term_info['term']}': TF={term_info['tf']}, IDF={term_info['idf']:.3f}, Score={term_info['term_score']:.4f}")
                else:
                    print("❌ Número de documento inválido")
            except ValueError:
                print("❌ Por favor ingresa un número válido")
        
        elif choice == "3":
            term = input("\n🔤 Término a analizar: ").strip().lower()
            if term:
                stats = tfidf_retriever.get_term_statistics(term)
                if "error" in stats:
                    print(f"❌ {stats['error']}")
                else:
                    print(f"\n📈 ESTADÍSTICAS DEL TÉRMINO '{term}':")
                    print(f"Aparece en {stats['document_frequency']} documentos")
                    print(f"Score IDF: {stats['idf_score']:.4f}")
                    print(f"Porcentaje de documentos: {stats['appears_in_percent']:.1f}%")
        
        elif choice == "4":
            print(f"\n📚 DOCUMENTOS EN LA BASE DE CONOCIMIENTO:")
            for i, doc in enumerate(documents, 1):
                print(f"{i}. {doc.title}")
                print(f"   Tags: {', '.join(doc.tags)}")
                print(f"   Palabras: {doc.word_count()}")
                print()
        
        elif choice == "5":
            print("\n👋 ¡Hasta luego!")
            break
        
        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    # Ejecutar pruebas automatizadas
    test_retrievers()
    
    # Demostración interactiva  
    print("\n¿Quieres probar el demo interactivo? (y/n): ", end="")
    if input().lower().startswith('y'):
        interactive_retriever_demo()
    
    print("\n=== EJERCICIOS ADICIONALES ===")
    print("1. Implementa stemming/lemmatización en el preprocessor")
    print("2. Agrega soporte para frases (n-gramas)")
    print("3. Implementa normalización L2 para vectores TF-IDF")
    print("4. Crea índice invertido para búsqueda más eficiente") 
    print("5. Agrega soporte para sinónimos y expansión de consultas")
