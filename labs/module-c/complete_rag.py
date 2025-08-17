"""
Laboratorio Módulo C: RAG Completo
Integración completa de Retrieval + Ranking + Generation con agente PEC
"""

import os
import sys
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from collections import defaultdict

# Importar componentes de laboratorios anteriores
try:
    from basic_retriever import TFIDFRetriever, BM25Retriever, Document, TextPreprocessor
    from citation_system import Citation, CitationGenerator, CitationValidator, CitationManager
    # Importar del módulo B si está disponible
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'module-b'))
    from pec_agent import PECAgent, ExecutionStatus, ExecutionResult
except ImportError as e:
    print(f"⚠️ Advertencia: No se pudieron importar algunos componentes: {e}")
    print("Asegúrate de que los laboratorios anteriores estén en la misma carpeta.")


@dataclass
class RAGResult:
    """Resultado completo de una consulta RAG"""
    query: str
    response: str
    sources: List[Citation]
    retrieval_time: float
    generation_time: float
    total_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            'query': self.query,
            'response': self.response,
            'sources': [citation.to_dict() for citation in self.sources],
            'retrieval_time': self.retrieval_time,
            'generation_time': self.generation_time,
            'total_time': self.total_time,
            'metadata': self.metadata,
            'timestamp': datetime.now().isoformat()
        }


class MMRRanker:
    """
    Re-ranker usando Maximal Marginal Relevance para diversidad
    
    TODO: Implementar MMR para evitar redundancia en resultados
    """
    
    def __init__(self, lambda_param: float = 0.7):
        self.lambda_param = lambda_param  # Balance relevancia vs diversidad
        self.preprocessor = TextPreprocessor()
    
    def rank(self, 
             query: str,
             documents: List[Tuple[Document, float]],
             top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        TODO: Re-rankear documentos usando MMR
        
        Algoritmo MMR:
        1. Seleccionar documento más relevante
        2. Para cada documento restante, calcular:
           MMR = λ * sim(d, q) - (1-λ) * max(sim(d, d_i)) para d_i ya seleccionados
        3. Seleccionar documento con mayor MMR
        4. Repetir hasta completar top_k
        
        Args:
            query: Consulta del usuario
            documents: Lista de (Document, relevance_score)
            top_k: Número de documentos a retornar
            
        Returns:
            Lista re-rankeada de (Document, final_score)
        """
        if not documents:
            return []
        
        if len(documents) <= 1:
            return documents[:top_k]
        
        print(f"🔄 Re-ranking con MMR (λ={self.lambda_param}, top_k={top_k})")
        
        # Preparar datos
        docs, original_scores = zip(*documents)
        query_tokens = set(self.preprocessor.tokenize(query))
        
        # Vectorizar documentos (representación simple con tokens)
        doc_token_sets = []
        for doc in docs:
            doc_tokens = set(self.preprocessor.tokenize(doc.content))
            doc_token_sets.append(doc_tokens)
        
        # MMR iterativo
        selected_indices = []
        remaining_indices = list(range(len(docs)))
        selected_docs = []
        
        for iteration in range(min(top_k, len(docs))):
            best_mmr_score = -float('inf')
            best_idx = None
            
            for idx in remaining_indices:
                # Relevancia con la consulta (normalizada)
                relevance = original_scores[idx]
                
                # Máxima similaridad con documentos ya seleccionados
                max_similarity = 0.0
                if selected_indices:
                    current_tokens = doc_token_sets[idx]
                    
                    for selected_idx in selected_indices:
                        selected_tokens = doc_token_sets[selected_idx]
                        
                        # Similaridad Jaccard
                        intersection = len(current_tokens & selected_tokens)
                        union = len(current_tokens | selected_tokens)
                        similarity = intersection / union if union > 0 else 0
                        
                        max_similarity = max(max_similarity, similarity)
                
                # Calcular MMR score
                mmr_score = (self.lambda_param * relevance - 
                           (1 - self.lambda_param) * max_similarity)
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = idx
            
            # Seleccionar mejor documento
            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
                selected_docs.append((docs[best_idx], best_mmr_score))
                
                print(f"  {iteration+1}. {docs[best_idx].title[:50]}... (MMR: {best_mmr_score:.4f})")
        
        return selected_docs
    
    def _compute_content_similarity(self, doc1: Document, doc2: Document) -> float:
        """Calcular similaridad entre contenidos de documentos"""
        tokens1 = set(self.preprocessor.tokenize(doc1.content))
        tokens2 = set(self.preprocessor.tokenize(doc2.content))
        
        if not tokens1 and not tokens2:
            return 1.0
        
        if not tokens1 or not tokens2:
            return 0.0
        
        # Similaridad Jaccard
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union


class ResponseGenerator:
    """
    Generador de respuestas usando contexto recuperado
    
    TODO: Implementar generación de respuestas con citas
    """
    
    def __init__(self, 
                 max_context_length: int = 2000,
                 citation_format: str = "canonical"):
        self.max_context_length = max_context_length
        self.citation_format = citation_format
        self.generation_stats = {
            'total_generations': 0,
            'context_truncated': 0,
            'avg_response_length': 0
        }
    
    def generate_response(self,
                         query: str,
                         ranked_documents: List[Tuple[Document, float]],
                         citations: List[Citation]) -> Tuple[str, Dict[str, Any]]:
        """
        TODO: Generar respuesta usando documentos recuperados
        
        Proceso:
        1. Construir contexto desde documentos rankeados
        2. Crear prompt con consulta y contexto
        3. Generar respuesta (simulada)
        4. Agregar citas al final
        
        Args:
            query: Consulta del usuario
            ranked_documents: Documentos ordenados por relevancia
            citations: Citas correspondientes a los documentos
            
        Returns:
            (respuesta_generada, metadatos)
        """
        start_time = time.time()
        
        # Paso 1: Construir contexto
        context = self._build_context(ranked_documents)
        
        # Paso 2: Crear prompt
        prompt = self._create_prompt(query, context)
        
        # Paso 3: Generar respuesta (simulación)
        response = self._simulate_llm_response(prompt, ranked_documents)
        
        # Paso 4: Agregar citas
        response_with_citations = self._add_citations(response, citations)
        
        generation_time = time.time() - start_time
        
        # Actualizar estadísticas
        self.generation_stats['total_generations'] += 1
        current_avg = self.generation_stats['avg_response_length']
        total_gens = self.generation_stats['total_generations']
        new_avg = (current_avg * (total_gens - 1) + len(response_with_citations)) / total_gens
        self.generation_stats['avg_response_length'] = new_avg
        
        metadata = {
            'generation_time': generation_time,
            'context_length': len(context),
            'context_truncated': len(context) >= self.max_context_length,
            'sources_used': len(ranked_documents),
            'citations_added': len(citations)
        }
        
        return response_with_citations, metadata
    
    def _build_context(self, ranked_documents: List[Tuple[Document, float]]) -> str:
        """Construir contexto desde documentos rankeados"""
        context_parts = []
        current_length = 0
        
        for i, (doc, score) in enumerate(ranked_documents):
            # Crear entrada de contexto
            doc_context = f"[Fuente {i+1}: {doc.title}]\n{doc.content}\n"
            
            # Verificar límite de longitud
            if current_length + len(doc_context) > self.max_context_length:
                if not context_parts:  # Al menos incluir una fuente
                    context_parts.append(doc_context[:self.max_context_length])
                self.generation_stats['context_truncated'] += 1
                break
            
            context_parts.append(doc_context)
            current_length += len(doc_context)
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Crear prompt para el LLM"""
        prompt = f"""Basándote en el siguiente contexto, responde a la pregunta del usuario de manera precisa y completa.

CONTEXTO:
{context}

PREGUNTA: {query}

INSTRUCCIONES:
- Responde únicamente basándote en la información proporcionada en el contexto
- Si no hay información suficiente, indícalo claramente
- Sé preciso y evita inventar información
- Estructura tu respuesta de manera clara y legible

RESPUESTA:"""

        return prompt
    
    def _simulate_llm_response(self, prompt: str, ranked_documents: List[Tuple[Document, float]]) -> str:
        """
        Simular respuesta de LLM (en producción usarías OpenAI, etc.)
        """
        # Esta es una simulación básica para fines educativos
        # En producción usarías una API de LLM real
        
        if not ranked_documents:
            return "Lo siento, no encontré información relevante para responder tu pregunta."
        
        # Extraer información clave de los documentos
        key_topics = []
        for doc, score in ranked_documents[:3]:  # Top 3 documentos
            # Extraer algunas líneas relevantes
            lines = doc.content.split('\n')
            relevant_lines = [line.strip() for line in lines if line.strip()][:2]
            key_topics.extend(relevant_lines)
        
        # Generar respuesta estructurada
        response_parts = [
            "Basándome en la información disponible:",
            "",
        ]
        
        # Agregar puntos clave
        for i, topic in enumerate(key_topics[:5], 1):
            if topic:
                response_parts.append(f"• {topic}")
        
        response_parts.extend([
            "",
            "Esta información proviene de las fuentes citadas a continuación."
        ])
        
        return "\n".join(response_parts)
    
    def _add_citations(self, response: str, citations: List[Citation]) -> str:
        """Agregar citas al final de la respuesta"""
        if not citations:
            return response
        
        citation_section = "\n\n**Fuentes:**\n"
        
        for i, citation in enumerate(citations, 1):
            if self.citation_format == "canonical":
                citation_line = f"{i}. {citation.to_canonical()}"
            else:
                citation_line = f"{i}. {citation.uri} (líneas {citation.start_line}-{citation.end_line})"
            
            citation_section += citation_line + "\n"
        
        return response + citation_section


class CompleteRAG:
    """
    Sistema RAG completo integrando todos los componentes
    
    TODO: Implementar pipeline completo de RAG
    """
    
    def __init__(self,
                 documents: List[Document],
                 retriever_type: str = "bm25",
                 use_mmr: bool = True,
                 lambda_mmr: float = 0.7):
        
        self.documents = documents
        self.use_mmr = use_mmr
        
        # Componentes del pipeline
        self.preprocessor = TextPreprocessor()
        self._init_retriever(retriever_type)
        self.mmr_ranker = MMRRanker(lambda_mmr) if use_mmr else None
        self.citation_generator = CitationGenerator(line_context=1)
        self.citation_manager = CitationManager()
        self.response_generator = ResponseGenerator()
        
        # Estadísticas
        self.query_stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'avg_retrieval_time': 0,
            'avg_generation_time': 0,
            'avg_sources_per_query': 0
        }
        
        print(f"✅ RAG System inicializado:")
        print(f"  - Documentos: {len(documents)}")
        print(f"  - Retriever: {retriever_type.upper()}")
        print(f"  - MMR: {'Habilitado' if use_mmr else 'Deshabilitado'}")
    
    def _init_retriever(self, retriever_type: str):
        """Inicializar el retriever especificado"""
        if retriever_type.lower() == "tfidf":
            self.retriever = TFIDFRetriever(self.documents, self.preprocessor)
        elif retriever_type.lower() == "bm25":
            self.retriever = BM25Retriever(self.documents, preprocessor=self.preprocessor)
        else:
            raise ValueError(f"Tipo de retriever no soportado: {retriever_type}")
    
    def query(self, 
              user_query: str,
              top_k: int = 5,
              min_relevance: float = 0.1) -> RAGResult:
        """
        TODO: Procesar consulta completa usando pipeline RAG
        
        Pipeline:
        1. RETRIEVAL: Buscar documentos relevantes
        2. RANKING: Re-rankear con MMR (opcional)
        3. CITATION: Generar citas canónicas
        4. GENERATION: Generar respuesta con contexto
        
        Args:
            user_query: Consulta del usuario
            top_k: Número máximo de documentos a usar
            min_relevance: Score mínimo de relevancia
            
        Returns:
            RAGResult con respuesta completa y metadatos
        """
        start_time = time.time()
        
        try:
            print(f"\n🔍 Procesando consulta: '{user_query}'")
            
            # PASO 1: RETRIEVAL
            retrieval_start = time.time()
            raw_results = self.retriever.search(user_query, top_k=top_k * 2)  # Buscar más para filtrar
            
            # Filtrar por relevancia mínima
            filtered_results = [(doc, score) for doc, score in raw_results if score >= min_relevance]
            retrieval_time = time.time() - retrieval_start
            
            print(f"📊 Retrieval: {len(filtered_results)} documentos (filtrados por score >= {min_relevance})")
            
            # PASO 2: RANKING con MMR (opcional)
            if self.use_mmr and len(filtered_results) > 1:
                ranked_results = self.mmr_ranker.rank(user_query, filtered_results, top_k)
            else:
                ranked_results = filtered_results[:top_k]
            
            print(f"🔄 Ranking: {len(ranked_results)} documentos finales")
            
            # PASO 3: GENERAR CITAS
            citations = self._generate_citations(user_query, ranked_results)
            print(f"📝 Citas: {len(citations)} citas generadas")
            
            # PASO 4: GENERAR RESPUESTA
            generation_start = time.time()
            response, gen_metadata = self.response_generator.generate_response(
                user_query, ranked_results, citations
            )
            generation_time = time.time() - generation_start
            
            total_time = time.time() - start_time
            
            # Crear resultado
            result = RAGResult(
                query=user_query,
                response=response,
                sources=citations,
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                total_time=total_time,
                metadata={
                    'raw_results_count': len(raw_results),
                    'filtered_results_count': len(filtered_results),
                    'final_results_count': len(ranked_results),
                    'mmr_used': self.use_mmr,
                    'min_relevance_threshold': min_relevance,
                    **gen_metadata
                }
            )
            
            # Actualizar estadísticas
            self._update_stats(result)
            
            print(f"✅ Consulta completada en {total_time:.3f}s")
            print(f"   Retrieval: {retrieval_time:.3f}s | Generation: {generation_time:.3f}s")
            
            return result
            
        except Exception as e:
            print(f"❌ Error procesando consulta: {e}")
            
            # Resultado de error
            return RAGResult(
                query=user_query,
                response=f"Lo siento, ocurrió un error procesando tu consulta: {str(e)}",
                sources=[],
                retrieval_time=0,
                generation_time=0,
                total_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
    
    def _generate_citations(self, query: str, ranked_results: List[Tuple[Document, float]]) -> List[Citation]:
        """Generar citas para los documentos rankeados"""
        citations = []
        
        for doc, score in ranked_results:
            # Simular posición del match (en producción usarías búsqueda real)
            query_tokens = self.preprocessor.tokenize(query)
            best_match_start = 0
            best_match_length = min(len(doc.content), 100)  # Primeros 100 chars como aproximación
            
            citation = self.citation_generator.generate_citation(
                document_uri=doc.uri,
                full_content=doc.content,
                match_start_char=best_match_start,
                match_end_char=best_match_start + best_match_length,
                relevance_score=score,
                metadata={
                    'query': query,
                    'document_title': doc.title,
                    'document_id': doc.id
                }
            )
            
            if citation:
                citations.append(citation)
                
                # Agregar al manager para tracking
                success, cite_id = self.citation_manager.add_citation(citation)
                if success:
                    citation.metadata['citation_id'] = cite_id
        
        return citations
    
    def _update_stats(self, result: RAGResult):
        """Actualizar estadísticas del sistema"""
        self.query_stats['total_queries'] += 1
        
        if not result.metadata.get('error'):
            self.query_stats['successful_queries'] += 1
        
        # Actualizar promedios
        total = self.query_stats['total_queries']
        
        # Tiempo de retrieval promedio
        current_retr_avg = self.query_stats['avg_retrieval_time']
        self.query_stats['avg_retrieval_time'] = (current_retr_avg * (total - 1) + result.retrieval_time) / total
        
        # Tiempo de generación promedio
        current_gen_avg = self.query_stats['avg_generation_time']
        self.query_stats['avg_generation_time'] = (current_gen_avg * (total - 1) + result.generation_time) / total
        
        # Fuentes promedio por consulta
        current_sources_avg = self.query_stats['avg_sources_per_query']
        self.query_stats['avg_sources_per_query'] = (current_sources_avg * (total - 1) + len(result.sources)) / total
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas completas del sistema"""
        retriever_stats = {}
        if hasattr(self.retriever, 'get_term_statistics'):
            # Stats específicas del retriever si están disponibles
            pass
        
        return {
            'query_stats': self.query_stats,
            'retriever_type': type(self.retriever).__name__,
            'document_count': len(self.documents),
            'mmr_enabled': self.use_mmr,
            'citation_manager_stats': self.citation_manager.get_manager_stats(),
            'response_generator_stats': self.response_generator.generation_stats
        }
    
    def batch_query(self, queries: List[str], **kwargs) -> List[RAGResult]:
        """Procesar múltiples consultas en lote"""
        results = []
        
        print(f"🔄 Procesando {len(queries)} consultas en lote...")
        
        for i, query in enumerate(queries, 1):
            print(f"\n--- Consulta {i}/{len(queries)} ---")
            result = self.query(query, **kwargs)
            results.append(result)
        
        return results


class RAGWithAgent:
    """
    Integración de RAG con agente PEC del módulo B
    
    TODO: Integrar RAG como herramienta en agente PEC
    """
    
    def __init__(self, documents: List[Document]):
        self.rag_system = CompleteRAG(documents)
        
        # Intentar crear agente PEC si está disponible
        try:
            self.pec_agent = PECAgent(['rag_search', 'summarize', 'format_response'])
            self.agent_available = True
            self._register_rag_tool()
        except (NameError, ImportError):
            print("⚠️ Agente PEC no disponible. Funcionando solo con RAG.")
            self.pec_agent = None
            self.agent_available = False
    
    def _register_rag_tool(self):
        """Registrar RAG como herramienta en el agente"""
        if not self.agent_available:
            return
        
        # Implementación de la herramienta RAG
        def rag_search_impl(query: str, top_k: int = 3) -> Dict[str, Any]:
            """Implementación de búsqueda RAG"""
            result = self.rag_system.query(query, top_k=top_k)
            
            return {
                'response': result.response,
                'sources_count': len(result.sources),
                'sources': [cite.to_canonical() for cite in result.sources],
                'retrieval_time': result.retrieval_time
            }
        
        # Registrar herramienta (código simplificado)
        try:
            from tool_registry import ToolDefinition
            
            rag_tool = ToolDefinition(
                name="rag_search",
                description="Buscar información usando RAG con citas canónicas",
                parameters_schema={
                    "query": {"type": "string", "required": True, "min_length": 1},
                    "top_k": {"type": "number", "required": False, "min_value": 1, "max_value": 10}
                },
                function=rag_search_impl
            )
            
            self.pec_agent.tool_registry.register_tool(rag_tool)
            print("✅ Herramienta RAG registrada en agente PEC")
            
        except ImportError:
            print("⚠️ No se pudo registrar herramienta RAG en agente")
    
    def process_with_agent(self, user_goal: str) -> Dict[str, Any]:
        """Procesar solicitud usando agente PEC + RAG"""
        if not self.agent_available:
            # Fallback a RAG directo
            result = self.rag_system.query(user_goal)
            return {
                'success': True,
                'response': result.response,
                'sources': [cite.to_canonical() for cite in result.sources],
                'method': 'rag_direct'
            }
        
        # Usar agente PEC
        return self.pec_agent.process_request(user_goal)
    
    def direct_rag_query(self, query: str, **kwargs) -> RAGResult:
        """Acceso directo al sistema RAG"""
        return self.rag_system.query(query, **kwargs)


def create_knowledge_base() -> List[Document]:
    """Crear base de conocimiento ampliada para pruebas"""
    knowledge_docs = [
        {
            "id": "kb_001",
            "uri": "kb://ai-fundamentals.md",
            "title": "Fundamentos de Inteligencia Artificial",
            "content": """La Inteligencia Artificial (IA) es una disciplina que busca crear sistemas capaces de realizar tareas que normalmente requieren inteligencia humana.

Los principales enfoques de IA incluyen:

1. Machine Learning (Aprendizaje Automático)
   - Algoritmos que mejoran con la experiencia
   - Incluye aprendizaje supervisado, no supervisado y por refuerzo
   - Aplicaciones en reconocimiento de patrones y predicción

2. Deep Learning (Aprendizaje Profundo)
   - Redes neuronales con múltiples capas
   - Excelente para procesamiento de imágenes, texto y audio
   - Requiere grandes cantidades de datos para entrenamiento

3. Natural Language Processing (NLP)
   - Procesamiento y comprensión del lenguaje natural
   - Incluye análisis de sentimientos, traducción y generación de texto
   - Los transformers han revolucionado este campo

4. Computer Vision
   - Interpretación de información visual
   - Aplicaciones en reconocimiento facial, detección de objetos
   - Uso de CNNs (Convolutional Neural Networks)

La IA moderna se basa en tres pilares fundamentales: algoritmos avanzados, poder computacional y grandes datasets.""",
            "tags": ["ai", "machine-learning", "deep-learning", "nlp", "computer-vision"]
        },
        {
            "id": "kb_002",
            "uri": "kb://rag-systems.md",
            "title": "Sistemas RAG: Retrieval-Augmented Generation",
            "content": """RAG (Retrieval-Augmented Generation) es una arquitectura que combina recuperación de información con generación de texto para crear respuestas más precisas y actualizadas.

Componentes principales de RAG:

1. Retriever (Recuperador)
   - Busca documentos relevantes en una base de conocimiento
   - Algoritmos comunes: TF-IDF, BM25, embeddings densos
   - Optimización importante para la calidad final

2. Ranker (Clasificador)
   - Re-ordena documentos por relevancia
   - Técnicas como MMR (Maximal Marginal Relevance)
   - Reduce redundancia y mejora diversidad

3. Generator (Generador)
   - LLM que genera respuesta usando contexto recuperado
   - Modelos como GPT, T5, BART
   - Importante el diseño del prompt

Ventajas de RAG:
- Información siempre actualizada sin reentrenamiento
- Respuestas con fuentes verificables
- Reduce alucinaciones del modelo
- Más eficiente que reentrenar modelos

Desafíos:
- Calidad del retrieval afecta resultado final
- Latencia adicional por búsqueda
- Necesidad de bases de conocimiento bien curadas
- Balancear relevancia vs diversidad""",
            "tags": ["rag", "retrieval", "generation", "nlp", "information-retrieval"]
        },
        {
            "id": "kb_003",
            "uri": "kb://python-ecosystem.md",
            "title": "Ecosistema Python para Data Science",
            "content": """Python se ha establecido como el lenguaje líder para ciencia de datos, machine learning e inteligencia artificial.

Bibliotecas fundamentales:

1. NumPy - Computación numérica
   - Arrays multidimensionales eficientes
   - Operaciones matemáticas vectorizadas
   - Base para muchas otras bibliotecas

2. Pandas - Manipulación de datos
   - DataFrames para datos estructurados
   - Limpieza y transformación de datos
   - Análisis exploratorio de datos (EDA)

3. Scikit-learn - Machine Learning
   - Algoritmos de clasificación, regresión, clustering
   - Herramientas de preprocesamiento
   - Métricas de evaluación

4. TensorFlow/PyTorch - Deep Learning
   - Frameworks para redes neuronales
   - Soporte para GPU y entrenamiento distribuido
   - Ecosistemas completos con herramientas adicionales

5. Matplotlib/Seaborn - Visualización
   - Gráficos estáticos y dinámicos
   - Visualizaciones estadísticas especializadas

Herramientas de desarrollo:
- Jupyter Notebooks para prototipado
- IDEs como PyCharm, VSCode
- Entornos virtuales con conda o venv
- Gestión de dependencias con pip/conda

El ecosistema Python permite un flujo completo desde exploración de datos hasta despliegue en producción.""",
            "tags": ["python", "data-science", "numpy", "pandas", "scikit-learn", "visualization"]
        },
        {
            "id": "kb_004",
            "uri": "kb://api-design-best-practices.md",
            "title": "Mejores Prácticas en Diseño de APIs",
            "content": """El diseño de APIs efectivas es crucial para el desarrollo de sistemas escalables y mantenibles.

Principios REST fundamentales:

1. Recursos y URIs
   - URLs descriptivas y jerárquicas
   - Uso de sustantivos, no verbos
   - Ejemplo: /api/users/123/orders

2. Métodos HTTP apropiados
   - GET: obtener recursos
   - POST: crear nuevos recursos
   - PUT: actualizar recursos completos
   - PATCH: actualizaciones parciales
   - DELETE: eliminar recursos

3. Códigos de estado HTTP
   - 200 OK: operación exitosa
   - 201 Created: recurso creado
   - 400 Bad Request: solicitud malformada
   - 401 Unauthorized: autenticación requerida
   - 404 Not Found: recurso no encontrado
   - 500 Internal Server Error: error del servidor

Mejores prácticas de seguridad:
- Autenticación con tokens JWT
- Validación estricta de entrada
- Rate limiting para prevenir abuso
- HTTPS obligatorio en producción
- Sanitización de datos de salida

Documentación:
- Especificaciones OpenAPI/Swagger
- Ejemplos de uso claros
- Descripción de errores posibles
- Versionado de API explícito

Monitoreo y observabilidad:
- Logging detallado de requests
- Métricas de rendimiento
- Alertas por errores frecuentes
- Health checks automatizados""",
            "tags": ["api", "rest", "http", "security", "documentation", "best-practices"]
        },
        {
            "id": "kb_005",
            "uri": "kb://vector-databases.md",
            "title": "Bases de Datos Vectoriales para IA",
            "content": """Las bases de datos vectoriales son esenciales para aplicaciones modernas de IA que manejan embeddings y búsqueda semántica.

¿Qué son los embeddings?
- Representaciones numéricas densas de datos
- Capturan similitud semántica
- Generados por modelos de ML entrenados
- Típicamente vectores de 384, 768, 1536 dimensiones

Operaciones principales:
1. Indexación - Almacenar vectores eficientemente
2. Búsqueda - Encontrar vectores similares (KNN)
3. Filtrado - Combinar búsqueda vectorial con metadatos

Algoritmos de búsqueda:
- Exact search: búsqueda exhaustiva (lento pero preciso)
- HNSW: Hierarchical Navigable Small World
- IVF: Inverted File Index
- LSH: Locality Sensitive Hashing

Bases de datos populares:
1. Pinecone - Servicio managed en la nube
2. Weaviate - Open source con capacidades de NLP
3. Qdrant - Rápido y eficiente en memoria
4. Chroma - Ligero para prototipado
5. Faiss - Biblioteca de Facebook para búsqueda

Casos de uso:
- Búsqueda semántica en documentos
- Sistemas de recomendación
- Detección de duplicados
- Análisis de similitud de imágenes
- RAG (Retrieval-Augmented Generation)

Consideraciones de rendimiento:
- Trade-off entre precisión y velocidad
- Tamaño del índice vs memoria disponible
- Latencia de consulta vs throughput
- Consistencia en escrituras concurrentes""",
            "tags": ["vector-database", "embeddings", "similarity-search", "ai", "indexing"]
        }
    ]
    
    documents = []
    for doc_data in knowledge_docs:
        doc = Document(
            id=doc_data["id"],
            uri=doc_data["uri"],
            title=doc_data["title"],
            content=doc_data["content"].strip(),
            author="Knowledge Base Team",
            created_at=datetime.now(),
            tags=doc_data["tags"],
            metadata={"source": "knowledge_base", "domain": "ai_tech"}
        )
        documents.append(doc)
    
    return documents


def test_complete_rag():
    """Pruebas del sistema RAG completo"""
    print("=== PRUEBAS DEL SISTEMA RAG COMPLETO ===\n")
    
    # Crear base de conocimiento
    documents = create_knowledge_base()
    print(f"📚 Base de conocimiento: {len(documents)} documentos")
    
    # Crear sistemas RAG con diferentes configuraciones
    systems = {
        "BM25 + MMR": CompleteRAG(documents, retriever_type="bm25", use_mmr=True),
        "TF-IDF sin MMR": CompleteRAG(documents, retriever_type="tfidf", use_mmr=False),
        "BM25 sin MMR": CompleteRAG(documents, retriever_type="bm25", use_mmr=False)
    }
    
    # Consultas de prueba
    test_queries = [
        "¿Qué es RAG y cuáles son sus componentes principales?",
        "¿Cuáles son las mejores bibliotecas de Python para machine learning?",
        "¿Cómo diseñar APIs REST seguras?",
        "¿Qué son las bases de datos vectoriales y para qué se usan?",
        "¿Cuáles son los principales enfoques de inteligencia artificial?"
    ]
    
    print(f"\n🧪 Ejecutando {len(test_queries)} consultas en {len(systems)} sistemas...\n")
    
    all_results = {}
    
    for system_name, rag_system in systems.items():
        print(f"{'='*60}")
        print(f"SISTEMA: {system_name}")
        print(f"{'='*60}")
        
        system_results = []
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Consulta {i}: {query[:50]}... ---")
            
            result = rag_system.query(query, top_k=3, min_relevance=0.1)
            system_results.append(result)
            
            print(f"📝 Respuesta generada ({len(result.response)} chars)")
            print(f"📊 Fuentes: {len(result.sources)} citas")
            print(f"⏱️ Tiempo total: {result.total_time:.3f}s")
            
            # Mostrar respuesta resumida
            response_preview = result.response[:200] + "..." if len(result.response) > 200 else result.response
            print(f"💬 Vista previa: {response_preview}")
        
        all_results[system_name] = system_results
        
        # Estadísticas del sistema
        stats = rag_system.get_system_stats()
        print(f"\n📈 ESTADÍSTICAS DEL SISTEMA:")
        print(f"  Consultas exitosas: {stats['query_stats']['successful_queries']}/{stats['query_stats']['total_queries']}")
        print(f"  Tiempo promedio retrieval: {stats['query_stats']['avg_retrieval_time']:.3f}s")
        print(f"  Tiempo promedio generación: {stats['query_stats']['avg_generation_time']:.3f}s")
        print(f"  Fuentes promedio por consulta: {stats['query_stats']['avg_sources_per_query']:.1f}")
    
    # Comparación entre sistemas
    print(f"\n{'='*60}")
    print(f"COMPARACIÓN ENTRE SISTEMAS")
    print(f"{'='*60}")
    
    for query_idx, query in enumerate(test_queries):
        print(f"\n📋 Consulta {query_idx + 1}: {query}")
        print(f"{'─'*40}")
        
        for system_name in systems.keys():
            result = all_results[system_name][query_idx]
            print(f"  {system_name}:")
            print(f"    Tiempo: {result.total_time:.3f}s | Fuentes: {len(result.sources)} | Chars: {len(result.response)}")


def interactive_rag_demo():
    """Demostración interactiva del sistema RAG"""
    print("=== DEMO INTERACTIVO - SISTEMA RAG COMPLETO ===\n")
    print("Explora el sistema RAG con consultas personalizadas.")
    
    # Crear sistema RAG
    documents = create_knowledge_base()
    rag_system = CompleteRAG(documents, retriever_type="bm25", use_mmr=True)
    
    print(f"\n🚀 Sistema RAG inicializado:")
    print(f"  - {len(documents)} documentos en la base de conocimiento")
    print(f"  - Retriever: BM25")
    print(f"  - Re-ranking: MMR habilitado")
    print(f"  - Generación: Con citas canónicas")
    
    # Mostrar temas disponibles
    all_tags = set()
    for doc in documents:
        all_tags.update(doc.tags)
    
    print(f"\n📚 Temas disponibles: {', '.join(sorted(all_tags))}")
    
    while True:
        print(f"\nOpciones:")
        print("1. Hacer consulta RAG")
        print("2. Consulta con configuración personalizada")
        print("3. Ver estadísticas del sistema")
        print("4. Explorar documentos de la base")
        print("5. Exportar resultados")
        print("6. Salir")
        
        choice = input("\nElige una opción (1-6): ").strip()
        
        if choice == "1":
            query = input("\n🔍 Tu pregunta: ").strip()
            if query:
                print(f"\n🚀 Procesando consulta...")
                
                result = rag_system.query(query)
                
                print(f"\n{'='*60}")
                print(f"RESULTADO RAG")
                print(f"{'='*60}")
                print(f"📝 Respuesta:\n{result.response}")
                
                if result.sources:
                    print(f"\n📚 Fuentes utilizadas:")
                    for i, citation in enumerate(result.sources, 1):
                        print(f"  {i}. {citation.to_canonical()}")
                        print(f"     Score: {citation.relevance_score:.3f}")
                
                print(f"\n⏱️ Rendimiento:")
                print(f"  Retrieval: {result.retrieval_time:.3f}s")
                print(f"  Generación: {result.generation_time:.3f}s")
                print(f"  Total: {result.total_time:.3f}s")
        
        elif choice == "2":
            query = input("\n🔍 Tu pregunta: ").strip()
            if query:
                print(f"\n⚙️ Configuración personalizada:")
                top_k = int(input("Número máximo de fuentes (3): ") or "3")
                min_rel = float(input("Relevancia mínima (0.1): ") or "0.1")
                
                result = rag_system.query(query, top_k=top_k, min_relevance=min_rel)
                
                print(f"\n📝 Respuesta: {result.response}")
                print(f"📊 Configuración usada: top_k={top_k}, min_relevance={min_rel}")
                print(f"🔢 Fuentes encontradas: {len(result.sources)}")
        
        elif choice == "3":
            stats = rag_system.get_system_stats()
            
            print(f"\n📊 ESTADÍSTICAS DEL SISTEMA RAG:")
            print(f"Consultas procesadas: {stats['query_stats']['total_queries']}")
            print(f"Consultas exitosas: {stats['query_stats']['successful_queries']}")
            print(f"Tasa de éxito: {stats['query_stats']['successful_queries']/max(1, stats['query_stats']['total_queries']):.1%}")
            print(f"Tiempo promedio retrieval: {stats['query_stats']['avg_retrieval_time']:.3f}s")
            print(f"Tiempo promedio generación: {stats['query_stats']['avg_generation_time']:.3f}s")
            print(f"Fuentes promedio por consulta: {stats['query_stats']['avg_sources_per_query']:.1f}")
            
            print(f"\nEstadísticas de citas:")
            cite_stats = stats['citation_manager_stats']
            print(f"Total citas generadas: {cite_stats['total_citations']}")
            print(f"URIs únicos: {cite_stats['unique_uris']}")
            print(f"Score promedio de relevancia: {cite_stats['avg_relevance_score']:.3f}")
        
        elif choice == "4":
            print(f"\n📚 DOCUMENTOS EN LA BASE DE CONOCIMIENTO:")
            for i, doc in enumerate(documents, 1):
                print(f"\n{i}. {doc.title}")
                print(f"   URI: {doc.uri}")
                print(f"   Tags: {', '.join(doc.tags)}")
                print(f"   Contenido: {len(doc.content)} caracteres")
                
                # Mostrar vista previa
                preview = doc.content[:150] + "..." if len(doc.content) > 150 else doc.content
                print(f"   Vista previa: {preview}")
        
        elif choice == "5":
            filename = input("\n💾 Nombre del archivo JSON (rag_results.json): ").strip() or "rag_results.json"
            
            # Exportar estadísticas del sistema
            export_data = {
                'exported_at': datetime.now().isoformat(),
                'system_stats': rag_system.get_system_stats(),
                'documents_count': len(documents),
                'available_tags': list(all_tags)
            }
            
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                print(f"✅ Estadísticas exportadas a {filename}")
            except Exception as e:
                print(f"❌ Error exportando: {e}")
        
        elif choice == "6":
            print("\n👋 ¡Hasta luego!")
            break
        
        print(f"\n{'='*50}")


if __name__ == "__main__":
    # Ejecutar pruebas automatizadas
    test_complete_rag()
    
    # Demostración interactiva
    print("\n¿Quieres probar el demo interactivo? (y/n): ", end="")
    if input().lower().startswith('y'):
        interactive_rag_demo()
    
    print("\n=== EJERCICIOS ADICIONALES ===")
    print("1. Integra embeddings densos (sentence-transformers)")
    print("2. Implementa caché de resultados para consultas frecuentes") 
    print("3. Agrega métricas de evaluación automática (BLEU, ROUGE)")
    print("4. Crea interfaz web con FastAPI para el sistema RAG")
    print("5. Implementa RAG híbrido (sparse + dense retrieval)")
