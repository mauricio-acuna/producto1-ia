"""
RepoGPT - Capstone Project: Agente IA Completo para Análisis de Repositorios
===========================================================================

Este es el proyecto capstone que integra todos los módulos del curso:
- Módulo A: Conceptos esenciales de agentes IA
- Módulo B: Arquitectura PEC (Planner-Executor-Critic)
- Módulo C: Sistema RAG con citas canónicas
- Módulo D: Evaluación automática y monitoreo

Autor: Sistema de IA Educativo
Curso: Agentes IA para Desarrolladores
"""

import os
import json
import time
import ast
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

# Importar módulos del curso (simulados para el capstone)
# En implementación real, estos serían imports de los módulos anteriores
from abc import ABC, abstractmethod

# =============================================================================
# 1. CONFIGURACIÓN Y MODELOS DE DATOS
# =============================================================================

@dataclass
class RepoConfig:
    """Configuración del repositorio a analizar"""
    repo_path: str
    repo_name: str
    languages: List[str]
    exclude_patterns: List[str] = None
    max_file_size: int = 1000000  # 1MB
    
    def __post_init__(self):
        if self.exclude_patterns is None:
            self.exclude_patterns = [
                "*.pyc", "__pycache__", ".git", "node_modules", 
                "venv", ".env", "*.log", "dist", "build"
            ]

@dataclass
class CodeFile:
    """Representación de un archivo de código"""
    file_path: str
    content: str
    language: str
    size: int
    lines: int
    functions: List[str]
    classes: List[str]
    imports: List[str]
    complexity_score: float
    last_modified: float
    file_hash: str

@dataclass
class QueryResult:
    """Resultado de una consulta al agente"""
    query: str
    answer: str
    sources: List[str]
    citations: List[str]
    confidence: float
    response_time: float
    cost_usd: float
    evaluation_scores: Dict[str, float]
    timestamp: float

# =============================================================================
# 2. HERRAMIENTAS PARA ANÁLISIS DE CÓDIGO
# =============================================================================

class CodeAnalyzer:
    """Analizador de código con soporte multi-lenguaje"""
    
    def __init__(self):
        self.language_patterns = {
            'python': ['.py'],
            'javascript': ['.js', '.jsx'],
            'typescript': ['.ts', '.tsx'],
            'java': ['.java'],
            'cpp': ['.cpp', '.cc', '.cxx'],
            'c': ['.c'],
            'go': ['.go'],
            'rust': ['.rs']
        }
    
    def detect_language(self, file_path: str) -> str:
        """Detectar el lenguaje de programación"""
        ext = Path(file_path).suffix.lower()
        
        for language, extensions in self.language_patterns.items():
            if ext in extensions:
                return language
        
        return 'unknown'
    
    def analyze_python_file(self, content: str) -> Dict[str, Any]:
        """Análisis específico para archivos Python"""
        try:
            tree = ast.parse(content)
            
            functions = []
            classes = []
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            
            # Calcular complejidad básica
            complexity = self._calculate_complexity(tree)
            
            return {
                'functions': functions,
                'classes': classes,
                'imports': imports,
                'complexity': complexity
            }
            
        except SyntaxError:
            return {
                'functions': [],
                'classes': [],
                'imports': [],
                'complexity': 0.0
            }
    
    def analyze_javascript_file(self, content: str) -> Dict[str, Any]:
        """Análisis básico para archivos JavaScript"""
        # Implementación simplificada usando regex
        
        # Buscar funciones
        function_pattern = r'function\s+(\w+)\s*\('
        arrow_function_pattern = r'const\s+(\w+)\s*=\s*\([^)]*\)\s*=>'
        functions = re.findall(function_pattern, content) + re.findall(arrow_function_pattern, content)
        
        # Buscar clases
        class_pattern = r'class\s+(\w+)'
        classes = re.findall(class_pattern, content)
        
        # Buscar imports
        import_pattern = r'import\s+.*?from\s+[\'"]([^\'"]+)[\'"]'
        require_pattern = r'require\([\'"]([^\'"]+)[\'"]\)'
        imports = re.findall(import_pattern, content) + re.findall(require_pattern, content)
        
        return {
            'functions': functions,
            'classes': classes,
            'imports': imports,
            'complexity': len(functions) * 2 + len(classes) * 3  # Heurística simple
        }
    
    def analyze_file(self, file_path: str, content: str) -> CodeFile:
        """Analizar un archivo de código"""
        language = self.detect_language(file_path)
        
        # Análisis específico por lenguaje
        if language == 'python':
            analysis = self.analyze_python_file(content)
        elif language in ['javascript', 'typescript']:
            analysis = self.analyze_javascript_file(content)
        else:
            # Análisis genérico
            analysis = {
                'functions': [],
                'classes': [],
                'imports': [],
                'complexity': 0.0
            }
        
        # Métricas generales
        lines = len(content.split('\n'))
        size = len(content.encode('utf-8'))
        file_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        
        return CodeFile(
            file_path=file_path,
            content=content,
            language=language,
            size=size,
            lines=lines,
            functions=analysis['functions'],
            classes=analysis['classes'],
            imports=analysis['imports'],
            complexity_score=analysis['complexity'],
            last_modified=time.time(),
            file_hash=file_hash
        )
    
    def _calculate_complexity(self, tree: ast.AST) -> float:
        """Calcular complejidad ciclomática básica"""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity

class RepositoryIndexer:
    """Indexador de repositorios de código"""
    
    def __init__(self, config: RepoConfig):
        self.config = config
        self.analyzer = CodeAnalyzer()
        self.indexed_files: List[CodeFile] = []
        self.knowledge_base: Dict[str, Any] = {}
    
    def should_exclude_file(self, file_path: str) -> bool:
        """Verificar si un archivo debe ser excluido"""
        path_str = str(file_path)
        
        for pattern in self.config.exclude_patterns:
            if pattern.startswith('*'):
                if path_str.endswith(pattern[1:]):
                    return True
            elif pattern in path_str:
                return True
        
        return False
    
    def index_repository(self) -> Dict[str, Any]:
        """Indexar todo el repositorio"""
        print(f"🔍 Indexing repository: {self.config.repo_name}")
        print(f"📂 Path: {self.config.repo_path}")
        
        repo_path = Path(self.config.repo_path)
        
        if not repo_path.exists():
            raise FileNotFoundError(f"Repository path not found: {self.config.repo_path}")
        
        files_processed = 0
        files_skipped = 0
        
        # Recorrer todos los archivos
        for file_path in repo_path.rglob('*'):
            if file_path.is_file():
                if self.should_exclude_file(file_path):
                    files_skipped += 1
                    continue
                
                if file_path.stat().st_size > self.config.max_file_size:
                    files_skipped += 1
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Analizar archivo
                    code_file = self.analyzer.analyze_file(str(file_path), content)
                    self.indexed_files.append(code_file)
                    files_processed += 1
                    
                    if files_processed % 10 == 0:
                        print(f"  📄 Processed {files_processed} files...")
                
                except Exception as e:
                    print(f"  ⚠️ Error processing {file_path}: {e}")
                    files_skipped += 1
        
        # Generar knowledge base
        self.knowledge_base = self._build_knowledge_base()
        
        print(f"✅ Indexing complete!")
        print(f"  📊 Files processed: {files_processed}")
        print(f"  ⏭️ Files skipped: {files_skipped}")
        print(f"  🎯 Languages found: {self._get_language_stats()}")
        
        return self.knowledge_base
    
    def _build_knowledge_base(self) -> Dict[str, Any]:
        """Construir base de conocimiento estructurada"""
        kb = {
            'repository': {
                'name': self.config.repo_name,
                'path': self.config.repo_path,
                'total_files': len(self.indexed_files),
                'total_lines': sum(f.lines for f in self.indexed_files),
                'total_size': sum(f.size for f in self.indexed_files),
                'languages': self._get_language_stats(),
                'last_indexed': time.time()
            },
            'files': {},
            'functions': {},
            'classes': {},
            'imports': {},
            'structure': self._analyze_structure()
        }
        
        # Indexar archivos
        for file in self.indexed_files:
            kb['files'][file.file_path] = {
                'language': file.language,
                'lines': file.lines,
                'functions': file.functions,
                'classes': file.classes,
                'complexity': file.complexity_score,
                'content_preview': file.content[:500] + '...' if len(file.content) > 500 else file.content
            }
            
            # Indexar funciones
            for func in file.functions:
                if func not in kb['functions']:
                    kb['functions'][func] = []
                kb['functions'][func].append(file.file_path)
            
            # Indexar clases
            for cls in file.classes:
                if cls not in kb['classes']:
                    kb['classes'][cls] = []
                kb['classes'][cls].append(file.file_path)
            
            # Indexar imports
            for imp in file.imports:
                if imp not in kb['imports']:
                    kb['imports'][imp] = []
                kb['imports'][imp].append(file.file_path)
        
        return kb
    
    def _get_language_stats(self) -> Dict[str, int]:
        """Obtener estadísticas de lenguajes"""
        stats = {}
        for file in self.indexed_files:
            if file.language not in stats:
                stats[file.language] = 0
            stats[file.language] += 1
        return stats
    
    def _analyze_structure(self) -> Dict[str, Any]:
        """Analizar estructura del proyecto"""
        structure = {
            'directories': set(),
            'file_types': {},
            'depth': 0
        }
        
        for file in self.indexed_files:
            path = Path(file.file_path)
            
            # Directorios
            for parent in path.parents:
                structure['directories'].add(str(parent))
            
            # Tipos de archivo
            ext = path.suffix
            if ext not in structure['file_types']:
                structure['file_types'][ext] = 0
            structure['file_types'][ext] += 1
            
            # Profundidad máxima
            depth = len(path.parts)
            structure['depth'] = max(structure['depth'], depth)
        
        structure['directories'] = list(structure['directories'])
        return structure

# =============================================================================
# 3. SISTEMA RAG PARA CÓDIGO (Simplificado del Módulo C)
# =============================================================================

class CodeRAGEngine:
    """Motor RAG especializado para código"""
    
    def __init__(self, knowledge_base: Dict[str, Any]):
        self.knowledge_base = knowledge_base
        self.indexed_content = self._prepare_searchable_content()
    
    def _prepare_searchable_content(self) -> List[Dict[str, Any]]:
        """Preparar contenido para búsqueda"""
        searchable = []
        
        # Indexar archivos
        for file_path, file_info in self.knowledge_base['files'].items():
            doc = {
                'id': f"file_{hashlib.md5(file_path.encode()).hexdigest()[:8]}",
                'type': 'file',
                'path': file_path,
                'content': file_info['content_preview'],
                'metadata': {
                    'language': file_info['language'],
                    'lines': file_info['lines'],
                    'functions': file_info['functions'],
                    'classes': file_info['classes'],
                    'complexity': file_info['complexity']
                }
            }
            searchable.append(doc)
        
        # Indexar funciones
        for func_name, file_paths in self.knowledge_base['functions'].items():
            doc = {
                'id': f"func_{hashlib.md5(func_name.encode()).hexdigest()[:8]}",
                'type': 'function',
                'name': func_name,
                'content': f"Function: {func_name} defined in {', '.join(file_paths)}",
                'metadata': {
                    'files': file_paths,
                    'type': 'function'
                }
            }
            searchable.append(doc)
        
        # Indexar clases
        for class_name, file_paths in self.knowledge_base['classes'].items():
            doc = {
                'id': f"class_{hashlib.md5(class_name.encode()).hexdigest()[:8]}",
                'type': 'class',
                'name': class_name,
                'content': f"Class: {class_name} defined in {', '.join(file_paths)}",
                'metadata': {
                    'files': file_paths,
                    'type': 'class'
                }
            }
            searchable.append(doc)
        
        return searchable
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Búsqueda simple basada en palabras clave"""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        scored_docs = []
        
        for doc in self.indexed_content:
            content_lower = doc['content'].lower()
            content_words = set(content_lower.split())
            
            # Score basado en overlap de palabras
            overlap = len(query_words & content_words)
            
            # Boost por tipo de documento
            type_boost = {
                'function': 1.2 if 'function' in query_lower else 1.0,
                'class': 1.2 if 'class' in query_lower else 1.0,
                'file': 1.0
            }
            
            score = overlap * type_boost.get(doc['type'], 1.0)
            
            # Boost por coincidencia exacta en nombres
            if doc['type'] in ['function', 'class']:
                if query_lower in doc['name'].lower():
                    score *= 2.0
            
            if score > 0:
                scored_docs.append((score, doc))
        
        # Ordenar por score y retornar top-k
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:top_k]]
    
    def generate_response(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
        """Generar respuesta basada en documentos recuperados"""
        
        if not retrieved_docs:
            return "No encontré información relevante en el repositorio para responder tu consulta.", []
        
        # Construir contexto
        context_parts = []
        citations = []
        
        for i, doc in enumerate(retrieved_docs):
            if doc['type'] == 'file':
                context_parts.append(f"Archivo {doc['path']}:\n{doc['content']}")
                citations.append(f"[{doc['path']}#L1-L{doc['metadata']['lines']}]")
            elif doc['type'] == 'function':
                context_parts.append(f"Función {doc['name']} en: {', '.join(doc['metadata']['files'])}")
                citations.append(f"[{doc['metadata']['files'][0]}#function-{doc['name']}]")
            elif doc['type'] == 'class':
                context_parts.append(f"Clase {doc['name']} en: {', '.join(doc['metadata']['files'])}")
                citations.append(f"[{doc['metadata']['files'][0]}#class-{doc['name']}]")
        
        # Generar respuesta (simulada - en implementación real usaríamos LLM)
        response = self._generate_mock_response(query, context_parts, retrieved_docs)
        
        return response, citations[:3]  # Limitar a 3 citas
    
    def _generate_mock_response(self, query: str, context_parts: List[str], docs: List[Dict[str, Any]]) -> str:
        """Generar respuesta simulada basada en el contexto"""
        
        query_lower = query.lower()
        
        # Detectar tipo de consulta
        if 'cómo funciona' in query_lower or 'how does' in query_lower:
            return f"Basándome en el análisis del código, he encontrado {len(docs)} elementos relevantes. " + \
                   f"El funcionamiento se implementa principalmente en los archivos encontrados, " + \
                   f"que incluyen las funciones y clases relacionadas con tu consulta."
        
        elif 'qué es' in query_lower or 'what is' in query_lower:
            return f"Según el código del repositorio, he identificado {len(docs)} definiciones relevantes. " + \
                   f"Los elementos encontrados están distribuidos en varios archivos del proyecto."
        
        elif 'función' in query_lower or 'function' in query_lower:
            functions = [doc for doc in docs if doc['type'] == 'function']
            if functions:
                func_names = [doc['name'] for doc in functions]
                return f"Encontré las siguientes funciones relacionadas: {', '.join(func_names)}. " + \
                       f"Estas funciones implementan la funcionalidad que consultas."
            else:
                return "No encontré funciones específicas relacionadas con tu consulta."
        
        elif 'clase' in query_lower or 'class' in query_lower:
            classes = [doc for doc in docs if doc['type'] == 'class']
            if classes:
                class_names = [doc['name'] for doc in classes]
                return f"Encontré las siguientes clases relacionadas: {', '.join(class_names)}. " + \
                       f"Estas clases definen la estructura y comportamiento del código."
            else:
                return "No encontré clases específicas relacionadas con tu consulta."
        
        else:
            return f"He analizado el repositorio y encontré {len(docs)} elementos relevantes " + \
                   f"distribuidos en los archivos del proyecto. La información está disponible " + \
                   f"en las referencias citadas."

# =============================================================================
# 4. SISTEMA PEC SIMPLIFICADO (Del Módulo B)
# =============================================================================

class SimplePlanner:
    """Planner simplificado para consultas de código"""
    
    def plan(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Crear plan para responder consulta"""
        
        query_lower = query.lower()
        
        # Detectar tipo de consulta
        if any(word in query_lower for word in ['función', 'function', 'método', 'method']):
            query_type = 'function_search'
        elif any(word in query_lower for word in ['clase', 'class']):
            query_type = 'class_search'
        elif any(word in query_lower for word in ['archivo', 'file']):
            query_type = 'file_search'
        elif any(word in query_lower for word in ['estructura', 'architecture', 'organización']):
            query_type = 'structure_analysis'
        elif any(word in query_lower for word in ['dependencia', 'import', 'require']):
            query_type = 'dependency_analysis'
        else:
            query_type = 'general_search'
        
        plan = {
            'query_type': query_type,
            'search_strategy': 'semantic' if len(query.split()) > 3 else 'keyword',
            'max_results': 5,
            'include_citations': True,
            'analysis_depth': 'standard'
        }
        
        return plan

class SimpleExecutor:
    """Executor que utiliza el motor RAG"""
    
    def __init__(self, rag_engine: CodeRAGEngine):
        self.rag_engine = rag_engine
    
    def execute(self, plan: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Ejecutar el plan utilizando RAG"""
        
        # Buscar documentos relevantes
        retrieved_docs = self.rag_engine.search(query, top_k=plan['max_results'])
        
        # Generar respuesta
        response, citations = self.rag_engine.generate_response(query, retrieved_docs)
        
        return {
            'answer': response,
            'sources': [doc['id'] for doc in retrieved_docs],
            'citations': citations,
            'retrieved_docs': retrieved_docs,
            'confidence': min(1.0, len(retrieved_docs) / plan['max_results'])
        }

class SimpleCritic:
    """Critic que evalúa la calidad de las respuestas"""
    
    def evaluate(self, query: str, result: Dict[str, Any]) -> Dict[str, float]:
        """Evaluar calidad de la respuesta"""
        
        scores = {}
        
        # Relevance: basado en número de documentos encontrados
        scores['relevance'] = min(1.0, len(result['retrieved_docs']) / 3)
        
        # Completeness: basado en longitud de respuesta
        answer_length = len(result['answer'])
        scores['completeness'] = min(1.0, answer_length / 200)
        
        # Citation quality: basado en número de citas
        citation_count = len(result['citations'])
        scores['citation_quality'] = min(1.0, citation_count / 2)
        
        # Overall confidence
        scores['overall'] = (scores['relevance'] + scores['completeness'] + scores['citation_quality']) / 3
        
        return scores

# =============================================================================
# 5. AGENTE PRINCIPAL REPOGPT
# =============================================================================

class RepoGPTAgent:
    """Agente principal que integra todos los componentes"""
    
    def __init__(self, repo_config: RepoConfig):
        self.config = repo_config
        self.indexer = RepositoryIndexer(repo_config)
        self.knowledge_base: Optional[Dict[str, Any]] = None
        self.rag_engine: Optional[CodeRAGEngine] = None
        
        # Componentes PEC
        self.planner = SimplePlanner()
        self.executor: Optional[SimpleExecutor] = None
        self.critic = SimpleCritic()
        
        # Métricas y monitoreo (simplificado)
        self.query_history: List[QueryResult] = []
        self.performance_metrics = {
            'total_queries': 0,
            'avg_response_time': 0.0,
            'avg_confidence': 0.0,
            'total_cost': 0.0
        }
    
    def initialize(self) -> bool:
        """Inicializar el agente indexando el repositorio"""
        try:
            print("🚀 Initializing RepoGPT Agent...")
            
            # Indexar repositorio
            self.knowledge_base = self.indexer.index_repository()
            
            # Inicializar RAG
            self.rag_engine = CodeRAGEngine(self.knowledge_base)
            self.executor = SimpleExecutor(self.rag_engine)
            
            print("✅ Agent initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize agent: {e}")
            return False
    
    def query(self, question: str) -> QueryResult:
        """Procesar una consulta usando arquitectura PEC"""
        start_time = time.time()
        
        print(f"\n🔍 Processing query: {question}")
        
        # PLANNER: Crear plan
        plan = self.planner.plan(question, {'knowledge_base': self.knowledge_base})
        print(f"📋 Plan: {plan['query_type']} ({plan['search_strategy']})")
        
        # EXECUTOR: Ejecutar plan
        execution_result = self.executor.execute(plan, question)
        print(f"⚡ Found {len(execution_result['retrieved_docs'])} relevant sources")
        
        # CRITIC: Evaluar resultado
        evaluation_scores = self.critic.evaluate(question, execution_result)
        print(f"🔍 Evaluation scores: {evaluation_scores}")
        
        # Calcular métricas
        response_time = time.time() - start_time
        estimated_cost = 0.001  # Costo simulado
        
        # Crear resultado
        result = QueryResult(
            query=question,
            answer=execution_result['answer'],
            sources=execution_result['sources'],
            citations=execution_result['citations'],
            confidence=execution_result['confidence'],
            response_time=response_time,
            cost_usd=estimated_cost,
            evaluation_scores=evaluation_scores,
            timestamp=time.time()
        )
        
        # Actualizar historial y métricas
        self.query_history.append(result)
        self._update_metrics(result)
        
        return result
    
    def get_repository_summary(self) -> Dict[str, Any]:
        """Obtener resumen del repositorio"""
        if not self.knowledge_base:
            return {}
        
        repo_info = self.knowledge_base['repository']
        
        return {
            'name': repo_info['name'],
            'total_files': repo_info['total_files'],
            'total_lines': repo_info['total_lines'],
            'languages': repo_info['languages'],
            'total_functions': len(self.knowledge_base['functions']),
            'total_classes': len(self.knowledge_base['classes']),
            'unique_imports': len(self.knowledge_base['imports']),
            'last_indexed': datetime.fromtimestamp(repo_info['last_indexed']).strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Obtener métricas de performance del agente"""
        return {
            'performance_metrics': self.performance_metrics.copy(),
            'query_count': len(self.query_history),
            'recent_queries': [
                {
                    'query': q.query,
                    'confidence': q.confidence,
                    'response_time': q.response_time,
                    'cost': q.cost_usd
                }
                for q in self.query_history[-5:]
            ]
        }
    
    def export_knowledge_base(self, filename: str) -> bool:
        """Exportar knowledge base a archivo JSON"""
        if not self.knowledge_base:
            return False
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_base, f, indent=2, ensure_ascii=False, default=str)
            print(f"📄 Knowledge base exported to {filename}")
            return True
        except Exception as e:
            print(f"❌ Failed to export knowledge base: {e}")
            return False
    
    def _update_metrics(self, result: QueryResult):
        """Actualizar métricas de performance"""
        self.performance_metrics['total_queries'] += 1
        
        # Promedio móvil para response time
        current_avg = self.performance_metrics['avg_response_time']
        new_avg = (current_avg * (self.performance_metrics['total_queries'] - 1) + result.response_time) / self.performance_metrics['total_queries']
        self.performance_metrics['avg_response_time'] = new_avg
        
        # Promedio móvil para confidence
        current_avg = self.performance_metrics['avg_confidence']
        new_avg = (current_avg * (self.performance_metrics['total_queries'] - 1) + result.confidence) / self.performance_metrics['total_queries']
        self.performance_metrics['avg_confidence'] = new_avg
        
        # Costo total
        self.performance_metrics['total_cost'] += result.cost_usd

# =============================================================================
# 6. INTERFAZ DE LÍNEA DE COMANDOS
# =============================================================================

class RepoGPTCLI:
    """Interfaz de línea de comandos para RepoGPT"""
    
    def __init__(self):
        self.agent: Optional[RepoGPTAgent] = None
    
    def run(self):
        """Ejecutar interfaz de línea de comandos"""
        print("🤖 RepoGPT - AI Agent for Code Repository Analysis")
        print("=" * 60)
        
        # Configurar repositorio
        repo_path = self._get_repository_path()
        if not repo_path:
            return
        
        # Crear configuración
        config = RepoConfig(
            repo_path=repo_path,
            repo_name=Path(repo_path).name,
            languages=['python', 'javascript', 'typescript']
        )
        
        # Inicializar agente
        self.agent = RepoGPTAgent(config)
        if not self.agent.initialize():
            return
        
        # Mostrar resumen del repositorio
        self._show_repository_summary()
        
        # Loop de consultas
        self._query_loop()
    
    def _get_repository_path(self) -> Optional[str]:
        """Obtener path del repositorio a analizar"""
        print("\n📂 Repository Setup")
        print("-" * 20)
        
        # Usar repositorio de ejemplo por defecto
        default_path = "capstone/sample_repo"
        
        repo_path = input(f"Enter repository path (default: {default_path}): ").strip()
        if not repo_path:
            repo_path = default_path
        
        if not Path(repo_path).exists():
            print(f"❌ Repository path not found: {repo_path}")
            
            # Crear repositorio de ejemplo si no existe
            if repo_path == default_path:
                print("🔧 Creating sample repository...")
                self._create_sample_repository(repo_path)
                return repo_path
            
            return None
        
        return repo_path
    
    def _create_sample_repository(self, repo_path: str):
        """Crear repositorio de ejemplo"""
        Path(repo_path).mkdir(parents=True, exist_ok=True)
        
        # README.md
        readme_content = """# Sample Repository for RepoGPT

This is a sample Python project for demonstrating RepoGPT capabilities.

## Features

- User management
- Data processing
- API endpoints
- Utility functions

## Structure

- `main.py`: Main application entry point
- `models/`: Data models
- `utils/`: Utility functions
- `api/`: API endpoints
"""
        
        with open(f"{repo_path}/README.md", 'w') as f:
            f.write(readme_content)
        
        # main.py
        main_content = '''"""
Main application entry point
"""

from models.user import User
from utils.helpers import format_date
from api.endpoints import create_app

def main():
    """Main function to start the application"""
    app = create_app()
    
    # Create sample user
    user = User("Alice", "alice@example.com")
    print(f"Created user: {user.name}")
    
    # Format current date
    current_date = format_date()
    print(f"Current date: {current_date}")
    
    # Start app
    app.run(debug=True)

if __name__ == "__main__":
    main()
'''
        
        with open(f"{repo_path}/main.py", 'w') as f:
            f.write(main_content)
        
        # models/user.py
        Path(f"{repo_path}/models").mkdir(exist_ok=True)
        user_model_content = '''"""
User model for the application
"""

from datetime import datetime

class User:
    """User class representing application users"""
    
    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email
        self.created_at = datetime.now()
        self.is_active = True
    
    def activate(self):
        """Activate the user account"""
        self.is_active = True
    
    def deactivate(self):
        """Deactivate the user account"""
        self.is_active = False
    
    def update_email(self, new_email: str):
        """Update user email address"""
        self.email = new_email
    
    def __str__(self):
        return f"User(name={self.name}, email={self.email})"
'''
        
        with open(f"{repo_path}/models/user.py", 'w') as f:
            f.write(user_model_content)
        
        # utils/helpers.py
        Path(f"{repo_path}/utils").mkdir(exist_ok=True)
        helpers_content = '''"""
Utility functions for the application
"""

from datetime import datetime
from typing import List, Dict, Any

def format_date(date: datetime = None) -> str:
    """Format date to string representation"""
    if date is None:
        date = datetime.now()
    return date.strftime("%Y-%m-%d %H:%M:%S")

def validate_email(email: str) -> bool:
    """Validate email address format"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def process_data(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Process list of data dictionaries"""
    result = {
        'count': len(data),
        'keys': set(),
        'sample': data[:3] if data else []
    }
    
    for item in data:
        result['keys'].update(item.keys())
    
    result['keys'] = list(result['keys'])
    return result

def calculate_metrics(values: List[float]) -> Dict[str, float]:
    """Calculate basic metrics for a list of values"""
    if not values:
        return {}
    
    return {
        'mean': sum(values) / len(values),
        'min': min(values),
        'max': max(values),
        'count': len(values)
    }
'''
        
        with open(f"{repo_path}/utils/helpers.py", 'w') as f:
            f.write(helpers_content)
        
        # api/endpoints.py
        Path(f"{repo_path}/api").mkdir(exist_ok=True)
        api_content = '''"""
API endpoints for the web application
"""

from flask import Flask, jsonify, request
from models.user import User

def create_app() -> Flask:
    """Create and configure Flask application"""
    app = Flask(__name__)
    
    # In-memory storage for demo
    users = []
    
    @app.route('/users', methods=['GET'])
    def get_users():
        """Get all users"""
        return jsonify([
            {'name': user.name, 'email': user.email, 'active': user.is_active}
            for user in users
        ])
    
    @app.route('/users', methods=['POST'])
    def create_user():
        """Create a new user"""
        data = request.json
        user = User(data['name'], data['email'])
        users.append(user)
        return jsonify({'message': 'User created', 'user_id': len(users) - 1})
    
    @app.route('/users/<int:user_id>', methods=['GET'])
    def get_user(user_id: int):
        """Get specific user"""
        if user_id < len(users):
            user = users[user_id]
            return jsonify({
                'name': user.name,
                'email': user.email,
                'active': user.is_active,
                'created_at': user.created_at.isoformat()
            })
        return jsonify({'error': 'User not found'}), 404
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({'status': 'healthy', 'users_count': len(users)})
    
    return app
'''
        
        with open(f"{repo_path}/api/endpoints.py", 'w') as f:
            f.write(api_content)
        
        print(f"✅ Sample repository created at: {repo_path}")
    
    def _show_repository_summary(self):
        """Mostrar resumen del repositorio"""
        summary = self.agent.get_repository_summary()
        
        print("\n📊 Repository Summary")
        print("-" * 25)
        print(f"Name: {summary['name']}")
        print(f"Files: {summary['total_files']}")
        print(f"Lines of code: {summary['total_lines']:,}")
        print(f"Functions: {summary['total_functions']}")
        print(f"Classes: {summary['total_classes']}")
        print(f"Languages: {', '.join(summary['languages'].keys())}")
        print(f"Last indexed: {summary['last_indexed']}")
    
    def _query_loop(self):
        """Loop principal de consultas"""
        print("\n💬 Query Interface")
        print("-" * 20)
        print("Enter your questions about the repository. Type 'help' for commands or 'quit' to exit.")
        
        while True:
            try:
                query = input("\n🤖 RepoGPT> ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if query.lower() in ['help', 'h']:
                    self._show_help()
                    continue
                
                if query.lower() in ['stats', 'metrics']:
                    self._show_metrics()
                    continue
                
                if query.lower().startswith('export'):
                    filename = query.split(' ')[1] if len(query.split()) > 1 else 'knowledge_base.json'
                    self.agent.export_knowledge_base(filename)
                    continue
                
                # Procesar consulta
                result = self.agent.query(query)
                
                # Mostrar respuesta
                print(f"\n📝 Answer:")
                print(f"{result.answer}")
                
                if result.citations:
                    print(f"\n📎 Sources:")
                    for citation in result.citations:
                        print(f"  {citation}")
                
                print(f"\n📊 Metrics:")
                print(f"  Confidence: {result.confidence:.2f}")
                print(f"  Response time: {result.response_time:.3f}s")
                print(f"  Cost: ${result.cost_usd:.4f}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _show_help(self):
        """Mostrar ayuda"""
        print("\n📖 Available Commands:")
        print("  help, h          - Show this help")
        print("  stats, metrics   - Show performance metrics")
        print("  export <file>    - Export knowledge base to JSON")
        print("  quit, exit, q    - Exit RepoGPT")
        print("\n💡 Example queries:")
        print("  - ¿Qué funciones hay en el proyecto?")
        print("  - ¿Cómo funciona la clase User?")
        print("  - ¿Qué hace el archivo main.py?")
        print("  - ¿Cuáles son las dependencias del proyecto?")
    
    def _show_metrics(self):
        """Mostrar métricas de performance"""
        metrics = self.agent.get_performance_metrics()
        
        print("\n📊 Performance Metrics:")
        print("-" * 25)
        perf = metrics['performance_metrics']
        print(f"Total queries: {perf['total_queries']}")
        print(f"Average response time: {perf['avg_response_time']:.3f}s")
        print(f"Average confidence: {perf['avg_confidence']:.2f}")
        print(f"Total cost: ${perf['total_cost']:.4f}")
        
        if metrics['recent_queries']:
            print(f"\n🕒 Recent Queries:")
            for i, q in enumerate(metrics['recent_queries'], 1):
                print(f"  {i}. {q['query'][:50]}... (conf: {q['confidence']:.2f})")

# =============================================================================
# 7. FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """Función principal para ejecutar RepoGPT"""
    cli = RepoGPTCLI()
    cli.run()

if __name__ == "__main__":
    main()
