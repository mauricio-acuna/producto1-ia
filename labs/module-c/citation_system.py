"""
Laboratorio Módulo C: Sistema de Citas Canónicas
Implementación de generación y validación de citas en formato uri#Lx-Ly
"""

import re
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path


@dataclass
class Citation:
    """
    Cita canónica con metadatos completos
    
    Formato: uri#Lx-Ly donde:
    - uri: Identificador único del documento
    - Lx: Línea de inicio (1-indexed)
    - Ly: Línea de fin (1-indexed)
    """
    uri: str
    start_line: int
    end_line: int
    content: str
    relevance_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_canonical(self) -> str:
        """Generar cita en formato canónico uri#Lx-Ly"""
        return f"{self.uri}#L{self.start_line}-L{self.end_line}"
    
    def get_line_range(self) -> Tuple[int, int]:
        """Obtener rango de líneas como tupla"""
        return (self.start_line, self.end_line)
    
    def get_line_count(self) -> int:
        """Obtener número de líneas citadas"""
        return self.end_line - self.start_line + 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para serialización"""
        return {
            **asdict(self),
            'canonical_reference': self.to_canonical(),
            'line_count': self.get_line_count(),
            'created_at': self.created_at.isoformat()
        }
    
    def __str__(self) -> str:
        """Representación string de la cita"""
        content_preview = self.content[:100] + "..." if len(self.content) > 100 else self.content
        return f"[{self.to_canonical()}] {content_preview}"
    
    def __repr__(self) -> str:
        return f"Citation(uri='{self.uri}', lines={self.start_line}-{self.end_line}, score={self.relevance_score:.3f})"


class CitationValidator:
    """
    Validador de citas canónicas con reglas estrictas
    
    TODO: Implementar validación completa de formato y contenido
    """
    
    # Patrón regex para formato canónico
    CITATION_PATTERN = re.compile(r'^([^#]+)#L(\d+)-L(\d+)$')
    
    # Esquemas URI válidos
    VALID_URI_SCHEMES = {
        'http', 'https', 'file', 'doc', 'docs', 'ref', 'book', 'article', 'wiki'
    }
    
    @classmethod
    def validate_format(cls, citation_ref: str) -> Tuple[bool, str]:
        """
        TODO: Validar formato de cita canónica
        
        Validaciones:
        1. Formato general uri#Lx-Ly
        2. Números de línea válidos
        3. Rango de líneas lógico
        4. URI bien formada
        """
        if not citation_ref or not isinstance(citation_ref, str):
            return False, "La cita debe ser una cadena no vacía"
        
        # Validación 1: Formato general
        match = cls.CITATION_PATTERN.match(citation_ref.strip())
        if not match:
            return False, "Formato inválido. Debe ser: uri#Lx-Ly (ej: doc://file.md#L10-L15)"
        
        uri, start_line_str, end_line_str = match.groups()
        
        # Validación 2: Números de línea válidos
        try:
            start_line = int(start_line_str)
            end_line = int(end_line_str)
        except ValueError:
            return False, "Los números de línea deben ser enteros válidos"
        
        # Validación 3: Rango de líneas lógico
        if start_line < 1:
            return False, "Los números de línea deben ser >= 1"
        
        if start_line > end_line:
            return False, f"Línea inicial ({start_line}) no puede ser mayor que línea final ({end_line})"
        
        # Validación 4: URI básica
        uri_valid, uri_error = cls._validate_uri(uri)
        if not uri_valid:
            return False, f"URI inválida: {uri_error}"
        
        return True, "Formato de cita válido"
    
    @classmethod
    def _validate_uri(cls, uri: str) -> Tuple[bool, str]:
        """Validar formato básico de URI"""
        if not uri:
            return False, "URI no puede estar vacía"
        
        # Verificar esquema básico
        if '://' in uri:
            scheme = uri.split('://')[0].lower()
            if scheme not in cls.VALID_URI_SCHEMES:
                return False, f"Esquema URI '{scheme}' no es válido. Válidos: {', '.join(cls.VALID_URI_SCHEMES)}"
        elif not uri.startswith(('/', './', '../')):
            # Permitir paths relativos y absolutos sin esquema
            return False, "URI debe tener esquema válido o ser path relativo/absoluto"
        
        return True, "URI válida"
    
    @classmethod
    def parse_citation(cls, citation_ref: str) -> Optional[Dict[str, Union[str, int]]]:
        """
        TODO: Parsear cita canónica en componentes
        
        Returns:
            Dict con 'uri', 'start_line', 'end_line' o None si inválida
        """
        is_valid, error = cls.validate_format(citation_ref)
        if not is_valid:
            return None
        
        match = cls.CITATION_PATTERN.match(citation_ref.strip())
        if match:
            uri, start_line_str, end_line_str = match.groups()
            return {
                'uri': uri,
                'start_line': int(start_line_str),
                'end_line': int(end_line_str)
            }
        
        return None
    
    @classmethod
    def validate_citation_object(cls, citation: Citation) -> Tuple[bool, List[str]]:
        """Validar objeto Citation completo"""
        errors = []
        
        # Validar formato canónico
        canonical_ref = citation.to_canonical()
        is_valid, format_error = cls.validate_format(canonical_ref)
        if not is_valid:
            errors.append(f"Formato canónico inválido: {format_error}")
        
        # Validar contenido no vacío
        if not citation.content or not citation.content.strip():
            errors.append("El contenido de la cita no puede estar vacío")
        
        # Validar score de relevancia
        if not (0 <= citation.relevance_score <= 1):
            errors.append(f"Score de relevancia debe estar entre 0 y 1, recibido: {citation.relevance_score}")
        
        # Validar consistencia de líneas con contenido
        content_lines = citation.content.split('\n')
        expected_lines = citation.get_line_count()
        if len(content_lines) != expected_lines:
            errors.append(f"Inconsistencia: contenido tiene {len(content_lines)} líneas, pero rango indica {expected_lines}")
        
        return len(errors) == 0, errors


class CitationGenerator:
    """
    Generador de citas canónicas con contexto y metadatos
    
    TODO: Implementar generación inteligente de citas
    """
    
    def __init__(self, 
                 line_context: int = 2,
                 max_content_length: int = 1000,
                 min_relevance_score: float = 0.1):
        self.line_context = line_context
        self.max_content_length = max_content_length
        self.min_relevance_score = min_relevance_score
        
        # Estadísticas
        self.citations_generated = 0
        self.generation_stats = {
            'total_generated': 0,
            'with_context_added': 0,
            'content_truncated': 0,
            'below_min_score': 0
        }
    
    def generate_citation(self,
                         document_uri: str,
                         full_content: str,
                         match_start_char: int,
                         match_end_char: int,
                         relevance_score: float,
                         metadata: Optional[Dict[str, Any]] = None,
                         include_context: bool = True) -> Optional[Citation]:
        """
        TODO: Generar cita con contexto de líneas
        
        Args:
            document_uri: URI del documento fuente
            full_content: Contenido completo del documento
            match_start_char: Posición inicial del match en caracteres
            match_end_char: Posición final del match en caracteres
            relevance_score: Score de relevancia (0-1)
            metadata: Metadatos adicionales
            include_context: Si incluir líneas de contexto
            
        Returns:
            Citation object o None si no cumple criterios mínimos
        """
        self.generation_stats['total_generated'] += 1
        
        # Filtrar por score mínimo
        if relevance_score < self.min_relevance_score:
            self.generation_stats['below_min_score'] += 1
            return None
        
        try:
            lines = full_content.split('\n')
            total_lines = len(lines)
            
            # Encontrar líneas que contienen el match
            start_line, end_line = self._find_match_lines(full_content, match_start_char, match_end_char)
            
            if start_line is None or end_line is None:
                return None
            
            # Agregar contexto si está habilitado
            if include_context and self.line_context > 0:
                context_start = max(1, start_line - self.line_context)
                context_end = min(total_lines, end_line + self.line_context)
                
                if context_start < start_line or context_end > end_line:
                    self.generation_stats['with_context_added'] += 1
            else:
                context_start = start_line
                context_end = end_line
            
            # Extraer contenido citado
            cited_lines = lines[context_start-1:context_end]  # Convert to 0-indexed
            cited_content = '\n'.join(cited_lines)
            
            # Truncar contenido si es muy largo
            if len(cited_content) > self.max_content_length:
                cited_content = cited_content[:self.max_content_length] + "..."
                self.generation_stats['content_truncated'] += 1
            
            # Crear metadatos enriquecidos
            enriched_metadata = {
                'original_match_chars': (match_start_char, match_end_char),
                'exact_match_lines': (start_line, end_line),
                'context_added': include_context and self.line_context > 0,
                'content_truncated': len('\n'.join(cited_lines)) > self.max_content_length,
                'total_document_lines': total_lines,
                'generation_timestamp': datetime.now().isoformat()
            }
            
            if metadata:
                enriched_metadata.update(metadata)
            
            citation = Citation(
                uri=document_uri,
                start_line=context_start,
                end_line=context_end,
                content=cited_content,
                relevance_score=relevance_score,
                metadata=enriched_metadata
            )
            
            self.citations_generated += 1
            return citation
            
        except Exception as e:
            print(f"Error generando cita: {e}")
            return None
    
    def _find_match_lines(self, content: str, start_char: int, end_char: int) -> Tuple[Optional[int], Optional[int]]:
        """
        TODO: Encontrar números de línea basados en posiciones de caracteres
        """
        lines = content.split('\n')
        current_pos = 0
        start_line = None
        end_line = None
        
        for i, line in enumerate(lines):
            line_start = current_pos
            line_end = current_pos + len(line)
            
            # Encontrar línea de inicio
            if start_line is None and line_start <= start_char <= line_end:
                start_line = i + 1  # 1-indexed
            
            # Encontrar línea de fin
            if line_start <= end_char <= line_end:
                end_line = i + 1  # 1-indexed
            
            # Mover al siguiente carácter (incluyendo \n)
            current_pos = line_end + 1
            
            # Si ya encontramos ambas líneas, terminar
            if start_line is not None and end_line is not None:
                break
        
        return start_line, end_line
    
    def generate_citations_from_matches(self,
                                       document_uri: str,
                                       full_content: str,
                                       matches: List[Tuple[int, int, float]],
                                       metadata: Optional[Dict[str, Any]] = None) -> List[Citation]:
        """
        TODO: Generar múltiples citas desde lista de matches
        
        Args:
            document_uri: URI del documento
            full_content: Contenido completo
            matches: Lista de (start_char, end_char, score)
            metadata: Metadatos base para todas las citas
            
        Returns:
            Lista de citations válidas
        """
        citations = []
        
        for i, (start_char, end_char, score) in enumerate(matches):
            # Metadatos específicos del match
            match_metadata = {'match_index': i, 'total_matches': len(matches)}
            if metadata:
                match_metadata.update(metadata)
            
            citation = self.generate_citation(
                document_uri=document_uri,
                full_content=full_content,
                match_start_char=start_char,
                match_end_char=end_char,
                relevance_score=score,
                metadata=match_metadata
            )
            
            if citation:
                citations.append(citation)
        
        return citations
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de generación"""
        total = self.generation_stats['total_generated']
        return {
            **self.generation_stats,
            'success_rate': (self.citations_generated / total) if total > 0 else 0,
            'context_addition_rate': (self.generation_stats['with_context_added'] / total) if total > 0 else 0,
            'truncation_rate': (self.generation_stats['content_truncated'] / total) if total > 0 else 0
        }


class CitationExtractor:
    """
    Extractor de snippets desde citas canónicas
    
    TODO: Implementar extracción segura de contenido
    """
    
    def __init__(self):
        self.extraction_cache = {}
        self.extractions_performed = 0
    
    def extract_snippet(self, 
                       citation: Citation, 
                       source_content: str,
                       validate_consistency: bool = True) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        TODO: Extraer snippet desde cita y contenido fuente
        
        Args:
            citation: Objeto Citation
            source_content: Contenido fuente del documento
            validate_consistency: Si validar consistencia con contenido original
            
        Returns:
            (éxito, snippet_extraído, mensaje_error)
        """
        try:
            lines = source_content.split('\n')
            total_lines = len(lines)
            
            # Validar rango de líneas
            if citation.start_line < 1 or citation.end_line > total_lines:
                return False, None, f"Rango de líneas {citation.start_line}-{citation.end_line} fuera de rango (1-{total_lines})"
            
            if citation.start_line > citation.end_line:
                return False, None, f"Línea inicial ({citation.start_line}) mayor que línea final ({citation.end_line})"
            
            # Extraer snippet
            snippet_lines = lines[citation.start_line-1:citation.end_line]  # Convert to 0-indexed
            extracted_snippet = '\n'.join(snippet_lines)
            
            # Validar consistencia si está habilitado
            if validate_consistency:
                consistency_valid, consistency_error = self._validate_consistency(citation, extracted_snippet)
                if not consistency_valid:
                    return False, extracted_snippet, f"Inconsistencia detectada: {consistency_error}"
            
            self.extractions_performed += 1
            return True, extracted_snippet, None
            
        except Exception as e:
            return False, None, f"Error extrayendo snippet: {str(e)}"
    
    def _validate_consistency(self, citation: Citation, extracted_snippet: str) -> Tuple[bool, Optional[str]]:
        """Validar que el snippet extraído sea consistente con la cita original"""
        
        # Comparar número de líneas
        citation_lines = citation.content.count('\n') + 1
        extracted_lines = extracted_snippet.count('\n') + 1
        
        if citation_lines != extracted_lines:
            return False, f"Número de líneas difiere: cita={citation_lines}, extraído={extracted_lines}"
        
        # Comparar contenido (ignorando espacios en blanco al inicio/final)
        citation_normalized = citation.content.strip()
        extracted_normalized = extracted_snippet.strip()
        
        # Si el contenido de la cita fue truncado, solo comparar el inicio
        if citation.content.endswith('...'):
            citation_prefix = citation_normalized[:-3].strip()
            if not extracted_normalized.startswith(citation_prefix):
                return False, "Contenido no coincide con snippet extraído"
        else:
            if citation_normalized != extracted_normalized:
                return False, "Contenido no coincide exactamente"
        
        return True, None
    
    def batch_extract(self, 
                     citations: List[Citation], 
                     content_provider: callable) -> List[Tuple[Citation, bool, Optional[str]]]:
        """
        TODO: Extraer snippets en lote
        
        Args:
            citations: Lista de citas
            content_provider: Función que retorna contenido dado un URI
            
        Returns:
            Lista de (citation, éxito, snippet_o_error)
        """
        results = []
        
        for citation in citations:
            try:
                # Obtener contenido usando el proveedor
                source_content = content_provider(citation.uri)
                
                if source_content is None:
                    results.append((citation, False, f"No se pudo obtener contenido para {citation.uri}"))
                    continue
                
                # Extraer snippet
                success, snippet, error = self.extract_snippet(citation, source_content)
                
                if success:
                    results.append((citation, True, snippet))
                else:
                    results.append((citation, False, error))
                    
            except Exception as e:
                results.append((citation, False, f"Error procesando {citation.uri}: {str(e)}"))
        
        return results


class CitationManager:
    """
    Gestor completo del sistema de citas canónicas
    
    TODO: Implementar gestión completa de citas
    """
    
    def __init__(self):
        self.validator = CitationValidator()
        self.generator = CitationGenerator()
        self.extractor = CitationExtractor()
        
        # Almacén de citas
        self.citations: Dict[str, Citation] = {}  # citation_id -> Citation
        self.uri_index: Dict[str, List[str]] = {}  # uri -> [citation_ids]
        
        # Estadísticas
        self.stats = {
            'total_citations': 0,
            'valid_citations': 0,
            'invalid_citations': 0,
            'unique_uris': 0
        }
    
    def add_citation(self, citation: Citation, citation_id: Optional[str] = None) -> Tuple[bool, str]:
        """
        TODO: Agregar cita al sistema con validación
        
        Args:
            citation: Objeto Citation
            citation_id: ID personalizado (opcional)
            
        Returns:
            (éxito, mensaje_o_id)
        """
        # Validar cita
        is_valid, errors = self.validator.validate_citation_object(citation)
        
        if not is_valid:
            self.stats['invalid_citations'] += 1
            return False, f"Cita inválida: {'; '.join(errors)}"
        
        # Generar ID si no se proporciona
        if citation_id is None:
            citation_id = self._generate_citation_id(citation)
        
        # Verificar duplicados
        if citation_id in self.citations:
            return False, f"Ya existe una cita con ID: {citation_id}"
        
        # Agregar al almacén
        self.citations[citation_id] = citation
        
        # Actualizar índice por URI
        if citation.uri not in self.uri_index:
            self.uri_index[citation.uri] = []
            self.stats['unique_uris'] += 1
        
        self.uri_index[citation.uri].append(citation_id)
        
        # Actualizar estadísticas
        self.stats['total_citations'] += 1
        self.stats['valid_citations'] += 1
        
        return True, citation_id
    
    def _generate_citation_id(self, citation: Citation) -> str:
        """Generar ID único para una cita"""
        # Usar hash del contenido canónico + timestamp
        canonical = citation.to_canonical()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"cite_{abs(hash(canonical))}_{timestamp}"
    
    def get_citation(self, citation_id: str) -> Optional[Citation]:
        """Obtener cita por ID"""
        return self.citations.get(citation_id)
    
    def get_citations_by_uri(self, uri: str) -> List[Citation]:
        """Obtener todas las citas de un URI específico"""
        citation_ids = self.uri_index.get(uri, [])
        return [self.citations[cid] for cid in citation_ids]
    
    def search_citations(self, 
                        query: str = None,
                        uri_pattern: str = None,
                        min_score: float = None,
                        max_results: int = None) -> List[Citation]:
        """
        TODO: Buscar citas con filtros múltiples
        """
        results = []
        
        for citation in self.citations.values():
            # Filtro por consulta en contenido
            if query and query.lower() not in citation.content.lower():
                continue
            
            # Filtro por patrón de URI
            if uri_pattern and not re.search(uri_pattern, citation.uri):
                continue
            
            # Filtro por score mínimo
            if min_score is not None and citation.relevance_score < min_score:
                continue
            
            results.append(citation)
        
        # Ordenar por relevancia
        results.sort(key=lambda c: c.relevance_score, reverse=True)
        
        # Limitar resultados
        if max_results:
            results = results[:max_results]
        
        return results
    
    def export_citations(self, filepath: str, format: str = 'json') -> bool:
        """Exportar citas a archivo"""
        try:
            if format.lower() == 'json':
                citations_data = {
                    'metadata': {
                        'exported_at': datetime.now().isoformat(),
                        'total_citations': len(self.citations),
                        'stats': self.stats
                    },
                    'citations': {cid: citation.to_dict() for cid, citation in self.citations.items()}
                }
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(citations_data, f, indent=2, ensure_ascii=False)
                
                return True
            else:
                return False
                
        except Exception as e:
            print(f"Error exportando citas: {e}")
            return False
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas completas del manager"""
        return {
            **self.stats,
            'generator_stats': self.generator.get_generation_stats(),
            'extractions_performed': self.extractor.extractions_performed,
            'avg_relevance_score': sum(c.relevance_score for c in self.citations.values()) / len(self.citations) if self.citations else 0
        }


def create_sample_citations() -> List[Citation]:
    """Crear citas de muestra para pruebas"""
    sample_citations = [
        Citation(
            uri="docs://python-tutorial.md",
            start_line=10,
            end_line=15,
            content="""Python es un lenguaje de programación interpretado de alto nivel.
Es conocido por su sintaxis clara y legible, lo que lo hace ideal para principiantes.
Python soporta múltiples paradigmas de programación incluyendo programación orientada a objetos,
programación funcional y programación procedural.

Las características principales de Python incluyen:""",
            relevance_score=0.95,
            metadata={"topic": "python-intro", "difficulty": "beginner"}
        ),
        Citation(
            uri="docs://machine-learning-intro.md",
            start_line=5,
            end_line=8,
            content="""Machine Learning (ML) es una rama de la inteligencia artificial que permite a los sistemas
aprender y mejorar automáticamente a partir de la experiencia sin ser programados explícitamente.

Los principales tipos de machine learning son:
1. Aprendizaje supervisado: usa datos etiquetados para entrenar modelos""",
            relevance_score=0.87,
            metadata={"topic": "machine-learning", "difficulty": "intermediate"}
        ),
        Citation(
            uri="https://api-docs.example.com/auth",
            start_line=25,
            end_line=30,
            content="""Para autorización usa headers Authorization: Bearer <token>
El token debe ser válido y no expirado.
Los tokens expiran después de 24 horas.
Para renovar un token, usa el endpoint /auth/refresh.
Todos los endpoints protegidos requieren autenticación.""",
            relevance_score=0.78,
            metadata={"topic": "api-auth", "section": "authorization"}
        )
    ]
    
    return sample_citations


def test_citation_system():
    """Función de prueba para el sistema completo de citas"""
    print("=== PRUEBAS DEL SISTEMA DE CITAS ===\n")
    
    # 1. Pruebas de validación
    print("🔍 PRUEBAS DE VALIDACIÓN:")
    
    test_references = [
        "docs://file.md#L10-L15",  # Válida
        "https://example.com/doc#L1-L5",  # Válida
        "file.md#L10-L15",  # Inválida - sin esquema
        "docs://file.md#L15-L10",  # Inválida - rango invertido
        "docs://file.md#L0-L5",  # Inválida - línea 0
        "not-a-citation",  # Inválida - formato incorrecto
    ]
    
    for ref in test_references:
        is_valid, message = CitationValidator.validate_format(ref)
        status = "✅" if is_valid else "❌"
        print(f"  {status} {ref}: {message}")
    
    # 2. Pruebas de generación
    print(f"\n🏗️ PRUEBAS DE GENERACIÓN:")
    
    generator = CitationGenerator(line_context=1)
    
    sample_content = """Línea 1: Introducción al tema
Línea 2: Conceptos básicos importantes
Línea 3: Esta es la línea objetivo principal
Línea 4: Información adicional relevante
Línea 5: Conclusiones del tema"""
    
    # Simular match en línea 3
    match_start = sample_content.find("Esta es la línea objetivo")
    match_end = match_start + len("Esta es la línea objetivo principal")
    
    citation = generator.generate_citation(
        document_uri="docs://example.md",
        full_content=sample_content,
        match_start_char=match_start,
        match_end_char=match_end,
        relevance_score=0.9,
        metadata={"test": True}
    )
    
    if citation:
        print(f"  ✅ Cita generada: {citation.to_canonical()}")
        print(f"  📄 Contenido: {citation.content}")
        print(f"  📊 Score: {citation.relevance_score}")
    else:
        print(f"  ❌ No se pudo generar la cita")
    
    # 3. Pruebas de extracción
    print(f"\n📤 PRUEBAS DE EXTRACCIÓN:")
    
    extractor = CitationExtractor()
    
    if citation:
        success, extracted, error = extractor.extract_snippet(citation, sample_content)
        if success:
            print(f"  ✅ Snippet extraído exitosamente")
            print(f"  📄 Contenido extraído: {extracted}")
        else:
            print(f"  ❌ Error en extracción: {error}")
    
    # 4. Pruebas del manager
    print(f"\n🗂️ PRUEBAS DEL MANAGER:")
    
    manager = CitationManager()
    
    # Agregar citas de muestra
    sample_citations = create_sample_citations()
    
    for i, cite in enumerate(sample_citations):
        success, result = manager.add_citation(cite)
        if success:
            print(f"  ✅ Cita {i+1} agregada: ID = {result}")
        else:
            print(f"  ❌ Error agregando cita {i+1}: {result}")
    
    # Buscar citas
    python_citations = manager.search_citations(query="python", min_score=0.8)
    print(f"  🔍 Citas sobre Python (score >= 0.8): {len(python_citations)}")
    
    # Estadísticas
    stats = manager.get_manager_stats()
    print(f"  📊 Estadísticas del manager:")
    print(f"    - Total citas: {stats['total_citations']}")
    print(f"    - Citas válidas: {stats['valid_citations']}")
    print(f"    - URIs únicos: {stats['unique_uris']}")
    print(f"    - Score promedio: {stats['avg_relevance_score']:.3f}")


def interactive_citation_demo():
    """Demostración interactiva del sistema de citas"""
    print("=== DEMO INTERACTIVO - SISTEMA DE CITAS ===\n")
    print("Explora el sistema de citas canónicas con ejemplos reales.")
    
    manager = CitationManager()
    
    # Cargar citas de muestra
    sample_citations = create_sample_citations()
    for cite in sample_citations:
        manager.add_citation(cite)
    
    print(f"\n📚 Citas de muestra cargadas: {len(sample_citations)}")
    
    while True:
        print("\nOpciones:")
        print("1. Validar formato de cita")
        print("2. Buscar citas por contenido")
        print("3. Ver citas por URI")
        print("4. Generar nueva cita")
        print("5. Ver estadísticas")
        print("6. Exportar citas")
        print("7. Salir")
        
        choice = input("\nElige una opción (1-7): ").strip()
        
        if choice == "1":
            citation_ref = input("\n📝 Ingresa cita a validar (ej: docs://file.md#L10-L15): ").strip()
            if citation_ref:
                is_valid, message = CitationValidator.validate_format(citation_ref)
                status = "✅ Válida" if is_valid else "❌ Inválida"
                print(f"\n{status}: {message}")
                
                if is_valid:
                    parsed = CitationValidator.parse_citation(citation_ref)
                    if parsed:
                        print(f"📊 Componentes parseados:")
                        print(f"  URI: {parsed['uri']}")
                        print(f"  Líneas: {parsed['start_line']}-{parsed['end_line']}")
        
        elif choice == "2":
            query = input("\n🔍 Buscar en contenido: ").strip()
            if query:
                results = manager.search_citations(query=query)
                
                print(f"\n📋 Encontradas {len(results)} citas:")
                for i, citation in enumerate(results, 1):
                    print(f"\n{i}. {citation.to_canonical()}")
                    print(f"   Score: {citation.relevance_score:.3f}")
                    print(f"   Contenido: {citation.content[:100]}...")
        
        elif choice == "3":
            uri = input("\n🔗 URI a buscar: ").strip()
            if uri:
                citations = manager.get_citations_by_uri(uri)
                
                print(f"\n📄 Encontradas {len(citations)} citas para {uri}:")
                for i, citation in enumerate(citations, 1):
                    print(f"\n{i}. Líneas {citation.start_line}-{citation.end_line}")
                    print(f"   Score: {citation.relevance_score:.3f}")
                    print(f"   Contenido: {citation.content[:150]}...")
        
        elif choice == "4":
            print(f"\n🏗️ GENERADOR DE CITAS:")
            uri = input("URI del documento: ").strip()
            content = input("Contenido del documento: ").strip()
            search_term = input("Término a buscar: ").strip()
            
            if uri and content and search_term:
                # Buscar término en contenido
                start_pos = content.find(search_term)
                if start_pos >= 0:
                    end_pos = start_pos + len(search_term)
                    
                    generator = CitationGenerator()
                    citation = generator.generate_citation(
                        document_uri=uri,
                        full_content=content,
                        match_start_char=start_pos,
                        match_end_char=end_pos,
                        relevance_score=0.8,
                        metadata={"generated_interactively": True}
                    )
                    
                    if citation:
                        success, cite_id = manager.add_citation(citation)
                        if success:
                            print(f"\n✅ Cita generada y agregada:")
                            print(f"   ID: {cite_id}")
                            print(f"   Referencia: {citation.to_canonical()}")
                            print(f"   Contenido: {citation.content}")
                        else:
                            print(f"\n❌ Error agregando cita: {cite_id}")
                    else:
                        print(f"\n❌ No se pudo generar la cita")
                else:
                    print(f"\n❌ Término '{search_term}' no encontrado en el contenido")
        
        elif choice == "5":
            stats = manager.get_manager_stats()
            print(f"\n📊 ESTADÍSTICAS DEL SISTEMA:")
            print(f"Total de citas: {stats['total_citations']}")
            print(f"Citas válidas: {stats['valid_citations']}")
            print(f"Citas inválidas: {stats['invalid_citations']}")
            print(f"URIs únicos: {stats['unique_uris']}")
            print(f"Score promedio: {stats['avg_relevance_score']:.3f}")
            
            gen_stats = stats['generator_stats']
            print(f"\nEstadísticas de generación:")
            print(f"Tasa de éxito: {gen_stats['success_rate']:.1%}")
            print(f"Con contexto agregado: {gen_stats['context_addition_rate']:.1%}")
            print(f"Contenido truncado: {gen_stats['truncation_rate']:.1%}")
        
        elif choice == "6":
            filename = input("\n💾 Nombre del archivo (ej: citas.json): ").strip()
            if filename:
                success = manager.export_citations(filename)
                if success:
                    print(f"✅ Citas exportadas a {filename}")
                else:
                    print(f"❌ Error exportando citas")
        
        elif choice == "7":
            print("\n👋 ¡Hasta luego!")
            break
        
        print("\n" + "="*50)


if __name__ == "__main__":
    # Ejecutar pruebas automatizadas
    test_citation_system()
    
    # Demostración interactiva
    print("\n¿Quieres probar el demo interactivo? (y/n): ", end="")
    if input().lower().startswith('y'):
        interactive_citation_demo()
    
    print("\n=== EJERCICIOS ADICIONALES ===")
    print("1. Implementa validación de checksums para detectar contenido modificado")
    print("2. Agrega soporte para citas anidadas (citas que referencian otras citas)")
    print("3. Implementa versionado de citas para documentos que cambian")
    print("4. Crea sistema de resolución de citas rotas (dead citations)")
    print("5. Agrega métricas de calidad para citas (precisión, cobertura, etc.)")
