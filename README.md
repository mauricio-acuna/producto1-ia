# 🤖 Portal 1: Fundamentos de IA para Desarrolladores

Este repositorio contiene el desarrollo del **Portal 1**, un curso de capacitación en IA diseñado específicamente para desarrolladores sin experiencia previa en agentes o RAG.

## 📋 **Tabla de Contenidos**
- [🎯 Objetivo](#-objetivo)
- [📚 Estructura del Curso](#-estructura-del-curso)
- [🏗️ Arquitectura del Proyecto](#️-arquitectura-del-proyecto)
- [🚀 Inicio Rápido](#-inicio-rápido)
- [🧪 Laboratorios Disponibles](#-laboratorios-disponibles)
- [📋 Estado del Desarrollo](#-estado-del-desarrollo)
- [📊 Métricas del Proyecto](#-métricas-del-proyecto)
- [🛠️ Tecnologías Utilizadas](#️-tecnologías-utilizadas)
- [📚 Documentación Adicional](#-documentación-adicional)
- [🤝 Contribución](#-contribución)

## 📋 **Navegación Rápida**

| 📚 **Documentación** | 🔧 **Desarrollo** | 🎯 **Laboratorios** |
|---------------------|-------------------|---------------------|
| [📖 PRD Completo](./PRD.md) | [🚀 Guía de Despliegue](docs/deployment/guia-despliegue-produccion.md) | [🧪 Labs Módulo A](labs/module-a/) |
| [⚡ Benchmarks](docs/performance/marco-benchmarks-rendimiento.md) | [🔌 API Docs](docs/api/README.md) | [🏗️ Arquitectura](docs/adr/README-ES.md) |
| [🔒 Seguridad](docs/security/marco-seguridad-etica.md) | [📊 Análisis Industrial](analysis/documentation_standards_assessment.md) | [📁 Código Fuente](src/) |

## 🎯 Objetivo

Proporcionar una ruta clara y modular para que desarrolladores junior/mid-level puedan:
- Comprender los fundamentos de agentes de IA
- Crear un mini-agente funcional
- Implementar RAG básico con citas
- Aplicar seguridad mínima y métricas de calidad
- Generar un repositorio inicial para entrevistas técnicas

## 📚 Estructura del Curso

| Módulo | Estado | Descripción | Enlaces |
|--------|---------|-------------|---------|
| **A** | ✅ Completo | Conceptos esenciales | [📁 Contenido](content/modules/module-a-conceptos/) \| [🧪 Labs](labs/module-a/) |
| **B** | 🚧 En desarrollo | Primer mini-agente | [📁 Contenido](content/modules/module-b-mini-agente/) |
| **C** | ⏳ Pendiente | RAG básico con citas | [📁 Contenido](content/modules/module-c-rag-basico/) |
| **D** | ⏳ Pendiente | Métricas de calidad | [📁 Contenido](content/modules/module-d-metricas/) |
| **E** | ⏳ Pendiente | Capstone final | [📁 Contenido](content/modules/module-e-capstone/) |

## 🏗️ Arquitectura del Proyecto

```
producto1/
├── � PRD.md                          # Documento de requerimientos
├── 📄 project.config.json             # Configuración del curso
├── 🐳 Dockerfile & docker-compose.yml # Containerización
├── 📦 requirements.txt                # Dependencias Python
├── 
├── 📁 src/                            # Código fuente de la aplicación
│   ├── 🌐 main.py                     # FastAPI app principal
│   ├── 📁 api/routes/                 # Endpoints API
│   ├── 📁 models/                     # Modelos de datos
│   ├── 📁 services/                   # Lógica de negocio
│   └── 📁 utils/                      # Utilidades
├── 
├── 📁 content/modules/                # Contenido educativo
│   ├── 📁 module-a-conceptos/         # ✅ Módulo A completo
│   ├── 📁 module-b-mini-agente/       # 🚧 En desarrollo
│   ├── 📁 module-c-rag-basico/        # ⏳ Pendiente
│   ├── 📁 module-d-metricas/          # ⏳ Pendiente
│   └── 📁 module-e-capstone/          # ⏳ Pendiente
├── 
├── 📁 labs/                           # Laboratorios prácticos
│   ├── 📁 module-a/                   # ✅ 3 labs implementados
│   │   ├── 🐍 chat_vs_agent.py       # Comparación sistemas
│   │   ├── 🐍 structured_output.py   # JSON estructurado
│   │   └── 🐍 security_validator.py  # Validación seguridad
│   └── 📁 [otros módulos]/
├── 
├── 📁 templates/                      # Plantillas y recursos
│   ├── 📄 safety.min.yaml            # Configuración de seguridad
│   └── 📄 plan-spec.json             # Schema de planes de agente
├── 
├── 📁 datasets/                       # Datos para prácticas
│   └── 📄 knowledge-base-mini.json   # Dataset RAG (10 documentos)
├── 
└── 📁 public/                        # Archivos estáticos web
```

## 🚀 Inicio Rápido

### 🐳 Opción 1: Docker (Recomendado)
```bash
# Clonar repositorio
git clone https://github.com/mauricio-acuna/producto1-ia.git
cd producto1-ia

# Levantar con Docker Compose
docker-compose up --build

# Acceder a:
# - Portal web: http://localhost:8000
# - API docs: http://localhost:8000/api/docs
# - Jupyter: http://localhost:8888 (opcional)
```

### 🐍 Opción 2: Local
```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar aplicación
cd src
python main.py
```

## 🧪 Laboratorios Disponibles

### 📘 Módulo A: Conceptos Esenciales
| Lab | Descripción | Archivo | Estado |
|-----|-------------|---------|---------|
| 1 | Chat vs Agente | [chat_vs_agent.py](labs/module-a/chat_vs_agent.py) | ✅ |
| 2 | JSON estructurado | [structured_output.py](labs/module-a/structured_output.py) | ✅ |
| 3 | Validación seguridad | [security_validator.py](labs/module-a/security_validator.py) | ✅ |

```bash
# Ejecutar laboratorios
cd labs/module-a
python chat_vs_agent.py        # Lab 1
python structured_output.py    # Lab 2  
python security_validator.py   # Lab 3
```

## 📋 Estado del Desarrollo

### ✅ Completado
- [x] **PRD inicial** - Documento completo de requerimientos
- [x] **Estructura del proyecto** - Arquitectura modular establecida
- [x] **Módulo A completo** - Conceptos esenciales con 3 laboratorios
- [x] **API REST** - Endpoints para módulos, progreso y autenticación
- [x] **Base de datos** - Modelo SQLite para usuarios y progreso
- [x] **Containerización** - Docker y docker-compose configurados
- [x] **Landing page** - Portal web con información del curso

### 🚧 En Desarrollo
- [ ] **Módulo B** - Primer mini-agente (Planner→Executor→Critic)
- [ ] **Sistema de evaluación** - Quizzes y validación automática
- [ ] **Dashboard de progreso** - Interfaz para tracking de estudiantes

### ⏳ Pendiente
- [ ] **Módulo C** - RAG básico con citas canónicas
- [ ] **Módulo D** - Métricas de calidad, coste y latencia  
- [ ] **Módulo E** - Capstone final
- [ ] **Frontend completo** - Interfaz web para estudiantes
- [ ] **Sistema de certificados** - Generación automática
- [ ] **Analytics avanzado** - Métricas detalladas del curso

## 🧪 Laboratorios Disponibles

### Módulo A: Conceptos Esenciales
```bash
# Ejecutar laboratorios
cd labs/module-a

# 1. Comparación Chat vs Agente
python chat_vs_agent.py

# 2. Generación de JSON estructurado
python structured_output.py

# 3. Validación de seguridad
python security_validator.py
```

## 📊 Métricas del Proyecto

- **Líneas de código**: ~2,500+ (Python, Markdown, configs)
- **Módulos educativos**: 1/5 completos
- **Laboratorios prácticos**: 3 implementados
- **Endpoints API**: 15+ funcionales
- **Documentos de conocimiento**: 10 (dataset RAG)

## 🛠️ Tecnologías Utilizadas

### Backend
- **FastAPI** - Framework web moderno y rápido
- **SQLite** - Base de datos ligera para desarrollo
- **Pydantic** - Validación de datos y serialización
- **JWT** - Autenticación con tokens

### IA y ML
- **OpenAI API** - Integración con GPT (preparado)
- **scikit-learn** - Algoritmos ML para RAG
- **NLTK** - Procesamiento de lenguaje natural
- **rank-bm25** - Algoritmo de ranking para búsqueda

### DevOps
- **Docker** - Containerización
- **pytest** - Testing automatizado
- **Black/Flake8** - Formateo y linting de código

## 📚 Documentación Adicional

### 📋 **Documentos Principales**
- [📄 PRD.md](./PRD.md) - Documento de Requerimientos del Producto
- [⚙️ project.config.json](./project.config.json) - Configuración del curso
- [📁 Plantillas](./templates/) - Recursos descargables para estudiantes

### 🏗️ **Arquitectura y Desarrollo**
- [🏛️ ADRs (Decisiones Arquitectónicas)](docs/adr/README-ES.md)
- [🔌 Documentación API Completa](docs/api/README.md)
- [🚀 Guías de Despliegue](docs/deployment/guia-despliegue-produccion.md)
- [⚡ Benchmarks de Rendimiento](docs/performance/marco-benchmarks-rendimiento.md)
- [🔒 Marco de Seguridad y Ética](docs/security/marco-seguridad-etica.md)

### 📊 **Análisis y Estándares**
- [🏭 Análisis de Estándares Industriales](analysis/documentation_standards_assessment.md)
- [📋 README Industrial](README_INDUSTRY_STANDARD.md)

## 🤝 Contribución

Este proyecto está en desarrollo activo. Los commits se realizan frecuentemente para registrar el progreso.

### Flujo de Desarrollo
1. **Desarrollo modular** - Un módulo completo por iteración
2. **Testing continuo** - Verificación de funcionalidades
3. **Commits frecuentes** - Registro detallado de progreso
4. **Documentación** - Actualización continua

---

**🎯 Objetivo próximo**: Completar Módulo B (Mini-agente) con patrón Planner-Executor-Critic

**Autor**: Mauricio Acuña  
**Repositorio**: https://github.com/mauricio-acuna/producto1-ia.git  
**Última actualización**: Agosto 2025
