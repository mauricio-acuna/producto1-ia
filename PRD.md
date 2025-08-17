# AI Agents: From Zero to Production - CURSO COMPLETADO ✅
## Curso Completo de Desarrollo de Agentes de IA

**Estado:** ✅ Completado - 100% terminado
**Última actualización:** Enero 2024

### 📋 Progreso del Curso

- ✅ **Módulo A**: Fundamentos de Agentes de IA (100%)
- ✅ **Módulo B**: Arquitectura PEC (Planner-Executor-Critic) (100%)
- ✅ **Módulo C**: RAG con Citas Canónicas (100%)
- ✅ **Módulo D**: Quick Evals y Monitoreo (100%)
- ✅ **Módulo E**: Proyecto Capstone - RepoGPT (100%)

🎯 **¡CURSO FINALIZADO!** Has desarrollado un agente de IA completo y funcional.

---

# 📄 PRD — Portal 1 "Fundamentos de IA para Devs"# 📄 PRD — Portal 1 “Fundamentos de IA para Devs”

## 1. Introducción

### 1.1 Propósito

El **Portal 1** es la puerta de entrada al recorrido de capacitación en IA para desarrolladores. Está diseñado para quienes **no tienen experiencia previa con agentes o RAG** pero poseen base en desarrollo de software.
El objetivo es que, al finalizar, el alumno pueda **comprender los fundamentos**, **crear un mini-agente** y un **RAG básico**, **aplicar seguridad mínima** y **medir calidad/coste/latencia**, dejando preparado un **repo inicial** para entrevistas técnicas.

### 1.2 Alcance

Este portal cubrirá:

* Diferencia entre chat tradicional y agentes.
* Uso de salidas estructuradas (JSON válido).
* Tool calling seguro con reglas mínimas.
* RAG básico con citas canónicas.
* Métricas básicas de calidad/coste/latencia.
* Capstone corto (proyecto final).

No incluye: técnicas avanzadas (LangGraph, Graph-RAG, observabilidad avanzada, CI/CD, costes multi-tenant) que se cubren en portales posteriores.

---

## 2. Público objetivo y usuarios

### 2.1 Perfil primario

* **Ingenieros de software junior o mid-level** con conocimientos básicos de Git, Python/TypeScript, Docker y APIs REST.
* Idioma base: español (pero con opción de internacionalización EN).

### 2.2 Problemas a resolver

* Falta de entendimiento de qué es un agente y cómo se diferencia de un simple chatbot.
* Dificultad para aplicar *good practices* mínimas en proyectos de IA.
* Necesidad de resultados prácticos y medibles (un repo, un capstone).

---

## 3. Objetivos y métricas de éxito

### 3.1 Objetivos de producto

1. Proporcionar una **ruta clara y modular** de aprendizaje.
2. Asegurar que los estudiantes produzcan un **artefacto real** (mini-agente + RAG básico).
3. Permitir que cualquier dev pueda **medir coste, latencia y calidad** con herramientas mínimas.

### 3.2 KPIs / métricas de éxito

* **Tasa de finalización del curso:** ≥ 65%.
* **Entrega de capstone:** ≥ 40% de inscritos.
* **NPS (satisfacción del curso):** ≥ 8/10.
* **Precisión promedio en quick evals de capstone:** ≥ 70%.

---

## 4. Requisitos funcionales

### 4.1 Estructura curricular

* **Módulo A — Conceptos esenciales** (agentes vs chat, JSON estructurado, seguridad mínima).
* **Módulo B — Primer mini-agente** (Planner→Executor→Critic simple).
* **Módulo C — RAG básico con citas** (BM25/TF-IDF, MMR, citas `uri#Lx-Ly`).
* **Módulo D — Calidad/coste/tests mínimos** (quick evals, coste tokens, latencia básica). ✅
* **Módulo E — Capstone corto** (mini-agente + RAG que responde sobre un repo demo).

### 4.2 Funcionalidades del portal

* **Landing page:** pitch del curso, CTA (“Empezar ahora”).
* **Curriculum interactivo:** listado de módulos con progreso del alumno.
* **Lecciones en formato Markdown/HTML:** cada una con objetivos, contenido, ejercicios y quiz.
* **Laboratorios prácticos:** guías paso a paso con datasets incluidos.
* **Descargables:** plantillas (`safety.min.yaml`, `plan-spec.json` básico, cheat-sheets).
* **Capstone final:** rúbrica clara, entregables, checklist de evaluación.
* **Analítica interna:** seguimiento de progreso, descargas, finalización.

### 4.3 Internacionalización (i18n)

* Contenido base en **ES**, opción de traducción a **EN** mediante `front-matter lang:` en las lecciones.

### 4.4 Accesibilidad

* Compatibilidad con lectores de pantalla (`aria-label`, tabindex).
* Contraste mínimo AA (WCAG).
* Subtítulos en materiales audiovisuales.

---

## 5. Requisitos no funcionales

### 5.1 UX

* Diseño limpio, con barra de progreso en cada módulo.
* Botones de “copiar código” en ejemplos.
* FAQs y recursos colapsables para reducir scroll.

### 5.2 SEO

* Palabras clave: “curso agentes IA para devs”, “RAG básico”, “tool calling seguro”.
* Schema.org: `Course`, `HowTo`, `FAQPage`.
* Sitemap.xml + robots.txt.

### 5.3 Performance

* Carga inicial < 2s en desktop, < 3s en móvil.
* Imágenes optimizadas (WebP).
* Lecciones cargadas de forma diferida.

---

## 6. Roadmap de contenidos

| Semana | Entregable principal             |
| ------ | -------------------------------- |
| 1      | Landing + Módulo A               |
| 2      | Módulo B + laboratorio           |
| 3      | Módulo C + laboratorio           |
| 4      | Módulo D ✅                      |
| 5      | Módulo E (capstone) + rúbrica    |
| 6      | SEO, analítica, QA y publicación |

---

## 7. Recursos de aprendizaje incluidos

* **Plantillas YAML/JSON** listas para adaptar.
* **Datasets mini** (30–50 fragmentos) para RAG básico.
* **Cheat-sheets** descargables.
* **Capstone repo demo** para práctica.

---

## 8. Entregables para el alumno

* Repo inicial (mini-agente + RAG + evals mínimos).
* Documentación clara (README).
* Resultados de quick evals.
* Grabación corta del flujo del capstone.

---

## 9. Glosario (extracto)

* **Agente:** sistema IA que planifica y ejecuta acciones con herramientas externas.
* **RAG:** Retrieval-Augmented Generation; recuperar contexto antes de generar respuesta.
* **Precision\@k:** métrica que mide cuántos de los documentos recuperados son relevantes.
* **Eval gate:** prueba automática que bloquea despliegue si se incumplen métricas mínimas.
* **Citas canónicas:** referencias verificables `uri#Lx-Ly` a la fuente original.

---

## 10. Riesgos y mitigaciones

* **Riesgo:** sobrecarga técnica para principiantes.

  * **Mitigación:** usar ejemplos claros, datasets pequeños, checklists.
* **Riesgo:** abandono a mitad de curso.

  * **Mitigación:** quick wins en cada módulo, gamificación de progreso.
* **Riesgo:** confusión entre chatbots y agentes.

  * **Mitigación:** comparativas visuales y demos simples.

---

## 11. KPI de seguimiento interno

* % completado por módulo.
* % de descargas de plantillas.
* Tiempo promedio de permanencia por módulo.
* Tasa de entrega de capstone.

---

✅ Con este **PRD.md** tienes el blueprint completo para levantar el **Portal 1** en carpetas separadas, con Sonnet + VS Code, sin depender de código inicial mío.

