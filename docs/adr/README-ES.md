# Registros de Decisiones Arquitectónicas (ADR)

Este directorio contiene los Registros de Decisiones Arquitectónicas (ADR) para el Framework de Agentes IA, documentando decisiones técnicas clave siguiendo las mejores prácticas de la industria.

## ¿Qué son los ADRs?

Los Registros de Decisiones Arquitectónicas (ADR) son documentos que capturan una decisión arquitectónica importante junto con su contexto y consecuencias. Siguen el estándar establecido por ThoughtWorks y organizaciones líderes como Google, Microsoft y Netflix.

## Estructura de ADR

Cada ADR sigue esta estructura estándar:

- **Estado**: Propuesto, Aceptado, Rechazado, Reemplazado
- **Contexto**: La situación que motiva la decisión
- **Decisión**: La decisión tomada
- **Consecuencias**: Las implicaciones de la decisión

## Índice de ADRs

### ADRs Críticos

| ADR | Título | Estado | Fecha | Impacto |
|-----|--------|--------|-------|---------|
| [001](001-pec-sobre-react.md) | PEC sobre React para Framework de Agentes IA | ✅ Aceptado | 2025-08-17 | 🔴 Crítico |
| [002](002-rag-sobre-finetuning.md) | RAG sobre Fine-tuning para Conocimiento | ✅ Aceptado | 2025-08-17 | 🔴 Crítico |
| [003](003-patrones-multiagente.md) | Patrones de Coordinación Multi-Agente | ✅ Aceptado | 2025-08-17 | 🟡 Alto |
| [004](004-diseño-seguridad-primero.md) | Diseño de Seguridad Primero | ✅ Aceptado | 2025-08-17 | 🔴 Crítico |

### ADRs Planificados

| ADR | Título | Estado | Prioridad |
|-----|--------|--------|-----------|
| 005 | Estrategia de Cache Multi-Nivel | 📋 Planificado | Alta |
| 006 | Patrones de Observabilidad | 📋 Planificado | Alta |
| 007 | Estrategia de Persistencia de Datos | 📋 Planificado | Media |
| 008 | Arquitectura de API Gateway | 📋 Planificado | Media |

## Proceso de ADR

### 1. Propuesta de ADR

Cuando se enfrenta a una decisión arquitectónica significativa:

1. Crear un nuevo archivo ADR usando la plantilla
2. Establecer estado como "Propuesto"
3. Documentar el contexto y opciones consideradas
4. Solicitar revisión del equipo

### 2. Revisión y Discusión

- Revisar con el equipo de arquitectura
- Evaluar impacto en el sistema existente
- Considerar alternativas y trade-offs
- Documentar feedback y modificaciones

### 3. Decisión

- Actualizar estado a "Aceptado" o "Rechazado"
- Documentar la decisión final y rationale
- Registrar consecuencias esperadas
- Comunicar al equipo

### 4. Implementación

- Usar el ADR como guía durante implementación
- Actualizar documentación relacionada
- Monitorear consecuencias reales vs esperadas

## Plantilla de ADR

```markdown
# ADR-XXX: [Título de la Decisión]

## Estado
[Propuesto | Aceptado | Rechazado | Reemplazado por ADR-YYY]

## Contexto
[Descripción del problema y factores que influyen en la decisión]

## Decisión
[La decisión tomada]

## Consecuencias
### Positivas
- [Beneficio 1]
- [Beneficio 2]

### Negativas
- [Trade-off 1]
- [Limitación 1]

### Neutras
- [Cambio neutral 1]

## Alternativas Consideradas
1. **Opción 1**: [Descripción y razón de rechazo]
2. **Opción 2**: [Descripción y razón de rechazo]

## Notas de Implementación
[Detalles específicos de implementación si es relevante]

## Referencias
- [Link 1]
- [Link 2]
```

## Principios para ADRs Efectivos

### 1. **Inmutabilidad**
Los ADRs son inmutables una vez aceptados. Si una decisión cambia, crear un nuevo ADR que reemplace al anterior.

### 2. **Contexto Rico**
Documentar no solo QUÉ se decidió, sino POR QUÉ y en qué CONTEXTO.

### 3. **Consecuencias Honesas**
Incluir tanto beneficios como limitaciones de la decisión.

### 4. **Trazabilidad**
Mantener enlaces entre ADRs relacionados y referencias externas.

### 5. **Brevedad**
Mantener los ADRs concisos pero completos.

## Herramientas y Automatización

### Validación de ADRs

```bash
#!/bin/bash
# validar-adr.sh

ADR_FILE=$1

# Verificar estructura requerida
required_sections=("Estado" "Contexto" "Decisión" "Consecuencias")

for section in "${required_sections[@]}"; do
    if ! grep -q "## $section" "$ADR_FILE"; then
        echo "ERROR: Sección faltante '$section' en $ADR_FILE"
        exit 1
    fi
done

echo "✅ ADR $ADR_FILE tiene estructura válida"
```

### Generación de Índice

```python
# generar-indice-adr.py
import os
import re
from datetime import datetime

def generar_indice_adr():
    adr_dir = "docs/adr"
    adrs = []
    
    for filename in sorted(os.listdir(adr_dir)):
        if filename.endswith('.md') and filename != 'README.md':
            with open(os.path.join(adr_dir, filename), 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Extraer metadatos
            title_match = re.search(r'# ADR-(\d+): (.+)', content)
            status_match = re.search(r'## Estado\s*\n(.+)', content)
            
            if title_match and status_match:
                number = title_match.group(1)
                title = title_match.group(2)
                status = status_match.group(1).strip()
                
                adrs.append({
                    'number': number,
                    'title': title,
                    'status': status,
                    'filename': filename
                })
    
    # Generar tabla markdown
    print("| ADR | Título | Estado | Archivo |")
    print("|-----|--------|--------|---------|")
    
    for adr in adrs:
        status_icon = "✅" if "Aceptado" in adr['status'] else "📋"
        print(f"| {adr['number']} | {adr['title']} | {status_icon} {adr['status']} | [{adr['filename']}]({adr['filename']}) |")

if __name__ == "__main__":
    generar_indice_adr()
```

## Mejores Prácticas

### 1. **Decisiones Significativas**
Solo crear ADRs para decisiones que:
- Tienen impacto arquitectónico significativo
- Son difíciles de revertir
- Afectan múltiples equipos
- Establecen precedentes importantes

### 2. **Participación del Equipo**
- Involucrar a stakeholders relevantes
- Buscar input de expertos en dominio
- Considerar perspectivas de operaciones y seguridad

### 3. **Documentación Viva**
- Revisar ADRs periódicamente
- Actualizar consecuencias basadas en experiencia real
- Crear nuevos ADRs cuando las decisiones evolucionen

### 4. **Integración con Desarrollo**
- Referenciar ADRs en pull requests
- Incluir validación de ADRs en CI/CD
- Usar ADRs en onboarding de nuevos miembros

## Métricas y Evolución

Tracking de efectividad de ADRs:

- **Tasa de Implementación**: % de ADRs implementados según especificación
- **Precisión de Consecuencias**: Qué tan precisas fueron las predicciones
- **Frecuencia de Revisión**: Qué tan a menudo se revisan decisiones
- **Impacto en Decisiones Futuras**: Cuántos ADRs referencian decisiones anteriores

## Recursos Adicionales

- [ADR Tools](https://github.com/npryce/adr-tools) - Herramientas de línea de comandos
- [ADR GitHub Template](https://github.com/joelparkerhenderson/architecture_decision_record) - Plantillas y ejemplos
- [ThoughtWorks ADR](https://www.thoughtworks.com/radar/techniques/lightweight-architecture-decision-records) - Filosofía original

---

Los ADRs son una herramienta fundamental para mantener la coherencia arquitectónica y facilitar la evolución del Framework de Agentes IA a medida que crece y se adapta a nuevos requisitos.
