"""
API Router para módulos del curso
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import json
import os

router = APIRouter()

@router.get("/")
async def get_all_modules():
    """Obtener lista de todos los módulos"""
    try:
        with open("project.config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        
        modules = []
        for module in config.get("modules", []):
            modules.append({
                "id": module["id"],
                "name": module["name"],
                "description": module["description"],
                "duration": module["duration"],
                "objectives": module["objectives"]
            })
        
        return {
            "total_modules": len(modules),
            "modules": modules
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading modules: {str(e)}")

@router.get("/{module_id}")
async def get_module_detail(module_id: str):
    """Obtener detalles de un módulo específico"""
    try:
        with open("project.config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        
        module = next((m for m in config["modules"] if m["id"] == module_id), None)
        
        if not module:
            raise HTTPException(status_code=404, detail="Module not found")
        
        # Intentar cargar contenido del módulo
        content_path = module["contentPath"]
        readme_path = os.path.join(content_path, "README.md")
        
        content = ""
        if os.path.exists(readme_path):
            with open(readme_path, "r", encoding="utf-8") as f:
                content = f.read()
        
        return {
            **module,
            "content": content,
            "has_labs": os.path.exists(module["labPath"]),
            "lab_files": get_lab_files(module["labPath"]) if os.path.exists(module["labPath"]) else []
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading module: {str(e)}")

def get_lab_files(lab_path: str) -> List[str]:
    """Obtener lista de archivos de laboratorio"""
    try:
        files = []
        for file in os.listdir(lab_path):
            if file.endswith('.py'):
                files.append(file)
        return files
    except:
        return []

@router.get("/{module_id}/labs")
async def get_module_labs(module_id: str):
    """Obtener laboratorios de un módulo"""
    try:
        with open("project.config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        
        module = next((m for m in config["modules"] if m["id"] == module_id), None)
        
        if not module:
            raise HTTPException(status_code=404, detail="Module not found")
        
        lab_path = module["labPath"]
        
        if not os.path.exists(lab_path):
            return {"labs": []}
        
        labs = []
        for file in os.listdir(lab_path):
            if file.endswith('.py'):
                file_path = os.path.join(lab_path, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                labs.append({
                    "filename": file,
                    "title": extract_title_from_content(content),
                    "description": extract_description_from_content(content),
                    "content": content
                })
        
        return {"labs": labs}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading labs: {str(e)}")

def extract_title_from_content(content: str) -> str:
    """Extraer título del contenido del archivo"""
    lines = content.split('\n')
    for line in lines[:10]:  # Buscar en las primeras 10 líneas
        if line.strip().startswith('"""') and len(line.strip()) > 3:
            return line.strip().replace('"""', '').strip()
        elif line.strip().startswith('#') and 'Laboratorio' in line:
            return line.strip().replace('#', '').strip()
    return "Lab sin título"

def extract_description_from_content(content: str) -> str:
    """Extraer descripción del contenido del archivo"""
    lines = content.split('\n')
    in_docstring = False
    description_lines = []
    
    for line in lines[:20]:  # Buscar en las primeras 20 líneas
        if line.strip().startswith('"""'):
            if in_docstring:
                break
            in_docstring = True
            continue
        elif in_docstring:
            if line.strip():
                description_lines.append(line.strip())
            elif description_lines:  # Si ya tenemos descripción y encontramos línea vacía
                break
    
    return ' '.join(description_lines) if description_lines else "Sin descripción disponible"
