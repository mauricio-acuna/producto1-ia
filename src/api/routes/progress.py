"""
API Router para tracking de progreso del estudiante
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime
import json
import os

router = APIRouter()

# Modelos Pydantic
class ProgressUpdate(BaseModel):
    module_id: str
    lesson_id: Optional[str] = None
    completed: bool
    score: Optional[float] = None
    time_spent: Optional[int] = None  # en segundos

class ModuleProgress(BaseModel):
    module_id: str
    completion_percentage: float
    lessons_completed: int
    total_lessons: int
    average_score: Optional[float] = None
    total_time_spent: int = 0

class StudentProgress(BaseModel):
    student_id: str
    overall_completion: float
    modules: List[ModuleProgress]
    total_time_spent: int
    last_activity: datetime

# Storage simple en archivo JSON (en producción usar base de datos)
PROGRESS_FILE = "data/student_progress.json"

def load_progress_data() -> Dict:
    """Cargar datos de progreso desde archivo"""
    if not os.path.exists(PROGRESS_FILE):
        os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
        return {}
    
    try:
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {}

def save_progress_data(data: Dict):
    """Guardar datos de progreso en archivo"""
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)

@router.get("/student/{student_id}")
async def get_student_progress(student_id: str):
    """Obtener progreso completo de un estudiante"""
    try:
        progress_data = load_progress_data()
        
        if student_id not in progress_data:
            # Crear progreso inicial
            return {
                "student_id": student_id,
                "overall_completion": 0.0,
                "modules": [],
                "total_time_spent": 0,
                "last_activity": None,
                "achievements": [],
                "current_module": None
            }
        
        student_data = progress_data[student_id]
        
        # Calcular estadísticas actualizadas
        total_modules = 5  # Según nuestro config
        completed_modules = sum(1 for m in student_data.get("modules", []) if m.get("completion_percentage", 0) >= 100)
        
        return {
            **student_data,
            "completed_modules": completed_modules,
            "total_modules": total_modules,
            "is_course_completed": completed_modules == total_modules
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading progress: {str(e)}")

@router.post("/student/{student_id}/update")
async def update_progress(student_id: str, progress: ProgressUpdate):
    """Actualizar progreso de un estudiante"""
    try:
        progress_data = load_progress_data()
        
        # Inicializar datos del estudiante si no existen
        if student_id not in progress_data:
            progress_data[student_id] = {
                "student_id": student_id,
                "overall_completion": 0.0,
                "modules": [],
                "total_time_spent": 0,
                "last_activity": datetime.now().isoformat(),
                "achievements": []
            }
        
        student_data = progress_data[student_id]
        
        # Buscar o crear módulo en progreso
        module_progress = None
        for module in student_data["modules"]:
            if module["module_id"] == progress.module_id:
                module_progress = module
                break
        
        if not module_progress:
            module_progress = {
                "module_id": progress.module_id,
                "completion_percentage": 0.0,
                "lessons_completed": 0,
                "total_lessons": get_module_lesson_count(progress.module_id),
                "scores": [],
                "total_time_spent": 0
            }
            student_data["modules"].append(module_progress)
        
        # Actualizar progreso del módulo
        if progress.completed:
            module_progress["lessons_completed"] += 1
        
        if progress.score is not None:
            if "scores" not in module_progress:
                module_progress["scores"] = []
            module_progress["scores"].append(progress.score)
        
        if progress.time_spent:
            module_progress["total_time_spent"] += progress.time_spent
            student_data["total_time_spent"] += progress.time_spent
        
        # Recalcular porcentaje de completitud
        if module_progress["total_lessons"] > 0:
            module_progress["completion_percentage"] = (
                module_progress["lessons_completed"] / module_progress["total_lessons"]
            ) * 100
        
        # Calcular promedio de scores
        if module_progress.get("scores"):
            module_progress["average_score"] = sum(module_progress["scores"]) / len(module_progress["scores"])
        
        # Actualizar progreso general
        total_completion = sum(m["completion_percentage"] for m in student_data["modules"])
        student_data["overall_completion"] = total_completion / (len(student_data["modules"]) * 100) * 100
        
        # Actualizar última actividad
        student_data["last_activity"] = datetime.now().isoformat()
        
        # Verificar logros
        check_achievements(student_data)
        
        # Guardar datos
        save_progress_data(progress_data)
        
        return {
            "success": True,
            "message": "Progress updated successfully",
            "student_progress": student_data
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating progress: {str(e)}")

def get_module_lesson_count(module_id: str) -> int:
    """Obtener número de lecciones en un módulo"""
    lesson_counts = {
        "module-a": 6,  # 5 conceptos + 1 quiz
        "module-b": 8,  # Más complejo
        "module-c": 7,
        "module-d": 5,
        "module-e": 3   # Capstone es más libre
    }
    return lesson_counts.get(module_id, 5)

def check_achievements(student_data: Dict):
    """Verificar y otorgar logros al estudiante"""
    achievements = student_data.get("achievements", [])
    
    # Logro: Primer módulo completado
    if any(m["completion_percentage"] >= 100 for m in student_data["modules"]):
        if "first_module_completed" not in achievements:
            achievements.append("first_module_completed")
    
    # Logro: Curso completado
    completed_modules = sum(1 for m in student_data["modules"] if m["completion_percentage"] >= 100)
    if completed_modules >= 5:
        if "course_completed" not in achievements:
            achievements.append("course_completed")
    
    # Logro: Puntuaciones altas
    high_scores = sum(1 for m in student_data["modules"] 
                     if m.get("average_score", 0) >= 90)
    if high_scores >= 3:
        if "high_achiever" not in achievements:
            achievements.append("high_achiever")
    
    student_data["achievements"] = achievements

@router.get("/analytics/overview")
async def get_analytics_overview():
    """Obtener analytics generales del curso"""
    try:
        progress_data = load_progress_data()
        
        if not progress_data:
            return {
                "total_students": 0,
                "average_completion": 0.0,
                "completion_rate": 0.0,
                "most_popular_module": None
            }
        
        students = list(progress_data.values())
        total_students = len(students)
        
        # Calcular métricas
        total_completion = sum(s.get("overall_completion", 0) for s in students)
        average_completion = total_completion / total_students if total_students > 0 else 0
        
        completed_students = sum(1 for s in students if s.get("overall_completion", 0) >= 95)
        completion_rate = (completed_students / total_students * 100) if total_students > 0 else 0
        
        # Módulo más popular (más estudiantes activos)
        module_activity = {}
        for student in students:
            for module in student.get("modules", []):
                module_id = module["module_id"]
                module_activity[module_id] = module_activity.get(module_id, 0) + 1
        
        most_popular_module = max(module_activity.items(), key=lambda x: x[1])[0] if module_activity else None
        
        return {
            "total_students": total_students,
            "average_completion": round(average_completion, 2),
            "completion_rate": round(completion_rate, 2),
            "most_popular_module": most_popular_module,
            "module_activity": module_activity
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading analytics: {str(e)}")

@router.get("/leaderboard")
async def get_leaderboard(limit: int = 10):
    """Obtener tabla de líderes"""
    try:
        progress_data = load_progress_data()
        
        students = []
        for student_id, data in progress_data.items():
            # Calcular score total
            total_score = 0
            score_count = 0
            
            for module in data.get("modules", []):
                if module.get("scores"):
                    total_score += sum(module["scores"])
                    score_count += len(module["scores"])
            
            average_score = total_score / score_count if score_count > 0 else 0
            
            students.append({
                "student_id": student_id,
                "overall_completion": data.get("overall_completion", 0),
                "average_score": average_score,
                "total_time_spent": data.get("total_time_spent", 0),
                "achievements_count": len(data.get("achievements", [])),
                "last_activity": data.get("last_activity")
            })
        
        # Ordenar por completitud y luego por score promedio
        students.sort(
            key=lambda x: (x["overall_completion"], x["average_score"]), 
            reverse=True
        )
        
        return {
            "leaderboard": students[:limit],
            "total_students": len(students)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading leaderboard: {str(e)}")
