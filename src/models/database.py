"""
Modelos de base de datos para el Portal 1
"""

import sqlite3
import os
from typing import Optional
import json

DATABASE_PATH = "data/portal.db"

async def init_db():
    """Inicializar base de datos SQLite"""
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Tabla de usuarios
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'student',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            is_active BOOLEAN DEFAULT 1
        )
    ''')
    
    # Tabla de progreso por módulo
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS module_progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            module_id TEXT NOT NULL,
            completion_percentage REAL DEFAULT 0.0,
            lessons_completed INTEGER DEFAULT 0,
            total_lessons INTEGER DEFAULT 0,
            average_score REAL,
            total_time_spent INTEGER DEFAULT 0,
            last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id),
            UNIQUE(user_id, module_id)
        )
    ''')
    
    # Tabla de progreso por lección
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS lesson_progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            module_id TEXT NOT NULL,
            lesson_id TEXT NOT NULL,
            completed BOOLEAN DEFAULT 0,
            score REAL,
            time_spent INTEGER DEFAULT 0,
            completed_at TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id),
            UNIQUE(user_id, module_id, lesson_id)
        )
    ''')
    
    # Tabla de logros
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS achievements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            achievement_type TEXT NOT NULL,
            achieved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id),
            UNIQUE(user_id, achievement_type)
        )
    ''')
    
    # Tabla de evaluaciones
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            module_id TEXT NOT NULL,
            evaluation_type TEXT NOT NULL, -- quiz, lab, capstone
            questions TEXT, -- JSON con preguntas y respuestas
            score REAL,
            max_score REAL,
            time_spent INTEGER,
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Tabla de feedback y comentarios
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            module_id TEXT,
            feedback_type TEXT NOT NULL, -- rating, comment, suggestion
            content TEXT NOT NULL,
            rating INTEGER, -- 1-10 para NPS
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()
    
    print("✅ Base de datos inicializada correctamente")

def get_db_connection():
    """Obtener conexión a la base de datos"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row  # Para acceder a columnas por nombre
    return conn

class UserModel:
    """Modelo para manejo de usuarios"""
    
    @staticmethod
    def create_user(email: str, name: str, password_hash: str, role: str = "student"):
        """Crear nuevo usuario"""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO users (email, name, password_hash, role)
            VALUES (?, ?, ?, ?)
        ''', (email, name, password_hash, role))
        
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return user_id
    
    @staticmethod
    def get_user_by_email(email: str):
        """Obtener usuario por email"""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()
        
        conn.close()
        return dict(user) if user else None
    
    @staticmethod
    def update_last_login(email: str):
        """Actualizar último login"""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE users SET last_login = CURRENT_TIMESTAMP 
            WHERE email = ?
        ''', (email,))
        
        conn.commit()
        conn.close()

class ProgressModel:
    """Modelo para manejo de progreso"""
    
    @staticmethod
    def update_module_progress(user_id: int, module_id: str, **kwargs):
        """Actualizar progreso de módulo"""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Verificar si existe el registro
        cursor.execute('''
            SELECT id FROM module_progress 
            WHERE user_id = ? AND module_id = ?
        ''', (user_id, module_id))
        
        existing = cursor.fetchone()
        
        if existing:
            # Actualizar registro existente
            update_fields = []
            values = []
            
            for key, value in kwargs.items():
                if value is not None:
                    update_fields.append(f"{key} = ?")
                    values.append(value)
            
            if update_fields:
                values.append(user_id)
                values.append(module_id)
                
                query = f'''
                    UPDATE module_progress 
                    SET {", ".join(update_fields)}, last_activity = CURRENT_TIMESTAMP
                    WHERE user_id = ? AND module_id = ?
                '''
                
                cursor.execute(query, values)
        else:
            # Crear nuevo registro
            cursor.execute('''
                INSERT INTO module_progress 
                (user_id, module_id, completion_percentage, lessons_completed, 
                 total_lessons, average_score, total_time_spent)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id, module_id,
                kwargs.get('completion_percentage', 0.0),
                kwargs.get('lessons_completed', 0),
                kwargs.get('total_lessons', 0),
                kwargs.get('average_score'),
                kwargs.get('total_time_spent', 0)
            ))
        
        conn.commit()
        conn.close()
    
    @staticmethod
    def get_user_progress(user_id: int):
        """Obtener progreso completo del usuario"""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Progreso por módulos
        cursor.execute('''
            SELECT * FROM module_progress 
            WHERE user_id = ?
            ORDER BY module_id
        ''', (user_id,))
        
        modules = [dict(row) for row in cursor.fetchall()]
        
        # Logros
        cursor.execute('''
            SELECT achievement_type, achieved_at FROM achievements 
            WHERE user_id = ?
            ORDER BY achieved_at DESC
        ''', (user_id,))
        
        achievements = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            "modules": modules,
            "achievements": achievements
        }

class EvaluationModel:
    """Modelo para evaluaciones y quizzes"""
    
    @staticmethod
    def save_evaluation(user_id: int, module_id: str, eval_type: str, 
                       questions: dict, score: float, max_score: float, 
                       time_spent: int):
        """Guardar resultado de evaluación"""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO evaluations 
            (user_id, module_id, evaluation_type, questions, score, max_score, time_spent)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, module_id, eval_type, json.dumps(questions), 
              score, max_score, time_spent))
        
        conn.commit()
        conn.close()
    
    @staticmethod
    def get_user_evaluations(user_id: int, module_id: Optional[str] = None):
        """Obtener evaluaciones del usuario"""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if module_id:
            cursor.execute('''
                SELECT * FROM evaluations 
                WHERE user_id = ? AND module_id = ?
                ORDER BY submitted_at DESC
            ''', (user_id, module_id))
        else:
            cursor.execute('''
                SELECT * FROM evaluations 
                WHERE user_id = ?
                ORDER BY submitted_at DESC
            ''', (user_id,))
        
        evaluations = []
        for row in cursor.fetchall():
            eval_dict = dict(row)
            eval_dict['questions'] = json.loads(eval_dict['questions'])
            evaluations.append(eval_dict)
        
        conn.close()
        return evaluations
