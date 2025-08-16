"""
API Router para autenticación básica
"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional
import jwt
import json
import os
from datetime import datetime, timedelta
import hashlib

router = APIRouter()
security = HTTPBearer()

# Configuración JWT (en producción usar variables de entorno)
SECRET_KEY = "portal1-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 horas

# Modelos
class UserRegister(BaseModel):
    email: str
    name: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class UserProfile(BaseModel):
    email: str
    name: str
    role: str = "student"
    created_at: datetime
    last_login: Optional[datetime] = None

# Storage simple (en producción usar base de datos)
USERS_FILE = "data/users.json"

def load_users() -> dict:
    """Cargar usuarios desde archivo"""
    if not os.path.exists(USERS_FILE):
        os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
        return {}
    
    try:
        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {}

def save_users(users: dict):
    """Guardar usuarios en archivo"""
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, indent=2, default=str)

def hash_password(password: str) -> str:
    """Hash de contraseña simple (en producción usar bcrypt)"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    """Verificar contraseña"""
    return hash_password(password) == hashed

def create_access_token(data: dict) -> str:
    """Crear token JWT"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Verificar token JWT"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token inválido"
            )
        return payload
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido"
        )

@router.post("/register")
async def register_user(user: UserRegister):
    """Registrar nuevo usuario"""
    try:
        users = load_users()
        
        # Verificar si el usuario ya existe
        if user.email in users:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El usuario ya existe"
            )
        
        # Validaciones básicas
        if len(user.password) < 6:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="La contraseña debe tener al menos 6 caracteres"
            )
        
        if "@" not in user.email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email inválido"
            )
        
        # Crear usuario
        users[user.email] = {
            "email": user.email,
            "name": user.name,
            "password": hash_password(user.password),
            "role": "student",
            "created_at": datetime.now().isoformat(),
            "last_login": None
        }
        
        save_users(users)
        
        # Crear token
        access_token = create_access_token({"sub": user.email})
        
        return {
            "message": "Usuario registrado exitosamente",
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "email": user.email,
                "name": user.name,
                "role": "student"
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en el registro: {str(e)}"
        )

@router.post("/login")
async def login_user(user: UserLogin):
    """Iniciar sesión"""
    try:
        users = load_users()
        
        # Verificar usuario existe
        if user.email not in users:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Credenciales inválidas"
            )
        
        user_data = users[user.email]
        
        # Verificar contraseña
        if not verify_password(user.password, user_data["password"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Credenciales inválidas"
            )
        
        # Actualizar último login
        user_data["last_login"] = datetime.now().isoformat()
        users[user.email] = user_data
        save_users(users)
        
        # Crear token
        access_token = create_access_token({"sub": user.email})
        
        return {
            "message": "Login exitoso",
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "email": user_data["email"],
                "name": user_data["name"],
                "role": user_data["role"]
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en el login: {str(e)}"
        )

@router.get("/profile")
async def get_user_profile(current_user: dict = Depends(verify_token)):
    """Obtener perfil del usuario actual"""
    try:
        users = load_users()
        email = current_user["sub"]
        
        if email not in users:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Usuario no encontrado"
            )
        
        user_data = users[email]
        
        return {
            "email": user_data["email"],
            "name": user_data["name"],
            "role": user_data["role"],
            "created_at": user_data["created_at"],
            "last_login": user_data.get("last_login")
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo perfil: {str(e)}"
        )

@router.put("/profile")
async def update_user_profile(
    name: str,
    current_user: dict = Depends(verify_token)
):
    """Actualizar perfil del usuario"""
    try:
        users = load_users()
        email = current_user["sub"]
        
        if email not in users:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Usuario no encontrado"
            )
        
        # Actualizar nombre
        users[email]["name"] = name
        save_users(users)
        
        return {
            "message": "Perfil actualizado exitosamente",
            "user": {
                "email": users[email]["email"],
                "name": users[email]["name"],
                "role": users[email]["role"]
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error actualizando perfil: {str(e)}"
        )

@router.post("/logout")
async def logout_user(current_user: dict = Depends(verify_token)):
    """Cerrar sesión (principalmente para documentación)"""
    return {
        "message": "Sesión cerrada exitosamente. Elimina el token del cliente."
    }

@router.get("/verify")
async def verify_user_token(current_user: dict = Depends(verify_token)):
    """Verificar si el token es válido"""
    return {
        "valid": True,
        "user": current_user["sub"],
        "expires": current_user.get("exp")
    }
