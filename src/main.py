"""
Portal 1: Fundamentos de IA para Desarrolladores
FastAPI Application Main Entry Point
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import os

from api.routes import modules, progress, auth
from models.database import init_db

# Crear instancia de FastAPI
app = FastAPI(
    title="Portal 1: Fundamentos de IA para Desarrolladores",
    description="Curso de capacitación en IA para desarrolladores sin experiencia previa en agentes o RAG",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios exactos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar archivos estáticos
if os.path.exists("public"):
    app.mount("/static", StaticFiles(directory="public"), name="static")

# Incluir rutas de API
app.include_router(modules.router, prefix="/api/modules", tags=["modules"])
app.include_router(progress.router, prefix="/api/progress", tags=["progress"])
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])

@app.on_event("startup")
async def startup_event():
    """Inicializar base de datos y configuraciones al arrancar"""
    await init_db()
    print("🚀 Portal 1 iniciado correctamente")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Landing page del portal"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Portal 1: Fundamentos de IA para Desarrolladores</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                min-height: 100vh;
            }
            .container {
                max-width: 1000px;
                margin: 0 auto;
                text-align: center;
            }
            .hero {
                padding: 60px 0;
            }
            h1 {
                font-size: 3em;
                margin-bottom: 0.5em;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            .subtitle {
                font-size: 1.3em;
                margin-bottom: 2em;
                opacity: 0.9;
            }
            .cta-button {
                display: inline-block;
                background: #ff6b6b;
                color: white;
                padding: 15px 30px;
                text-decoration: none;
                border-radius: 50px;
                font-size: 1.1em;
                font-weight: bold;
                margin: 10px;
                transition: transform 0.3s ease;
            }
            .cta-button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }
            .features {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 30px;
                margin: 60px 0;
            }
            .feature {
                background: rgba(255,255,255,0.1);
                padding: 30px;
                border-radius: 15px;
                backdrop-filter: blur(10px);
            }
            .feature h3 {
                margin-bottom: 15px;
                font-size: 1.3em;
            }
            .modules {
                margin: 60px 0;
                text-align: left;
            }
            .module {
                background: rgba(255,255,255,0.1);
                padding: 20px;
                margin: 15px 0;
                border-radius: 10px;
                border-left: 4px solid #ff6b6b;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="hero">
                <h1>🤖 Portal 1: Fundamentos de IA</h1>
                <p class="subtitle">Tu puerta de entrada al mundo de los agentes de IA y RAG</p>
                
                <a href="/api/docs" class="cta-button">📚 Explorar API</a>
                <a href="/api/modules" class="cta-button">🎯 Ver Módulos</a>
            </div>
            
            <div class="features">
                <div class="feature">
                    <h3>🧠 Agentes Inteligentes</h3>
                    <p>Aprende la diferencia entre chatbots y agentes de IA, y cómo construir sistemas que planifican y actúan.</p>
                </div>
                <div class="feature">
                    <h3>🔍 RAG Básico</h3>
                    <p>Implementa Retrieval-Augmented Generation con citas canónicas y técnicas de re-ranking.</p>
                </div>
                <div class="feature">
                    <h3>🛡️ Seguridad Mínima</h3>
                    <p>Aplica buenas prácticas de seguridad y validación en sistemas de IA desde el principio.</p>
                </div>
                <div class="feature">
                    <h3>📊 Métricas y Evaluación</h3>
                    <p>Mide calidad, coste y latencia con herramientas automatizadas de evaluación.</p>
                </div>
            </div>
            
            <div class="modules">
                <h2>📋 Módulos del Curso</h2>
                <div class="module">
                    <h3>Módulo A: Conceptos Esenciales</h3>
                    <p>Agentes vs chat • JSON estructurado • Seguridad mínima</p>
                </div>
                <div class="module">
                    <h3>Módulo B: Primer Mini-Agente</h3>
                    <p>Patrón Planner→Executor→Critic • Tool calling seguro</p>
                </div>
                <div class="module">
                    <h3>Módulo C: RAG Básico</h3>
                    <p>BM25/TF-IDF • Citas canónicas • MMR re-ranking</p>
                </div>
                <div class="module">
                    <h3>Módulo D: Métricas y Evaluación</h3>
                    <p>Quick evals • Coste tokens • Latencia • Gates de calidad</p>
                </div>
                <div class="module">
                    <h3>Módulo E: Capstone</h3>
                    <p>Proyecto final integrador • Repo para entrevistas</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """Endpoint de verificación de salud"""
    return {"status": "healthy", "version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
