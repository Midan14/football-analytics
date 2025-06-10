"""
main_simple.py - Servidor Backend Simplificado Football Analytics

Versión simplificada del servidor para resolver problemas de dependencias
"""

import os
import sys
from datetime import datetime
from typing import Dict, Any

# Añadir el directorio app al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    import uvicorn
except ImportError as e:
    print(f"❌ Error importando dependencias básicas: {e}")
    print("🔧 Ejecuta: pip install fastapi uvicorn")
    sys.exit(1)

# ================================
# CONFIGURACIÓN BÁSICA
# ================================

app = FastAPI(
    title="Football Analytics API - Simplified",
    description="Sistema simplificado de análisis de fútbol",
    version="2.1.0-simple",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ================================
# MIDDLEWARE
# ================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================
# SERVICIOS MOCK SIMPLIFICADOS
# ================================

class MockServices:
    """Servicios mock para testing básico"""
    
    @staticmethod
    def get_teams(league: str = "PL"):
        """Retorna equipos mock"""
        teams = {
            "PL": ["Arsenal", "Chelsea", "Liverpool", "Manchester City", "Manchester United"],
            "ES": ["Real Madrid", "Barcelona", "Atletico Madrid", "Sevilla", "Valencia"]
        }
        return teams.get(league, ["Team A", "Team B", "Team C"])
    
    @staticmethod
    def predict_match(home_team: str, away_team: str, league: str = "PL"):
        """Predicción mock"""
        import random
        predictions = ["home_win", "draw", "away_win"]
        return {
            "home_team": home_team,
            "away_team": away_team,
            "league": league,
            "prediction": random.choice(predictions),
            "confidence": round(random.uniform(0.4, 0.9), 2),
            "odds": {
                "home": round(random.uniform(1.5, 4.0), 2),
                "draw": round(random.uniform(2.8, 4.5), 2),
                "away": round(random.uniform(1.5, 4.0), 2)
            },
            "timestamp": datetime.now().isoformat(),
            "mock": True
        }

# ================================
# RUTAS PRINCIPALES
# ================================

@app.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "message": "🚀 Football Analytics API - Simplificado",
        "version": "2.1.0-simple",
        "status": "operational",
        "docs": "/docs",
        "health": "/health",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check simplificado"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.1.0-simple",
        "environment": "development",
        "components": {
            "api": {"status": "healthy"},
            "services": {"status": "healthy"}
        }
    }

@app.get("/teams/{league}")
async def get_teams(league: str):
    """Obtener equipos por liga"""
    try:
        teams = MockServices.get_teams(league)
        return {
            "success": True,
            "league": league,
            "teams": teams,
            "count": len(teams),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict_match(
    home_team: str,
    away_team: str,
    league: str = "PL"
):
    """Predicción de partido"""
    try:
        prediction = MockServices.predict_match(home_team, away_team, league)
        return {
            "success": True,
            "data": prediction,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/matches/recent")
async def get_recent_matches(league: str = "PL"):
    """Partidos recientes mock"""
    import random
    teams = MockServices.get_teams(league)
    matches = []
    
    for i in range(5):
        home = random.choice(teams)
        away = random.choice([t for t in teams if t != home])
        matches.append({
            "id": f"match_{i+1}",
            "home_team": home,
            "away_team": away,
            "date": f"2024-06-{str(i+1).zfill(2)}",
            "status": "completed",
            "league": league
        })
    
    return {
        "success": True,
        "league": league,
        "matches": matches,
        "count": len(matches),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/stats")
async def get_stats():
    """Estadísticas del sistema"""
    return {
        "success": True,
        "stats": {
            "total_predictions": 0,
            "active_leagues": ["PL", "ES", "DE", "IT", "FR"],
            "uptime": "Sistema iniciado",
            "version": "2.1.0-simple"
        },
        "timestamp": datetime.now().isoformat()
    }

# ================================
# FUNCIÓN PRINCIPAL
# ================================

def main():
    """Función principal"""
    print(
        f"""
🏈 FOOTBALL ANALYTICS - SERVIDOR SIMPLIFICADO
{'='*50}
📦 Versión: 2.1.0-simple
🌍 Entorno: development
🌐 API: http://localhost:8000
📊 Docs: http://localhost:8000/docs
📋 Health: http://localhost:8000/health
{'='*50}
    """
    )
    
    try:
        uvicorn.run(
            "main_simple:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🔄 Cerrando servidor...")
    except Exception as e:
        print(f"❌ Error ejecutando servidor: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
