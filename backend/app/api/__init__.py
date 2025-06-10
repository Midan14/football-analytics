"""
API Package Initialization for Football Analytics

Este módulo contiene toda la lógica de la API REST para el sistema de análisis
de fútbol, incluyendo endpoints para equipos, jugadores, partidos y predicciones.
"""

from fastapi import APIRouter

from .routes import leagues, live, matches, players, predictions, teams

# Información del módulo API
__version__ = "1.0.0"
__author__ = "Football Analytics Team"
__description__ = "API REST para análisis predictivo de fútbol"

# Metadatos del API
API_METADATA = {
    "title": "Football Analytics API",
    "description": "API completa para análisis predictivo de fútbol con ML",
    "version": __version__,
    "contact": {
        "name": "Football Analytics Support",
        "email": "support@football-analytics.com",
    },
    "license": {
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    "tags_metadata": [
        {
            "name": "leagues",
            "description": "Gestión de ligas de fútbol",
            "externalDocs": {
                "description": "Documentación de ligas",
                "url": "https://docs.football-analytics.com/leagues",
            },
        },
        {
            "name": "teams",
            "description": "Gestión de equipos de fútbol",
            "externalDocs": {
                "description": "Documentación de equipos",
                "url": "https://docs.football-analytics.com/teams",
            },
        },
        {
            "name": "players",
            "description": "Gestión de jugadores de fútbol",
            "externalDocs": {
                "description": "Documentación de jugadores",
                "url": "https://docs.football-analytics.com/players",
            },
        },
        {
            "name": "matches",
            "description": "Gestión de partidos de fútbol",
            "externalDocs": {
                "description": "Documentación de partidos",
                "url": "https://docs.football-analytics.com/matches",
            },
        },
        {
            "name": "predictions",
            "description": "Predicciones ML para análisis de fútbol",
            "externalDocs": {
                "description": "Documentación de predicciones",
                "url": "https://docs.football-analytics.com/predictions",
            },
        },
        {
            "name": "live",
            "description": "Datos en tiempo real y WebSockets",
            "externalDocs": {
                "description": "Documentación de tiempo real",
                "url": "https://docs.football-analytics.com/live",
            },
        },
    ],
}

# Router principal del API
main_router = APIRouter()

# Registrar todos los routers de los módulos
main_router.include_router(leagues.router, prefix="/leagues", tags=["leagues"])

main_router.include_router(teams.router, prefix="/teams", tags=["teams"])

main_router.include_router(players.router, prefix="/players", tags=["players"])

main_router.include_router(matches.router, prefix="/matches", tags=["matches"])

main_router.include_router(
    predictions.router, prefix="/predictions", tags=["predictions"]
)

main_router.include_router(live.router, prefix="/live", tags=["live"])


# Endpoint de salud para el API
@main_router.get("/health", summary="Health Check", tags=["system"])
async def health_check():
    """
    Endpoint simple para verificar que la API está funcionando correctamente.

    Returns:
        dict: Estado del sistema y metadatos básicos
    """
    return {
        "status": "healthy",
        "message": "Football Analytics API is running",
        "version": __version__,
        "api_docs": "/docs",
        "api_redoc": "/redoc",
        "endpoints_available": [
            "/api/v1/leagues",
            "/api/v1/teams",
            "/api/v1/players",
            "/api/v1/matches",
            "/api/v1/predictions",
            "/api/v1/live",
        ],
    }


# Endpoint de información del API
@main_router.get("/info", summary="API Information", tags=["system"])
async def api_info():
    """
    Información detallada sobre la API y sus capacidades.

    Returns:
        dict: Metadatos completos del API
    """
    return {
        "api_metadata": API_METADATA,
        "modules": {
            "leagues": "Gestión completa de ligas mundiales",
            "teams": "Análisis de equipos con estadísticas avanzadas",
            "players": "Perfiles de jugadores con métricas ML",
            "matches": "Partidos con datos en tiempo real",
            "predictions": "Predicciones ML para apuestas deportivas",
            "live": "WebSockets para seguimiento en vivo",
        },
        "features": [
            "Machine Learning predictions",
            "Real-time data",
            "Advanced analytics",
            "Betting odds integration",
            "Injury tracking",
            "Performance metrics",
            "Head-to-head statistics",
        ],
        "technologies": [
            "FastAPI",
            "PostgreSQL",
            "Redis",
            "SQLAlchemy",
            "Scikit-learn",
            "XGBoost",
            "WebSockets",
            "Celery",
        ],
    }


# Funciones de utilidad para otros módulos
def get_api_version():
    """Obtener versión actual del API"""
    return __version__


def get_api_metadata():
    """Obtener metadatos completos del API"""
    return API_METADATA


def get_available_endpoints():
    """Obtener lista de endpoints disponibles"""
    return [
        {"path": "/leagues", "description": "Gestión de ligas"},
        {"path": "/teams", "description": "Gestión de equipos"},
        {"path": "/players", "description": "Gestión de jugadores"},
        {"path": "/matches", "description": "Gestión de partidos"},
        {"path": "/predictions", "description": "Predicciones ML"},
        {"path": "/live", "description": "Datos en tiempo real"},
    ]


# Exportar componentes principales
__all__ = [
    "main_router",
    "API_METADATA",
    "get_api_version",
    "get_api_metadata",
    "get_available_endpoints",
]
