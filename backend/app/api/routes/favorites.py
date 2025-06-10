"""
=============================================================================
FOOTBALL ANALYTICS - RUTAS DE SISTEMA DE FAVORITOS
=============================================================================
Endpoints para el sistema de favoritos de usuarios.
Permite gestionar equipos, jugadores y partidos favoritos.

Funcionalidades:
- Agregar/quitar equipos favoritos
- Gestionar jugadores favoritos
- Seguimiento de partidos favoritos
- Listas personalizadas de favoritos
- Notificaciones de favoritos
"""

import logging
from datetime import datetime

from fastapi import APIRouter
from fastapi.responses import JSONResponse

# =============================================================================
# CONFIGURACIÓN DE LOGGING
# =============================================================================
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURACIÓN DEL ROUTER
# =============================================================================
router = APIRouter()

# =============================================================================
# ENDPOINTS DE FAVORITOS
# =============================================================================


@router.get(
    "/health",
    summary="Health check del módulo de favoritos",
    description="Verificar el estado del sistema de favoritos",
    tags=["favorites", "health"],
)
async def favorites_health():
    """
    Health check específico del módulo de favoritos
    """
    try:
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "module": "favorites",
                "timestamp": datetime.now().isoformat(),
                "message": "Sistema de favoritos operativo",
            },
        )
    except Exception as e:
        logger.error(f"❌ Error en health check de favoritos: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "module": "favorites",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
            },
        )


# =============================================================================
# LOG DE INICIALIZACIÓN
# =============================================================================
logger.info("⭐ Router de favoritos inicializado")
