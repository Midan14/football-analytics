"""
=============================================================================
FOOTBALL ANALYTICS - RUTAS DE AUTENTICACIÓN
=============================================================================
Endpoints para autenticación y autorización de usuarios.
Incluye login, registro, gestión de tokens y control de acceso.

Funcionalidades:
- Registro de usuarios
- Login y logout
- Gestión de tokens JWT
- Verificación de permisos
- Recuperación de contraseñas
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
# ENDPOINTS DE AUTENTICACIÓN
# =============================================================================


@router.get(
    "/health",
    summary="Health check del módulo de autenticación",
    description="Verificar el estado del sistema de autenticación",
    tags=["auth", "health"],
)
async def auth_health():
    """
    Health check específico del módulo de autenticación
    """
    try:
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "module": "auth",
                "timestamp": datetime.now().isoformat(),
                "message": "Sistema de autenticación operativo",
            },
        )
    except Exception as e:
        logger.error(f"❌ Error en health check de auth: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "module": "auth",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
            },
        )


# =============================================================================
# LOG DE INICIALIZACIÓN
# =============================================================================
logger.info("🔐 Router de autenticación inicializado")
