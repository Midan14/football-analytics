"""
=============================================================================
FOOTBALL ANALYTICS - CONFIGURACIÓN CENTRALIZADA DE RUTAS API
=============================================================================
Configuración principal de todas las rutas de la API REST.
Este archivo registra todos los routers y los hace disponibles
para el sistema FastAPI del backend.

Rutas incluidas:
- /leagues      : Gestión de ligas y competiciones
- /teams        : Gestión de equipos y estadísticas
- /players      : Gestión de jugadores y perfiles
- /matches      : Gestión de partidos y eventos
- /predictions  : Predicciones AI y análisis
- /live         : Datos en tiempo real y WebSockets
- /injuries     : Sistema de seguimiento de lesiones
- /auth         : Autenticación y autorización
- /favorites    : Sistema de favoritos de usuarios
- /analytics    : Análisis avanzados y métricas
"""

import logging
import os
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

# =============================================================================
# CONFIGURACIÓN DE LOGGING
# =============================================================================
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTES
# =============================================================================
ROUTER_NOT_FOUND_MSG = "Router attribute not found or None"

# =============================================================================
# IMPORTACIÓN CONDICIONAL DE ROUTERS
# =============================================================================
# Lista para almacenar configuración de routers
routers_config = []
successfully_imported = []
failed_imports = []

# Router de ligas y competiciones
try:
    from . import leagues

    # Verificar que el módulo tiene el atributo router
    if getattr(leagues, "router", None) is not None:
        routers_config.append(
            {
                "router": leagues.router,
                "prefix": "/leagues",
                "tags": ["leagues"],
                "name": "Gestión de Ligas",
                "description": "Operaciones con ligas y competiciones",
            }
        )
        successfully_imported.append("leagues")
        logger.debug("✅ Router de ligas cargado")
    else:
        failed_imports.append(("leagues", ROUTER_NOT_FOUND_MSG))
        logger.warning("⚠️  Router de ligas sin atributo router")
except ImportError as e:
    failed_imports.append(("leagues", str(e)))
    logger.warning("⚠️  Router de ligas no encontrado")
except AttributeError as e:
    failed_imports.append(("leagues", f"AttributeError: {str(e)}"))
    logger.warning("⚠️  Error de atributo en router de ligas")

# Router de equipos
try:
    from . import teams

    # Verificar que el módulo tiene el atributo router
    if getattr(teams, "router", None) is not None:
        routers_config.append(
            {
                "router": teams.router,
                "prefix": "/teams",
                "tags": ["teams"],
                "name": "Gestión de Equipos",
                "description": "Operaciones con equipos y estadísticas",
            }
        )
        successfully_imported.append("teams")
        logger.debug("✅ Router de equipos cargado")
    else:
        failed_imports.append(("teams", ROUTER_NOT_FOUND_MSG))
        logger.warning("⚠️  Router de equipos sin atributo router")
except ImportError as e:
    failed_imports.append(("teams", str(e)))
    logger.warning("⚠️  Router de equipos no encontrado")
except AttributeError as e:
    failed_imports.append(("teams", f"AttributeError: {str(e)}"))
    logger.warning("⚠️  Error de atributo en router de equipos")

# Router de jugadores
try:
    from . import players

    # Verificar que el módulo tiene el atributo router
    if getattr(players, "router", None) is not None:
        routers_config.append(
            {
                "router": players.router,
                "prefix": "/players",
                "tags": ["players"],
                "name": "Gestión de Jugadores",
                "description": "Perfiles de jugadores y estadísticas",
            }
        )
        successfully_imported.append("players")
        logger.debug("✅ Router de jugadores cargado")
    else:
        failed_imports.append(("players", ROUTER_NOT_FOUND_MSG))
        logger.warning("⚠️  Router de jugadores sin atributo router")
except ImportError as e:
    failed_imports.append(("players", str(e)))
    logger.warning("⚠️  Router de jugadores no encontrado")
except AttributeError as e:
    failed_imports.append(("players", f"AttributeError: {str(e)}"))
    logger.warning("⚠️  Error de atributo en router de jugadores")

# Router de partidos
try:
    from . import matches

    # Verificar que el módulo tiene el atributo router
    if getattr(matches, "router", None) is not None:
        routers_config.append(
            {
                "router": matches.router,
                "prefix": "/matches",
                "tags": ["matches"],
                "name": "Gestión de Partidos",
                "description": "Partidos, eventos en vivo y resultados",
            }
        )
        successfully_imported.append("matches")
        logger.debug("✅ Router de partidos cargado")
    else:
        failed_imports.append(("matches", ROUTER_NOT_FOUND_MSG))
        logger.warning("⚠️  Router de partidos sin atributo router")
except ImportError as e:
    failed_imports.append(("matches", str(e)))
    logger.warning("⚠️  Router de partidos no encontrado")
except AttributeError as e:
    failed_imports.append(("matches", f"AttributeError: {str(e)}"))
    logger.warning("⚠️  Error de atributo en router de partidos")

# Router de predicciones AI
try:
    from . import predictions

    # Verificar que el módulo tiene el atributo router
    if getattr(predictions, "router", None) is not None:
        routers_config.append(
            {
                "router": predictions.router,
                "prefix": "/predictions",
                "tags": ["predictions", "ai"],
                "name": "Predicciones AI",
                "description": "Predicciones de resultados y análisis",
            }
        )
        successfully_imported.append("predictions")
        logger.debug("✅ Router de predicciones cargado")
    else:
        failed_imports.append(("predictions", ROUTER_NOT_FOUND_MSG))
        logger.warning("⚠️  Router de predicciones sin atributo router")
except ImportError as e:
    failed_imports.append(("predictions", str(e)))
    logger.warning("⚠️  Router de predicciones no encontrado")
except AttributeError as e:
    failed_imports.append(("predictions", f"AttributeError: {str(e)}"))
    logger.warning("⚠️  Error de atributo en router de predicciones")

# Router de datos en tiempo real
try:
    from . import live

    # Verificar que el módulo tiene el atributo router
    if getattr(live, "router", None) is not None:
        routers_config.append(
            {
                "router": live.router,
                "prefix": "/live",
                "tags": ["live", "web-sockets", "realtime"],
                "name": "Datos en Tiempo Real",
                "description": "WebSockets y partidos en vivo",
            }
        )
        successfully_imported.append("live")
        logger.debug("✅ Router de tiempo real cargado")
    else:
        failed_imports.append(("live", ROUTER_NOT_FOUND_MSG))
        logger.warning("⚠️  Router de tiempo real sin atributo router")
except ImportError as e:
    failed_imports.append(("live", str(e)))
    logger.warning("⚠️  Router de tiempo real no encontrado")
except AttributeError as e:
    failed_imports.append(("live", f"AttributeError: {str(e)}"))
    logger.warning("⚠️  Error de atributo en router de tiempo real")

# Router de lesiones (nuevo)
try:
    from . import injuries

    # Verificar que el módulo tiene el atributo router y que es válido
    if getattr(injuries, "router", None) is not None:
        routers_config.append(
            {
                "router": injuries.router,
                "prefix": "/injuries",
                "tags": ["injuries", "medical"],
                "name": "Seguimiento de Lesiones",
                "description": "Sistema de tracking de lesiones",
            }
        )
        successfully_imported.append("injuries")
        logger.debug("✅ Router de lesiones cargado")
    else:
        failed_imports.append(("injuries", ROUTER_NOT_FOUND_MSG))
        logger.warning("⚠️  Router de lesiones sin atributo router")
except ImportError as e:
    failed_imports.append(("injuries", str(e)))
    logger.warning("⚠️  Router de lesiones no encontrado")
except AttributeError as e:
    failed_imports.append(("injuries", f"AttributeError: {str(e)}"))
    logger.warning("⚠️  Error de atributo en router de lesiones")

# Router de autenticación (nuevo)
try:
    from . import auth

    # Verificar que el módulo tiene el atributo router
    if getattr(auth, "router", None) is not None:
        routers_config.append(
            {
                "router": auth.router,
                "prefix": "/auth",
                "tags": ["authentication", "security"],
                "name": "Autenticación",
                "description": "Login, registro y gestión de tokens",
            }
        )
        successfully_imported.append("auth")
        logger.debug("✅ Router de autenticación cargado")
    else:
        failed_imports.append(("auth", ROUTER_NOT_FOUND_MSG))
        logger.warning("⚠️  Router de autenticación sin atributo router")
except ImportError as e:
    failed_imports.append(("auth", str(e)))
    logger.warning("⚠️  Router de autenticación no encontrado")
except AttributeError as e:
    failed_imports.append(("auth", f"AttributeError: {str(e)}"))
    logger.warning("⚠️  Error de atributo en router de autenticación")

# Router de favoritos (nuevo)
try:
    from . import favorites

    # Verificar que el módulo tiene el atributo router
    if getattr(favorites, "router", None) is not None:
        routers_config.append(
            {
                "router": favorites.router,
                "prefix": "/favorites",
                "tags": ["favorites", "user"],
                "name": "Sistema de Favoritos",
                "description": "Gestión de equipos, jugadores y partidos",
            }
        )
        successfully_imported.append("favorites")
        logger.debug("✅ Router de favoritos cargado")
    else:
        failed_imports.append(("favorites", ROUTER_NOT_FOUND_MSG))
        logger.warning("⚠️  Router de favoritos sin atributo router")
except ImportError as e:
    failed_imports.append(("favorites", str(e)))
    logger.warning("⚠️  Router de favoritos no encontrado")
except AttributeError as e:
    failed_imports.append(("favorites", f"AttributeError: {str(e)}"))
    logger.warning("⚠️  Error de atributo en router de favoritos")

# Router de analytics avanzados (nuevo)
try:
    from . import analytics

    # Verificar que el módulo tiene el atributo router
    if getattr(analytics, "router", None) is not None:
        routers_config.append(
            {
                "router": analytics.router,
                "prefix": "/analytics",
                "tags": ["analytics", "statistics"],
                "name": "Análisis Avanzados",
                "description": "Métricas, dashboards y análisis estadísticos",
            }
        )
        successfully_imported.append("analytics")
        logger.debug("✅ Router de analytics cargado")
    else:
        failed_imports.append(("analytics", ROUTER_NOT_FOUND_MSG))
        logger.warning("⚠️  Router de analytics sin atributo router")
except ImportError as e:
    failed_imports.append(("analytics", str(e)))
    logger.warning("⚠️  Router de analytics no encontrado")
except AttributeError as e:
    failed_imports.append(("analytics", f"AttributeError: {str(e)}"))
    logger.warning("⚠️  Error de atributo en router de analytics")

# =============================================================================
# CONFIGURACIÓN DEL ROUTER PRINCIPAL
# =============================================================================
# Crear router principal que incluye todas las rutas
api_router = APIRouter()

# Registrar todos los routers disponibles
registered_routers = 0
registration_errors = []

for router_config in routers_config:
    try:
        api_router.include_router(
            router_config["router"],
            prefix=router_config["prefix"],
            tags=router_config["tags"],
        )
        registered_routers += 1
        logger.debug("✅ Router registrado: %s", router_config["prefix"])
    except Exception as e:
        error_msg = "Error registrando router %s: %s"
        formatted_msg = error_msg % (router_config["prefix"], e)
        registration_errors.append(formatted_msg)
        logger.error("❌ %s", formatted_msg)

# =============================================================================
# INFORMACIÓN DE RUTAS DISPONIBLES (COMPATIBLE CON VERSIÓN ORIGINAL)
# =============================================================================
# Lista de rutas disponibles (manteniendo compatibilidad)
available_routes = [
    {
        "prefix": config["prefix"],
        "name": config["name"],
        "module": config["prefix"].replace("/", ""),
        "description": config["description"],
        "tags": config["tags"],
    }
    for config in routers_config
]


# =============================================================================
# FUNCIONES DE INFORMACIÓN Y ESTADÍSTICAS
# =============================================================================
def get_routes_info() -> dict[str, Any]:
    """
    Retorna información sobre todas las rutas disponibles.
    Útil para debugging y documentación automática.
    """
    return {
        "total_routes": len(available_routes),
        "registered_routes": registered_routers,
        "failed_routes": len(routers_config) - registered_routers,
        "routes": available_routes,
        "api_version": "v1",
        "description": "Football Analytics API - Análisis predictivo",
        "successfully_imported": successfully_imported,
        "failed_imports": [
            {"module": name, "error": error} for name, error in failed_imports
        ],
        "registration_errors": registration_errors,
    }


def get_api_health() -> dict[str, Any]:
    """
    Información de salud de la API
    """
    health_status = "healthy" if registered_routers > 0 else "degraded"
    if (
        registered_routers == 0 and routers_config
    ):  # Check if routers_config is not empty
        health_status = "unhealthy"
    elif not routers_config:  # No routers configured at all
        health_status = "unhealthy"

    return {
        "status": health_status,
        "registered_routers": registered_routers,
        "total_routers_configured": len(routers_config),  # Renamed for clarity
        "timestamp": datetime.now().isoformat(),
        "environment": os.getenv("NODE_ENV", "development"),
    }


def get_api_statistics() -> dict[str, Any]:
    """
    Estadísticas detalladas de la API con detección comprehensiva
    de características
    """
    # Calcular porcentaje de éxito
    if routers_config:  # Check if routers_config is not empty before division
        success_pct = (
            (registered_routers / len(routers_config) * 100)
            if len(routers_config) > 0
            else 0
        )
        success_rate = f"{success_pct:.1f}%"
    else:
        # "N/A" si se prefiere cuando no hay routers configurados
        success_rate = "0%"

    # Enhanced route configuration with comprehensive feature detection

    # Check for authentication system availability
    has_auth_system = any(
        "auth" in config.get("prefix", "").lower()
        or "authentication" in config.get("name", "").lower()
        or config.get("auth_required", False)
        for config in routers_config
    )

    # Check for favorites/bookmarks system availability
    has_favorites_system = any(
        "favorites" in config.get("prefix", "").lower()
        or "bookmarks" in config.get("prefix", "").lower()
        or "likes" in config.get("prefix", "").lower()
        or config.get("features", {}).get("favorites", False)
        for config in routers_config
    )

    # Check for user management system
    has_user_management = any(
        "users" in config.get("prefix", "").lower()
        or "profiles" in config.get("prefix", "").lower()
        or config.get("features", {}).get("user_management", False)
        for config in routers_config
    )

    # Check for real-time features
    has_realtime_features = any(
        "live" in config.get("prefix", "").lower()
        or "realtime" in config.get("name", "").lower()
        or "web socket" in config.get("description", "").lower()
        or any(
            "live" in tag.lower() or "realtime" in tag.lower()
            for tag in config.get("tags", [])
        )
        for config in routers_config
    )

    # Check for AI/ML capabilities
    has_ai_features = any(
        "predictions" in config.get("prefix", "").lower()
        or "ai" in config.get("name", "").lower()
        or "machine learning" in config.get("description", "").lower()
        or any(
            "ai" in tag.lower() or "predictions" in tag.lower()
            for tag in config.get("tags", [])
        )
        for config in routers_config
    )

    # Check for analytics capabilities
    has_analytics_features = any(
        "analytics" in config.get("prefix", "").lower()
        or "statistics" in config.get("name", "").lower()
        or "metrics" in config.get("description", "").lower()
        or any(
            "analytics" in tag.lower() or "statistics" in tag.lower()
            for tag in config.get("tags", [])
        )
        for config in routers_config
    )

    # Check for injury tracking system
    has_injury_tracking = any(
        "injuries" in config.get("prefix", "").lower()
        or "medical" in config.get("name", "").lower()
        or "tracking" in config.get("description", "").lower()
        or any(
            "injuries" in tag.lower() or "medical" in tag.lower()
            for tag in config.get("tags", [])
        )
        for config in routers_config
    )

    return {
        "routers": {
            "total_configured": len(routers_config),
            "successfully_registered": registered_routers,
            "failed_registrations": len(routers_config) - registered_routers,
            "success_rate": success_rate,
        },
        "features": {
            # Características principales del sistema
            "authentication_system": has_auth_system,
            "favorites_system": has_favorites_system,
            "user_management": has_user_management,
            "real_time_data": has_realtime_features,
            "ai_predictions": has_ai_features,
            "advanced_analytics": has_analytics_features,
            "injury_tracking": has_injury_tracking,
            # Compatibilidad con versión anterior (deprecated)
            # Deprecated: use authentication_system
            "user_authentication": has_auth_system,
        },
        "feature_summary": {
            "total_features_detected": sum(
                [
                    has_auth_system,
                    has_favorites_system,
                    has_user_management,
                    has_realtime_features,
                    has_ai_features,
                    has_analytics_features,
                    has_injury_tracking,
                ]
            ),
            "core_systems": {
                "user_systems": has_auth_system or has_user_management,
                "data_systems": (has_realtime_features or has_analytics_features),
                "ai_systems": has_ai_features,
                "tracking_systems": (has_injury_tracking or has_favorites_system),
            },
        },
        "health": get_api_health(),
    }


# =============================================================================
# ENDPOINTS DE INFORMACIÓN Y DIAGNÓSTICO
# =============================================================================


@api_router.get(
    "/info",
    summary="Información de la API",
    description=(
        "Endpoint que retorna información sobre todas las rutas "
        "disponibles y el estado general de la API"
    ),
    tags=["system", "info"],
)
async def api_info():
    """
    Endpoint que retorna información sobre todas las rutas disponibles
    y el estado general de la API.
    """
    try:
        logger.info("📊 Información de API solicitada")
        return JSONResponse(
            status_code=200,
            content={
                "message": "Football Analytics API",
                "version": "1.0.0",
                "timestamp": datetime.now().isoformat(),
                "routes_info": get_routes_info(),
                "documentation": "/docs",
                "health_check": "/health",
                "docs_alternative": "/docs-alt",
            },
        )
    except Exception as e:
        logger.error("❌ Error obteniendo información de API: %s", str(e))
        raise HTTPException(status_code=500, detail="Error interno del servidor") from e


@api_router.get(
    "/health",
    summary="Health Check de la API",
    description="Verificar el estado de salud de la API y servicios",
    tags=["system", "health"],
)
async def api_health():
    """
    Health check de la API y servicios
    """
    try:
        logger.debug("🏥 Health check solicitado")
        health_data = get_api_health()

        # Determinar código de estado basado en la salud
        status_code = 200 if health_data.get("status") == "healthy" else 503

        return JSONResponse(status_code=status_code, content=health_data)
    except Exception as e:
        logger.error("❌ Error en health check: %s", str(e))
        return JSONResponse(
            status_code=503,  # Service Unavailable
            content={
                "status": "error",
                "message": "Health check failed due to an internal error.",
                "timestamp": datetime.now().isoformat(),
                # More specific error for internal logging
                "error_details": str(e),
            },
        )


@api_router.get(
    "/stats",
    summary="Estadísticas de la API",
    description="Estadísticas de uso, configuración y estado de la API",
    tags=["system", "statistics"],
)
async def api_statistics():
    """
    Estadísticas detalladas de la API
    """
    try:
        logger.info("📈 Estadísticas de API solicitadas")
        return JSONResponse(
            status_code=200,
            content={
                "message": "Estadísticas de Football Analytics API",
                "timestamp": datetime.now().isoformat(),
                **get_api_statistics(),
            },
        )
    except Exception as e:
        logger.error("❌ Error obteniendo estadísticas: %s", str(e))
        raise HTTPException(
            status_code=500, detail="Error interno al obtener estadísticas"
        ) from e


@api_router.get(
    "/routes",
    summary="Lista de rutas disponibles",
    description="Endpoint que retorna todas las rutas de la API",
    tags=["system", "documentation"],
)
async def list_routes():
    """
    Lista todas las rutas disponibles en la API
    """
    try:
        logger.debug("📋 Lista de rutas solicitada")
        return JSONResponse(
            status_code=200,
            content={
                "message": "Rutas disponibles en Football Analytics API",
                "timestamp": datetime.now().isoformat(),
                "routes": available_routes,
                "summary": {
                    # Renamed for clarity
                    "total_available": len(available_routes),
                    "registered_successfully": registered_routers,
                    "failed_to_register": (len(routers_config) - registered_routers),
                },
            },
        )
    except Exception as e:
        logger.error("❌ Error listando rutas: %s", str(e))
        raise HTTPException(
            status_code=500, detail="Error interno al listar rutas"
        ) from e


@api_router.get(
    "/features",
    summary="Análisis avanzado de características",
    description=(
        "Endpoint que retorna un análisis detallado de las capacidades "
        "y características del sistema"
    ),
    tags=["system", "features", "analysis"],
)
async def enhanced_features_analysis():
    """
    Análisis avanzado de características disponibles en el sistema
    """
    try:
        logger.info("🔍 Análisis de características solicitado")
        analysis_data = get_enhanced_feature_analysis()

        return JSONResponse(
            status_code=200,
            content={
                "message": ("Análisis de características de Football Analytics API"),
                "timestamp": datetime.now().isoformat(),
                "analysis": analysis_data,
                "metadata": {
                    "total_routers_analyzed": len(routers_config),
                    "analysis_version": "1.0.0",
                    "generated_at": datetime.now().isoformat(),
                },
            },
        )
    except Exception as e:
        logger.error("❌ Error en análisis de características: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail="Error interno al analizar características del sistema",
        ) from e


# =============================================================================
# FUNCIONES HELPER PARA COMPATIBILIDAD Y DETECCIÓN AVANZADA
# =============================================================================
def get_enhanced_feature_analysis() -> dict[str, Any]:
    """
    Análisis avanzado de características disponibles en el sistema

    Returns:
        Análisis detallado de capacidades y características del sistema
    """
    # Detectar características por categorías
    auth_features = []
    data_features = []
    ai_features = []
    user_features = []

    for config in routers_config:
        prefix = config.get("prefix", "").lower()
        name = config.get("name", "").lower()

        # Categorizar características de autenticación
        if any(
            keyword in prefix or keyword in name
            for keyword in ["auth", "login", "security"]
        ):
            auth_features.extend(
                ["user_authentication", "session_management", "token_handling"]
            )

        # Categorizar características de datos
        if any(
            keyword in prefix or keyword in name
            for keyword in ["live", "real", "stream"]
        ):
            data_features.extend(["real_time_data", "live_updates", "streaming"])

        if any(
            keyword in prefix or keyword in name
            for keyword in ["analytics", "stats", "metrics"]
        ):
            data_features.extend(
                [
                    "advanced_analytics",
                    "statistical_analysis",
                    "data_visualization",
                ]
            )

        # Categorizar características de IA
        if any(
            keyword in prefix or keyword in name
            for keyword in ["prediction", "ai", "ml"]
        ):
            ai_features.extend(
                ["predictive_analytics", "machine_learning", "ai_insights"]
            )

        # Categorizar características de usuario
        if any(
            keyword in prefix or keyword in name
            for keyword in ["favorites", "bookmarks", "profile"]
        ):
            user_features.extend(["user_preferences", "personalization", "bookmarking"])

        if any(
            keyword in prefix or keyword in name
            for keyword in ["injuries", "medical", "health"]
        ):
            data_features.extend(
                ["injury_tracking", "medical_monitoring", "health_analytics"]
            )

    # Eliminar duplicados
    auth_features = list(set(auth_features))
    data_features = list(set(data_features))
    ai_features = list(set(ai_features))
    user_features = list(set(user_features))

    return {
        "feature_categories": {
            "authentication": {
                "available": len(auth_features) > 0,
                "features": auth_features,
                "count": len(auth_features),
            },
            "data_management": {
                "available": len(data_features) > 0,
                "features": data_features,
                "count": len(data_features),
            },
            "artificial_intelligence": {
                "available": len(ai_features) > 0,
                "features": ai_features,
                "count": len(ai_features),
            },
            "user_experience": {
                "available": len(user_features) > 0,
                "features": user_features,
                "count": len(user_features),
            },
        },
        "system_capabilities": {
            "total_feature_categories": sum(
                [
                    len(auth_features) > 0,
                    len(data_features) > 0,
                    len(ai_features) > 0,
                    len(user_features) > 0,
                ]
            ),
            "total_features_detected": len(auth_features)
            + len(data_features)
            + len(ai_features)
            + len(user_features),
            "system_maturity": (
                "enterprise"
                if (
                    len(auth_features) > 0
                    and len(data_features) > 0
                    and len(ai_features) > 0
                )
                else "standard"
            ),
            "feature_coverage": {
                "has_core_auth": len(auth_features) > 0,
                "has_data_processing": len(data_features) > 0,
                "has_ai_capabilities": len(ai_features) > 0,
                "has_user_features": len(user_features) > 0,
            },
        },
    }


def get_api_router() -> APIRouter:
    """
    Obtener el router principal configurado

    Returns:
        APIRouter con todas las rutas registradas
    """
    logger.info(
        "🔌 Router principal con %s rutas registradas de %s configuradas.",
        registered_routers,
        len(routers_config),
    )
    return api_router


def get_router_summary() -> dict[str, Any]:
    """
    Obtener resumen de routers para logging/debugging

    Returns:
        Resumen de configuración de routers
    """
    return {
        "total_configured": len(routers_config),
        "successfully_registered": registered_routers,
        "failed_registrations": len(routers_config) - registered_routers,
        "prefixes_configured": [
            config.get("prefix") for config in routers_config
        ],  # Added .get for safety
        # Renamed for clarity
        "successfully_imported_modules": successfully_imported,
        "failed_module_imports": failed_imports,  # Renamed for clarity
    }


# =============================================================================
# EXPORTS PRINCIPALES
# =============================================================================
__all__ = [
    # Router principal (compatibilidad con versión original)
    "api_router",
    "available_routes",
    "get_routes_info",
    # Nuevas funciones
    "get_api_router",
    "get_api_health",
    "get_api_statistics",
    "get_router_summary",
    "get_enhanced_feature_analysis",
    # Configuración
    "routers_config",
    "registered_routers",
    "successfully_imported",
    "failed_imports",
]

# =============================================================================
# LOG DE INICIALIZACIÓN
# =============================================================================
logger.info("🏆 Sistema de rutas API inicializado.")
total_configured_routers = len(routers_config)  # Renamed for clarity
logger.info(
    "📊 Resumen de registro: %s/%s routers registrados exitosamente.",
    registered_routers,
    total_configured_routers,
)

if failed_imports:
    logger.warning(
        "⚠️ %s módulo(s) no pudieron ser importados. Ver detalles abajo:",
        len(failed_imports),
    )
    for name, error in failed_imports:
        logger.warning("   ModuleName: %s, Error: %s", name, error)

if registration_errors:
    logger.error(
        (
            "❌ Se encontraron %s error(es) durante el registro de routers. "
            "Ver detalles abajo:"
        ),
        len(registration_errors),
    )
    for error_detail in registration_errors:
        logger.error("   RegistrationError: %s", error_detail)

# En modo debug, mostrar información detallada
if os.getenv("NODE_ENV") == "development":
    logger.debug("🔧 Rutas configuradas para registro:")
    for idx, config in enumerate(routers_config):
        # Verificar si el router está registrado (simplificado)
        status_symbol = "✅" if config.get("router") else "❌"
        logger.debug(
            "   %s Router %s: Prefix='%s', Name='%s'",
            status_symbol,
            idx + 1,
            config.get("prefix"),
            config.get("name"),
        )

    if successfully_imported:
        logger.debug("✅ Módulos importados exitosamente:")
        for module_name in successfully_imported:
            logger.debug("   👍 %s", module_name)

    # Detailed failed imports already logged above if any

    logger.debug("🌐 Endpoints de sistema disponibles en el router principal:")
    logger.debug("   🔗 /info - Información general de la API")
    logger.debug("   🔗 /health - Verificación de estado de la API")
    logger.debug("   🔗 /stats - Estadísticas de la API")
    logger.debug("   🔗 /routes - Lista de todas las rutas configuradas")
    logger.debug("   📚 /docs - Interfaz de documentación Swagger UI")
    logger.debug("   📖 /docs-alt - Documentación API alternativa")

    logger.info("🚀 Configuración de Football Analytics API completada.")
