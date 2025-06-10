"""
main.py - Punto de Entrada Principal Football Analytics

Servidor principal que inicializa y ejecuta el sistema completo de Football Analytics.
Incluye API REST, WebSocket, monitoreo en tiempo real y gestión de servicios.

Author: Football Analytics Team
Version: 2.1.0
Date: 2024-06-02
"""

import asyncio
import json
import signal
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import uvicorn
# import websockets  # Comentado temporalmente para resolver error de importación
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.gzip import GZipMiddleware

from app import (
    FootballAnalyticsApp,
    cleanup_app,
    get_app,
    get_health_status,
    initialize_app,
)

# Imports del proyecto
from app.config import (
    FootballAnalyticsConfig,
    get_config,
    print_config_summary,
    validate_config,
)
from app.services.data_collector import DataCollectorService
from app.services.live_tracker import LiveTrackerService
from app.services.odds_calculator import OddsCalculatorService
from app.services.predictor import PredictorService
from app.utils.constants import SUCCESS_MESSAGES
from app.utils.helpers import safe_execution, timing_decorator

# ================================
# CONFIGURACIÓN GLOBAL
# ================================

config: Optional[FootballAnalyticsConfig] = None
football_app: Optional[FootballAnalyticsApp] = None


# ================================
# CONTEXT MANAGER PARA APLICACIÓN
# ================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Context manager para el ciclo de vida de la aplicación"""
    global config, football_app

    try:
        print("🚀 Iniciando Football Analytics...")

        # Cargar y validar configuración
        config = get_config()
        print_config_summary(config)

        errors = validate_config(config)
        if errors:
            print("❌ Errores de configuración:")
            for error in errors:
                print(f"   - {error}")
            raise RuntimeError("Configuración inválida")

        # Inicializar aplicación
        print("🔧 Inicializando servicios...")
        initialize_app()
        football_app = get_app()

        if not football_app:
            raise RuntimeError("Error al inicializar Football Analytics")

        # Inicializar servicios en background
        if config.monitoring.enabled:
            asyncio.create_task(start_monitoring())

        # Inicializar WebSocket server
        # asyncio.create_task(start_websocket_server())  # Comentado temporalmente

        print("✅ Football Analytics iniciado correctamente")
        print(f"🌐 API disponible en: http://{config.web.host}:{config.web.port}")
        print(
            f"📡 WebSocket disponible en: ws://{config.web.websocket_host}:{config.web.websocket_port}"
        )

        yield

    except Exception as e:
        print(f"❌ Error durante el inicio: {e}")
        raise
    finally:
        print("🔄 Cerrando Football Analytics...")
        if football_app:
            await cleanup_app()
        print("✅ Football Analytics cerrado correctamente")


# ================================
# CREACIÓN DE LA APP FASTAPI
# ================================

app = FastAPI(
    title="Football Analytics API",
    description="Sistema avanzado de análisis y predicción deportiva",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ================================
# MIDDLEWARE
# ================================


@app.middleware("http")
async def add_process_time_header(request, call_next):
    """Middleware para medir tiempo de respuesta"""
    start_time = datetime.now()
    response = await call_next(request)
    process_time = (datetime.now() - start_time).total_seconds()
    response.headers["X-Process-Time"] = str(process_time)
    return response


# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción usar orígenes específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gzip Middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)


# ================================
# DEPENDENCIAS
# ================================


def get_predictor() -> PredictorService:
    """Dependencia para obtener el servicio de predicción"""
    if not football_app:
        raise HTTPException(status_code=503, detail="Servicio no disponible")
    return football_app.services.predictor


def get_data_collector() -> DataCollectorService:
    """Dependencia para obtener el recolector de datos"""
    if not football_app:
        raise HTTPException(status_code=503, detail="Servicio no disponible")
    return football_app.services.data_collector


def get_odds_calculator() -> OddsCalculatorService:
    """Dependencia para obtener el calculador de cuotas"""
    if not football_app:
        raise HTTPException(status_code=503, detail="Servicio no disponible")
    return football_app.services.odds_calculator


def get_live_tracker() -> LiveTrackerService:
    """Dependencia para obtener el tracker en vivo"""
    if not football_app:
        raise HTTPException(status_code=503, detail="Servicio no disponible")
    return football_app.services.live_tracker


# ================================
# RUTAS PRINCIPALES DE LA API
# ================================


@app.get("/")
async def root():
    """Endpoint raíz con información del sistema"""
    return {
        "message": "🚀 Football Analytics API",
        "version": "2.1.0",
        "status": "operational",
        "docs": "/docs",
        "health": "/health",
        "websocket": (
            f"ws://{config.web.websocket_host}:{config.web.websocket_port}"
            if config
            else None
        ),
    }


@app.get("/health")
async def health_check():
    """Health check del sistema"""
    try:
        health_status = get_health_status()

        # Determinar estado general
        all_healthy = all(
            component.get("status") == "healthy"
            for component in health_status.get("components", {}).values()
        )

        status_code = 200 if all_healthy else 503
        health_status["overall_status"] = "healthy" if all_healthy else "unhealthy"

        return JSONResponse(content=health_status, status_code=status_code)

    except Exception as e:
        return JSONResponse(
            content={
                "overall_status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            },
            status_code=503,
        )


@app.get("/config")
async def get_system_config():
    """Obtiene configuración del sistema (sin datos sensibles)"""
    if not config:
        raise HTTPException(status_code=503, detail="Configuración no disponible")

    safe_config = {
        "app_name": config.app_name,
        "version": config.version,
        "environment": config.environment.value,
        "debug": config.debug,
        "api": {
            "football_data_configured": bool(config.api.football_data_api_key),
            "rate_limits": {
                "football_data": config.api.football_data_rate_limit,
                "rapidapi": config.api.rapidapi_rate_limit,
            },
        },
        "ml": {
            "default_model": config.ml.default_model.value,
            "available_models": [model.value for model in config.ml.available_models],
            "min_accuracy": config.ml.min_accuracy,
        },
        "cache": {
            "enabled": config.cache.enabled,
            "backend": config.cache.backend,
        },
    }

    return safe_config


# ================================
# RUTAS DE PREDICCIÓN
# ================================


@app.post("/predict/match")
@timing_decorator
async def predict_match(
    home_team: str,
    away_team: str,
    league: str = "PL",
    match_date: Optional[str] = None,
    predictor: PredictorService = Depends(get_predictor),
):
    """Predice el resultado de un partido"""
    try:
        prediction = await predictor.predict_match_async(
            home_team=home_team,
            away_team=away_team,
            league=league,
            match_date=match_date,
        )

        return {
            "success": True,
            "message": SUCCESS_MESSAGES["PREDICTION_CREATED"],
            "data": prediction,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predicción: {str(e)}")


@app.post("/predict/multiple")
async def predict_multiple_matches(
    matches: list,
    predictor: PredictorService = Depends(get_predictor),
    background_tasks: BackgroundTasks = None,
):
    """Predice múltiples partidos"""
    try:
        predictions = []

        for match in matches:
            prediction = await predictor.predict_match_async(
                home_team=match.get("home_team"),
                away_team=match.get("away_team"),
                league=match.get("league", "PL"),
                match_date=match.get("match_date"),
            )
            predictions.append(prediction)

        return {
            "success": True,
            "message": f"Predicciones generadas para {len(predictions)} partidos",
            "data": predictions,
            "count": len(predictions),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error en predicciones múltiples: {str(e)}",
        )


# ================================
# RUTAS DE DATOS
# ================================


@app.get("/data/teams/{league}")
async def get_teams_by_league(
    league: str,
    data_collector: DataCollectorService = Depends(get_data_collector),
):
    """Obtiene equipos de una liga específica"""
    try:
        teams = await data_collector.get_teams_by_league(league)

        return {
            "success": True,
            "data": teams,
            "league": league,
            "count": len(teams),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error obteniendo equipos: {str(e)}"
        )


@app.get("/data/matches/recent")
async def get_recent_matches(
    league: str = "PL",
    days: int = 7,
    data_collector: DataCollectorService = Depends(get_data_collector),
):
    """Obtiene partidos recientes"""
    try:
        matches = await data_collector.get_recent_matches(league, days)

        return {
            "success": True,
            "data": matches,
            "league": league,
            "days": days,
            "count": len(matches),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error obteniendo partidos: {str(e)}"
        )


@app.post("/data/collect")
async def trigger_data_collection(
    league: str = "PL",
    force_update: bool = False,
    background_tasks: BackgroundTasks = None,
    data_collector: DataCollectorService = Depends(get_data_collector),
):
    """Dispara recolección de datos en background"""
    try:
        # Ejecutar en background
        background_tasks.add_task(
            data_collector.collect_league_data, league, force_update
        )

        return {
            "success": True,
            "message": f"Recolección de datos iniciada para {league}",
            "league": league,
            "force_update": force_update,
            "status": "in_progress",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error iniciando recolección: {str(e)}"
        )


# ================================
# RUTAS DE CUOTAS
# ================================


@app.get("/odds/{home_team}/{away_team}")
async def get_match_odds(
    home_team: str,
    away_team: str,
    league: str = "PL",
    odds_calculator: OddsCalculatorService = Depends(get_odds_calculator),
):
    """Obtiene cuotas para un partido específico"""
    try:
        odds_data = await odds_calculator.get_match_odds(home_team, away_team, league)

        return {
            "success": True,
            "data": odds_data,
            "match": f"{home_team} vs {away_team}",
            "league": league,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error obteniendo cuotas: {str(e)}"
        )


@app.post("/odds/analyze")
async def analyze_odds(
    odds_data: dict,
    odds_calculator: OddsCalculatorService = Depends(get_odds_calculator),
):
    """Analiza cuotas para encontrar valor y arbitraje"""
    try:
        analysis = await odds_calculator.analyze_odds(odds_data)

        return {
            "success": True,
            "data": analysis,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error analizando cuotas: {str(e)}"
        )


# ================================
# RUTAS DE MONITOREO EN VIVO
# ================================


@app.get("/live/matches")
async def get_live_matches(
    live_tracker: LiveTrackerService = Depends(get_live_tracker),
):
    """Obtiene partidos en vivo"""
    try:
        live_matches = await live_tracker.get_live_matches()

        return {
            "success": True,
            "data": live_matches,
            "count": len(live_matches),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error obteniendo partidos en vivo: {str(e)}",
        )


@app.post("/live/track/{match_id}")
async def start_tracking_match(
    match_id: str, live_tracker: LiveTrackerService = Depends(get_live_tracker)
):
    """Inicia el tracking de un partido específico"""
    try:
        result = await live_tracker.start_tracking(match_id)

        return {
            "success": True,
            "message": f"Tracking iniciado para partido {match_id}",
            "data": result,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error iniciando tracking: {str(e)}"
        )


# ================================
# SERVIDOR WEBSOCKET
# ================================

connected_clients = set()


async def websocket_handler(websocket, path):
    """Handler para conexiones WebSocket"""
    connected_clients.add(websocket)
    try:
        print(f"🔌 Cliente WebSocket conectado. Total: {len(connected_clients)}")

        # Enviar mensaje de bienvenida
        welcome_message = {
            "type": "welcome",
            "message": "Conectado a Football Analytics",
            "timestamp": datetime.now().isoformat(),
            "server_version": "2.1.0",
        }
        await websocket.send(json.dumps(welcome_message))

        # Mantener conexión activa
        async for message in websocket:
            try:
                data = json.loads(message)
                await handle_websocket_message(websocket, data)
            except json.JSONDecodeError:
                error_msg = {
                    "type": "error",
                    "message": "Formato de mensaje inválido",
                    "timestamp": datetime.now().isoformat(),
                }
                await websocket.send(json.dumps(error_msg))

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected_clients.remove(websocket)
        print(f"🔌 Cliente WebSocket desconectado. Total: {len(connected_clients)}")


async def handle_websocket_message(websocket, data):
    """Maneja mensajes recibidos por WebSocket"""
    message_type = data.get("type")

    if message_type == "ping":
        response = {"type": "pong", "timestamp": datetime.now().isoformat()}
        await websocket.send(json.dumps(response))

    elif message_type == "subscribe_predictions":
        # Suscribir a predicciones en tiempo real
        response = {
            "type": "subscription_confirmed",
            "subscription": "predictions",
            "timestamp": datetime.now().isoformat(),
        }
        await websocket.send(json.dumps(response))

    elif message_type == "get_live_matches":
        # Enviar partidos en vivo
        if football_app:
            live_matches = await football_app.services.live_tracker.get_live_matches()
            response = {
                "type": "live_matches",
                "data": live_matches,
                "timestamp": datetime.now().isoformat(),
            }
            await websocket.send(json.dumps(response))


async def broadcast_to_clients(message):
    """Envía mensaje a todos los clientes conectados"""
    if connected_clients:
        disconnected = set()
        for client in connected_clients.copy():
            try:
                await client.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)

        # Remover clientes desconectados
        connected_clients -= disconnected


async def start_websocket_server():
    """Inicia el servidor WebSocket"""
    if not config:
        return

    host = config.web.websocket_host
    port = config.web.websocket_port

    try:
        print(f"🚀 Iniciando servidor WebSocket en {host}:{port}")
        server = await websockets.serve(websocket_handler, host, port)
        print(f"✅ Servidor WebSocket iniciado en ws://{host}:{port}")

        # Mantener el servidor activo
        await server.wait_closed()

    except Exception as e:
        print(f"❌ Error iniciando servidor WebSocket: {e}")


# ================================
# MONITOREO EN BACKGROUND
# ================================


async def start_monitoring():
    """Inicia tareas de monitoreo en background"""
    print("📊 Iniciando monitoreo del sistema...")

    # Monitoreo de salud cada minuto
    asyncio.create_task(health_monitor())

    # Broadcast de estadísticas cada 30 segundos
    asyncio.create_task(stats_broadcaster())

    # Limpieza de cache cada hora
    asyncio.create_task(cache_cleaner())


async def health_monitor():
    """Monitorea la salud del sistema"""
    while True:
        try:
            health = get_health_status()

            # Si hay problemas, enviar alerta
            if any(
                component.get("status") != "healthy"
                for component in health.get("components", {}).values()
            ):
                alert_message = {
                    "type": "system_alert",
                    "level": "warning",
                    "message": "Componentes del sistema con problemas",
                    "health_status": health,
                    "timestamp": datetime.now().isoformat(),
                }
                await broadcast_to_clients(alert_message)

            await asyncio.sleep(60)  # Cada minuto

        except Exception as e:
            print(f"❌ Error en health monitor: {e}")
            await asyncio.sleep(60)


async def stats_broadcaster():
    """Envía estadísticas a clientes WebSocket"""
    while True:
        try:
            if connected_clients and football_app:
                stats = {
                    "type": "system_stats",
                    "data": {
                        "connected_clients": len(connected_clients),
                        "uptime": (
                            datetime.now() - football_app.start_time
                        ).total_seconds(),
                        "memory_usage": "N/A",  # Implementar si es necesario
                        "predictions_count": "N/A",  # Implementar contador
                        "active_matches": "N/A",  # Implementar contador
                    },
                    "timestamp": datetime.now().isoformat(),
                }
                await broadcast_to_clients(stats)

            await asyncio.sleep(30)  # Cada 30 segundos

        except Exception as e:
            print(f"❌ Error en stats broadcaster: {e}")
            await asyncio.sleep(30)


async def cache_cleaner():
    """Limpia cache periódicamente"""
    while True:
        try:
            if football_app and hasattr(football_app, "cache"):
                await football_app.cache.cleanup_expired()

            await asyncio.sleep(3600)  # Cada hora

        except Exception as e:
            print(f"❌ Error en cache cleaner: {e}")
            await asyncio.sleep(3600)


# ================================
# MANEJADORES DE SEÑALES
# ================================


def signal_handler(signum, frame):
    """Maneja señales del sistema para cierre graceful"""
    print(f"📶 Señal {signum} recibida. Cerrando aplicación...")

    # Aquí se podría implementar cierre graceful
    # Por ahora, usar el context manager de lifespan

    sys.exit(0)


# Registrar manejadores de señales
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ================================
# FUNCIÓN PRINCIPAL
# ================================


@safe_execution(default_return=None, log_errors=True)
def main():
    """Función principal de la aplicación"""
    global config

    # Cargar configuración
    config = get_config()

    print(
        f"""
🏈 FOOTBALL ANALYTICS - SERVIDOR PRINCIPAL
{'='*50}
📦 Versión: {config.version}
🌍 Entorno: {config.environment.value}
🐛 Debug: {config.debug}
🌐 API: http://{config.web.host}:{config.web.port}
📡 WebSocket: ws://{config.web.websocket_host}:{config.web.websocket_port}
📊 Docs: http://{config.web.host}:{config.web.port}/docs
{'='*50}
    """
    )

    # Configurar uvicorn
    uvicorn_config = {
        "app": "app.main:app",
        "host": config.web.host,
        "port": config.web.port,
        "reload": config.web.reload,
        "log_level": "info" if not config.debug else "debug",
        "access_log": True,
        "use_colors": True,
    }

    # Ejecutar servidor
    try:
        uvicorn.run(**uvicorn_config)
    except KeyboardInterrupt:
        print("\n🔄 Cerrando servidor...")
    except Exception as e:
        print(f"❌ Error ejecutando servidor: {e}")
        return 1

    return 0


# ================================
# PUNTO DE ENTRADA
# ================================

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)  # PUNTO DE ENTRADA
# ================================

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
