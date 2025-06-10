# backend/app/api/routes/live.py
"""
Endpoints para datos en tiempo real, WebSockets y seguimiento de partidos en vivo.
Incluye actualizaciones en tiempo real, notificaciones push y streaming de datos.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

import redis
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from sqlalchemy.orm import Session

from ...config import settings

# Imports internos
from ...database.connection import get_db
from ...database.models import League, Match, MatchEvent, Team
from ...services.data_collector import DataCollector
from ...services.live_tracker import LiveTracker

# Configurar logging
logger = logging.getLogger(__name__)

# Crear router
router = APIRouter()

# Configurar Redis para tiempo real
redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)

# Instanciar servicios
data_collector = DataCollector()
live_tracker = LiveTracker()

# =====================================================
# GESTIÓN DE CONEXIONES WEBSOCKET
# =====================================================


class ConnectionManager:
    """
    Gestor de conexiones WebSocket para tiempo real.
    Maneja múltiples conexiones y broadcasting.
    """

    def __init__(self):
        # Conexiones activas: {connection_id: websocket}
        self.active_connections: Dict[str, WebSocket] = {}

        # Suscripciones por partido: {match_id: set(connection_ids)}
        self.match_subscriptions: Dict[int, Set[str]] = {}

        # Suscripciones generales: set(connection_ids)
        self.general_subscriptions: Set[str] = set()

        # Metadata de conexiones
        self.connection_metadata: Dict[str, Dict] = {}

    async def connect(
        self, websocket: WebSocket, connection_id: str = None
    ) -> str:
        """Conectar un nuevo WebSocket."""
        try:
            await websocket.accept()

            if not connection_id:
                connection_id = str(uuid.uuid4())

            self.active_connections[connection_id] = websocket
            self.connection_metadata[connection_id] = {
                "connected_at": datetime.now().isoformat(),
                "subscriptions": [],
                "user_agent": websocket.headers.get("user-agent", "unknown"),
            }

            logger.info(f"✅ Nueva conexión WebSocket: {connection_id}")
            return connection_id

        except Exception as e:
            logger.error(f"❌ Error conectando WebSocket: {e}")
            raise

    def disconnect(self, connection_id: str):
        """Desconectar un WebSocket."""
        try:
            # Remover de conexiones activas
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]

            # Remover de suscripciones generales
            self.general_subscriptions.discard(connection_id)

            # Remover de suscripciones de partidos
            for match_id, subscribers in self.match_subscriptions.items():
                subscribers.discard(connection_id)

            # Limpiar suscripciones vacías
            self.match_subscriptions = {
                match_id: subs
                for match_id, subs in self.match_subscriptions.items()
                if subs
            }

            # Remover metadata
            if connection_id in self.connection_metadata:
                del self.connection_metadata[connection_id]

            logger.info(f"🔌 Conexión WebSocket desconectada: {connection_id}")

        except Exception as e:
            logger.error(f"❌ Error desconectando WebSocket: {e}")

    async def send_personal_message(self, message: Dict, connection_id: str):
        """Enviar mensaje a una conexión específica."""
        try:
            if connection_id in self.active_connections:
                websocket = self.active_connections[connection_id]
                await websocket.send_text(json.dumps(message))

        except Exception as e:
            logger.error(f"❌ Error enviando mensaje personal: {e}")
            # Remover conexión problemática
            self.disconnect(connection_id)

    async def broadcast_to_match(self, message: Dict, match_id: int):
        """Enviar mensaje a todos los suscritos a un partido."""
        try:
            if match_id in self.match_subscriptions:
                subscribers = list(self.match_subscriptions[match_id])

                for connection_id in subscribers:
                    await self.send_personal_message(message, connection_id)

                logger.info(
                    f"📢 Broadcast a {len(subscribers)} conexiones (partido {match_id})"
                )

        except Exception as e:
            logger.error(f"❌ Error en broadcast a partido: {e}")

    async def broadcast_general(self, message: Dict):
        """Enviar mensaje a todas las conexiones generales."""
        try:
            subscribers = list(self.general_subscriptions)

            for connection_id in subscribers:
                await self.send_personal_message(message, connection_id)

            logger.info(
                f"📢 Broadcast general a {len(subscribers)} conexiones"
            )

        except Exception as e:
            logger.error(f"❌ Error en broadcast general: {e}")

    def subscribe_to_match(self, connection_id: str, match_id: int):
        """Suscribir conexión a actualizaciones de un partido."""
        try:
            if match_id not in self.match_subscriptions:
                self.match_subscriptions[match_id] = set()

            self.match_subscriptions[match_id].add(connection_id)

            # Actualizar metadata
            if connection_id in self.connection_metadata:
                self.connection_metadata[connection_id][
                    "subscriptions"
                ].append(f"match_{match_id}")

            logger.info(
                f"🔔 Conexión {connection_id} suscrita a partido {match_id}"
            )

        except Exception as e:
            logger.error(f"❌ Error suscribiendo a partido: {e}")

    def unsubscribe_from_match(self, connection_id: str, match_id: int):
        """Desuscribir conexión de un partido."""
        try:
            if match_id in self.match_subscriptions:
                self.match_subscriptions[match_id].discard(connection_id)

            # Actualizar metadata
            if connection_id in self.connection_metadata:
                subscriptions = self.connection_metadata[connection_id][
                    "subscriptions"
                ]
                if f"match_{match_id}" in subscriptions:
                    subscriptions.remove(f"match_{match_id}")

            logger.info(
                f"🔕 Conexión {connection_id} desuscrita de partido {match_id}"
            )

        except Exception as e:
            logger.error(f"❌ Error desuscribiendo de partido: {e}")

    def subscribe_to_general(self, connection_id: str):
        """Suscribir a actualizaciones generales."""
        self.general_subscriptions.add(connection_id)

        if connection_id in self.connection_metadata:
            self.connection_metadata[connection_id]["subscriptions"].append(
                "general"
            )

    def get_stats(self) -> Dict:
        """Obtener estadísticas de conexiones."""
        return {
            "total_connections": len(self.active_connections),
            "general_subscriptions": len(self.general_subscriptions),
            "match_subscriptions": {
                str(match_id): len(subs)
                for match_id, subs in self.match_subscriptions.items()
            },
            "active_connection_ids": list(self.active_connections.keys()),
        }


# Instancia global del gestor de conexiones
manager = ConnectionManager()

# =====================================================
# ENDPOINTS REST API
# =====================================================


@router.get("/", summary="Obtener estado de datos en tiempo real")
async def get_live_status(db: Session = Depends(get_db)):
    """
    Obtener estado actual del sistema de tiempo real.

    **Retorna:**
    - Conexiones WebSocket activas
    - Partidos en vivo
    - Últimas actualizaciones
    - Estado de servicios
    """
    try:
        # Obtener partidos en vivo
        live_matches = (
            db.query(Match)
            .filter(Match.status == "live")
            .join(Team, Match.home_team_id == Team.id)
            .join(Team, Match.away_team_id == Team.id, isouter=True)
            .all()
        )

        live_matches_data = []
        for match in live_matches:
            match_data = {
                "id": match.id,
                "home_team": (
                    match.home_team.name if match.home_team else "TBD"
                ),
                "away_team": (
                    match.away_team.name if match.away_team else "TBD"
                ),
                "home_score": match.home_score,
                "away_score": match.away_score,
                "minutes_played": match.minutes_played,
                "league": match.league.name if match.league else "Unknown",
                "last_updated": (
                    match.updated_at.isoformat() if match.updated_at else None
                ),
            }
            live_matches_data.append(match_data)

        # Estadísticas de conexiones
        connection_stats = manager.get_stats()

        # Estado de Redis
        try:
            redis_info = redis_client.ping()
            redis_status = "connected" if redis_info else "disconnected"
        except:
            redis_status = "error"

        return {
            "success": True,
            "data": {
                "live_matches": live_matches_data,
                "total_live_matches": len(live_matches_data),
                "websocket_connections": connection_stats,
                "services": {
                    "redis": redis_status,
                    "live_tracker": "active",
                    "data_collector": "active",
                },
                "last_update": datetime.now().isoformat(),
                "system_time": datetime.now().isoformat(),
            },
        }

    except Exception as e:
        logger.error(f"❌ Error obteniendo estado live: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo estado en tiempo real: {str(e)}",
        )


@router.get("/matches", summary="Obtener partidos en vivo")
async def get_live_matches(
    include_events: bool = False,
    include_stats: bool = False,
    db: Session = Depends(get_db),
):
    """
    Obtener todos los partidos que están siendo jugados actualmente.

    **Parámetros:**
    - `include_events`: Incluir eventos del partido (goles, tarjetas, etc.)
    - `include_stats`: Incluir estadísticas en vivo

    **Retorna:**
    - Lista de partidos en vivo
    - Información detallada de cada partido
    - Eventos y estadísticas opcionales
    """
    try:
        # Query para partidos en vivo
        live_matches = (
            db.query(Match).filter(Match.status == "live").join(League).all()
        )

        matches_data = []

        for match in live_matches:
            # Datos básicos del partido
            match_data = {
                "id": match.id,
                "league": {"id": match.league.id, "name": match.league.name},
                "home_team": {
                    "id": match.home_team_id,
                    "name": match.home_team.name if match.home_team else "TBD",
                },
                "away_team": {
                    "id": match.away_team_id,
                    "name": match.away_team.name if match.away_team else "TBD",
                },
                "score": {"home": match.home_score, "away": match.away_score},
                "time": {
                    "minutes_played": match.minutes_played,
                    "added_time": match.added_time_first_half
                    + match.added_time_second_half,
                    "status": match.status,
                },
                "match_date": match.match_date.isoformat(),
                "last_updated": (
                    match.updated_at.isoformat() if match.updated_at else None
                ),
            }

            # Incluir eventos si se solicita
            if include_events:
                events = (
                    db.query(MatchEvent)
                    .filter(MatchEvent.match_id == match.id)
                    .order_by(MatchEvent.minute.desc())
                    .limit(10)
                    .all()
                )

                match_data["recent_events"] = [
                    {
                        "id": event.id,
                        "type": event.event_type,
                        "detail": event.event_detail,
                        "minute": event.minute,
                        "player": event.player.name if event.player else None,
                        "team": event.team.name if event.team else None,
                    }
                    for event in events
                ]

            # Incluir estadísticas si se solicita
            if include_stats:
                # Obtener desde Redis o calcular
                stats_key = f"match_stats:{match.id}"
                cached_stats = redis_client.get(stats_key)

                if cached_stats:
                    match_data["stats"] = json.loads(cached_stats)
                else:
                    # Generar estadísticas básicas
                    match_data["stats"] = {
                        "possession": {"home": 50, "away": 50},  # Placeholder
                        "shots": {"home": 0, "away": 0},
                        "corners": {"home": 0, "away": 0},
                    }

            matches_data.append(match_data)

        return {
            "success": True,
            "data": {
                "live_matches": matches_data,
                "total_matches": len(matches_data),
                "includes": {"events": include_events, "stats": include_stats},
                "timestamp": datetime.now().isoformat(),
            },
        }

    except Exception as e:
        logger.error(f"❌ Error obteniendo partidos en vivo: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo partidos en vivo: {str(e)}",
        )


@router.get("/match/{match_id}", summary="Obtener partido en vivo específico")
async def get_live_match(match_id: int, db: Session = Depends(get_db)):
    """
    Obtener información detallada de un partido específico en vivo.

    **Incluye:**
    - Estado actual del partido
    - Eventos recientes
    - Estadísticas en tiempo real
    - Timeline completo
    """
    try:
        # Buscar partido
        match = db.query(Match).filter(Match.id == match_id).first()

        if not match:
            raise HTTPException(
                status_code=404, detail="Partido no encontrado"
            )

        # Obtener eventos del partido
        events = (
            db.query(MatchEvent)
            .filter(MatchEvent.match_id == match_id)
            .order_by(MatchEvent.minute.asc())
            .all()
        )

        events_data = [
            {
                "id": event.id,
                "type": event.event_type,
                "detail": event.event_detail,
                "minute": event.minute,
                "added_time": event.added_time,
                "description": event.description,
                "player": {
                    "id": event.player.id if event.player else None,
                    "name": event.player.name if event.player else None,
                },
                "team": {
                    "id": event.team.id if event.team else None,
                    "name": event.team.name if event.team else None,
                },
            }
            for event in events
        ]

        # Obtener estadísticas desde Redis
        stats_key = f"match_stats:{match_id}"
        live_stats = redis_client.get(stats_key)

        if live_stats:
            stats = json.loads(live_stats)
        else:
            stats = {
                "possession": {"home": 50, "away": 50},
                "shots": {"home": 0, "away": 0},
                "shots_on_target": {"home": 0, "away": 0},
                "corners": {"home": 0, "away": 0},
                "fouls": {"home": 0, "away": 0},
                "yellow_cards": {"home": 0, "away": 0},
                "red_cards": {"home": 0, "away": 0},
            }

        return {
            "success": True,
            "data": {
                "match": {
                    "id": match.id,
                    "home_team": {
                        "id": match.home_team_id,
                        "name": (
                            match.home_team.name if match.home_team else "TBD"
                        ),
                    },
                    "away_team": {
                        "id": match.away_team_id,
                        "name": (
                            match.away_team.name if match.away_team else "TBD"
                        ),
                    },
                    "score": {
                        "home": match.home_score,
                        "away": match.away_score,
                    },
                    "time": {
                        "minutes_played": match.minutes_played,
                        "added_time_first_half": match.added_time_first_half,
                        "added_time_second_half": match.added_time_second_half,
                        "status": match.status,
                    },
                    "league": (
                        {"id": match.league.id, "name": match.league.name}
                        if match.league
                        else None
                    ),
                },
                "events": events_data,
                "stats": stats,
                "subscribers": len(
                    manager.match_subscriptions.get(match_id, set())
                ),
                "last_updated": datetime.now().isoformat(),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error obteniendo partido en vivo: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error obteniendo partido: {str(e)}"
        )


@router.post("/match/{match_id}/simulate", summary="Simular evento en partido")
async def simulate_match_event(
    match_id: int,
    event_type: str,
    minute: int,
    team_id: int,
    player_id: Optional[int] = None,
    description: Optional[str] = None,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db),
):
    """
    Simular un evento en un partido (para testing y demos).

    **Eventos soportados:**
    - goal, yellow_card, red_card, substitution, penalty

    **Parámetros:**
    - `event_type`: Tipo de evento
    - `minute`: Minuto del evento
    - `team_id`: ID del equipo
    - `player_id`: ID del jugador (opcional)
    - `description`: Descripción adicional
    """
    try:
        # Verificar que el partido existe
        match = db.query(Match).filter(Match.id == match_id).first()
        if not match:
            raise HTTPException(
                status_code=404, detail="Partido no encontrado"
            )

        # Crear evento
        new_event = MatchEvent(
            match_id=match_id,
            team_id=team_id,
            player_id=player_id,
            event_type=event_type,
            minute=minute,
            description=description or f"{event_type} en minuto {minute}",
        )

        db.add(new_event)

        # Actualizar score si es gol
        if event_type == "goal":
            if team_id == match.home_team_id:
                match.home_score += 1
            elif team_id == match.away_team_id:
                match.away_score += 1

        # Actualizar minutos jugados
        if minute > match.minutes_played:
            match.minutes_played = minute

        db.commit()

        # Crear mensaje para WebSocket
        event_message = {
            "type": "match_event",
            "match_id": match_id,
            "event": {
                "id": new_event.id,
                "type": event_type,
                "minute": minute,
                "team_id": team_id,
                "player_id": player_id,
                "description": description,
            },
            "score": {"home": match.home_score, "away": match.away_score},
            "timestamp": datetime.now().isoformat(),
        }

        # Enviar via WebSocket en background
        background_tasks.add_task(
            manager.broadcast_to_match, event_message, match_id
        )

        return {
            "success": True,
            "data": {
                "event_id": new_event.id,
                "message": f"Evento {event_type} simulado en minuto {minute}",
                "updated_score": {
                    "home": match.home_score,
                    "away": match.away_score,
                },
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error simulando evento: {e}")
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"Error simulando evento: {str(e)}"
        )


# =====================================================
# WEBSOCKET ENDPOINTS
# =====================================================


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket principal para actualizaciones en tiempo real.

    **Protocolos soportados:**
    - Suscripción a partidos específicos
    - Actualizaciones generales
    - Ping/Pong para keep-alive

    **Mensajes de entrada:**
    ```json
    {
        "action": "subscribe_match",
        "match_id": 123
    }
    ```

    **Mensajes de salida:**
    ```json
    {
        "type": "match_event",
        "match_id": 123,
        "data": {...}
    }
    ```
    """
    connection_id = None

    try:
        # Conectar WebSocket
        connection_id = await manager.connect(websocket)

        # Enviar mensaje de bienvenida
        welcome_message = {
            "type": "connection_established",
            "connection_id": connection_id,
            "message": "Conectado al sistema de tiempo real",
            "available_actions": [
                "subscribe_match",
                "unsubscribe_match",
                "subscribe_general",
                "ping",
            ],
            "timestamp": datetime.now().isoformat(),
        }

        await manager.send_personal_message(welcome_message, connection_id)

        # Loop principal de manejo de mensajes
        while True:
            try:
                # Recibir mensaje del cliente
                data = await websocket.receive_text()
                message = json.loads(data)

                action = message.get("action")

                if action == "subscribe_match":
                    # Suscribirse a un partido específico
                    match_id = message.get("match_id")
                    if match_id:
                        manager.subscribe_to_match(connection_id, match_id)

                        response = {
                            "type": "subscription_confirmed",
                            "action": "subscribe_match",
                            "match_id": match_id,
                            "message": f"Suscrito a partido {match_id}",
                            "timestamp": datetime.now().isoformat(),
                        }
                        await manager.send_personal_message(
                            response, connection_id
                        )

                elif action == "unsubscribe_match":
                    # Desuscribirse de un partido
                    match_id = message.get("match_id")
                    if match_id:
                        manager.unsubscribe_from_match(connection_id, match_id)

                        response = {
                            "type": "unsubscription_confirmed",
                            "action": "unsubscribe_match",
                            "match_id": match_id,
                            "message": f"Desuscrito de partido {match_id}",
                            "timestamp": datetime.now().isoformat(),
                        }
                        await manager.send_personal_message(
                            response, connection_id
                        )

                elif action == "subscribe_general":
                    # Suscribirse a actualizaciones generales
                    manager.subscribe_to_general(connection_id)

                    response = {
                        "type": "subscription_confirmed",
                        "action": "subscribe_general",
                        "message": "Suscrito a actualizaciones generales",
                        "timestamp": datetime.now().isoformat(),
                    }
                    await manager.send_personal_message(
                        response, connection_id
                    )

                elif action == "ping":
                    # Responder a ping
                    pong_response = {
                        "type": "pong",
                        "timestamp": datetime.now().isoformat(),
                        "connection_id": connection_id,
                    }
                    await manager.send_personal_message(
                        pong_response, connection_id
                    )

                elif action == "get_status":
                    # Enviar estado de la conexión
                    status_response = {
                        "type": "status",
                        "connection_id": connection_id,
                        "subscriptions": manager.connection_metadata.get(
                            connection_id, {}
                        ).get("subscriptions", []),
                        "connected_since": manager.connection_metadata.get(
                            connection_id, {}
                        ).get("connected_at"),
                        "total_connections": len(manager.active_connections),
                        "timestamp": datetime.now().isoformat(),
                    }
                    await manager.send_personal_message(
                        status_response, connection_id
                    )

                else:
                    # Acción no reconocida
                    error_response = {
                        "type": "error",
                        "message": f"Acción no reconocida: {action}",
                        "available_actions": [
                            "subscribe_match",
                            "unsubscribe_match",
                            "subscribe_general",
                            "ping",
                            "get_status",
                        ],
                        "timestamp": datetime.now().isoformat(),
                    }
                    await manager.send_personal_message(
                        error_response, connection_id
                    )

            except json.JSONDecodeError:
                # Error de JSON
                error_response = {
                    "type": "error",
                    "message": "Formato JSON inválido",
                    "timestamp": datetime.now().isoformat(),
                }
                await manager.send_personal_message(
                    error_response, connection_id
                )

            except Exception as e:
                logger.error(f"❌ Error procesando mensaje WebSocket: {e}")
                error_response = {
                    "type": "error",
                    "message": f"Error interno: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                }
                await manager.send_personal_message(
                    error_response, connection_id
                )

    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket desconectado: {connection_id}")

    except Exception as e:
        logger.error(f"❌ Error en WebSocket: {e}")

    finally:
        # Limpiar conexión
        if connection_id:
            manager.disconnect(connection_id)


@router.websocket("/ws/match/{match_id}")
async def websocket_match_specific(websocket: WebSocket, match_id: int):
    """
    WebSocket específico para un partido.
    Automáticamente suscribe a actualizaciones de ese partido.
    """
    connection_id = None

    try:
        # Conectar y suscribir automáticamente
        connection_id = await manager.connect(websocket)
        manager.subscribe_to_match(connection_id, match_id)

        # Mensaje de bienvenida específico
        welcome_message = {
            "type": "match_connection_established",
            "connection_id": connection_id,
            "match_id": match_id,
            "message": f"Conectado a actualizaciones del partido {match_id}",
            "timestamp": datetime.now().isoformat(),
        }

        await manager.send_personal_message(welcome_message, connection_id)

        # Mantener conexión activa
        while True:
            try:
                # Escuchar por ping o mensajes del cliente
                data = await websocket.receive_text()
                message = json.loads(data)

                if message.get("action") == "ping":
                    pong_response = {
                        "type": "pong",
                        "match_id": match_id,
                        "timestamp": datetime.now().isoformat(),
                    }
                    await manager.send_personal_message(
                        pong_response, connection_id
                    )

            except asyncio.TimeoutError:
                # Timeout - continuar el loop
                continue

            except json.JSONDecodeError:
                continue

    except WebSocketDisconnect:
        logger.info(
            f"🔌 WebSocket de partido {match_id} desconectado: {connection_id}"
        )

    except Exception as e:
        logger.error(f"❌ Error en WebSocket de partido: {e}")

    finally:
        if connection_id:
            manager.disconnect(connection_id)


# =====================================================
# TAREAS EN BACKGROUND
# =====================================================


async def start_live_updates():
    """
    Tarea en background para enviar actualizaciones periódicas.
    Se ejecuta cada 10 segundos.
    """
    while True:
        try:
            # Obtener timestamp actual
            current_time = datetime.now().isoformat()

            # Mensaje de actualización general
            general_update = {
                "type": "system_heartbeat",
                "message": "Sistema activo",
                "active_connections": len(manager.active_connections),
                "active_matches": len(manager.match_subscriptions),
                "timestamp": current_time,
            }

            # Enviar a suscriptores generales
            await manager.broadcast_general(general_update)

            # Actualizar datos de partidos en vivo desde Redis
            for match_id in manager.match_subscriptions.keys():
                try:
                    # Obtener datos actualizados del partido
                    match_data_key = f"live_match:{match_id}"
                    cached_data = redis_client.get(match_data_key)

                    if cached_data:
                        match_update = json.loads(cached_data)
                        match_update.update(
                            {"type": "match_update", "timestamp": current_time}
                        )

                        await manager.broadcast_to_match(
                            match_update, match_id
                        )

                except Exception as e:
                    logger.error(
                        f"❌ Error actualizando partido {match_id}: {e}"
                    )

            # Esperar 10 segundos antes de la siguiente actualización
            await asyncio.sleep(10)

        except Exception as e:
            logger.error(f"❌ Error en actualizaciones en vivo: {e}")
            await asyncio.sleep(5)  # Esperar menos tiempo si hay error


@router.on_event("startup")
async def startup_live_system():
    """
    Inicializar sistema de tiempo real al arrancar la aplicación.
    """
    try:
        logger.info("🚀 Iniciando sistema de tiempo real...")

        # Limpiar datos antiguos de Redis
        redis_client.flushdb()

        # Iniciar tarea de actualizaciones en background
        asyncio.create_task(start_live_updates())

        logger.info("✅ Sistema de tiempo real iniciado correctamente")

    except Exception as e:
        logger.error(f"❌ Error iniciando sistema de tiempo real: {e}")


# =====================================================
# UTILIDADES Y HELPERS
# =====================================================


async def simulate_live_match_data(match_id: int):
    """
    Simular datos de un partido en vivo (para testing).
    """
    try:
        # Simular datos básicos del partido
        simulated_data = {
            "match_id": match_id,
            "minute": 45 + (datetime.now().second % 45),
            "score": {"home": 1, "away": 0},
            "stats": {
                "possession": {
                    "home": 55 + (datetime.now().second % 20),
                    "away": 45 - (datetime.now().second % 20),
                },
                "shots": {"home": 8, "away": 4},
                "corners": {"home": 3, "away": 1},
            },
            "last_event": {"type": "corner", "minute": 43, "team": "home"},
        }

        # Guardar en Redis
        redis_key = f"live_match:{match_id}"
        redis_client.setex(
            redis_key, 300, json.dumps(simulated_data)
        )  # Expira en 5 minutos

        return simulated_data

    except Exception as e:
        logger.error(f"❌ Error simulando datos de partido: {e}")
        return None


@router.get("/debug/connections", summary="Debug: Ver conexiones activas")
async def debug_connections():
    """
    Endpoint de debug para ver todas las conexiones WebSocket activas.
    Solo para desarrollo y testing.
    """
    try:
        stats = manager.get_stats()

        detailed_connections = {}
        for conn_id, metadata in manager.connection_metadata.items():
            detailed_connections[conn_id] = {
                "connected_at": metadata.get("connected_at"),
                "subscriptions": metadata.get("subscriptions", []),
                "user_agent": metadata.get("user_agent", "unknown"),
                "is_active": conn_id in manager.active_connections,
            }

        return {
            "success": True,
            "data": {
                "summary": stats,
                "detailed_connections": detailed_connections,
                "redis_status": (
                    "connected" if redis_client.ping() else "disconnected"
                ),
                "debug_timestamp": datetime.now().isoformat(),
            },
        }

    except Exception as e:
        logger.error(f"❌ Error en debug de conexiones: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error en debug: {str(e)}"
        )


@router.post("/debug/broadcast", summary="Debug: Enviar mensaje broadcast")
async def debug_broadcast(
    message: str, match_id: Optional[int] = None, general: bool = False
):
    """
    Endpoint de debug para enviar mensajes de prueba via WebSocket.
    """
    try:
        test_message = {
            "type": "debug_message",
            "message": message,
            "sent_by": "debug_endpoint",
            "timestamp": datetime.now().isoformat(),
        }

        if match_id:
            await manager.broadcast_to_match(test_message, match_id)
            target = f"partido {match_id}"
        elif general:
            await manager.broadcast_general(test_message)
            target = "suscriptores generales"
        else:
            return {
                "success": False,
                "error": "Debe especificar match_id o general=true",
            }

        return {
            "success": True,
            "data": {
                "message_sent": message,
                "target": target,
                "timestamp": datetime.now().isoformat(),
            },
        }

    except Exception as e:
        logger.error(f"❌ Error en debug broadcast: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error enviando mensaje: {str(e)}"
        )


@router.get("/health", summary="Health check del sistema de tiempo real")
async def live_health_check():
    """
    Verificar el estado de salud del sistema de tiempo real.
    """
    try:
        health_status = {
            "websocket_manager": "healthy",
            "redis_connection": "unknown",
            "active_connections": len(manager.active_connections),
            "active_subscriptions": len(manager.match_subscriptions),
            "uptime": "unknown",
        }

        # Test Redis
        try:
            redis_client.ping()
            health_status["redis_connection"] = "healthy"
        except:
            health_status["redis_connection"] = "unhealthy"

        # Determinar estado general
        overall_status = "healthy"
        if health_status["redis_connection"] == "unhealthy":
            overall_status = "degraded"

        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "details": health_status,
        }

    except Exception as e:
        logger.error(f"❌ Error en health check: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
        }


# =====================================================
# EVENTOS Y NOTIFICACIONES
# =====================================================


async def send_match_notification(match_id: int, event_type: str, data: Dict):
    """
    Enviar notificación específica de evento de partido.
    """
    try:
        notification = {
            "type": "match_notification",
            "match_id": match_id,
            "event_type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat(),
            "priority": (
                "high"
                if event_type in ["goal", "red_card", "penalty"]
                else "normal"
            ),
        }

        # Enviar a suscriptores del partido
        await manager.broadcast_to_match(notification, match_id)

        # Si es un evento importante, también enviar a suscriptores generales
        if notification["priority"] == "high":
            await manager.broadcast_general(notification)

        logger.info(
            f"📢 Notificación enviada: {event_type} en partido {match_id}"
        )

    except Exception as e:
        logger.error(f"❌ Error enviando notificación: {e}")


class LiveMatchUpdater:
    """
    Clase para manejar actualizaciones de partidos en vivo.
    """

    @staticmethod
    async def update_score(match_id: int, home_score: int, away_score: int):
        """Actualizar marcador de un partido."""
        try:
            update_data = {
                "type": "score_update",
                "match_id": match_id,
                "score": {"home": home_score, "away": away_score},
                "timestamp": datetime.now().isoformat(),
            }

            # Guardar en Redis
            redis_key = f"live_match:{match_id}:score"
            redis_client.setex(redis_key, 3600, json.dumps(update_data))

            # Broadcast via WebSocket
            await manager.broadcast_to_match(update_data, match_id)

        except Exception as e:
            logger.error(f"❌ Error actualizando marcador: {e}")

    @staticmethod
    async def update_time(match_id: int, minute: int, added_time: int = 0):
        """Actualizar tiempo de juego."""
        try:
            time_data = {
                "type": "time_update",
                "match_id": match_id,
                "time": {
                    "minute": minute,
                    "added_time": added_time,
                    "display": f"{minute}'"
                    + (f"+{added_time}" if added_time > 0 else ""),
                },
                "timestamp": datetime.now().isoformat(),
            }

            await manager.broadcast_to_match(time_data, match_id)

        except Exception as e:
            logger.error(f"❌ Error actualizando tiempo: {e}")

    @staticmethod
    async def update_stats(match_id: int, stats: Dict):
        """Actualizar estadísticas del partido."""
        try:
            stats_data = {
                "type": "stats_update",
                "match_id": match_id,
                "stats": stats,
                "timestamp": datetime.now().isoformat(),
            }

            # Guardar en Redis
            redis_key = f"live_match:{match_id}:stats"
            redis_client.setex(redis_key, 3600, json.dumps(stats))

            # Broadcast via WebSocket
            await manager.broadcast_to_match(stats_data, match_id)

        except Exception as e:
            logger.error(f"❌ Error actualizando estadísticas: {e}")


# Instancia del actualizador
live_updater = LiveMatchUpdater()

# =====================================================
# EXPORTAR FUNCIONES PARA USO EXTERNO
# =====================================================

# Estas funciones pueden ser importadas desde otros módulos
__all__ = [
    "manager",
    "send_match_notification",
    "live_updater",
    "simulate_live_match_data",
]
