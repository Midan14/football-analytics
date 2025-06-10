"""
Football Analytics - Live Tracker Service
Sistema de seguimiento en tiempo real de partidos de fútbol
"""

import asyncio
import json
import logging
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import aiohttp
import numpy as np
import websockets
from websockets.exceptions import ConnectionClosed

# WebSocket y notificaciones
from websockets.server import serve

# Configuración real
from .data_collector import get_production_config

logger = logging.getLogger(__name__)


class MatchStatus(Enum):
    """Estados de un partido"""

    SCHEDULED = "scheduled"
    LIVE = "live"
    HALF_TIME = "half_time"
    SECOND_HALF = "second_half"
    FINISHED = "finished"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"
    SUSPENDED = "suspended"


class EventType(Enum):
    """Tipos de eventos en vivo"""

    GOAL = "goal"
    YELLOW_CARD = "yellow_card"
    RED_CARD = "red_card"
    SUBSTITUTION = "substitution"
    PENALTY = "penalty"
    CORNER = "corner"
    OFFSIDE = "offside"
    FOUL = "foul"
    KICK_OFF = "kick_off"
    HALF_TIME_START = "half_time"
    SECOND_HALF_START = "second_half"
    FULL_TIME = "full_time"
    VAR_CHECK = "var_check"


@dataclass
class LiveMatchEvent:
    """Evento en vivo de un partido"""

    event_id: str
    match_id: str
    minute: int
    event_type: EventType
    team: str
    player: Optional[str] = None
    description: str = ""
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class LiveMatchData:
    """Datos en vivo de un partido"""

    match_id: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    minute: int
    status: MatchStatus
    league: str
    season: str

    # Estadísticas en vivo
    home_shots: int = 0
    away_shots: int = 0
    home_shots_on_target: int = 0
    away_shots_on_target: int = 0
    home_possession: float = 50.0
    away_possession: float = 50.0
    home_corners: int = 0
    away_corners: int = 0
    home_fouls: int = 0
    away_fouls: int = 0
    home_yellow_cards: int = 0
    away_yellow_cards: int = 0
    home_red_cards: int = 0
    away_red_cards: int = 0

    # Metadatos
    last_updated: datetime = None
    events: List[LiveMatchEvent] = None

    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()
        if self.events is None:
            self.events = []

    @property
    def is_live(self) -> bool:
        """Verifica si el partido está en vivo"""
        return self.status in [MatchStatus.LIVE, MatchStatus.SECOND_HALF]

    @property
    def total_goals(self) -> int:
        """Total de goles en el partido"""
        return self.home_score + self.away_score


class FootballDataLiveTracker:
    """Tracker usando Football-Data.org API para datos en vivo"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.football-data.org/v4"
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={"X-Auth-Token": self.api_key},
            timeout=aiohttp.ClientTimeout(total=30),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_live_matches(self) -> List[LiveMatchData]:
        """Obtiene partidos en vivo"""
        live_matches = []

        try:
            # Obtener partidos de hoy
            today = datetime.now().strftime("%Y-%m-%d")
            url = f"{self.base_url}/matches"
            params = {"dateFrom": today, "dateTo": today}

            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    logger.error(f"Error API: {response.status}")
                    return []

                data = await response.json()

                for match in data.get("matches", []):
                    # Solo procesar partidos en vivo o que han terminado hoy
                    if match["status"] in ["IN_PLAY", "PAUSED", "FINISHED"]:
                        live_match = self._parse_match_data(match)
                        if live_match:
                            live_matches.append(live_match)

            logger.info(
                f"📡 Encontrados {len(live_matches)} partidos en vivo/recientes"
            )
            return live_matches

        except Exception as e:
            logger.error(f"Error obteniendo partidos en vivo: {e}")
            return []

    def _parse_match_data(self, match_data: Dict) -> Optional[LiveMatchData]:
        """Parsea datos de partido de la API"""
        try:
            # Determinar estado del partido
            status_map = {
                "IN_PLAY": MatchStatus.LIVE,
                "PAUSED": MatchStatus.HALF_TIME,
                "FINISHED": MatchStatus.FINISHED,
                "SCHEDULED": MatchStatus.SCHEDULED,
                "POSTPONED": MatchStatus.POSTPONED,
                "CANCELLED": MatchStatus.CANCELLED,
                "SUSPENDED": MatchStatus.SUSPENDED,
            }

            status = status_map.get(
                match_data["status"], MatchStatus.SCHEDULED
            )

            # Extraer goles
            home_score = match_data["score"]["fullTime"]["home"] or 0
            away_score = match_data["score"]["fullTime"]["away"] or 0

            # Si está en vivo, usar score actual
            if status == MatchStatus.LIVE and match_data["score"]["halfTime"]:
                home_score = match_data["score"]["halfTime"]["home"] or 0
                away_score = match_data["score"]["halfTime"]["away"] or 0

            live_match = LiveMatchData(
                match_id=str(match_data["id"]),
                home_team=match_data["homeTeam"]["name"],
                away_team=match_data["awayTeam"]["name"],
                home_score=home_score,
                away_score=away_score,
                minute=0,  # Football-Data no proporciona minuto exacto
                status=status,
                league=match_data["competition"]["code"],
                season=str(match_data["season"]["startDate"][:4]),
            )

            return live_match

        except Exception as e:
            logger.error(f"Error parseando partido: {e}")
            return None


class LiveMatchDatabase:
    """Base de datos para almacenar datos en vivo"""

    def __init__(self, db_path: str = "live_matches.db"):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Inicializa la base de datos"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Tabla de partidos en vivo
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS live_matches (
                    match_id TEXT PRIMARY KEY,
                    home_team TEXT,
                    away_team TEXT,
                    home_score INTEGER,
                    away_score INTEGER,
                    minute INTEGER,
                    status TEXT,
                    league TEXT,
                    season TEXT,
                    home_possession REAL,
                    away_possession REAL,
                    last_updated TIMESTAMP,
                    match_data JSON
                )
            """
            )

            # Tabla de eventos en vivo
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS live_events (
                    event_id TEXT PRIMARY KEY,
                    match_id TEXT,
                    minute INTEGER,
                    event_type TEXT,
                    team TEXT,
                    player TEXT,
                    description TEXT,
                    timestamp TIMESTAMP,
                    FOREIGN KEY (match_id) REFERENCES live_matches (match_id)
                )
            """
            )

            conn.commit()

    def save_match(self, match: LiveMatchData):
        """Guarda datos de un partido en vivo"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO live_matches
                (match_id, home_team, away_team, home_score, away_score, 
                 minute, status, league, season, home_possession, away_possession,
                 last_updated, match_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    match.match_id,
                    match.home_team,
                    match.away_team,
                    match.home_score,
                    match.away_score,
                    match.minute,
                    match.status.value,
                    match.league,
                    match.season,
                    match.home_possession,
                    match.away_possession,
                    match.last_updated,
                    json.dumps(asdict(match), default=str),
                ),
            )

            conn.commit()

    def save_event(self, event: LiveMatchEvent):
        """Guarda un evento en vivo"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO live_events
                (event_id, match_id, minute, event_type, team, player, 
                 description, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    event.event_id,
                    event.match_id,
                    event.minute,
                    event.event_type.value,
                    event.team,
                    event.player,
                    event.description,
                    event.timestamp,
                ),
            )

            conn.commit()

    def get_live_matches(self) -> List[LiveMatchData]:
        """Obtiene todos los partidos en vivo"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT match_data FROM live_matches 
                WHERE status IN ('live', 'second_half', 'half_time')
                ORDER BY last_updated DESC
            """
            )

            matches = []
            for row in cursor.fetchall():
                try:
                    match_dict = json.loads(row[0])
                    # Reconstruir objeto LiveMatchData
                    match_dict["status"] = MatchStatus(match_dict["status"])
                    matches.append(LiveMatchData(**match_dict))
                except Exception as e:
                    logger.error(f"Error cargando partido: {e}")

            return matches


class WebSocketServer:
    """Servidor WebSocket para notificaciones en tiempo real"""

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.server = None

    async def register_client(
        self, websocket: websockets.WebSocketServerProtocol
    ):
        """Registra un nuevo cliente"""
        self.clients.add(websocket)
        logger.info(f"Cliente conectado. Total: {len(self.clients)}")

        # Enviar datos iniciales
        try:
            initial_data = {
                "type": "connection",
                "message": "Conectado al Live Tracker",
                "timestamp": datetime.now().isoformat(),
            }
            await websocket.send(json.dumps(initial_data))
        except Exception as e:
            logger.error(f"Error enviando datos iniciales: {e}")

    async def unregister_client(
        self, websocket: websockets.WebSocketServerProtocol
    ):
        """Desregistra un cliente"""
        self.clients.discard(websocket)
        logger.info(f"Cliente desconectado. Total: {len(self.clients)}")

    async def broadcast_match_update(self, match: LiveMatchData):
        """Envía actualización de partido a todos los clientes"""
        if not self.clients:
            return

        message = {
            "type": "match_update",
            "match_id": match.match_id,
            "home_team": match.home_team,
            "away_team": match.away_team,
            "home_score": match.home_score,
            "away_score": match.away_score,
            "minute": match.minute,
            "status": match.status.value,
            "timestamp": datetime.now().isoformat(),
        }

        # Enviar a todos los clientes conectados
        disconnected = set()
        for client in self.clients:
            try:
                await client.send(json.dumps(message))
            except ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                logger.error(f"Error enviando a cliente: {e}")
                disconnected.add(client)

        # Limpiar clientes desconectados
        self.clients -= disconnected

    async def broadcast_event(self, event: LiveMatchEvent):
        """Envía evento en vivo a todos los clientes"""
        if not self.clients:
            return

        message = {
            "type": "live_event",
            "event_id": event.event_id,
            "match_id": event.match_id,
            "minute": event.minute,
            "event_type": event.event_type.value,
            "team": event.team,
            "player": event.player,
            "description": event.description,
            "timestamp": event.timestamp.isoformat(),
        }

        # Enviar a todos los clientes
        disconnected = set()
        for client in self.clients:
            try:
                await client.send(json.dumps(message))
            except ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                logger.error(f"Error enviando evento: {e}")
                disconnected.add(client)

        self.clients -= disconnected

    async def handle_client(
        self, websocket: websockets.WebSocketServerProtocol, path: str
    ):
        """Maneja conexión de cliente WebSocket"""
        await self.register_client(websocket)
        try:
            async for message in websocket:
                # Procesar mensajes del cliente si es necesario
                try:
                    data = json.loads(message)
                    logger.info(f"Mensaje de cliente: {data}")
                except json.JSONDecodeError:
                    logger.warning(f"Mensaje inválido: {message}")
        except ConnectionClosed:
            pass
        finally:
            await self.unregister_client(websocket)

    async def start_server(self):
        """Inicia el servidor WebSocket"""
        try:
            self.server = await serve(self.handle_client, self.host, self.port)
            logger.info(
                f"🌐 Servidor WebSocket iniciado en ws://{self.host}:{self.port}"
            )
            return self.server
        except Exception as e:
            logger.error(f"Error iniciando servidor WebSocket: {e}")
            raise

    async def stop_server(self):
        """Detiene el servidor WebSocket"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("🔻 Servidor WebSocket detenido")


class LiveTrackerService:
    """Servicio principal de seguimiento en vivo"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get("football_data_api_key")

        # Componentes
        self.tracker = FootballDataLiveTracker(self.api_key)
        self.database = LiveMatchDatabase(
            config.get("database", {})
            .get("path", "live_matches.db")
            .replace(".db", "_live.db")
        )
        self.websocket_server = WebSocketServer()

        # Estado del servicio
        self.is_running = False
        self.tracked_matches: Dict[str, LiveMatchData] = {}
        self.last_scores: Dict[str, Tuple[int, int]] = {}

        # Configuración de polling
        self.polling_interval = config.get(
            "live_polling_interval", 30
        )  # 30 segundos

        # Callbacks para eventos
        self.event_callbacks: List[Callable] = []

        logger.info("🔴 Live Tracker Service inicializado")

    def add_event_callback(self, callback: Callable):
        """Agrega callback para eventos en vivo"""
        self.event_callbacks.append(callback)

    async def start_tracking(self):
        """Inicia el seguimiento en vivo"""
        if self.is_running:
            logger.warning("Live Tracker ya está ejecutándose")
            return

        self.is_running = True
        logger.info("🚀 Iniciando Live Tracker Service")

        # Iniciar servidor WebSocket
        await self.websocket_server.start_server()

        # Iniciar loop de seguimiento
        tracking_task = asyncio.create_task(self._tracking_loop())

        try:
            await tracking_task
        except Exception as e:
            logger.error(f"Error en tracking loop: {e}")
        finally:
            await self.stop_tracking()

    async def stop_tracking(self):
        """Detiene el seguimiento en vivo"""
        self.is_running = False
        await self.websocket_server.stop_server()
        logger.info("🔻 Live Tracker Service detenido")

    async def _tracking_loop(self):
        """Loop principal de seguimiento"""
        logger.info(
            f"🔄 Loop de tracking iniciado (intervalo: {self.polling_interval}s)"
        )

        while self.is_running:
            try:
                # Obtener partidos en vivo
                async with self.tracker:
                    live_matches = await self.tracker.get_live_matches()

                # Procesar cada partido
                for match in live_matches:
                    await self._process_match_update(match)

                # Esperar antes del siguiente poll
                await asyncio.sleep(self.polling_interval)

            except Exception as e:
                logger.error(f"Error en tracking loop: {e}")
                await asyncio.sleep(self.polling_interval)

    async def _process_match_update(self, match: LiveMatchData):
        """Procesa actualización de un partido"""
        match_id = match.match_id
        previous_match = self.tracked_matches.get(match_id)

        # Detectar cambios importantes
        significant_change = False
        events = []

        if previous_match:
            # Detectar goles
            if match.home_score > previous_match.home_score:
                events.append(
                    LiveMatchEvent(
                        event_id=f"{match_id}_goal_home_{match.home_score}",
                        match_id=match_id,
                        minute=match.minute,
                        event_type=EventType.GOAL,
                        team=match.home_team,
                        description=f"¡GOL! {match.home_team} {match.home_score}-{match.away_score} {match.away_team}",
                    )
                )
                significant_change = True

            if match.away_score > previous_match.away_score:
                events.append(
                    LiveMatchEvent(
                        event_id=f"{match_id}_goal_away_{match.away_score}",
                        match_id=match_id,
                        minute=match.minute,
                        event_type=EventType.GOAL,
                        team=match.away_team,
                        description=f"¡GOL! {match.home_team} {match.home_score}-{match.away_score} {match.away_team}",
                    )
                )
                significant_change = True

            # Detectar cambio de estado
            if match.status != previous_match.status:
                if match.status == MatchStatus.HALF_TIME:
                    events.append(
                        LiveMatchEvent(
                            event_id=f"{match_id}_half_time",
                            match_id=match_id,
                            minute=45,
                            event_type=EventType.HALF_TIME_START,
                            team="",
                            description="Descanso",
                        )
                    )
                elif match.status == MatchStatus.FINISHED:
                    events.append(
                        LiveMatchEvent(
                            event_id=f"{match_id}_full_time",
                            match_id=match_id,
                            minute=90,
                            event_type=EventType.FULL_TIME,
                            team="",
                            description=f"Final: {match.home_team} {match.home_score}-{match.away_score} {match.away_team}",
                        )
                    )
                significant_change = True

        else:
            # Primer tracking de este partido
            if match.is_live:
                events.append(
                    LiveMatchEvent(
                        event_id=f"{match_id}_tracking_start",
                        match_id=match_id,
                        minute=match.minute,
                        event_type=EventType.KICK_OFF,
                        team="",
                        description=f"Siguiendo: {match.home_team} vs {match.away_team}",
                    )
                )
                significant_change = True

        # Actualizar estado interno
        self.tracked_matches[match_id] = match

        # Guardar en base de datos
        self.database.save_match(match)

        # Procesar eventos
        for event in events:
            self.database.save_event(event)
            await self.websocket_server.broadcast_event(event)

            # Ejecutar callbacks
            for callback in self.event_callbacks:
                try:
                    await callback(event)
                except Exception as e:
                    logger.error(f"Error en callback: {e}")

        # Enviar actualización de partido
        if significant_change or previous_match is None:
            await self.websocket_server.broadcast_match_update(match)
            logger.info(
                f"📊 {match.home_team} {match.home_score}-{match.away_score} {match.away_team} ({match.status.value})"
            )

    def get_live_matches(self) -> List[LiveMatchData]:
        """Obtiene partidos actualmente en vivo"""
        return list(self.tracked_matches.values())

    def get_match_by_id(self, match_id: str) -> Optional[LiveMatchData]:
        """Obtiene un partido específico"""
        return self.tracked_matches.get(match_id)


# Funciones de conveniencia
async def create_live_tracker() -> LiveTrackerService:
    """Crea instancia del live tracker con configuración de producción"""
    config = get_production_config()
    if not config or not config.get("football_data_api_key"):
        raise ValueError("❌ Configuración inválida para Live Tracker")

    return LiveTrackerService(config)


async def start_live_tracking():
    """Inicia el sistema de tracking en vivo"""
    tracker = await create_live_tracker()

    # Agregar callback de ejemplo
    async def log_event(event: LiveMatchEvent):
        if event.event_type == EventType.GOAL:
            logger.info(f"🥅 {event.description}")
        elif event.event_type == EventType.FULL_TIME:
            logger.info(f"⏰ {event.description}")

    tracker.add_event_callback(log_event)

    # Iniciar tracking
    await tracker.start_tracking()


if __name__ == "__main__":
    # Sistema de Live Tracking en producción
    async def main():
        print("🔴 Football Analytics - Live Tracker")
        print("Sistema de seguimiento en tiempo real")
        print("=" * 50)

        try:
            # Crear live tracker
            tracker = await create_live_tracker()

            print("✅ Live Tracker configurado")
            print("🌐 Servidor WebSocket: ws://localhost:8765")
            print("📡 Polling cada 30 segundos")
            print("🔄 Iniciando seguimiento...")

            # Agregar callbacks para logging
            async def on_goal(event: LiveMatchEvent):
                if event.event_type == EventType.GOAL:
                    print(f"🥅 ¡GOL! {event.description}")

            async def on_match_end(event: LiveMatchEvent):
                if event.event_type == EventType.FULL_TIME:
                    print(f"⏰ FINAL: {event.description}")

            tracker.add_event_callback(on_goal)
            tracker.add_event_callback(on_match_end)

            # Iniciar tracking (bloquea hasta Ctrl+C)
            await tracker.start_tracking()

        except KeyboardInterrupt:
            print("\n🛑 Deteniendo Live Tracker...")
        except Exception as e:
            print(f"❌ Error: {e}")

    # Ejecutar sistema
    asyncio.run(main())
