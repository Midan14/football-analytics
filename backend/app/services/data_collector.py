"""
Football Analytics - Data Collector Service
Recolector centralizado de datos de múltiples fuentes para análisis de fútbol
"""

import asyncio
import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import aiohttp
import asyncio_throttle
import pandas as pd

# Rate limiting y retry
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


@dataclass
class MatchData:
    """Estructura de datos para un partido"""

    match_id: str
    date: datetime
    home_team: str
    away_team: str
    home_goals: Optional[int] = None
    away_goals: Optional[int] = None
    league: str = ""
    season: str = ""
    matchday: Optional[int] = None
    status: str = "scheduled"  # scheduled, live, finished
    home_team_id: Optional[str] = None
    away_team_id: Optional[str] = None
    venue: Optional[str] = None
    referee: Optional[str] = None
    attendance: Optional[int] = None

    # Estadísticas del partido
    home_shots: Optional[int] = None
    away_shots: Optional[int] = None
    home_shots_on_target: Optional[int] = None
    away_shots_on_target: Optional[int] = None
    home_possession: Optional[float] = None
    away_possession: Optional[float] = None
    home_corners: Optional[int] = None
    away_corners: Optional[int] = None
    home_fouls: Optional[int] = None
    away_fouls: Optional[int] = None
    home_yellow_cards: Optional[int] = None
    away_yellow_cards: Optional[int] = None
    home_red_cards: Optional[int] = None
    away_red_cards: Optional[int] = None

    # Metadatos
    source: str = "unknown"
    collected_at: datetime = None

    def __post_init__(self):
        if self.collected_at is None:
            self.collected_at = datetime.now()

    @property
    def result(self) -> Optional[str]:
        """Resultado del partido (H/D/A)"""
        if self.home_goals is None or self.away_goals is None:
            return None
        if self.home_goals > self.away_goals:
            return "H"
        elif self.home_goals < self.away_goals:
            return "A"
        else:
            return "D"


@dataclass
class TeamData:
    """Estructura de datos para un equipo"""

    team_id: str
    name: str
    short_name: Optional[str] = None
    country: str = ""
    league: str = ""
    founded: Optional[int] = None
    venue: Optional[str] = None
    logo_url: Optional[str] = None
    market_value: Optional[float] = None

    # Estadísticas de la temporada
    matches_played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0
    points: int = 0
    position: Optional[int] = None

    # Metadatos
    source: str = "unknown"
    collected_at: datetime = None

    def __post_init__(self):
        if self.collected_at is None:
            self.collected_at = datetime.now()


@dataclass
class PlayerData:
    """Estructura de datos para un jugador"""

    player_id: str
    name: str
    team_id: str
    position: str
    age: Optional[int] = None
    nationality: str = ""
    market_value: Optional[float] = None
    goals: int = 0
    assists: int = 0
    minutes_played: int = 0
    matches_played: int = 0
    source: str = "unknown"
    collected_at: datetime = None

    def __post_init__(self):
        if self.collected_at is None:
            self.collected_at = datetime.now()


@dataclass
class OddsData:
    """Estructura de datos para cuotas"""

    match_id: str
    bookmaker: str
    home_win: Optional[float] = None
    draw: Optional[float] = None
    away_win: Optional[float] = None
    over_2_5: Optional[float] = None
    under_2_5: Optional[float] = None
    btts_yes: Optional[float] = None
    btts_no: Optional[float] = None
    timestamp: datetime = None
    source: str = "unknown"

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class BaseDataCollector(ABC):
    """Clase base para todos los collectors de datos"""

    def __init__(self, api_key: Optional[str] = None, rate_limit: int = 60):
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.session = None
        self.throttler = asyncio_throttle.Throttler(rate_limit=rate_limit, period=60)
        self.logger = logging.getLogger(self.__class__.__name__)

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30), headers=self._get_headers()
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    def _get_headers(self) -> Dict[str, str]:
        """Headers para las requests"""
        headers = {"User-Agent": "Football-Analytics/1.0", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
    )
    async def _make_request(self, url: str, params: Optional[Dict] = None) -> Dict:
        """Realiza request con rate limiting y retry"""
        async with self.throttler:
            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                return await response.json()

    @abstractmethod
    async def collect_matches(self, league_id: str, season: str) -> List[MatchData]:
        """Recolecta datos de partidos"""
        pass

    @abstractmethod
    async def collect_teams(self, league_id: str) -> List[TeamData]:
        """Recolecta datos de equipos"""
        pass


class FootballDataCollector(BaseDataCollector):
    """Collector para Football-Data.org API"""

    BASE_URL = "https://api.football-data.org/v4"

    def __init__(self, api_key: str):
        super().__init__(api_key, rate_limit=10)  # 10 requests per minute for free tier

    def _get_headers(self) -> Dict[str, str]:
        return {"X-Auth-Token": self.api_key, "Accept": "application/json"}

    async def collect_matches(self, league_id: str, season: str) -> List[MatchData]:
        """Recolecta partidos de Football-Data.org"""
        matches = []

        try:
            url = f"{self.BASE_URL}/competitions/{league_id}/matches"
            params = {"season": season}

            data = await self._make_request(url, params)

            for match in data.get("matches", []):
                match_data = MatchData(
                    match_id=str(match["id"]),
                    date=datetime.fromisoformat(
                        match["utcDate"].replace("Z", "+00:00")
                    ),
                    home_team=match["homeTeam"]["name"],
                    away_team=match["awayTeam"]["name"],
                    home_team_id=str(match["homeTeam"]["id"]),
                    away_team_id=str(match["awayTeam"]["id"]),
                    league=league_id,
                    season=season,
                    matchday=match.get("matchday"),
                    status=match["status"].lower(),
                    source="football-data.org",
                )

                # Agregar resultado si el partido terminó
                if match["status"] == "FINISHED" and match["score"]["fullTime"]:
                    match_data.home_goals = match["score"]["fullTime"]["home"]
                    match_data.away_goals = match["score"]["fullTime"]["away"]

                matches.append(match_data)

            self.logger.info(f"Recolectados {len(matches)} partidos de {league_id}")
            return matches

        except Exception as e:
            self.logger.error(f"Error recolectando partidos: {e}")
            return []

    async def collect_teams(self, league_id: str) -> List[TeamData]:
        """Recolecta equipos de Football-Data.org"""
        teams = []

        try:
            url = f"{self.BASE_URL}/competitions/{league_id}/teams"
            data = await self._make_request(url)

            for team in data.get("teams", []):
                team_data = TeamData(
                    team_id=str(team["id"]),
                    name=team["name"],
                    short_name=team.get("shortName"),
                    country=team.get("area", {}).get("name", ""),
                    league=league_id,
                    founded=team.get("founded"),
                    venue=team.get("venue"),
                    logo_url=team.get("crest"),
                    source="football-data.org",
                )
                teams.append(team_data)

            self.logger.info(f"Recolectados {len(teams)} equipos de {league_id}")
            return teams

        except Exception as e:
            self.logger.error(f"Error recolectando equipos: {e}")
            return []

    async def collect_standings(self, league_id: str, season: str) -> List[TeamData]:
        """Recolecta tabla de posiciones"""
        teams = []

        try:
            url = f"{self.BASE_URL}/competitions/{league_id}/standings"
            params = {"season": season}
            data = await self._make_request(url, params)

            standings = data.get("standings", [])
            if standings:
                table = standings[0].get("table", [])

                for entry in table:
                    team = entry["team"]
                    team_data = TeamData(
                        team_id=str(team["id"]),
                        name=team["name"],
                        short_name=team.get("shortName"),
                        league=league_id,
                        position=entry["position"],
                        matches_played=entry["playedGames"],
                        wins=entry["won"],
                        draws=entry["draw"],
                        losses=entry["lost"],
                        goals_for=entry["goalsFor"],
                        goals_against=entry["goalsAgainst"],
                        points=entry["points"],
                        source="football-data.org",
                    )
                    teams.append(team_data)

            return teams

        except Exception as e:
            self.logger.error(f"Error recolectando tabla: {e}")
            return []


class RapidAPICollector(BaseDataCollector):
    """Collector para RapidAPI (API-Football)"""

    BASE_URL = "https://api-football-v1.p.rapidapi.com/v3"

    def __init__(self, api_key: str):
        super().__init__(api_key, rate_limit=100)  # 100 requests per day for free tier

    def _get_headers(self) -> Dict[str, str]:
        return {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com",
        }

    async def collect_matches(self, league_id: str, season: str) -> List[MatchData]:
        """Recolecta partidos de RapidAPI"""
        matches = []

        try:
            url = f"{self.BASE_URL}/fixtures"
            params = {"league": league_id, "season": season}

            data = await self._make_request(url, params)

            for fixture in data.get("response", []):
                match_data = MatchData(
                    match_id=str(fixture["fixture"]["id"]),
                    date=datetime.fromisoformat(
                        fixture["fixture"]["date"].replace("Z", "+00:00")
                    ),
                    home_team=fixture["teams"]["home"]["name"],
                    away_team=fixture["teams"]["away"]["name"],
                    home_team_id=str(fixture["teams"]["home"]["id"]),
                    away_team_id=str(fixture["teams"]["away"]["id"]),
                    league=league_id,
                    season=season,
                    venue=fixture["fixture"]["venue"]["name"],
                    referee=fixture["fixture"]["referee"],
                    status=fixture["fixture"]["status"]["short"].lower(),
                    source="rapidapi",
                )

                # Agregar goles si están disponibles
                if fixture["goals"]["home"] is not None:
                    match_data.home_goals = fixture["goals"]["home"]
                    match_data.away_goals = fixture["goals"]["away"]

                # Agregar estadísticas si están disponibles
                if "statistics" in fixture:
                    stats = fixture["statistics"]
                    if len(stats) >= 2:
                        home_stats = (
                            stats[0]
                            if stats[0]["team"]["id"] == fixture["teams"]["home"]["id"]
                            else stats[1]
                        )
                        away_stats = (
                            stats[1]
                            if stats[1]["team"]["id"] == fixture["teams"]["away"]["id"]
                            else stats[0]
                        )

                        match_data.home_shots = self._extract_stat(
                            home_stats, "Total Shots"
                        )
                        match_data.away_shots = self._extract_stat(
                            away_stats, "Total Shots"
                        )
                        match_data.home_possession = self._extract_stat(
                            home_stats, "Ball Possession", True
                        )
                        match_data.away_possession = self._extract_stat(
                            away_stats, "Ball Possession", True
                        )

                matches.append(match_data)

            self.logger.info(f"Recolectados {len(matches)} partidos de {league_id}")
            return matches

        except Exception as e:
            self.logger.error(f"Error recolectando partidos: {e}")
            return []

    def _extract_stat(
        self, stats: Dict, stat_name: str, is_percentage: bool = False
    ) -> Optional[Union[int, float]]:
        """Extrae una estadística específica"""
        try:
            for stat in stats.get("statistics", []):
                if stat["type"] == stat_name:
                    value = stat["value"]
                    if value is None:
                        return None
                    if is_percentage:
                        return float(value.replace("%", ""))
                    return int(value)
        except:
            pass
        return None

    async def collect_teams(self, league_id: str) -> List[TeamData]:
        """Recolecta equipos de RapidAPI"""
        teams = []

        try:
            url = f"{self.BASE_URL}/teams"
            params = {"league": league_id, "season": "2024"}

            data = await self._make_request(url, params)

            for team_info in data.get("response", []):
                team = team_info["team"]
                venue = team_info.get("venue", {})

                team_data = TeamData(
                    team_id=str(team["id"]),
                    name=team["name"],
                    country=team["country"],
                    founded=team.get("founded"),
                    venue=venue.get("name"),
                    logo_url=team.get("logo"),
                    source="rapidapi",
                )
                teams.append(team_data)

            return teams

        except Exception as e:
            self.logger.error(f"Error recolectando equipos: {e}")
            return []


class OddsCollector(BaseDataCollector):
    """Collector especializado para cuotas"""

    def __init__(self, api_key: str):
        super().__init__(
            api_key, rate_limit=500
        )  # 500 requests per month for free tier
        self.base_url = "https://api.the-odds-api.com/v4"

    def _get_headers(self) -> Dict[str, str]:
        return {"Accept": "application/json"}

    async def collect_odds(self, sport: str = "soccer_epl") -> List[OddsData]:
        """Recolecta cuotas de partidos"""
        odds_data = []

        try:
            url = f"{self.base_url}/sports/{sport}/odds"
            params = {
                "apiKey": self.api_key,
                "regions": "eu",
                "markets": "h2h,totals,btts",
                "oddsFormat": "decimal",
            }

            data = await self._make_request(url, params)

            for game in data:
                match_id = game["id"]

                for bookmaker in game.get("bookmakers", []):
                    odds = OddsData(
                        match_id=match_id,
                        bookmaker=bookmaker["title"],
                        source="the-odds-api",
                    )

                    for market in bookmaker.get("markets", []):
                        if market["key"] == "h2h":
                            for outcome in market["outcomes"]:
                                if outcome["name"] == game["home_team"]:
                                    odds.home_win = outcome["price"]
                                elif outcome["name"] == game["away_team"]:
                                    odds.away_win = outcome["price"]
                                elif outcome["name"] == "Draw":
                                    odds.draw = outcome["price"]

                        elif market["key"] == "totals":
                            for outcome in market["outcomes"]:
                                if (
                                    outcome["name"] == "Over"
                                    and outcome["point"] == 2.5
                                ):
                                    odds.over_2_5 = outcome["price"]
                                elif (
                                    outcome["name"] == "Under"
                                    and outcome["point"] == 2.5
                                ):
                                    odds.under_2_5 = outcome["price"]

                    odds_data.append(odds)

            self.logger.info(f"Recolectadas cuotas para {len(data)} partidos")
            return odds_data

        except Exception as e:
            self.logger.error(f"Error recolectando cuotas: {e}")
            return []


class DataCollectorService:
    """Servicio principal que coordina todos los collectors"""

    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.collectors = {}
        self.database_path = "football_data.db"
        self.logger = logging.getLogger(__name__)

        # Inicializar collectors según configuración
        if "football_data_api_key" in config:
            self.collectors["football_data"] = FootballDataCollector(
                config["football_data_api_key"]
            )

        if "rapidapi_key" in config:
            self.collectors["rapidapi"] = RapidAPICollector(config["rapidapi_key"])

        if "odds_api_key" in config:
            self.collectors["odds"] = OddsCollector(config["odds_api_key"])

        # Inicializar base de datos
        self._init_database()

    def _init_database(self):
        """Inicializa la base de datos SQLite"""
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()

            # Tabla de partidos
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS matches (
                    match_id TEXT PRIMARY KEY,
                    date TIMESTAMP,
                    home_team TEXT,
                    away_team TEXT,
                    home_goals INTEGER,
                    away_goals INTEGER,
                    league TEXT,
                    season TEXT,
                    status TEXT,
                    source TEXT,
                    collected_at TIMESTAMP,
                    data JSON
                )
            """
            )

            # Tabla de equipos
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS teams (
                    team_id TEXT PRIMARY KEY,
                    name TEXT,
                    league TEXT,
                    country TEXT,
                    source TEXT,
                    collected_at TIMESTAMP,
                    data JSON
                )
            """
            )

            # Tabla de cuotas
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS odds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT,
                    bookmaker TEXT,
                    home_win REAL,
                    draw REAL,
                    away_win REAL,
                    timestamp TIMESTAMP,
                    source TEXT
                )
            """
            )

            conn.commit()

    async def collect_league_data(
        self, league_id: str, season: str, include_odds: bool = True
    ) -> Dict[str, Any]:
        """
        Recolecta datos completos de una liga

        Args:
            league_id: ID de la liga
            season: Temporada
            include_odds: Si incluir cuotas

        Returns:
            Diccionario con todos los datos recolectados
        """
        results = {
            "matches": [],
            "teams": [],
            "standings": [],
            "odds": [],
            "errors": [],
        }

        # Recolectar de múltiples fuentes en paralelo
        tasks = []

        for source_name, collector in self.collectors.items():
            if source_name == "odds" and not include_odds:
                continue

            if source_name == "odds":
                tasks.append(self._collect_odds_wrapper(collector))
            else:
                tasks.append(
                    self._collect_source_data(collector, league_id, season, source_name)
                )

        # Ejecutar todas las tareas en paralelo
        source_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Procesar resultados
        for i, result in enumerate(source_results):
            if isinstance(result, Exception):
                results["errors"].append(str(result))
                continue

            if "matches" in result:
                results["matches"].extend(result["matches"])
            if "teams" in result:
                results["teams"].extend(result["teams"])
            if "standings" in result:
                results["standings"].extend(result["standings"])
            if "odds" in result:
                results["odds"].extend(result["odds"])

        # Guardar en base de datos
        await self._save_to_database(results)

        # Estadísticas de recolección
        stats = {
            "matches_collected": len(results["matches"]),
            "teams_collected": len(results["teams"]),
            "odds_collected": len(results["odds"]),
            "sources_used": len(
                [r for r in source_results if not isinstance(r, Exception)]
            ),
            "errors": len(results["errors"]),
        }

        self.logger.info(f"Recolección completada: {stats}")

        return {**results, "stats": stats}

    async def _collect_source_data(
        self, collector, league_id: str, season: str, source_name: str
    ) -> Dict[str, List]:
        """Recolecta datos de una fuente específica"""
        result = {"matches": [], "teams": [], "standings": []}

        try:
            async with collector:
                # Recolectar partidos
                matches = await collector.collect_matches(league_id, season)
                result["matches"] = matches

                # Recolectar equipos
                teams = await collector.collect_teams(league_id)
                result["teams"] = teams

                # Recolectar tabla (si está disponible)
                if hasattr(collector, "collect_standings"):
                    standings = await collector.collect_standings(league_id, season)
                    result["standings"] = standings

        except Exception as e:
            self.logger.error(f"Error en collector {source_name}: {e}")
            raise

        return result

    async def _collect_odds_wrapper(self, odds_collector) -> Dict[str, List]:
        """Wrapper para recolectar cuotas"""
        try:
            async with odds_collector:
                odds = await odds_collector.collect_odds()
                return {"odds": odds}
        except Exception as e:
            self.logger.error(f"Error recolectando cuotas: {e}")
            return {"odds": []}

    async def _save_to_database(self, data: Dict[str, List]):
        """Guarda los datos recolectados en la base de datos"""
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()

            # Guardar partidos
            for match in data["matches"]:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO matches 
                    (match_id, date, home_team, away_team, home_goals, away_goals, 
                     league, season, status, source, collected_at, data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        match.match_id,
                        match.date,
                        match.home_team,
                        match.away_team,
                        match.home_goals,
                        match.away_goals,
                        match.league,
                        match.season,
                        match.status,
                        match.source,
                        match.collected_at,
                        json.dumps(asdict(match)),
                    ),
                )

            # Guardar equipos
            for team in data["teams"]:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO teams 
                    (team_id, name, league, country, source, collected_at, data)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        team.team_id,
                        team.name,
                        team.league,
                        team.country,
                        team.source,
                        team.collected_at,
                        json.dumps(asdict(team)),
                    ),
                )

            # Guardar cuotas
            for odds in data["odds"]:
                cursor.execute(
                    """
                    INSERT INTO odds 
                    (match_id, bookmaker, home_win, draw, away_win, timestamp, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        odds.match_id,
                        odds.bookmaker,
                        odds.home_win,
                        odds.draw,
                        odds.away_win,
                        odds.timestamp,
                        odds.source,
                    ),
                )

            conn.commit()

    def get_matches_dataframe(
        self, league: str = None, season: str = None
    ) -> pd.DataFrame:
        """Obtiene partidos como DataFrame de pandas"""
        with sqlite3.connect(self.database_path) as conn:
            query = "SELECT * FROM matches WHERE 1=1"
            params = []

            if league:
                query += " AND league = ?"
                params.append(league)

            if season:
                query += " AND season = ?"
                params.append(season)

            return pd.read_sql_query(query, conn, params=params)

    def get_teams_dataframe(self, league: str = None) -> pd.DataFrame:
        """Obtiene equipos como DataFrame de pandas"""
        with sqlite3.connect(self.database_path) as conn:
            query = "SELECT * FROM teams WHERE 1=1"
            params = []

            if league:
                query += " AND league = ?"
                params.append(league)

            return pd.read_sql_query(query, conn, params=params)

    async def schedule_collection(
        self, leagues: List[str], season: str, interval_hours: int = 24
    ):
        """Programa recolección automática de datos"""
        while True:
            try:
                for league in leagues:
                    self.logger.info(f"Iniciando recolección programada para {league}")
                    await self.collect_league_data(league, season)

                    # Esperar entre ligas para no sobrecargar APIs
                    await asyncio.sleep(60)

                # Esperar intervalo antes de la siguiente recolección
                await asyncio.sleep(interval_hours * 3600)

            except Exception as e:
                self.logger.error(f"Error en recolección programada: {e}")
                await asyncio.sleep(300)  # Esperar 5 minutos antes de reintentar


# Funciones de conveniencia
def create_data_collector(config: Dict[str, str]) -> DataCollectorService:
    """Factory function para crear el data collector"""
    return DataCollectorService(config)


async def collect_premier_league_data(api_keys: Dict[str, str]) -> Dict[str, Any]:
    """Función de conveniencia para recolectar datos de Premier League"""
    collector = create_data_collector(api_keys)
    return await collector.collect_league_data("PL", "2024", include_odds=True)


# Configuración de producción con tu key real de Football-Data.org
PRODUCTION_CONFIG = {
    # ✅ TU KEY REAL DE FOOTBALL-DATA.ORG CONFIGURADA
    "football_data_api_key": "9c9a42cbff2e8eb387eac2755c5e1e97",  # Tu key real configurada
    # 🔄 APIs opcionales (se pueden agregar después)
    "rapidapi_key": None,  # Opcional: para estadísticas detalladas
    "odds_api_key": None,  # Opcional: para cuotas de apuestas
    # Configuración de ligas (Football-Data.org códigos)
    "leagues": {
        "premier_league": "PL",  # ✅ Disponible con tu key
        "championship": "ELC",  # ✅ Disponible con tu key
        "la_liga": "PD",  # ✅ Disponible con tu key
        "serie_a": "SA",  # ✅ Disponible con tu key
        "bundesliga": "BL1",  # ✅ Disponible con tu key
        "ligue_1": "FL1",  # ✅ Disponible con tu key
        "eredivisie": "DED",  # ✅ Disponible con tu key
        "primeira_liga": "PPL",  # ✅ Disponible con tu key
        "champions_league": "CL",  # ✅ Disponible con tu key
        "europa_league": "EL",  # ✅ Disponible con tu key
        "world_cup": "WC",  # ✅ Disponible con tu key
        "european_championship": "EC",  # ✅ Disponible con tu key
    },
    # Configuración específica para Football-Data.org
    "football_data_config": {
        "base_url": "https://api.football-data.org/v4",
        "rate_limit": 10,  # 10 calls per minute (tier gratuito)
        "daily_limit": 100,  # 100 calls per day (tier gratuito)
        "timeout": 30,
        "retry_attempts": 3,
    },
    # Configuración de temporadas
    "current_season": "2024",
    "historical_seasons": ["2023", "2022", "2021"],
    # Base de datos de producción
    "database": {
        "type": "sqlite",
        "path": "/Users/miguelantonio/Desktop/football-analytics/backend/data/football_analytics.db",
        "backup_path": "/Users/miguelantonio/Desktop/football-analytics/backend/backups/",
        "max_connections": 20,
    },
    # Configuración de recolección automática
    "schedule": {
        "daily_collection_hour": 6,  # 6 AM UTC
        "match_data_interval": 3600,  # 1 hora
        "standings_interval": 86400,  # 24 horas
    },
    # Logging de producción
    "logging": {
        "level": "INFO",
        "file": "/Users/miguelantonio/Desktop/football-analytics/backend/logs/data_collector.log",
        "max_size": "100MB",
        "backup_count": 10,
    },
}

# Configuración por entorno (RECOMENDADO para seguridad)
import os
from typing import Any, Dict


def get_production_config() -> Dict[str, Any]:
    """
    Obtiene configuración de producción con tu key real
    """
    config = PRODUCTION_CONFIG.copy()

    # Prioridad 1: Variable de entorno (MÁS SEGURO)
    if os.getenv("FOOTBALL_DATA_API_KEY"):
        config["football_data_api_key"] = os.getenv("FOOTBALL_DATA_API_KEY")
        print("✅ Usando Football-Data key desde variable de entorno")

    # Prioridad 2: Key hardcoded (SOLO PARA DESARROLLO)
    elif config["football_data_api_key"] == "PEGA_TU_KEY_REAL_AQUI":
        print("⚠️  ATENCIÓN: Necesitas configurar tu Football-Data API key real")
        print("📝 Opciones:")
        print("   1. Editar PRODUCTION_CONFIG['football_data_api_key']")
        print("   2. Usar variable de entorno: export FOOTBALL_DATA_API_KEY='tu_key'")
        return None

    # Verificar que la key tiene formato válido
    if config["football_data_api_key"] and len(config["football_data_api_key"]) > 10:
        print("✅ Football-Data API key configurada correctamente")

    # Configurar otras APIs como opcionales
    if os.getenv("RAPIDAPI_KEY"):
        config["rapidapi_key"] = os.getenv("RAPIDAPI_KEY")
        print("✅ RapidAPI key encontrada - estadísticas detalladas habilitadas")

    if os.getenv("ODDS_API_KEY"):
        config["odds_api_key"] = os.getenv("ODDS_API_KEY")
        print("✅ Odds API key encontrada - cuotas de apuestas habilitadas")

    return config


async def test_football_data_connection(api_key: str) -> bool:
    """
    Prueba la conexión con tu key real de Football-Data.org
    """
    try:
        async with FootballDataCollector(api_key) as collector:
            # Probar con una liga simple
            matches = await collector.collect_matches("PL", "2024")
            if matches:
                print(
                    f"🎉 ¡CONEXIÓN EXITOSA! Encontrados {len(matches)} partidos de Premier League"
                )
                return True
            else:
                print("⚠️  Conexión OK pero sin datos. Verifica la temporada.")
                return True
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        print("🔍 Verifica que tu API key sea correcta")
        return False


async def initialize_football_data_only() -> DataCollectorService:
    """
    Inicializa el collector SOLO con Football-Data.org (tu key real)
    """
    config = get_production_config()

    if not config:
        raise ValueError(
            "❌ Configuración inválida. Configura tu Football-Data API key."
        )

    # Crear directorios necesarios
    os.makedirs(os.path.dirname(config["database"]["path"]), exist_ok=True)
    os.makedirs(config["database"]["backup_path"], exist_ok=True)
    os.makedirs(os.path.dirname(config["logging"]["file"]), exist_ok=True)

    # Configurar logging
    logging.basicConfig(
        level=getattr(logging, config["logging"]["level"]),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(config["logging"]["file"]),
            logging.StreamHandler(),
        ],
    )

    logger.info("🚀 Inicializando Football Analytics - SOLO Football-Data.org")
    logger.info(f"📊 Ligas disponibles: {len(config['leagues'])}")

    # Probar conexión
    if not await test_football_data_connection(config["football_data_api_key"]):
        raise ValueError("❌ No se pudo conectar con Football-Data.org")

    return DataCollectorService(config)


async def collect_premier_league_sample():
    """
    Función de prueba para recolectar datos de Premier League con tu key
    """
    print("🔄 Probando recolección de Premier League...")

    collector = await initialize_football_data_only()

    # Recolectar datos de Premier League
    data = await collector.collect_league_data("PL", "2024", include_odds=False)

    print("\n📊 RESULTADOS:")
    print(f"⚽ Partidos recolectados: {data['stats']['matches_collected']}")
    print(f"🏟️  Equipos recolectados: {data['stats']['teams_collected']}")
    print(f"📈 Fuentes exitosas: {data['stats']['sources_used']}")

    if data["stats"]["matches_collected"] > 0:
        print("\n✅ ¡ÉXITO! Tu sistema está funcionando correctamente")

        # Mostrar algunos datos de ejemplo
        matches_df = collector.get_matches_dataframe("PL", "2024")
        if len(matches_df) > 0:
            print("\n📋 Muestra de datos recolectados:")
            print(
                f"   Último partido: {matches_df.iloc[-1]['home_team']} vs {matches_df.iloc[-1]['away_team']}"
            )
            print(f"   Total partidos en BD: {len(matches_df)}")

    return data


if __name__ == "__main__":
    # INSTRUCCIONES PARA USAR TU KEY REAL:
    print("🔑 CONFIGURACIÓN DE TU KEY REAL DE FOOTBALL-DATA.ORG")
    print("=" * 60)
    print("📝 OPCIÓN 1 (RECOMENDADA): Variable de entorno")
    print("   export FOOTBALL_DATA_API_KEY='tu_key_real_aqui'")
    print("   python -m app.services.data_collector")
    print("")
    print("📝 OPCIÓN 2: Editar código directamente")
    print("   Reemplaza 'PEGA_TU_KEY_REAL_AQUI' con tu key real")
    print("   en la línea 4 de PRODUCTION_CONFIG")
    print("")
    print("🧪 EJECUTANDO PRUEBA CON PREMIER LEAGUE...")
    print("=" * 60)

    # Ejecutar prueba
    async def main():
        try:
            await collect_premier_league_sample()
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("\n🔧 SOLUCIONES:")
            print("1. Verifica que tu API key sea correcta")
            print("2. Asegúrate de tener conexión a internet")
            print("3. Verifica que no hayas excedido el límite de 100 calls/día")

    asyncio.run(main())
