"""
SQLAlchemy Models for Football Analytics Database

Este módulo contiene todos los modelos de base de datos para el sistema de análisis
de fútbol, incluyendo equipos, jugadores, partidos, estadísticas y predicciones.
"""

import enum

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

# Base para todos los modelos
Base = declarative_base()

# =====================================================
# ENUMS
# =====================================================


class MatchStatus(enum.Enum):
    SCHEDULED = "scheduled"
    LIVE = "live"
    HALFTIME = "halftime"
    FINISHED = "finished"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"
    SUSPENDED = "suspended"


class PlayerPosition(enum.Enum):
    GOALKEEPER = "Goalkeeper"
    DEFENDER = "Defender"
    MIDFIELDER = "Midfielder"
    FORWARD = "Forward"


class InjurySeverity(enum.Enum):
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    SEVERE = "severe"


class TransferType(enum.Enum):
    PERMANENT = "permanent"
    LOAN = "loan"
    FREE = "free"
    UNKNOWN = "unknown"


class EventType(enum.Enum):
    GOAL = "goal"
    ASSIST = "assist"
    YELLOW_CARD = "yellow_card"
    RED_CARD = "red_card"
    SUBSTITUTION = "substitution"
    PENALTY = "penalty"
    OWN_GOAL = "own_goal"
    VAR = "var"


# =====================================================
# MIXIN CLASSES (Funcionalidades comunes)
# =====================================================


class TimestampMixin:
    """Mixin para campos de timestamp automáticos."""

    created_at = Column(DateTime, default=func.current_timestamp(), nullable=False)
    updated_at = Column(
        DateTime,
        default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
        nullable=False,
    )


class ActiveMixin:
    """Mixin para soft delete."""

    is_active = Column(Boolean, default=True, nullable=False)


# =====================================================
# MODELOS PRINCIPALES
# =====================================================


class Country(Base, TimestampMixin):
    """Modelo para países."""

    __tablename__ = "countries"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    code = Column(String(3), nullable=False, unique=True)  # ISO code
    flag_url = Column(String(255))

    # Relaciones
    leagues = relationship("League", back_populates="country")

    def __repr__(self):
        return f"<Country(name='{self.name}', code='{self.code}')>"


class League(Base, TimestampMixin, ActiveMixin):
    """Modelo para ligas de fútbol."""

    __tablename__ = "leagues"

    id = Column(Integer, primary_key=True)
    name = Column(String(150), nullable=False)
    short_name = Column(String(50))
    country_id = Column(Integer, ForeignKey("countries.id"), nullable=False)
    logo_url = Column(String(255))
    tier = Column(Integer, default=1)  # División (1=Primera, 2=Segunda, etc.)
    current_season = Column(String(20))  # "2023-24"

    # Configuración de la liga
    teams_count = Column(Integer, default=20)
    rounds_count = Column(Integer, default=38)  # Número de jornadas
    points_for_win = Column(Integer, default=3)
    points_for_draw = Column(Integer, default=1)

    # Relaciones
    country = relationship("Country", back_populates="leagues")
    teams = relationship("Team", back_populates="league")
    matches = relationship("Match", back_populates="league")
    seasons = relationship("Season", back_populates="league")

    # Índices
    __table_args__ = (
        Index("idx_league_country", "country_id"),
        Index("idx_league_active", "is_active"),
    )

    def __repr__(self):
        return f"<League(name='{self.name}', country='{self.country.name if self.country else 'N/A'}')>"


class Stadium(Base, TimestampMixin):
    """Modelo para estadios."""

    __tablename__ = "stadiums"

    id = Column(Integer, primary_key=True)
    name = Column(String(150), nullable=False)
    city = Column(String(100), nullable=False)
    country_id = Column(Integer, ForeignKey("countries.id"))
    capacity = Column(Integer)
    surface = Column(String(50), default="grass")  # grass, artificial, hybrid

    # Coordenadas geográficas
    latitude = Column(Numeric(10, 8))
    longitude = Column(Numeric(11, 8))

    # Información adicional
    opened_year = Column(Integer)
    image_url = Column(String(255))

    # Relaciones
    country = relationship("Country")
    teams = relationship("Team", back_populates="stadium")
    matches = relationship("Match", back_populates="stadium")

    def __repr__(self):
        return f"<Stadium(name='{self.name}', city='{self.city}', capacity={self.capacity})>"


class Team(Base, TimestampMixin, ActiveMixin):
    """Modelo para equipos de fútbol."""

    __tablename__ = "teams"

    id = Column(Integer, primary_key=True)
    name = Column(String(150), nullable=False)
    short_name = Column(String(50))
    league_id = Column(Integer, ForeignKey("leagues.id"), nullable=False)
    stadium_id = Column(Integer, ForeignKey("stadiums.id"))

    # Información básica
    founded_year = Column(Integer)
    logo_url = Column(String(255))
    website = Column(String(255))
    colors = Column(String(100))  # "Red, White"

    # Información financiera
    market_value = Column(Numeric(15, 2))  # Valor de mercado en euros

    # Relaciones
    league = relationship("League", back_populates="teams")
    stadium = relationship("Stadium", back_populates="teams")
    players = relationship("Player", back_populates="team")
    coach = relationship("Coach", back_populates="team", uselist=False)

    # Partidos como local y visitante
    home_matches = relationship(
        "Match", foreign_keys="Match.home_team_id", back_populates="home_team"
    )
    away_matches = relationship(
        "Match", foreign_keys="Match.away_team_id", back_populates="away_team"
    )

    # Estadísticas
    stats = relationship("TeamStats", back_populates="team")
    transfers = relationship("TeamTransfer", back_populates="team")

    # Índices
    __table_args__ = (
        Index("idx_team_league", "league_id"),
        Index("idx_team_active", "is_active"),
        Index("idx_team_name", "name"),
    )

    def __repr__(self):
        return f"<Team(name='{self.name}', league='{self.league.name if self.league else 'N/A'}')>"


class Coach(Base, TimestampMixin):
    """Modelo para entrenadores."""

    __tablename__ = "coaches"

    id = Column(Integer, primary_key=True)
    name = Column(String(150), nullable=False)
    nationality = Column(String(100))
    birth_date = Column(Date)
    team_id = Column(Integer, ForeignKey("teams.id"))

    # Información profesional
    appointed_date = Column(Date)
    contract_expires = Column(Date)
    preferred_formation = Column(String(20))  # "4-4-2", "3-5-2"

    # Estadísticas
    matches_coached = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    draws = Column(Integer, default=0)
    losses = Column(Integer, default=0)

    # Relación
    team = relationship("Team", back_populates="coach")

    def __repr__(self):
        return f"<Coach(name='{self.name}', team='{self.team.name if self.team else 'N/A'}')>"


class Player(Base, TimestampMixin, ActiveMixin):
    """Modelo para jugadores de fútbol."""

    __tablename__ = "players"

    id = Column(Integer, primary_key=True)
    name = Column(String(150), nullable=False)
    full_name = Column(String(200))
    team_id = Column(Integer, ForeignKey("teams.id"))

    # Información personal
    birth_date = Column(Date)
    nationality = Column(String(100))
    height = Column(Integer)  # en cm
    weight = Column(Integer)  # en kg
    preferred_foot = Column(String(10))  # "left", "right", "both"

    # Información del jugador
    position = Column(String(50), nullable=False)
    jersey_number = Column(Integer)
    market_value = Column(Numeric(12, 2))  # Valor en euros

    # Información contractual
    contract_expires = Column(Date)

    # URLs
    photo_url = Column(String(255))

    # Relaciones
    team = relationship("Team", back_populates="players")
    stats = relationship("PlayerStats", back_populates="player")
    injuries = relationship("PlayerInjury", back_populates="player")
    transfers = relationship("PlayerTransfer", back_populates="player")
    match_events = relationship("MatchEvent", back_populates="player")

    # Índices
    __table_args__ = (
        Index("idx_player_team", "team_id"),
        Index("idx_player_position", "position"),
        Index("idx_player_active", "is_active"),
        Index("idx_player_name", "name"),
        UniqueConstraint("team_id", "jersey_number", name="uq_team_jersey"),
    )

    def __repr__(self):
        return f"<Player(name='{self.name}', position='{self.position}', team='{self.team.name if self.team else 'N/A'}')>"


class Match(Base, TimestampMixin):
    """Modelo para partidos de fútbol."""

    __tablename__ = "matches"

    id = Column(Integer, primary_key=True)
    league_id = Column(Integer, ForeignKey("leagues.id"), nullable=False)
    season_id = Column(Integer, ForeignKey("seasons.id"), nullable=True)
    round_number = Column(Integer)  # Jornada

    # Equipos
    home_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)

    # Fechas y horarios
    match_date = Column(DateTime, nullable=False)

    # Estadio
    stadium_id = Column(Integer, ForeignKey("stadiums.id"))

    # Estado del partido
    status = Column(SQLEnum(MatchStatus), default=MatchStatus.SCHEDULED, nullable=False)

    # Resultados
    home_score = Column(Integer, default=0)
    away_score = Column(Integer, default=0)
    home_score_ht = Column(Integer, default=0)  # Medio tiempo
    away_score_ht = Column(Integer, default=0)

    # Información del partido
    attendance = Column(Integer)
    referee = Column(String(100))
    weather = Column(String(100))
    temperature = Column(Integer)  # Celsius

    # Metadatos
    importance = Column(String(20), default="normal")  # normal, high, final, derby
    is_neutral_venue = Column(Boolean, default=False)

    # Relaciones
    league = relationship("League", back_populates="matches")
    home_team = relationship(
        "Team", foreign_keys=[home_team_id], back_populates="home_matches"
    )
    away_team = relationship(
        "Team", foreign_keys=[away_team_id], back_populates="away_matches"
    )
    stadium = relationship("Stadium", back_populates="matches")
    events = relationship(
        "MatchEvent", back_populates="match", cascade="all, delete-orphan"
    )
    betting_odds = relationship("BettingOdds", back_populates="match")
    predictions = relationship("PredictionHistory", back_populates="match")
    season = relationship("Season", back_populates="matches")

    # Índices
    __table_args__ = (
        Index("idx_match_date", "match_date"),
        Index("idx_match_teams", "home_team_id", "away_team_id"),
        Index("idx_match_league_season", "league_id", "season_id"),
        Index("idx_match_status", "status"),
        CheckConstraint("home_team_id != away_team_id", name="check_different_teams"),
    )

    def __repr__(self):
        return f"<Match({self.home_team.name if self.home_team else 'TBD'} vs {self.away_team.name if self.away_team else 'TBD'}, {self.match_date})>"


class MatchEvent(Base, TimestampMixin):
    """Modelo para eventos durante un partido."""

    __tablename__ = "match_events"

    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    player_id = Column(Integer, ForeignKey("players.id"))

    # Información del evento
    event_type = Column(SQLEnum(EventType), nullable=False)
    minute = Column(Integer, nullable=False)
    extra_minute = Column(Integer, default=0)  # Tiempo añadido
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)

    # Detalles específicos
    description = Column(String(255))
    player_in = Column(Integer, ForeignKey("players.id"))  # Para sustituciones
    player_out = Column(Integer, ForeignKey("players.id"))  # Para sustituciones
    assist_player_id = Column(Integer, ForeignKey("players.id"))  # Para goles

    # Metadatos
    x_coordinate = Column(Float)  # Posición en el campo (0-100)
    y_coordinate = Column(Float)  # Posición en el campo (0-100)

    # Relaciones
    match = relationship("Match", back_populates="events")
    player = relationship(
        "Player", foreign_keys=[player_id], back_populates="match_events"
    )
    team = relationship("Team")
    player_in_rel = relationship("Player", foreign_keys=[player_in])
    player_out_rel = relationship("Player", foreign_keys=[player_out])
    assist_player = relationship("Player", foreign_keys=[assist_player_id])

    # Índices
    __table_args__ = (
        Index("idx_event_match", "match_id"),
        Index("idx_event_player", "player_id"),
        Index("idx_event_type", "event_type"),
        Index("idx_event_minute", "minute"),
    )

    def __repr__(self):
        return f"<MatchEvent({self.event_type.value}, minute {self.minute}, {self.player.name if self.player else 'N/A'})>"


class Season(Base, TimestampMixin):
    """Modelo para temporadas de ligas."""

    __tablename__ = "seasons"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)  # e.g., "2023/24"
    year_start = Column(Integer, nullable=False)  # e.g., 2023
    year_end = Column(Integer, nullable=False)  # e.g., 2024
    is_current = Column(Boolean, default=False, nullable=False)
    start_date = Column(Date, nullable=True)
    end_date = Column(Date, nullable=True)

    # Foreign Keys
    league_id = Column(Integer, ForeignKey("leagues.id"), nullable=False)

    # Relationships
    league = relationship("League", back_populates="seasons")
    matches = relationship("Match", back_populates="season")

    # Constraints
    __table_args__ = (
        Index("idx_season_league", "league_id"),
        Index("idx_season_current", "is_current"),
        UniqueConstraint("league_id", "year_start", name="uq_league_season"),
    )

    def __repr__(self):
        return f"<Season(name='{self.name}', league='{self.league.name if self.league else 'N/A'}')>"


# =====================================================
# MODELOS DE ESTADÍSTICAS
# =====================================================


class TeamStats(Base, TimestampMixin):
    """Estadísticas de equipos por temporada."""

    __tablename__ = "team_stats"

    id = Column(Integer, primary_key=True)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    season = Column(String(20), nullable=False)

    # Estadísticas generales
    matches_played = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    draws = Column(Integer, default=0)
    losses = Column(Integer, default=0)

    # Goles
    goals_for = Column(Integer, default=0)
    goals_against = Column(Integer, default=0)
    goal_difference = Column(Integer, default=0)

    # Puntos y posición
    points = Column(Integer, default=0)
    position = Column(Integer)

    # Estadísticas local/visitante
    home_wins = Column(Integer, default=0)
    home_draws = Column(Integer, default=0)
    home_losses = Column(Integer, default=0)
    away_wins = Column(Integer, default=0)
    away_draws = Column(Integer, default=0)
    away_losses = Column(Integer, default=0)

    # Estadísticas avanzadas
    shots_on_target = Column(Integer, default=0)
    shots_off_target = Column(Integer, default=0)
    possession_percentage = Column(Float)
    pass_accuracy = Column(Float)
    corners = Column(Integer, default=0)
    fouls = Column(Integer, default=0)
    yellow_cards = Column(Integer, default=0)
    red_cards = Column(Integer, default=0)

    # Expected Goals (xG)
    expected_goals_for = Column(Float)
    expected_goals_against = Column(Float)

    # Relación
    team = relationship("Team", back_populates="stats")

    # Índices
    __table_args__ = (
        Index("idx_team_stats_season", "team_id", "season"),
        UniqueConstraint("team_id", "season", name="uq_team_season_stats"),
    )

    def __repr__(self):
        return f"<TeamStats({self.team.name if self.team else 'N/A'}, {self.season}, {self.points}pts)>"


class PlayerStats(Base, TimestampMixin):
    """Estadísticas de jugadores por temporada."""

    __tablename__ = "player_stats"

    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    season = Column(String(20), nullable=False)

    # Estadísticas básicas
    matches_played = Column(Integer, default=0)
    minutes_played = Column(Integer, default=0)

    # Goles y asistencias
    goals = Column(Integer, default=0)
    assists = Column(Integer, default=0)

    # Disciplina
    yellow_cards = Column(Integer, default=0)
    red_cards = Column(Integer, default=0)

    # Estadísticas ofensivas
    shots_on_target = Column(Integer, default=0)
    shots_off_target = Column(Integer, default=0)
    key_passes = Column(Integer, default=0)

    # Estadísticas defensivas
    tackles = Column(Integer, default=0)
    interceptions = Column(Integer, default=0)
    clearances = Column(Integer, default=0)
    blocks = Column(Integer, default=0)

    # Estadísticas de pase
    passes_completed = Column(Integer, default=0)
    passes_attempted = Column(Integer, default=0)
    pass_accuracy = Column(Float)

    # Rating promedio
    rating = Column(Float)

    # Expected Goals/Assists
    expected_goals = Column(Float)
    expected_assists = Column(Float)

    # Relación
    player = relationship("Player", back_populates="stats")

    # Índices
    __table_args__ = (
        Index("idx_player_stats_season", "player_id", "season"),
        UniqueConstraint("player_id", "season", name="uq_player_season_stats"),
    )

    def __repr__(self):
        return f"<PlayerStats({self.player.name if self.player else 'N/A'}, {self.season}, {self.goals}G/{self.assists}A)>"


# =====================================================
# MODELOS DE LESIONES Y TRANSFERENCIAS
# =====================================================


class PlayerInjury(Base, TimestampMixin):
    """Modelo para lesiones de jugadores."""

    __tablename__ = "player_injuries"

    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)

    # Información de la lesión
    injury_type = Column(String(100), nullable=False)  # "Muscle", "Knee", etc.
    injury_date = Column(Date, nullable=False)
    expected_return_date = Column(Date)
    actual_return_date = Column(Date)

    # Severidad y descripción
    severity = Column(SQLEnum(InjurySeverity), nullable=False)
    description = Column(Text)

    # Relación
    player = relationship("Player", back_populates="injuries")

    # Índices
    __table_args__ = (
        Index("idx_injury_player", "player_id"),
        Index("idx_injury_date", "injury_date"),
        Index("idx_injury_severity", "severity"),
    )

    def __repr__(self):
        return f"<PlayerInjury({self.player.name if self.player else 'N/A'}, {self.injury_type}, {self.severity.value})>"


class PlayerTransfer(Base, TimestampMixin):
    """Modelo para transferencias de jugadores."""

    __tablename__ = "player_transfers"

    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)

    # Información de la transferencia
    from_team = Column(String(150))
    to_team = Column(String(150))
    transfer_date = Column(Date, nullable=False)
    transfer_fee = Column(Numeric(12, 2))  # En euros
    transfer_type = Column(SQLEnum(TransferType), nullable=False)

    # Contrato
    contract_duration = Column(Integer)  # años
    annual_salary = Column(Numeric(10, 2))

    # Relación
    player = relationship("Player", back_populates="transfers")

    # Índices
    __table_args__ = (
        Index("idx_transfer_player", "player_id"),
        Index("idx_transfer_date", "transfer_date"),
    )

    def __repr__(self):
        return f"<PlayerTransfer({self.player.name if self.player else 'N/A'}, {self.from_team} → {self.to_team})>"


class TeamTransfer(Base, TimestampMixin):
    """Transferencias desde perspectiva del equipo."""

    __tablename__ = "team_transfers"

    id = Column(Integer, primary_key=True)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)

    # Información de la transferencia
    player_name = Column(String(150), nullable=False)
    from_team_id = Column(Integer, ForeignKey("teams.id"))
    to_team_id = Column(Integer, ForeignKey("teams.id"))
    from_team = Column(String(150))
    to_team = Column(String(150))

    transfer_date = Column(Date, nullable=False)
    transfer_fee = Column(Numeric(12, 2))
    transfer_type = Column(SQLEnum(TransferType), nullable=False)
    is_incoming = Column(Boolean, nullable=False)  # True si llega al equipo

    # Relación
    team = relationship("Team", back_populates="transfers")

    def __repr__(self):
        direction = "IN" if self.is_incoming else "OUT"
        return f"<TeamTransfer({self.team.name if self.team else 'N/A'}, {self.player_name} {direction})>"


# =====================================================
# MODELOS DE PREDICCIONES Y APUESTAS
# =====================================================


class BettingOdds(Base, TimestampMixin):
    """Modelo para cuotas de apuestas."""

    __tablename__ = "betting_odds"

    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    bookmaker = Column(String(100), nullable=False)

    # Mercado 1X2
    home_win = Column(Float)
    draw = Column(Float)
    away_win = Column(Float)

    # Over/Under 2.5
    over_2_5 = Column(Float)
    under_2_5 = Column(Float)

    # Both Teams to Score
    btts_yes = Column(Float)
    btts_no = Column(Float)

    # Otros mercados
    total_goals_over_1_5 = Column(Float)
    total_goals_under_1_5 = Column(Float)
    total_goals_over_3_5 = Column(Float)
    total_goals_under_3_5 = Column(Float)

    # Timestamp de las cuotas
    odds_timestamp = Column(DateTime, default=func.now())

    # Relación
    match = relationship("Match", back_populates="betting_odds")

    # Índices
    __table_args__ = (
        Index("idx_odds_match", "match_id"),
        Index("idx_odds_bookmaker", "bookmaker"),
        Index("idx_odds_timestamp", "odds_timestamp"),
    )

    def __repr__(self):
        return f"<BettingOdds({self.bookmaker}, {self.home_win}-{self.draw}-{self.away_win})>"


class PredictionHistory(Base, TimestampMixin):
    """Historial de predicciones ML."""

    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True)
    prediction_id = Column(String(100), nullable=False, unique=True)
    match_id = Column(Integer, ForeignKey("matches.id"))

    # Equipos de la predicción
    home_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)

    # Información de la predicción
    prediction_type = Column(
        String(50), nullable=False
    )  # "match_result", "over_under", etc.
    predicted_value = Column(JSON)  # Datos de la predicción en JSON
    confidence = Column(Float, nullable=False)
    model_version = Column(String(20))

    # Resultado real (para evaluar precisión)
    actual_result = Column(JSON)
    is_correct = Column(Boolean)

    # Relaciones
    match = relationship("Match", back_populates="predictions")
    home_team = relationship("Team", foreign_keys=[home_team_id])
    away_team = relationship("Team", foreign_keys=[away_team_id])

    # Índices
    __table_args__ = (
        Index("idx_prediction_match", "match_id"),
        Index("idx_prediction_type", "prediction_type"),
        Index("idx_prediction_confidence", "confidence"),
        Index("idx_prediction_date", "created_at"),
    )

    def __repr__(self):
        return f"<PredictionHistory({self.prediction_type}, confidence={self.confidence:.2f})>"


# =====================================================
# MODELO DE CONFIGURACIÓN
# =====================================================


class SystemConfig(Base, TimestampMixin):
    """Configuración del sistema."""

    __tablename__ = "system_config"

    id = Column(Integer, primary_key=True)
    key = Column(String(100), nullable=False, unique=True)
    value = Column(Text)
    description = Column(Text)
    config_type = Column(String(50), default="string")  # string, int, float, bool, json

    def __repr__(self):
        return f"<SystemConfig({self.key}={self.value})>"


# =====================================================
# INFORMACIÓN DE MODELOS
# =====================================================

# Lista de todos los modelos para referencia
ALL_MODELS = [
    Country,
    League,
    Stadium,
    Team,
    Coach,
    Player,
    Match,
    MatchEvent,
    TeamStats,
    PlayerStats,
    PlayerInjury,
    PlayerTransfer,
    TeamTransfer,
    BettingOdds,
    PredictionHistory,
    SystemConfig,
]

# Metadatos de los modelos
MODELS_INFO = {
    "core_entities": [
        "Country",
        "League",
        "Stadium",
        "Team",
        "Coach",
        "Player",
    ],
    "match_data": ["Match", "MatchEvent", "BettingOdds"],
    "statistics": ["TeamStats", "PlayerStats"],
    "tracking": ["PlayerInjury", "PlayerTransfer", "TeamTransfer"],
    "ml_predictions": ["PredictionHistory"],
    "system": ["SystemConfig"],
    "total_models": len(ALL_MODELS),
}


# Función de utilidad para obtener info de modelos
def get_models_info():
    """Obtener información sobre todos los modelos."""
    return {
        "models": [model.__name__ for model in ALL_MODELS],
        "categories": MODELS_INFO,
        "total_tables": len(ALL_MODELS),
    }
