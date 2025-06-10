#!/usr/bin/env python3
"""
Football Analytics - System Constants
Constantes del sistema Football Analytics para configuración y datos estáticos

Autor: Sistema Football Analytics
Versión: 2.1.0
Fecha: 2024-06-02
"""

from datetime import timedelta
from enum import Enum
from typing import Dict, List, Set, Tuple

# =============================================================================
# CONFIGURACIÓN GENERAL DEL SISTEMA
# =============================================================================

# Información del sistema
SYSTEM_INFO = {
    "name": "Football Analytics",
    "version": "2.1.0",
    "build_date": "2024-06-02",
    "author": "Football Analytics Team",
    "description": "Sistema avanzado de análisis y predicción deportiva",
    "api_version": "v1",
}

# Configuración de APIs
API_CONFIG = {
    "football_data_base_url": "https://api.football-data.org/v4",
    "rapidapi_base_url": "https://api-football-v1.p.rapidapi.com/v3",
    "odds_api_base_url": "https://api.the-odds-api.com/v4",
    "timeout_seconds": 30,
    "max_retries": 3,
    "retry_delay": 1.0,
}

# Rate limits por API (requests por minuto/día)
RATE_LIMITS = {
    "football_data": {"per_minute": 10, "per_day": 100},
    "rapidapi": {"per_minute": 100, "per_day": 500},
    "odds_api": {"per_minute": 500, "per_day": 1000},
}

# =============================================================================
# LIGAS Y COMPETICIONES
# =============================================================================

# Códigos de ligas principales
LEAGUE_CODES = {
    "PL": "Premier League",
    "PD": "La Liga",
    "SA": "Serie A",
    "BL1": "Bundesliga",
    "FL1": "Ligue 1",
    "CL": "Champions League",
    "EL": "Europa League",
    "WC": "World Cup",
    "EC": "European Championship",
}

# Ligas por país
LEAGUES_BY_COUNTRY = {
    "England": ["PL"],
    "Spain": ["PD"],
    "Italy": ["SA"],
    "Germany": ["BL1"],
    "France": ["FL1"],
    "International": ["CL", "EL", "WC", "EC"],
}

# Número de equipos por liga
TEAMS_PER_LEAGUE = {
    "PL": 20,
    "PD": 20,
    "SA": 20,
    "BL1": 18,
    "FL1": 20,
    "CL": 32,
    "EL": 48,
}

# Temporadas de partidos por liga
MATCHES_PER_SEASON = {
    "PL": 380,  # 20 equipos × 19 jornadas × 2 (ida y vuelta)
    "PD": 380,  # 20 equipos × 19 jornadas × 2
    "SA": 380,  # 20 equipos × 19 jornadas × 2
    "BL1": 306,  # 18 equipos × 17 jornadas × 2
    "FL1": 380,  # 20 equipos × 19 jornadas × 2
    "CL": 125,  # Fase de grupos + eliminatorias
    "EL": 205,  # Fase de grupos + eliminatorias
}

# =============================================================================
# EQUIPOS POR LIGA
# =============================================================================

# Premier League
PREMIER_LEAGUE_TEAMS = {
    "Arsenal",
    "Aston Villa",
    "Bournemouth",
    "Brighton & Hove Albion",
    "Burnley",
    "Chelsea",
    "Crystal Palace",
    "Everton",
    "Fulham",
    "Liverpool",
    "Luton Town",
    "Manchester City",
    "Manchester United",
    "Newcastle United",
    "Nottingham Forest",
    "Sheffield United",
    "Tottenham Hotspur",
    "West Ham United",
    "Wolverhampton Wanderers",
    "Brentford",
}

# La Liga
LA_LIGA_TEAMS = {
    "Real Madrid",
    "Barcelona",
    "Atletico Madrid",
    "Real Sociedad",
    "Athletic Bilbao",
    "Real Betis",
    "Villarreal",
    "Valencia",
    "Sevilla",
    "Getafe",
    "Osasuna",
    "Las Palmas",
    "Rayo Vallecano",
    "Mallorca",
    "Girona",
    "Cadiz",
    "Celta Vigo",
    "Alaves",
    "Granada",
    "Almeria",
}

# Serie A
SERIE_A_TEAMS = {
    "Inter Milan",
    "AC Milan",
    "Juventus",
    "Roma",
    "Lazio",
    "Napoli",
    "Atalanta",
    "Fiorentina",
    "Bologna",
    "Torino",
    "Monza",
    "Genoa",
    "Lecce",
    "Udinese",
    "Frosinone",
    "Hellas Verona",
    "Cagliari",
    "Sassuolo",
    "Empoli",
    "Salernitana",
}

# Bundesliga
BUNDESLIGA_TEAMS = {
    "Bayern Munich",
    "Borussia Dortmund",
    "RB Leipzig",
    "Union Berlin",
    "SC Freiburg",
    "Bayer Leverkusen",
    "Eintracht Frankfurt",
    "Wolfsburg",
    "Borussia Monchengladbach",
    "Mainz 05",
    "FC Koln",
    "Hoffenheim",
    "Werder Bremen",
    "FC Augsburg",
    "Heidenheim",
    "VfL Bochum",
    "Stuttgart",
    "Darmstadt 98",
}

# Ligue 1
LIGUE_1_TEAMS = {
    "Paris Saint-Germain",
    "AS Monaco",
    "Lille",
    "Olympique Marseille",
    "Rennes",
    "Lyon",
    "Nice",
    "Lens",
    "Nantes",
    "Strasbourg",
    "Montpellier",
    "Brest",
    "Reims",
    "Toulouse",
    "Le Havre",
    "Metz",
    "Lorient",
    "Clermont Foot",
    "Troyes",
    "Ajaccio",
}

# Todos los equipos conocidos
ALL_KNOWN_TEAMS = {
    "PL": PREMIER_LEAGUE_TEAMS,
    "PD": LA_LIGA_TEAMS,
    "SA": SERIE_A_TEAMS,
    "BL1": BUNDESLIGA_TEAMS,
    "FL1": LIGUE_1_TEAMS,
}

# =============================================================================
# TIPOS DE RESULTADO Y MERCADOS
# =============================================================================


class MatchResult(Enum):
    """Tipos de resultado de partido."""

    HOME_WIN = "H"
    DRAW = "D"
    AWAY_WIN = "A"


class MatchStatus(Enum):
    """Estados de partido."""

    SCHEDULED = "SCHEDULED"
    LIVE = "LIVE"
    IN_PLAY = "IN_PLAY"
    PAUSED = "PAUSED"
    FINISHED = "FINISHED"
    POSTPONED = "POSTPONED"
    CANCELLED = "CANCELLED"
    SUSPENDED = "SUSPENDED"


class BettingMarket(Enum):
    """Mercados de apuestas."""

    MATCH_RESULT = "1x2"  # Victoria local/empate/visitante
    OVER_UNDER = "over_under"  # Más/menos goles
    BOTH_TEAMS_SCORE = "btts"  # Ambos equipos marcan
    ASIAN_HANDICAP = "asian_handicap"  # Hándicap asiático
    CORRECT_SCORE = "correct_score"  # Resultado exacto
    HALF_TIME_RESULT = "ht_result"  # Resultado al descanso
    DOUBLE_CHANCE = "double_chance"  # Doble oportunidad
    DRAW_NO_BET = "draw_no_bet"  # Empate anula apuesta


class OddsFormat(Enum):
    """Formatos de cuotas."""

    DECIMAL = "decimal"  # 2.50
    FRACTIONAL = "fractional"  # 3/2
    AMERICAN = "american"  # +150
    IMPLIED = "implied"  # 40%


# =============================================================================
# CONFIGURACIÓN DE MODELOS ML
# =============================================================================

# Algoritmos de ML disponibles
ML_ALGORITHMS = {
    "xgboost": "XGBoost Classifier",
    "lightgbm": "LightGBM Classifier",
    "catboost": "CatBoost Classifier",
    "random_forest": "Random Forest Classifier",
    "logistic_regression": "Logistic Regression",
}

# Hiperparámetros por defecto
DEFAULT_HYPERPARAMETERS = {
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    "lightgbm": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    "catboost": {
        "iterations": 100,
        "depth": 6,
        "learning_rate": 0.1,
        "verbose": False,
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
    },
    "logistic_regression": {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"},
}

# Features de entrenamiento
TRAINING_FEATURES = [
    "home_team_form",
    "away_team_form",
    "home_attack_strength",
    "away_attack_strength",
    "home_defense_strength",
    "away_defense_strength",
    "home_points_per_game",
    "away_points_per_game",
    "home_win_percentage",
    "away_win_percentage",
    "home_goals_per_game",
    "away_goals_per_game",
    "league_position_difference",
    "points_difference",
    "market_value_ratio",
    "home_advantage",
    "h2h_home_win_rate",
    "h2h_avg_goals",
    "days_since_last_match",
    "importance_factor",
]

# Targets de predicción
PREDICTION_TARGETS = ["result", "total_goals", "btts", "over_under_2_5"]

# =============================================================================
# MÉTRICAS Y THRESHOLDS
# =============================================================================

# Niveles de confianza
CONFIDENCE_LEVELS = {
    "VERY_LOW": (0.0, 0.4),
    "LOW": (0.4, 0.6),
    "MEDIUM": (0.6, 0.75),
    "HIGH": (0.75, 0.85),
    "VERY_HIGH": (0.85, 1.0),
}

# Thresholds de calidad de modelo
MODEL_QUALITY_THRESHOLDS = {
    "minimum_accuracy": 0.45,  # Mejor que azar (33.3%)
    "good_accuracy": 0.55,  # Buena precisión
    "excellent_accuracy": 0.65,  # Excelente precisión
    "minimum_precision": 0.40,  # Por clase
    "minimum_recall": 0.35,  # Por clase
    "minimum_f1": 0.37,  # Por clase
    "max_brier_score": 0.25,  # Calibración
    "max_calibration_error": 0.10,  # Error de calibración
}

# Límites de datos válidos
DATA_VALIDATION_LIMITS = {
    "max_goals_per_team": 20,
    "max_expected_goals": 8.0,
    "min_probability": 0.01,
    "max_probability": 0.99,
    "min_odds": 1.01,
    "max_odds": 1000.0,
    "max_future_days": 365,
    "min_team_name_length": 2,
    "max_team_name_length": 50,
}

# =============================================================================
# CONFIGURACIÓN DE BASE DE DATOS
# =============================================================================

# Esquemas de tablas
DATABASE_TABLES = {
    "matches": {
        "columns": [
            "match_id",
            "date",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "league",
            "season",
            "status",
            "created_at",
        ],
        "primary_key": "match_id",
    },
    "predictions": {
        "columns": [
            "prediction_id",
            "match_id",
            "home_prob",
            "draw_prob",
            "away_prob",
            "home_goals_expected",
            "away_goals_expected",
            "confidence_score",
            "model_version",
            "created_at",
        ],
        "primary_key": "prediction_id",
    },
    "odds": {
        "columns": [
            "odds_id",
            "match_id",
            "bookmaker",
            "home_odds",
            "draw_odds",
            "away_odds",
            "market_type",
            "timestamp",
        ],
        "primary_key": "odds_id",
    },
    "teams": {
        "columns": [
            "team_id",
            "team_name",
            "league",
            "country",
            "founded_year",
            "stadium",
            "capacity",
            "website",
            "created_at",
        ],
        "primary_key": "team_id",
    },
}

# Configuración de SQLite
SQLITE_CONFIG = {
    "database_path": "/app/data/football_analytics.db",
    "backup_path": "/app/backups/",
    "pragmas": {
        "journal_mode": "WAL",
        "cache_size": -1000000,  # 1GB cache
        "foreign_keys": 1,
        "ignore_check_constraints": 0,
        "synchronous": 0,
    },
}

# =============================================================================
# CONFIGURACIÓN DE LOGGING
# =============================================================================

# Niveles de logging
LOG_LEVELS = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}

# Configuración de archivos de log
LOG_CONFIG = {
    "main_log": "/app/logs/football_analytics.log",
    "api_log": "/app/logs/api.log",
    "data_collector_log": "/app/logs/data_collector.log",
    "predictor_log": "/app/logs/predictor.log",
    "errors_log": "/app/logs/errors.log",
    "max_size": "100MB",
    "backup_count": 10,
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
}

# =============================================================================
# CONFIGURACIÓN DE CACHE
# =============================================================================

# Configuración de Redis (si disponible)
REDIS_CONFIG = {
    "host": "localhost",
    "port": 6379,
    "db": 0,
    "decode_responses": True,
    "socket_timeout": 5.0,
    "socket_connect_timeout": 5.0,
}

# TTL (Time To Live) para cache
CACHE_TTL = {
    "match_data": timedelta(hours=1),
    "team_stats": timedelta(hours=6),
    "predictions": timedelta(minutes=30),
    "odds_data": timedelta(minutes=5),
    "league_standings": timedelta(hours=2),
    "model_performance": timedelta(days=1),
}

# =============================================================================
# CONFIGURACIÓN DE NOTIFICACIONES
# =============================================================================

# Tipos de notificaciones
NOTIFICATION_TYPES = {
    "GOAL": "goal",
    "MATCH_START": "match_start",
    "MATCH_END": "match_end",
    "VALUE_BET": "value_bet",
    "MODEL_UPDATE": "model_update",
    "ERROR": "error",
}

# Configuración de WebSocket
WEBSOCKET_CONFIG = {
    "host": "localhost",
    "port": 8765,
    "max_connections": 100,
    "ping_interval": 30,
    "ping_timeout": 10,
    "close_timeout": 10,
}

# =============================================================================
# CONFIGURACIÓN DE SEGURIDAD
# =============================================================================

# Headers de seguridad
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": "default-src 'self'",
}

# Rate limiting para API
API_RATE_LIMITS = {
    "requests_per_minute": 60,
    "requests_per_hour": 1000,
    "requests_per_day": 10000,
}

# =============================================================================
# MENSAJES DEL SISTEMA
# =============================================================================

# Mensajes de error
ERROR_MESSAGES = {
    "INVALID_TEAM": "Nombre de equipo inválido o no reconocido",
    "INVALID_LEAGUE": "Liga no válida o no soportada",
    "INVALID_DATE": "Fecha inválida o fuera de rango",
    "INVALID_ODDS": "Cuotas inválidas o fuera de rango",
    "MODEL_NOT_FOUND": "Modelo no encontrado o no inicializado",
    "API_LIMIT_EXCEEDED": "Límite de API excedido",
    "DATABASE_ERROR": "Error de base de datos",
    "VALIDATION_FAILED": "Validación de datos falló",
}

# Mensajes de éxito
SUCCESS_MESSAGES = {
    "PREDICTION_CREATED": "Predicción creada exitosamente",
    "DATA_COLLECTED": "Datos recolectados exitosamente",
    "MODEL_TRAINED": "Modelo entrenado exitosamente",
    "ODDS_ANALYZED": "Análisis de cuotas completado",
    "CACHE_UPDATED": "Cache actualizado exitosamente",
}

# =============================================================================
# CONFIGURACIÓN DE DESARROLLO
# =============================================================================

# Configuración para testing
TEST_CONFIG = {
    "use_mock_apis": True,
    "mock_data_path": "/app/tests/mock_data/",
    "test_database": ":memory:",
    "fast_mode": True,
    "verbose_logging": False,
}

# Configuración de desarrollo
DEV_CONFIG = {
    "debug_mode": True,
    "auto_reload": True,
    "cors_enabled": True,
    "api_docs_enabled": True,
    "profiling_enabled": False,
}

# Configuración de producción
PROD_CONFIG = {
    "debug_mode": False,
    "auto_reload": False,
    "cors_enabled": False,
    "api_docs_enabled": False,
    "profiling_enabled": True,
    "ssl_enabled": True,
    "workers": 4,
}

# =============================================================================
# UTILIDADES Y HELPERS
# =============================================================================


def get_league_teams(league_code: str) -> Set[str]:
    """
    Obtiene los equipos de una liga específica.

    Args:
        league_code: Código de la liga (ej: 'PL', 'PD')

    Returns:
        Set con nombres de equipos
    """
    return ALL_KNOWN_TEAMS.get(league_code, set())


def is_valid_league(league_code: str) -> bool:
    """
    Verifica si un código de liga es válido.

    Args:
        league_code: Código de la liga

    Returns:
        True si la liga es válida
    """
    return league_code in LEAGUE_CODES


def get_league_name(league_code: str) -> str:
    """
    Obtiene el nombre completo de una liga.

    Args:
        league_code: Código de la liga

    Returns:
        Nombre completo de la liga
    """
    return LEAGUE_CODES.get(league_code, "Unknown League")


def get_confidence_level(score: float) -> str:
    """
    Determina el nivel de confianza basado en score.

    Args:
        score: Score de confianza (0-1)

    Returns:
        Nivel de confianza como string
    """
    for level, (min_val, max_val) in CONFIDENCE_LEVELS.items():
        if min_val <= score < max_val:
            return level
    return "UNKNOWN"


def get_algorithm_name(algorithm_code: str) -> str:
    """
    Obtiene el nombre completo de un algoritmo ML.

    Args:
        algorithm_code: Código del algoritmo

    Returns:
        Nombre completo del algoritmo
    """
    return ML_ALGORITHMS.get(algorithm_code, "Unknown Algorithm")


# =============================================================================
# CONFIGURACIÓN AVANZADA DE APUESTAS
# =============================================================================

# Bookmakers principales
BOOKMAKERS = {
    "bet365": "Bet365",
    "pinnacle": "Pinnacle",
    "betfair": "Betfair",
    "william_hill": "William Hill",
    "unibet": "Unibet",
    "betway": "Betway",
    "bwin": "Bwin",
    "ladbrokes": "Ladbrokes",
}

# Límites de apuesta por bookmaker (en euros)
BETTING_LIMITS = {
    "bet365": {"min": 1, "max": 25000},
    "pinnacle": {"min": 1, "max": 50000},
    "betfair": {"min": 2, "max": 100000},
    "william_hill": {"min": 1, "max": 25000},
}

# Márgenes típicos por bookmaker (overround %)
BOOKMAKER_MARGINS = {
    "pinnacle": 2.5,  # Mejores cuotas, menor margen
    "bet365": 5.0,  # Margen estándar
    "betfair": 2.0,  # Exchange, menor margen
    "william_hill": 6.0,  # Margen alto
}

# Criterios de Kelly para gestión de bankroll
KELLY_CRITERIA = {
    "conservative": 0.25,  # 25% del Kelly completo
    "moderate": 0.50,  # 50% del Kelly completo
    "aggressive": 0.75,  # 75% del Kelly completo
    "full_kelly": 1.0,  # Kelly completo (riesgoso)
}

# Límites de value betting
VALUE_BETTING_LIMITS = {
    "minimum_value_percentage": 2.0,  # Mínimo 2% de valor
    "excellent_value_percentage": 10.0,  # Excelente valor 10%+
    "maximum_stake_percentage": 5.0,  # Máximo 5% del bankroll
    "minimum_confidence": 0.6,  # Confianza mínima 60%
    "minimum_odds": 1.5,  # Cuotas mínimas 1.50
    "maximum_odds": 10.0,  # Cuotas máximas 10.0
}

# =============================================================================
# CONFIGURACIÓN DE LIVE TRACKING
# =============================================================================

# Intervalos de actualización
UPDATE_INTERVALS = {
    "live_matches": 30,  # 30 segundos para partidos en vivo
    "pre_match_odds": 300,  # 5 minutos para cuotas pre-partido
    "live_odds": 60,  # 1 minuto para cuotas en vivo
    "team_stats": 3600,  # 1 hora para estadísticas
    "league_standings": 7200,  # 2 horas para tablas
    "injury_reports": 1800,  # 30 minutos para lesiones
}

# Estados de partido en tiempo real
LIVE_MATCH_STATES = {
    "PRE_MATCH": "Antes del partido",
    "KICK_OFF": "Inicio del partido",
    "FIRST_HALF": "Primera parte",
    "HALF_TIME": "Descanso",
    "SECOND_HALF": "Segunda parte",
    "EXTRA_TIME": "Tiempo extra",
    "PENALTY_SHOOTOUT": "Tanda de penales",
    "FULL_TIME": "Tiempo completo",
    "MATCH_ABANDONED": "Partido suspendido",
}

# Eventos importantes a trackear
TRACKED_EVENTS = {
    "GOAL": {"priority": 1, "notify": True},
    "OWN_GOAL": {"priority": 1, "notify": True},
    "PENALTY_GOAL": {"priority": 1, "notify": True},
    "YELLOW_CARD": {"priority": 3, "notify": False},
    "RED_CARD": {"priority": 2, "notify": True},
    "SUBSTITUTION": {"priority": 4, "notify": False},
    "INJURY": {"priority": 2, "notify": True},
    "VAR_DECISION": {"priority": 2, "notify": True},
}

# =============================================================================
# FEATURES AVANZADAS DE ML
# =============================================================================

# Features categóricas que requieren encoding
CATEGORICAL_FEATURES = [
    "home_team",
    "away_team",
    "league",
    "season",
    "referee",
    "weather_conditions",
    "day_of_week",
]

# Features numéricas que requieren scaling
NUMERICAL_FEATURES = [
    "home_team_form",
    "away_team_form",
    "home_attack_strength",
    "away_attack_strength",
    "home_defense_strength",
    "away_defense_strength",
    "market_value_ratio",
    "h2h_avg_goals",
    "days_since_last_match",
]

# Features derivadas (calculadas automáticamente)
DERIVED_FEATURES = [
    "attack_difference",  # home_attack - away_attack
    "defense_difference",  # home_defense - away_defense
    "form_difference",  # home_form - away_form
    "total_strength",  # Suma de fortalezas
    "competitive_balance",  # Qué tan equilibrado es el partido
    "expected_competitiveness",  # Nivel de competitividad esperado
]

# Pesos para features específicas de fútbol
FEATURE_WEIGHTS = {
    "team_form": 0.25,  # Forma reciente es muy importante
    "attack_strength": 0.20,  # Capacidad ofensiva crucial
    "defense_strength": 0.20,  # Capacidad defensiva crucial
    "home_advantage": 0.15,  # Ventaja de local significativa
    "h2h_record": 0.10,  # Historial H2H relevante
    "league_position": 0.05,  # Posición en liga menos importante
    "market_value": 0.05,  # Valor de mercado como contexto
}

# =============================================================================
# CONFIGURACIÓN DE ANÁLISIS ESTADÍSTICO
# =============================================================================

# Distribuciones esperadas en fútbol
FOOTBALL_DISTRIBUTIONS = {
    "goals_per_match": {
        "mean": 2.6,  # Promedio de goles por partido
        "std": 1.6,  # Desviación estándar
        "min": 0,  # Mínimo posible
        "max": 20,  # Máximo razonable
    },
    "home_win_probability": {
        "mean": 0.45,  # 45% probabilidad promedio local
        "std": 0.15,  # Variación estándar
        "min": 0.15,  # Mínimo realista
        "max": 0.85,  # Máximo realista
    },
    "draw_probability": {
        "mean": 0.27,  # 27% probabilidad promedio empate
        "std": 0.08,  # Menor variación
        "min": 0.15,  # Mínimo realista
        "max": 0.40,  # Máximo realista
    },
}

# Correlaciones importantes en fútbol
FOOTBALL_CORRELATIONS = {
    "form_vs_goals": 0.75,  # Forma correlaciona con goles
    "defense_vs_conceded": -0.80,  # Mejor defensa = menos goles recibidos
    "attack_vs_scored": 0.85,  # Mejor ataque = más goles anotados
    "home_advantage": 0.65,  # Ventaja local es real
    "league_position_vs_form": 0.70,  # Posición refleja forma
}

# =============================================================================
# CONFIGURACIÓN DE BACKTESTING
# =============================================================================

# Períodos de backtesting
BACKTESTING_PERIODS = {
    "short_term": timedelta(days=30),  # 1 mes
    "medium_term": timedelta(days=90),  # 3 meses
    "long_term": timedelta(days=365),  # 1 año
    "full_season": timedelta(days=270),  # Temporada completa
}

# Métricas de evaluación para backtesting
BACKTESTING_METRICS = [
    "accuracy",  # Precisión general
    "precision_by_class",  # Precisión por resultado (H/D/A)
    "recall_by_class",  # Recall por resultado
    "f1_by_class",  # F1-score por resultado
    "log_loss",  # Loss logarítmica
    "brier_score",  # Score de Brier (calibración)
    "profit_loss",  # P&L de apuestas
    "roi",  # Return on Investment
    "max_drawdown",  # Máxima pérdida consecutiva
    "sharpe_ratio",  # Ratio de Sharpe
]

# Estrategias de apuesta para backtesting
BETTING_STRATEGIES = {
    "value_betting": {
        "min_value": 0.05,  # Mínimo 5% valor
        "max_stake": 0.02,  # Máximo 2% bankroll
        "min_confidence": 0.65,  # Mínimo 65% confianza
    },
    "conservative": {
        "min_value": 0.10,  # Mínimo 10% valor
        "max_stake": 0.01,  # Máximo 1% bankroll
        "min_confidence": 0.75,  # Mínimo 75% confianza
    },
    "aggressive": {
        "min_value": 0.03,  # Mínimo 3% valor
        "max_stake": 0.05,  # Máximo 5% bankroll
        "min_confidence": 0.55,  # Mínimo 55% confianza
    },
}

# =============================================================================
# CONFIGURACIÓN DE REPORTES
# =============================================================================

# Tipos de reportes disponibles
REPORT_TYPES = {
    "daily_predictions": "Predicciones Diarias",
    "weekly_performance": "Rendimiento Semanal",
    "monthly_summary": "Resumen Mensual",
    "betting_analysis": "Análisis de Apuestas",
    "model_performance": "Rendimiento de Modelos",
    "value_opportunities": "Oportunidades de Valor",
    "arbitrage_report": "Reporte de Arbitraje",
}

# Formatos de exportación
EXPORT_FORMATS = {
    "pdf": "application/pdf",
    "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "csv": "text/csv",
    "json": "application/json",
    "html": "text/html",
}

# =============================================================================
# CONFIGURACIÓN DE MONITOREO
# =============================================================================

# Métricas de sistema a monitorear
SYSTEM_METRICS = {
    "api_response_time": {"threshold": 2.0, "unit": "seconds"},
    "prediction_accuracy": {"threshold": 0.50, "unit": "percentage"},
    "database_size": {"threshold": 1000, "unit": "MB"},
    "memory_usage": {"threshold": 80, "unit": "percentage"},
    "cpu_usage": {"threshold": 75, "unit": "percentage"},
    "error_rate": {"threshold": 0.05, "unit": "percentage"},
}

# Alertas del sistema
ALERT_LEVELS = {
    "INFO": {"color": "blue", "priority": 1},
    "WARNING": {"color": "yellow", "priority": 2},
    "ERROR": {"color": "red", "priority": 3},
    "CRITICAL": {"color": "red", "priority": 4},
}

# =============================================================================
# FUNCIONES AVANZADAS DE UTILIDAD
# =============================================================================


def get_bookmaker_name(bookmaker_code: str) -> str:
    """
    Obtiene el nombre completo de un bookmaker.

    Args:
        bookmaker_code: Código del bookmaker

    Returns:
        Nombre completo del bookmaker
    """
    return BOOKMAKERS.get(bookmaker_code, "Unknown Bookmaker")


def get_betting_limits(bookmaker_code: str) -> Dict[str, int]:
    """
    Obtiene los límites de apuesta de un bookmaker.

    Args:
        bookmaker_code: Código del bookmaker

    Returns:
        Diccionario con límites min y max
    """
    return BETTING_LIMITS.get(bookmaker_code, {"min": 1, "max": 1000})


def calculate_overround(odds: Dict[str, float]) -> float:
    """
    Calcula el overround (margen) de un conjunto de cuotas.

    Args:
        odds: Diccionario con cuotas {home: 2.0, draw: 3.5, away: 4.0}

    Returns:
        Overround como porcentaje
    """
    if not odds or len(odds) == 0:
        return 0.0

    total_probability = sum(1 / odd for odd in odds.values() if odd > 0)
    overround = (total_probability - 1.0) * 100
    return max(0.0, overround)


def is_arbitrage_opportunity(odds_dict: Dict[str, Dict[str, float]]) -> bool:
    """
    Detecta si existe una oportunidad de arbitraje.

    Args:
        odds_dict: Cuotas de múltiples bookmakers

    Returns:
        True si hay arbitraje disponible
    """
    if not odds_dict or len(odds_dict) < 2:
        return False

    # Encontrar las mejores cuotas para cada resultado
    best_odds = {}
    for bookmaker, odds in odds_dict.items():
        for outcome, odd in odds.items():
            if outcome not in best_odds or odd > best_odds[outcome]:
                best_odds[outcome] = odd

    # Calcular overround con las mejores cuotas
    overround = calculate_overround(best_odds)
    return overround < -0.5  # Arbitraje si overround es negativo


def get_kelly_multiplier(strategy: str) -> float:
    """
    Obtiene el multiplicador de Kelly para una estrategia.

    Args:
        strategy: Estrategia de apuesta

    Returns:
        Multiplicador de Kelly
    """
    return KELLY_CRITERIA.get(strategy, 0.25)


def is_value_bet(
    predicted_prob: float, odds: float, min_value: float = 0.05
) -> bool:
    """
    Determina si una apuesta tiene valor.

    Args:
        predicted_prob: Probabilidad predicha por el modelo
        odds: Cuotas del bookmaker
        min_value: Valor mínimo requerido

    Returns:
        True si la apuesta tiene valor
    """
    if predicted_prob <= 0 or odds <= 1:
        return False

    implied_prob = 1 / odds
    value = (predicted_prob - implied_prob) / implied_prob
    return value >= min_value


def get_feature_importance_tier(feature_name: str) -> str:
    """
    Obtiene el tier de importancia de una feature.

    Args:
        feature_name: Nombre de la feature

    Returns:
        Tier de importancia ('high', 'medium', 'low')
    """
    high_importance = ["team_form", "attack_strength", "defense_strength"]
    medium_importance = ["home_advantage", "h2h_record", "league_position"]

    if any(important in feature_name for important in high_importance):
        return "high"
    elif any(important in feature_name for important in medium_importance):
        return "medium"
    else:
        return "low"


def validate_football_score(home_goals: int, away_goals: int) -> bool:
    """
    Valida que un marcador de fútbol sea realista.

    Args:
        home_goals: Goles del equipo local
        away_goals: Goles del equipo visitante

    Returns:
        True si el marcador es válido
    """
    max_goals = DATA_VALIDATION_LIMITS["max_goals_per_team"]

    return (
        0 <= home_goals <= max_goals
        and 0 <= away_goals <= max_goals
        and (home_goals + away_goals) <= max_goals
    )


def get_match_importance_factor(
    league: str, match_type: str = "regular"
) -> float:
    """
    Calcula el factor de importancia de un partido.

    Args:
        league: Liga del partido
        match_type: Tipo de partido ('regular', 'derby', 'final', 'relegation')

    Returns:
        Factor de importancia (1.0 = normal, >1.0 = más importante)
    """
    base_importance = 1.0

    # Ajustes por tipo de partido
    importance_multipliers = {
        "regular": 1.0,
        "derby": 1.3,  # Derbis son más importantes
        "final": 2.0,  # Finales muy importantes
        "relegation": 1.5,  # Partidos de descenso importantes
        "title_decider": 1.8,  # Partidos que deciden el título
    }

    # Ajustes por liga
    league_multipliers = {
        "CL": 1.5,  # Champions League más importante
        "EL": 1.3,  # Europa League importante
        "PL": 1.2,  # Premier League alta importancia
        "PD": 1.2,  # La Liga alta importancia
        "SA": 1.1,  # Serie A importancia media-alta
        "BL1": 1.1,  # Bundesliga importancia media-alta
        "FL1": 1.0,  # Ligue 1 importancia normal
    }

    match_multiplier = importance_multipliers.get(match_type, 1.0)
    league_multiplier = league_multipliers.get(league, 1.0)

    return base_importance * match_multiplier * league_multiplier


# =============================================================================
# EXPORTACIONES ACTUALIZADAS
# =============================================================================

__all__ = [
    # Información del sistema
    "SYSTEM_INFO",
    "API_CONFIG",
    "RATE_LIMITS",
    # Ligas y equipos
    "LEAGUE_CODES",
    "LEAGUES_BY_COUNTRY",
    "TEAMS_PER_LEAGUE",
    "MATCHES_PER_SEASON",
    "ALL_KNOWN_TEAMS",
    "PREMIER_LEAGUE_TEAMS",
    "LA_LIGA_TEAMS",
    "SERIE_A_TEAMS",
    "BUNDESLIGA_TEAMS",
    "LIGUE_1_TEAMS",
    # Enums
    "MatchResult",
    "MatchStatus",
    "BettingMarket",
    "OddsFormat",
    # ML y modelos
    "ML_ALGORITHMS",
    "DEFAULT_HYPERPARAMETERS",
    "TRAINING_FEATURES",
    "PREDICTION_TARGETS",
    "CATEGORICAL_FEATURES",
    "NUMERICAL_FEATURES",
    "DERIVED_FEATURES",
    "FEATURE_WEIGHTS",
    # Métricas y validación
    "CONFIDENCE_LEVELS",
    "MODEL_QUALITY_THRESHOLDS",
    "DATA_VALIDATION_LIMITS",
    "FOOTBALL_DISTRIBUTIONS",
    "FOOTBALL_CORRELATIONS",
    # Configuración de apuestas
    "BOOKMAKERS",
    "BETTING_LIMITS",
    "BOOKMAKER_MARGINS",
    "KELLY_CRITERIA",
    "VALUE_BETTING_LIMITS",
    "BETTING_STRATEGIES",
    # Live tracking
    "UPDATE_INTERVALS",
    "LIVE_MATCH_STATES",
    "TRACKED_EVENTS",
    # Backtesting y análisis
    "BACKTESTING_PERIODS",
    "BACKTESTING_METRICS",
    # Reportes y monitoreo
    "REPORT_TYPES",
    "EXPORT_FORMATS",
    "SYSTEM_METRICS",
    "ALERT_LEVELS",
    # Configuración técnica
    "DATABASE_TABLES",
    "SQLITE_CONFIG",
    "LOG_LEVELS",
    "LOG_CONFIG",
    "REDIS_CONFIG",
    "CACHE_TTL",
    "WEBSOCKET_CONFIG",
    "SECURITY_HEADERS",
    "API_RATE_LIMITS",
    # Mensajes
    "ERROR_MESSAGES",
    "SUCCESS_MESSAGES",
    "NOTIFICATION_TYPES",
    # Configuración de entorno
    "TEST_CONFIG",
    "DEV_CONFIG",
    "PROD_CONFIG",
    # Funciones de utilidad básicas
    "get_league_teams",
    "is_valid_league",
    "get_league_name",
    "get_confidence_level",
    "get_algorithm_name",
    # Funciones de utilidad avanzadas
    "get_bookmaker_name",
    "get_betting_limits",
    "calculate_overround",
    "is_arbitrage_opportunity",
    "get_kelly_multiplier",
    "is_value_bet",
    "get_feature_importance_tier",
    "validate_football_score",
    "get_match_importance_factor",
]
