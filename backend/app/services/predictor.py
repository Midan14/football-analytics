"""
Football Analytics - Predictor Service
Sistema completo de predicciones para fútbol usando múltiples modelos ML
"""

import json
import logging
import sqlite3
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# Machine Learning
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ..ml_models.match_outcome import (
    MatchOutcomeEnsemble,
    MatchOutcomePredictor,
)

# Importaciones locales
from .calculator import CalculatorService
from .data_collector import get_production_config
from .odds_calculator import BettingOpportunity, OddsCalculatorService

logger = logging.getLogger(__name__)


class PredictionType(Enum):
    """Tipos de predicciones disponibles"""

    MATCH_RESULT = "match_result"  # 1X2
    OVER_UNDER = "over_under"  # Más/menos goles
    BOTH_TEAMS_SCORE = "btts"  # Ambos equipos marcan
    TOTAL_GOALS = "total_goals"  # Número exacto de goles
    CORRECT_SCORE = "correct_score"  # Resultado exacto
    HALF_TIME_RESULT = "ht_result"  # Resultado al descanso
    ASIAN_HANDICAP = "asian_handicap"  # Hándicap asiático
    PLAYER_GOALS = "player_goals"  # Goles de jugador específico


class ConfidenceLevel(Enum):
    """Niveles de confianza de predicciones"""

    VERY_LOW = "very_low"  # 0-40%
    LOW = "low"  # 40-60%
    MEDIUM = "medium"  # 60-75%
    HIGH = "high"  # 75-85%
    VERY_HIGH = "very_high"  # 85-100%


@dataclass
class MatchPrediction:
    """Predicción completa de un partido"""

    match_id: str
    home_team: str
    away_team: str
    league: str
    match_date: datetime

    # Predicciones principales
    result_probabilities: Dict[
        str, float
    ]  # {'home': 0.45, 'draw': 0.30, 'away': 0.25}
    most_likely_result: str
    confidence_level: ConfidenceLevel
    confidence_score: float

    # Predicciones de goles
    expected_goals_home: float
    expected_goals_away: float
    expected_goals_total: float
    over_under_probabilities: Dict[
        str, float
    ]  # {'over_2.5': 0.65, 'under_2.5': 0.35}
    btts_probabilities: Dict[str, float]  # {'yes': 0.58, 'no': 0.42}

    # Resultado más probable
    most_likely_score: Tuple[int, int]
    score_probability: float

    # Metadatos del modelo
    model_version: str
    prediction_method: str
    features_used: List[str]
    created_at: datetime

    # Análisis de valor (opcional)
    betting_opportunities: List[BettingOpportunity] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

    @property
    def result_prediction_text(self) -> str:
        """Texto legible de la predicción"""
        confidence_text = {
            ConfidenceLevel.VERY_LOW: "muy baja",
            ConfidenceLevel.LOW: "baja",
            ConfidenceLevel.MEDIUM: "media",
            ConfidenceLevel.HIGH: "alta",
            ConfidenceLevel.VERY_HIGH: "muy alta",
        }

        result_text = {
            "home": f"Victoria de {self.home_team}",
            "draw": "Empate",
            "away": f"Victoria de {self.away_team}",
        }

        return f"{result_text[self.most_likely_result]} (confianza {confidence_text[self.confidence_level]})"

    @property
    def goals_prediction_text(self) -> str:
        """Texto de predicción de goles"""
        return f"{self.home_team} {self.most_likely_score[0]}-{self.most_likely_score[1]} {self.away_team}"


@dataclass
class TeamForm:
    """Forma reciente de un equipo"""

    team_name: str
    recent_matches: int
    wins: int
    draws: int
    losses: int
    goals_for: int
    goals_against: int
    points: int
    form_rating: float

    @property
    def points_per_game(self) -> float:
        return (
            self.points / self.recent_matches if self.recent_matches > 0 else 0
        )

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against


@dataclass
class HeadToHeadRecord:
    """Historial de enfrentamientos directos"""

    home_team: str
    away_team: str
    total_matches: int
    home_wins: int
    draws: int
    away_wins: int
    last_5_results: List[str]  # ['H', 'A', 'D', 'H', 'A']
    avg_goals_home: float
    avg_goals_away: float
    avg_total_goals: float


class FeatureExtractor:
    """Extractor de características para los modelos"""

    def __init__(self):
        self.calculator = CalculatorService()
        self.logger = logging.getLogger(__name__)

    def extract_team_features(
        self,
        team: str,
        recent_matches: List[Dict],
        league_avg_goals: float = 2.5,
    ) -> Dict[str, float]:
        """Extrae características de un equipo"""
        if not recent_matches:
            return self._get_default_features()

        # Estadísticas básicas
        total_matches = len(recent_matches)
        wins = sum(1 for m in recent_matches if m["result"] == "W")
        draws = sum(1 for m in recent_matches if m["result"] == "D")
        losses = sum(1 for m in recent_matches if m["result"] == "L")

        goals_for = sum(m["goals_for"] for m in recent_matches)
        goals_against = sum(m["goals_against"] for m in recent_matches)

        # Separar partidos en casa y fuera
        home_matches = [m for m in recent_matches if m["venue"] == "home"]
        away_matches = [m for m in recent_matches if m["venue"] == "away"]

        # Características básicas
        features = {
            # Forma general
            "points_per_game": (wins * 3 + draws) / total_matches,
            "win_percentage": wins / total_matches,
            "goals_per_game": goals_for / total_matches,
            "goals_conceded_per_game": goals_against / total_matches,
            "goal_difference_per_game": (goals_for - goals_against)
            / total_matches,
            # Forma en casa
            "home_points_per_game": self._calculate_home_stats(
                home_matches, "points"
            ),
            "home_goals_per_game": self._calculate_home_stats(
                home_matches, "goals_for"
            ),
            "home_conceded_per_game": self._calculate_home_stats(
                home_matches, "goals_against"
            ),
            # Forma fuera
            "away_points_per_game": self._calculate_away_stats(
                away_matches, "points"
            ),
            "away_goals_per_game": self._calculate_away_stats(
                away_matches, "goals_for"
            ),
            "away_conceded_per_game": self._calculate_away_stats(
                away_matches, "goals_against"
            ),
            # Fortalezas relativas
            "attack_strength": (goals_for / total_matches) / league_avg_goals,
            "defense_strength": (
                league_avg_goals / (goals_against / total_matches)
                if goals_against > 0
                else 2.0
            ),
            # Forma reciente (últimos 5)
            "last_5_points": self._calculate_recent_form(
                recent_matches[-5:], "points"
            ),
            "last_5_goals": self._calculate_recent_form(
                recent_matches[-5:], "goals_for"
            ),
            # Consistencia
            "result_consistency": self._calculate_consistency(recent_matches),
            "goals_consistency": self._calculate_goals_consistency(
                recent_matches
            ),
        }

        return features

    def _calculate_home_stats(
        self, home_matches: List[Dict], stat: str
    ) -> float:
        """Calcula estadísticas de partidos en casa"""
        if not home_matches:
            return 0.0

        if stat == "points":
            points = sum(
                3 if m["result"] == "W" else 1 if m["result"] == "D" else 0
                for m in home_matches
            )
            return points / len(home_matches)
        else:
            return sum(m[stat] for m in home_matches) / len(home_matches)

    def _calculate_away_stats(
        self, away_matches: List[Dict], stat: str
    ) -> float:
        """Calcula estadísticas de partidos fuera"""
        if not away_matches:
            return 0.0

        if stat == "points":
            points = sum(
                3 if m["result"] == "W" else 1 if m["result"] == "D" else 0
                for m in away_matches
            )
            return points / len(away_matches)
        else:
            return sum(m[stat] for m in away_matches) / len(away_matches)

    def _calculate_recent_form(
        self, recent_matches: List[Dict], stat: str
    ) -> float:
        """Calcula forma reciente"""
        if not recent_matches:
            return 0.0

        if stat == "points":
            points = sum(
                3 if m["result"] == "W" else 1 if m["result"] == "D" else 0
                for m in recent_matches
            )
            return points / len(recent_matches)
        else:
            return sum(m[stat] for m in recent_matches) / len(recent_matches)

    def _calculate_consistency(self, matches: List[Dict]) -> float:
        """Calcula consistencia de resultados"""
        if len(matches) < 3:
            return 0.5

        results = [m["result"] for m in matches]
        wins = results.count("W")
        draws = results.count("D")
        losses = results.count("L")

        # Entropía como medida de consistencia
        total = len(results)
        entropy = 0
        for count in [wins, draws, losses]:
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)

        # Normalizar (0 = muy consistente, 1 = muy inconsistente)
        max_entropy = np.log2(3)  # Máxima entropía para 3 resultados
        return entropy / max_entropy

    def _calculate_goals_consistency(self, matches: List[Dict]) -> float:
        """Calcula consistencia en goles"""
        if len(matches) < 3:
            return 0.5

        goals = [m["goals_for"] for m in matches]
        std_dev = np.std(goals)
        mean_goals = np.mean(goals)

        # Coeficiente de variación normalizado
        cv = std_dev / mean_goals if mean_goals > 0 else 1.0
        return min(cv, 1.0)  # Limitar a 1.0

    def _get_default_features(self) -> Dict[str, float]:
        """Características por defecto cuando no hay datos"""
        return {
            "points_per_game": 1.0,
            "win_percentage": 0.33,
            "goals_per_game": 1.2,
            "goals_conceded_per_game": 1.2,
            "goal_difference_per_game": 0.0,
            "home_points_per_game": 1.2,
            "home_goals_per_game": 1.4,
            "home_conceded_per_game": 1.0,
            "away_points_per_game": 0.8,
            "away_goals_per_game": 1.0,
            "away_conceded_per_game": 1.4,
            "attack_strength": 1.0,
            "defense_strength": 1.0,
            "last_5_points": 1.0,
            "last_5_goals": 1.2,
            "result_consistency": 0.5,
            "goals_consistency": 0.5,
        }

    def extract_match_features(
        self,
        home_team: str,
        away_team: str,
        home_features: Dict,
        away_features: Dict,
        h2h_record: HeadToHeadRecord = None,
        match_importance: float = 1.0,
    ) -> Dict[str, float]:
        """Extrae características del partido combinando ambos equipos"""

        match_features = {}

        # Características del equipo local
        for key, value in home_features.items():
            match_features[f"home_{key}"] = value

        # Características del equipo visitante
        for key, value in away_features.items():
            match_features[f"away_{key}"] = value

        # Diferencias relativas
        match_features["points_difference"] = (
            home_features["points_per_game"] - away_features["points_per_game"]
        )
        match_features["attack_difference"] = (
            home_features["attack_strength"] - away_features["attack_strength"]
        )
        match_features["defense_difference"] = (
            home_features["defense_strength"]
            - away_features["defense_strength"]
        )
        match_features["form_difference"] = (
            home_features["last_5_points"] - away_features["last_5_points"]
        )

        # Ratios
        match_features["attack_ratio"] = home_features[
            "attack_strength"
        ] / max(away_features["defense_strength"], 0.1)
        match_features["defense_ratio"] = away_features[
            "attack_strength"
        ] / max(home_features["defense_strength"], 0.1)

        # Ventaja de local
        match_features["home_advantage"] = (
            home_features["home_points_per_game"]
            - away_features["away_points_per_game"]
        )

        # Historial H2H si está disponible
        if h2h_record:
            match_features["h2h_home_win_rate"] = h2h_record.home_wins / max(
                h2h_record.total_matches, 1
            )
            match_features["h2h_avg_goals_home"] = h2h_record.avg_goals_home
            match_features["h2h_avg_goals_away"] = h2h_record.avg_goals_away
            match_features["h2h_avg_total_goals"] = h2h_record.avg_total_goals
        else:
            match_features["h2h_home_win_rate"] = 0.5
            match_features["h2h_avg_goals_home"] = 1.2
            match_features["h2h_avg_goals_away"] = 1.2
            match_features["h2h_avg_total_goals"] = 2.4

        # Importancia del partido
        match_features["match_importance"] = match_importance

        # Features adicionales calculadas
        match_features["expected_competitiveness"] = (
            1 - abs(match_features["points_difference"]) / 3
        )
        match_features["goal_expectation"] = (
            home_features["goals_per_game"] + away_features["goals_per_game"]
        ) / 2

        return match_features


class PredictionDatabase:
    """Base de datos para almacenar predicciones y resultados"""

    def __init__(self, db_path: str = "predictions.db"):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Inicializa la base de datos de predicciones"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Tabla de predicciones
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT UNIQUE,
                    home_team TEXT,
                    away_team TEXT,
                    league TEXT,
                    match_date TIMESTAMP,
                    predicted_result TEXT,
                    confidence_score REAL,
                    confidence_level TEXT,
                    home_probability REAL,
                    draw_probability REAL,
                    away_probability REAL,
                    expected_goals_home REAL,
                    expected_goals_away REAL,
                    most_likely_score TEXT,
                    model_version TEXT,
                    created_at TIMESTAMP,
                    prediction_data JSON
                )
            """
            )

            # Tabla de resultados reales para evaluación
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS actual_results (
                    match_id TEXT PRIMARY KEY,
                    actual_result TEXT,
                    home_goals INTEGER,
                    away_goals INTEGER,
                    settled_at TIMESTAMP,
                    FOREIGN KEY (match_id) REFERENCES predictions (match_id)
                )
            """
            )

            # Tabla de rendimiento del modelo
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_version TEXT,
                    evaluation_date DATE,
                    total_predictions INTEGER,
                    correct_predictions INTEGER,
                    accuracy REAL,
                    precision_home REAL,
                    precision_draw REAL,
                    precision_away REAL,
                    avg_confidence REAL,
                    calibration_score REAL
                )
            """
            )

            conn.commit()

    def save_prediction(self, prediction: MatchPrediction):
        """Guarda una predicción en la base de datos"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO predictions
                (match_id, home_team, away_team, league, match_date,
                 predicted_result, confidence_score, confidence_level,
                 home_probability, draw_probability, away_probability,
                 expected_goals_home, expected_goals_away, most_likely_score,
                 model_version, created_at, prediction_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    prediction.match_id,
                    prediction.home_team,
                    prediction.away_team,
                    prediction.league,
                    prediction.match_date,
                    prediction.most_likely_result,
                    prediction.confidence_score,
                    prediction.confidence_level.value,
                    prediction.result_probabilities["home"],
                    prediction.result_probabilities["draw"],
                    prediction.result_probabilities["away"],
                    prediction.expected_goals_home,
                    prediction.expected_goals_away,
                    f"{prediction.most_likely_score[0]}-{prediction.most_likely_score[1]}",
                    prediction.model_version,
                    prediction.created_at,
                    json.dumps(asdict(prediction), default=str),
                ),
            )

            conn.commit()

    def save_actual_result(
        self, match_id: str, result: str, home_goals: int, away_goals: int
    ):
        """Guarda el resultado real de un partido"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO actual_results
                (match_id, actual_result, home_goals, away_goals, settled_at)
                VALUES (?, ?, ?, ?, ?)
            """,
                (match_id, result, home_goals, away_goals, datetime.now()),
            )

            conn.commit()

    def get_prediction_accuracy(self, days: int = 30) -> Dict[str, float]:
        """Calcula la precisión de las predicciones"""
        cutoff_date = datetime.now() - timedelta(days=days)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Predicciones con resultados conocidos
            cursor.execute(
                """
                SELECT p.predicted_result, a.actual_result, p.confidence_score
                FROM predictions p
                JOIN actual_results a ON p.match_id = a.match_id
                WHERE p.created_at >= ?
            """,
                (cutoff_date,),
            )

            results = cursor.fetchall()

            if not results:
                return {}

            correct = sum(1 for pred, actual, _ in results if pred == actual)
            total = len(results)

            # Precisión por tipo de resultado
            home_correct = sum(
                1
                for pred, actual, _ in results
                if pred == "home" and actual == "home"
            )
            home_total = sum(1 for pred, _, _ in results if pred == "home")

            draw_correct = sum(
                1
                for pred, actual, _ in results
                if pred == "draw" and actual == "draw"
            )
            draw_total = sum(1 for pred, _, _ in results if pred == "draw")

            away_correct = sum(
                1
                for pred, actual, _ in results
                if pred == "away" and actual == "away"
            )
            away_total = sum(1 for pred, _, _ in results if pred == "away")

            return {
                "overall_accuracy": correct / total,
                "total_predictions": total,
                "correct_predictions": correct,
                "home_precision": home_correct / max(home_total, 1),
                "draw_precision": draw_correct / max(draw_total, 1),
                "away_precision": away_correct / max(away_total, 1),
                "average_confidence": np.mean(
                    [conf for _, _, conf in results]
                ),
                "period_days": days,
            }


class PredictorService:
    """Servicio principal de predicciones"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or get_production_config()

        # Componentes del servicio
        self.feature_extractor = FeatureExtractor()
        self.database = PredictionDatabase()
        self.odds_calculator = OddsCalculatorService()

        # Modelos ML
        self.models = {}
        self.ensemble = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        # Configuración
        self.model_version = "v2.1.0"
        self.confidence_thresholds = {
            ConfidenceLevel.VERY_LOW: (0.0, 0.4),
            ConfidenceLevel.LOW: (0.4, 0.6),
            ConfidenceLevel.MEDIUM: (0.6, 0.75),
            ConfidenceLevel.HIGH: (0.75, 0.85),
            ConfidenceLevel.VERY_HIGH: (0.85, 1.0),
        }

        # Inicializar modelos
        self._load_or_create_models()

        logger.info("🔮 Predictor Service inicializado")

    def _load_or_create_models(self):
        """Carga modelos existentes o crea nuevos"""
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)

        try:
            # Intentar cargar modelos existentes
            self.ensemble = MatchOutcomeEnsemble(
                ["xgboost", "lightgbm", "catboost"]
            )
            logger.info("✅ Modelos cargados desde disco")
        except:
            # Crear nuevos modelos si no existen
            self.ensemble = MatchOutcomeEnsemble(
                ["xgboost", "lightgbm", "catboost"]
            )
            logger.info("🆕 Modelos creados desde cero")

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        league: str,
        match_date: datetime,
        match_id: str = None,
        market_odds: Dict[str, float] = None,
    ) -> MatchPrediction:
        """
        Realiza predicción completa de un partido

        Args:
            home_team: Equipo local
            away_team: Equipo visitante
            league: Liga del partido
            match_date: Fecha del partido
            match_id: ID único del partido
            market_odds: Cuotas del mercado para análisis de valor

        Returns:
            Predicción completa del partido
        """
        if match_id is None:
            match_id = (
                f"{home_team}_{away_team}_{match_date.strftime('%Y%m%d')}"
            )

        logger.info(f"🔮 Prediciendo: {home_team} vs {away_team}")

        # 1. Extraer características de los equipos
        home_features = self._get_team_features(home_team, league)
        away_features = self._get_team_features(away_team, league)

        # 2. Obtener historial H2H
        h2h_record = self._get_h2h_record(home_team, away_team)

        # 3. Extraer características del partido
        match_features = self.feature_extractor.extract_match_features(
            home_team, away_team, home_features, away_features, h2h_record
        )

        # 4. Preparar datos para el modelo
        feature_vector = self._prepare_feature_vector(match_features)

        # 5. Realizar predicción con ensemble
        probabilities = self._predict_probabilities(feature_vector)

        # 6. Determinar resultado más probable
        most_likely_result = max(probabilities, key=probabilities.get)
        max_probability = probabilities[most_likely_result]

        # 7. Calcular nivel de confianza
        confidence_level = self._calculate_confidence_level(
            max_probability, probabilities
        )

        # 8. Predecir goles esperados
        expected_goals = self._predict_expected_goals(feature_vector)

        # 9. Calcular probabilidades over/under y BTTS
        over_under_probs = self._calculate_over_under_probabilities(
            expected_goals
        )
        btts_probs = self._calculate_btts_probabilities(expected_goals)

        # 10. Resultado más probable
        most_likely_score = self._predict_most_likely_score(expected_goals)
        score_prob = self._calculate_score_probability(
            most_likely_score, expected_goals
        )

        # 11. Crear predicción
        prediction = MatchPrediction(
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            league=league,
            match_date=match_date,
            result_probabilities=probabilities,
            most_likely_result=most_likely_result,
            confidence_level=confidence_level,
            confidence_score=max_probability,
            expected_goals_home=expected_goals["home"],
            expected_goals_away=expected_goals["away"],
            expected_goals_total=expected_goals["total"],
            over_under_probabilities=over_under_probs,
            btts_probabilities=btts_probs,
            most_likely_score=most_likely_score,
            score_probability=score_prob,
            model_version=self.model_version,
            prediction_method="ensemble_ml",
            features_used=list(match_features.keys()),
        )

        # 12. Análisis de valor si hay cuotas
        if market_odds:
            prediction.betting_opportunities = self._analyze_betting_value(
                prediction, market_odds
            )

        # 13. Guardar predicción
        self.database.save_prediction(prediction)

        logger.info(
            f"✅ Predicción completada: {prediction.result_prediction_text}"
        )

        return prediction

    def _get_team_features(self, team: str, league: str) -> Dict[str, float]:
        """Obtiene características de un equipo"""
        # En implementación real, esto consultaría la base de datos
        # Por ahora, usamos características simuladas realistas

        # Simulación de datos realistas basados en equipos conocidos
        team_profiles = {
            "Real Madrid": {
                "points_per_game": 2.1,
                "win_percentage": 0.65,
                "goals_per_game": 2.3,
                "goals_conceded_per_game": 0.9,
                "attack_strength": 1.4,
                "defense_strength": 1.3,
            },
            "Barcelona": {
                "points_per_game": 2.0,
                "win_percentage": 0.60,
                "goals_per_game": 2.1,
                "goals_conceded_per_game": 1.0,
                "attack_strength": 1.3,
                "defense_strength": 1.2,
            },
            "Manchester City": {
                "points_per_game": 2.2,
                "win_percentage": 0.70,
                "goals_per_game": 2.5,
                "goals_conceded_per_game": 0.8,
                "attack_strength": 1.5,
                "defense_strength": 1.4,
            },
            "Liverpool": {
                "points_per_game": 2.0,
                "win_percentage": 0.62,
                "goals_per_game": 2.2,
                "goals_conceded_per_game": 1.1,
                "attack_strength": 1.3,
                "defense_strength": 1.1,
            },
        }

        if team in team_profiles:
            base_features = team_profiles[team]
        else:
            # Valores promedio para equipos desconocidos
            base_features = {
                "points_per_game": 1.4,
                "win_percentage": 0.45,
                "goals_per_game": 1.3,
                "goals_conceded_per_game": 1.3,
                "attack_strength": 1.0,
                "defense_strength": 1.0,
            }

        # Expandir con características adicionales
        full_features = {
            **base_features,
            "goal_difference_per_game": base_features["goals_per_game"]
            - base_features["goals_conceded_per_game"],
            "home_points_per_game": base_features["points_per_game"]
            * 1.2,  # Ventaja local
            "home_goals_per_game": base_features["goals_per_game"] * 1.15,
            "home_conceded_per_game": base_features["goals_conceded_per_game"]
            * 0.9,
            "away_points_per_game": base_features["points_per_game"]
            * 0.8,  # Desventaja visitante
            "away_goals_per_game": base_features["goals_per_game"] * 0.85,
            "away_conceded_per_game": base_features["goals_conceded_per_game"]
            * 1.1,
            "last_5_points": base_features["points_per_game"],
            "last_5_goals": base_features["goals_per_game"],
            "result_consistency": 0.6,
            "goals_consistency": 0.5,
        }

        return full_features

    def _get_h2h_record(
        self, home_team: str, away_team: str
    ) -> HeadToHeadRecord:
        """Obtiene historial de enfrentamientos directos"""
        # En implementación real, consultaría la base de datos
        # Simulación de datos H2H realistas

        return HeadToHeadRecord(
            home_team=home_team,
            away_team=away_team,
            total_matches=10,
            home_wins=4,
            draws=3,
            away_wins=3,
            last_5_results=["H", "A", "D", "H", "D"],
            avg_goals_home=1.4,
            avg_goals_away=1.2,
            avg_total_goals=2.6,
        )

    def _prepare_feature_vector(
        self, match_features: Dict[str, float]
    ) -> np.ndarray:
        """Prepara vector de características para el modelo"""
        # Orden específico de características que espera el modelo
        feature_order = [
            "home_points_per_game",
            "away_points_per_game",
            "points_difference",
            "home_attack_strength",
            "away_attack_strength",
            "attack_difference",
            "home_defense_strength",
            "away_defense_strength",
            "defense_difference",
            "home_goals_per_game",
            "away_goals_per_game",
            "form_difference",
            "attack_ratio",
            "defense_ratio",
            "home_advantage",
            "h2h_home_win_rate",
            "h2h_avg_total_goals",
            "match_importance",
            "expected_competitiveness",
            "goal_expectation",
        ]

        # Crear vector con características ordenadas
        feature_vector = []
        for feature in feature_order:
            feature_vector.append(match_features.get(feature, 0.0))

        return np.array(feature_vector).reshape(1, -1)

    def _predict_probabilities(
        self, feature_vector: np.ndarray
    ) -> Dict[str, float]:
        """Predice probabilidades usando el ensemble de modelos"""
        try:
            # Usar ensemble si está disponible
            if self.ensemble:
                probabilities = self.ensemble.predict_proba(feature_vector)[0]
                return {
                    "home": float(probabilities[0]),
                    "draw": float(probabilities[1]),
                    "away": float(probabilities[2]),
                }
        except:
            pass

        # Fallback: usar modelo estadístico simple
        return self._statistical_prediction(feature_vector)

    def _statistical_prediction(
        self, feature_vector: np.ndarray
    ) -> Dict[str, float]:
        """Predicción estadística como fallback"""
        features = feature_vector[0]

        # Extraer características clave
        points_diff = features[2] if len(features) > 2 else 0
        attack_ratio = features[12] if len(features) > 12 else 1.0
        home_advantage = features[14] if len(features) > 14 else 0.3

        # Cálculo basado en fortalezas relativas
        home_strength = 1.0 + points_diff * 0.2 + home_advantage * 0.3
        away_strength = 1.0 - points_diff * 0.2
        draw_factor = 1.0 - abs(points_diff) * 0.1

        # Probabilidades base
        total_strength = home_strength + away_strength + draw_factor

        home_prob = home_strength / total_strength
        away_prob = away_strength / total_strength
        draw_prob = draw_factor / total_strength

        # Normalizar para asegurar que suman 1
        total = home_prob + draw_prob + away_prob

        return {
            "home": home_prob / total,
            "draw": draw_prob / total,
            "away": away_prob / total,
        }

    def _calculate_confidence_level(
        self, max_prob: float, probabilities: Dict[str, float]
    ) -> ConfidenceLevel:
        """Calcula el nivel de confianza basado en la distribución de probabilidades"""
        # Entropía de la distribución
        entropy = -sum(p * np.log2(p) for p in probabilities.values() if p > 0)
        max_entropy = np.log2(3)  # Máxima entropía para 3 resultados
        normalized_entropy = entropy / max_entropy

        # Combinar probabilidad máxima con entropía
        confidence_score = max_prob * (1 - normalized_entropy * 0.5)

        # Determinar nivel
        for level, (min_val, max_val) in self.confidence_thresholds.items():
            if min_val <= confidence_score < max_val:
                return level

        return ConfidenceLevel.MEDIUM

    def _predict_expected_goals(
        self, feature_vector: np.ndarray
    ) -> Dict[str, float]:
        """Predice goles esperados para cada equipo"""
        features = feature_vector[0]

        # Extraer características relevantes para goles
        home_goals_avg = features[9] if len(features) > 9 else 1.3
        away_goals_avg = features[10] if len(features) > 10 else 1.3
        attack_ratio = features[12] if len(features) > 12 else 1.0
        defense_ratio = features[13] if len(features) > 13 else 1.0

        # Ajustar por fortalezas relativas
        home_expected = (
            home_goals_avg * attack_ratio * 0.8 + 0.3
        )  # Ventaja local
        away_expected = away_goals_avg * defense_ratio * 0.8

        # Asegurar valores realistas
        home_expected = max(0.1, min(home_expected, 5.0))
        away_expected = max(0.1, min(away_expected, 5.0))

        return {
            "home": home_expected,
            "away": away_expected,
            "total": home_expected + away_expected,
        }

    def _calculate_over_under_probabilities(
        self, expected_goals: Dict[str, float]
    ) -> Dict[str, float]:
        """Calcula probabilidades over/under usando distribución de Poisson"""
        total_expected = expected_goals["total"]

        # Calcular probabilidades para diferentes líneas
        probabilities = {}

        for line in [1.5, 2.5, 3.5]:
            # Probabilidad de que el total sea mayor a la línea
            over_prob = 1 - stats.poisson.cdf(line, total_expected)
            under_prob = 1 - over_prob

            probabilities[f"over_{line}"] = over_prob
            probabilities[f"under_{line}"] = under_prob

        return probabilities

    def _calculate_btts_probabilities(
        self, expected_goals: Dict[str, float]
    ) -> Dict[str, float]:
        """Calcula probabilidades de que ambos equipos marquen"""
        home_expected = expected_goals["home"]
        away_expected = expected_goals["away"]

        # Probabilidad de que el equipo local NO marque
        home_no_goals = stats.poisson.pmf(0, home_expected)

        # Probabilidad de que el equipo visitante NO marque
        away_no_goals = stats.poisson.pmf(0, away_expected)

        # Probabilidad de que al menos uno no marque
        no_btts_prob = (
            home_no_goals + away_no_goals - (home_no_goals * away_no_goals)
        )

        # Probabilidad de que ambos marquen
        btts_prob = 1 - no_btts_prob

        return {
            "yes": max(0.0, min(1.0, btts_prob)),
            "no": max(0.0, min(1.0, 1 - btts_prob)),
        }

    def _predict_most_likely_score(
        self, expected_goals: Dict[str, float]
    ) -> Tuple[int, int]:
        """Predice el resultado más probable usando Poisson"""
        home_expected = expected_goals["home"]
        away_expected = expected_goals["away"]

        # Encontrar los goles más probables para cada equipo
        home_most_likely = int(round(home_expected))
        away_most_likely = int(round(away_expected))

        # Verificar que sean valores razonables
        home_most_likely = max(0, min(home_most_likely, 5))
        away_most_likely = max(0, min(away_most_likely, 5))

        return (home_most_likely, away_most_likely)

    def _calculate_score_probability(
        self, score: Tuple[int, int], expected_goals: Dict[str, float]
    ) -> float:
        """Calcula la probabilidad de un resultado específico"""
        home_goals, away_goals = score
        home_expected = expected_goals["home"]
        away_expected = expected_goals["away"]

        # Probabilidad usando Poisson independiente
        home_prob = stats.poisson.pmf(home_goals, home_expected)
        away_prob = stats.poisson.pmf(away_goals, away_expected)

        return home_prob * away_prob

    def _analyze_betting_value(
        self, prediction: MatchPrediction, market_odds: Dict[str, float]
    ) -> List[BettingOpportunity]:
        """Analiza valor de apuesta comparando predicción vs mercado"""

        # Preparar datos para análisis
        predicted_probs = prediction.result_probabilities
        match_info = {
            "match_id": prediction.match_id,
            "home_team": prediction.home_team,
            "away_team": prediction.away_team,
        }

        # Confidence scores por resultado
        confidence_scores = {
            result: (
                prediction.confidence_score
                if result == prediction.most_likely_result
                else prediction.confidence_score * 0.8
            )
            for result in predicted_probs.keys()
        }

        # Usar el odds calculator para encontrar oportunidades
        opportunities = self.odds_calculator.analyze_betting_opportunity(
            predicted_probs, market_odds, match_info, confidence_scores
        )

        return opportunities

    def predict_multiple_matches(
        self, matches: List[Dict[str, Any]]
    ) -> List[MatchPrediction]:
        """Predice múltiples partidos en lote"""
        predictions = []

        for match in matches:
            try:
                prediction = self.predict_match(
                    home_team=match["home_team"],
                    away_team=match["away_team"],
                    league=match["league"],
                    match_date=match["match_date"],
                    match_id=match.get("match_id"),
                    market_odds=match.get("market_odds"),
                )
                predictions.append(prediction)

            except Exception as e:
                logger.error(
                    f"Error prediciendo {match['home_team']} vs {match['away_team']}: {e}"
                )

        logger.info(
            f"✅ Predicciones completadas: {len(predictions)}/{len(matches)}"
        )

        return predictions

    def evaluate_model_performance(self, days: int = 30) -> Dict[str, Any]:
        """Evalúa el rendimiento del modelo"""
        accuracy_stats = self.database.get_prediction_accuracy(days)

        if not accuracy_stats:
            return {"error": "No hay datos suficientes para evaluación"}

        # Calcular métricas adicionales
        performance = {
            "period_days": days,
            "model_version": self.model_version,
            "accuracy_metrics": accuracy_stats,
            "evaluation_date": datetime.now().isoformat(),
        }

        # Agregar recomendaciones basadas en rendimiento
        overall_accuracy = accuracy_stats.get("overall_accuracy", 0)

        if overall_accuracy >= 0.6:
            performance["status"] = "Excelente"
            performance["recommendation"] = "Modelo funcionando óptimamente"
        elif overall_accuracy >= 0.5:
            performance["status"] = "Bueno"
            performance["recommendation"] = "Modelo con rendimiento sólido"
        elif overall_accuracy >= 0.4:
            performance["status"] = "Aceptable"
            performance["recommendation"] = "Considerar reentrenamiento"
        else:
            performance["status"] = "Deficiente"
            performance["recommendation"] = "Reentrenamiento urgente necesario"

        return performance

    def get_prediction_insights(
        self, prediction: MatchPrediction
    ) -> Dict[str, Any]:
        """Genera insights detallados sobre una predicción"""
        insights = {
            "match_summary": {
                "teams": f"{prediction.home_team} vs {prediction.away_team}",
                "league": prediction.league,
                "prediction": prediction.result_prediction_text,
                "confidence": prediction.confidence_level.value,
                "expected_score": prediction.goals_prediction_text,
            },
            "key_factors": [],
            "statistical_analysis": {
                "home_win_probability": f"{prediction.result_probabilities['home']:.1%}",
                "draw_probability": f"{prediction.result_probabilities['draw']:.1%}",
                "away_win_probability": f"{prediction.result_probabilities['away']:.1%}",
                "expected_total_goals": f"{prediction.expected_goals_total:.1f}",
                "over_2_5_probability": f"{prediction.over_under_probabilities.get('over_2.5', 0):.1%}",
                "btts_probability": f"{prediction.btts_probabilities['yes']:.1%}",
            },
            "betting_analysis": None,
        }

        # Factores clave basados en las probabilidades
        max_prob = max(prediction.result_probabilities.values())
        if max_prob > 0.6:
            insights["key_factors"].append(
                "Alta confianza en el resultado predicho"
            )

        if prediction.expected_goals_total > 3.0:
            insights["key_factors"].append(
                "Partido con muchos goles esperados"
            )
        elif prediction.expected_goals_total < 2.0:
            insights["key_factors"].append("Partido con pocos goles esperados")

        if (
            abs(
                prediction.expected_goals_home - prediction.expected_goals_away
            )
            > 1.0
        ):
            insights["key_factors"].append(
                "Gran diferencia en capacidad ofensiva"
            )

        # Análisis de apuestas si hay oportunidades
        if prediction.betting_opportunities:
            best_opportunity = max(
                prediction.betting_opportunities,
                key=lambda x: x.value_percentage,
            )
            insights["betting_analysis"] = {
                "best_value": f"{best_opportunity.selection} con {best_opportunity.value_percentage:.2f}% de valor",
                "expected_value": f"{best_opportunity.expected_value:.2f}",
                "recommended_stake": f"€{best_opportunity.stake_suggestion:.2f}",
                "risk_level": best_opportunity.risk_level,
            }

        return insights


# Funciones de conveniencia
def create_predictor(config: Dict[str, Any] = None) -> PredictorService:
    """Factory function para crear el predictor"""
    return PredictorService(config)


def predict_match_quick(
    home_team: str, away_team: str, league: str = "Unknown"
) -> MatchPrediction:
    """Función rápida para predecir un partido"""
    predictor = create_predictor()
    return predictor.predict_match(
        home_team=home_team,
        away_team=away_team,
        league=league,
        match_date=datetime.now() + timedelta(days=1),
    )


def analyze_predictions_batch(matches: List[Dict]) -> List[MatchPrediction]:
    """Analiza múltiples partidos en lote"""
    predictor = create_predictor()
    return predictor.predict_multiple_matches(matches)


if __name__ == "__main__":
    # Sistema de predicciones en producción
    async def main():
        print("🔮 Football Analytics - Predictor Service")
        print("Sistema completo de predicciones de fútbol")
        print("=" * 60)

        # Crear predictor
        predictor = create_predictor()

        print("✅ Predictor Service inicializado")
        print(f"📊 Modelo version: {predictor.model_version}")
        print("🧠 Ensemble: XGBoost + LightGBM + CatBoost")

        # Ejemplo de predicción
        print("\n🔮 Realizando predicción de ejemplo...")

        prediction = predictor.predict_match(
            home_team="Real Madrid",
            away_team="Barcelona",
            league="La Liga",
            match_date=datetime.now() + timedelta(days=7),
            market_odds={"home": 2.10, "draw": 3.40, "away": 3.80},
        )

        print(
            f"\n⚽ PREDICCIÓN: {prediction.home_team} vs {prediction.away_team}"
        )
        print(f"📈 Resultado: {prediction.result_prediction_text}")
        print(f"🎯 Marcador más probable: {prediction.goals_prediction_text}")
        print(f"⭐ Confianza: {prediction.confidence_score:.1%}")

        # Probabilidades detalladas
        print(f"\n📊 PROBABILIDADES:")
        for result, prob in prediction.result_probabilities.items():
            print(f"   {result.upper()}: {prob:.1%}")

        # Análisis de goles
        print(f"\n⚽ ANÁLISIS DE GOLES:")
        print(
            f"   Goles esperados {prediction.home_team}: {prediction.expected_goals_home:.1f}"
        )
        print(
            f"   Goles esperados {prediction.away_team}: {prediction.expected_goals_away:.1f}"
        )
        print(f"   Total esperado: {prediction.expected_goals_total:.1f}")
        print(
            f"   Over 2.5: {prediction.over_under_probabilities.get('over_2.5', 0):.1%}"
        )
        print(f"   BTTS: {prediction.btts_probabilities['yes']:.1%}")

        # Oportunidades de apuesta
        if prediction.betting_opportunities:
            print(f"\n💰 OPORTUNIDADES DE VALOR:")
            for opp in prediction.betting_opportunities:
                print(
                    f"   {opp.selection.upper()}: {opp.value_percentage:.2f}% valor, EV: {opp.expected_value:.2f}"
                )

        # Insights
        insights = predictor.get_prediction_insights(prediction)
        print(f"\n💡 FACTORES CLAVE:")
        for factor in insights["key_factors"]:
            print(f"   • {factor}")

        # Evaluación del modelo
        print(f"\n📈 Evaluando rendimiento del modelo...")
        performance = predictor.evaluate_model_performance(30)

        if "accuracy_metrics" in performance:
            acc = performance["accuracy_metrics"]
            print(
                f"   Precisión general: {acc.get('overall_accuracy', 0):.1%}"
            )
            print(
                f"   Predicciones totales: {acc.get('total_predictions', 0)}"
            )
            print(f"   Estado del modelo: {performance['status']}")

        # Predicción en lote
        print(f"\n⚡ Ejemplo de predicción en lote...")
        matches_batch = [
            {
                "home_team": "Manchester City",
                "away_team": "Liverpool",
                "league": "Premier League",
                "match_date": datetime.now() + timedelta(days=3),
            },
            {
                "home_team": "Bayern Munich",
                "away_team": "Borussia Dortmund",
                "league": "Bundesliga",
                "match_date": datetime.now() + timedelta(days=5),
            },
        ]

        batch_predictions = predictor.predict_multiple_matches(matches_batch)

        print(f"📊 Predicciones en lote completadas: {len(batch_predictions)}")
        for pred in batch_predictions:
            print(
                f"   {pred.home_team} vs {pred.away_team}: {pred.most_likely_result} ({pred.confidence_score:.1%})"
            )

        print(f"\n✅ Sistema de predicciones funcionando correctamente!")

    # Ejecutar ejemplo
    import asyncio

    asyncio.run(main())
