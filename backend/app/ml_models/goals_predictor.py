"""
Goals Predictor for Football Analytics

Este módulo contiene modelos especializados en predecir goles en partidos de fútbol,
incluyendo total de goles, goles por equipo, Over/Under y Both Teams to Score.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.stats import poisson

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import PoissonRegressor
from xgboost import XGBClassifier, XGBRegressor

from app.ml_models import MLConfig

# Base model
from .base_model import (
    BaseFootballPredictor,
    FootballClassifierMixin,
    FootballRegressorMixin,
)

# Configurar logging
logger = logging.getLogger(__name__)

# =====================================================
# PREDICTOR DE TOTAL DE GOLES
# =====================================================


class TotalGoalsPredictor(BaseFootballPredictor, FootballRegressorMixin):
    """
    Predictor para el total de goles en un partido.

    Predice el número total de goles que se marcarán en un partido,
    útil para mercados Over/Under.
    """

    def __init__(self, model_algorithm: str = "xgboost"):
        """
        Inicializar predictor de goles totales.

        Args:
            model_algorithm: Algoritmo ML ('xgboost', 'lightgbm', 'poisson', 'catboost')
        """
        super().__init__(
            model_name="total_goals_predictor",
            prediction_type="total_goals",
            model_type="regressor",
        )

        self.model_algorithm = model_algorithm

        # Features específicas para predicción de goles
        self.add_football_features(
            team_features=[
                "home_team_goals_scored_avg",
                "away_team_goals_scored_avg",
                "home_team_goals_conceded_avg",
                "away_team_goals_conceded_avg",
                "home_team_attack_strength",
                "away_team_attack_strength",
                "home_team_defense_strength",
                "away_team_defense_strength",
                "home_team_form_goals",
                "away_team_form_goals",
                "home_team_goals_last_5",
                "away_team_goals_last_5",
            ],
            player_features=[
                "home_top_scorer_available",
                "away_top_scorer_available",
                "home_team_injured_attackers",
                "away_team_injured_attackers",
                "home_team_key_players_goals",
                "away_team_key_players_goals",
                "home_team_striker_form",
                "away_team_striker_form",
            ],
            match_features=[
                "head_to_head_goals_avg",
                "league_goals_per_game_avg",
                "match_importance",
                "weather_impact_on_goals",
                "stadium_goals_factor",
                "referee_impact_on_goals",
                "season_period_goals_trend",
                "tv_match_factor",
            ],
        )

        logger.info(
            f"Initialized Total Goals Predictor with {model_algorithm}"
        )

    def _create_model(self):
        """Crear modelo según algoritmo especificado."""
        if self.model_algorithm == "xgboost":
            return XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                objective="count:poisson",  # Ideal para conteos como goles
            )

        elif self.model_algorithm == "lightgbm":
            return LGBMRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                objective="poisson",
            )

        elif self.model_algorithm == "catboost":
            return CatBoostRegressor(
                iterations=300,
                depth=6,
                learning_rate=0.08,
                random_state=self.random_state,
                loss_function="Poisson",
                verbose=False,
            )

        elif self.model_algorithm == "poisson":
            return PoissonRegressor(alpha=0.5, max_iter=1000)

        elif self.model_algorithm == "randomforest":
            return RandomForestRegressor(
                n_estimators=200,
                max_depth=8,
                min_samples_split=5,
                random_state=self.random_state,
            )

        else:
            raise ValueError(f"Unknown algorithm: {self.model_algorithm}")

    def _get_default_hyperparameters(self):
        """Hiperparámetros por defecto según algoritmo."""
        params_map = {
            "xgboost": {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.08,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
            "lightgbm": {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.08,
                "subsample": 0.8,
            },
            "catboost": {"iterations": 300, "depth": 6, "learning_rate": 0.08},
            "poisson": {"alpha": 0.5},
            "randomforest": {
                "n_estimators": 200,
                "max_depth": 8,
                "min_samples_split": 5,
            },
        }
        return params_map.get(self.model_algorithm, {})

    def _get_hyperparameter_grid(self):
        """Grid para optimización de hiperparámetros."""
        grids_map = {
            "xgboost": {
                "n_estimators": [200, 300, 400],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.05, 0.08, 0.1],
                "subsample": [0.7, 0.8, 0.9],
            },
            "lightgbm": {
                "n_estimators": [200, 300, 400],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.05, 0.08, 0.1],
            },
            "catboost": {
                "iterations": [200, 300, 400],
                "depth": [4, 6, 8],
                "learning_rate": [0.05, 0.08, 0.1],
            },
            "poisson": {"alpha": [0.1, 0.5, 1.0, 2.0]},
            "randomforest": {
                "n_estimators": [150, 200, 250],
                "max_depth": [6, 8, 10],
                "min_samples_split": [3, 5, 7],
            },
        }
        return grids_map.get(self.model_algorithm, {})

    def create_goals_features(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Crear features específicas para predicción de goles.

        Args:
            matches_df: DataFrame con datos de partidos

        Returns:
            pd.DataFrame: DataFrame con features de goles
        """
        df = self.create_football_features(matches_df.copy())

        # Features ofensivas por equipo
        if "home_team_id" in df.columns:
            # Promedio de goles por equipo
            df["home_team_goals_avg"] = df.groupby("home_team_id")[
                "home_goals"
            ].transform("mean")
            df["away_team_goals_avg"] = df.groupby("away_team_id")[
                "away_goals"
            ].transform("mean")

            # Promedio de goles concedidos
            df["home_team_conceded_avg"] = df.groupby("home_team_id")[
                "away_goals"
            ].transform("mean")
            df["away_team_conceded_avg"] = df.groupby("away_team_id")[
                "home_goals"
            ].transform("mean")

            # Fortaleza ofensiva y defensiva
            league_avg_goals = df["total_goals"].mean()
            df["home_attack_strength"] = (
                df["home_team_goals_avg"] / league_avg_goals
            )
            df["away_attack_strength"] = (
                df["away_team_goals_avg"] / league_avg_goals
            )
            df["home_defense_strength"] = 1 - (
                df["home_team_conceded_avg"] / league_avg_goals
            )
            df["away_defense_strength"] = 1 - (
                df["away_team_conceded_avg"] / league_avg_goals
            )

            # Rolling averages (forma reciente)
            df["home_goals_last_5"] = df.groupby("home_team_id")[
                "home_goals"
            ].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
            df["away_goals_last_5"] = df.groupby("away_team_id")[
                "away_goals"
            ].transform(lambda x: x.rolling(window=5, min_periods=1).mean())

        # Features de enfrentamientos directos
        if "home_team_id" in df.columns and "away_team_id" in df.columns:
            # Crear ID del enfrentamiento
            df["h2h_id"] = df[["home_team_id", "away_team_id"]].apply(
                lambda x: f"{min(x)}-{max(x)}", axis=1
            )

            # Promedio de goles en enfrentamientos previos
            df["h2h_goals_avg"] = df.groupby("h2h_id")[
                "total_goals"
            ].transform("mean")
            df["h2h_home_goals_avg"] = df.groupby(
                ["home_team_id", "away_team_id"]
            )["home_goals"].transform("mean")
            df["h2h_away_goals_avg"] = df.groupby(
                ["home_team_id", "away_team_id"]
            )["away_goals"].transform("mean")

        # Features temporales específicas para goles
        if "match_date" in df.columns:
            df["match_date"] = pd.to_datetime(df["match_date"])

            # Mes del año (algunos meses son más goleadores)
            df["month"] = df["match_date"].dt.month
            df["is_spring_summer"] = (
                (df["month"] >= 3) & (df["month"] <= 8)
            ).astype(int)

            # Período de la temporada
            df["season_week"] = df["match_date"].dt.isocalendar().week
            df["early_season"] = (df["season_week"] <= 10).astype(int)
            df["mid_season"] = (
                (df["season_week"] > 10) & (df["season_week"] <= 35)
            ).astype(int)
            df["late_season"] = (df["season_week"] > 35).astype(int)

        # Features de liga y competición
        if "league_id" in df.columns:
            # Promedio de goles por liga
            df["league_goals_avg"] = df.groupby("league_id")[
                "total_goals"
            ].transform("mean")

            # Factor de liga (algunas ligas son más goleadoras)
            df["league_goals_factor"] = (
                df["league_goals_avg"] / df["total_goals"].mean()
            )

        # Features del estadio
        if "stadium_id" in df.columns:
            df["stadium_goals_avg"] = df.groupby("stadium_id")[
                "total_goals"
            ].transform("mean")

            # Algunos estadios favorecen más goles
            df["high_scoring_stadium"] = (
                df["stadium_goals_avg"] > df["total_goals"].quantile(0.75)
            ).astype(int)

        # Features meteorológicas
        if "weather" in df.columns:
            # Impacto del clima en los goles
            weather_goals_factor = {
                "sunny": 1.05,
                "cloudy": 1.0,
                "rainy": 0.9,
                "windy": 0.95,
                "snow": 0.8,
            }
            df["weather_goals_factor"] = (
                df["weather"].map(weather_goals_factor).fillna(1.0)
            )

        # Features de importancia del partido
        if "importance" in df.columns:
            # Partidos importantes pueden tener menos goles (más cautelosos)
            importance_goals_factor = {
                "low": 1.1,
                "normal": 1.0,
                "high": 0.9,
                "final": 0.85,
                "derby": 1.05,  # Los derbys pueden ser impredecibles
            }
            df["importance_goals_factor"] = (
                df["importance"].map(importance_goals_factor).fillna(1.0)
            )

        # Features de motivación
        if "round_number" in df.columns:
            # Última jornada puede tener resultados peculiares
            max_round = df["round_number"].max()
            df["is_last_round"] = (df["round_number"] == max_round).astype(int)
            df["is_first_round"] = (df["round_number"] == 1).astype(int)

        logger.info(f"Created goals features, final shape: {df.shape}")
        return df

    def predict_goals_detailed(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Predecir goles con análisis detallado.

        Args:
            X: Features para predicción

        Returns:
            dict: Predicciones detalladas de goles
        """
        # Predicciones básicas
        raw_predictions = self.predict(X)

        # Analizar distribución de goles
        goals_predictions = []
        for pred in raw_predictions:
            exact_goals = max(0, pred)  # No puede ser negativo
            rounded_goals = max(0, round(pred))

            goals_analysis = {
                "predicted_goals_exact": round(exact_goals, 2),
                "predicted_goals_rounded": int(rounded_goals),
                "over_under_analysis": {
                    "over_0_5": "over" if exact_goals > 0.5 else "under",
                    "over_1_5": "over" if exact_goals > 1.5 else "under",
                    "over_2_5": "over" if exact_goals > 2.5 else "under",
                    "over_3_5": "over" if exact_goals > 3.5 else "under",
                    "over_4_5": "over" if exact_goals > 4.5 else "under",
                },
                "goals_probability_distribution": self._calculate_goals_distribution(
                    exact_goals
                ),
                "betting_value_analysis": self._analyze_betting_value(
                    exact_goals
                ),
                "confidence_level": self._calculate_goals_confidence(
                    exact_goals
                ),
            }

            goals_predictions.append(goals_analysis)

        return {
            "predictions": goals_predictions,
            "model_name": self.model_name,
            "algorithm": self.model_algorithm,
            "average_goals_predicted": float(
                np.mean(
                    [p["predicted_goals_exact"] for p in goals_predictions]
                )
            ),
            "high_scoring_matches": len(
                [
                    p
                    for p in goals_predictions
                    if p["predicted_goals_exact"] >= 3.5
                ]
            ),
            "low_scoring_matches": len(
                [
                    p
                    for p in goals_predictions
                    if p["predicted_goals_exact"] <= 1.5
                ]
            ),
            "prediction_timestamp": datetime.now().isoformat(),
        }

    def _calculate_goals_distribution(
        self, predicted_goals: float
    ) -> Dict[str, float]:
        """Calcular distribución de probabilidades usando Poisson."""
        lambda_param = max(0.1, predicted_goals)

        distribution = {}
        for goals in range(0, 8):  # 0 a 7+ goles
            if goals == 7:
                prob = 1 - poisson.cdf(6, lambda_param)
                distribution["7+_goals"] = round(prob, 3)
            else:
                prob = poisson.pmf(goals, lambda_param)
                distribution[f"{goals}_goals"] = round(prob, 3)

        return distribution

    def _analyze_betting_value(self, predicted_goals: float) -> Dict[str, Any]:
        """Analizar valor de apuesta para diferentes mercados."""
        analysis = {
            "best_over_under_bet": None,
            "btts_recommendation": None,
            "total_goals_range": None,
        }

        # Mejor apuesta Over/Under
        thresholds = [0.5, 1.5, 2.5, 3.5, 4.5]
        best_threshold = None
        best_confidence = 0

        for threshold in thresholds:
            if predicted_goals > threshold:
                confidence = min(
                    0.9, (predicted_goals - threshold) / threshold
                )
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_threshold = f"Over {threshold}"
            else:
                confidence = min(
                    0.9, (threshold - predicted_goals) / threshold
                )
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_threshold = f"Under {threshold}"

        analysis["best_over_under_bet"] = {
            "market": best_threshold,
            "confidence": round(best_confidence, 2),
        }

        # Recomendación BTTS (simplificada)
        if predicted_goals >= 2.0:
            analysis["btts_recommendation"] = {
                "bet": "BTTS Yes",
                "reasoning": "High goal expectation suggests both teams likely to score",
            }
        elif predicted_goals <= 1.5:
            analysis["btts_recommendation"] = {
                "bet": "BTTS No",
                "reasoning": "Low goal expectation suggests clean sheet likely",
            }
        else:
            analysis["btts_recommendation"] = {
                "bet": "Avoid BTTS market",
                "reasoning": "Uncertain goal distribution",
            }

        # Rango de goles totales
        if predicted_goals <= 1.5:
            analysis["total_goals_range"] = "0-2 goals"
        elif predicted_goals <= 2.5:
            analysis["total_goals_range"] = "2-3 goals"
        elif predicted_goals <= 3.5:
            analysis["total_goals_range"] = "3-4 goals"
        else:
            analysis["total_goals_range"] = "4+ goals"

        return analysis

    def _calculate_goals_confidence(self, predicted_goals: float) -> str:
        """Calcular nivel de confianza basado en la predicción."""
        # Partidos con predicciones extremas (muy pocos o muchos goles) son menos confiables
        if predicted_goals <= 0.8 or predicted_goals >= 5.0:
            return "low"
        elif 1.8 <= predicted_goals <= 3.2:
            return "high"  # Rango más común y predecible
        else:
            return "medium"


# =====================================================
# PREDICTOR DE GOLES POR EQUIPO
# =====================================================


class TeamGoalsPredictor(BaseFootballPredictor, FootballRegressorMixin):
    """
    Predictor para goles específicos por equipo.

    Predice cuántos goles marcará cada equipo individual.
    """

    def __init__(
        self, team_type: str = "both", model_algorithm: str = "xgboost"
    ):
        """
        Inicializar predictor de goles por equipo.

        Args:
            team_type: Tipo de equipo ('home', 'away', 'both')
            model_algorithm: Algoritmo ML
        """
        model_name = f"{team_type}_goals_predictor"
        super().__init__(
            model_name=model_name,
            prediction_type="team_goals",
            model_type="regressor",
        )

        self.team_type = team_type
        self.model_algorithm = model_algorithm

        # Features específicas para goles por equipo
        self.add_football_features(
            team_features=[
                "team_attack_efficiency",
                "team_conversion_rate",
                "team_shots_on_target_ratio",
                "team_big_chances_created",
                "team_goals_from_set_pieces",
                "team_counter_attack_goals",
                "team_home_away_goal_difference",
                "team_goals_by_half",
            ],
            player_features=[
                "top_scorer_goals_per_game",
                "creative_players_assists",
                "injured_key_attackers",
                "suspended_goal_scorers",
                "new_signings_goal_impact",
                "bench_strength_attack",
            ],
            match_features=[
                "opponent_defensive_record",
                "opponent_clean_sheets",
                "opponent_goals_conceded_trend",
                "tactical_matchup",
                "set_pieces_advantage",
                "pace_vs_defense_strength",
            ],
        )

    def _create_model(self):
        """Crear modelo para predicción por equipo."""
        if self.model_algorithm == "xgboost":
            return XGBRegressor(
                n_estimators=250,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=self.random_state,
                objective="count:poisson",
            )
        elif self.model_algorithm == "lightgbm":
            return LGBMRegressor(
                n_estimators=250,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state,
                objective="poisson",
            )
        else:
            return PoissonRegressor(alpha=0.3)

    def _get_default_hyperparameters(self):
        return {"n_estimators": 250, "max_depth": 5, "learning_rate": 0.1}

    def _get_hyperparameter_grid(self):
        return {
            "n_estimators": [200, 250, 300],
            "max_depth": [4, 5, 6],
            "learning_rate": [0.08, 0.1, 0.12],
        }

    def predict_team_goals_detailed(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Predecir goles por equipo con análisis detallado.

        Args:
            X: Features para predicción

        Returns:
            dict: Predicciones detalladas por equipo
        """
        if self.team_type == "both":
            # Predecir para ambos equipos (necesita dos modelos o features específicas)
            home_predictions = self.predict(X)  # Simplificado para ejemplo
            away_predictions = home_predictions * 0.85  # Factor visitante
        else:
            predictions = self.predict(X)
            if self.team_type == "home":
                home_predictions = predictions
                away_predictions = None
            else:
                away_predictions = predictions
                home_predictions = None

        team_goals_analysis = []
        for i, _ in enumerate(X.index):
            analysis = {}

            if home_predictions is not None:
                home_goals = max(0, home_predictions[i])
                analysis["home_team"] = {
                    "predicted_goals": round(home_goals, 2),
                    "goals_probability": self._calculate_team_goals_probability(
                        home_goals
                    ),
                    "goal_scoring_likelihood": self._categorize_goal_likelihood(
                        home_goals
                    ),
                    "key_metrics": {
                        "expected_conversion": round(
                            home_goals * 0.15, 2
                        ),  # Estimación shots on target
                        "clean_sheet_against": round(
                            1 - min(0.95, home_goals / 3), 2
                        ),
                    },
                }

            if away_predictions is not None:
                away_goals = max(0, away_predictions[i])
                analysis["away_team"] = {
                    "predicted_goals": round(away_goals, 2),
                    "goals_probability": self._calculate_team_goals_probability(
                        away_goals
                    ),
                    "goal_scoring_likelihood": self._categorize_goal_likelihood(
                        away_goals
                    ),
                    "key_metrics": {
                        "expected_conversion": round(away_goals * 0.15, 2),
                        "clean_sheet_against": round(
                            1 - min(0.95, away_goals / 3), 2
                        ),
                    },
                }

            # Análisis combinado si ambos equipos
            if home_predictions is not None and away_predictions is not None:
                home_goals = max(0, home_predictions[i])
                away_goals = max(0, away_predictions[i])

                analysis["match_analysis"] = {
                    "total_goals": round(home_goals + away_goals, 2),
                    "goal_difference": round(abs(home_goals - away_goals), 2),
                    "both_teams_score_probability": self._calculate_btts_probability(
                        home_goals, away_goals
                    ),
                    "dominant_attack": (
                        "home"
                        if home_goals > away_goals * 1.2
                        else (
                            "away"
                            if away_goals > home_goals * 1.2
                            else "balanced"
                        )
                    ),
                    "expected_result": self._determine_expected_result(
                        home_goals, away_goals
                    ),
                }

            team_goals_analysis.append(analysis)

        return {
            "team_type": self.team_type,
            "predictions": team_goals_analysis,
            "model_name": self.model_name,
            "algorithm": self.model_algorithm,
            "prediction_timestamp": datetime.now().isoformat(),
        }

    def _calculate_team_goals_probability(
        self, predicted_goals: float
    ) -> Dict[str, float]:
        """Calcular probabilidades de goles específicas para un equipo."""
        lambda_param = max(0.1, predicted_goals)

        return {
            "0_goals": round(poisson.pmf(0, lambda_param), 3),
            "1_goal": round(poisson.pmf(1, lambda_param), 3),
            "2_goals": round(poisson.pmf(2, lambda_param), 3),
            "3+_goals": round(1 - poisson.cdf(2, lambda_param), 3),
        }

    def _categorize_goal_likelihood(self, predicted_goals: float) -> str:
        """Categorizar la probabilidad de marcar goles."""
        if predicted_goals >= 2.5:
            return "very_high"
        elif predicted_goals >= 1.8:
            return "high"
        elif predicted_goals >= 1.2:
            return "medium"
        elif predicted_goals >= 0.7:
            return "low"
        else:
            return "very_low"

    def _calculate_btts_probability(
        self, home_goals: float, away_goals: float
    ) -> float:
        """Calcular probabilidad de que ambos equipos marquen."""
        # P(home > 0) * P(away > 0)
        home_scores = 1 - poisson.pmf(0, max(0.1, home_goals))
        away_scores = 1 - poisson.pmf(0, max(0.1, away_goals))

        return round(home_scores * away_scores, 3)

    def _determine_expected_result(
        self, home_goals: float, away_goals: float
    ) -> Dict[str, Any]:
        """Determinar resultado esperado basado en goles predichos."""
        goal_difference = home_goals - away_goals

        if goal_difference > 0.5:
            result = "home_win"
            margin = "comfortable" if goal_difference > 1.5 else "narrow"
        elif goal_difference < -0.5:
            result = "away_win"
            margin = "comfortable" if goal_difference < -1.5 else "narrow"
        else:
            result = "draw"
            margin = "tight"

        return {
            "most_likely_result": result,
            "margin_type": margin,
            "goal_difference": round(goal_difference, 2),
            "result_confidence": self._calculate_result_confidence(
                abs(goal_difference)
            ),
        }

    def _calculate_result_confidence(self, goal_difference: float) -> str:
        """Calcular confianza en el resultado basado en diferencia de goles."""
        if goal_difference >= 2.0:
            return "high"
        elif goal_difference >= 1.0:
            return "medium"
        else:
            return "low"


# =====================================================
# PREDICTOR BOTH TEAMS TO SCORE (BTTS)
# =====================================================


class BothTeamsScorePredictor(BaseFootballPredictor, FootballClassifierMixin):
    """
    Predictor especializado en Both Teams to Score (BTTS).

    Predice si ambos equipos marcarán al menos un gol.
    """

    def __init__(self, model_algorithm: str = "xgboost"):
        super().__init__(
            model_name="btts_predictor",
            prediction_type="both_teams_score",
            model_type="classifier",
        )

        self.model_algorithm = model_algorithm

        # Features específicas para BTTS
        self.add_football_features(
            team_features=[
                "team_btts_percentage",
                "team_clean_sheets_ratio",
                "team_failed_to_score_ratio",
                "team_goals_consistency",
                "team_attacking_vs_defensive_balance",
            ],
            player_features=[
                "reliable_goal_scorers_available",
                "defensive_stability_rating",
                "set_piece_specialists_both_teams",
                "injury_impact_on_scoring",
            ],
            match_features=[
                "historical_btts_percentage_h2h",
                "league_btts_average",
                "referee_impact_on_goals",
                "match_attacking_tendency",
            ],
        )

    def _create_model(self):
        """Crear clasificador para BTTS."""
        if self.model_algorithm == "xgboost":
            return XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
            )
        elif self.model_algorithm == "lightgbm":
            return LGBMClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state,
            )
        else:
            return RandomForestClassifier(
                n_estimators=200, max_depth=6, random_state=self.random_state
            )

    def _get_default_hyperparameters(self):
        return {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1}

    def _get_hyperparameter_grid(self):
        return {
            "n_estimators": [150, 200, 250],
            "max_depth": [4, 5, 6],
            "learning_rate": [0.08, 0.1, 0.12],
        }

    def predict_btts_detailed(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Predecir BTTS con análisis detallado.

        Args:
            X: Features para predicción

        Returns:
            dict: Análisis detallado de BTTS
        """
        # Obtener probabilidades
        probabilities = self.predict_proba(X)
        predictions = self.predict(X)

        btts_analysis = []
        for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
            # Asumiendo que clase 1 = BTTS Yes, clase 0 = BTTS No
            btts_yes_prob = proba[1] if len(proba) > 1 else proba[0]
            btts_no_prob = 1 - btts_yes_prob

            analysis = {
                "btts_prediction": "yes" if pred == 1 else "no",
                "btts_yes_probability": round(btts_yes_prob, 3),
                "btts_no_probability": round(btts_no_prob, 3),
                "confidence_level": self.get_prediction_confidence_level(
                    max(btts_yes_prob, btts_no_prob)
                ),
                "betting_recommendation": self._generate_btts_betting_advice(
                    btts_yes_prob
                ),
                "risk_assessment": {
                    "high_confidence_yes": btts_yes_prob >= 0.7,
                    "high_confidence_no": btts_no_prob >= 0.7,
                    "uncertain": 0.4 <= btts_yes_prob <= 0.6,
                },
                "supporting_factors": self._analyze_btts_factors(
                    btts_yes_prob
                ),
            }

            btts_analysis.append(analysis)

        return {
            "predictions": btts_analysis,
            "model_name": self.model_name,
            "algorithm": self.model_algorithm,
            "average_btts_probability": float(
                np.mean([p["btts_yes_probability"] for p in btts_analysis])
            ),
            "high_btts_matches": len(
                [p for p in btts_analysis if p["btts_yes_probability"] > 0.7]
            ),
            "prediction_timestamp": datetime.now().isoformat(),
        }

    def _generate_btts_betting_advice(
        self, btts_yes_prob: float
    ) -> Dict[str, str]:
        """Generar consejos de apuesta para BTTS."""
        if btts_yes_prob >= 0.7:
            return {
                "recommendation": "Bet BTTS Yes",
                "confidence": "high",
                "reasoning": f"Strong probability ({btts_yes_prob:.1%}) both teams score",
            }
        elif btts_yes_prob <= 0.3:
            return {
                "recommendation": "Bet BTTS No",
                "confidence": "high",
                "reasoning": f"Low probability ({btts_yes_prob:.1%}) both teams score",
            }
        elif btts_yes_prob >= 0.6:
            return {
                "recommendation": "Consider BTTS Yes",
                "confidence": "medium",
                "reasoning": f"Good probability ({btts_yes_prob:.1%}) both teams score",
            }
        elif btts_yes_prob <= 0.4:
            return {
                "recommendation": "Consider BTTS No",
                "confidence": "medium",
                "reasoning": f"Lower probability ({btts_yes_prob:.1%}) both teams score",
            }
        else:
            return {
                "recommendation": "Avoid BTTS market",
                "confidence": "low",
                "reasoning": f"Uncertain outcome ({btts_yes_prob:.1%})",
            }

    def _analyze_btts_factors(self, btts_yes_prob: float) -> Dict[str, str]:
        """Analizar factores que apoyan la predicción BTTS."""
        factors = {}

        if btts_yes_prob >= 0.6:
            factors["primary"] = "Both teams have good attacking records"
            factors["secondary"] = "Defensive vulnerabilities on both sides"
            factors["tertiary"] = (
                "Historical tendency for goals in this matchup"
            )
        elif btts_yes_prob <= 0.4:
            factors["primary"] = (
                "At least one team has strong defensive record"
            )
            factors["secondary"] = "Low scoring tendency in recent matches"
            factors["tertiary"] = (
                "One or both teams struggle to score consistently"
            )
        else:
            factors["primary"] = "Mixed attacking and defensive capabilities"
            factors["secondary"] = "Uncertain tactical approach expected"
            factors["tertiary"] = "Historical data shows varied outcomes"

        return factors


# =====================================================
# ENSEMBLE DE PREDICTORES DE GOLES
# =====================================================


class GoalsEnsemblePredictor:
    """
    Ensemble que combina múltiples predictores de goles.
    """

    def __init__(self):
        self.total_goals_predictor = TotalGoalsPredictor()
        self.home_goals_predictor = TeamGoalsPredictor(team_type="home")
        self.away_goals_predictor = TeamGoalsPredictor(team_type="away")
        self.btts_predictor = BothTeamsScorePredictor()

        self.predictors = {
            "total_goals": self.total_goals_predictor,
            "home_goals": self.home_goals_predictor,
            "away_goals": self.away_goals_predictor,
            "btts": self.btts_predictor,
        }

        logger.info("Initialized Goals Ensemble Predictor")

    def train_all_predictors(
        self, X: pd.DataFrame, y_dict: Dict[str, pd.Series]
    ) -> Dict[str, Any]:
        """
        Entrenar todos los predictores del ensemble.

        Args:
            X: Features de entrenamiento
            y_dict: Targets para cada predictor

        Returns:
            dict: Resultados de entrenamiento
        """
        training_results = {}

        for name, predictor in self.predictors.items():
            if name in y_dict:
                logger.info(f"Training {name} predictor...")
                try:
                    result = predictor.train(X, y_dict[name])
                    training_results[name] = result
                    logger.info(f"{name} training completed successfully")
                except Exception as e:
                    logger.error(f"Error training {name}: {e}")
                    training_results[name] = {
                        "success": False,
                        "error": str(e),
                    }

        return training_results

    def predict_all_goals(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Hacer predicciones con todos los predictores de goles.

        Args:
            X: Features para predicción

        Returns:
            dict: Predicciones completas de goles
        """
        all_predictions = {}

        # Predicciones individuales
        if self.total_goals_predictor.is_trained:
            all_predictions["total_goals"] = (
                self.total_goals_predictor.predict_goals_detailed(X)
            )

        if self.home_goals_predictor.is_trained:
            all_predictions["home_goals"] = (
                self.home_goals_predictor.predict_team_goals_detailed(X)
            )

        if self.away_goals_predictor.is_trained:
            all_predictions["away_goals"] = (
                self.away_goals_predictor.predict_team_goals_detailed(X)
            )

        if self.btts_predictor.is_trained:
            all_predictions["btts"] = (
                self.btts_predictor.predict_btts_detailed(X)
            )

        # Crear resumen consolidado
        consolidated_summary = self._create_consolidated_goals_summary(
            all_predictions
        )

        return {
            "individual_predictions": all_predictions,
            "consolidated_summary": consolidated_summary,
            "ensemble_info": {
                "active_predictors": len(all_predictions),
                "prediction_timestamp": datetime.now().isoformat(),
            },
        }

    def _create_consolidated_goals_summary(
        self, predictions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Crear resumen consolidado de todas las predicciones de goles."""
        summary = {
            "match_outlook": "moderate_scoring",
            "key_insights": [],
            "best_betting_opportunities": [],
            "over_under_recommendations": {},
            "btts_analysis": {},
        }

        # Analizar predicciones de goles totales
        if "total_goals" in predictions:
            total_pred = predictions["total_goals"]
            avg_goals = total_pred.get("average_goals_predicted", 0)

            if avg_goals >= 3.5:
                summary["match_outlook"] = "high_scoring"
                summary["key_insights"].append(
                    f"High-scoring match expected: {avg_goals:.1f} goals average"
                )
                summary["over_under_recommendations"][
                    "primary"
                ] = "Over 2.5 goals"
                summary["over_under_recommendations"][
                    "alternative"
                ] = "Over 3.5 goals"
            elif avg_goals <= 2.0:
                summary["match_outlook"] = "low_scoring"
                summary["key_insights"].append(
                    f"Low-scoring match expected: {avg_goals:.1f} goals average"
                )
                summary["over_under_recommendations"][
                    "primary"
                ] = "Under 2.5 goals"
                summary["over_under_recommendations"][
                    "alternative"
                ] = "Under 1.5 goals"
            else:
                summary["over_under_recommendations"][
                    "primary"
                ] = "Over 2.5 goals"
                summary["over_under_recommendations"][
                    "alternative"
                ] = "Consider both Over/Under 2.5"

        # Analizar BTTS
        if "btts" in predictions:
            btts_pred = predictions["btts"]
            avg_btts_prob = btts_pred.get("average_btts_probability", 0)

            if avg_btts_prob >= 0.65:
                summary["btts_analysis"]["recommendation"] = "BTTS Yes"
                summary["btts_analysis"]["confidence"] = "high"
                summary["key_insights"].append("Both teams likely to score")
                summary["best_betting_opportunities"].append(
                    {
                        "market": "Both Teams to Score - Yes",
                        "confidence": "high",
                        "reasoning": f"Strong BTTS probability: {avg_btts_prob:.1%}",
                    }
                )
            elif avg_btts_prob <= 0.35:
                summary["btts_analysis"]["recommendation"] = "BTTS No"
                summary["btts_analysis"]["confidence"] = "high"
                summary["key_insights"].append(
                    "Clean sheet likely for at least one team"
                )
                summary["best_betting_opportunities"].append(
                    {
                        "market": "Both Teams to Score - No",
                        "confidence": "high",
                        "reasoning": f"Low BTTS probability: {avg_btts_prob:.1%}",
                    }
                )
            else:
                summary["btts_analysis"][
                    "recommendation"
                ] = "Avoid BTTS market"
                summary["btts_analysis"]["confidence"] = "low"

        # Analizar distribución de goles por equipo
        home_goals_info = (
            predictions.get("home_goals", {}).get("predictions", [{}])[0]
            if "home_goals" in predictions
            else {}
        )
        away_goals_info = (
            predictions.get("away_goals", {}).get("predictions", [{}])[0]
            if "away_goals" in predictions
            else {}
        )

        if home_goals_info and away_goals_info:
            home_goals = home_goals_info.get("home_team", {}).get(
                "predicted_goals", 0
            )
            away_goals = away_goals_info.get("away_team", {}).get(
                "predicted_goals", 0
            )

            if home_goals > away_goals * 1.5:
                summary["key_insights"].append(
                    "Home team expected to dominate scoring"
                )
            elif away_goals > home_goals * 1.5:
                summary["key_insights"].append(
                    "Away team expected to dominate scoring"
                )
            else:
                summary["key_insights"].append(
                    "Balanced scoring expected from both teams"
                )

        # Identificar mejores oportunidades de apuesta
        if "total_goals" in predictions:
            total_preds = predictions["total_goals"]["predictions"]
            for pred in total_preds[:3]:  # Primeros 3 partidos
                best_bet = pred.get("betting_value_analysis", {}).get(
                    "best_over_under_bet", {}
                )
                if best_bet.get("confidence", 0) >= 0.7:
                    summary["best_betting_opportunities"].append(
                        {
                            "market": best_bet.get("market", "Unknown"),
                            "confidence": "high",
                            "reasoning": "Strong model confidence in Over/Under prediction",
                        }
                    )

        return summary

    def create_betting_strategy(
        self, predictions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Crear estrategia de apuestas basada en todas las predicciones.

        Args:
            predictions: Predicciones del ensemble

        Returns:
            dict: Estrategia de apuestas recomendada
        """
        strategy = {
            "recommended_bets": [],
            "avoid_markets": [],
            "risk_level": "medium",
            "expected_roi": 0,
            "bankroll_allocation": {},
        }

        summary = predictions.get("consolidated_summary", {})

        # Recomendaciones Over/Under
        ou_recommendations = summary.get("over_under_recommendations", {})
        if ou_recommendations.get("primary"):
            strategy["recommended_bets"].append(
                {
                    "market": ou_recommendations["primary"],
                    "allocation": "40%",
                    "confidence": "high",
                    "stake_size": "medium",
                }
            )

        # Recomendaciones BTTS
        btts_analysis = summary.get("btts_analysis", {})
        if btts_analysis.get("confidence") == "high":
            strategy["recommended_bets"].append(
                {
                    "market": f"BTTS {btts_analysis.get('recommendation', 'Unknown')}",
                    "allocation": "30%",
                    "confidence": "high",
                    "stake_size": "medium",
                }
            )

        # Mejores oportunidades
        best_opportunities = summary.get("best_betting_opportunities", [])
        for opportunity in best_opportunities[:2]:  # Top 2 oportunidades
            if opportunity.get("confidence") == "high":
                strategy["recommended_bets"].append(
                    {
                        "market": opportunity.get("market"),
                        "allocation": "15%",
                        "confidence": opportunity.get("confidence"),
                        "stake_size": "small",
                    }
                )

        # Mercados a evitar
        if btts_analysis.get("confidence") == "low":
            strategy["avoid_markets"].append("Both Teams to Score markets")

        # Calcular nivel de riesgo
        high_conf_bets = len(
            [
                bet
                for bet in strategy["recommended_bets"]
                if bet.get("confidence") == "high"
            ]
        )
        if high_conf_bets >= 2:
            strategy["risk_level"] = "low"
        elif high_conf_bets == 1:
            strategy["risk_level"] = "medium"
        else:
            strategy["risk_level"] = "high"

        return strategy

    def get_ensemble_info(self) -> Dict[str, Any]:
        """Obtener información del ensemble de goles."""
        return {
            "predictors": {
                name: {
                    "is_trained": predictor.is_trained,
                    "model_type": predictor.model_type,
                    "prediction_type": predictor.prediction_type,
                }
                for name, predictor in self.predictors.items()
            },
            "ensemble_capabilities": [
                "Total Goals Prediction",
                "Team-specific Goals",
                "Both Teams to Score",
                "Over/Under Markets",
                "Betting Strategy Generation",
                "Risk Assessment",
            ],
            "supported_markets": [
                "Over/Under 0.5, 1.5, 2.5, 3.5, 4.5 goals",
                "Both Teams to Score Yes/No",
                "Team Goals Over/Under",
                "Exact Goals markets",
                "First Half Goals",
            ],
        }


# =====================================================
# UTILIDADES PARA ANÁLISIS DE GOLES
# =====================================================


class GoalsAnalyzer:
    """Utilidades para análisis avanzado de goles."""

    @staticmethod
    def calculate_expected_goals(
        shots: int, shots_on_target: int, big_chances: int
    ) -> float:
        """
        Calcular Expected Goals (xG) simplificado.

        Args:
            shots: Total de disparos
            shots_on_target: Disparos a puerta
            big_chances: Grandes ocasiones

        Returns:
            float: xG estimado
        """
        if shots == 0:
            return 0.0

        # Modelo simplificado de xG
        shot_quality = shots_on_target / shots if shots > 0 else 0
        big_chance_conversion = (
            big_chances * 0.4
        )  # ~40% conversión de grandes ocasiones
        regular_shots_xg = (
            shots_on_target - big_chances
        ) * 0.1  # ~10% conversión disparos normales

        total_xg = big_chance_conversion + regular_shots_xg
        return round(max(0, total_xg), 2)

    @staticmethod
    def analyze_scoring_patterns(team_history: pd.DataFrame) -> Dict[str, Any]:
        """
        Analizar patrones de goles de un equipo.

        Args:
            team_history: Historial del equipo

        Returns:
            dict: Análisis de patrones de goles
        """
        if team_history.empty:
            return {"pattern": "insufficient_data"}

        # Tendencias por período
        first_half_goals = team_history["first_half_goals"].mean()
        second_half_goals = team_history["second_half_goals"].mean()

        # Consistencia de goles
        goals_std = team_history["goals_scored"].std()
        consistency = (
            "high"
            if goals_std < 1.0
            else "medium" if goals_std < 1.5 else "low"
        )

        # Patrones por local/visitante
        if "is_home" in team_history.columns:
            home_avg = team_history[team_history["is_home"]][
                "goals_scored"
            ].mean()
            away_avg = team_history[~team_history["is_home"]][
                "goals_scored"
            ].mean()
            home_advantage_goals = home_avg - away_avg
        else:
            home_advantage_goals = 0

        return {
            "avg_goals_per_game": round(
                team_history["goals_scored"].mean(), 2
            ),
            "first_half_avg": round(first_half_goals, 2),
            "second_half_avg": round(second_half_goals, 2),
            "preferred_half": (
                "first" if first_half_goals > second_half_goals else "second"
            ),
            "consistency": consistency,
            "home_advantage_goals": round(home_advantage_goals, 2),
            "high_scoring_games_ratio": (
                team_history["goals_scored"] >= 3
            ).mean(),
            "failed_to_score_ratio": (
                team_history["goals_scored"] == 0
            ).mean(),
        }

    @staticmethod
    def calculate_poisson_probabilities(
        home_xg: float, away_xg: float
    ) -> Dict[str, Dict[str, float]]:
        """
        Calcular probabilidades de resultados exactos usando distribución de Poisson.

        Args:
            home_xg: Expected goals equipo local
            away_xg: Expected goals equipo visitante

        Returns:
            dict: Matriz de probabilidades de resultados
        """
        probabilities = {}

        # Calcular probabilidades para scores 0-5 para cada equipo
        for home_goals in range(6):
            probabilities[home_goals] = {}
            for away_goals in range(6):
                home_prob = poisson.pmf(home_goals, home_xg)
                away_prob = poisson.pmf(away_goals, away_xg)
                probabilities[home_goals][away_goals] = round(
                    home_prob * away_prob, 4
                )

        # Calcular probabilidades de mercados principales
        market_probabilities = {
            "home_win": sum(
                probabilities[h][a]
                for h in range(6)
                for a in range(6)
                if h > a
            ),
            "draw": sum(probabilities[h][h] for h in range(6)),
            "away_win": sum(
                probabilities[h][a]
                for h in range(6)
                for a in range(6)
                if h < a
            ),
            "over_2_5": sum(
                probabilities[h][a]
                for h in range(6)
                for a in range(6)
                if h + a > 2.5
            ),
            "under_2_5": sum(
                probabilities[h][a]
                for h in range(6)
                for a in range(6)
                if h + a < 2.5
            ),
            "btts_yes": sum(
                probabilities[h][a] for h in range(1, 6) for a in range(1, 6)
            ),
            "btts_no": sum(
                probabilities[h][a]
                for h in range(6)
                for a in range(6)
                if h == 0 or a == 0
            ),
        }

        return {
            "exact_scores": probabilities,
            "market_probabilities": {
                k: round(v, 3) for k, v in market_probabilities.items()
            },
        }


# =====================================================
# FUNCIÓN PRINCIPAL PARA CREAR PREDICTORES
# =====================================================


def create_goals_predictor(
    predictor_type: str = "ensemble", **kwargs
) -> object:
    """
    Factory function para crear predictores de goles.

    Args:
        predictor_type: Tipo de predictor ('total', 'team', 'btts', 'ensemble')
        **kwargs: Argumentos adicionales

    Returns:
        object: Predictor configurado
    """
    if predictor_type == "total":
        algorithm = kwargs.get("algorithm", "xgboost")
        return TotalGoalsPredictor(model_algorithm=algorithm)

    elif predictor_type == "team":
        team_type = kwargs.get("team_type", "both")
        algorithm = kwargs.get("algorithm", "xgboost")
        return TeamGoalsPredictor(
            team_type=team_type, model_algorithm=algorithm
        )

    elif predictor_type == "btts":
        algorithm = kwargs.get("algorithm", "xgboost")
        return BothTeamsScorePredictor(model_algorithm=algorithm)

    elif predictor_type == "ensemble":
        return GoalsEnsemblePredictor()

    else:
        raise ValueError(f"Unknown predictor type: {predictor_type}")


# =====================================================
# EXPORTACIONES
# =====================================================

__all__ = [
    # Predictores principales
    "TotalGoalsPredictor",
    "TeamGoalsPredictor",
    "BothTeamsScorePredictor",
    "GoalsEnsemblePredictor",
    # Utilidades
    "GoalsAnalyzer",
    "create_goals_predictor",
]
