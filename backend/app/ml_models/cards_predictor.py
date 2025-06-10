"""
Cards Predictor for Football Analytics

Este módulo contiene modelos especializados en predecir tarjetas amarillas y rojas
en partidos de fútbol, incluyendo total de tarjetas, tarjetas por equipo y jugador.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor

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
# PREDICTOR DE TOTAL DE TARJETAS
# =====================================================


class TotalCardsPredictor(BaseFootballPredictor, FootballRegressorMixin):
    """
    Predictor para el total de tarjetas en un partido.

    Predice el número total de tarjetas amarillas y rojas
    que se mostrarán durante un partido.
    """

    def __init__(
        self, card_type: str = "yellow", model_algorithm: str = "xgboost"
    ):
        """
        Inicializar predictor de tarjetas totales.

        Args:
            card_type: Tipo de tarjeta ('yellow', 'red', 'total')
            model_algorithm: Algoritmo ML ('xgboost', 'randomforest', 'poisson')
        """
        model_name = f"total_{card_type}_cards_predictor"
        super().__init__(
            model_name=model_name,
            prediction_type="cards",
            model_type="regressor",
        )

        self.card_type = card_type
        self.model_algorithm = model_algorithm

        # Features específicas para predicción de tarjetas
        self.add_football_features(
            team_features=[
                "home_team_avg_cards",
                "away_team_avg_cards",
                "home_team_cards_last_5",
                "away_team_cards_last_5",
                "home_team_discipline_rating",
                "away_team_discipline_rating",
                "home_team_aggressive_style",
                "away_team_aggressive_style",
            ],
            player_features=[
                "home_team_suspended_players",
                "away_team_suspended_players",
                "home_team_aggressive_players",
                "away_team_aggressive_players",
                "key_players_card_history",
            ],
            match_features=[
                "referee_cards_per_game",
                "referee_strictness",
                "match_importance",
                "rivalry_factor",
                "league_discipline_level",
                "season_period",
                "weather_conditions",
                "stadium_atmosphere",
            ],
        )

        logger.info(f"Initialized {model_name} with {model_algorithm}")

    def _create_model(self):
        """Crear modelo según algoritmo especificado."""
        if self.model_algorithm == "xgboost":
            return XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                objective="count:poisson",  # Apropiado para conteos
            )

        elif self.model_algorithm == "randomforest":
            return RandomForestRegressor(
                n_estimators=200,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
            )

        elif self.model_algorithm == "poisson":
            return PoissonRegressor(alpha=1.0, max_iter=1000)

        else:
            raise ValueError(f"Unknown algorithm: {self.model_algorithm}")

    def _get_default_hyperparameters(self):
        """Hiperparámetros por defecto según algoritmo."""
        if self.model_algorithm == "xgboost":
            return {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            }
        elif self.model_algorithm == "randomforest":
            return {
                "n_estimators": 200,
                "max_depth": 8,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
            }
        else:
            return {"alpha": 1.0}

    def _get_hyperparameter_grid(self):
        """Grid para optimización de hiperparámetros."""
        if self.model_algorithm == "xgboost":
            return {
                "n_estimators": [100, 200, 300],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.05, 0.1, 0.15],
                "subsample": [0.7, 0.8, 0.9],
                "colsample_bytree": [0.7, 0.8, 0.9],
            }
        elif self.model_algorithm == "randomforest":
            return {
                "n_estimators": [100, 200, 300],
                "max_depth": [6, 8, 10, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            }
        else:
            return {"alpha": [0.1, 0.5, 1.0, 2.0, 5.0]}

    def create_cards_features(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Crear features específicas para predicción de tarjetas.

        Args:
            matches_df: DataFrame con datos de partidos

        Returns:
            pd.DataFrame: DataFrame con features de tarjetas
        """
        df = self.create_football_features(matches_df.copy())

        # Features de historial de tarjetas por equipo
        if "home_team_id" in df.columns:
            # Promedio de tarjetas por equipo (últimos partidos)
            df["home_team_avg_cards_season"] = df.groupby("home_team_id")[
                "home_team_cards"
            ].transform("mean")
            df["away_team_avg_cards_season"] = df.groupby("away_team_id")[
                "away_team_cards"
            ].transform("mean")

            # Rolling average últimos 5 partidos
            df["home_team_cards_last_5"] = df.groupby("home_team_id")[
                "home_team_cards"
            ].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
            df["away_team_cards_last_5"] = df.groupby("away_team_id")[
                "away_team_cards"
            ].transform(lambda x: x.rolling(window=5, min_periods=1).mean())

        # Features del árbitro
        if "referee" in df.columns:
            df["referee_avg_cards"] = df.groupby("referee")[
                "total_cards"
            ].transform("mean")
            df["referee_strictness"] = (
                df.groupby("referee")["total_cards"].transform("std").fillna(0)
            )

        # Features temporales específicas para tarjetas
        if "match_date" in df.columns:
            df["match_date"] = pd.to_datetime(df["match_date"])

            # Período de la temporada (más tarjetas al final)
            df["days_from_season_start"] = (
                df["match_date"] - df["match_date"].min()
            ).dt.days
            df["season_progress"] = (
                df["days_from_season_start"]
                / df["days_from_season_start"].max()
            )

            # Mes del año (más agresividad en ciertos meses)
            df["month"] = df["match_date"].dt.month
            df["is_winter"] = (
                (df["month"] >= 12) | (df["month"] <= 2)
            ).astype(int)

        # Features de rivalidad
        if "home_team_id" in df.columns and "away_team_id" in df.columns:
            # Crear ID único para el enfrentamiento
            df["matchup_id"] = df[["home_team_id", "away_team_id"]].apply(
                lambda x: f"{min(x)}-{max(x)}", axis=1
            )

            # Promedio de tarjetas en enfrentamientos previos
            df["historical_cards_h2h"] = df.groupby("matchup_id")[
                "total_cards"
            ].transform("mean")

            # Derby factor (equipos de la misma liga/ciudad)
            df["is_local_derby"] = (
                df["home_team_id"] == df["away_team_id"]
            ).astype(int)

        # Features de importancia del partido
        if "importance" in df.columns:
            importance_cards_factor = {
                "low": 0.8,
                "normal": 1.0,
                "high": 1.3,
                "final": 1.5,
                "derby": 1.4,
            }
            df["importance_cards_multiplier"] = (
                df["importance"].map(importance_cards_factor).fillna(1.0)
            )

        # Features de estilo de juego
        if "home_team_possession_avg" in df.columns:
            # Equipos con menos posesión tienden a hacer más faltas
            df["possession_diff"] = (
                df["home_team_possession_avg"] - df["away_team_possession_avg"]
            )
            df["defensive_style_factor"] = np.where(
                df["possession_diff"] < -10,
                1.2,  # Equipo visitante más defensivo
                np.where(
                    df["possession_diff"] > 10, 1.1, 1.0
                ),  # Equipo local más defensivo
            )

        # Features de clima (más agresividad en mal tiempo)
        if "weather" in df.columns:
            weather_aggression = {
                "sunny": 1.0,
                "cloudy": 1.05,
                "rainy": 1.15,
                "stormy": 1.25,
                "snow": 1.3,
            }
            df["weather_aggression_factor"] = (
                df["weather"].map(weather_aggression).fillna(1.0)
            )

        logger.info(f"Created cards features, final shape: {df.shape}")
        return df

    def predict_cards_detailed(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Predecir tarjetas con análisis detallado.

        Args:
            X: Features para predicción

        Returns:
            dict: Predicciones detalladas de tarjetas
        """
        # Predicciones básicas
        predictions = self.predict_goals(X)  # Usar método del mixin

        # Analizar distribución de tarjetas
        card_predictions = []
        for pred in predictions["predictions"]:
            exact_cards = pred[
                "predicted_goals_exact"
            ]  # Reutilizar estructura
            rounded_cards = pred["predicted_goals_rounded"]

            card_analysis = {
                "predicted_cards_exact": exact_cards,
                "predicted_cards_rounded": max(
                    0, rounded_cards
                ),  # No puede ser negativo
                "over_under_analysis": {
                    "over_1_5": "over" if exact_cards > 1.5 else "under",
                    "over_2_5": "over" if exact_cards > 2.5 else "under",
                    "over_3_5": "over" if exact_cards > 3.5 else "under",
                    "over_4_5": "over" if exact_cards > 4.5 else "under",
                },
                "card_probability_ranges": {
                    "0-1_cards": self._calculate_range_probability(
                        exact_cards, 0, 1
                    ),
                    "2-3_cards": self._calculate_range_probability(
                        exact_cards, 2, 3
                    ),
                    "4-5_cards": self._calculate_range_probability(
                        exact_cards, 4, 5
                    ),
                    "6+_cards": 1.0 if exact_cards >= 6 else 0.0,
                },
                "betting_recommendations": self._generate_cards_betting_advice(
                    exact_cards
                ),
            }

            card_predictions.append(card_analysis)

        return {
            "card_type": self.card_type,
            "predictions": card_predictions,
            "model_name": self.model_name,
            "algorithm": self.model_algorithm,
            "average_cards_predicted": float(
                np.mean([p["predicted_cards_exact"] for p in card_predictions])
            ),
            "prediction_timestamp": datetime.now().isoformat(),
        }

    def _calculate_range_probability(
        self, prediction: float, min_val: int, max_val: int
    ) -> float:
        """Calcular probabilidad de que las tarjetas estén en un rango."""
        # Usar distribución de Poisson para calcular probabilidad
        from scipy.stats import poisson

        lambda_param = max(0.1, prediction)  # Evitar lambda = 0

        if max_val == float("inf"):
            prob = 1 - poisson.cdf(min_val - 1, lambda_param)
        else:
            prob = poisson.cdf(max_val, lambda_param) - poisson.cdf(
                min_val - 1, lambda_param
            )

        return round(float(prob), 3)

    def _generate_cards_betting_advice(
        self, predicted_cards: float
    ) -> Dict[str, Any]:
        """Generar consejos de apuestas para tarjetas."""
        advice = {
            "primary_bet": None,
            "confidence": "low",
            "alternative_bets": [],
            "avoid_bets": [],
        }

        if predicted_cards <= 1.5:
            advice["primary_bet"] = {
                "market": "Under 2.5 Cards",
                "reasoning": f"Low cards prediction ({predicted_cards:.1f})",
            }
            advice["confidence"] = (
                "medium" if predicted_cards <= 1.2 else "low"
            )
            advice["avoid_bets"] = ["Over 3.5 Cards", "Over 4.5 Cards"]

        elif predicted_cards >= 4.5:
            advice["primary_bet"] = {
                "market": "Over 3.5 Cards",
                "reasoning": f"High cards prediction ({predicted_cards:.1f})",
            }
            advice["confidence"] = (
                "medium" if predicted_cards >= 5.0 else "low"
            )
            advice["alternative_bets"] = ["Over 4.5 Cards"]
            advice["avoid_bets"] = ["Under 1.5 Cards"]

        else:
            # Predicción moderada
            advice["primary_bet"] = {
                "market": "Over 2.5 Cards",
                "reasoning": f"Moderate cards prediction ({predicted_cards:.1f})",
            }
            advice["confidence"] = "low"
            advice["alternative_bets"] = ["Under 4.5 Cards"]

        return advice


# =====================================================
# PREDICTOR DE TARJETAS POR EQUIPO
# =====================================================


class TeamCardsPredictor(BaseFootballPredictor, FootballRegressorMixin):
    """
    Predictor para tarjetas específicas por equipo.

    Predice cuántas tarjetas recibirá cada equipo individual.
    """

    def __init__(self, model_algorithm: str = "xgboost"):
        super().__init__(
            model_name="team_cards_predictor",
            prediction_type="team_cards",
            model_type="regressor",
        )

        self.model_algorithm = model_algorithm

        # Features específicas para tarjetas por equipo
        self.add_football_features(
            team_features=[
                "team_discipline_record",
                "team_playing_style",
                "team_cards_home_vs_away",
                "team_pressure_response",
                "team_youth_players_ratio",
                "team_captain_leadership",
            ],
            player_features=[
                "team_aggressive_players_count",
                "team_suspended_players",
                "team_key_players_cards_history",
                "team_new_signings_discipline",
            ],
            match_features=[
                "opponent_pressing_intensity",
                "match_pressure_level",
                "referee_bias_history",
                "venue_atmosphere_effect",
            ],
        )

    def _create_model(self):
        """Crear modelo para predicción por equipo."""
        if self.model_algorithm == "xgboost":
            return XGBRegressor(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=self.random_state,
            )
        else:
            return RandomForestRegressor(
                n_estimators=150, max_depth=7, random_state=self.random_state
            )

    def _get_default_hyperparameters(self):
        return {"n_estimators": 150, "max_depth": 5, "learning_rate": 0.1}

    def _get_hyperparameter_grid(self):
        return {
            "n_estimators": [100, 150, 200],
            "max_depth": [4, 5, 6],
            "learning_rate": [0.08, 0.1, 0.12],
        }

    def predict_team_cards(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Predecir tarjetas por equipo con análisis detallado.

        Args:
            X: Features para predicción

        Returns:
            dict: Predicciones por equipo
        """
        predictions = self.predict_with_confidence(X)

        # Separar predicciones para cada equipo
        team_predictions = []
        for i, pred in enumerate(predictions["predictions"]):
            # Asumiendo que las predicciones vienen en pares (home, away)
            home_cards = max(0, pred)
            away_cards = max(0, pred * 0.9)  # Ajuste para equipo visitante

            team_analysis = {
                "home_team_cards": round(home_cards, 1),
                "away_team_cards": round(away_cards, 1),
                "total_match_cards": round(home_cards + away_cards, 1),
                "cards_distribution": {
                    "home_percentage": (
                        round(home_cards / (home_cards + away_cards) * 100, 1)
                        if (home_cards + away_cards) > 0
                        else 50
                    ),
                    "away_percentage": (
                        round(away_cards / (home_cards + away_cards) * 100, 1)
                        if (home_cards + away_cards) > 0
                        else 50
                    ),
                },
                "team_discipline_comparison": {
                    "more_disciplined_team": (
                        "home" if home_cards < away_cards else "away"
                    ),
                    "discipline_difference": abs(home_cards - away_cards),
                },
            }

            team_predictions.append(team_analysis)

        return {
            "predictions": team_predictions,
            "model_name": self.model_name,
            "prediction_type": "team_specific_cards",
            "average_home_cards": float(
                np.mean([p["home_team_cards"] for p in team_predictions])
            ),
            "average_away_cards": float(
                np.mean([p["away_team_cards"] for p in team_predictions])
            ),
            "timestamp": datetime.now().isoformat(),
        }


# =====================================================
# PREDICTOR DE TARJETAS ROJAS
# =====================================================


class RedCardsPredictor(BaseFootballPredictor, FootballClassifierMixin):
    """
    Predictor especializado en tarjetas rojas.

    Predice la probabilidad de que haya tarjetas rojas en un partido.
    """

    def __init__(self, model_algorithm: str = "xgboost"):
        super().__init__(
            model_name="red_cards_predictor",
            prediction_type="red_cards",
            model_type="classifier",
        )

        self.model_algorithm = model_algorithm

        # Features específicas para tarjetas rojas
        self.add_football_features(
            team_features=[
                "team_red_cards_history",
                "team_aggressive_incidents",
                "team_disciplinary_points",
                "team_fair_play_ranking",
            ],
            player_features=[
                "players_with_red_card_history",
                "hot_headed_players",
                "suspended_players_returning",
                "young_players_ratio",
            ],
            match_features=[
                "referee_red_cards_ratio",
                "high_stakes_match",
                "historical_red_cards_h2h",
                "match_intensity_rating",
            ],
        )

    def _create_model(self):
        """Crear clasificador para tarjetas rojas."""
        if self.model_algorithm == "xgboost":
            return XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                scale_pos_weight=3,  # Ajustar por desbalance (pocas tarjetas rojas)
                random_state=self.random_state,
            )
        else:
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                class_weight="balanced",
                random_state=self.random_state,
            )

    def _get_default_hyperparameters(self):
        return {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.1,
            "scale_pos_weight": 3,
        }

    def _get_hyperparameter_grid(self):
        return {
            "n_estimators": [150, 200, 250],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.08, 0.1, 0.12],
            "scale_pos_weight": [2, 3, 4],
        }

    def predict_red_cards_detailed(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Predecir tarjetas rojas con análisis de probabilidad.

        Args:
            X: Features para predicción

        Returns:
            dict: Análisis detallado de tarjetas rojas
        """
        predictions = self.predict_match_outcome(X)  # Usar método del mixin

        red_card_analysis = []
        for pred in predictions["predictions"]:
            # Adaptar las probabilidades para tarjetas rojas
            red_card_prob = pred["probabilities"][
                "home_win"
            ]  # Reutilizar estructura
            no_red_card_prob = 1 - red_card_prob

            analysis = {
                "red_card_probability": round(red_card_prob, 3),
                "no_red_card_probability": round(no_red_card_prob, 3),
                "prediction": (
                    "red_card_likely"
                    if red_card_prob > 0.3
                    else "no_red_card_expected"
                ),
                "confidence_level": self.get_prediction_confidence_level(
                    max(red_card_prob, no_red_card_prob)
                ),
                "risk_assessment": {
                    "very_low": red_card_prob < 0.1,
                    "low": 0.1 <= red_card_prob < 0.2,
                    "moderate": 0.2 <= red_card_prob < 0.35,
                    "high": red_card_prob >= 0.35,
                },
                "betting_advice": self._generate_red_card_betting_advice(
                    red_card_prob
                ),
            }

            red_card_analysis.append(analysis)

        return {
            "predictions": red_card_analysis,
            "model_name": self.model_name,
            "average_red_card_probability": float(
                np.mean([p["red_card_probability"] for p in red_card_analysis])
            ),
            "high_risk_matches": len(
                [
                    p
                    for p in red_card_analysis
                    if p["red_card_probability"] > 0.3
                ]
            ),
            "timestamp": datetime.now().isoformat(),
        }

    def _generate_red_card_betting_advice(
        self, red_card_prob: float
    ) -> Dict[str, str]:
        """Generar consejos de apuesta para tarjetas rojas."""
        if red_card_prob >= 0.35:
            return {
                "recommendation": "Bet on Red Card Yes",
                "confidence": "medium",
                "reasoning": f"High probability ({red_card_prob:.1%}) of red card",
            }
        elif red_card_prob <= 0.15:
            return {
                "recommendation": "Bet on Red Card No",
                "confidence": "medium",
                "reasoning": f"Low probability ({red_card_prob:.1%}) of red card",
            }
        else:
            return {
                "recommendation": "Avoid red card bets",
                "confidence": "low",
                "reasoning": f"Uncertain probability ({red_card_prob:.1%})",
            }


# =====================================================
# ENSEMBLE DE PREDICTORES DE TARJETAS
# =====================================================


class CardsEnsemblePredictor:
    """
    Ensemble que combina múltiples predictores de tarjetas.
    """

    def __init__(self):
        self.total_cards_predictor = TotalCardsPredictor(card_type="total")
        self.yellow_cards_predictor = TotalCardsPredictor(card_type="yellow")
        self.red_cards_predictor = RedCardsPredictor()
        self.team_cards_predictor = TeamCardsPredictor()

        self.predictors = {
            "total_cards": self.total_cards_predictor,
            "yellow_cards": self.yellow_cards_predictor,
            "red_cards": self.red_cards_predictor,
            "team_cards": self.team_cards_predictor,
        }

        logger.info("Initialized Cards Ensemble Predictor")

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

    def predict_all_cards(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Hacer predicciones con todos los predictores.

        Args:
            X: Features para predicción

        Returns:
            dict: Predicciones completas de tarjetas
        """
        all_predictions = {}

        # Predicciones individuales
        if self.total_cards_predictor.is_trained:
            all_predictions["total_cards"] = (
                self.total_cards_predictor.predict_cards_detailed(X)
            )

        if self.yellow_cards_predictor.is_trained:
            all_predictions["yellow_cards"] = (
                self.yellow_cards_predictor.predict_cards_detailed(X)
            )

        if self.red_cards_predictor.is_trained:
            all_predictions["red_cards"] = (
                self.red_cards_predictor.predict_red_cards_detailed(X)
            )

        if self.team_cards_predictor.is_trained:
            all_predictions["team_cards"] = (
                self.team_cards_predictor.predict_team_cards(X)
            )

        # Crear resumen consolidado
        consolidated_summary = self._create_consolidated_summary(
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

    def _create_consolidated_summary(
        self, predictions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Crear resumen consolidado de todas las predicciones."""
        summary = {
            "overall_cards_forecast": "moderate",
            "key_insights": [],
            "betting_recommendations": [],
            "risk_factors": [],
        }

        # Analizar predicciones de tarjetas totales
        if "total_cards" in predictions:
            total_pred = predictions["total_cards"]
            avg_cards = total_pred.get("average_cards_predicted", 0)

            if avg_cards >= 4.5:
                summary["overall_cards_forecast"] = "high"
                summary["key_insights"].append(
                    f"High cards expected: {avg_cards:.1f} average"
                )
            elif avg_cards <= 2.0:
                summary["overall_cards_forecast"] = "low"
                summary["key_insights"].append(
                    f"Low cards expected: {avg_cards:.1f} average"
                )

        # Analizar riesgo de tarjetas rojas
        if "red_cards" in predictions:
            red_pred = predictions["red_cards"]
            avg_red_prob = red_pred.get("average_red_card_probability", 0)

            if avg_red_prob >= 0.3:
                summary["risk_factors"].append("High red card risk")
                summary["betting_recommendations"].append(
                    {
                        "market": "Red Card - Yes",
                        "confidence": "medium",
                        "reasoning": f"Red card probability: {avg_red_prob:.1%}",
                    }
                )
            elif avg_red_prob <= 0.15:
                summary["betting_recommendations"].append(
                    {
                        "market": "Red Card - No",
                        "confidence": "medium",
                        "reasoning": f"Low red card probability: {avg_red_prob:.1%}",
                    }
                )

        # Analizar distribución por equipos
        if "team_cards" in predictions:
            team_pred = predictions["team_cards"]
            if team_pred.get("predictions"):
                first_pred = team_pred["predictions"][0]
                home_cards = first_pred.get("home_team_cards", 0)
                away_cards = first_pred.get("away_team_cards", 0)

                if abs(home_cards - away_cards) >= 1.5:
                    more_cards_team = (
                        "home" if home_cards > away_cards else "away"
                    )
                    summary["key_insights"].append(
                        f"Uneven cards distribution: {more_cards_team} team expected to receive more cards"
                    )

        return summary

    def get_ensemble_info(self) -> Dict[str, Any]:
        """Obtener información del ensemble."""
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
                "Total Cards Prediction",
                "Yellow Cards Specific",
                "Red Cards Probability",
                "Team-specific Cards",
                "Over/Under Markets",
                "Betting Recommendations",
            ],
        }


# =====================================================
# UTILIDADES PARA ANÁLISIS DE TARJETAS
# =====================================================


class CardsAnalyzer:
    """Utilidades para análisis avanzado de tarjetas."""

    @staticmethod
    def calculate_referee_impact(
        referee_history: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Calcular el impacto del árbitro en las tarjetas.

        Args:
            referee_history: Historial del árbitro

        Returns:
            dict: Métricas del árbitro
        """
        if referee_history.empty:
            return {
                "avg_cards_per_game": 2.5,  # Promedio general
                "strictness_rating": 0.5,
                "red_card_frequency": 0.1,
            }

        avg_cards = referee_history["total_cards"].mean()
        std_cards = referee_history["total_cards"].std()
        red_card_freq = (referee_history["red_cards"] > 0).mean()

        # Rating de severidad (0-1)
        strictness = min(1.0, (avg_cards - 2.0) / 4.0)  # Normalizar

        return {
            "avg_cards_per_game": round(avg_cards, 2),
            "strictness_rating": round(max(0, strictness), 2),
            "red_card_frequency": round(red_card_freq, 3),
            "consistency": round(
                1 / (1 + std_cards), 2
            ),  # Más consistente = menor desviación
        }

    @staticmethod
    def analyze_team_discipline_trends(
        team_history: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Analizar tendencias de disciplina de un equipo.

        Args:
            team_history: Historial del equipo

        Returns:
            dict: Análisis de disciplina
        """
        if team_history.empty:
            return {"trend": "stable", "risk_level": "medium"}

        # Ordenar por fecha
        team_history = team_history.sort_values("match_date")

        # Calcular tendencia reciente (últimos 10 partidos vs anteriores)
        recent_games = team_history.tail(10)
        previous_games = (
            team_history.head(-10) if len(team_history) > 10 else team_history
        )

        recent_avg = recent_games["total_cards"].mean()
        previous_avg = previous_games["total_cards"].mean()

        # Determinar tendencia
        if recent_avg > previous_avg * 1.2:
            trend = "worsening"
            risk_level = "high"
        elif recent_avg < previous_avg * 0.8:
            trend = "improving"
            risk_level = "low"
        else:
            trend = "stable"
            risk_level = "medium"

        # Factores adicionales
        home_away_diff = (
            team_history[team_history["is_home"]]["total_cards"].mean()
            - team_history[~team_history["is_home"]]["total_cards"].mean()
        )

        return {
            "trend": trend,
            "risk_level": risk_level,
            "recent_avg_cards": round(recent_avg, 2),
            "previous_avg_cards": round(previous_avg, 2),
            "home_away_difference": round(home_away_diff, 2),
            "most_cards_in_game": int(team_history["total_cards"].max()),
            "discipline_consistency": round(
                team_history["total_cards"].std(), 2
            ),
        }

    @staticmethod
    def calculate_rivalry_factor(h2h_history: pd.DataFrame) -> float:
        """
        Calcular factor de rivalidad basado en historial.

        Args:
            h2h_history: Historial cara a cara

        Returns:
            float: Factor de rivalidad (1.0 = normal, >1.0 = más agresivo)
        """
        if h2h_history.empty:
            return 1.0

        avg_cards_h2h = h2h_history["total_cards"].mean()
        general_avg = 3.0  # Promedio general en fútbol

        rivalry_factor = avg_cards_h2h / general_avg

        # Factores adicionales
        red_cards_frequency = (h2h_history["red_cards"] > 0).mean()
        if red_cards_frequency > 0.2:  # Más del 20% de partidos con roja
            rivalry_factor *= 1.2

        return round(min(2.0, rivalry_factor), 2)  # Cap en 2.0


# =====================================================
# FUNCIÓN PRINCIPAL PARA CREAR PREDICTORES
# =====================================================


def create_cards_predictor(
    predictor_type: str = "ensemble", **kwargs
) -> object:
    """
    Factory function para crear predictores de tarjetas.

    Args:
        predictor_type: Tipo de predictor ('total', 'red', 'team', 'ensemble')
        **kwargs: Argumentos adicionales

    Returns:
        object: Predictor configurado
    """
    if predictor_type == "total":
        card_type = kwargs.get("card_type", "total")
        algorithm = kwargs.get("algorithm", "xgboost")
        return TotalCardsPredictor(
            card_type=card_type, model_algorithm=algorithm
        )

    elif predictor_type == "red":
        algorithm = kwargs.get("algorithm", "xgboost")
        return RedCardsPredictor(model_algorithm=algorithm)

    elif predictor_type == "team":
        algorithm = kwargs.get("algorithm", "xgboost")
        return TeamCardsPredictor(model_algorithm=algorithm)

    elif predictor_type == "ensemble":
        return CardsEnsemblePredictor()

    else:
        raise ValueError(f"Unknown predictor type: {predictor_type}")


# =====================================================
# MÉTRICAS ESPECÍFICAS PARA EVALUACIÓN
# =====================================================


class CardsMetrics:
    """Métricas específicas para evaluación de predictores de tarjetas."""

    @staticmethod
    def calculate_cards_accuracy(
        y_true: np.ndarray, y_pred: np.ndarray, tolerance: int = 1
    ) -> float:
        """
        Calcular precisión con tolerancia para predicciones de tarjetas.

        Args:
            y_true: Valores reales
            y_pred: Valores predichos
            tolerance: Tolerancia permitida

        Returns:
            float: Precisión con tolerancia
        """
        correct_predictions = np.abs(y_true - y_pred) <= tolerance
        return np.mean(correct_predictions)

    @staticmethod
    def calculate_over_under_accuracy(
        y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 2.5
    ) -> Dict[str, float]:
        """
        Calcular precisión para mercados over/under.

        Args:
            y_true: Valores reales
            y_pred: Valores predichos
            threshold: Umbral over/under

        Returns:
            dict: Métricas over/under
        """
        true_over = y_true > threshold
        pred_over = y_pred > threshold

        correct_over_under = true_over == pred_over
        accuracy = np.mean(correct_over_under)

        # Precisión específica para over y under
        over_precision = (
            np.mean(true_over[pred_over]) if np.any(pred_over) else 0.0
        )
        under_precision = (
            np.mean(~true_over[~pred_over]) if np.any(~pred_over) else 0.0
        )

        return {
            "overall_accuracy": round(accuracy, 3),
            "over_precision": round(over_precision, 3),
            "under_precision": round(under_precision, 3),
            "over_predictions": int(np.sum(pred_over)),
            "under_predictions": int(np.sum(~pred_over)),
        }

    @staticmethod
    def evaluate_betting_performance(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        odds: np.ndarray,
        threshold: float = 2.5,
    ) -> Dict[str, float]:
        """
        Evaluar rendimiento de apuestas simulado.

        Args:
            y_true: Valores reales
            y_pred: Predicciones
            odds: Cuotas para over/under
            threshold: Umbral

        Returns:
            dict: Métricas de rendimiento de apuestas
        """
        pred_over = y_pred > threshold
        true_over = y_true > threshold

        # Simular apuestas (apostar cuando confianza > 60%)
        confidence = np.abs(y_pred - threshold)
        bet_mask = confidence > 0.6

        if not np.any(bet_mask):
            return {"roi": 0.0, "accuracy": 0.0, "total_bets": 0}

        # Calcular ROI
        bet_outcomes = true_over[bet_mask] == pred_over[bet_mask]
        bet_odds = odds[bet_mask]

        winnings = np.sum(bet_odds[bet_outcomes]) - np.sum(bet_outcomes)
        losses = np.sum(~bet_outcomes)
        roi = (winnings - losses) / len(bet_outcomes) * 100

        return {
            "roi": round(roi, 2),
            "accuracy": round(np.mean(bet_outcomes), 3),
            "total_bets": int(np.sum(bet_mask)),
            "winning_bets": int(np.sum(bet_outcomes)),
            "losing_bets": int(np.sum(~bet_outcomes)),
        }


# =====================================================
# EXPORTACIONES
# =====================================================

__all__ = [
    # Predictores principales
    "TotalCardsPredictor",
    "TeamCardsPredictor",
    "RedCardsPredictor",
    "CardsEnsemblePredictor",
    # Utilidades
    "CardsAnalyzer",
    "CardsMetrics",
    "create_cards_predictor",
]
