"""
Match Outcome Predictor - Predicción de resultados 1X2 (Home Win/Draw/Away Win)

Este módulo contiene los predictores especializados para predecir el resultado
final de partidos de fútbol utilizando clasificación multiclase.

Funcionalidades:
- Predicción 1X2 (Victoria Local/Empate/Victoria Visitante)
- Análisis de probabilidades para cada resultado
- Recomendaciones de apuesta con análisis de valor
- Explicabilidad de predicciones con SHAP
- Ensemble de múltiples algoritmos
"""

import logging
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

from ..utils.logger_config import get_logger
from .base_model import BaseFootballPredictor, FootballClassifierMixin

# Logger específico para predicciones de resultado
logger = get_logger(__name__)


class MatchOutcomePredictor(BaseFootballPredictor, FootballClassifierMixin):
    """
    Predictor principal para resultados de partido (1X2).

    Predice la probabilidad de:
    - Home Win (1): Victoria del equipo local
    - Draw (X): Empate
    - Away Win (2): Victoria del equipo visitante
    """

    def __init__(self, model_algorithm: str = "xgboost"):
        """
        Inicializar predictor de resultados.

        Args:
            model_algorithm: Algoritmo a usar ('xgboost', 'lightgbm', 'catboost', 'random_forest')
        """
        super().__init__(
            model_name=f"match_outcome_{model_algorithm}",
            prediction_type="match_outcome",
            model_type="classifier",
        )

        self.model_algorithm = model_algorithm
        self.label_encoder = LabelEncoder()
        self.outcome_classes = ["home_win", "draw", "away_win"]

        # Features específicas para resultado de partido
        self.match_outcome_features = [
            "home_team_form",
            "away_team_form",
            "home_advantage",
            "head_to_head_record",
            "league_position_diff",
            "market_value_ratio",
            "home_goals_avg",
            "away_goals_avg",
            "home_conceded_avg",
            "away_conceded_avg",
            "home_attack_strength",
            "away_attack_strength",
            "home_defense_strength",
            "away_defense_strength",
            "match_importance",
            "referee_bias",
            "weather_conditions",
            "injuries_impact",
        ]

        logger.info(
            f"MatchOutcomePredictor initialized with {model_algorithm}"
        )

    def _create_model(self):
        """Crear modelo según algoritmo especificado."""
        if self.model_algorithm == "xgboost":
            return xgb.XGBClassifier(
                objective="multi:softprob",
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric="mlogloss",
            )
        elif self.model_algorithm == "lightgbm":
            return lgb.LGBMClassifier(
                objective="multiclass",
                num_class=3,
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbose=-1,
            )
        elif self.model_algorithm == "catboost":
            return CatBoostClassifier(
                objective="MultiClass",
                iterations=200,
                depth=6,
                learning_rate=0.1,
                random_seed=self.random_state,
                verbose=False,
            )
        elif self.model_algorithm == "random_forest":
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"Algoritmo no soportado: {self.model_algorithm}")

    def _get_default_hyperparameters(self) -> Dict:
        """Hiperparámetros por defecto según algoritmo."""
        defaults = {
            "xgboost": {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
            "lightgbm": {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
            "catboost": {"iterations": 200, "depth": 6, "learning_rate": 0.1},
            "random_forest": {
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
            },
        }
        return defaults.get(self.model_algorithm, {})

    def _get_hyperparameter_grid(self) -> Dict:
        """Grid de hiperparámetros para optimización."""
        grids = {
            "xgboost": {
                "n_estimators": [100, 200, 300],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.05, 0.1, 0.15],
                "subsample": [0.7, 0.8, 0.9],
                "colsample_bytree": [0.7, 0.8, 0.9],
            },
            "lightgbm": {
                "n_estimators": [100, 200, 300],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.05, 0.1, 0.15],
                "subsample": [0.7, 0.8, 0.9],
                "colsample_bytree": [0.7, 0.8, 0.9],
            },
            "catboost": {
                "iterations": [100, 200, 300],
                "depth": [4, 6, 8],
                "learning_rate": [0.05, 0.1, 0.15],
            },
            "random_forest": {
                "n_estimators": [100, 200, 300],
                "max_depth": [8, 10, 12],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
        }
        return grids.get(self.model_algorithm, {})

    def create_match_outcome_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crear features específicas para predicción de resultados.

        Args:
            df: DataFrame con datos del partido

        Returns:
            DataFrame con features engineered
        """
        logger.info("Creating match outcome features...")

        # Copiar DataFrame
        features_df = df.copy()

        # 1. Forma reciente de equipos (últimos 5 partidos)
        features_df["home_team_form"] = self._calculate_team_form(
            features_df, "home_team_id"
        )
        features_df["away_team_form"] = self._calculate_team_form(
            features_df, "away_team_id"
        )

        # 2. Fortaleza relativa
        features_df["attack_strength_ratio"] = (
            features_df["home_attack_strength"]
            / features_df["away_defense_strength"]
        )
        features_df["defense_strength_ratio"] = (
            features_df["home_defense_strength"]
            / features_df["away_attack_strength"]
        )

        # 3. Diferencia de posiciones en liga
        features_df["league_position_diff"] = (
            features_df["away_league_position"]
            - features_df["home_league_position"]
        )

        # 4. Ratio de valor de mercado
        features_df["market_value_ratio"] = (
            features_df["home_team_market_value"]
            / features_df["away_team_market_value"]
        )

        # 5. Ventaja de local (histórica)
        features_df["home_advantage"] = self._calculate_home_advantage(
            features_df
        )

        # 6. Historial cara a cara
        features_df["h2h_home_wins_ratio"] = self._calculate_h2h_ratio(
            features_df, "home_wins"
        )
        features_df["h2h_draws_ratio"] = self._calculate_h2h_ratio(
            features_df, "draws"
        )
        features_df["h2h_away_wins_ratio"] = self._calculate_h2h_ratio(
            features_df, "away_wins"
        )

        # 7. Factor de importancia del partido
        features_df["match_importance"] = self._calculate_match_importance(
            features_df
        )

        # 8. Impacto de lesiones
        features_df["home_injuries_impact"] = self._calculate_injuries_impact(
            features_df, "home"
        )
        features_df["away_injuries_impact"] = self._calculate_injuries_impact(
            features_df, "away"
        )

        # 9. Bias del árbitro (si está disponible)
        if "referee_id" in features_df.columns:
            features_df["referee_home_bias"] = self._calculate_referee_bias(
                features_df
            )

        # 10. Condiciones del partido
        features_df["weather_impact"] = self._calculate_weather_impact(
            features_df
        )

        logger.info(
            f"Created {len(features_df.columns)} features for match outcome prediction"
        )
        return features_df

    def predict_match_outcome_detailed(
        self, X: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Predicción detallada de resultado de partido.

        Args:
            X: Features del partido

        Returns:
            Diccionario con predicciones detalladas
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        logger.info("Making detailed match outcome predictions...")

        # Preparar features
        if not hasattr(X, "columns") or len(X.columns) < len(
            self.match_outcome_features
        ):
            X = self.create_match_outcome_features(X)

        # Predicciones de probabilidad
        probabilities = self.model.predict_proba(X)
        predictions = self.model.predict(X)

        # Análisis detallado
        results = []
        for i, (probs, pred) in enumerate(zip(probabilities, predictions)):

            # Probabilidades por resultado
            home_prob = probs[0]
            draw_prob = probs[1]
            away_prob = probs[2]

            # Resultado más probable
            outcome_map = {0: "home_win", 1: "draw", 2: "away_win"}
            predicted_outcome = outcome_map[pred]

            # Confianza de la predicción
            confidence = np.max(probs)
            confidence_level = self._get_confidence_level(confidence)

            # Análisis de valor para apuestas
            betting_analysis = self._analyze_betting_value(
                home_prob, draw_prob, away_prob
            )

            result = {
                "probabilities": {
                    "home_win": round(home_prob, 3),
                    "draw": round(draw_prob, 3),
                    "away_win": round(away_prob, 3),
                },
                "predicted_outcome": predicted_outcome,
                "confidence_score": round(confidence, 3),
                "confidence_level": confidence_level,
                "betting_analysis": betting_analysis,
                "match_outlook": self._get_match_outlook(
                    home_prob, draw_prob, away_prob
                ),
            }

            results.append(result)

        return {
            "prediction_type": "match_outcome",
            "model_algorithm": self.model_algorithm,
            "predictions": results,
            "model_info": {
                "accuracy": getattr(self, "last_accuracy", None),
                "log_loss": getattr(self, "last_log_loss", None),
            },
        }

    def _calculate_team_form(
        self, df: pd.DataFrame, team_col: str
    ) -> pd.Series:
        """Calcular forma reciente del equipo (últimos 5 partidos)."""
        # Simulación de forma reciente (en producción vendría de BD)
        np.random.seed(42)
        return pd.Series(np.random.uniform(0.3, 0.9, len(df)))

    def _calculate_home_advantage(self, df: pd.DataFrame) -> pd.Series:
        """Calcular ventaja de jugar en casa."""
        # Factor base de ventaja local (estadísticamente ~0.6)
        base_advantage = 0.6

        # Ajustar según características del estadio/equipo
        stadium_factor = (
            getattr(df, "stadium_capacity", pd.Series([30000] * len(df)))
            / 50000
        )
        return pd.Series([base_advantage] * len(df)) + stadium_factor * 0.1

    def _calculate_h2h_ratio(
        self, df: pd.DataFrame, result_type: str
    ) -> pd.Series:
        """Calcular ratio de resultados en historial cara a cara."""
        # Simulación (en producción vendría de BD histórica)
        np.random.seed(42)
        return pd.Series(np.random.uniform(0.2, 0.5, len(df)))

    def _calculate_match_importance(self, df: pd.DataFrame) -> pd.Series:
        """Calcular importancia del partido."""
        importance_factors = {
            "league": 1.0,
            "cup": 1.2,
            "champions_league": 1.5,
            "derby": 1.3,
            "relegation_battle": 1.4,
            "title_race": 1.4,
        }

        # Por defecto es partido de liga
        return pd.Series([1.0] * len(df))

    def _calculate_injuries_impact(
        self, df: pd.DataFrame, team_side: str
    ) -> pd.Series:
        """Calcular impacto de lesiones en el equipo."""
        # Simulación basada en número de jugadores lesionados
        np.random.seed(42)
        injured_players = np.random.randint(0, 5, len(df))
        impact = injured_players * 0.05  # 5% de impacto por jugador lesionado
        return pd.Series(np.minimum(impact, 0.25))  # Máximo 25% de impacto

    def _calculate_referee_bias(self, df: pd.DataFrame) -> pd.Series:
        """Calcular bias histórico del árbitro hacia equipos locales."""
        # Simulación (en producción sería análisis histórico del árbitro)
        np.random.seed(42)
        return pd.Series(np.random.uniform(-0.1, 0.1, len(df)))

    def _calculate_weather_impact(self, df: pd.DataFrame) -> pd.Series:
        """Calcular impacto de condiciones climáticas."""
        weather_impacts = {
            "sunny": 0.0,
            "cloudy": 0.0,
            "light_rain": -0.05,
            "heavy_rain": -0.15,
            "snow": -0.2,
            "strong_wind": -0.1,
        }

        # Por defecto condiciones normales
        return pd.Series([0.0] * len(df))

    def _get_confidence_level(self, confidence: float) -> str:
        """Determinar nivel de confianza textual."""
        if confidence >= 0.75:
            return "very_high"
        elif confidence >= 0.65:
            return "high"
        elif confidence >= 0.55:
            return "medium"
        elif confidence >= 0.45:
            return "low"
        else:
            return "very_low"

    def _analyze_betting_value(
        self, home_prob: float, draw_prob: float, away_prob: float
    ) -> Dict:
        """
        Analizar valor de apuesta basado en probabilidades.

        Args:
            home_prob: Probabilidad victoria local
            draw_prob: Probabilidad empate
            away_prob: Probabilidad victoria visitante

        Returns:
            Análisis de valor para apuestas
        """
        # Cuotas implícitas (simuladas)
        home_odds = 1 / home_prob if home_prob > 0 else 1.0
        draw_odds = 1 / draw_prob if draw_prob > 0 else 1.0
        away_odds = 1 / away_prob if away_prob > 0 else 1.0

        # Determinar mejor apuesta
        max_prob = max(home_prob, draw_prob, away_prob)

        if max_prob == home_prob:
            best_bet = "home_win"
            best_odds = home_odds
            best_prob = home_prob
        elif max_prob == draw_prob:
            best_bet = "draw"
            best_odds = draw_odds
            best_prob = draw_prob
        else:
            best_bet = "away_win"
            best_odds = away_odds
            best_prob = away_prob

        # Calcular valor esperado
        expected_value = (best_prob * best_odds) - 1

        return {
            "recommended_bet": best_bet,
            "implied_odds": round(best_odds, 2),
            "win_probability": round(best_prob, 3),
            "expected_value": round(expected_value, 3),
            "value_rating": (
                "positive"
                if expected_value > 0.05
                else "neutral" if expected_value > -0.05 else "negative"
            ),
            "confidence_recommendation": (
                "bet"
                if best_prob > 0.6
                else "consider" if best_prob > 0.5 else "avoid"
            ),
        }

    def _get_match_outlook(
        self, home_prob: float, draw_prob: float, away_prob: float
    ) -> str:
        """Determinar perspectiva general del partido."""
        max_prob = max(home_prob, draw_prob, away_prob)

        if max_prob > 0.6:
            if max_prob == home_prob:
                return "strong_home_favorite"
            elif max_prob == away_prob:
                return "strong_away_favorite"
            else:
                return "draw_likely"
        elif max_prob > 0.4:
            if max_prob == home_prob:
                return "home_slight_favorite"
            elif max_prob == away_prob:
                return "away_slight_favorite"
            else:
                return "competitive_match"
        else:
            return "very_competitive"

    def evaluate_model_detailed(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict:
        """Evaluación detallada del modelo."""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        logger.info("Evaluating match outcome model...")

        # Predicciones
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)

        # Métricas básicas
        accuracy = np.mean(y_pred == y_test)
        log_loss_score = log_loss(y_test, y_pred_proba)

        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_test, y_test, cv=5, scoring="accuracy"
        )

        # Reporte de clasificación
        class_report = classification_report(y_test, y_pred, output_dict=True)

        # Matriz de confusión
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Guardar métricas para uso posterior
        self.last_accuracy = accuracy
        self.last_log_loss = log_loss_score

        return {
            "accuracy": round(accuracy, 4),
            "log_loss": round(log_loss_score, 4),
            "cross_validation": {
                "mean_accuracy": round(cv_scores.mean(), 4),
                "std_accuracy": round(cv_scores.std(), 4),
                "individual_scores": [round(score, 4) for score in cv_scores],
            },
            "classification_report": class_report,
            "confusion_matrix": conf_matrix.tolist(),
            "model_algorithm": self.model_algorithm,
        }


class MatchOutcomeEnsemble:
    """
    Ensemble de múltiples predictores de resultado para máxima precisión.
    """

    def __init__(self, algorithms: List[str] = None):
        """
        Inicializar ensemble de predictores.

        Args:
            algorithms: Lista de algoritmos a usar. Por defecto: ['xgboost', 'lightgbm', 'catboost']
        """
        if algorithms is None:
            algorithms = ["xgboost", "lightgbm", "catboost"]

        self.algorithms = algorithms
        self.predictors = {}
        self.ensemble_model = None
        self.is_trained = False

        # Crear predictores individuales
        for algo in algorithms:
            self.predictors[algo] = MatchOutcomePredictor(model_algorithm=algo)

        logger.info(
            f"MatchOutcomeEnsemble initialized with algorithms: {algorithms}"
        )

    def train_ensemble(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        optimize_hyperparameters: bool = False,
    ):
        """
        Entrenar todos los predictores del ensemble.

        Args:
            X_train: Features de entrenamiento
            y_train: Labels de entrenamiento
            optimize_hyperparameters: Si optimizar hiperparámetros
        """
        logger.info("Training ensemble of match outcome predictors...")

        # Entrenar cada predictor individual
        trained_models = []
        for algo, predictor in self.predictors.items():
            logger.info(f"Training {algo} predictor...")

            predictor.train(
                X_train,
                y_train,
                optimize_hyperparameters=optimize_hyperparameters,
            )
            trained_models.append((algo, predictor.model))

        # Crear ensemble con voting
        voting_models = [(name, model) for name, model in trained_models]
        self.ensemble_model = VotingClassifier(
            estimators=voting_models, voting="soft"  # Usar probabilidades
        )

        # Entrenar ensemble
        X_features = self._prepare_features_for_ensemble(X_train)
        self.ensemble_model.fit(X_features, y_train)

        self.is_trained = True
        logger.info("Ensemble training completed successfully")

    def predict_ensemble(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Predicción usando ensemble completo.

        Args:
            X: Features para predicción

        Returns:
            Predicciones consolidadas del ensemble
        """
        if not self.is_trained:
            raise ValueError(
                "Ensemble must be trained before making predictions"
            )

        logger.info("Making ensemble predictions...")

        # Predicciones individuales
        individual_predictions = {}
        for algo, predictor in self.predictors.items():
            pred = predictor.predict_match_outcome_detailed(X)
            individual_predictions[algo] = pred

        # Predicción del ensemble
        X_features = self._prepare_features_for_ensemble(X)
        ensemble_proba = self.ensemble_model.predict_proba(X_features)
        ensemble_pred = self.ensemble_model.predict(X_features)

        # Consolidar resultados
        consolidated_results = []
        for i, (probs, pred) in enumerate(zip(ensemble_proba, ensemble_pred)):

            # Probabilidades ensemble
            home_prob = probs[0]
            draw_prob = probs[1]
            away_prob = probs[2]

            # Resultado más probable
            outcome_map = {0: "home_win", 1: "draw", 2: "away_win"}
            predicted_outcome = outcome_map[pred]

            # Confianza y análisis
            confidence = np.max(probs)

            # Acuerdo entre modelos
            individual_outcomes = [
                pred_data["predictions"][i]["predicted_outcome"]
                for pred_data in individual_predictions.values()
            ]
            model_agreement = len(set(individual_outcomes)) == 1

            result = {
                "ensemble_probabilities": {
                    "home_win": round(home_prob, 3),
                    "draw": round(draw_prob, 3),
                    "away_win": round(away_prob, 3),
                },
                "ensemble_prediction": predicted_outcome,
                "ensemble_confidence": round(confidence, 3),
                "model_agreement": model_agreement,
                "individual_predictions": {
                    algo: pred_data["predictions"][i]
                    for algo, pred_data in individual_predictions.items()
                },
                "consensus_strength": "strong" if model_agreement else "weak",
            }

            consolidated_results.append(result)

        return {
            "prediction_type": "match_outcome_ensemble",
            "algorithms_used": self.algorithms,
            "predictions": consolidated_results,
            "ensemble_info": {
                "total_models": len(self.predictors),
                "voting_method": "soft",
            },
        }

    def _prepare_features_for_ensemble(self, X: pd.DataFrame) -> pd.DataFrame:
        """Preparar features para el ensemble."""
        # Usar features del primer predictor como base
        first_predictor = next(iter(self.predictors.values()))
        return first_predictor.create_match_outcome_features(X)

    def evaluate_ensemble(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict:
        """Evaluar performance del ensemble."""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before evaluation")

        logger.info("Evaluating ensemble performance...")

        # Evaluar cada modelo individual
        individual_evaluations = {}
        for algo, predictor in self.predictors.items():
            eval_result = predictor.evaluate_model_detailed(X_test, y_test)
            individual_evaluations[algo] = eval_result

        # Evaluar ensemble
        X_features = self._prepare_features_for_ensemble(X_test)
        ensemble_pred = self.ensemble_model.predict(X_features)
        ensemble_proba = self.ensemble_model.predict_proba(X_features)

        ensemble_accuracy = np.mean(ensemble_pred == y_test)
        ensemble_log_loss = log_loss(y_test, ensemble_proba)

        return {
            "ensemble_metrics": {
                "accuracy": round(ensemble_accuracy, 4),
                "log_loss": round(ensemble_log_loss, 4),
            },
            "individual_metrics": individual_evaluations,
            "best_individual_model": max(
                individual_evaluations.items(), key=lambda x: x[1]["accuracy"]
            )[0],
        }


# Funciones de utilidad para crear predictores
def create_match_outcome_predictor(
    predictor_type: str = "xgboost",
) -> Union[MatchOutcomePredictor, MatchOutcomeEnsemble]:
    """
    Crear predictor de resultado de partido.

    Args:
        predictor_type: Tipo de predictor ('xgboost', 'lightgbm', 'catboost', 'random_forest', 'ensemble')

    Returns:
        Predictor configurado
    """
    if predictor_type == "ensemble":
        return MatchOutcomeEnsemble()
    else:
        return MatchOutcomePredictor(model_algorithm=predictor_type)


class MatchOutcomeAnalyzer:
    """Utilidades para análisis de resultados de partidos."""

    @staticmethod
    def calculate_expected_points(
        home_prob: float,
        draw_prob: float,
        away_prob: float,
        perspective: str = "home",
    ) -> float:
        """
        Calcular puntos esperados desde perspectiva de un equipo.

        Args:
            home_prob: Probabilidad victoria local
            draw_prob: Probabilidad empate
            away_prob: Probabilidad victoria visitante
            perspective: Perspectiva del cálculo ('home' o 'away')

        Returns:
            Puntos esperados (0-3)
        """
        if perspective == "home":
            return (home_prob * 3) + (draw_prob * 1) + (away_prob * 0)
        else:
            return (away_prob * 3) + (draw_prob * 1) + (home_prob * 0)

    @staticmethod
    def analyze_betting_efficiency(
        predictions: List[Dict], actual_odds: List[Dict]
    ) -> Dict:
        """
        Analizar eficiencia de predicciones vs cuotas reales del mercado.

        Args:
            predictions: Lista de predicciones del modelo
            actual_odds: Lista de cuotas reales del mercado

        Returns:
            Análisis de eficiencia y value bets
        """
        value_bets = []
        total_value = 0

        for pred, odds in zip(predictions, actual_odds):
            model_probs = pred["probabilities"]

            # Calcular probabilidades implícitas del mercado
            market_probs = {
                "home_win": (
                    1 / odds["home_win"] if odds["home_win"] > 0 else 0
                ),
                "draw": 1 / odds["draw"] if odds["draw"] > 0 else 0,
                "away_win": (
                    1 / odds["away_win"] if odds["away_win"] > 0 else 0
                ),
            }

            # Normalizar probabilidades del mercado
            total_market_prob = sum(market_probs.values())
            market_probs = {
                k: v / total_market_prob for k, v in market_probs.items()
            }

            # Encontrar value bets
            for outcome in ["home_win", "draw", "away_win"]:
                model_prob = model_probs[outcome]
                market_prob = market_probs[outcome]

                # Value bet si modelo predice mayor probabilidad que el mercado
                if model_prob > market_prob * 1.1:  # 10% threshold
                    value = (model_prob * odds[outcome]) - 1
                    if value > 0.05:  # Mínimo 5% de value
                        value_bets.append(
                            {
                                "outcome": outcome,
                                "model_probability": round(model_prob, 3),
                                "market_probability": round(market_prob, 3),
                                "odds": odds[outcome],
                                "expected_value": round(value, 3),
                            }
                        )
                        total_value += value

        return {
            "total_value_bets": len(value_bets),
            "average_value": round(
                total_value / len(predictions) if predictions else 0, 3
            ),
            "value_bets": value_bets,
            "market_efficiency": (
                "low"
                if len(value_bets) > len(predictions) * 0.3
                else (
                    "medium"
                    if len(value_bets) > len(predictions) * 0.1
                    else "high"
                )
            ),
        }

    @staticmethod
    def calculate_form_momentum(recent_results: List[str]) -> Dict:
        """
        Calcular momentum de forma basado en resultados recientes.

        Args:
            recent_results: Lista de resultados recientes ['W', 'D', 'L', 'W', 'L']

        Returns:
            Análisis de momentum
        """
        if not recent_results:
            return {"momentum": "unknown", "trend": "stable", "confidence": 0}

        # Puntajes por resultado
        result_points = {"W": 3, "D": 1, "L": 0}

        # Calcular puntos con pesos decrecientes (más reciente = más peso)
        weighted_points = 0
        total_weight = 0

        for i, result in enumerate(reversed(recent_results)):
            weight = 1.0 + (
                i * 0.2
            )  # Peso incrementa para resultados más recientes
            points = result_points.get(result, 0)
            weighted_points += points * weight
            total_weight += weight * 3  # Máximo 3 puntos por partido

        form_score = weighted_points / total_weight if total_weight > 0 else 0

        # Determinar momentum
        if form_score >= 0.7:
            momentum = "excellent"
        elif form_score >= 0.5:
            momentum = "good"
        elif form_score >= 0.3:
            momentum = "average"
        else:
            momentum = "poor"

        # Analizar tendencia
        if len(recent_results) >= 3:
            recent_3 = recent_results[-3:]
            previous_3 = (
                recent_results[-6:-3] if len(recent_results) >= 6 else []
            )

            recent_avg = sum(result_points.get(r, 0) for r in recent_3) / (
                len(recent_3) * 3
            )
            previous_avg = (
                sum(result_points.get(r, 0) for r in previous_3)
                / (len(previous_3) * 3)
                if previous_3
                else recent_avg
            )

            if recent_avg > previous_avg + 0.1:
                trend = "improving"
            elif recent_avg < previous_avg - 0.1:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        return {
            "momentum": momentum,
            "form_score": round(form_score, 3),
            "trend": trend,
            "recent_results": recent_results,
            "confidence": min(
                len(recent_results) / 5.0, 1.0
            ),  # Máxima confianza con 5+ partidos
        }


class MatchOutcomeMetrics:
    """Métricas especializadas para evaluación de predictores de resultados."""

    @staticmethod
    def calculate_outcome_accuracy(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict:
        """
        Calcular precisión específica por tipo de resultado.

        Args:
            y_true: Resultados reales
            y_pred: Predicciones del modelo

        Returns:
            Métricas detalladas por outcome
        """
        outcomes = ["home_win", "draw", "away_win"]
        metrics = {}

        for i, outcome in enumerate(outcomes):
            # Máscaras para este outcome
            true_mask = y_true == i
            pred_mask = y_pred == i

            # Métricas básicas
            true_positives = np.sum(true_mask & pred_mask)
            false_positives = np.sum(~true_mask & pred_mask)
            false_negatives = np.sum(true_mask & ~pred_mask)

            # Precision, Recall, F1
            precision = (
                true_positives / (true_positives + false_positives)
                if (true_positives + false_positives) > 0
                else 0
            )
            recall = (
                true_positives / (true_positives + false_negatives)
                if (true_positives + false_negatives) > 0
                else 0
            )
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            metrics[outcome] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "support": int(np.sum(true_mask)),
            }

        # Métricas generales
        overall_accuracy = np.mean(y_true == y_pred)

        return {
            "overall_accuracy": round(overall_accuracy, 4),
            "by_outcome": metrics,
            "total_samples": len(y_true),
        }

    @staticmethod
    def calculate_probability_calibration(
        y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
    ) -> Dict:
        """
        Evaluar calibración de probabilidades predichas.

        Args:
            y_true: Resultados reales
            y_proba: Probabilidades predichas
            n_bins: Número de bins para calibración

        Returns:
            Métricas de calibración
        """
        from sklearn.calibration import calibration_curve

        calibration_results = {}

        for outcome_idx in range(y_proba.shape[1]):
            # Convertir a problema binario para este outcome
            y_binary = (y_true == outcome_idx).astype(int)
            y_prob_outcome = y_proba[:, outcome_idx]

            # Curva de calibración
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_binary, y_prob_outcome, n_bins=n_bins
            )

            # Brier Score (lower is better)
            brier_score = np.mean((y_prob_outcome - y_binary) ** 2)

            outcome_name = ["home_win", "draw", "away_win"][outcome_idx]
            calibration_results[outcome_name] = {
                "brier_score": round(brier_score, 4),
                "calibration_curve": {
                    "fraction_of_positives": fraction_of_positives.tolist(),
                    "mean_predicted_value": mean_predicted_value.tolist(),
                },
            }

        return calibration_results

    @staticmethod
    def evaluate_betting_performance(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        odds_data: List[Dict],
        stake: float = 1.0,
    ) -> Dict:
        """
        Evaluar performance de estrategia de apuestas basada en predicciones.

        Args:
            y_true: Resultados reales
            y_proba: Probabilidades predichas
            odds_data: Cuotas para cada partido
            stake: Cantidad apostada por partido

        Returns:
            Métricas de performance de apuestas
        """
        total_stake = 0
        total_return = 0
        winning_bets = 0
        total_bets = 0
        bet_details = []

        outcome_names = ["home_win", "draw", "away_win"]

        for i, (true_outcome, probs, odds) in enumerate(
            zip(y_true, y_proba, odds_data)
        ):
            # Encontrar la apuesta con mayor valor esperado
            best_value = -1
            best_bet = None

            for outcome_idx, outcome_name in enumerate(outcome_names):
                prob = probs[outcome_idx]
                odd = odds.get(outcome_name, 0)

                if odd > 0:
                    expected_value = (prob * odd) - 1

                    # Solo apostar si hay valor positivo y probabilidad razonable
                    if expected_value > 0.05 and prob > 0.3:
                        if expected_value > best_value:
                            best_value = expected_value
                            best_bet = {
                                "outcome": outcome_name,
                                "outcome_idx": outcome_idx,
                                "probability": prob,
                                "odds": odd,
                                "expected_value": expected_value,
                            }

            # Realizar apuesta si encontramos valor
            if best_bet:
                total_bets += 1
                total_stake += stake

                # Verificar si ganamos
                if true_outcome == best_bet["outcome_idx"]:
                    winning_bets += 1
                    return_amount = stake * best_bet["odds"]
                    total_return += return_amount
                    profit = return_amount - stake
                else:
                    profit = -stake

                bet_details.append(
                    {
                        "match_id": i,
                        "bet_outcome": best_bet["outcome"],
                        "actual_outcome": outcome_names[true_outcome],
                        "odds": best_bet["odds"],
                        "probability": best_bet["probability"],
                        "stake": stake,
                        "profit": profit,
                        "won": true_outcome == best_bet["outcome_idx"],
                    }
                )

        # Calcular métricas
        net_profit = total_return - total_stake
        roi = (net_profit / total_stake * 100) if total_stake > 0 else 0
        win_rate = (winning_bets / total_bets * 100) if total_bets > 0 else 0
        average_odds = (
            np.mean([bet["odds"] for bet in bet_details]) if bet_details else 0
        )

        return {
            "total_bets": total_bets,
            "winning_bets": winning_bets,
            "win_rate_percent": round(win_rate, 2),
            "total_stake": round(total_stake, 2),
            "total_return": round(total_return, 2),
            "net_profit": round(net_profit, 2),
            "roi_percent": round(roi, 2),
            "average_odds": round(average_odds, 2),
            "profit_factor": (
                round(total_return / total_stake, 2) if total_stake > 0 else 0
            ),
            "bet_details": bet_details,
        }


# Configuración de logging
logger.info("Match outcome predictor module loaded successfully")

# Exportar clases principales
__all__ = [
    "MatchOutcomePredictor",
    "MatchOutcomeEnsemble",
    "MatchOutcomeAnalyzer",
    "MatchOutcomeMetrics",
    "create_match_outcome_predictor",
]
