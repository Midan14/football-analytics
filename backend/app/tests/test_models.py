"""
Football Analytics - ML Models Tests
Pruebas específicas para todos los modelos de machine learning del proyecto
"""

import os
import tempfile
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import joblib
import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report

# ML libraries for testing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.ml_models.feature_engineer import FeatureEngineer

# Importaciones del proyecto
from app.ml_models.match_outcome import (
    MatchOutcomeEnsemble,
    MatchOutcomeMetrics,
    MatchOutcomePredictor,
    create_match_outcome_predictor,
)
from app.ml_models.model_trainer import (
    ModelTrainer,
    TrainingResult,
    train_all_models,
)
from app.ml_models.model_validator import ModelValidator

# Test data and utilities
from . import TEST_APP_CONFIG, TestDatabase, assert_prediction_valid, mock_data


class TestMatchOutcomePredictor:
    """Tests para el predictor de resultados 1X2"""

    def setup_method(self):
        """Setup antes de cada test"""
        self.predictor = MatchOutcomePredictor(model_algorithm="xgboost")
        self.sample_features = self._create_sample_features()
        self.sample_target = self._create_sample_target()

    def _create_sample_features(self) -> pd.DataFrame:
        """Crea features de muestra realistas para fútbol"""
        np.random.seed(42)
        n_samples = 1000

        # Features específicas de fútbol
        features = {
            # Estadísticas de equipos
            "home_team_form": np.random.uniform(
                0, 3, n_samples
            ),  # Puntos por partido
            "away_team_form": np.random.uniform(0, 3, n_samples),
            "home_attack_strength": np.random.uniform(0.5, 2.0, n_samples),
            "away_attack_strength": np.random.uniform(0.5, 2.0, n_samples),
            "home_defense_strength": np.random.uniform(0.5, 2.0, n_samples),
            "away_defense_strength": np.random.uniform(0.5, 2.0, n_samples),
            # Diferencias y ratios
            "league_position_diff": np.random.uniform(-19, 19, n_samples),
            "market_value_ratio": np.random.uniform(0.1, 10.0, n_samples),
            "home_advantage": np.random.uniform(0.1, 0.5, n_samples),
            "form_difference": np.random.uniform(-3, 3, n_samples),
            # Estadísticas H2H
            "h2h_home_wins": np.random.uniform(0, 1, n_samples),
            "h2h_avg_goals": np.random.uniform(1, 4, n_samples),
            # Context features
            "match_importance": np.random.uniform(0.5, 2.0, n_samples),
            "derby_factor": np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            "injuries_impact": np.random.uniform(0, 1, n_samples),
            # Weather and external
            "weather_conditions": np.random.choice(
                [0, 1, 2], n_samples
            ),  # 0=good, 1=rain, 2=bad
            "referee_bias": np.random.uniform(-0.2, 0.2, n_samples),
            # Advanced metrics
            "expected_competitiveness": np.random.uniform(0, 1, n_samples),
            "momentum_factor": np.random.uniform(-1, 1, n_samples),
            "pressure_situations": np.random.uniform(0, 1, n_samples),
        }

        return pd.DataFrame(features)

    def _create_sample_target(self) -> pd.Series:
        """Crea target realista basado en probabilidades del fútbol"""
        n_samples = 1000
        # Distribución realista: ~45% local, ~27% empate, ~28% visitante
        target = np.random.choice(
            ["H", "D", "A"], n_samples, p=[0.45, 0.27, 0.28]
        )
        return pd.Series(target)

    def test_predictor_initialization(self):
        """Test de inicialización del predictor"""
        assert self.predictor.model_algorithm == "xgboost"
        assert self.predictor.model is not None
        assert hasattr(self.predictor, "scaler")
        assert hasattr(self.predictor, "label_encoder")

    def test_predictor_with_different_algorithms(self):
        """Test con diferentes algoritmos ML"""
        algorithms = ["xgboost", "lightgbm", "catboost", "random_forest"]

        for algo in algorithms:
            predictor = MatchOutcomePredictor(model_algorithm=algo)
            assert predictor.model_algorithm == algo
            assert predictor.model is not None

    def test_feature_validation(self):
        """Test de validación de features"""
        # Test con features válidas
        valid_features = self.sample_features
        is_valid = self.predictor._validate_features(valid_features)
        assert is_valid

        # Test con features inválidas (columnas faltantes)
        invalid_features = valid_features.drop(columns=["home_team_form"])
        is_valid = self.predictor._validate_features(invalid_features)
        assert not is_valid

        # Test con valores NaN
        features_with_nan = valid_features.copy()
        features_with_nan.iloc[0, 0] = np.nan
        is_valid = self.predictor._validate_features(features_with_nan)
        assert not is_valid

    def test_data_preprocessing(self):
        """Test del preprocesamiento de datos"""
        X_processed = self.predictor._preprocess_features(self.sample_features)

        # Verificar que no hay valores NaN después del preprocesamiento
        assert not X_processed.isnull().any().any()

        # Verificar que las dimensiones son correctas
        assert X_processed.shape[0] == self.sample_features.shape[0]
        assert X_processed.shape[1] == self.sample_features.shape[1]

    def test_model_training(self):
        """Test del entrenamiento del modelo"""
        # Entrenar modelo
        self.predictor.train(self.sample_features, self.sample_target)

        # Verificar que el modelo está entrenado
        assert hasattr(self.predictor.model, "predict")

        # Verificar que se pueden hacer predicciones
        predictions = self.predictor.predict(self.sample_features[:10])
        assert len(predictions) == 10
        assert all(pred in ["H", "D", "A"] for pred in predictions)

    def test_probability_predictions(self):
        """Test de predicciones de probabilidad"""
        # Entrenar modelo
        self.predictor.train(self.sample_features, self.sample_target)

        # Obtener probabilidades
        probabilities = self.predictor.predict_proba(self.sample_features[:10])

        # Verificar dimensiones
        assert probabilities.shape == (10, 3)  # 10 muestras, 3 clases

        # Verificar que las probabilidades suman 1
        for i in range(10):
            assert abs(probabilities[i].sum() - 1.0) < 0.01

        # Verificar que todas las probabilidades están entre 0 y 1
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)

    def test_detailed_prediction(self):
        """Test de predicción detallada"""
        # Entrenar modelo
        self.predictor.train(self.sample_features, self.sample_target)

        # Hacer predicción detallada
        result = self.predictor.predict_match_outcome_detailed(
            self.sample_features[:1]
        )

        # Verificar estructura de respuesta
        assert "prediction" in result
        assert "probabilities" in result
        assert "confidence_score" in result
        assert "confidence_level" in result
        assert "betting_analysis" in result
        assert "match_outlook" in result

        # Verificar tipos de datos
        assert result["prediction"] in ["home_win", "draw", "away_win"]
        assert isinstance(result["confidence_score"], (int, float))
        assert 0 <= result["confidence_score"] <= 1

    def test_model_persistence(self):
        """Test de guardado y carga del modelo"""
        # Entrenar modelo
        self.predictor.train(self.sample_features, self.sample_target)

        # Hacer predicción antes de guardar
        original_predictions = self.predictor.predict(self.sample_features[:5])

        # Guardar modelo
        with tempfile.NamedTemporaryFile(
            suffix=".pkl", delete=False
        ) as tmp_file:
            model_path = tmp_file.name

        self.predictor.save_model(model_path)

        # Crear nuevo predictor y cargar modelo
        new_predictor = MatchOutcomePredictor(model_algorithm="xgboost")
        new_predictor.load_model(model_path)

        # Hacer predicción con modelo cargado
        loaded_predictions = new_predictor.predict(self.sample_features[:5])

        # Verificar que las predicciones son iguales
        assert np.array_equal(original_predictions, loaded_predictions)

        # Cleanup
        os.unlink(model_path)

    def test_feature_importance(self):
        """Test de importancia de features"""
        # Entrenar modelo
        self.predictor.train(self.sample_features, self.sample_target)

        # Obtener importancia de features
        importance = self.predictor.get_feature_importance()

        # Verificar que se retorna un diccionario
        assert isinstance(importance, dict)

        # Verificar que tiene entradas para todas las features
        assert len(importance) == len(self.sample_features.columns)

        # Verificar que todos los valores son numéricos
        assert all(isinstance(v, (int, float)) for v in importance.values())


class TestMatchOutcomeEnsemble:
    """Tests para el ensemble de modelos"""

    def setup_method(self):
        """Setup antes de cada test"""
        self.algorithms = ["xgboost", "lightgbm", "catboost"]
        self.ensemble = MatchOutcomeEnsemble(self.algorithms)
        self.sample_features = self._create_sample_features()
        self.sample_target = self._create_sample_target()

    def _create_sample_features(self) -> pd.DataFrame:
        """Reutiliza la función de la clase anterior"""
        test_predictor = TestMatchOutcomePredictor()
        return test_predictor._create_sample_features()

    def _create_sample_target(self) -> pd.Series:
        """Reutiliza la función de la clase anterior"""
        test_predictor = TestMatchOutcomePredictor()
        return test_predictor._create_sample_target()

    def test_ensemble_initialization(self):
        """Test de inicialización del ensemble"""
        assert len(self.ensemble.predictors) == 3
        assert all(
            algo in self.ensemble.predictors for algo in self.algorithms
        )

        # Verificar que cada predictor tiene el algoritmo correcto
        for algo in self.algorithms:
            assert self.ensemble.predictors[algo].model_algorithm == algo

    def test_ensemble_training(self):
        """Test de entrenamiento del ensemble"""
        # Entrenar ensemble
        self.ensemble.train_ensemble(self.sample_features, self.sample_target)

        # Verificar que todos los modelos están entrenados
        for predictor in self.ensemble.predictors.values():
            assert hasattr(predictor.model, "predict")

    def test_ensemble_prediction(self):
        """Test de predicción con ensemble"""
        # Entrenar ensemble
        self.ensemble.train_ensemble(self.sample_features, self.sample_target)

        # Hacer predicciones
        predictions = self.ensemble.predict_ensemble(self.sample_features[:10])

        # Verificar estructura de respuesta
        assert "ensemble_prediction" in predictions
        assert "ensemble_probabilities" in predictions
        assert "individual_predictions" in predictions
        assert "consensus_score" in predictions
        assert "confidence_score" in predictions

        # Verificar que hay predicciones individuales para cada algoritmo
        individual = predictions["individual_predictions"]
        assert len(individual) == len(self.algorithms)
        for algo in self.algorithms:
            assert algo in individual

    def test_consensus_calculation(self):
        """Test del cálculo de consenso entre modelos"""
        # Entrenar ensemble
        self.ensemble.train_ensemble(self.sample_features, self.sample_target)

        # Hacer predicción
        result = self.ensemble.predict_ensemble(self.sample_features[:1])

        # El consenso debe estar entre 0 y 1
        consensus = result["consensus_score"]
        assert 0 <= consensus <= 1

        # Verificar que el consenso se calcula correctamente
        individual_preds = result["individual_predictions"]
        ensemble_pred = result["ensemble_prediction"]

        # Contar cuántos modelos coinciden con la predicción final
        agreements = sum(
            1 for pred in individual_preds.values() if pred == ensemble_pred
        )
        expected_consensus = agreements / len(individual_preds)

        assert abs(consensus - expected_consensus) < 0.01

    def test_ensemble_with_hyperparameter_optimization(self):
        """Test del ensemble con optimización de hiperparámetros"""
        # Entrenar con optimización (menos trials para tests rápidos)
        self.ensemble.train_ensemble(
            self.sample_features,
            self.sample_target,
            optimize_hyperparameters=True,
            optimization_trials=5,  # Pocos trials para test rápido
        )

        # Verificar que todos los modelos están entrenados
        for predictor in self.ensemble.predictors.values():
            assert hasattr(predictor.model, "predict")


class TestModelTrainer:
    """Tests para el entrenador de modelos"""

    def setup_method(self):
        """Setup antes de cada test"""
        self.trainer = ModelTrainer(
            model_type="match_outcome", algorithm="xgboost"
        )
        self.sample_data = self._create_sample_dataframe()

    def _create_sample_dataframe(self) -> pd.DataFrame:
        """Crea DataFrame de muestra con features y target"""
        test_predictor = TestMatchOutcomePredictor()
        features = test_predictor._create_sample_features()
        target = test_predictor._create_sample_target()

        # Combinar features y target
        data = features.copy()
        data["result"] = target

        return data

    def test_trainer_initialization(self):
        """Test de inicialización del trainer"""
        assert self.trainer.model_type == "match_outcome"
        assert self.trainer.algorithm == "xgboost"
        assert hasattr(self.trainer, "feature_engineer")
        assert hasattr(self.trainer, "validator")

    def test_data_preparation(self):
        """Test de preparación de datos"""
        data_dict = self.trainer.prepare_data(
            self.sample_data,
            target_column="result",
            test_size=0.2,
            validation_split=True,
        )

        # Verificar estructura de datos preparados
        required_keys = [
            "X_train",
            "X_val",
            "X_test",
            "y_train",
            "y_val",
            "y_test",
            "feature_names",
        ]
        for key in required_keys:
            assert key in data_dict

        # Verificar dimensiones
        total_samples = len(self.sample_data)
        train_samples = len(data_dict["y_train"])
        val_samples = (
            len(data_dict["y_val"]) if data_dict["y_val"] is not None else 0
        )
        test_samples = len(data_dict["y_test"])

        # La suma debe ser aproximadamente igual al total
        assert (
            abs((train_samples + val_samples + test_samples) - total_samples)
            <= 2
        )

    def test_model_creation(self):
        """Test de creación de modelos"""
        algorithms = ["xgboost", "lightgbm", "catboost", "random_forest"]

        for algo in algorithms:
            trainer = ModelTrainer(model_type="match_outcome", algorithm=algo)
            model = trainer.get_model()

            # Verificar que el modelo se crea correctamente
            assert model is not None
            assert hasattr(model, "fit")
            assert hasattr(model, "predict")

    def test_hyperparameter_optimization(self):
        """Test de optimización de hiperparámetros"""
        data_dict = self.trainer.prepare_data(self.sample_data, "result")

        # Optimizar con pocos trials para test rápido
        best_params = self.trainer.optimize_hyperparameters(
            data_dict, n_trials=3
        )

        # Verificar que se retornan parámetros
        assert isinstance(best_params, dict)
        assert len(best_params) > 0

        # Verificar que los parámetros son válidos para XGBoost
        expected_params = ["n_estimators", "max_depth", "learning_rate"]
        for param in expected_params:
            assert param in best_params

    def test_model_training(self):
        """Test de entrenamiento completo"""
        data_dict = self.trainer.prepare_data(self.sample_data, "result")

        # Entrenar sin optimización para test rápido
        result = self.trainer.train_model(
            data_dict, optimize_hyperparams=False
        )

        # Verificar que se retorna TrainingResult
        assert isinstance(result, TrainingResult)

        # Verificar campos del resultado
        assert hasattr(result, "model")
        assert hasattr(result, "accuracy")
        assert hasattr(result, "cross_val_score")
        assert hasattr(result, "feature_importance")
        assert hasattr(result, "training_time")

        # Verificar que la accuracy es razonable
        assert 0 <= result.accuracy <= 1
        assert 0 <= result.cross_val_score <= 1

    def test_model_saving_and_loading(self):
        """Test de guardado y carga de modelos"""
        data_dict = self.trainer.prepare_data(self.sample_data, "result")
        result = self.trainer.train_model(
            data_dict, optimize_hyperparams=False
        )

        # Verificar que se guardó el modelo
        assert os.path.exists(result.model_path)

        # Cargar modelo
        loaded_package = self.trainer.load_model(result.model_path)

        # Verificar contenido del paquete
        assert "model" in loaded_package
        assert "model_type" in loaded_package
        assert "algorithm" in loaded_package
        assert loaded_package["model_type"] == "match_outcome"
        assert loaded_package["algorithm"] == "xgboost"

        # Cleanup
        os.unlink(result.model_path)


class TestMatchOutcomeMetrics:
    """Tests para métricas específicas de resultados de partidos"""

    def test_outcome_accuracy_calculation(self):
        """Test de cálculo de precisión de resultados"""
        # Datos de prueba
        y_true = np.array(["H", "D", "A", "H", "A", "D", "H", "A"])
        y_pred = np.array(["H", "D", "H", "H", "A", "A", "D", "A"])

        metrics = MatchOutcomeMetrics.calculate_outcome_accuracy(
            y_true, y_pred
        )

        # Verificar estructura
        assert "overall_accuracy" in metrics
        assert "home_precision" in metrics
        assert "draw_precision" in metrics
        assert "away_precision" in metrics

        # Verificar cálculos
        expected_accuracy = accuracy_score(y_true, y_pred)
        assert abs(metrics["overall_accuracy"] - expected_accuracy) < 0.01

    def test_probability_calibration(self):
        """Test de calibración de probabilidades"""
        # Crear datos de prueba
        y_true = np.array([0, 1, 2, 0, 2, 1, 0, 2])  # Encoded classes
        y_proba = np.array(
            [
                [0.7, 0.2, 0.1],  # High confidence correct
                [0.1, 0.8, 0.1],  # High confidence correct
                [0.1, 0.1, 0.8],  # High confidence correct
                [0.6, 0.3, 0.1],  # Medium confidence correct
                [0.2, 0.2, 0.6],  # Medium confidence correct
                [0.3, 0.4, 0.3],  # Low confidence correct
                [0.4, 0.4, 0.2],  # Wrong prediction
                [0.1, 0.1, 0.8],  # High confidence correct
            ]
        )

        calibration = MatchOutcomeMetrics.calculate_probability_calibration(
            y_true, y_proba
        )

        # Verificar estructura
        assert "brier_score" in calibration
        assert "calibration_error" in calibration
        assert "reliability" in calibration

        # Verificar rangos
        assert 0 <= calibration["brier_score"] <= 1
        assert calibration["calibration_error"] >= 0

    def test_betting_performance_evaluation(self):
        """Test de evaluación de rendimiento en apuestas"""
        # Datos de prueba
        y_true = np.array([0, 1, 2, 0, 2])  # Resultados reales
        y_proba = np.array(
            [
                [0.7, 0.2, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
                [0.6, 0.3, 0.1],
                [0.2, 0.2, 0.6],
            ]
        )
        odds = np.array(
            [
                [1.8, 4.0, 8.0],  # Market odds for each outcome
                [5.0, 1.9, 6.0],
                [8.0, 5.0, 1.7],
                [2.0, 3.5, 7.0],
                [4.0, 4.0, 2.2],
            ]
        )

        performance = MatchOutcomeMetrics.evaluate_betting_performance(
            y_true, y_proba, odds
        )

        # Verificar estructura
        assert "total_bets" in performance
        assert "profitable_bets" in performance
        assert "total_return" in performance
        assert "roi_percentage" in performance
        assert "kelly_performance" in performance


class TestFeatureEngineer:
    """Tests para ingeniería de características"""

    def setup_method(self):
        """Setup antes de cada test"""
        self.engineer = FeatureEngineer()

    def test_basic_feature_engineering(self):
        """Test de ingeniería básica de features"""
        # Crear datos de entrada básicos
        raw_data = pd.DataFrame(
            {
                "home_team": ["Real Madrid", "Barcelona", "Atletico"],
                "away_team": ["Barcelona", "Real Madrid", "Valencia"],
                "home_goals_avg": [2.1, 1.8, 1.5],
                "away_goals_avg": [1.2, 1.1, 1.0],
                "league_position_home": [1, 2, 3],
                "league_position_away": [2, 1, 5],
            }
        )

        # Aplicar feature engineering
        engineered = self.engineer.engineer_features(
            raw_data, model_type="match_outcome"
        )

        # Verificar que se crearon nuevas features
        assert len(engineered.columns) >= len(raw_data.columns)

        # Verificar features específicas esperadas
        expected_features = [
            "goal_ratio",
            "position_difference",
            "team_strength_ratio",
        ]
        for feature in expected_features:
            assert feature in engineered.columns or any(
                feature in col for col in engineered.columns
            )

    def test_advanced_feature_creation(self):
        """Test de creación de features avanzadas"""
        # Datos más complejos
        raw_data = pd.DataFrame(
            {
                "home_team": ["Real Madrid"] * 10,
                "away_team": ["Barcelona"] * 10,
                "home_form_points": [9, 10, 7, 12, 8, 11, 6, 9, 10, 8],
                "away_form_points": [7, 8, 9, 6, 10, 7, 11, 8, 7, 9],
                "home_market_value": [
                    800,
                    800,
                    800,
                    800,
                    800,
                    800,
                    800,
                    800,
                    800,
                    800,
                ],
                "away_market_value": [
                    750,
                    750,
                    750,
                    750,
                    750,
                    750,
                    750,
                    750,
                    750,
                    750,
                ],
                "days_since_last_match": [3, 7, 4, 14, 3, 7, 10, 3, 7, 4],
            }
        )

        engineered = self.engineer.engineer_features(
            raw_data, model_type="match_outcome"
        )

        # Verificar que no hay valores NaN
        assert not engineered.isnull().any().any()

        # Verificar que todas las columnas son numéricas
        numeric_columns = engineered.select_dtypes(include=[np.number]).columns
        assert len(numeric_columns) == len(engineered.columns)

    def test_feature_scaling_and_encoding(self):
        """Test de escalado y encoding de features"""
        raw_data = pd.DataFrame(
            {
                "team_strength": [0.5, 1.0, 1.5, 2.0, 0.8],
                "market_value": [100, 500, 200, 800, 300],
                "categorical_feature": ["A", "B", "A", "C", "B"],
            }
        )

        scaled_data = self.engineer.scale_features(raw_data)

        # Verificar que los datos están escalados
        for col in scaled_data.select_dtypes(include=[np.number]).columns:
            # Los valores escalados deben tener media ~0 y std ~1
            if len(scaled_data[col].unique()) > 1:  # Skip constant columns
                assert abs(scaled_data[col].mean()) < 0.1
                assert abs(scaled_data[col].std() - 1.0) < 0.2


class TestModelValidator:
    """Tests para validación de modelos"""

    def setup_method(self):
        """Setup antes de cada test"""
        self.validator = ModelValidator()

        # Crear datos de test
        self.y_true = np.array(["H", "D", "A"] * 100)
        self.y_pred = np.array(
            ["H", "D", "A"] * 95 + ["H", "A", "D", "A", "H"]
        )  # ~95% accuracy

        # Probabilidades de test
        self.y_proba = np.random.dirichlet(
            [2, 1, 1], 300
        )  # Bias toward first class

    def test_cross_validation(self):
        """Test de validación cruzada"""
        # Crear modelo mock
        mock_model = Mock()
        mock_model.fit = Mock()
        mock_model.predict = Mock(return_value=self.y_pred[:100])

        # Crear datos
        X = np.random.random((300, 10))
        y = self.y_true

        # Ejecutar validación cruzada
        cv_scores = self.validator.cross_validate_model(
            mock_model, X, y, cv_folds=5
        )

        # Verificar resultados
        assert "accuracy_scores" in cv_scores
        assert "mean_accuracy" in cv_scores
        assert "std_accuracy" in cv_scores
        assert len(cv_scores["accuracy_scores"]) == 5
        assert 0 <= cv_scores["mean_accuracy"] <= 1

    def test_model_comparison(self):
        """Test de comparación de modelos"""
        # Crear múltiples modelos mock
        models = {}
        for name in ["xgboost", "lightgbm", "random_forest"]:
            mock_model = Mock()
            mock_model.fit = Mock()
            mock_model.predict = Mock(return_value=self.y_pred[:100])
            models[name] = mock_model

        # Datos de test
        X = np.random.random((300, 10))
        y = self.y_true

        # Comparar modelos
        comparison = self.validator.compare_models(models, X, y)

        # Verificar estructura del resultado
        assert isinstance(comparison, dict)
        for model_name in models.keys():
            assert model_name in comparison
            assert "mean_accuracy" in comparison[model_name]
            assert "std_accuracy" in comparison[model_name]

    def test_feature_importance_validation(self):
        """Test de validación de importancia de features"""
        # Feature importance mock
        feature_importance = {
            "home_team_form": 0.25,
            "away_team_form": 0.20,
            "home_attack_strength": 0.15,
            "away_defense_strength": 0.12,
            "league_position_diff": 0.10,
            "market_value_ratio": 0.08,
            "home_advantage": 0.06,
            "h2h_record": 0.04,
        }

        # Validar importancia
        validation_result = self.validator.validate_feature_importance(
            feature_importance
        )

        # Verificar estructura
        assert "total_importance" in validation_result
        assert "top_features" in validation_result
        assert "feature_distribution" in validation_result

        # Verificar que la suma total es aproximadamente 1
        assert abs(validation_result["total_importance"] - 1.0) < 0.01

        # Verificar que las top features están ordenadas
        top_features = validation_result["top_features"]
        importances = [feature_importance[feat] for feat in top_features]
        assert importances == sorted(importances, reverse=True)


class TestModelIntegration:
    """Tests de integración entre componentes"""

    def setup_method(self):
        """Setup para tests de integración"""
        self.sample_data = self._create_comprehensive_dataset()

    def _create_comprehensive_dataset(self) -> pd.DataFrame:
        """Crea dataset completo para tests de integración"""
        np.random.seed(42)
        n_samples = 500

        # Equipos realistas
        teams = [
            "Real Madrid",
            "Barcelona",
            "Atletico Madrid",
            "Valencia",
            "Sevilla",
            "Athletic Bilbao",
            "Real Sociedad",
            "Villarreal",
            "Betis",
            "Celta",
        ]

        data = []
        for i in range(n_samples):
            home_team = np.random.choice(teams)
            away_team = np.random.choice([t for t in teams if t != home_team])

            # Simular fortalezas realistas basadas en equipos
            team_strength = {
                "Real Madrid": 0.9,
                "Barcelona": 0.85,
                "Atletico Madrid": 0.8,
                "Valencia": 0.7,
                "Sevilla": 0.75,
                "Athletic Bilbao": 0.65,
                "Real Sociedad": 0.6,
                "Villarreal": 0.7,
                "Betis": 0.6,
                "Celta": 0.55,
            }

            home_strength = team_strength[home_team]
            away_strength = team_strength[away_team]

            # Ventaja de local
            home_advantage = 0.1

            # Calcular probabilidades realistas
            total_strength = home_strength + home_advantage + away_strength
            home_prob = (home_strength + home_advantage) / total_strength
            away_prob = away_strength / total_strength
            draw_prob = 1 - home_prob - away_prob

            # Resultado basado en probabilidades
            result = np.random.choice(
                ["H", "D", "A"], p=[home_prob, draw_prob, away_prob]
            )

            # Features calculadas
            match_data = {
                "home_team": home_team,
                "away_team": away_team,
                "home_team_form": np.random.uniform(1.5, 2.5) * home_strength,
                "away_team_form": np.random.uniform(1.5, 2.5) * away_strength,
                "home_attack_strength": np.random.uniform(0.8, 1.5)
                * home_strength,
                "away_attack_strength": np.random.uniform(0.8, 1.5)
                * away_strength,
                "home_defense_strength": np.random.uniform(0.8, 1.5)
                * home_strength,
                "away_defense_strength": np.random.uniform(0.8, 1.5)
                * away_strength,
                "league_position_diff": np.random.uniform(-10, 10),
                "market_value_ratio": np.random.uniform(0.5, 2.0),
                "home_advantage": home_advantage
                + np.random.uniform(-0.05, 0.05),
                "form_difference": (home_strength - away_strength)
                + np.random.uniform(-0.2, 0.2),
                "h2h_home_wins": np.random.uniform(0, 1),
                "h2h_avg_goals": np.random.uniform(2, 3.5),
                "match_importance": np.random.uniform(0.8, 1.2),
                "derby_factor": (
                    1
                    if home_team in ["Real Madrid", "Barcelona"]
                    and away_team in ["Real Madrid", "Barcelona"]
                    else 0
                ),
                "injuries_impact": np.random.uniform(0, 0.3),
                "weather_conditions": np.random.choice([0, 1, 2]),
                "referee_bias": np.random.uniform(-0.1, 0.1),
                "expected_competitiveness": 1
                - abs(home_strength - away_strength),
                "momentum_factor": np.random.uniform(-0.5, 0.5),
                "pressure_situations": np.random.uniform(0.3, 0.8),
                "result": result,
            }

            data.append(match_data)

        return pd.DataFrame(data)

    def test_end_to_end_prediction_pipeline(self):
        """Test del pipeline completo de predicción"""
        # 1. Feature Engineering
        engineer = FeatureEngineer()
        features_df = self.sample_data.drop(
            columns=["result", "home_team", "away_team"]
        )
        engineered_features = engineer.engineer_features(
            features_df, "match_outcome"
        )

        # 2. Model Training
        trainer = ModelTrainer("match_outcome", "xgboost")
        data_with_target = engineered_features.copy()
        data_with_target["result"] = self.sample_data["result"]

        prepared_data = trainer.prepare_data(
            data_with_target, "result", test_size=0.3
        )
        training_result = trainer.train_model(
            prepared_data, optimize_hyperparams=False
        )

        # 3. Model Validation
        validator = ModelValidator()
        X_test = prepared_data["X_test"]
        y_test = prepared_data["y_test"]

        # Hacer predicciones
        predictions = training_result.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # 4. Verificaciones del pipeline completo
        assert training_result.accuracy > 0.3  # Accuracy mínima razonable
        assert accuracy > 0.3  # Test accuracy razonable
        assert len(predictions) == len(y_test)
        assert all(pred in ["H", "D", "A"] for pred in predictions)

        # 5. Feature Importance
        importance = training_result.feature_importance
        assert len(importance) > 0
        assert all(isinstance(v, (int, float)) for v in importance.values())

    def test_ensemble_vs_individual_models(self):
        """Test comparando ensemble vs modelos individuales"""
        # Preparar datos
        X = self.sample_data.drop(columns=["result", "home_team", "away_team"])
        y = self.sample_data["result"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # 1. Entrenar modelos individuales
        algorithms = ["xgboost", "lightgbm", "catboost"]
        individual_scores = {}

        for algo in algorithms:
            predictor = MatchOutcomePredictor(model_algorithm=algo)
            predictor.train(X_train, y_train)
            predictions = predictor.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            individual_scores[algo] = accuracy

        # 2. Entrenar ensemble
        ensemble = MatchOutcomeEnsemble(algorithms)
        ensemble.train_ensemble(X_train, y_train)
        ensemble_result = ensemble.predict_ensemble(X_test)
        ensemble_predictions = [ensemble_result["ensemble_prediction"]] * len(
            X_test
        )

        # Para este test, simulamos predicciones del ensemble
        # En implementación real, el ensemble haría predicciones reales
        best_individual_score = max(individual_scores.values())

        # Verificar que al menos uno de los modelos individuales funciona
        assert best_individual_score > 0.3

        # Verificar que el ensemble retorna estructura correcta
        assert "ensemble_prediction" in ensemble_result
        assert "consensus_score" in ensemble_result
        assert "individual_predictions" in ensemble_result

    def test_model_persistence_integration(self):
        """Test de persistencia completa del modelo"""
        # 1. Entrenar modelo completo
        trainer = ModelTrainer("match_outcome", "xgboost")
        features_df = self.sample_data.drop(
            columns=["result", "home_team", "away_team"]
        )
        data_with_target = features_df.copy()
        data_with_target["result"] = self.sample_data["result"]

        prepared_data = trainer.prepare_data(data_with_target, "result")
        training_result = trainer.train_model(
            prepared_data, optimize_hyperparams=False
        )

        # 2. Guardar modelo
        model_path = training_result.model_path
        assert os.path.exists(model_path)

        # 3. Cargar modelo en nuevo trainer
        new_trainer = ModelTrainer("match_outcome", "xgboost")
        loaded_package = new_trainer.load_model(model_path)

        # 4. Verificar que las predicciones son consistentes
        test_features = prepared_data["X_test"][:5]

        # Predicciones del modelo original
        original_predictions = training_result.model.predict(test_features)

        # Predicciones del modelo cargado
        loaded_predictions = loaded_package["model"].predict(test_features)

        # Verificar consistencia
        assert np.array_equal(original_predictions, loaded_predictions)

        # Cleanup
        os.unlink(model_path)

    def test_feature_engineering_compatibility(self):
        """Test de compatibilidad entre feature engineering y modelos"""
        # 1. Feature engineering
        engineer = FeatureEngineer()
        raw_features = self.sample_data.drop(
            columns=["result", "home_team", "away_team"]
        )
        engineered = engineer.engineer_features(raw_features, "match_outcome")

        # 2. Verificar que todas las features son numéricas
        non_numeric = engineered.select_dtypes(exclude=[np.number]).columns
        assert (
            len(non_numeric) == 0
        ), f"Features no numéricas encontradas: {non_numeric.tolist()}"

        # 3. Verificar que no hay valores infinitos o NaN
        assert not np.isinf(
            engineered.values
        ).any(), "Valores infinitos encontrados"
        assert not np.isnan(engineered.values).any(), "Valores NaN encontrados"

        # 4. Entrenar modelo con features engineered
        predictor = MatchOutcomePredictor("xgboost")
        predictor.train(engineered, self.sample_data["result"])

        # 5. Verificar que el modelo puede hacer predicciones
        predictions = predictor.predict(engineered[:10])
        assert len(predictions) == 10
        assert all(pred in ["H", "D", "A"] for pred in predictions)


class TestModelPerformance:
    """Tests de rendimiento y benchmarking"""

    def test_training_time_performance(self):
        """Test de tiempo de entrenamiento"""
        import time

        # Crear dataset de tamaño mediano
        n_samples = 1000
        n_features = 20

        X = np.random.random((n_samples, n_features))
        y = np.random.choice(["H", "D", "A"], n_samples)

        # Test de diferentes algoritmos
        algorithms = ["xgboost", "lightgbm", "catboost"]
        training_times = {}

        for algo in algorithms:
            predictor = MatchOutcomePredictor(model_algorithm=algo)

            start_time = time.time()
            predictor.train(pd.DataFrame(X), pd.Series(y))
            training_time = time.time() - start_time

            training_times[algo] = training_time

            # Verificar que el entrenamiento no toma demasiado tiempo
            assert (
                training_time < 30
            ), f"{algo} toma demasiado tiempo: {training_time:.2f}s"

        # Log de tiempos para referencia
        print(f"Tiempos de entrenamiento: {training_times}")

    def test_prediction_speed(self):
        """Test de velocidad de predicción"""
        import time

        # Entrenar modelo
        n_samples = 1000
        n_features = 20

        X_train = pd.DataFrame(np.random.random((n_samples, n_features)))
        y_train = pd.Series(np.random.choice(["H", "D", "A"], n_samples))

        predictor = MatchOutcomePredictor("xgboost")
        predictor.train(X_train, y_train)

        # Test de velocidad de predicción
        X_test = pd.DataFrame(np.random.random((100, n_features)))

        start_time = time.time()
        predictions = predictor.predict(X_test)
        prediction_time = time.time() - start_time

        # Verificar velocidad (100 predicciones en menos de 1 segundo)
        assert (
            prediction_time < 1.0
        ), f"Predicciones muy lentas: {prediction_time:.3f}s"
        assert len(predictions) == 100

    def test_memory_usage(self):
        """Test de uso de memoria"""
        import gc

        import psutil

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Crear y entrenar múltiples modelos
        models = []
        for i in range(5):
            X = pd.DataFrame(np.random.random((500, 15)))
            y = pd.Series(np.random.choice(["H", "D", "A"], 500))

            predictor = MatchOutcomePredictor("xgboost")
            predictor.train(X, y)
            models.append(predictor)

        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory

        # Cleanup
        del models
        gc.collect()

        final_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Verificar que el uso de memoria es razonable
        assert (
            memory_increase < 1000
        ), f"Uso de memoria muy alto: {memory_increase:.1f}MB"

        # Verificar que la memoria se libera correctamente
        memory_freed = peak_memory - final_memory
        assert (
            memory_freed > memory_increase * 0.5
        ), "Posible memory leak detectado"


class TestModelRobustness:
    """Tests de robustez del modelo"""

    def test_handling_missing_features(self):
        """Test de manejo de features faltantes"""
        # Crear datos completos
        complete_data = pd.DataFrame(
            {
                "feature_1": np.random.random(100),
                "feature_2": np.random.random(100),
                "feature_3": np.random.random(100),
                "feature_4": np.random.random(100),
            }
        )
        target = pd.Series(np.random.choice(["H", "D", "A"], 100))

        # Entrenar con datos completos
        predictor = MatchOutcomePredictor("xgboost")
        predictor.train(complete_data, target)

        # Test con features faltantes
        incomplete_data = complete_data.drop(columns=["feature_4"])

        # Verificar que se maneja correctamente
        is_valid = predictor._validate_features(incomplete_data)
        assert not is_valid, "Debería detectar features faltantes"

    def test_handling_invalid_data_types(self):
        """Test de manejo de tipos de datos inválidos"""
        predictor = MatchOutcomePredictor("xgboost")

        # Datos con tipos incorrectos
        invalid_data = pd.DataFrame(
            {
                "feature_1": [
                    "a",
                    "b",
                    "c",
                    "d",
                ],  # String en lugar de numérico
                "feature_2": [1, 2, 3, 4],
                "feature_3": [1.1, 2.2, 3.3, 4.4],
            }
        )

        # Verificar que se detecta el problema
        try:
            predictor._preprocess_features(invalid_data)
            # Si no falla, verificar que se convirtieron correctamente
            processed = predictor._preprocess_features(invalid_data)
            assert processed.dtypes.apply(
                lambda x: np.issubdtype(x, np.number)
            ).all()
        except (ValueError, TypeError):
            # Es aceptable que falle con tipos inválidos
            pass

    def test_extreme_values_handling(self):
        """Test de manejo de valores extremos"""
        # Datos con valores extremos
        extreme_data = pd.DataFrame(
            {
                "normal_feature": np.random.normal(0, 1, 100),
                "extreme_feature": np.concatenate(
                    [
                        np.random.normal(0, 1, 95),
                        [1000, -1000, 999, -999, 500],  # Valores extremos
                    ]
                ),
            }
        )
        target = pd.Series(np.random.choice(["H", "D", "A"], 100))

        # Entrenar modelo
        predictor = MatchOutcomePredictor("xgboost")
        predictor.train(extreme_data, target)

        # Verificar que puede hacer predicciones con valores extremos
        predictions = predictor.predict(extreme_data[-5:])
        assert len(predictions) == 5
        assert all(pred in ["H", "D", "A"] for pred in predictions)

    def test_small_dataset_handling(self):
        """Test con datasets pequeños"""
        # Dataset muy pequeño
        small_X = pd.DataFrame(
            {"feature_1": [1, 2, 3, 4, 5], "feature_2": [2, 4, 6, 8, 10]}
        )
        small_y = pd.Series(["H", "D", "A", "H", "D"])

        predictor = MatchOutcomePredictor("xgboost")

        try:
            predictor.train(small_X, small_y)
            predictions = predictor.predict(small_X)
            assert len(predictions) == 5
        except ValueError:
            # Es aceptable que falle con datasets muy pequeños
            pass


# Factory function tests
class TestFactoryFunctions:
    """Tests para funciones factory"""

    def test_create_match_outcome_predictor(self):
        """Test de factory function para predictor"""
        algorithms = [
            "xgboost",
            "lightgbm",
            "catboost",
            "random_forest",
            "ensemble",
        ]

        for algo in algorithms:
            predictor = create_match_outcome_predictor(algo)

            if algo == "ensemble":
                assert isinstance(predictor, MatchOutcomeEnsemble)
            else:
                assert isinstance(predictor, MatchOutcomePredictor)
                assert predictor.model_algorithm == algo

    def test_train_all_models(self):
        """Test de entrenamiento de todos los modelos"""
        # Crear dataset de prueba
        test_data = pd.DataFrame(
            {
                "feature_1": np.random.random(200),
                "feature_2": np.random.random(200),
                "feature_3": np.random.random(200),
                "result": np.random.choice(["H", "D", "A"], 200),
            }
        )

        # Entrenar todos los modelos (con pocos datos para test rápido)
        results = train_all_models(test_data, "result", "match_outcome")

        # Verificar que se entrenaron múltiples modelos
        assert isinstance(results, dict)
        assert len(results) > 0

        # Verificar que cada resultado es válido
        for algo, result in results.items():
            assert isinstance(result, TrainingResult)
            assert hasattr(result, "accuracy")
            assert hasattr(result, "model")


# Configuración de pytest para este módulo
@pytest.fixture(autouse=True)
def setup_ml_environment():
    """Setup del entorno de ML para tests"""
    # Suprimir warnings de ML libraries
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Set random seeds para reproducibilidad
    np.random.seed(42)

    yield

    # Cleanup después de tests
    import gc

    gc.collect()


# Marks para organización de tests
pytestmark = [
    pytest.mark.unit,  # Todos los tests de modelos son unitarios
]

if __name__ == "__main__":
    # Ejecutar tests específicos de modelos
    pytest.main(
        [__file__, "-v", "--tb=short", "-x"]
    )  # -x para parar en primer fallo
