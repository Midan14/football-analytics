"""
Football Analytics - Model Trainer
Entrenador centralizado para todos los modelos de ML del sistema
"""

import json
import logging
import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import lightgbm as lgb
import numpy as np

# Hyperparameter optimization
import optuna
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Model validation and metrics
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ML Libraries
from sklearn.model_selection import (
    GridSearchCV,
    TimeSeriesSplit,
    cross_val_score,
    learning_curve,
    train_test_split,
    validation_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler

# Custom imports
from ..utils.config import MODEL_CONFIG, TRAINING_CONFIG
from ..utils.logger import get_logger
from .feature_engineer import FeatureEngineer
from .model_validator import ModelValidator

logger = get_logger(__name__)


@dataclass
class TrainingResult:
    """Resultado del entrenamiento de un modelo"""

    model: Any
    accuracy: float
    cross_val_score: float
    feature_importance: Dict[str, float]
    training_time: float
    hyperparameters: Dict[str, Any]
    calibration_score: float
    model_path: str


class ModelTrainer:
    """
    Entrenador centralizado para todos los modelos de ML
    Maneja entrenamiento, validación, optimización y persistencia
    """

    def __init__(
        self, model_type: str = "match_outcome", algorithm: str = "xgboost"
    ):
        self.model_type = model_type
        self.algorithm = algorithm
        self.feature_engineer = FeatureEngineer()
        self.validator = ModelValidator()

        # Configuración de paths
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)

        # Configuración de logging
        self.logger = get_logger(f"ModelTrainer_{model_type}_{algorithm}")

        # Inicializar scalers
        self.scaler = None
        self.label_encoder = None

        # Métricas de entrenamiento
        self.training_history = []

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        validation_split: bool = True,
    ) -> Dict[str, Any]:
        """
        Prepara los datos para entrenamiento

        Args:
            df: DataFrame con los datos
            target_column: Nombre de la columna objetivo
            test_size: Proporción para test set
            validation_split: Si crear validation set adicional

        Returns:
            Dictionary con datos preparados
        """
        self.logger.info(f"Preparando datos para {self.model_type}")

        # Separar features y target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Aplicar feature engineering
        X_engineered = self.feature_engineer.engineer_features(
            X, self.model_type
        )

        # Encoding para variables categóricas si es necesario
        if self.model_type == "match_outcome":
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                y_encoded = self.label_encoder.fit_transform(y)
            else:
                y_encoded = self.label_encoder.transform(y)
        else:
            y_encoded = y

        # Split inicial train/test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_engineered,
            y_encoded,
            test_size=test_size,
            random_state=42,
            stratify=y_encoded if self.model_type == "match_outcome" else None,
        )

        if validation_split:
            # Split adicional train/validation
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp,
                y_temp,
                test_size=0.25,  # 0.25 de 0.8 = 0.2 total
                random_state=42,
                stratify=(
                    y_temp if self.model_type == "match_outcome" else None
                ),
            )
        else:
            X_train, X_val, y_train, y_val = X_temp, None, y_temp, None

        # Scaling para algoritmos que lo requieren
        if self.algorithm in ["logistic", "svm", "neural_network"]:
            if self.scaler is None:
                self.scaler = RobustScaler()
                X_train_scaled = pd.DataFrame(
                    self.scaler.fit_transform(X_train),
                    columns=X_train.columns,
                    index=X_train.index,
                )
            else:
                X_train_scaled = pd.DataFrame(
                    self.scaler.transform(X_train),
                    columns=X_train.columns,
                    index=X_train.index,
                )

            if X_val is not None:
                X_val_scaled = pd.DataFrame(
                    self.scaler.transform(X_val),
                    columns=X_val.columns,
                    index=X_val.index,
                )
            else:
                X_val_scaled = None

            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index,
            )
        else:
            X_train_scaled = X_train
            X_val_scaled = X_val
            X_test_scaled = X_test

        data_dict = {
            "X_train": X_train_scaled,
            "X_val": X_val_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "feature_names": list(X_train.columns),
            "target_classes": (
                self.label_encoder.classes_ if self.label_encoder else None
            ),
        }

        self.logger.info(
            f"Datos preparados: Train={len(X_train)}, Val={len(X_val) if X_val is not None else 0}, Test={len(X_test)}"
        )

        return data_dict

    def get_model(self, hyperparameters: Optional[Dict] = None) -> Any:
        """
        Obtiene el modelo según el algoritmo especificado

        Args:
            hyperparameters: Hiperparámetros específicos

        Returns:
            Modelo inicializado
        """
        if hyperparameters:
            params = hyperparameters
        else:
            params = self._get_default_hyperparameters()

        if self.algorithm == "xgboost":
            if self.model_type == "match_outcome":
                return xgb.XGBClassifier(**params)
            else:
                return xgb.XGBRegressor(**params)

        elif self.algorithm == "lightgbm":
            if self.model_type == "match_outcome":
                return lgb.LGBMClassifier(**params)
            else:
                return lgb.LGBMRegressor(**params)

        elif self.algorithm == "catboost":
            if self.model_type == "match_outcome":
                return CatBoostClassifier(**params, verbose=False)
            else:
                return CatBoostRegressor(**params, verbose=False)

        elif self.algorithm == "random_forest":
            if self.model_type == "match_outcome":
                return RandomForestClassifier(**params)
            else:
                return RandomForestRegressor(**params)

        elif self.algorithm == "logistic":
            return LogisticRegression(**params)

        else:
            raise ValueError(f"Algoritmo no soportado: {self.algorithm}")

    def _get_default_hyperparameters(self) -> Dict[str, Any]:
        """Obtiene hiperparámetros por defecto según el algoritmo"""

        defaults = {
            "xgboost": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            },
            "lightgbm": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "verbose": -1,
            },
            "catboost": {
                "iterations": 100,
                "depth": 6,
                "learning_rate": 0.1,
                "random_seed": 42,
            },
            "random_forest": {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42,
            },
            "logistic": {"random_state": 42, "max_iter": 1000},
        }

        return defaults.get(self.algorithm, {})

    def optimize_hyperparameters(
        self, data: Dict[str, Any], n_trials: int = 50
    ) -> Dict[str, Any]:
        """
        Optimiza hiperparámetros usando Optuna

        Args:
            data: Datos preparados
            n_trials: Número de trials para optimización

        Returns:
            Mejores hiperparámetros encontrados
        """
        self.logger.info(
            f"Iniciando optimización de hiperparámetros para {self.algorithm}"
        )

        def objective(trial):
            # Definir espacio de búsqueda según algoritmo
            params = self._suggest_hyperparameters(trial)

            # Crear modelo con parámetros sugeridos
            model = self.get_model(params)

            # Cross-validation
            if self.model_type == "match_outcome":
                cv_scores = cross_val_score(
                    model,
                    data["X_train"],
                    data["y_train"],
                    cv=5,
                    scoring="accuracy",
                    n_jobs=-1,
                )
            else:
                cv_scores = cross_val_score(
                    model,
                    data["X_train"],
                    data["y_train"],
                    cv=5,
                    scoring="r2",
                    n_jobs=-1,
                )

            return cv_scores.mean()

        # Configurar estudio Optuna
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )

        # Ejecutar optimización
        study.optimize(
            objective, n_trials=n_trials, timeout=3600
        )  # 1 hora max

        best_params = study.best_params
        self.logger.info(f"Mejores parámetros encontrados: {best_params}")
        self.logger.info(f"Mejor score: {study.best_value:.4f}")

        return best_params

    def _suggest_hyperparameters(self, trial) -> Dict[str, Any]:
        """Sugiere hiperparámetros para optimización según algoritmo"""

        if self.algorithm == "xgboost":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.6, 1.0
                ),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
                "random_state": 42,
            }

        elif self.algorithm == "lightgbm":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.6, 1.0
                ),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
                "random_state": 42,
                "verbose": -1,
            }

        elif self.algorithm == "catboost":
            return {
                "iterations": trial.suggest_int("iterations", 50, 300),
                "depth": trial.suggest_int("depth", 3, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3
                ),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
                "random_seed": 42,
            }

        elif self.algorithm == "random_forest":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 5, 20),
                "min_samples_split": trial.suggest_int(
                    "min_samples_split", 2, 20
                ),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf", 1, 10
                ),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", "log2", None]
                ),
                "random_state": 42,
            }

        else:
            return {}

    def train_model(
        self,
        data: Dict[str, Any],
        optimize_hyperparams: bool = True,
        n_trials: int = 50,
    ) -> TrainingResult:
        """
        Entrena el modelo con los datos proporcionados

        Args:
            data: Datos preparados
            optimize_hyperparams: Si optimizar hiperparámetros
            n_trials: Número de trials para optimización

        Returns:
            Resultado del entrenamiento
        """
        start_time = datetime.now()
        self.logger.info(
            f"Iniciando entrenamiento {self.model_type} con {self.algorithm}"
        )

        # Optimizar hiperparámetros si se solicita
        if optimize_hyperparams:
            best_params = self.optimize_hyperparameters(data, n_trials)
        else:
            best_params = self._get_default_hyperparameters()

        # Crear y entrenar modelo final
        model = self.get_model(best_params)
        model.fit(data["X_train"], data["y_train"])

        # Evaluación
        if data["X_val"] is not None:
            val_predictions = model.predict(data["X_val"])
            if self.model_type == "match_outcome":
                accuracy = accuracy_score(data["y_val"], val_predictions)
            else:
                from sklearn.metrics import r2_score

                accuracy = r2_score(data["y_val"], val_predictions)
        else:
            test_predictions = model.predict(data["X_test"])
            if self.model_type == "match_outcome":
                accuracy = accuracy_score(data["y_test"], test_predictions)
            else:
                from sklearn.metrics import r2_score

                accuracy = r2_score(data["y_test"], test_predictions)

        # Cross-validation score
        cv_scores = cross_val_score(
            model,
            data["X_train"],
            data["y_train"],
            cv=5,
            scoring="accuracy" if self.model_type == "match_outcome" else "r2",
        )
        cv_score = cv_scores.mean()

        # Feature importance
        feature_importance = self._get_feature_importance(
            model, data["feature_names"]
        )

        # Calibración (solo para clasificación)
        calibration_score = 0.0
        if self.model_type == "match_outcome":
            calibrated_model = CalibratedClassifierCV(
                model, method="isotonic", cv=3
            )
            calibrated_model.fit(data["X_train"], data["y_train"])

            if data["X_val"] is not None:
                proba_pred = calibrated_model.predict_proba(data["X_val"])
                calibration_score = self._calculate_calibration_score(
                    data["y_val"], proba_pred
                )

            model = calibrated_model  # Usar modelo calibrado

        # Tiempo de entrenamiento
        training_time = (datetime.now() - start_time).total_seconds()

        # Guardar modelo
        model_path = self.save_model(model, best_params, accuracy)

        # Crear resultado
        result = TrainingResult(
            model=model,
            accuracy=accuracy,
            cross_val_score=cv_score,
            feature_importance=feature_importance,
            training_time=training_time,
            hyperparameters=best_params,
            calibration_score=calibration_score,
            model_path=model_path,
        )

        # Guardar en historial
        self.training_history.append(
            {
                "timestamp": datetime.now(),
                "model_type": self.model_type,
                "algorithm": self.algorithm,
                "accuracy": accuracy,
                "cv_score": cv_score,
                "training_time": training_time,
                "hyperparameters": best_params,
            }
        )

        self.logger.info(
            f"Entrenamiento completado. Accuracy: {accuracy:.4f}, CV Score: {cv_score:.4f}"
        )

        return result

    def _get_feature_importance(
        self, model: Any, feature_names: List[str]
    ) -> Dict[str, float]:
        """Extrae feature importance del modelo"""
        try:
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                importances = (
                    np.abs(model.coef_[0])
                    if len(model.coef_.shape) > 1
                    else np.abs(model.coef_)
                )
            else:
                return {}

            return dict(zip(feature_names, importances))
        except Exception as e:
            self.logger.warning(f"No se pudo extraer feature importance: {e}")
            return {}

    def _calculate_calibration_score(
        self, y_true: np.ndarray, y_proba: np.ndarray
    ) -> float:
        """Calcula score de calibración"""
        try:
            # Usar la clase más probable
            y_prob_max = np.max(y_proba, axis=1)
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_prob_max, n_bins=10
            )

            # Calcular Brier Score como medida de calibración
            brier_score = np.mean((y_prob_max - y_true) ** 2)
            return 1 - brier_score  # Invertir para que mayor sea mejor

        except Exception as e:
            self.logger.warning(f"Error calculando calibración: {e}")
            return 0.0

    def save_model(
        self, model: Any, hyperparameters: Dict, accuracy: float
    ) -> str:
        """
        Guarda el modelo entrenado

        Args:
            model: Modelo entrenado
            hyperparameters: Hiperparámetros usados
            accuracy: Accuracy obtenida

        Returns:
            Ruta donde se guardó el modelo
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.model_type}_{self.algorithm}_{timestamp}.pkl"
        model_path = self.models_dir / filename

        # Crear paquete del modelo
        model_package = {
            "model": model,
            "model_type": self.model_type,
            "algorithm": self.algorithm,
            "hyperparameters": hyperparameters,
            "accuracy": accuracy,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "timestamp": timestamp,
            "version": "1.0",
        }

        # Guardar
        joblib.dump(model_package, model_path)

        # Guardar también metadatos
        metadata = {
            "model_path": str(model_path),
            "model_type": self.model_type,
            "algorithm": self.algorithm,
            "accuracy": accuracy,
            "timestamp": timestamp,
            "hyperparameters": hyperparameters,
        }

        metadata_path = (
            self.models_dir
            / f"{self.model_type}_{self.algorithm}_metadata.json"
        )
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        self.logger.info(f"Modelo guardado en: {model_path}")

        return str(model_path)

    def load_model(self, model_path: str) -> Any:
        """
        Carga un modelo guardado

        Args:
            model_path: Ruta del modelo

        Returns:
            Paquete del modelo cargado
        """
        try:
            model_package = joblib.load(model_path)

            # Restaurar componentes
            self.scaler = model_package.get("scaler")
            self.label_encoder = model_package.get("label_encoder")

            self.logger.info(f"Modelo cargado desde: {model_path}")

            return model_package

        except Exception as e:
            self.logger.error(f"Error cargando modelo: {e}")
            raise

    def evaluate_model(
        self, model: Any, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evalúa el modelo con métricas detalladas

        Args:
            model: Modelo a evaluar
            data: Datos de test

        Returns:
            Métricas de evaluación
        """
        X_test = data["X_test"]
        y_test = data["y_test"]

        # Predicciones
        predictions = model.predict(X_test)

        if self.model_type == "match_outcome":
            # Métricas de clasificación
            accuracy = accuracy_score(y_test, predictions)
            report = classification_report(
                y_test, predictions, output_dict=True
            )
            cm = confusion_matrix(y_test, predictions)

            # Probabilidades si están disponibles
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(X_test)
                # Calcular log loss
                from sklearn.metrics import log_loss

                logloss = log_loss(y_test, probabilities)
            else:
                probabilities = None
                logloss = None

            metrics = {
                "accuracy": accuracy,
                "classification_report": report,
                "confusion_matrix": cm.tolist(),
                "log_loss": logloss,
                "probabilities_sample": (
                    probabilities[:10].tolist()
                    if probabilities is not None
                    else None
                ),
            }

        else:
            # Métricas de regresión
            from sklearn.metrics import (
                mean_absolute_error,
                mean_squared_error,
                r2_score,
            )

            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            rmse = np.sqrt(mse)

            metrics = {
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2_score": r2,
                "predictions_sample": predictions[:10].tolist(),
            }

        return metrics

    def get_training_history(self) -> List[Dict]:
        """Obtiene el historial de entrenamientos"""
        return self.training_history

    def create_ensemble(
        self, algorithms: List[str], data: Dict[str, Any]
    ) -> Any:
        """
        Crea un ensemble de múltiples algoritmos

        Args:
            algorithms: Lista de algoritmos a incluir
            data: Datos de entrenamiento

        Returns:
            Modelo ensemble entrenado
        """
        self.logger.info(f"Creando ensemble con algoritmos: {algorithms}")

        # Crear modelos individuales
        models = []
        for algo in algorithms:
            trainer = ModelTrainer(self.model_type, algo)
            result = trainer.train_model(
                data, optimize_hyperparams=True, n_trials=30
            )
            models.append((algo, result.model))

        # Crear voting classifier/regressor
        if self.model_type == "match_outcome":
            ensemble = VotingClassifier(
                estimators=models, voting="soft"  # Usar probabilidades
            )
        else:
            from sklearn.ensemble import VotingRegressor

            ensemble = VotingRegressor(estimators=models)

        # Entrenar ensemble
        ensemble.fit(data["X_train"], data["y_train"])

        self.logger.info("Ensemble creado y entrenado exitosamente")

        return ensemble


def create_model_trainer(model_type: str, algorithm: str) -> ModelTrainer:
    """
    Factory function para crear trainer

    Args:
        model_type: Tipo de modelo (match_outcome, goals, etc.)
        algorithm: Algoritmo a usar

    Returns:
        ModelTrainer configurado
    """
    return ModelTrainer(model_type, algorithm)


def train_all_models(
    df: pd.DataFrame, target_column: str, model_type: str = "match_outcome"
) -> Dict[str, TrainingResult]:
    """
    Entrena todos los algoritmos disponibles y retorna resultados

    Args:
        df: DataFrame con datos
        target_column: Columna objetivo
        model_type: Tipo de modelo

    Returns:
        Resultados de todos los entrenamientos
    """
    algorithms = ["xgboost", "lightgbm", "catboost", "random_forest"]
    results = {}

    for algo in algorithms:
        try:
            trainer = ModelTrainer(model_type, algo)
            data = trainer.prepare_data(df, target_column)
            result = trainer.train_model(data, optimize_hyperparams=True)
            results[algo] = result

            print(
                f"✅ {algo.upper()}: Accuracy={result.accuracy:.4f}, CV={result.cross_val_score:.4f}"
            )

        except Exception as e:
            print(f"❌ Error entrenando {algo}: {e}")
            continue

    return results


if __name__ == "__main__":
    # Ejemplo de uso
    print("🚀 Football Analytics - Model Trainer")
    print("Entrenador de modelos ML para predicciones deportivas")

    # Ejemplo básico
    # df = pd.read_csv("data/matches.csv")
    # results = train_all_models(df, "result", "match_outcome")
    #
    # # Mostrar mejores resultados
    # best_algo = max(results.keys(), key=lambda x: results[x].accuracy)
    # print(f"\n🏆 Mejor algoritmo: {best_algo.upper()}")
    # print(f"📊 Accuracy: {results[best_algo].accuracy:.4f}")
