"""
Base Model Class for Football Analytics ML Models

Esta clase base proporciona funcionalidades comunes para todos los modelos
de Machine Learning, incluyendo entrenamiento, evaluación, persistencia y explicabilidad.
"""

import logging
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# ML Libraries
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Explicabilidad
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")

try:
    import lime
    from lime.lime_tabular import LimeTabularExplainer

    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    warnings.warn("LIME not available. Install with: pip install lime")


# Configuración
from app.ml_models import MLConfig

# Configurar logging
logger = logging.getLogger(__name__)

# =====================================================
# CLASE BASE ABSTRACTA
# =====================================================


class BaseFootballModel(ABC):
    """
    Clase base abstracta para todos los modelos de fútbol.

    Proporciona funcionalidades comunes como:
    - Entrenamiento y evaluación
    - Persistencia de modelos
    - Explicabilidad (SHAP/LIME)
    - Validación cruzada
    - Optimización de hiperparámetros
    """

    def __init__(
        self,
        model_name: str,
        model_type: str = "classifier",
        random_state: int = MLConfig.RANDOM_STATE,
    ):
        """
        Inicializar modelo base.

        Args:
            model_name: Nombre único del modelo
            model_type: Tipo de modelo ('classifier' o 'regressor')
            random_state: Semilla para reproducibilidad
        """
        self.model_name = model_name
        self.model_type = model_type
        self.random_state = random_state

        # Estado del modelo
        self.is_trained = False
        self.model = None
        self.preprocessor = None
        self.feature_names = []
        self.target_names = []

        # Métricas y metadatos
        self.training_metrics = {}
        self.validation_metrics = {}
        self.feature_importance = {}
        self.training_time = None
        self.model_version = MLConfig.MODEL_VERSION

        # Configuración
        self.test_size = MLConfig.TEST_SIZE
        self.cv_folds = MLConfig.CV_FOLDS

        # Explicabilidad
        self.shap_explainer = None
        self.lime_explainer = None

        logger.info(f"Initialized {self.model_type} model: {self.model_name}")

    # =====================================================
    # MÉTODOS ABSTRACTOS (IMPLEMENTAR EN SUBCLASES)
    # =====================================================

    @abstractmethod
    def _create_model(self) -> BaseEstimator:
        """
        Crear instancia del modelo específico.

        Returns:
            BaseEstimator: Modelo de sklearn/XGBoost/etc.
        """
        pass

    @abstractmethod
    def _get_default_hyperparameters(self) -> Dict[str, Any]:
        """
        Obtener hiperparámetros por defecto.

        Returns:
            dict: Hiperparámetros por defecto
        """
        pass

    @abstractmethod
    def _get_hyperparameter_grid(self) -> Dict[str, List]:
        """
        Obtener grid de hiperparámetros para búsqueda.

        Returns:
            dict: Grid de hiperparámetros
        """
        pass

    # =====================================================
    # PREPROCESAMIENTO DE DATOS
    # =====================================================

    def create_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Crear pipeline de preprocesamiento.

        Args:
            X: DataFrame con features

        Returns:
            ColumnTransformer: Pipeline de preprocesamiento
        """
        # Identificar tipos de columnas
        numeric_features = X.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        categorical_features = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        # Pipeline para features numéricas
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        # Pipeline para features categóricas
        categorical_transformer = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="constant", fill_value="unknown"),
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore", sparse_output=False
                    ),
                ),
            ]
        )

        # Combinar transformadores
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ],
            remainder="passthrough",
        )

        logger.info(
            f"Created preprocessor with {len(numeric_features)} numeric and {len(categorical_features)} categorical features"
        )
        return preprocessor

    def prepare_data(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preparar datos para entrenamiento/predicción.

        Args:
            X: Features
            y: Target (opcional para predicción)

        Returns:
            Tuple: (X_processed, y_processed)
        """
        # Crear preprocessor si no existe
        if self.preprocessor is None:
            self.preprocessor = self.create_preprocessor(X)

        # Ajustar preprocessor solo durante entrenamiento
        if not self.is_trained and y is not None:
            X_processed = self.preprocessor.fit_transform(X)
            self.feature_names = self._get_feature_names_after_preprocessing(X)
        else:
            X_processed = self.preprocessor.transform(X)

        # Procesar target si existe
        y_processed = None
        if y is not None:
            if self.model_type == "classifier":
                # Para clasificación, asegurar que las clases están en formato correcto
                y_processed = np.array(y)
                self.target_names = list(np.unique(y_processed))
            else:
                # Para regresión
                y_processed = np.array(y)

        logger.debug(
            f"Data prepared: X shape {X_processed.shape}, y shape {y_processed.shape if y_processed is not None else 'None'}"
        )
        return X_processed, y_processed

    def _get_feature_names_after_preprocessing(
        self, X_original: pd.DataFrame
    ) -> List[str]:
        """Obtener nombres de features después del preprocesamiento."""
        feature_names = []

        # Features numéricas (mantienen el nombre original)
        numeric_features = X_original.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        feature_names.extend(numeric_features)

        # Features categóricas (OneHot genera múltiples columnas)
        categorical_features = X_original.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        for cat_feature in categorical_features:
            # Obtener categorías únicas
            categories = X_original[cat_feature].unique()
            for category in categories:
                if pd.notna(category):  # Ignorar NaN
                    feature_names.append(f"{cat_feature}_{category}")

        return feature_names

    # =====================================================
    # ENTRENAMIENTO DEL MODELO
    # =====================================================

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: bool = True,
        optimize_hyperparameters: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Entrenar el modelo.

        Args:
            X: Features de entrenamiento
            y: Target de entrenamiento
            validation_split: Si dividir en train/validation
            optimize_hyperparameters: Si optimizar hiperparámetros
            **kwargs: Argumentos adicionales

        Returns:
            dict: Métricas de entrenamiento
        """
        start_time = datetime.now()
        logger.info(f"Starting training for {self.model_name}")

        try:
            # Preparar datos
            X_processed, y_processed = self.prepare_data(X, y)

            # Dividir en train/validation si se solicita
            if validation_split:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_processed,
                    y_processed,
                    test_size=self.test_size,
                    random_state=self.random_state,
                    stratify=(
                        y_processed
                        if self.model_type == "classifier"
                        else None
                    ),
                )
            else:
                X_train, y_train = X_processed, y_processed
                X_val, y_val = None, None

            # Crear modelo
            if self.model is None:
                self.model = self._create_model()

            # Optimizar hiperparámetros si se solicita
            if optimize_hyperparameters:
                logger.info("Optimizing hyperparameters...")
                self.model = self._optimize_hyperparameters(X_train, y_train)

            # Entrenar modelo
            logger.info("Training model...")
            self.model.fit(X_train, y_train)

            # Evaluar en training
            train_predictions = self.model.predict(X_train)
            self.training_metrics = self._calculate_metrics(
                y_train, train_predictions
            )

            # Evaluar en validation si existe
            if X_val is not None:
                val_predictions = self.model.predict(X_val)
                self.validation_metrics = self._calculate_metrics(
                    y_val, val_predictions
                )

            # Calcular importancia de features
            self._calculate_feature_importance()

            # Configurar explicabilidad
            self._setup_explainability(X_train, y_train)

            # Actualizar estado
            self.is_trained = True
            self.training_time = datetime.now() - start_time

            # Preparar resultado
            result = {
                "success": True,
                "training_metrics": self.training_metrics,
                "validation_metrics": self.validation_metrics,
                "training_time_seconds": self.training_time.total_seconds(),
                "feature_count": len(self.feature_names),
                "sample_count": len(X),
            }

            logger.info(
                f"Training completed in {self.training_time.total_seconds():.2f} seconds"
            )
            return result

        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

    def _optimize_hyperparameters(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> BaseEstimator:
        """Optimizar hiperparámetros usando GridSearch."""
        param_grid = self._get_hyperparameter_grid()

        # Usar GridSearchCV o RandomizedSearchCV dependiendo del tamaño
        if (
            len(param_grid) > 50
        ):  # Si hay muchas combinaciones, usar RandomizedSearch
            search = RandomizedSearchCV(
                self.model,
                param_grid,
                n_iter=20,
                cv=self.cv_folds,
                scoring=(
                    "accuracy"
                    if self.model_type == "classifier"
                    else "neg_mean_squared_error"
                ),
                random_state=self.random_state,
                n_jobs=-1,
            )
        else:
            search = GridSearchCV(
                self.model,
                param_grid,
                cv=self.cv_folds,
                scoring=(
                    "accuracy"
                    if self.model_type == "classifier"
                    else "neg_mean_squared_error"
                ),
                n_jobs=-1,
            )

        search.fit(X_train, y_train)

        logger.info(f"Best hyperparameters: {search.best_params_}")
        logger.info(f"Best cross-validation score: {search.best_score_:.4f}")

        return search.best_estimator_

    # =====================================================
    # EVALUACIÓN DEL MODELO
    # =====================================================

    def _calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calcular métricas de evaluación."""
        metrics = {}

        if self.model_type == "classifier":
            # Métricas de clasificación
            metrics["accuracy"] = accuracy_score(y_true, y_pred)
            metrics["precision"] = precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            )
            metrics["recall"] = recall_score(
                y_true, y_pred, average="weighted", zero_division=0
            )
            metrics["f1_score"] = f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            )

            # ROC AUC para clasificación binaria
            if len(np.unique(y_true)) == 2:
                try:
                    y_proba = self.model.predict_proba(
                        self.preprocessor.transform(pd.DataFrame())
                    )
                    if y_proba.shape[1] == 2:
                        metrics["roc_auc"] = roc_auc_score(
                            y_true, y_proba[:, 1]
                        )
                except:
                    pass  # No es crítico si falla

        else:
            # Métricas de regresión
            metrics["mse"] = mean_squared_error(y_true, y_pred)
            metrics["rmse"] = np.sqrt(metrics["mse"])
            metrics["mae"] = mean_absolute_error(y_true, y_pred)
            metrics["r2"] = r2_score(y_true, y_pred)

        return metrics

    def cross_validate(
        self, X: pd.DataFrame, y: pd.Series, cv: int = None
    ) -> Dict[str, Any]:
        """
        Realizar validación cruzada.

        Args:
            X: Features
            y: Target
            cv: Número de folds (default: self.cv_folds)

        Returns:
            dict: Resultados de validación cruzada
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before cross-validation")

        cv_folds = cv or self.cv_folds
        X_processed, y_processed = self.prepare_data(X, y)

        # Métrica para scoring
        scoring = (
            "accuracy"
            if self.model_type == "classifier"
            else "neg_mean_squared_error"
        )

        # Realizar validación cruzada
        cv_scores = cross_val_score(
            self.model,
            X_processed,
            y_processed,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
        )

        result = {
            "cv_scores": cv_scores.tolist(),
            "mean_score": cv_scores.mean(),
            "std_score": cv_scores.std(),
            "cv_folds": cv_folds,
            "scoring_metric": scoring,
        }

        logger.info(
            f"Cross-validation: {result['mean_score']:.4f} (+/- {result['std_score']*2:.4f})"
        )
        return result

    def _calculate_feature_importance(self):
        """Calcular importancia de features."""
        if not self.is_trained:
            return

        try:
            # Diferentes métodos según el tipo de modelo
            if hasattr(self.model, "feature_importances_"):
                # Tree-based models (RandomForest, XGBoost, etc.)
                importances = self.model.feature_importances_
            elif hasattr(self.model, "coef_"):
                # Linear models
                importances = np.abs(self.model.coef_)
                if importances.ndim > 1:
                    importances = np.mean(importances, axis=0)
            else:
                logger.warning(
                    "Model doesn't support feature importance calculation"
                )
                return

            # Crear diccionario de importancias
            if len(self.feature_names) == len(importances):
                self.feature_importance = dict(
                    zip(self.feature_names, importances)
                )

                # Ordenar por importancia
                self.feature_importance = dict(
                    sorted(
                        self.feature_importance.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                )

                logger.info(
                    f"Feature importance calculated for {len(self.feature_importance)} features"
                )

        except Exception as e:
            logger.warning(f"Error calculating feature importance: {e}")

    # =====================================================
    # PREDICCIONES
    # =====================================================

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Hacer predicciones.

        Args:
            X: Features para predicción

        Returns:
            np.ndarray: Predicciones
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        X_processed, _ = self.prepare_data(X)
        predictions = self.model.predict(X_processed)

        logger.debug(f"Generated {len(predictions)} predictions")
        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predecir probabilidades (solo para clasificadores).

        Args:
            X: Features para predicción

        Returns:
            np.ndarray: Probabilidades de clase
        """
        if self.model_type != "classifier":
            raise ValueError("predict_proba only available for classifiers")

        if not hasattr(self.model, "predict_proba"):
            raise ValueError("Model doesn't support probability prediction")

        X_processed, _ = self.prepare_data(X)
        probabilities = self.model.predict_proba(X_processed)

        logger.debug(
            f"Generated probability predictions for {len(probabilities)} samples"
        )
        return probabilities

    def predict_with_confidence(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Predecir con información de confianza.

        Args:
            X: Features para predicción

        Returns:
            dict: Predicciones con confianza y metadatos
        """
        predictions = self.predict(X)

        result = {
            "predictions": predictions.tolist(),
            "sample_count": len(predictions),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "prediction_timestamp": datetime.now().isoformat(),
        }

        # Agregar probabilidades si es clasificador
        if self.model_type == "classifier" and hasattr(
            self.model, "predict_proba"
        ):
            probabilities = self.predict_proba(X)
            result["probabilities"] = probabilities.tolist()

            # Calcular confianza como máxima probabilidad
            max_probabilities = np.max(probabilities, axis=1)
            result["confidence_scores"] = max_probabilities.tolist()
            result["mean_confidence"] = float(np.mean(max_probabilities))

            # Clasificar predicciones por confianza
            high_conf = np.sum(
                max_probabilities > MLConfig.HIGH_CONFIDENCE_THRESHOLD
            )
            medium_conf = np.sum(
                (max_probabilities > MLConfig.MEDIUM_CONFIDENCE_THRESHOLD)
                & (max_probabilities <= MLConfig.HIGH_CONFIDENCE_THRESHOLD)
            )
            low_conf = len(predictions) - high_conf - medium_conf

            result["confidence_distribution"] = {
                "high_confidence": int(high_conf),
                "medium_confidence": int(medium_conf),
                "low_confidence": int(low_conf),
            }

        return result

    # =====================================================
    # EXPLICABILIDAD
    # =====================================================

    def _setup_explainability(self, X_train: np.ndarray, y_train: np.ndarray):
        """Configurar herramientas de explicabilidad."""
        try:
            # Configurar SHAP
            if SHAP_AVAILABLE:
                if hasattr(self.model, "predict_proba"):
                    # Para clasificadores
                    self.shap_explainer = shap.Explainer(
                        self.model.predict_proba, X_train
                    )
                else:
                    # Para regresores
                    self.shap_explainer = shap.Explainer(
                        self.model.predict, X_train
                    )

                logger.info("SHAP explainer configured")

            # Configurar LIME
            if LIME_AVAILABLE:
                mode = (
                    "classification"
                    if self.model_type == "classifier"
                    else "regression"
                )
                self.lime_explainer = LimeTabularExplainer(
                    X_train,
                    feature_names=self.feature_names,
                    mode=mode,
                    random_state=self.random_state,
                )
                logger.info("LIME explainer configured")

        except Exception as e:
            logger.warning(f"Error setting up explainability: {e}")

    def explain_prediction(
        self, X: pd.DataFrame, method: str = "shap"
    ) -> Dict[str, Any]:
        """
        Explicar predicciones usando SHAP o LIME.

        Args:
            X: Sample para explicar
            method: Método de explicación ('shap' o 'lime')

        Returns:
            dict: Explicación de la predicción
        """
        if method == "shap" and self.shap_explainer is not None:
            return self._explain_with_shap(X)
        elif method == "lime" and self.lime_explainer is not None:
            return self._explain_with_lime(X)
        else:
            return {"error": f"Explainer {method} not available"}

    def _explain_with_shap(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Explicar con SHAP."""
        try:
            X_processed, _ = self.prepare_data(X)
            shap_values = self.shap_explainer(X_processed)

            # Convertir a formato serializable
            explanation = {
                "method": "shap",
                "shap_values": (
                    shap_values.values.tolist()
                    if hasattr(shap_values, "values")
                    else shap_values.tolist()
                ),
                "feature_names": self.feature_names,
                "base_value": (
                    float(shap_values.base_values[0])
                    if hasattr(shap_values, "base_values")
                    else 0.0
                ),
            }

            return explanation

        except Exception as e:
            logger.error(f"Error with SHAP explanation: {e}")
            return {"error": str(e)}

    def _explain_with_lime(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Explicar con LIME."""
        try:
            X_processed, _ = self.prepare_data(X)

            # LIME explica una muestra a la vez
            instance = X_processed[0]

            if self.model_type == "classifier":
                explanation = self.lime_explainer.explain_instance(
                    instance,
                    self.model.predict_proba,
                    num_features=min(10, len(self.feature_names)),
                )
            else:
                explanation = self.lime_explainer.explain_instance(
                    instance,
                    self.model.predict,
                    num_features=min(10, len(self.feature_names)),
                )

            # Convertir a formato serializable
            feature_importance = explanation.as_list()

            return {
                "method": "lime",
                "feature_importance": feature_importance,
                "prediction_probability": (
                    explanation.predict_proba.tolist()
                    if hasattr(explanation, "predict_proba")
                    else None
                ),
            }

        except Exception as e:
            logger.error(f"Error with LIME explanation: {e}")
            return {"error": str(e)}

    # =====================================================
    # PERSISTENCIA
    # =====================================================

    def save_model(self, filepath: Optional[str] = None) -> str:
        """
        Guardar modelo entrenado.

        Args:
            filepath: Ruta donde guardar (opcional)

        Returns:
            str: Ruta del archivo guardado
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        # Generar filepath si no se proporciona
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = (
                f"{self.model_name}_{self.model_version}_{timestamp}.pkl"
            )
            filepath = MLConfig.MODELS_DIR / filename

        # Preparar datos para guardar
        model_data = {
            "model": self.model,
            "preprocessor": self.preprocessor,
            "feature_names": self.feature_names,
            "target_names": self.target_names,
            "training_metrics": self.training_metrics,
            "validation_metrics": self.validation_metrics,
            "feature_importance": self.feature_importance,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "model_version": self.model_version,
            "training_time": (
                self.training_time.total_seconds()
                if self.training_time
                else None
            ),
            "saved_at": datetime.now().isoformat(),
            "random_state": self.random_state,
        }

        # Guardar
        joblib.dump(model_data, filepath, compress=3)

        logger.info(f"Model saved to {filepath}")
        return str(filepath)

    def load_model(self, filepath: str):
        """
        Cargar modelo desde archivo.

        Args:
            filepath: Ruta del archivo del modelo
        """
        model_data = joblib.load(filepath)

        # Restaurar estado
        self.model = model_data["model"]
        self.preprocessor = model_data["preprocessor"]
        self.feature_names = model_data["feature_names"]
        self.target_names = model_data["target_names"]
        self.training_metrics = model_data["training_metrics"]
        self.validation_metrics = model_data["validation_metrics"]
        self.feature_importance = model_data["feature_importance"]
        self.model_name = model_data["model_name"]
        self.model_type = model_data["model_type"]
        self.model_version = model_data["model_version"]
        self.random_state = model_data["random_state"]

        self.is_trained = True

        logger.info(f"Model loaded from {filepath}")

    # =====================================================
    # UTILIDADES
    # =====================================================

    def get_model_info(self) -> Dict[str, Any]:
        """Obtener información completa del modelo."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "model_version": self.model_version,
            "is_trained": self.is_trained,
            "feature_count": len(self.feature_names),
            "training_metrics": self.training_metrics,
            "validation_metrics": self.validation_metrics,
            "training_time_seconds": (
                self.training_time.total_seconds()
                if self.training_time
                else None
            ),
            "has_feature_importance": bool(self.feature_importance),
            "explainability": {
                "shap_available": self.shap_explainer is not None,
                "lime_available": self.lime_explainer is not None,
            },
            "random_state": self.random_state,
        }

    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """
        Obtener top N features más importantes.

        Args:
            n: Número de features a retornar

        Returns:
            List: Lista de tuplas (feature_name, importance)
        """
        if not self.feature_importance:
            return []

        return list(self.feature_importance.items())[:n]

    def __repr__(self):
        status = "trained" if self.is_trained else "untrained"
        return f"<{self.__class__.__name__}(name='{self.model_name}', type='{self.model_type}', status='{status}')>"


# =====================================================
# MIXINS PARA FUNCIONALIDADES ESPECÍFICAS
# =====================================================


class FootballClassifierMixin:
    """Mixin con funcionalidades específicas para clasificadores de fútbol."""

    def predict_match_outcome(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Predecir resultado de partido con probabilidades."""
        if self.model_type != "classifier":
            raise ValueError("This method is only for classifiers")

        predictions = self.predict(X)
        probabilities = self.predict_proba(X)

        # Mapear a resultados de fútbol
        outcome_mapping = {0: "Away Win", 1: "Draw", 2: "Home Win"}

        results = []
        for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
            results.append(
                {
                    "predicted_outcome": outcome_mapping.get(pred, pred),
                    "probabilities": {
                        "home_win": float(proba[2]) if len(proba) > 2 else 0.0,
                        "draw": float(proba[1]) if len(proba) > 1 else 0.0,
                        "away_win": float(proba[0]) if len(proba) > 0 else 0.0,
                    },
                    "confidence": float(max(proba)),
                }
            )

        return {
            "predictions": results,
            "model_name": self.model_name,
            "prediction_count": len(results),
        }


class FootballRegressorMixin:
    """Mixin con funcionalidades específicas para regresores de fútbol."""

    def predict_goals(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Predecir número de goles."""
        if self.model_type != "regressor":
            raise ValueError("This method is only for regressors")

        predictions = self.predict(X)

        # Redondear a enteros para goles
        goal_predictions = np.round(np.maximum(predictions, 0)).astype(int)

        results = []
        for pred, goal_pred in zip(predictions, goal_predictions):
            results.append(
                {
                    "predicted_goals_exact": float(pred),
                    "predicted_goals_rounded": int(goal_pred),
                    "over_under_2_5": "over" if goal_pred > 2.5 else "under",
                    "over_under_1_5": "over" if goal_pred > 1.5 else "under",
                }
            )

        return {
            "predictions": results,
            "model_name": self.model_name,
            "average_goals": float(np.mean(goal_predictions)),
        }


# =====================================================
# CLASE BASE ESPECÍFICA PARA FÚTBOL
# =====================================================


class BaseFootballPredictor(BaseFootballModel):
    """
    Clase base específica para predictores de fútbol.
    Combina funcionalidades comunes de ML con lógica específica de fútbol.
    """

    def __init__(
        self,
        model_name: str,
        prediction_type: str,
        model_type: str = "classifier",
        random_state: int = MLConfig.RANDOM_STATE,
    ):
        """
        Inicializar predictor de fútbol.

        Args:
            model_name: Nombre del modelo
            prediction_type: Tipo de predicción ('match_result', 'over_under', etc.)
            model_type: Tipo de modelo ML ('classifier' o 'regressor')
            random_state: Semilla para reproducibilidad
        """
        super().__init__(model_name, model_type, random_state)
        self.prediction_type = prediction_type

        # Configuración específica de fútbol
        self.football_features = []
        self.team_features = []
        self.player_features = []
        self.match_features = []

        logger.info(f"Initialized football predictor: {prediction_type}")

    def add_football_features(
        self,
        team_features: List[str] = None,
        player_features: List[str] = None,
        match_features: List[str] = None,
    ):
        """
        Agregar features específicas de fútbol.

        Args:
            team_features: Features relacionadas con equipos
            player_features: Features relacionadas con jugadores
            match_features: Features relacionadas con partidos
        """
        self.team_features = team_features or []
        self.player_features = player_features or []
        self.match_features = match_features or []

        self.football_features = (
            self.team_features + self.player_features + self.match_features
        )

        logger.info(
            f"Added {len(self.football_features)} football-specific features"
        )

    def validate_football_data(self, X: pd.DataFrame) -> bool:
        """
        Validar que los datos contienen features necesarias para fútbol.

        Args:
            X: DataFrame con features

        Returns:
            bool: True si los datos son válidos
        """
        missing_features = []

        for feature in self.football_features:
            if feature not in X.columns:
                missing_features.append(feature)

        if missing_features:
            logger.warning(f"Missing football features: {missing_features}")
            return False

        return True

    def create_football_features(
        self, matches_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Crear features específicas de fútbol a partir de datos de partidos.

        Args:
            matches_df: DataFrame con datos de partidos

        Returns:
            pd.DataFrame: DataFrame con features creadas
        """
        df = matches_df.copy()

        # Features básicas de equipos
        if "home_team_id" in df.columns and "away_team_id" in df.columns:
            # Feature de rivalidad (mismo ID = derby)
            df["is_derby"] = (df["home_team_id"] == df["away_team_id"]).astype(
                int
            )

        # Features temporales
        if "match_date" in df.columns:
            df["match_date"] = pd.to_datetime(df["match_date"])
            df["month"] = df["match_date"].dt.month
            df["day_of_week"] = df["match_date"].dt.dayofweek
            df["is_weekend"] = (
                (df["day_of_week"] == 5) | (df["day_of_week"] == 6)
            ).astype(int)

        # Features de importancia
        if "importance" in df.columns:
            importance_mapping = {"low": 1, "normal": 2, "high": 3, "final": 4}
            df["importance_numeric"] = (
                df["importance"].map(importance_mapping).fillna(2)
            )

        # Features de ubicación
        if "is_neutral_venue" in df.columns:
            df["home_advantage"] = (~df["is_neutral_venue"]).astype(int)

        logger.info(f"Created football features, final shape: {df.shape}")
        return df

    def get_prediction_confidence_level(self, confidence_score: float) -> str:
        """
        Clasificar nivel de confianza de predicción.

        Args:
            confidence_score: Score de confianza (0-1)

        Returns:
            str: Nivel de confianza ('high', 'medium', 'low')
        """
        if confidence_score >= MLConfig.HIGH_CONFIDENCE_THRESHOLD:
            return "high"
        elif confidence_score >= MLConfig.MEDIUM_CONFIDENCE_THRESHOLD:
            return "medium"
        else:
            return "low"

    def generate_betting_insights(
        self, predictions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generar insights para apuestas basados en predicciones.

        Args:
            predictions: Predicciones del modelo

        Returns:
            dict: Insights para apuestas
        """
        insights = {
            "recommended_bets": [],
            "risk_level": "medium",
            "confidence_summary": {},
            "betting_strategy": "conservative",
        }

        if (
            self.prediction_type == "match_result"
            and "predictions" in predictions
        ):
            for pred in predictions["predictions"]:
                if (
                    pred.get("confidence", 0)
                    >= MLConfig.HIGH_CONFIDENCE_THRESHOLD
                ):
                    insights["recommended_bets"].append(
                        {
                            "bet_type": "match_result",
                            "prediction": pred["predicted_outcome"],
                            "confidence": pred["confidence"],
                            "rationale": f"High confidence ({pred['confidence']:.2f}) prediction",
                        }
                    )

        elif (
            self.prediction_type == "over_under"
            and "predictions" in predictions
        ):
            for pred in predictions["predictions"]:
                if (
                    pred.get("confidence", 0)
                    >= MLConfig.MEDIUM_CONFIDENCE_THRESHOLD
                ):
                    insights["recommended_bets"].append(
                        {
                            "bet_type": "over_under_2.5",
                            "prediction": pred.get(
                                "over_under_2_5", "unknown"
                            ),
                            "confidence": pred.get("confidence", 0),
                            "rationale": "Medium+ confidence prediction for goals",
                        }
                    )

        # Determinar estrategia
        high_conf_bets = len(
            [
                bet
                for bet in insights["recommended_bets"]
                if bet["confidence"] >= MLConfig.HIGH_CONFIDENCE_THRESHOLD
            ]
        )

        if high_conf_bets >= 2:
            insights["betting_strategy"] = "aggressive"
            insights["risk_level"] = "low"
        elif high_conf_bets == 1:
            insights["betting_strategy"] = "moderate"
            insights["risk_level"] = "medium"
        else:
            insights["betting_strategy"] = "conservative"
            insights["risk_level"] = "high"

        return insights

    def get_football_model_info(self) -> Dict[str, Any]:
        """Obtener información específica del modelo de fútbol."""
        base_info = self.get_model_info()

        football_info = {
            "prediction_type": self.prediction_type,
            "football_features": {
                "team_features": len(self.team_features),
                "player_features": len(self.player_features),
                "match_features": len(self.match_features),
                "total_football_features": len(self.football_features),
            },
            "betting_ready": True,
            "supported_markets": self._get_supported_betting_markets(),
        }

        # Combinar información
        base_info.update(football_info)
        return base_info

    def _get_supported_betting_markets(self) -> List[str]:
        """Obtener mercados de apuestas soportados según el tipo de predicción."""
        market_mapping = {
            "match_result": ["1X2", "Double Chance", "Home/Away"],
            "over_under": [
                "Over/Under 0.5",
                "Over/Under 1.5",
                "Over/Under 2.5",
                "Over/Under 3.5",
            ],
            "both_teams_score": ["BTTS Yes/No"],
            "correct_score": ["Exact Score", "Score Groups"],
            "first_half": ["First Half Result", "First Half Over/Under"],
            "cards": ["Total Cards", "Player Cards"],
            "corners": ["Total Corners", "Corner Handicap"],
        }

        return market_mapping.get(self.prediction_type, ["Generic"])


# =====================================================
# FUNCIONES DE UTILIDAD
# =====================================================


def create_ensemble_model(
    models: List[BaseFootballModel],
    weights: Optional[List[float]] = None,
    voting: str = "soft",
) -> "EnsembleFootballModel":
    """
    Crear modelo ensemble combinando múltiples modelos.

    Args:
        models: Lista de modelos entrenados
        weights: Pesos para cada modelo (opcional)
        voting: Tipo de voting ('soft' o 'hard')

    Returns:
        EnsembleFootballModel: Modelo ensemble
    """
    from sklearn.ensemble import VotingClassifier, VotingRegressor

    # Validar que todos los modelos son del mismo tipo
    model_types = set(model.model_type for model in models)
    if len(model_types) > 1:
        raise ValueError(
            "All models must be of the same type (classifier or regressor)"
        )

    model_type = models[0].model_type

    # Crear estimadores para ensemble
    estimators = [
        (f"model_{i}", model.model) for i, model in enumerate(models)
    ]

    # Crear ensemble
    if model_type == "classifier":
        ensemble = VotingClassifier(
            estimators=estimators, voting=voting, weights=weights
        )
    else:
        ensemble = VotingRegressor(estimators=estimators, weights=weights)

    # Crear modelo wrapper
    ensemble_model = EnsembleFootballModel(
        models=models, ensemble_estimator=ensemble, weights=weights
    )

    return ensemble_model


class EnsembleFootballModel(BaseFootballModel):
    """Modelo ensemble que combina múltiples modelos de fútbol."""

    def __init__(
        self,
        models: List[BaseFootballModel],
        ensemble_estimator,
        weights: Optional[List[float]] = None,
    ):

        # Usar el nombre del primer modelo como base
        model_name = f"ensemble_{models[0].model_name}"
        model_type = models[0].model_type

        super().__init__(model_name, model_type)

        self.base_models = models
        self.ensemble_estimator = ensemble_estimator
        self.weights = weights or [1.0] * len(models)

        # Heredar configuración del primer modelo
        if models:
            self.feature_names = models[0].feature_names
            self.preprocessor = models[0].preprocessor
            self.is_trained = all(model.is_trained for model in models)

    def _create_model(self):
        """El ensemble ya está creado."""
        return self.ensemble_estimator

    def _get_default_hyperparameters(self):
        """Ensemble no tiene hiperparámetros propios."""
        return {}

    def _get_hyperparameter_grid(self):
        """Ensemble no tiene grid de hiperparámetros."""
        return {}

    def predict_with_individual_models(
        self, X: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Hacer predicciones con modelos individuales y ensemble.

        Args:
            X: Features para predicción

        Returns:
            dict: Predicciones de todos los modelos
        """
        results = {
            "individual_predictions": {},
            "ensemble_prediction": None,
            "confidence_scores": {},
            "model_agreement": None,
        }

        # Predicciones individuales
        individual_preds = []
        for i, model in enumerate(self.base_models):
            pred = model.predict_with_confidence(X)
            results["individual_predictions"][
                f"model_{i}_{model.model_name}"
            ] = pred
            individual_preds.append(pred["predictions"])

        # Predicción ensemble
        ensemble_pred = self.predict_with_confidence(X)
        results["ensemble_prediction"] = ensemble_pred

        # Calcular acuerdo entre modelos
        if individual_preds:
            # Para clasificación, calcular porcentaje de acuerdo
            if self.model_type == "classifier":
                agreements = []
                for i in range(len(individual_preds[0])):
                    predictions_for_sample = [
                        pred[i] for pred in individual_preds
                    ]
                    most_common = max(
                        set(predictions_for_sample),
                        key=predictions_for_sample.count,
                    )
                    agreement = predictions_for_sample.count(
                        most_common
                    ) / len(predictions_for_sample)
                    agreements.append(agreement)

                results["model_agreement"] = {
                    "average_agreement": float(np.mean(agreements)),
                    "min_agreement": float(np.min(agreements)),
                    "max_agreement": float(np.max(agreements)),
                    "agreement_per_sample": agreements,
                }

        return results


# =====================================================
# EXPORTACIONES
# =====================================================

__all__ = [
    # Clase base principal
    "BaseFootballModel",
    "BaseFootballPredictor",
    # Mixins
    "FootballClassifierMixin",
    "FootballRegressorMixin",
    # Ensemble
    "EnsembleFootballModel",
    "create_ensemble_model",
    # Utilidades
    "SHAP_AVAILABLE",
    "LIME_AVAILABLE",
]
