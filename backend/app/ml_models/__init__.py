"""
Machine Learning Models Package for Football Analytics

Este módulo contiene todos los modelos de Machine Learning para predicciones
de fútbol, incluyendo clasificadores, regresores, ensemble methods y utilidades.
"""

import logging
import os
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib

# Suprimir warnings de sklearn
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================
# CONFIGURACIÓN DE MODELOS ML
# =====================================================


class MLConfig:
    """Configuración centralizada para modelos de Machine Learning."""

    # Directorios
    BASE_DIR = Path(__file__).parent
    MODELS_DIR = BASE_DIR / "saved_models"
    TRAINING_DATA_DIR = BASE_DIR / "training_data"
    LOGS_DIR = BASE_DIR / "logs"

    # Versiones de modelos
    MODEL_VERSION = "v2.0"
    FEATURE_VERSION = "v1.3"

    # Configuración de entrenamiento
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.1
    CV_FOLDS = 5

    # Parámetros de modelos
    MAX_ITER = 1000
    N_ESTIMATORS = 100
    LEARNING_RATE = 0.1

    # Umbrales de confianza
    HIGH_CONFIDENCE_THRESHOLD = 0.8
    MEDIUM_CONFIDENCE_THRESHOLD = 0.6
    LOW_CONFIDENCE_THRESHOLD = 0.4

    # Configuración de features
    FEATURE_SELECTION = True
    FEATURE_SCALING = True
    HANDLE_MISSING = True

    @classmethod
    def ensure_directories(cls):
        """Crear directorios necesarios si no existen."""
        for directory in [cls.MODELS_DIR, cls.TRAINING_DATA_DIR, cls.LOGS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")


# =====================================================
# GESTIÓN DE MODELOS
# =====================================================


class ModelRegistry:
    """Registro centralizado de todos los modelos ML."""

    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._model_metadata: Dict[str, Dict] = {}
        self._loaded_models: Dict[str, bool] = {}

    def register_model(
        self, name: str, model: Any, metadata: Optional[Dict] = None
    ):
        """
        Registrar un modelo en el registry.

        Args:
            name: Nombre único del modelo
            model: Instancia del modelo
            metadata: Metadatos del modelo (versión, métricas, etc.)
        """
        self._models[name] = model
        self._model_metadata[name] = metadata or {}
        self._loaded_models[name] = True

        logger.info(f"Model '{name}' registered successfully")

    def get_model(self, name: str) -> Optional[Any]:
        """Obtener modelo por nombre."""
        if name not in self._models:
            logger.warning(f"Model '{name}' not found in registry")
            return None

        if not self._loaded_models.get(name, False):
            logger.warning(f"Model '{name}' is registered but not loaded")
            return None

        return self._models[name]

    def get_metadata(self, name: str) -> Dict:
        """Obtener metadatos de un modelo."""
        return self._model_metadata.get(name, {})

    def list_models(self) -> List[str]:
        """Listar todos los modelos registrados."""
        return list(self._models.keys())

    def get_loaded_models(self) -> List[str]:
        """Obtener lista de modelos cargados."""
        return [name for name, loaded in self._loaded_models.items() if loaded]

    def unload_model(self, name: str):
        """Descargar modelo de la memoria."""
        if name in self._models:
            del self._models[name]
            self._loaded_models[name] = False
            logger.info(f"Model '{name}' unloaded from memory")

    def get_registry_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del registry."""
        return {
            "total_models": len(self._models),
            "loaded_models": len(self.get_loaded_models()),
            "registered_models": self.list_models(),
            "model_types": self._get_model_types(),
            "registry_size_mb": self._estimate_memory_usage(),
        }

    def _get_model_types(self) -> Dict[str, int]:
        """Obtener tipos de modelos registrados."""
        types = {}
        for model in self._models.values():
            model_type = type(model).__name__
            types[model_type] = types.get(model_type, 0) + 1
        return types

    def _estimate_memory_usage(self) -> float:
        """Estimar uso de memoria (aproximado)."""
        import sys

        total_size = 0
        for model in self._models.values():
            total_size += sys.getsizeof(model)
        return round(total_size / (1024 * 1024), 2)  # MB


# =====================================================
# GESTIÓN DE ARCHIVOS DE MODELOS
# =====================================================


class ModelPersistence:
    """Utilidades para guardar y cargar modelos."""

    @staticmethod
    def save_model(
        model: Any, name: str, metadata: Optional[Dict] = None
    ) -> str:
        """
        Guardar modelo en disco.

        Args:
            model: Modelo a guardar
            name: Nombre del modelo
            metadata: Metadatos adicionales

        Returns:
            str: Ruta del archivo guardado
        """
        try:
            MLConfig.ensure_directories()

            # Crear nombre de archivo con timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{MLConfig.MODEL_VERSION}_{timestamp}.pkl"
            filepath = MLConfig.MODELS_DIR / filename

            # Preparar datos para guardar
            model_data = {
                "model": model,
                "metadata": metadata or {},
                "version": MLConfig.MODEL_VERSION,
                "saved_at": datetime.now().isoformat(),
                "name": name,
            }

            # Guardar con joblib (mejor para modelos sklearn)
            joblib.dump(model_data, filepath, compress=3)

            logger.info(f"Model '{name}' saved to {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Error saving model '{name}': {e}")
            raise

    @staticmethod
    def load_model(filepath: str) -> Dict[str, Any]:
        """
        Cargar modelo desde disco.

        Args:
            filepath: Ruta del archivo del modelo

        Returns:
            dict: Datos del modelo cargado
        """
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Model file not found: {filepath}")

            model_data = joblib.load(filepath)

            logger.info(f"Model loaded from {filepath}")
            return model_data

        except Exception as e:
            logger.error(f"Error loading model from '{filepath}': {e}")
            raise

    @staticmethod
    def list_saved_models() -> List[Dict[str, Any]]:
        """Listar todos los modelos guardados."""
        try:
            MLConfig.ensure_directories()
            model_files = []

            for filepath in MLConfig.MODELS_DIR.glob("*.pkl"):
                try:
                    # Obtener información básica del archivo
                    stat = filepath.stat()

                    model_info = {
                        "filename": filepath.name,
                        "filepath": str(filepath),
                        "size_mb": round(stat.st_size / (1024 * 1024), 2),
                        "modified": datetime.fromtimestamp(
                            stat.st_mtime
                        ).isoformat(),
                        "accessible": True,
                    }

                    # Intentar obtener metadatos sin cargar el modelo completo
                    try:
                        with open(filepath, "rb") as f:
                            # Solo leer metadatos básicos
                            data = pickle.load(f)
                            if isinstance(data, dict):
                                model_info.update(
                                    {
                                        "name": data.get("name", "unknown"),
                                        "version": data.get(
                                            "version", "unknown"
                                        ),
                                        "saved_at": data.get(
                                            "saved_at", "unknown"
                                        ),
                                    }
                                )
                    except:
                        model_info["metadata_error"] = True

                    model_files.append(model_info)

                except Exception as e:
                    logger.warning(f"Error reading model file {filepath}: {e}")

            return sorted(
                model_files, key=lambda x: x["modified"], reverse=True
            )

        except Exception as e:
            logger.error(f"Error listing saved models: {e}")
            return []

    @staticmethod
    def delete_model_file(filepath: str) -> bool:
        """Eliminar archivo de modelo."""
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"Model file deleted: {filepath}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting model file '{filepath}': {e}")
            return False


# =====================================================
# INSTANCIAS GLOBALES
# =====================================================

# Registry global de modelos
model_registry = ModelRegistry()

# =====================================================
# FUNCIONES DE INICIALIZACIÓN
# =====================================================


def initialize_ml_environment():
    """Inicializar entorno de Machine Learning."""
    try:
        logger.info("Initializing ML environment...")

        # Crear directorios necesarios
        MLConfig.ensure_directories()

        # Cargar modelos disponibles
        load_available_models()

        # Configurar logging específico para ML
        setup_ml_logging()

        logger.info("ML environment initialized successfully")
        return True

    except Exception as e:
        logger.error(f"Error initializing ML environment: {e}")
        return False


def load_available_models():
    """Cargar modelos disponibles automáticamente."""
    try:
        saved_models = ModelPersistence.list_saved_models()
        loaded_count = 0

        for model_info in saved_models:
            try:
                # Solo cargar modelos recientes y válidos
                if model_info.get("accessible", False) and not model_info.get(
                    "metadata_error", False
                ):
                    # Por ahora, solo registrar sin cargar (lazy loading)
                    model_name = model_info.get(
                        "name", f"model_{loaded_count}"
                    )
                    model_registry._model_metadata[model_name] = model_info
                    model_registry._loaded_models[model_name] = False
                    loaded_count += 1

            except Exception as e:
                logger.warning(
                    f"Error registering model {model_info.get('filename', 'unknown')}: {e}"
                )

        logger.info(f"Registered {loaded_count} available models")

    except Exception as e:
        logger.warning(f"Error loading available models: {e}")


def setup_ml_logging():
    """Configurar logging específico para ML."""
    try:
        # Crear logger específico para ML
        ml_logger = logging.getLogger("football_analytics.ml")
        ml_logger.setLevel(logging.INFO)

        # Handler para archivo de logs
        log_file = (
            MLConfig.LOGS_DIR
            / f"ml_models_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Formato de logs
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)

        ml_logger.addHandler(file_handler)
        logger.info(f"ML logging configured, log file: {log_file}")

    except Exception as e:
        logger.warning(f"Error setting up ML logging: {e}")


# =====================================================
# FUNCIONES DE UTILIDAD
# =====================================================


def get_ml_info() -> Dict[str, Any]:
    """Obtener información completa del módulo ML."""
    return {
        "config": {
            "model_version": MLConfig.MODEL_VERSION,
            "feature_version": MLConfig.FEATURE_VERSION,
            "random_state": MLConfig.RANDOM_STATE,
            "models_directory": str(MLConfig.MODELS_DIR),
            "confidence_thresholds": {
                "high": MLConfig.HIGH_CONFIDENCE_THRESHOLD,
                "medium": MLConfig.MEDIUM_CONFIDENCE_THRESHOLD,
                "low": MLConfig.LOW_CONFIDENCE_THRESHOLD,
            },
        },
        "registry": model_registry.get_registry_stats(),
        "saved_models": len(ModelPersistence.list_saved_models()),
        "available_predictors": get_available_predictors(),
        "initialization_time": datetime.now().isoformat(),
    }


def get_available_predictors() -> List[str]:
    """Obtener lista de predictores disponibles."""
    predictors = []

    # Buscar archivos de predictores
    try:
        from . import predictors

        predictor_files = [
            "match_result_predictor",
            "over_under_predictor",
            "both_teams_score_predictor",
            "correct_score_predictor",
            "player_performance_predictor",
        ]

        for predictor in predictor_files:
            try:
                # Verificar si el archivo existe
                predictor_path = (
                    MLConfig.BASE_DIR / "predictors" / f"{predictor}.py"
                )
                if predictor_path.exists():
                    predictors.append(predictor)
            except:
                continue

    except ImportError:
        logger.warning("Predictors module not found")

    return predictors


def cleanup_old_models(days_old: int = 30) -> int:
    """
    Limpiar modelos antiguos.

    Args:
        days_old: Días de antigüedad para considerar eliminar

    Returns:
        int: Número de modelos eliminados
    """
    try:
        import time

        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        deleted_count = 0

        for model_file in MLConfig.MODELS_DIR.glob("*.pkl"):
            if model_file.stat().st_mtime < cutoff_time:
                try:
                    model_file.unlink()
                    deleted_count += 1
                    logger.info(f"Deleted old model: {model_file.name}")
                except Exception as e:
                    logger.warning(
                        f"Error deleting old model {model_file.name}: {e}"
                    )

        logger.info(f"Cleanup completed: {deleted_count} old models deleted")
        return deleted_count

    except Exception as e:
        logger.error(f"Error during model cleanup: {e}")
        return 0


# =====================================================
# IMPORTS DE MÓDULOS
# =====================================================

# Importar componentes principales
try:
    from .predictors import *

    logger.info("Predictors module imported successfully")
except ImportError as e:
    logger.warning(f"Could not import predictors: {e}")

try:
    from .utils import *

    logger.info("ML utilities imported successfully")
except ImportError as e:
    logger.warning(f"Could not import ML utilities: {e}")

try:
    from .feature_engineering import *

    logger.info("Feature engineering module imported successfully")
except ImportError as e:
    logger.warning(f"Could not import feature engineering: {e}")

try:
    from .model_evaluation import *

    logger.info("Model evaluation module imported successfully")
except ImportError as e:
    logger.warning(f"Could not import model evaluation: {e}")

# =====================================================
# INFORMACIÓN DEL MÓDULO
# =====================================================

__version__ = "2.0.0"
__author__ = "Football Analytics Team"
__description__ = "Módulo de Machine Learning para predicciones de fútbol"

# Metadata del módulo ML
ML_MODULE_INFO = {
    "version": __version__,
    "description": __description__,
    "algorithms": [
        "Random Forest",
        "XGBoost",
        "LightGBM",
        "CatBoost",
        "Neural Networks",
        "Ensemble Methods",
    ],
    "prediction_types": [
        "Match Result (1X2)",
        "Over/Under Goals",
        "Both Teams Score",
        "Correct Score",
        "Player Performance",
        "Card Predictions",
        "First Half Results",
    ],
    "features": [
        "Feature Engineering",
        "Model Persistence",
        "Hyperparameter Optimization",
        "Model Evaluation",
        "Explainable AI (SHAP)",
        "Real-time Predictions",
        "Ensemble Voting",
    ],
    "supported_leagues": "All major football leagues worldwide",
    "data_sources": "Historical match data, player stats, team performance",
}

# =====================================================
# AUTO-INICIALIZACIÓN
# =====================================================

# Inicializar automáticamente al importar el módulo
_initialization_success = initialize_ml_environment()

if _initialization_success:
    logger.info(
        f"Football Analytics ML Module v{__version__} loaded successfully"
    )
else:
    logger.error("Failed to initialize ML module properly")

# =====================================================
# EXPORTACIONES
# =====================================================

__all__ = [
    # Configuración
    "MLConfig",
    # Gestión de modelos
    "ModelRegistry",
    "ModelPersistence",
    "model_registry",
    # Funciones principales
    "initialize_ml_environment",
    "get_ml_info",
    "get_available_predictors",
    "cleanup_old_models",
    # Información
    "ML_MODULE_INFO",
    "__version__",
]
