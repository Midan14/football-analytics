#!/usr/bin/env python3
"""
Football Analytics - Utils Package Initialization
Inicialización del paquete de utilidades del sistema Football Analytics

Autor: Sistema Football Analytics
Versión: 2.1.0
Fecha: 2024-06-02
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

# Importar todos los módulos de utilidades
from .validators import (  # Funciones de conveniencia; Decoradores
    ComprehensiveValidator,
    FeatureValidator,
    MatchValidator,
    OddsValidator,
    PredictionValidator,
    TeamValidator,
    ValidationError,
    ValidationLevel,
    ValidationResult,
    validate_feature_dataframe,
    validate_full_pipeline,
    validate_input,
    validate_match_data,
    validate_odds,
    validate_prediction_probabilities,
    validate_team_name,
)

# Configuración del sistema de utilidades
UTILS_CONFIG = {
    "logging_level": "INFO",
    "validation_level": ValidationLevel.NORMAL,
    "date_format": "%Y-%m-%d %H:%M:%S",
    "timezone": "UTC",
    "max_log_size": "100MB",
    "backup_count": 10,
    "cache_size": 1000,
    "performance_monitoring": True,
}


# Configuración de logging para utilidades
def configure_logging(
    level: str = "INFO", log_file: Optional[str] = None
) -> logging.Logger:
    """
    Configura el sistema de logging para utilidades.

    Args:
        level: Nivel de logging ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Archivo opcional para logs

    Returns:
        Logger configurado
    """
    logger = logging.getLogger("football_analytics.utils")
    logger.setLevel(getattr(logging, level.upper()))

    # Formato de logging
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler para archivo si se especifica
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Logger principal para utilidades
utils_logger = configure_logging(
    level=UTILS_CONFIG["logging_level"],
    log_file="/app/logs/utils.log" if os.path.exists("/app/logs") else None,
)


class UtilsManager:
    """
    Gestor centralizado de todas las utilidades del sistema.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa el gestor de utilidades.

        Args:
            config: Configuración opcional para utilidades
        """
        self.config = config or UTILS_CONFIG.copy()
        self.validators = self._initialize_validators()
        self.performance_metrics = {}

        utils_logger.info("🔧 UtilsManager inicializado")

    def _initialize_validators(self) -> Dict[str, Any]:
        """Inicializa todos los validadores."""
        return {
            "team": TeamValidator(),
            "match": MatchValidator(),
            "prediction": PredictionValidator(),
            "odds": OddsValidator(),
            "feature": FeatureValidator(),
            "comprehensive": ComprehensiveValidator(),
        }

    def validate_data(
        self, data_type: str, data: Any, **kwargs
    ) -> ValidationResult:
        """
        Valida datos usando el validador apropiado.

        Args:
            data_type: Tipo de dato ('team', 'match', 'prediction', 'odds', 'feature')
            data: Datos a validar
            **kwargs: Argumentos adicionales para validador

        Returns:
            Resultado de validación
        """
        if data_type not in self.validators:
            raise ValueError(f"Tipo de validador desconocido: {data_type}")

        validator = self.validators[data_type]

        # Mapeo de métodos de validación
        validation_methods = {
            "team": validator.validate_team_data,
            "match": validator.validate_match_data,
            "prediction": validator.validate_prediction_data,
            "odds": validator.validate_odds_data,
            "feature": validator.validate_feature_dataframe,
        }

        method = validation_methods.get(data_type)
        if not method:
            raise ValueError(
                f"Método de validación no encontrado para: {data_type}"
            )

        return method(data, **kwargs)

    def get_validator(self, validator_type: str):
        """Obtiene un validador específico."""
        return self.validators.get(validator_type)

    def set_validation_level(self, level: ValidationLevel):
        """Establece el nivel de validación global."""
        self.config["validation_level"] = level
        utils_logger.info(
            f"🔧 Nivel de validación establecido a: {level.value}"
        )

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas de rendimiento de utilidades."""
        return self.performance_metrics.copy()

    def log_performance(self, operation: str, duration: float):
        """Registra métricas de rendimiento."""
        if operation not in self.performance_metrics:
            self.performance_metrics[operation] = []

        self.performance_metrics[operation].append(
            {"duration": duration, "timestamp": datetime.now()}
        )

        # Mantener solo últimas 100 métricas por operación
        if len(self.performance_metrics[operation]) > 100:
            self.performance_metrics[operation] = self.performance_metrics[
                operation
            ][-100:]


# Instancia global del gestor de utilidades
utils_manager = UtilsManager()


# Funciones de conveniencia para acceso rápido
def get_validator(validator_type: str):
    """Obtiene un validador específico."""
    return utils_manager.get_validator(validator_type)


def validate_data(data_type: str, data: Any, **kwargs) -> ValidationResult:
    """Valida datos usando el validador apropiado."""
    return utils_manager.validate_data(data_type, data, **kwargs)


def set_validation_level(level: ValidationLevel):
    """Establece el nivel de validación global."""
    utils_manager.set_validation_level(level)


def get_performance_metrics() -> Dict[str, Any]:
    """Obtiene métricas de rendimiento."""
    return utils_manager.get_performance_metrics()


# Decoradores de utilidad
def performance_monitor(operation_name: str):
    """
    Decorador para monitorear rendimiento de funciones.

    Args:
        operation_name: Nombre de la operación para métricas
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = (datetime.now() - start_time).total_seconds()
                utils_manager.log_performance(operation_name, duration)

                if duration > 1.0:  # Log operaciones lentas
                    utils_logger.warning(
                        f"⚠️ Operación lenta: {operation_name} tomó {duration:.2f}s"
                    )

        return wrapper

    return decorator


def async_performance_monitor(operation_name: str):
    """
    Decorador para monitorear rendimiento de funciones asíncronas.

    Args:
        operation_name: Nombre de la operación para métricas
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = datetime.now()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = (datetime.now() - start_time).total_seconds()
                utils_manager.log_performance(operation_name, duration)

                if duration > 1.0:  # Log operaciones lentas
                    utils_logger.warning(
                        f"⚠️ Operación async lenta: {operation_name} tomó {duration:.2f}s"
                    )

        return wrapper

    return decorator


# Utilidades de fecha y hora
def get_current_timestamp() -> str:
    """Obtiene timestamp actual en formato estándar."""
    return datetime.now().strftime(UTILS_CONFIG["date_format"])


def parse_date_string(date_str: str) -> datetime:
    """
    Parsea string de fecha a objeto datetime.

    Args:
        date_str: String de fecha en formato YYYY-MM-DD o YYYY-MM-DD HH:MM:SS

    Returns:
        Objeto datetime
    """
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    raise ValueError(f"Formato de fecha no reconocido: {date_str}")


def is_future_date(date_obj: datetime, days_ahead: int = 365) -> bool:
    """
    Verifica si una fecha está en el futuro (dentro de límites razonables).

    Args:
        date_obj: Fecha a verificar
        days_ahead: Días máximos en el futuro permitidos

    Returns:
        True si está en futuro válido
    """
    now = datetime.now()
    max_future = now + timedelta(days=days_ahead)

    return now <= date_obj <= max_future


# Utilidades de string
def normalize_team_name(team_name: str) -> str:
    """
    Normaliza nombre de equipo para comparaciones.

    Args:
        team_name: Nombre original del equipo

    Returns:
        Nombre normalizado
    """
    if not team_name:
        return ""

    # Remover espacios extra y convertir a title case
    normalized = " ".join(team_name.strip().split()).title()

    # Mapeos de nombres comunes
    name_mappings = {
        "Fc Barcelona": "Barcelona",
        "Real Madrid Cf": "Real Madrid",
        "Atletico De Madrid": "Atletico Madrid",
        "Manchester United Fc": "Manchester United",
        "Liverpool Fc": "Liverpool",
    }

    return name_mappings.get(normalized, normalized)


def sanitize_string(text: str, max_length: int = 100) -> str:
    """
    Sanitiza string removiendo caracteres peligrosos.

    Args:
        text: Texto a sanitizar
        max_length: Longitud máxima permitida

    Returns:
        Texto sanitizado
    """
    if not text:
        return ""

    # Remover caracteres peligrosos
    dangerous_chars = ["<", ">", '"', "'", "&", "\n", "\r", "\t"]
    sanitized = text

    for char in dangerous_chars:
        sanitized = sanitized.replace(char, "")

    # Truncar si es necesario
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip()

    return sanitized.strip()


# Utilidades numéricas
def safe_divide(
    numerator: float, denominator: float, default: float = 0.0
) -> float:
    """
    División segura que maneja división por cero.

    Args:
        numerator: Numerador
        denominator: Denominador
        default: Valor por defecto si denominador es 0

    Returns:
        Resultado de la división o valor por defecto
    """
    if denominator == 0:
        return default
    return numerator / denominator


def clamp_value(value: float, min_val: float, max_val: float) -> float:
    """
    Limita un valor entre un mínimo y máximo.

    Args:
        value: Valor a limitar
        min_val: Valor mínimo
        max_val: Valor máximo

    Returns:
        Valor limitado
    """
    return max(min_val, min(value, max_val))


def is_probability_distribution(
    values: List[float], tolerance: float = 0.01
) -> bool:
    """
    Verifica si una lista de valores forma una distribución de probabilidad válida.

    Args:
        values: Lista de valores
        tolerance: Tolerancia para la suma (debe ser ≈ 1.0)

    Returns:
        True si es distribución válida
    """
    if not values:
        return False

    # Verificar que todos sean positivos
    if any(v < 0 for v in values):
        return False

    # Verificar que sumen aproximadamente 1.0
    total = sum(values)
    return abs(total - 1.0) <= tolerance


# Inicialización del paquete
def initialize_utils(config: Optional[Dict[str, Any]] = None) -> UtilsManager:
    """
    Inicializa el sistema de utilidades.

    Args:
        config: Configuración opcional

    Returns:
        Gestor de utilidades inicializado
    """
    global utils_manager

    if config:
        utils_manager = UtilsManager(config)

    utils_logger.info("🔧 Sistema de utilidades inicializado correctamente")
    utils_logger.info(
        f"📊 Validadores disponibles: {len(utils_manager.validators)}"
    )
    utils_logger.info(
        f"⚙️ Nivel de validación: {utils_manager.config['validation_level'].value}"
    )

    return utils_manager


def get_utils_status() -> Dict[str, Any]:
    """
    Obtiene el estado del sistema de utilidades.

    Returns:
        Diccionario con información de estado
    """
    return {
        "initialized": utils_manager is not None,
        "validators_count": (
            len(utils_manager.validators) if utils_manager else 0
        ),
        "validation_level": (
            utils_manager.config["validation_level"].value
            if utils_manager
            else None
        ),
        "performance_metrics_count": (
            len(utils_manager.performance_metrics) if utils_manager else 0
        ),
        "timestamp": get_current_timestamp(),
    }


# Exportaciones principales
__all__ = [
    # Validadores
    "TeamValidator",
    "MatchValidator",
    "PredictionValidator",
    "OddsValidator",
    "FeatureValidator",
    "ComprehensiveValidator",
    "ValidationResult",
    "ValidationLevel",
    "ValidationError",
    # Funciones de validación
    "validate_team_name",
    "validate_match_data",
    "validate_prediction_probabilities",
    "validate_odds",
    "validate_feature_dataframe",
    "validate_full_pipeline",
    "validate_input",
    # Gestor de utilidades
    "UtilsManager",
    "utils_manager",
    # Funciones de conveniencia
    "get_validator",
    "validate_data",
    "set_validation_level",
    "get_performance_metrics",
    # Decoradores
    "performance_monitor",
    "async_performance_monitor",
    # Utilidades de fecha
    "get_current_timestamp",
    "parse_date_string",
    "is_future_date",
    # Utilidades de string
    "normalize_team_name",
    "sanitize_string",
    # Utilidades numéricas
    "safe_divide",
    "clamp_value",
    "is_probability_distribution",
    # Configuración y estado
    "UTILS_CONFIG",
    "configure_logging",
    "initialize_utils",
    "get_utils_status",
    "utils_logger",
]

# Inicialización automática al importar
utils_logger.info("🔧 Football Analytics Utils Package inicializado")
utils_logger.info(f"📦 Módulos exportados: {len(__all__)}")
utils_logger.info("✅ Sistema de utilidades listo para uso")

# Mensaje de bienvenida
if __name__ == "__main__":
    print("🔧 Football Analytics - Utils Package")
    print("=====================================")
    print("✅ Sistema de utilidades inicializado")
    print(f"📊 Validadores disponibles: {len(utils_manager.validators)}")
    print(
        f"⚙️ Nivel de validación: {utils_manager.config['validation_level'].value}"
    )
    print(f"📦 Funciones exportadas: {len(__all__)}")
    print("🚀 Listo para validar datos de fútbol!")
