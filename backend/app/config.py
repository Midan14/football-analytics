"""
config.py - Configuración del Sistema Football Analytics

Sistema de configuración centralizado y adaptativo para Football Analytics.
Gestiona configuraciones por entorno, APIs, base de datos, ML y monitoreo.

Author: Football Analytics Team
Version: 2.1.0
Date: 2024-06-02
"""

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


# Configuración para solucionar los errores de importación
class settings:
    DATABASE_URL = "sqlite:///./football.db"
    DB_ECHO = False
    DEBUG = True


# ================================
# ENUMS DE CONFIGURACIÓN
# ================================


class Environment(Enum):
    """Entornos de ejecución disponibles"""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(Enum):
    """Niveles de logging disponibles"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseType(Enum):
    """Tipos de base de datos soportados"""

    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"


class MLFramework(Enum):
    """Frameworks de ML soportados"""

    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    SKLEARN = "sklearn"


# ================================
# CONFIGURACIONES BASE
# ================================


@dataclass
class APIConfig:
    """Configuración de APIs externas"""

    # Football-Data.org (tu key real)
    football_data_api_key: str = "9c9a42cbff2e8eb387eac2755c5e1e97"
    football_data_base_url: str = "https://api.football-data.org/v4"
    football_data_rate_limit: int = 10  # requests per minute (free tier)
    football_data_timeout: int = 30

    # RapidAPI Football
    rapidapi_key: str = field(default_factory=lambda: os.getenv("RAPIDAPI_KEY", ""))
    rapidapi_base_url: str = "https://api-football-v1.p.rapidapi.com/v3"
    rapidapi_rate_limit: int = 100
    rapidapi_timeout: int = 30

    # The Odds API
    odds_api_key: str = field(default_factory=lambda: os.getenv("ODDS_API_KEY", ""))
    odds_api_base_url: str = "https://api.the-odds-api.com/v4"
    odds_api_rate_limit: int = 500
    odds_api_timeout: int = 30

    # Configuración general
    max_retries: int = 3
    backoff_factor: float = 1.0
    verify_ssl: bool = True

    def get_headers(self, api_name: str) -> Dict[str, str]:
        """Obtiene headers específicos para cada API"""
        headers = {
            "User-Agent": "FootballAnalytics/2.1.0",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        if api_name == "football_data":
            headers["X-Auth-Token"] = self.football_data_api_key
        elif api_name == "rapidapi":
            headers["X-RapidAPI-Key"] = self.rapidapi_key
            headers["X-RapidAPI-Host"] = "api-football-v1.p.rapidapi.com"
        elif api_name == "odds":
            headers["X-API-Key"] = self.odds_api_key

        return headers


@dataclass
class DatabaseConfig:
    """Configuración de base de datos"""

    type: DatabaseType = DatabaseType.SQLITE
    host: str = "localhost"
    port: int = 5432
    database: str = "football_analytics"
    username: str = "football_user"
    password: str = field(default_factory=lambda: os.getenv("DB_PASSWORD", ""))

    # SQLite específico (tu configuración actual)
    sqlite_path: str = "data/football_analytics.db"
    sqlite_timeout: float = 30.0

    # Pool de conexiones
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600

    # Configuración de performance
    echo: bool = False
    echo_pool: bool = False

    def get_connection_string(self) -> str:
        """Genera string de conexión según el tipo de DB"""
        if self.type == DatabaseType.SQLITE:
            return f"sqlite:///{self.sqlite_path}"
        elif self.type == DatabaseType.POSTGRESQL:
            return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        elif self.type == DatabaseType.MYSQL:
            return f"mysql+pymysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        else:
            raise ValueError(f"Tipo de base de datos no soportado: {self.type}")


@dataclass
class MLConfig:
    """Configuración de Machine Learning"""

    # Modelos disponibles
    available_models: List[MLFramework] = field(
        default_factory=lambda: [
            MLFramework.XGBOOST,
            MLFramework.LIGHTGBM,
            MLFramework.CATBOOST,
        ]
    )

    # Modelo por defecto
    default_model: MLFramework = MLFramework.XGBOOST

    # Paths de modelos
    models_dir: str = "models"
    model_cache_dir: str = "models/cache"
    training_data_dir: str = "data/training"

    # Configuración de entrenamiento
    test_size: float = 0.2
    random_state: int = 42
    cross_validation_folds: int = 5

    # Hyperparameters por modelo
    xgboost_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 1000,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
        }
    )

    lightgbm_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 1000,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
    )

    catboost_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "iterations": 1000,
            "depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "random_state": 42,
            "verbose": False,
            "thread_count": -1,
        }
    )

    # Thresholds de calidad
    min_accuracy: float = 0.45
    good_accuracy: float = 0.55
    excellent_accuracy: float = 0.65

    # Features
    max_features: int = 50
    feature_selection: bool = True
    feature_importance_threshold: float = 0.01

    def get_model_params(self, framework: MLFramework) -> Dict[str, Any]:
        """Obtiene parámetros para un framework específico"""
        params_map = {
            MLFramework.XGBOOST: self.xgboost_params,
            MLFramework.LIGHTGBM: self.lightgbm_params,
            MLFramework.CATBOOST: self.catboost_params,
        }
        return params_map.get(framework, {})


@dataclass
class CacheConfig:
    """Configuración de sistema de cache"""

    enabled: bool = True
    backend: str = "memory"  # memory, redis, file

    # Cache en memoria
    max_size: int = 1000
    ttl_seconds: int = 3600  # 1 hora

    # Cache de archivos
    cache_dir: str = "cache"

    # Redis (si se usa)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = field(default_factory=lambda: os.getenv("REDIS_PASSWORD", ""))

    # TTL específicos por tipo
    ttl_predictions: int = 1800  # 30 minutos
    ttl_odds: int = 300  # 5 minutos
    ttl_team_stats: int = 3600  # 1 hora
    ttl_match_data: int = 86400  # 24 horas


@dataclass
class WebConfig:
    """Configuración del servidor web"""

    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    reload: bool = False

    # CORS
    cors_origins: List[str] = field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:8000",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8000",
        ]
    )
    cors_methods: List[str] = field(
        default_factory=lambda: ["GET", "POST", "PUT", "DELETE"]
    )
    cors_headers: List[str] = field(default_factory=lambda: ["*"])

    # WebSocket
    websocket_host: str = "localhost"
    websocket_port: int = 8765

    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds

    # Security
    secret_key: str = field(
        default_factory=lambda: os.getenv(
            "SECRET_KEY", "football-analytics-secret-key-2024"
        )
    )
    access_token_expire_minutes: int = 30

    # File uploads
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    upload_dir: str = "uploads"


@dataclass
class LoggingConfig:
    """Configuración de logging"""

    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"

    # Archivos de log
    log_dir: str = "logs"
    main_log_file: str = "football_analytics.log"
    error_log_file: str = "errors.log"
    api_log_file: str = "api.log"

    # Rotación de logs
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5

    # Logging específico
    log_sql_queries: bool = False
    log_api_requests: bool = True
    log_predictions: bool = True
    log_errors_to_file: bool = True

    # Loggers específicos
    logger_levels: Dict[str, str] = field(
        default_factory=lambda: {
            "football_analytics": "INFO",
            "football_analytics.api": "INFO",
            "football_analytics.ml": "INFO",
            "football_analytics.services": "INFO",
            "sqlalchemy.engine": "WARNING",
            "uvicorn": "INFO",
        }
    )


@dataclass
class MonitoringConfig:
    """Configuración de monitoreo y métricas"""

    enabled: bool = True

    # Health checks
    health_check_interval: int = 60  # seconds
    component_timeout: int = 10  # seconds

    # Métricas
    collect_metrics: bool = True
    metrics_retention_days: int = 30

    # Performance monitoring
    track_response_times: bool = True
    track_memory_usage: bool = True
    track_cpu_usage: bool = True

    # Alertas
    enable_alerts: bool = False
    alert_email: str = field(default_factory=lambda: os.getenv("ALERT_EMAIL", ""))
    alert_webhook: str = field(default_factory=lambda: os.getenv("ALERT_WEBHOOK", ""))

    # Thresholds para alertas
    max_response_time_ms: int = 5000
    max_memory_usage_mb: int = 1024
    max_cpu_usage_percent: int = 80
    min_accuracy_threshold: float = 0.40


# ================================
# CONFIGURACIÓN PRINCIPAL
# ================================


@dataclass
class FootballAnalyticsConfig:
    """Configuración principal del sistema Football Analytics"""

    # Información del sistema
    app_name: str = "Football Analytics"
    version: str = "2.1.0"
    description: str = "Sistema avanzado de análisis y predicción deportiva"
    build_date: str = "2024-06-02"

    # Entorno
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = True

    # Directorios principales
    base_dir: str = "/Users/miguelantonio/Desktop/football-analytics/backend"
    app_dir: str = "/Users/miguelantonio/Desktop/football-analytics/backend/app"
    data_dir: str = "/Users/miguelantonio/Desktop/football-analytics/backend/data"
    logs_dir: str = "/Users/miguelantonio/Desktop/football-analytics/backend/logs"
    models_dir: str = "/Users/miguelantonio/Desktop/football-analytics/backend/models"
    cache_dir: str = "/Users/miguelantonio/Desktop/football-analytics/backend/cache"
    backups_dir: str = "/Users/miguelantonio/Desktop/football-analytics/backend/backups"

    # Configuraciones específicas
    api: APIConfig = field(default_factory=APIConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    web: WebConfig = field(default_factory=WebConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # Timezone
    timezone: str = "UTC"

    def __post_init__(self):
        """Configuración post-inicialización"""
        # Ajustar configuración según entorno
        if self.environment == Environment.PRODUCTION:
            self.debug = False
            self.web.debug = False
            self.web.reload = False
            self.logging.level = LogLevel.INFO
            self.monitoring.enabled = True
        elif self.environment == Environment.DEVELOPMENT:
            self.debug = True
            self.web.debug = True
            self.web.reload = True
            self.logging.level = LogLevel.DEBUG
            self.monitoring.enabled = False

        # Actualizar paths relativos
        self._update_relative_paths()

        # Crear directorios si no existen
        self._ensure_directories_exist()

    def _update_relative_paths(self):
        """Actualiza paths relativos basados en base_dir"""
        self.database.sqlite_path = os.path.join(self.data_dir, "football_analytics.db")
        self.ml.models_dir = self.models_dir
        self.ml.model_cache_dir = os.path.join(self.models_dir, "cache")
        self.ml.training_data_dir = os.path.join(self.data_dir, "training")
        self.cache.cache_dir = self.cache_dir
        self.logging.log_dir = self.logs_dir
        self.web.upload_dir = os.path.join(self.data_dir, "uploads")

    def _ensure_directories_exist(self):
        """Crea directorios necesarios si no existen"""
        directories = [
            self.data_dir,
            self.logs_dir,
            self.models_dir,
            self.cache_dir,
            self.backups_dir,
            self.ml.model_cache_dir,
            self.ml.training_data_dir,
            self.web.upload_dir,
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def get_database_url(self) -> str:
        """Obtiene URL de conexión a la base de datos"""
        return self.database.get_connection_string()

    def get_api_headers(self, api_name: str) -> Dict[str, str]:
        """Obtiene headers para API específica"""
        return self.api.get_headers(api_name)

    def is_production(self) -> bool:
        """Verifica si está en entorno de producción"""
        return self.environment == Environment.PRODUCTION

    def is_development(self) -> bool:
        """Verifica si está en entorno de desarrollo"""
        return self.environment == Environment.DEVELOPMENT

    def to_dict(self) -> Dict[str, Any]:
        """Convierte configuración a diccionario"""

        def convert_value(value):
            if hasattr(value, "__dict__"):
                return {k: convert_value(v) for k, v in value.__dict__.items()}
            elif isinstance(value, Enum):
                return value.value
            elif isinstance(value, (list, tuple)):
                return [convert_value(item) for item in value]
            else:
                return value

        return convert_value(self)

    def save_to_file(self, filepath: str):
        """Guarda configuración a archivo JSON"""
        config_dict = self.to_dict()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, filepath: str) -> "FootballAnalyticsConfig":
        """Carga configuración desde archivo JSON"""
        with open(filepath, encoding="utf-8") as f:
            config_dict = json.load(f)

        # Reconstruir la configuración (implementación simplificada)
        # En producción se usaría una biblioteca como dacite o similar
        return cls(**config_dict)


# ================================
# CONFIGURACIONES POR ENTORNO
# ================================


def get_development_config() -> FootballAnalyticsConfig:
    """Configuración para desarrollo"""
    config = FootballAnalyticsConfig()
    config.environment = Environment.DEVELOPMENT
    config.debug = True
    config.web.debug = True
    config.web.reload = True
    config.logging.level = LogLevel.DEBUG
    config.database.echo = True
    config.monitoring.enabled = False
    return config


def get_testing_config() -> FootballAnalyticsConfig:
    """Configuración para testing"""
    config = FootballAnalyticsConfig()
    config.environment = Environment.TESTING
    config.debug = False
    config.database.database = "football_analytics_test"
    config.database.sqlite_path = "data/test_football_analytics.db"
    config.logging.level = LogLevel.WARNING
    config.cache.enabled = False
    config.monitoring.enabled = False
    return config


def get_staging_config() -> FootballAnalyticsConfig:
    """Configuración para staging"""
    config = FootballAnalyticsConfig()
    config.environment = Environment.STAGING
    config.debug = False
    config.web.debug = False
    config.web.reload = False
    config.logging.level = LogLevel.INFO
    config.monitoring.enabled = True
    return config


def get_production_config() -> FootballAnalyticsConfig:
    """Configuración para producción"""
    config = FootballAnalyticsConfig()
    config.environment = Environment.PRODUCTION
    config.debug = False
    config.web.debug = False
    config.web.reload = False
    config.web.host = "0.0.0.0"
    config.logging.level = LogLevel.INFO
    config.logging.log_sql_queries = False
    config.database.echo = False
    config.monitoring.enabled = True
    config.monitoring.enable_alerts = True
    return config


# ================================
# FACTORY DE CONFIGURACIÓN
# ================================


class ConfigFactory:
    """Factory para crear configuraciones según el entorno"""

    _configs = {
        Environment.DEVELOPMENT: get_development_config,
        Environment.TESTING: get_testing_config,
        Environment.STAGING: get_staging_config,
        Environment.PRODUCTION: get_production_config,
    }

    @classmethod
    def create(cls, environment: Optional[str] = None) -> FootballAnalyticsConfig:
        """Crea configuración para el entorno especificado"""
        if environment is None:
            environment = os.getenv("ENVIRONMENT", "development")

        env_enum = Environment(environment.lower())
        config_func = cls._configs.get(env_enum, get_development_config)
        return config_func()

    @classmethod
    def create_from_env(cls) -> FootballAnalyticsConfig:
        """Crea configuración basada en variables de entorno"""
        environment = os.getenv("ENVIRONMENT", "development")
        return cls.create(environment)


# ================================
# CONFIGURACIÓN GLOBAL
# ================================

# Instancia global de configuración
_global_config: Optional[FootballAnalyticsConfig] = None


def get_config() -> FootballAnalyticsConfig:
    """Obtiene la configuración global"""
    global _global_config
    if _global_config is None:
        _global_config = ConfigFactory.create_from_env()
    return _global_config


def set_config(config: FootballAnalyticsConfig):
    """Establece la configuración global"""
    global _global_config
    _global_config = config


def reload_config():
    """Recarga la configuración desde variables de entorno"""
    global _global_config
    _global_config = ConfigFactory.create_from_env()


# ================================
# VALIDADORES DE CONFIGURACIÓN
# ================================


def validate_config(config: FootballAnalyticsConfig) -> List[str]:
    """Valida la configuración y retorna lista de errores"""
    errors = []

    # Validar API keys
    if not config.api.football_data_api_key:
        errors.append("Football-Data API key no está configurada")

    # Validar directorios
    required_dirs = [config.data_dir, config.logs_dir, config.models_dir]

    for directory in required_dirs:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception as e:
                errors.append(f"No se puede crear directorio {directory}: {e}")

    # Validar configuración de ML
    if config.ml.test_size <= 0 or config.ml.test_size >= 1:
        errors.append("test_size debe estar entre 0 y 1")

    if config.ml.min_accuracy < 0 or config.ml.min_accuracy > 1:
        errors.append("min_accuracy debe estar entre 0 y 1")

    # Validar configuración web
    if config.web.port < 1 or config.web.port > 65535:
        errors.append("Puerto web debe estar entre 1 y 65535")

    return errors


def print_config_summary(config: FootballAnalyticsConfig):
    """Imprime resumen de la configuración"""
    print(
        f"""
🔧 CONFIGURACIÓN FOOTBALL ANALYTICS
{'='*50}
📦 Aplicación: {config.app_name} v{config.version}
🌍 Entorno: {config.environment.value}
🐛 Debug: {config.debug}
📊 Base de datos: {config.database.type.value}
🤖 Modelo ML por defecto: {config.ml.default_model.value}
🌐 Servidor: {config.web.host}:{config.web.port}
📝 Nivel de log: {config.logging.level.value}
💾 Cache habilitado: {config.cache.enabled}
📈 Monitoreo habilitado: {config.monitoring.enabled}
🔑 API Football-Data: {'✅ Configurada' if config.api.football_data_api_key else '❌ No configurada'}
📁 Directorio base: {config.base_dir}
{'='*50}
    """
    )


# ================================
# CONFIGURACIÓN PARA DESARROLLO LOCAL
# ================================

if __name__ == "__main__":
    # Ejecutar para probar configuración
    config = get_config()
    print_config_summary(config)

    # Validar configuración
    errors = validate_config(config)
    if errors:
        print("❌ ERRORES DE CONFIGURACIÓN:")
        for error in errors:
            print(f"   - {error}")
    else:
        print("✅ Configuración válida")

    # Guardar configuración de ejemplo
    config.save_to_file("config_example.json")
    print("📄 Configuración guardada en config_example.json")
