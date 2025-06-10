"""
=============================================================================
FOOTBALL ANALYTICS - CONFIGURACIÓN DE LOGGING
=============================================================================
Sistema de logging centralizado para el backend de Football Analytics
Incluye: logging de APIs, base de datos, WebSockets, ML, y monitoreo
"""

import json
import logging
import logging.handlers
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Imports específicos para integraciones
try:
    import colorama
    from colorama import Back, Fore, Style

    colorama.init()
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False

try:
    import structlog

    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False


class FootballAnalyticsLogger:
    """
    Sistema de logging personalizado para Football Analytics

    Características:
    - Logging estructurado con contexto
    - Diferentes niveles para diferentes componentes
    - Rotación automática de archivos
    - Integración con APIs externas
    - Logging específico para fútbol (partidos, jugadores, etc.)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializar el sistema de logging

        Args:
            config: Diccionario de configuración opcional
        """
        self.config = config or self._get_default_config()
        self.loggers = {}
        self._setup_directories()
        self._setup_formatters()
        self._setup_handlers()
        self._setup_loggers()

    def _get_default_config(self) -> Dict[str, Any]:
        """Obtener configuración por defecto del logging"""
        return {
            # Configuración general
            "log_level": os.getenv("LOG_LEVEL", "INFO").upper(),
            "log_dir": os.getenv("LOG_DIR", "logs"),
            "max_file_size": int(os.getenv("LOG_MAX_SIZE", "10")) * 1024 * 1024,  # 10MB
            "backup_count": int(os.getenv("LOG_MAX_FILES", "5")),
            "environment": os.getenv("NODE_ENV", "development"),
            # Configuración específica por componente
            "component_levels": {
                "api": "INFO",
                "database": "WARNING",
                "websocket": "INFO",
                "ml_service": "INFO",
                "football_api": "INFO",
                "auth": "WARNING",
                "scheduler": "INFO",
                "performance": "WARNING",
            },
            # Configuración de archivos de log
            "log_files": {
                "main": "football-analytics.log",
                "error": "error.log",
                "api": "api.log",
                "database": "database.log",
                "websocket": "websocket.log",
                "ml": "ml-service.log",
                "football_data": "football-data.log",
                "performance": "performance.log",
                "security": "security.log",
            },
            # Integraciones externas
            "sentry_dsn": os.getenv("SENTRY_DSN"),
            "enable_console": os.getenv("ENABLE_CONSOLE_LOGS", "true").lower()
            == "true",
            "enable_colors": HAS_COLORAMA and sys.stdout.isatty(),
            "enable_structured": HAS_STRUCTLOG,
            # Configuración específica de Football Analytics
            "football_config": {
                "api_key_prefix": (
                    os.getenv("FOOTBALL_API_KEY", "")[:8]
                    if os.getenv("FOOTBALL_API_KEY")
                    else "NOT_SET"
                ),
                "mask_sensitive_data": True,
                "log_api_responses": os.getenv("LOG_API_RESPONSES", "false").lower()
                == "true",
                "log_prediction_details": os.getenv(
                    "LOG_PREDICTION_DETAILS", "true"
                ).lower()
                == "true",
            },
        }

    def _setup_directories(self):
        """Crear directorios de logs si no existen"""
        log_dir = Path(self.config["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)

        # Crear subdirectorios para diferentes tipos de logs
        subdirs = ["api", "database", "ml", "websocket", "performance"]
        for subdir in subdirs:
            (log_dir / subdir).mkdir(exist_ok=True)

    def _setup_formatters(self):
        """Configurar formateadores de logs"""
        self.formatters = {}

        # Formateador principal con información completa
        self.formatters["detailed"] = logging.Formatter(
            "[%(asctime)s] %(levelname)s in %(name)s (%(filename)s:%(lineno)d): %(message)s"
        )

        # Formateador para consola con colores (si está disponible)
        if self.config["enable_colors"]:
            self.formatters["console"] = ColoredFormatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            )
        else:
            self.formatters["console"] = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            )

        # Formateador para API calls
        self.formatters["api"] = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(method)s %(endpoint)s - %(status_code)s - %(response_time)sms"
        )

        # Formateador para eventos de fútbol
        self.formatters["football"] = logging.Formatter(
            "[%(asctime)s] %(levelname)s - FOOTBALL: %(event_type)s - %(message)s"
        )

        # Formateador JSON para logs estructurados
        self.formatters["json"] = JsonFormatter()

    def _setup_handlers(self):
        """Configurar manejadores de logs"""
        self.handlers = {}
        log_dir = Path(self.config["log_dir"])

        # Handler principal para archivo general
        self.handlers["main_file"] = logging.handlers.RotatingFileHandler(
            log_dir / self.config["log_files"]["main"],
            maxBytes=self.config["max_file_size"],
            backupCount=self.config["backup_count"],
            encoding="utf-8",
        )
        self.handlers["main_file"].setFormatter(self.formatters["detailed"])

        # Handler para errores críticos
        self.handlers["error_file"] = logging.handlers.RotatingFileHandler(
            log_dir / self.config["log_files"]["error"],
            maxBytes=self.config["max_file_size"],
            backupCount=self.config["backup_count"],
            encoding="utf-8",
        )
        self.handlers["error_file"].setFormatter(self.formatters["detailed"])
        self.handlers["error_file"].setLevel(logging.ERROR)

        # Handler para consola
        if self.config["enable_console"]:
            self.handlers["console"] = logging.StreamHandler(sys.stdout)
            self.handlers["console"].setFormatter(self.formatters["console"])

        # Handlers específicos para componentes
        component_handlers = {
            "api": "api.log",
            "database": "database.log",
            "websocket": "websocket.log",
            "ml": "ml-service.log",
            "football_data": "football-data.log",
            "performance": "performance.log",
            "security": "security.log",
        }

        for component, filename in component_handlers.items():
            handler = logging.handlers.RotatingFileHandler(
                log_dir / filename,
                maxBytes=self.config["max_file_size"],
                backupCount=self.config["backup_count"],
                encoding="utf-8",
            )
            handler.setFormatter(self.formatters["detailed"])
            self.handlers[f"{component}_file"] = handler

    def _setup_loggers(self):
        """Configurar loggers específicos"""
        # Logger principal
        self.main_logger = self._create_logger(
            "football_analytics",
            level=self.config["log_level"],
            handlers=["main_file", "error_file", "console"],
        )

        # Loggers específicos por componente
        components = [
            "api",
            "database",
            "websocket",
            "ml_service",
            "football_api",
            "auth",
            "scheduler",
            "performance",
        ]

        for component in components:
            level = self.config["component_levels"].get(component, "INFO")
            handlers = [f"{component}_file", "error_file"]
            if self.config["enable_console"]:
                handlers.append("console")

            self.loggers[component] = self._create_logger(
                f"football_analytics.{component}", level=level, handlers=handlers
            )

    def _create_logger(self, name: str, level: str, handlers: list) -> logging.Logger:
        """Crear un logger con configuración específica"""
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level))

        # Limpiar handlers existentes
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Agregar handlers especificados
        for handler_name in handlers:
            if handler_name in self.handlers:
                logger.addHandler(self.handlers[handler_name])

        # Evitar propagación para evitar duplicados
        logger.propagate = False

        return logger

    def get_logger(self, component: str = "main") -> logging.Logger:
        """
        Obtener logger para un componente específico

        Args:
            component: Nombre del componente ('api', 'database', 'websocket', etc.)

        Returns:
            Logger configurado para el componente
        """
        if component == "main":
            return self.main_logger

        return self.loggers.get(component, self.main_logger)

    def log_api_call(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        response_time: float,
        error: Optional[str] = None,
    ):
        """
        Logging específico para llamadas API

        Args:
            method: Método HTTP (GET, POST, etc.)
            endpoint: Endpoint llamado
            status_code: Código de respuesta HTTP
            response_time: Tiempo de respuesta en ms
            error: Mensaje de error si existe
        """
        logger = self.get_logger("api")

        # Crear record personalizado para API
        record = logging.LogRecord(
            name=logger.name,
            level=logging.INFO if status_code < 400 else logging.ERROR,
            pathname="",
            lineno=0,
            msg=f"{method} {endpoint}",
            args=(),
            exc_info=None,
        )

        # Agregar información adicional
        record.method = method
        record.endpoint = endpoint
        record.status_code = status_code
        record.response_time = response_time

        if error:
            record.levelno = logging.ERROR
            record.levelname = "ERROR"
            record.msg = f"{method} {endpoint} - ERROR: {error}"

        logger.handle(record)

    def log_football_event(self, event_type: str, event_data: Dict[str, Any]):
        """
        Logging específico para eventos de fútbol

        Args:
            event_type: Tipo de evento ('match_start', 'goal_scored', etc.)
            event_data: Datos del evento
        """
        logger = self.get_logger("football_api")

        # Enmascarar datos sensibles si está habilitado
        if self.config["football_config"]["mask_sensitive_data"]:
            event_data = self._mask_sensitive_data(event_data)

        record = logging.LogRecord(
            name=logger.name,
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=json.dumps(event_data, ensure_ascii=False),
            args=(),
            exc_info=None,
        )

        record.event_type = event_type
        logger.handle(record)

    def log_ml_prediction(
        self,
        match_id: int,
        prediction: Dict[str, Any],
        confidence: float,
        model_version: str,
    ):
        """
        Logging específico para predicciones de ML

        Args:
            match_id: ID del partido
            prediction: Predicción generada
            confidence: Nivel de confianza
            model_version: Versión del modelo usado
        """
        logger = self.get_logger("ml_service")

        if self.config["football_config"]["log_prediction_details"]:
            logger.info(
                f"ML Prediction - Match: {match_id}, "
                f"Result: {prediction}, Confidence: {confidence:.2f}, "
                f"Model: {model_version}"
            )
        else:
            logger.info(f"ML Prediction generated for match {match_id}")

    def log_database_operation(
        self, operation: str, table: str, execution_time: float, rows_affected: int = 0
    ):
        """
        Logging específico para operaciones de base de datos

        Args:
            operation: Tipo de operación (SELECT, INSERT, UPDATE, DELETE)
            table: Tabla afectada
            execution_time: Tiempo de ejecución en ms
            rows_affected: Número de filas afectadas
        """
        logger = self.get_logger("database")

        if execution_time > 1000:  # Log slow queries (>1s)
            logger.warning(
                f"SLOW QUERY - {operation} on {table}: {execution_time:.2f}ms, "
                f"rows: {rows_affected}"
            )
        else:
            logger.debug(
                f"DB - {operation} on {table}: {execution_time:.2f}ms, "
                f"rows: {rows_affected}"
            )

    def log_websocket_event(
        self, event: str, client_id: str, data: Optional[Dict] = None
    ):
        """
        Logging específico para eventos WebSocket

        Args:
            event: Tipo de evento ('connect', 'disconnect', 'message', etc.)
            client_id: ID del cliente
            data: Datos adicionales del evento
        """
        logger = self.get_logger("websocket")

        message = f"WebSocket {event} - Client: {client_id}"
        if data:
            message += f" - Data: {json.dumps(data, ensure_ascii=False)}"

        logger.info(message)

    def log_performance_metric(
        self,
        metric_name: str,
        value: float,
        unit: str = "ms",
        threshold: Optional[float] = None,
    ):
        """
        Logging específico para métricas de performance

        Args:
            metric_name: Nombre de la métrica
            value: Valor de la métrica
            unit: Unidad de medida
            threshold: Umbral para warning
        """
        logger = self.get_logger("performance")

        level = logging.INFO
        if threshold and value > threshold:
            level = logging.WARNING

        logger.log(level, f"METRIC - {metric_name}: {value}{unit}")

    def log_security_event(
        self,
        event_type: str,
        ip_address: str,
        user_id: Optional[str] = None,
        details: Optional[str] = None,
    ):
        """
        Logging específico para eventos de seguridad

        Args:
            event_type: Tipo de evento de seguridad
            ip_address: Dirección IP
            user_id: ID del usuario (si aplica)
            details: Detalles adicionales
        """
        logger = self.get_logger("security")  # Usa un logger dedicado para seguridad

        message = f"SECURITY - {event_type} from {ip_address}"
        if user_id:
            message += f" (User: {user_id})"
        if details:
            message += f" - {details}"

        logger.warning(message)

    def _mask_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enmascarar datos sensibles en logs"""
        masked_data = data.copy()
        sensitive_keys = ["password", "token", "api_key", "secret", "auth"]

        for key, value in masked_data.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                if isinstance(value, str) and len(value) > 8:
                    masked_data[key] = value[:4] + "***" + value[-4:]
                else:
                    masked_data[key] = "***"

        return masked_data


class ColoredFormatter(logging.Formatter):
    """Formateador con colores para consola"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if HAS_COLORAMA:
            self.COLORS = {
                "DEBUG": Fore.CYAN,
                "INFO": Fore.GREEN,
                "WARNING": Fore.YELLOW,
                "ERROR": Fore.RED,
                "CRITICAL": Fore.RED + Back.WHITE + Style.BRIGHT,
            }
        else:
            self.COLORS = {}

    def format(self, record):
        if HAS_COLORAMA and record.levelname in self.COLORS:
            record.levelname = (
                self.COLORS[record.levelname] + record.levelname + Style.RESET_ALL
            )

        return super().format(record)


class JsonFormatter(logging.Formatter):
    """Formateador JSON para logs estructurados"""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Agregar información de excepción si existe
        if record.exc_info:
            log_entry["exception"] = traceback.format_exception(*record.exc_info)

        # Agregar campos personalizados si existen
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "lineno",
                "getMessage",
            ]:
                log_entry[key] = value

        return json.dumps(log_entry, ensure_ascii=False)


# =============================================================================
# INSTANCIA GLOBAL DEL LOGGER
# =============================================================================
# Crear instancia global del sistema de logging
_logger_instance = None


def get_football_logger(component: str = "main") -> logging.Logger:
    """
    Obtener logger para Football Analytics

    Args:
        component: Componente específico ('api', 'database', 'websocket', etc.)

    Returns:
        Logger configurado
    """
    global _logger_instance

    if _logger_instance is None:
        _logger_instance = FootballAnalyticsLogger()

    return _logger_instance.get_logger(component)


def setup_logging(config: Optional[Dict[str, Any]] = None) -> FootballAnalyticsLogger:
    """
    Configurar sistema de logging para Football Analytics

    Args:
        config: Configuración personalizada

    Returns:
        Instancia del sistema de logging
    """
    global _logger_instance
    _logger_instance = FootballAnalyticsLogger(config)
    return _logger_instance


# =============================================================================
# FUNCIONES DE CONVENIENCIA
# =============================================================================
def log_api_call(
    method: str,
    endpoint: str,
    status_code: int,
    response_time: float,
    error: Optional[str] = None,
):
    """Función de conveniencia para logging de API calls"""
    logger_instance = _logger_instance or FootballAnalyticsLogger()
    logger_instance.log_api_call(method, endpoint, status_code, response_time, error)


def log_football_event(event_type: str, event_data: Dict[str, Any]):
    """Función de conveniencia para logging de eventos de fútbol"""
    logger_instance = _logger_instance or FootballAnalyticsLogger()
    logger_instance.log_football_event(event_type, event_data)


def log_ml_prediction(
    match_id: int, prediction: Dict[str, Any], confidence: float, model_version: str
):
    """Función de conveniencia para logging de predicciones ML"""
    logger_instance = _logger_instance or FootballAnalyticsLogger()
    logger_instance.log_ml_prediction(match_id, prediction, confidence, model_version)


def log_performance_metric(
    metric_name: str, value: float, unit: str = "ms", threshold: Optional[float] = None
):
    """Función de conveniencia para logging de métricas de performance"""
    logger_instance = _logger_instance or FootballAnalyticsLogger()
    logger_instance.log_performance_metric(metric_name, value, unit, threshold)


# =============================================================================
# DECORADOR PARA LOGGING AUTOMÁTICO
# =============================================================================
def log_execution_time(component: str = "main", threshold_ms: float = 1000):
    """
    Decorador para logging automático de tiempo de ejecución

    Args:
        component: Componente para logging
        threshold_ms: Umbral en ms para generar warning
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            import time

            start_time = time.time()
            logger = get_football_logger(component)

            try:
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000

                if execution_time > threshold_ms:
                    logger.warning(
                        f"SLOW EXECUTION - {func.__name__}: {execution_time:.2f}ms"
                    )
                else:
                    logger.debug(f"EXECUTION - {func.__name__}: {execution_time:.2f}ms")

                return result

            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                logger.error(
                    f"ERROR in {func.__name__} after {execution_time:.2f}ms: {str(e)}"
                )
                raise

        return wrapper

    return decorator


# =============================================================================
# CONFIGURACIÓN DE EJEMPLO PARA DESARROLLO
# =============================================================================
if __name__ == "__main__":
    # Ejemplo de uso para testing
    logger_system = setup_logging()

    # Test de diferentes tipos de logging
    main_logger = get_football_logger("main")
    main_logger.info("Sistema de logging inicializado para Football Analytics")

    # Test API logging
    log_api_call("GET", "/api/matches", 200, 150.5)
    log_api_call("POST", "/api/auth/login", 401, 89.2, "Invalid credentials")

    # Test football event logging
    log_football_event(
        "goal_scored",
        {
            "match_id": 12345,
            "player_id": 67890,
            "team_id": 111,
            "minute": 67,
            "api_key": "9c9a42cbff2e8eb387eac2755c5e1e97",  # Será enmascarado
        },
    )

    # Test ML prediction logging
    log_ml_prediction(
        match_id=12345,
        prediction={"home_win": 0.6, "draw": 0.25, "away_win": 0.15},
        confidence=0.75,
        model_version="v1.2.0",
    )

    # Test performance metric
    log_performance_metric("database_query_time", 1250.5, "ms", 1000)

    print("✅ Sistema de logging configurado y probado correctamente")
