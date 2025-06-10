"""
Football Analytics - Services Package
Inicialización y exposición de todos los servicios del sistema
"""

import logging

# Solo importar servicios que existen
try:
    from .data_collector import DataCollectorService
except ImportError:
    DataCollectorService = None

try:
    from .live_tracker import LiveTrackerService
except ImportError:
    LiveTrackerService = None

try:
    from .odds_calculator import OddsCalculatorService
except ImportError:
    OddsCalculatorService = None

try:
    from .predictor import PredictorService
except ImportError:
    PredictorService = None

# Versión del paquete de servicios
__version__ = "1.0.0"

# Exportar servicios disponibles
__all__ = [
    "DataCollectorService",
    "LiveTrackerService",
    "OddsCalculatorService",
    "PredictorService",
]

# Registry de servicios para gestión centralizada
_service_registry = {}
_initialized = False


def initialize_services(config: dict = None) -> bool:
    """
    Inicializa todos los servicios del sistema (versión simplificada)
    """
    global _service_registry, _initialized

    try:
        _service_registry = {
            "data": "DataService-placeholder",
            "prediction": "PredictionService-placeholder",
            "analysis": "AnalysisService-placeholder",
            "betting": "BettingService-placeholder",
            "cache": "CacheService-placeholder",
            "model": "ModelService-placeholder",
        }

        _initialized = True

        logger = logging.getLogger(__name__)
        logger.info(f"✅ Servicios inicializados: {len(_service_registry)} servicios")

        return True

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"❌ Error inicializando servicios: {e}")
        return False


def get_service_status() -> dict:
    """Obtiene el estado de todos los servicios"""
    return {
        "initialized": _initialized,
        "services_count": len(_service_registry),
        "services": list(_service_registry.keys()) if _initialized else [],
    }


def shutdown_services() -> bool:
    """Apaga todos los servicios"""
    global _service_registry, _initialized

    try:
        _service_registry.clear()
        _initialized = False
        return True
    except Exception:
        return False


# Configuración de logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
