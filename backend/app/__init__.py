"""
=============================================================================
FOOTBALL ANALYTICS - BACKEND PACKAGE INITIALIZATION
=============================================================================
Inicialización del paquete backend para la plataforma de análisis de fútbol

Este módulo:
- Inicializa el sistema de logging
- Configura las importaciones principales
- Establece la versión del backend
- Expone APIs y servicios principales
- Configura el entorno de la aplicación
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Agregar el directorio raíz al path para importaciones
backend_root = Path(__file__).parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

# =============================================================================
# INFORMACIÓN DE LA APLICACIÓN
# =============================================================================
__version__ = "1.0.0"
__title__ = "Football Analytics Backend"
__description__ = "Sistema backend para análisis de fútbol con predicciones AI y datos en tiempo real"
__author__ = "Miguel Antonio"
__license__ = "MIT"

# =============================================================================
# CONFIGURACIÓN DEL ENTORNO
# =============================================================================
# Detectar entorno de ejecución
ENVIRONMENT = os.getenv('NODE_ENV', 'development')
DEBUG = ENVIRONMENT == 'development'
TESTING = os.getenv('TESTING', 'false').lower() == 'true'

# Configuración de la aplicación
APP_CONFIG = {
    'environment': ENVIRONMENT,
    'debug': DEBUG,
    'testing': TESTING,
    'version': __version__,
    'api_prefix': '/api',
    'cors_origins': os.getenv('CORS_ORIGINS', 'http://localhost:3000').split(','),
    'host': os.getenv('HOST', '0.0.0.0'),
    'port': int(os.getenv('PORT', 3001)),
    'websocket_port': int(os.getenv('WS_PORT', 3002))
}

# =============================================================================
# INICIALIZACIÓN DEL SISTEMA DE LOGGING
# =============================================================================
try:
    from app.utils.logger_config import setup_logging, get_football_logger
    
    # Configurar logging al importar el paquete
    _logger_system = setup_logging()
    logger = get_football_logger('main')
    
    logger.info(f"🚀 Football Analytics Backend v{__version__} inicializando...")
    logger.info(f"📊 Entorno: {ENVIRONMENT}")
    logger.info(f"🔧 Debug Mode: {DEBUG}")
    
except ImportError as e:
    # Fallback si no está disponible el sistema de logging personalizado
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️  Sistema de logging personalizado no disponible: {e}")
    logger.info(f"🚀 Football Analytics Backend v{__version__} inicializando con logging básico...")

# =============================================================================
# IMPORTACIONES PRINCIPALES DEL BACKEND
# =============================================================================
# Importaciones condicionales para evitar errores si no existen los módulos

# Configuración y utilidades
try:
    from app.config import Config, get_config
    logger.debug("✅ Configuración cargada")
except ImportError:
    logger.warning("⚠️  Módulo de configuración no encontrado")
    Config = None
    get_config = None

# Base de datos
try:
    from app.database import db, init_db
    logger.debug("✅ Módulo de base de datos disponible")
except ImportError:
    logger.warning("⚠️  Módulo de base de datos no encontrado")
    db = None
    init_db = None

# Servicios de APIs externas
try:
    from app.services.football_api import FootballAPIService
    from app.services.prediction_service import PredictionService
    logger.debug("✅ Servicios de APIs cargados")
except ImportError:
    logger.warning("⚠️  Servicios de APIs no encontrados")
    FootballAPIService = None
    PredictionService = None

# WebSocket service
try:
    from app.services.websocket_service import WebSocketService
    logger.debug("✅ Servicio WebSocket disponible")
except ImportError:
    logger.warning("⚠️  Servicio WebSocket no encontrado")
    WebSocketService = None

# Modelos de datos
try:
    from app.models import User, Match, Player, Team, League, Injury, Prediction
    logger.debug("✅ Modelos de datos cargados")
except ImportError:
    logger.warning("⚠️  Modelos de datos no encontrados")
    User = Match = Player = Team = League = Injury = Prediction = None

# APIs/Endpoints
try:
    from app.api import api_blueprint
    logger.debug("✅ Blueprint de APIs disponible")
except ImportError:
    logger.warning("⚠️  Blueprint de APIs no encontrado")
    api_blueprint = None

# Middleware y autenticación
try:
    from app.middleware.auth import AuthMiddleware
    from app.middleware.cors import setup_cors
    logger.debug("✅ Middleware cargado")
except ImportError:
    logger.warning("⚠️  Middleware no encontrado")
    AuthMiddleware = None
    setup_cors = None

# =============================================================================
# CONFIGURACIÓN DE LA API DE FÚTBOL
# =============================================================================
# Configurar servicio de API de fútbol con la clave del usuario
FOOTBALL_API_CONFIG = {
    'api_key': os.getenv('FOOTBALL_API_KEY', '9c9a42cbff2e8eb387eac2755c5e1e97'),
    'base_url': os.getenv('FOOTBALL_API_URL', 'https://api.football-data.org/v4'),
    'rate_limit': int(os.getenv('API_RATE_LIMIT_REQUESTS', 100)),
    'rate_window': int(os.getenv('API_RATE_LIMIT_WINDOW', 3600)),
    'timeout': int(os.getenv('API_TIMEOUT', 30)),
    'retries': int(os.getenv('API_RETRIES', 3))
}

# Validar configuración de API
if FOOTBALL_API_CONFIG['api_key'] and FOOTBALL_API_CONFIG['api_key'] != 'your_api_key_here':
    logger.info("✅ API Key de fútbol configurada correctamente")
    # Enmascarar la clave en logs
    masked_key = FOOTBALL_API_CONFIG['api_key'][:8] + '***'
    logger.debug(f"🔑 API Key: {masked_key}")
else:
    logger.error("❌ API Key de fútbol no configurada - Verificar .env")

# =============================================================================
# CONFIGURACIÓN DE BASE DE DATOS
# =============================================================================
DATABASE_CONFIG = {
    'url': os.getenv('DATABASE_URL'),
    'pool_size': int(os.getenv('DB_POOL_SIZE', 10)),
    'max_overflow': int(os.getenv('DB_MAX_OVERFLOW', 20)),
    'pool_timeout': int(os.getenv('DB_POOL_TIMEOUT', 30)),
    'echo': DEBUG  # SQL logging solo en desarrollo
}

# Validar configuración de base de datos
if DATABASE_CONFIG['url']:
    logger.info("✅ Configuración de base de datos detectada")
else:
    logger.warning("⚠️  URL de base de datos no configurada")

# =============================================================================
# CONFIGURACIÓN DE CACHE/REDIS
# =============================================================================
REDIS_CONFIG = {
    'url': os.getenv('REDIS_URL', 'redis://localhost:6379'),
    'host': os.getenv('REDIS_HOST', 'localhost'),
    'port': int(os.getenv('REDIS_PORT', 6379)),
    'password': os.getenv('REDIS_PASSWORD'),
    'db': int(os.getenv('REDIS_DB', 0)),
    'decode_responses': True,
    'socket_timeout': int(os.getenv('REDIS_TIMEOUT', 5))
}

# =============================================================================
# INICIALIZACIÓN DE SERVICIOS PRINCIPALES
# =============================================================================
def initialize_services() -> Dict[str, Any]:
    """
    Inicializar servicios principales del backend
    
    Returns:
        Dict con instancias de servicios inicializados
    """
    services = {}
    
    try:
        # Inicializar servicio de API de fútbol
        if FootballAPIService:
            services['football_api'] = FootballAPIService(FOOTBALL_API_CONFIG)
            logger.info("✅ Servicio de API de fútbol inicializado")
        
        # Inicializar servicio de predicciones
        if PredictionService:
            services['predictions'] = PredictionService()
            logger.info("✅ Servicio de predicciones inicializado")
        
        # Inicializar servicio WebSocket
        if WebSocketService:
            services['websocket'] = WebSocketService(APP_CONFIG['websocket_port'])
            logger.info("✅ Servicio WebSocket inicializado")
        
        logger.info(f"✅ {len(services)} servicios inicializados correctamente")
        
    except Exception as e:
        logger.error(f"❌ Error inicializando servicios: {str(e)}")
    
    return services


def create_app(config: Optional[Dict[str, Any]] = None):
    """
    Factory function para crear la aplicación Flask/FastAPI
    
    Args:
        config: Configuración adicional opcional
    
    Returns:
        Instancia de la aplicación configurada
    """
    try:
        # Importar Flask o FastAPI dependiendo de lo que esté disponible
        try:
            from flask import Flask
            app = Flask(__name__)
            framework = 'Flask'
        except ImportError:
            try:
                from fastapi import FastAPI
                app = FastAPI(
                    title=__title__,
                    description=__description__,
                    version=__version__
                )
                framework = 'FastAPI'
            except ImportError:
                logger.error("❌ Ni Flask ni FastAPI están disponibles")
                return None
        
        logger.info(f"✅ Aplicación {framework} creada")
        
        # Aplicar configuración
        if config:
            APP_CONFIG.update(config)
        
        # Configurar CORS si está disponible
        if setup_cors:
            setup_cors(app, APP_CONFIG['cors_origins'])
            logger.info("✅ CORS configurado")
        
        # Registrar blueprints si están disponibles
        if api_blueprint and hasattr(app, 'register_blueprint'):
            app.register_blueprint(api_blueprint, url_prefix=APP_CONFIG['api_prefix'])
            logger.info("✅ API Blueprint registrado")
        
        # Inicializar base de datos si está disponible
        if init_db:
            init_db(app, DATABASE_CONFIG)
            logger.info("✅ Base de datos inicializada")
        
        # Inicializar servicios
        app.services = initialize_services()
        
        logger.info(f"🚀 Aplicación {framework} lista para ejecutar")
        return app
        
    except Exception as e:
        logger.error(f"❌ Error creando aplicación: {str(e)}")
        return None


# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================
def get_app_info() -> Dict[str, str]:
    """
    Obtener información de la aplicación
    
    Returns:
        Diccionario con información de la app
    """
    return {
        'name': __title__,
        'version': __version__,
        'description': __description__,
        'environment': ENVIRONMENT,
        'debug': str(DEBUG),
        'api_prefix': APP_CONFIG['api_prefix'],
        'host': APP_CONFIG['host'],
        'port': str(APP_CONFIG['port']),
        'websocket_port': str(APP_CONFIG['websocket_port'])
    }


def health_check() -> Dict[str, Any]:
    """
    Health check de la aplicación
    
    Returns:
        Estado de salud de los servicios
    """
    health = {
        'status': 'healthy',
        'timestamp': os.environ.get('BUILD_TIME', 'unknown'),
        'version': __version__,
        'environment': ENVIRONMENT,
        'services': {}
    }
    
    # Verificar servicios disponibles
    try:
        # Verificar base de datos
        if db:
            health['services']['database'] = 'available'
        else:
            health['services']['database'] = 'not_configured'
        
        # Verificar API de fútbol
        if FOOTBALL_API_CONFIG['api_key'] != 'your_api_key_here':
            health['services']['football_api'] = 'configured'
        else:
            health['services']['football_api'] = 'not_configured'
        
        # Verificar Redis
        health['services']['redis'] = 'configured' if REDIS_CONFIG['url'] else 'not_configured'
        
    except Exception as e:
        health['status'] = 'degraded'
        health['error'] = str(e)
        logger.warning(f"⚠️  Health check warning: {e}")
    
    return health


# =============================================================================
# EXPORTS PRINCIPALES
# =============================================================================
# Definir qué se exporta cuando se hace 'from app import *'
__all__ = [
    # Información de la aplicación
    '__version__',
    '__title__',
    '__description__',
    'APP_CONFIG',
    
    # Funciones principales
    'create_app',
    'initialize_services',
    'get_app_info',
    'health_check',
    
    # Configuraciones
    'FOOTBALL_API_CONFIG',
    'DATABASE_CONFIG',
    'REDIS_CONFIG',
    
    # Servicios (si están disponibles)
    'FootballAPIService',
    'PredictionService',
    'WebSocketService',
    
    # Modelos (si están disponibles)
    'User',
    'Match',
    'Player',
    'Team',
    'League',
    'Injury',
    'Prediction',
    
    # Base de datos
    'db',
    'init_db',
    
    # APIs
    'api_blueprint',
    
    # Middleware
    'AuthMiddleware',
    'setup_cors',
    
    # Logging
    'logger',
    'get_football_logger'
]

# =============================================================================
# INICIALIZACIÓN AUTOMÁTICA
# =============================================================================
# Log de inicialización completada
logger.info("🏆 Paquete Football Analytics Backend inicializado correctamente")

# Mostrar resumen de configuración en desarrollo
if DEBUG:
    logger.debug("🔧 Configuración de desarrollo:")
    logger.debug(f"   - Puerto: {APP_CONFIG['port']}")
    logger.debug(f"   - WebSocket: {APP_CONFIG['websocket_port']}")
    logger.debug(f"   - API Prefix: {APP_CONFIG['api_prefix']}")
    logger.debug(f"   - CORS Origins: {APP_CONFIG['cors_origins']}")

# Verificar dependencias críticas
missing_deps = []
if not FootballAPIService:
    missing_deps.append('FootballAPIService')
if not DATABASE_CONFIG['url']:
    missing_deps.append('Database URL')

if missing_deps:
    logger.warning(f"⚠️  Dependencias faltantes: {', '.join(missing_deps)}")
    logger.warning("📝 Revisar configuración en .env")
else:
    logger.info("✅ Todas las dependencias críticas están configuradas")

# =============================================================================
# INFORMACIÓN DE DEBUG PARA DESARROLLO
# =============================================================================
if DEBUG and __name__ == "__main__":
    print("\n" + "="*60)
    print("🏆 FOOTBALL ANALYTICS BACKEND")
    print("="*60)
    print(f"📊 Versión: {__version__}")
    print(f"🌍 Entorno: {ENVIRONMENT}")
    print(f"🔧 Debug: {DEBUG}")
    print(f"🔑 API Key: {'✅ Configurada' if FOOTBALL_API_CONFIG['api_key'] != 'your_api_key_here' else '❌ No configurada'}")
    print(f"🗄️  Base de datos: {'✅ Configurada' if DATABASE_CONFIG['url'] else '❌ No configurada'}")
    print(f"📦 Redis: {'✅ Configurado' if REDIS_CONFIG['url'] else '❌ No configurado'}")
    print("="*60)
    print("🚀 Listo para iniciar el servidor!")
    print("="*60 + "\n")