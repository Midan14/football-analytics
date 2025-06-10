"""
Database Package Initialization for Football Analytics

Este módulo contiene toda la lógica de base de datos para el sistema de análisis
de fútbol, incluyendo modelos SQLAlchemy, conexiones, migraciones y utilidades.
"""

import asyncio
import logging
from typing import Generator, Optional

from sqlalchemy import MetaData, create_engine, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================
# CONFIGURACIÓN BASE DE DATOS
# =====================================================

# Base declarativa para todos los modelos
Base = declarative_base()

# Metadatos para la base de datos
metadata = MetaData(
    naming_convention={
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    }
)

# Variables globales para conexión
engine = None
SessionLocal = None
database_url = None

# =====================================================
# FUNCIONES DE INICIALIZACIÓN
# =====================================================


def init_database(db_url: str, echo: bool = False) -> None:
    """
    Inicializar conexión a la base de datos.

    Args:
        db_url: URL de conexión a PostgreSQL
        echo: Si mostrar SQL queries en logs
    """
    global engine, SessionLocal, database_url

    try:
        database_url = db_url

        # Crear engine con configuración optimizada
        engine = create_engine(
            db_url,
            echo=echo,
            pool_size=20,  # Conexiones en pool
            max_overflow=30,  # Conexiones adicionales
            pool_timeout=30,  # Timeout para obtener conexión
            pool_recycle=3600,  # Reciclar conexiones cada hora
            pool_pre_ping=True,  # Verificar conexiones antes de usar
            connect_args={
                "options": "-c timezone=UTC",  # Usar UTC
                "application_name": "football_analytics",
            },
        )

        # Crear session factory
        SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=engine
        )

        logger.info(
            f"Database initialized successfully: {_mask_password(db_url)}"
        )

    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise


def create_all_tables() -> None:
    """
    Crear todas las tablas definidas en los modelos.
    """
    try:
        if engine is None:
            raise RuntimeError(
                "Database not initialized. Call init_database() first."
            )

        # Importar todos los modelos para que se registren
        from . import models

        # Crear todas las tablas
        Base.metadata.create_all(bind=engine)

        logger.info("All database tables created successfully")

    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        raise


def drop_all_tables() -> None:
    """
    Eliminar todas las tablas (usar con precaución).
    """
    try:
        if engine is None:
            raise RuntimeError("Database not initialized")

        # Importar modelos
        from . import models

        # Eliminar todas las tablas
        Base.metadata.drop_all(bind=engine)

        logger.warning("All database tables dropped")

    except Exception as e:
        logger.error(f"Error dropping tables: {e}")
        raise


def get_database_info() -> dict:
    """
    Obtener información sobre la base de datos.

    Returns:
        dict: Información de la base de datos
    """
    try:
        if engine is None:
            return {"status": "not_initialized"}

        # Inspeccionar base de datos
        inspector = inspect(engine)
        table_names = inspector.get_table_names()

        # Obtener información de conexión
        with engine.connect() as connection:
            result = connection.execute("SELECT version()")
            db_version = result.fetchone()[0]

        return {
            "status": "connected",
            "database_url": (
                _mask_password(database_url) if database_url else None
            ),
            "database_version": db_version,
            "total_tables": len(table_names),
            "table_names": sorted(table_names),
            "engine_info": {
                "pool_size": engine.pool.size(),
                "checked_in": engine.pool.checkedin(),
                "checked_out": engine.pool.checkedout(),
                "overflow": engine.pool.overflow(),
                "invalid": engine.pool.invalid(),
            },
        }

    except Exception as e:
        logger.error(f"Error getting database info: {e}")
        return {"status": "error", "error": str(e)}


# =====================================================
# GESTIÓN DE SESIONES
# =====================================================


def get_db() -> Generator[Session, None, None]:
    """
    Dependency para obtener sesión de base de datos en FastAPI.

    Yields:
        Session: Sesión de SQLAlchemy
    """
    if SessionLocal is None:
        raise RuntimeError(
            "Database not initialized. Call init_database() first."
        )

    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def create_session() -> Session:
    """
    Crear nueva sesión de base de datos.

    Returns:
        Session: Nueva sesión de SQLAlchemy
    """
    if SessionLocal is None:
        raise RuntimeError("Database not initialized")

    return SessionLocal()


class DatabaseManager:
    """
    Manager para operaciones avanzadas de base de datos.
    """

    @staticmethod
    def get_session() -> Session:
        """Obtener nueva sesión."""
        return create_session()

    @staticmethod
    def execute_raw_sql(sql: str, params: Optional[dict] = None) -> list:
        """
        Ejecutar SQL crudo.

        Args:
            sql: Query SQL
            params: Parámetros opcionales

        Returns:
            list: Resultados de la query
        """
        try:
            with engine.connect() as connection:
                if params:
                    result = connection.execute(sql, params)
                else:
                    result = connection.execute(sql)

                return result.fetchall()

        except Exception as e:
            logger.error(f"Error executing raw SQL: {e}")
            raise

    @staticmethod
    def get_table_row_count(table_name: str) -> int:
        """
        Obtener número de filas en una tabla.

        Args:
            table_name: Nombre de la tabla

        Returns:
            int: Número de filas
        """
        try:
            sql = f"SELECT COUNT(*) FROM {table_name}"
            result = DatabaseManager.execute_raw_sql(sql)
            return result[0][0] if result else 0

        except Exception as e:
            logger.error(f"Error getting row count for {table_name}: {e}")
            return 0

    @staticmethod
    def get_all_table_stats() -> dict:
        """
        Obtener estadísticas de todas las tablas.

        Returns:
            dict: Estadísticas por tabla
        """
        try:
            inspector = inspect(engine)
            table_names = inspector.get_table_names()

            stats = {}
            for table_name in table_names:
                stats[table_name] = {
                    "row_count": DatabaseManager.get_table_row_count(
                        table_name
                    ),
                    "columns": len(inspector.get_columns(table_name)),
                }

            return stats

        except Exception as e:
            logger.error(f"Error getting table stats: {e}")
            return {}

    @staticmethod
    def check_database_health() -> dict:
        """
        Verificar salud de la base de datos.

        Returns:
            dict: Estado de salud
        """
        try:
            # Test de conexión básico
            with engine.connect() as connection:
                start_time = time.time()
                connection.execute("SELECT 1")
                response_time = time.time() - start_time

            # Obtener estadísticas del pool
            pool_stats = {
                "size": engine.pool.size(),
                "checked_in": engine.pool.checkedin(),
                "checked_out": engine.pool.checkedout(),
                "overflow": engine.pool.overflow(),
                "invalid": engine.pool.invalid(),
            }

            return {
                "status": "healthy",
                "response_time_ms": round(response_time * 1000, 2),
                "pool_stats": pool_stats,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }


# =====================================================
# UTILIDADES
# =====================================================


def _mask_password(url: str) -> str:
    """
    Enmascarar password en URL de base de datos para logs.

    Args:
        url: URL de conexión

    Returns:
        str: URL con password enmascarado
    """
    try:
        if "://" in url and "@" in url:
            # postgresql://user:password@host:port/db
            parts = url.split("://")
            protocol = parts[0]
            rest = parts[1]

            if "@" in rest:
                auth_part, host_part = rest.split("@", 1)
                if ":" in auth_part:
                    user, _ = auth_part.split(":", 1)
                    return f"{protocol}://{user}:***@{host_part}"

        return url

    except Exception:
        return "***masked***"


def validate_database_connection(db_url: str) -> bool:
    """
    Validar que se puede conectar a la base de datos.

    Args:
        db_url: URL de conexión

    Returns:
        bool: True si conexión exitosa
    """
    try:
        test_engine = create_engine(db_url)
        with test_engine.connect() as connection:
            connection.execute("SELECT 1")

        test_engine.dispose()
        return True

    except Exception as e:
        logger.error(f"Database connection validation failed: {e}")
        return False


# =====================================================
# IMPORTS DE MODELOS Y CONEXIÓN
# =====================================================

# Importar modelos para que se registren con Base
try:
    from .connection import get_db as get_database_session
    from .models import *

    logger.info("Database models imported successfully")
except ImportError as e:
    logger.warning(f"Could not import some database modules: {e}")

# Importar utilidades adicionales
try:
    from .utils import *

    logger.info("Database utilities imported successfully")
except ImportError as e:
    logger.warning(f"Could not import database utilities: {e}")

# =====================================================
# INFORMACIÓN DEL MÓDULO
# =====================================================

__version__ = "1.0.0"
__author__ = "Football Analytics Team"
__description__ = "Database layer para análisis predictivo de fútbol"

# Información del módulo de base de datos
DATABASE_INFO = {
    "version": __version__,
    "orm": "SQLAlchemy",
    "database": "PostgreSQL",
    "features": [
        "Connection pooling",
        "Automatic migrations",
        "Health monitoring",
        "Query optimization",
        "Transaction management",
        "Model relationships",
    ],
    "models": [
        "Country",
        "League",
        "Team",
        "Player",
        "Match",
        "TeamStats",
        "PlayerStats",
        "MatchEvent",
        "Stadium",
        "Coach",
        "PlayerInjury",
        "PredictionHistory",
        "BettingOdds",
    ],
}

# =====================================================
# EXPORTACIONES
# =====================================================

__all__ = [
    # Base y configuración
    "Base",
    "metadata",
    "init_database",
    "create_all_tables",
    "drop_all_tables",
    "get_database_info",
    # Gestión de sesiones
    "get_db",
    "create_session",
    "DatabaseManager",
    # Utilidades
    "validate_database_connection",
    "DATABASE_INFO",
    # Variables globales
    "engine",
    "SessionLocal",
]

# Imports adicionales para compatibilidad
import time
from datetime import datetime
