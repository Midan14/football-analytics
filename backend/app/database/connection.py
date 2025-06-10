"""
Database Connection Management for Football Analytics

Este módulo maneja todas las conexiones a PostgreSQL, incluyendo
configuración del engine, pool de conexiones, health checks y utilidades.
"""

import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Optional

from psycopg2 import OperationalError as Psycopg2OperationalError
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DisconnectionError, OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from app.config import settings

# Variable global para el SessionLocal
SessionLocal = None


def init_db():
    """Inicializa la sesión de base de datos"""
    global SessionLocal
    if SessionLocal is None:
        engine = create_engine(settings.DATABASE_URL, echo=settings.DB_ECHO)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Proporciona una sesión de base de datos para las dependencias de FastAPI"""
    init_db()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================
# CONFIGURACIÓN DE CONEXIÓN
# =====================================================


class DatabaseConfig:
    """Configuración centralizada para la base de datos."""

    # URLs de conexión
    DATABASE_URL = settings.DATABASE_URL
    TEST_DATABASE_URL = getattr(settings, "TEST_DATABASE_URL", None)

    # Configuración del pool
    POOL_SIZE = getattr(settings, "DB_POOL_SIZE", 20)
    MAX_OVERFLOW = getattr(settings, "DB_MAX_OVERFLOW", 30)
    POOL_TIMEOUT = getattr(settings, "DB_POOL_TIMEOUT", 30)
    POOL_RECYCLE = getattr(settings, "DB_POOL_RECYCLE", 3600)  # 1 hora

    # Configuración de conexión
    CONNECT_TIMEOUT = getattr(settings, "DB_CONNECT_TIMEOUT", 10)
    STATEMENT_TIMEOUT = getattr(settings, "DB_STATEMENT_TIMEOUT", 30000)  # 30 segundos

    # Configuración de reintentos
    MAX_RETRIES = getattr(settings, "DB_MAX_RETRIES", 3)
    RETRY_DELAY = getattr(settings, "DB_RETRY_DELAY", 1.0)

    # Logging
    ECHO_SQL = getattr(settings, "DB_ECHO", False)

    @classmethod
    def get_connect_args(cls) -> Dict[str, Any]:
        """Obtener argumentos de conexión PostgreSQL."""
        return {
            "connect_timeout": cls.CONNECT_TIMEOUT,
            "options": f"-c statement_timeout={cls.STATEMENT_TIMEOUT}ms -c timezone=UTC",
            "application_name": "football_analytics",
            "client_encoding": "utf8",
        }


# =====================================================
# GESTIÓN DE ENGINE Y SESIONES
# =====================================================


class DatabaseConnection:
    """Gestor principal de conexiones a la base de datos."""

    def __init__(self):
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None
        self._is_initialized = False
        self._connection_retries = 0

    def initialize(self, database_url: Optional[str] = None, **kwargs) -> None:
        """
        Inicializar conexión a la base de datos.

        Args:
            database_url: URL de conexión (opcional, usa config por defecto)
            **kwargs: Argumentos adicionales para create_engine
        """
        try:
            url = database_url or DatabaseConfig.DATABASE_URL

            if not url:
                raise ValueError("Database URL is required")

            # Configuración del engine
            engine_config = {
                "echo": DatabaseConfig.ECHO_SQL,
                "pool_size": DatabaseConfig.POOL_SIZE,
                "max_overflow": DatabaseConfig.MAX_OVERFLOW,
                "pool_timeout": DatabaseConfig.POOL_TIMEOUT,
                "pool_recycle": DatabaseConfig.POOL_RECYCLE,
                "pool_pre_ping": True,  # Verificar conexiones antes de usar
                "connect_args": DatabaseConfig.get_connect_args(),
                "poolclass": QueuePool,
                **kwargs,
            }

            # Crear engine
            self._engine = create_engine(url, **engine_config)

            # Configurar event listeners
            self._setup_event_listeners()

            # Crear session factory
            self._session_factory = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self._engine,
                expire_on_commit=False,  # Mantener objetos después de commit
            )

            # Verificar conexión
            self._verify_connection()

            self._is_initialized = True
            self._connection_retries = 0

            logger.info("Database connection initialized successfully")
            logger.info(
                f"Pool size: {DatabaseConfig.POOL_SIZE}, Max overflow: {DatabaseConfig.MAX_OVERFLOW}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize database connection: {e}")
            raise

    def _setup_event_listeners(self) -> None:
        """Configurar event listeners para el engine."""

        @event.listens_for(self._engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """Configurar opciones de conexión PostgreSQL."""
            if "postgresql" in str(self._engine.url):
                with dbapi_connection.cursor() as cursor:
                    # Configurar timezone
                    cursor.execute("SET timezone TO 'UTC'")
                    # Configurar encoding
                    cursor.execute("SET client_encoding TO 'UTF8'")
                    # Configurar statement timeout
                    cursor.execute(
                        f"SET statement_timeout TO '{DatabaseConfig.STATEMENT_TIMEOUT}ms'"
                    )

        @event.listens_for(self._engine, "checkout")
        def checkout_listener(dbapi_connection, connection_record, connection_proxy):
            """Listener cuando se obtiene una conexión del pool."""
            logger.debug("Connection checked out from pool")

        @event.listens_for(self._engine, "checkin")
        def checkin_listener(dbapi_connection, connection_record):
            """Listener cuando se retorna una conexión al pool."""
            logger.debug("Connection checked in to pool")

        @event.listens_for(self._engine, "invalidate")
        def invalidate_listener(dbapi_connection, connection_record, exception):
            """Listener cuando una conexión se invalida."""
            logger.warning(f"Connection invalidated: {exception}")

    def _verify_connection(self) -> None:
        """Verificar que la conexión funciona correctamente."""
        try:
            with self._engine.connect() as connection:
                result = connection.execute(text("SELECT 1"))
                result.fetchone()
            logger.info("Database connection verified successfully")

        except Exception as e:
            logger.error(f"Database connection verification failed: {e}")
            raise

    def get_session(self) -> Session:
        """
        Crear nueva sesión de base de datos.

        Returns:
            Session: Nueva sesión de SQLAlchemy
        """
        if not self._is_initialized:
            raise RuntimeError("Database connection not initialized")

        return self._session_factory()

    def get_engine(self) -> Engine:
        """
        Obtener engine de la base de datos.

        Returns:
            Engine: Engine de SQLAlchemy
        """
        if not self._is_initialized:
            raise RuntimeError("Database connection not initialized")

        return self._engine

    def close(self) -> None:
        """Cerrar todas las conexiones."""
        if self._engine:
            self._engine.dispose()
            logger.info("Database connections closed")

    def get_pool_status(self) -> Dict[str, Any]:
        """
        Obtener estado del pool de conexiones.

        Returns:
            dict: Estado del pool
        """
        if not self._engine:
            return {"status": "not_initialized"}

        pool = self._engine.pool
        return {
            "size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "invalid": pool.invalid(),
            "total_connections": pool.size() + pool.overflow(),
            "available_connections": pool.checkedin(),
        }

    def health_check(self) -> Dict[str, Any]:
        """
        Realizar health check completo de la base de datos.

        Returns:
            dict: Estado de salud
        """
        if not self._is_initialized:
            return {
                "status": "not_initialized",
                "timestamp": datetime.now().isoformat(),
            }

        try:
            start_time = time.time()

            # Test básico de conexión
            with self._engine.connect() as connection:
                result = connection.execute(text("SELECT version(), current_timestamp"))
                db_version, db_time = result.fetchone()

            response_time = time.time() - start_time

            # Test de escritura/lectura
            write_test_success = self._test_write_operation()

            return {
                "status": "healthy",
                "response_time_ms": round(response_time * 1000, 2),
                "database_version": db_version,
                "database_time": str(db_time),
                "write_test": write_test_success,
                "pool_status": self.get_pool_status(),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "pool_status": self.get_pool_status(),
                "timestamp": datetime.now().isoformat(),
            }

    def _test_write_operation(self) -> bool:
        """Test de operación de escritura."""
        try:
            with self._engine.connect() as connection:
                # Crear tabla temporal para test
                connection.execute(
                    text(
                        """
                    CREATE TEMP TABLE health_check_test (
                        id SERIAL PRIMARY KEY,
                        test_data TEXT,
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """
                    )
                )

                # Insertar datos de test
                connection.execute(
                    text(
                        """
                    INSERT INTO health_check_test (test_data) 
                    VALUES ('health_check_test_data')
                """
                    )
                )

                # Leer datos de test
                result = connection.execute(
                    text(
                        """
                    SELECT test_data FROM health_check_test LIMIT 1
                """
                    )
                )

                test_data = result.fetchone()[0]
                connection.commit()

                return test_data == "health_check_test_data"

        except Exception as e:
            logger.warning(f"Write test failed: {e}")
            return False


# =====================================================
# INSTANCIA GLOBAL Y UTILIDADES
# =====================================================

# Instancia global del gestor de conexiones
db_connection = DatabaseConnection()


def init_db(database_url: Optional[str] = None) -> None:
    """
    Inicializar conexión global a la base de datos.

    Args:
        database_url: URL de conexión opcional
    """
    db_connection.initialize(database_url)


def get_db() -> Generator[Session, None, None]:
    """
    Dependency para FastAPI - obtener sesión de base de datos.

    Yields:
        Session: Sesión de SQLAlchemy
    """
    session = db_connection.get_session()
    try:
        yield session
    except SQLAlchemyError as e:
        logger.error(f"Database error in session: {e}")
        session.rollback()
        raise
    except Exception as e:
        logger.error(f"Unexpected error in database session: {e}")
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def get_db_context():
    """
    Context manager para usar sesión de base de datos.

    Usage:
        with get_db_context() as db:
            users = db.query(User).all()
    """
    session = db_connection.get_session()
    try:
        yield session
        session.commit()
    except Exception as e:
        logger.error(f"Database error: {e}")
        session.rollback()
        raise
    finally:
        session.close()


def get_engine() -> Engine:
    """
    Obtener engine global de la base de datos.

    Returns:
        Engine: Engine de SQLAlchemy
    """
    return db_connection.get_engine()


# =====================================================
# UTILIDADES DE CONEXIÓN
# =====================================================


class ConnectionRetry:
    """Utilidad para reintentar conexiones con backoff exponencial."""

    @staticmethod
    def with_retry(func, max_retries: int = 3, base_delay: float = 1.0):
        """
        Ejecutar función con reintentos automáticos.

        Args:
            func: Función a ejecutar
            max_retries: Número máximo de reintentos
            base_delay: Delay base en segundos
        """
        for attempt in range(max_retries + 1):
            try:
                return func()

            except (
                OperationalError,
                Psycopg2OperationalError,
                DisconnectionError,
            ) as e:
                if attempt == max_retries:
                    logger.error(f"Max retries ({max_retries}) exceeded: {e}")
                    raise

                delay = base_delay * (2**attempt)  # Backoff exponencial
                logger.warning(
                    f"Database connection failed (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay}s: {e}"
                )
                time.sleep(delay)

            except Exception as e:
                logger.error(f"Non-retryable error: {e}")
                raise


def test_connection(database_url: str, timeout: int = 10) -> Dict[str, Any]:
    """
    Probar conexión a una base de datos específica.

    Args:
        database_url: URL de la base de datos
        timeout: Timeout en segundos

    Returns:
        dict: Resultado del test
    """
    try:
        # Crear engine temporal
        test_engine = create_engine(
            database_url,
            pool_size=1,
            max_overflow=0,
            pool_timeout=timeout,
            connect_args={"connect_timeout": timeout},
        )

        start_time = time.time()

        with test_engine.connect() as connection:
            result = connection.execute(text("SELECT version(), current_timestamp"))
            db_version, db_time = result.fetchone()

        response_time = time.time() - start_time
        test_engine.dispose()

        return {
            "success": True,
            "response_time_ms": round(response_time * 1000, 2),
            "database_version": db_version,
            "database_time": str(db_time),
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }


def create_database_if_not_exists(database_url: str, database_name: str) -> bool:
    """
    Crear base de datos si no existe.

    Args:
        database_url: URL del servidor PostgreSQL
        database_name: Nombre de la base de datos a crear

    Returns:
        bool: True si se creó o ya existía
    """
    try:
        # Conectar al servidor (base de datos postgres por defecto)
        server_url = database_url.rsplit("/", 1)[0] + "/postgres"

        engine = create_engine(server_url, isolation_level="AUTOCOMMIT")

        with engine.connect() as connection:
            # Verificar si la base de datos existe
            result = connection.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :dbname"),
                {"dbname": database_name},
            )

            if result.fetchone():
                logger.info(f"Database '{database_name}' already exists")
                return True

            # Crear base de datos
            connection.execute(text(f'CREATE DATABASE "{database_name}"'))
            logger.info(f"Database '{database_name}' created successfully")
            return True

    except Exception as e:
        logger.error(f"Error creating database '{database_name}': {e}")
        return False
    finally:
        if "engine" in locals():
            engine.dispose()


# =====================================================
# FUNCIONES DE MONITOREO
# =====================================================


def get_database_stats() -> Dict[str, Any]:
    """
    Obtener estadísticas detalladas de la base de datos.

    Returns:
        dict: Estadísticas de la base de datos
    """
    try:
        with get_db_context() as db:
            # Estadísticas básicas
            stats = {}

            # Información de la base de datos
            result = db.execute(
                text(
                    """
                SELECT 
                    current_database() as database_name,
                    current_user as current_user,
                    version() as version,
                    current_timestamp as current_time
            """
                )
            )

            row = result.fetchone()
            stats["database_info"] = {
                "name": row[0],
                "user": row[1],
                "version": row[2],
                "current_time": str(row[3]),
            }

            # Estadísticas de conexiones
            result = db.execute(
                text(
                    """
                SELECT 
                    count(*) as total_connections,
                    count(*) FILTER (WHERE state = 'active') as active_connections,
                    count(*) FILTER (WHERE state = 'idle') as idle_connections
                FROM pg_stat_activity 
                WHERE datname = current_database()
            """
                )
            )

            row = result.fetchone()
            stats["connections"] = {
                "total": row[0],
                "active": row[1],
                "idle": row[2],
            }

            # Tamaño de la base de datos
            result = db.execute(
                text(
                    """
                SELECT pg_size_pretty(pg_database_size(current_database())) as database_size
            """
                )
            )

            stats["database_size"] = result.fetchone()[0]

            return stats

    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        return {"error": str(e)}


def get_slow_queries(limit: int = 10) -> list:
    """
    Obtener queries más lentas (requiere pg_stat_statements).

    Args:
        limit: Número de queries a retornar

    Returns:
        list: Lista de queries lentas
    """
    try:
        with get_db_context() as db:
            result = db.execute(
                text(
                    """
                SELECT 
                    query,
                    calls,
                    total_time,
                    mean_time,
                    max_time
                FROM pg_stat_statements 
                ORDER BY mean_time DESC 
                LIMIT :limit
            """
                ),
                {"limit": limit},
            )

            return [
                {
                    "query": (row[0][:100] + "..." if len(row[0]) > 100 else row[0]),
                    "calls": row[1],
                    "total_time": round(row[2], 2),
                    "mean_time": round(row[3], 2),
                    "max_time": round(row[4], 2),
                }
                for row in result.fetchall()
            ]

    except Exception as e:
        logger.warning(
            f"Could not get slow queries (pg_stat_statements not available?): {e}"
        )
        return []


# =====================================================
# EXPORTACIONES
# =====================================================

__all__ = [
    # Clases principales
    "DatabaseConnection",
    "DatabaseConfig",
    "ConnectionRetry",
    # Instancia global
    "db_connection",
    # Funciones de inicialización
    "init_db",
    "get_db",
    "get_db_context",
    "get_engine",
    # Utilidades
    "test_connection",
    "create_database_if_not_exists",
    "get_database_stats",
    "get_slow_queries",
]
