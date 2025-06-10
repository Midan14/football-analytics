"""
database/__init__.py - Módulo de Base de Datos Football Analytics

Módulo principal para gestión de base de datos, conexiones, modelos y operaciones
específicas del sistema Football Analytics.

Author: Football Analytics Team
Version: 2.1.0
Date: 2024-06-02

Estructura del módulo:
- Gestión de conexiones de base de datos
- Modelos de SQLAlchemy
- Utilidades de migración
- Operaciones específicas de fútbol
- Cache y optimizaciones
"""

import logging
import os
import sys
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Imports de SQLAlchemy
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    event,
    inspect,
    pool,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship, sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.sql import text

# Imports async
try:
    from sqlalchemy.ext.asyncio import (
        AsyncSession,
        async_sessionmaker,
        create_async_engine,
    )

    ASYNC_SUPPORT = True
except ImportError:
    ASYNC_SUPPORT = False
    AsyncSession = None
    async_sessionmaker = None

# Imports para migraciones
try:
    from alembic import command
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    ALEMBIC_SUPPORT = True
except ImportError:
    ALEMBIC_SUPPORT = False

# Configuración de logging
logger = logging.getLogger(__name__)

# ================================
# CONFIGURACIÓN Y CONSTANTES
# ================================

# Información del módulo
__version__ = "2.1.0"
__author__ = "Football Analytics Team"

# Configuración por defecto
DEFAULT_DATABASE_URL = "sqlite:///football_analytics.db"
DEFAULT_ASYNC_DATABASE_URL = "sqlite+aiosqlite:///football_analytics.db"

# Pool settings
DEFAULT_POOL_SIZE = 10
DEFAULT_MAX_OVERFLOW = 20
DEFAULT_POOL_TIMEOUT = 30
DEFAULT_POOL_RECYCLE = 3600

# Base declarativa
Base = declarative_base()

# Metadata
metadata = MetaData(
    naming_convention={
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    }
)

# ================================
# GESTIÓN DE CONEXIONES
# ================================


class DatabaseManager:
    """Gestor principal de conexiones de base de datos"""

    def __init__(self):
        self.engine: Optional[Engine] = None
        self.async_engine: Optional[Any] = None
        self.session_factory: Optional[sessionmaker] = None
        self.async_session_factory: Optional[Any] = None
        self._metadata = metadata
        self._is_initialized = False

    def initialize(
        self,
        database_url: Optional[str] = None,
        async_database_url: Optional[str] = None,
        **engine_kwargs,
    ) -> None:
        """Inicializa las conexiones de base de datos"""

        # Obtener URLs de base de datos
        db_url = database_url or os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
        async_db_url = async_database_url or os.getenv(
            "ASYNC_DATABASE_URL", DEFAULT_ASYNC_DATABASE_URL
        )

        logger.info(f"Inicializando base de datos: {db_url}")

        # Configuración del engine
        engine_config = {
            "pool_size": DEFAULT_POOL_SIZE,
            "max_overflow": DEFAULT_MAX_OVERFLOW,
            "pool_timeout": DEFAULT_POOL_TIMEOUT,
            "pool_recycle": DEFAULT_POOL_RECYCLE,
            "echo": os.getenv("DB_ECHO", "false").lower() == "true",
            **engine_kwargs,
        }

        # Configuración específica para SQLite
        if db_url.startswith("sqlite"):
            engine_config.update(
                {
                    "poolclass": StaticPool,
                    "connect_args": {
                        "check_same_thread": False,
                        "timeout": 30,
                        "isolation_level": None,
                    },
                }
            )

        # Crear engine síncrono
        try:
            self.engine = create_engine(db_url, **engine_config)
            self.session_factory = sessionmaker(bind=self.engine)
            logger.info("✅ Engine síncrono inicializado correctamente")
        except Exception as e:
            logger.error(f"❌ Error inicializando engine síncrono: {e}")
            raise

        # Crear engine asíncrono si está disponible
        if ASYNC_SUPPORT:
            try:
                async_config = engine_config.copy()
                # Remover configuraciones que no aplican para async
                async_config.pop("connect_args", None)

                self.async_engine = create_async_engine(async_db_url, **async_config)
                self.async_session_factory = async_sessionmaker(bind=self.async_engine)
                logger.info("✅ Engine asíncrono inicializado correctamente")
            except Exception as e:
                logger.warning(f"⚠️ No se pudo inicializar engine asíncrono: {e}")

        # Configurar event listeners
        self._setup_event_listeners()

        self._is_initialized = True
        logger.info("🎯 DatabaseManager inicializado completamente")

    def _setup_event_listeners(self) -> None:
        """Configura event listeners para optimización"""

        if self.engine:
            # Listener para SQLite optimizations
            @event.listens_for(self.engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                if "sqlite" in str(self.engine.url):
                    cursor = dbapi_connection.cursor()
                    # Optimizaciones para SQLite
                    cursor.execute("PRAGMA foreign_keys=ON")
                    cursor.execute("PRAGMA journal_mode=WAL")
                    cursor.execute("PRAGMA synchronous=NORMAL")
                    cursor.execute("PRAGMA cache_size=10000")
                    cursor.execute("PRAGMA temp_store=MEMORY")
                    cursor.close()

    @contextmanager
    def get_session(self) -> Session:
        """Context manager para sesiones síncronas"""
        if not self._is_initialized:
            raise RuntimeError("DatabaseManager no ha sido inicializado")

        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @asynccontextmanager
    async def get_async_session(self) -> AsyncSession:
        """Context manager para sesiones asíncronas"""
        if not self._is_initialized or not self.async_session_factory:
            raise RuntimeError("AsyncSession no disponible")

        session = self.async_session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    def create_tables(self, checkfirst: bool = True) -> None:
        """Crea todas las tablas en la base de datos"""
        if not self.engine:
            raise RuntimeError("Engine no inicializado")

        logger.info("📊 Creando tablas de base de datos...")
        try:
            Base.metadata.create_all(bind=self.engine, checkfirst=checkfirst)
            logger.info("✅ Tablas creadas exitosamente")
        except Exception as e:
            logger.error(f"❌ Error creando tablas: {e}")
            raise

    def drop_tables(self) -> None:
        """Elimina todas las tablas (¡CUIDADO!)"""
        if not self.engine:
            raise RuntimeError("Engine no inicializado")

        logger.warning("⚠️ Eliminando todas las tablas...")
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.info("✅ Tablas eliminadas")
        except Exception as e:
            logger.error(f"❌ Error eliminando tablas: {e}")
            raise

    def get_table_info(self) -> Dict[str, Any]:
        """Obtiene información de las tablas"""
        if not self.engine:
            raise RuntimeError("Engine no inicializado")

        inspector = inspect(self.engine)
        tables = inspector.get_table_names()

        info = {
            "total_tables": len(tables),
            "tables": {},
            "database_url": str(self.engine.url),
            "driver": self.engine.driver,
        }

        for table_name in tables:
            columns = inspector.get_columns(table_name)
            indexes = inspector.get_indexes(table_name)
            foreign_keys = inspector.get_foreign_keys(table_name)

            info["tables"][table_name] = {
                "columns": len(columns),
                "indexes": len(indexes),
                "foreign_keys": len(foreign_keys),
                "column_names": [col["name"] for col in columns],
            }

        return info

    def execute_sql(self, sql: str, params: Optional[Dict] = None) -> Any:
        """Ejecuta SQL directamente"""
        if not self.engine:
            raise RuntimeError("Engine no inicializado")

        with self.engine.connect() as conn:
            result = conn.execute(text(sql), params or {})
            return result.fetchall()

    def close(self) -> None:
        """Cierra todas las conexiones"""
        if self.engine:
            self.engine.dispose()
            logger.info("🔒 Engine síncrono cerrado")

        if self.async_engine:
            # Note: async engine disposal should be done with asyncio
            logger.info("🔒 Engine asíncrono marcado para cierre")

        self._is_initialized = False


# ================================
# INSTANCIA GLOBAL
# ================================

# Instancia global del gestor de base de datos
db_manager = DatabaseManager()

# ================================
# MODELOS DE BASE DE DATOS
# ================================


class League(Base):
    """Modelo para ligas de fútbol"""

    __tablename__ = "leagues"

    id = Column(Integer, primary_key=True)
    code = Column(String(10), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    country = Column(String(50), nullable=False)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relaciones
    teams = relationship("Team", back_populates="league")
    matches = relationship("Match", back_populates="league")


class Team(Base):
    """Modelo para equipos de fútbol"""

    __tablename__ = "teams"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, index=True)
    short_name = Column(String(20))
    code = Column(String(10))
    league_id = Column(Integer, ForeignKey("leagues.id"), nullable=False)
    founded = Column(Integer)
    venue = Column(String(100))
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relaciones
    league = relationship("League", back_populates="teams")
    home_matches = relationship(
        "Match", foreign_keys="Match.home_team_id", back_populates="home_team"
    )
    away_matches = relationship(
        "Match", foreign_keys="Match.away_team_id", back_populates="away_team"
    )

    # Índices
    __table_args__ = (Index("ix_team_league_name", "league_id", "name"),)


class Match(Base):
    """Modelo para partidos de fútbol"""

    __tablename__ = "matches"

    id = Column(Integer, primary_key=True)
    league_id = Column(Integer, ForeignKey("leagues.id"), nullable=False)
    home_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    match_date = Column(DateTime, nullable=False, index=True)
    matchday = Column(Integer)

    # Resultados
    status = Column(
        String(20), default="SCHEDULED"
    )  # SCHEDULED, LIVE, FINISHED, POSTPONED
    home_score = Column(Integer)
    away_score = Column(Integer)

    # Estadísticas
    home_shots = Column(Integer)
    away_shots = Column(Integer)
    home_shots_on_target = Column(Integer)
    away_shots_on_target = Column(Integer)
    home_corners = Column(Integer)
    away_corners = Column(Integer)
    home_fouls = Column(Integer)
    away_fouls = Column(Integer)
    home_yellow_cards = Column(Integer)
    away_yellow_cards = Column(Integer)
    home_red_cards = Column(Integer)
    away_red_cards = Column(Integer)

    # Metadatos
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relaciones
    league = relationship("League", back_populates="matches")
    home_team = relationship(
        "Team", foreign_keys=[home_team_id], back_populates="home_matches"
    )
    away_team = relationship(
        "Team", foreign_keys=[away_team_id], back_populates="away_matches"
    )
    predictions = relationship("Prediction", back_populates="match")

    # Índices
    __table_args__ = (
        Index("ix_match_date_league", "match_date", "league_id"),
        Index("ix_match_teams", "home_team_id", "away_team_id"),
    )


class Prediction(Base):
    """Modelo para predicciones de partidos"""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    model_name = Column(String(50), nullable=False)
    model_version = Column(String(20))

    # Probabilidades
    home_win_prob = Column(Float, nullable=False)
    draw_prob = Column(Float, nullable=False)
    away_win_prob = Column(Float, nullable=False)

    # Predicciones adicionales
    home_goals_pred = Column(Float)
    away_goals_pred = Column(Float)
    total_goals_pred = Column(Float)

    # Confianza y metadatos
    confidence_score = Column(Float)
    features_used = Column(Text)  # JSON string
    prediction_date = Column(DateTime, default=datetime.utcnow)

    # Evaluación (después del partido)
    is_correct = Column(Boolean)
    actual_result = Column(String(10))  # 'H', 'D', 'A'

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relaciones
    match = relationship("Match", back_populates="predictions")

    # Índices
    __table_args__ = (
        Index("ix_prediction_match_model", "match_id", "model_name"),
        Index("ix_prediction_date", "prediction_date"),
    )


class Odds(Base):
    """Modelo para cuotas de apuestas"""

    __tablename__ = "odds"

    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    bookmaker = Column(String(50), nullable=False)
    market = Column(String(50), nullable=False)  # '1x2', 'over_under', 'btts'

    # Cuotas principales (1X2)
    home_odds = Column(Float)
    draw_odds = Column(Float)
    away_odds = Column(Float)

    # Cuotas adicionales
    over_2_5_odds = Column(Float)
    under_2_5_odds = Column(Float)
    btts_yes_odds = Column(Float)
    btts_no_odds = Column(Float)

    # Metadatos
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    is_closing = Column(Boolean, default=False)  # Cuotas de cierre

    # Relación
    match = relationship("Match")

    # Índices
    __table_args__ = (
        Index("ix_odds_match_bookmaker", "match_id", "bookmaker"),
        Index("ix_odds_timestamp", "timestamp"),
    )


# ================================
# UTILIDADES DE MIGRACIÓN
# ================================


class MigrationManager:
    """Gestor de migraciones de base de datos"""

    def __init__(self, alembic_cfg_path: Optional[str] = None):
        self.alembic_cfg_path = alembic_cfg_path or "alembic.ini"

    def init_alembic(self, directory: str = "migrations") -> None:
        """Inicializa Alembic para migraciones"""
        if not ALEMBIC_SUPPORT:
            raise RuntimeError("Alembic no está instalado")

        config = Config()
        config.set_main_option("script_location", directory)
        config.set_main_option("sqlalchemy.url", str(db_manager.engine.url))

        command.init(config, directory)
        logger.info(f"✅ Alembic inicializado en {directory}")

    def create_migration(self, message: str) -> None:
        """Crea una nueva migración"""
        if not ALEMBIC_SUPPORT:
            raise RuntimeError("Alembic no está instalado")

        config = Config(self.alembic_cfg_path)
        command.revision(config, autogenerate=True, message=message)
        logger.info(f"✅ Migración creada: {message}")

    def upgrade(self, revision: str = "head") -> None:
        """Ejecuta migraciones hacia adelante"""
        if not ALEMBIC_SUPPORT:
            raise RuntimeError("Alembic no está instalado")

        config = Config(self.alembic_cfg_path)
        command.upgrade(config, revision)
        logger.info(f"✅ Migración ejecutada: {revision}")

    def downgrade(self, revision: str) -> None:
        """Ejecuta migraciones hacia atrás"""
        if not ALEMBIC_SUPPORT:
            raise RuntimeError("Alembic no está instalado")

        config = Config(self.alembic_cfg_path)
        command.downgrade(config, revision)
        logger.info(f"✅ Downgrade ejecutado: {revision}")


# ================================
# UTILIDADES DE DATOS
# ================================


class FootballDataUtils:
    """Utilidades para operaciones específicas de fútbol"""

    @staticmethod
    def get_team_by_name(
        name: str, league_code: Optional[str] = None
    ) -> Optional[Team]:
        """Busca un equipo por nombre"""
        with db_manager.get_session() as session:
            query = session.query(Team).filter(Team.name.ilike(f"%{name}%"))

            if league_code:
                query = query.join(League).filter(League.code == league_code)

            return query.first()

    @staticmethod
    def get_recent_matches(team_id: int, limit: int = 10) -> List[Match]:
        """Obtiene partidos recientes de un equipo"""
        with db_manager.get_session() as session:
            return (
                session.query(Match)
                .filter(
                    (Match.home_team_id == team_id) | (Match.away_team_id == team_id),
                    Match.status == "FINISHED",
                    Match.match_date <= datetime.utcnow(),
                )
                .order_by(Match.match_date.desc())
                .limit(limit)
                .all()
            )

    @staticmethod
    def get_upcoming_matches(league_code: str, days: int = 7) -> List[Match]:
        """Obtiene próximos partidos de una liga"""
        end_date = datetime.utcnow() + timedelta(days=days)

        with db_manager.get_session() as session:
            return (
                session.query(Match)
                .join(League)
                .filter(
                    League.code == league_code,
                    Match.status == "SCHEDULED",
                    Match.match_date.between(datetime.utcnow(), end_date),
                )
                .order_by(Match.match_date)
                .all()
            )

    @staticmethod
    def calculate_team_form(team_id: int, matches: int = 5) -> Dict[str, Any]:
        """Calcula la forma de un equipo"""
        recent_matches = FootballDataUtils.get_recent_matches(team_id, matches)

        points = 0
        goals_for = 0
        goals_against = 0

        for match in recent_matches:
            if match.home_team_id == team_id:
                goals_for += match.home_score or 0
                goals_against += match.away_score or 0

                if match.home_score > match.away_score:
                    points += 3
                elif match.home_score == match.away_score:
                    points += 1
            else:
                goals_for += match.away_score or 0
                goals_against += match.home_score or 0

                if match.away_score > match.home_score:
                    points += 3
                elif match.away_score == match.home_score:
                    points += 1

        return {
            "points": points,
            "goals_for": goals_for,
            "goals_against": goals_against,
            "goal_difference": goals_for - goals_against,
            "matches_played": len(recent_matches),
            "points_per_game": points / max(len(recent_matches), 1),
            "goals_per_game": goals_for / max(len(recent_matches), 1),
        }


# ================================
# FUNCIONES DE INICIALIZACIÓN
# ================================


def initialize_database(
    database_url: Optional[str] = None, create_tables: bool = True, **kwargs
) -> None:
    """Inicializa la base de datos completa"""
    logger.info("🚀 Inicializando sistema de base de datos...")

    # Inicializar gestor
    db_manager.initialize(database_url, **kwargs)

    # Crear tablas si se solicita
    if create_tables:
        db_manager.create_tables()

    logger.info("✅ Sistema de base de datos inicializado correctamente")


def get_session() -> Session:
    """Función de conveniencia para obtener sesión"""
    return db_manager.get_session()


async def get_async_session() -> AsyncSession:
    """Función de conveniencia para obtener sesión async"""
    return db_manager.get_async_session()


def close_database() -> None:
    """Cierra las conexiones de base de datos"""
    db_manager.close()
    logger.info("🔒 Base de datos cerrada")


# ================================
# HEALTH CHECK
# ================================


def health_check() -> Dict[str, Any]:
    """Verifica el estado de la base de datos"""
    try:
        if not db_manager._is_initialized:
            return {"status": "unhealthy", "error": "Database not initialized"}

        # Test de conexión
        with db_manager.get_session() as session:
            session.execute(text("SELECT 1"))

        # Información de tablas
        table_info = db_manager.get_table_info()

        return {
            "status": "healthy",
            "database_url": str(db_manager.engine.url).split("@")[
                -1
            ],  # Sin credenciales
            "driver": db_manager.engine.driver,
            "tables": table_info["total_tables"],
            "async_support": ASYNC_SUPPORT,
            "alembic_support": ALEMBIC_SUPPORT,
        }

    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


# ================================
# EXPORTACIONES
# ================================

# Exportar elementos principales
__all__ = [
    # Gestor principal
    "db_manager",
    "DatabaseManager",
    # Modelos
    "Base",
    "League",
    "Team",
    "Match",
    "Prediction",
    "Odds",
    # Utilidades
    "MigrationManager",
    "FootballDataUtils",
    # Funciones principales
    "initialize_database",
    "get_session",
    "get_async_session",
    "close_database",
    "health_check",
    # Constantes
    "metadata",
    "ASYNC_SUPPORT",
    "ALEMBIC_SUPPORT",
]

# ================================
# INICIALIZACIÓN AUTOMÁTICA
# ================================

# Solo para development/testing
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO)

    # Inicializar con base de datos de prueba
    initialize_database("sqlite:///test_football_analytics.db")

    # Mostrar información
    health = health_check()
    print(f"🏥 Database Health: {health}")

    # Cerrar
    close_database()  # ================================

# Solo para development/testing
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO)

    # Inicializar con base de datos de prueba
    initialize_database("sqlite:///test_football_analytics.db")

    # Mostrar información
    health = health_check()
    print(f"🏥 Database Health: {health}")

    # Cerrar
    close_database()
