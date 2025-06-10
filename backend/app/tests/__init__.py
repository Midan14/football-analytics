"""
Football Analytics - Tests Package
Sistema de pruebas para validar todos los componentes del proyecto
"""

import asyncio
import logging
import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, Mock

import pytest

# Configurar path para importaciones
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configurar logging para tests
logging.basicConfig(
    level=logging.WARNING,  # Solo warnings y errores en tests
    format="%(name)s - %(levelname)s - %(message)s",
)

# Versión del sistema de pruebas
__version__ = "1.0.0"

# Test configuration
TEST_CONFIG = {
    "use_real_apis": False,  # Usar mocks por defecto
    "test_databases": True,  # Crear bases de datos temporales
    "cleanup_after_tests": True,  # Limpiar archivos después de tests
    "parallel_execution": False,  # Ejecutar tests en paralelo
    "coverage_threshold": 80,  # Umbral mínimo de cobertura
}

# Configuración de la aplicación para tests
TEST_APP_CONFIG = {
    "football_data_api_key": "test_key_football_data_123",
    "rapidapi_key": "test_key_rapidapi_456",
    "odds_api_key": "test_key_odds_789",
    "database": {
        "type": "sqlite",
        "path": ":memory:",  # Base de datos en memoria para tests
        "backup_path": "/tmp/test_backups/",
        "max_connections": 5,
    },
    "logging": {
        "level": "WARNING",
        "file": "/tmp/test_logs/football_analytics_test.log",
    },
    "leagues": {"test_league": "TL", "premier_league": "PL", "la_liga": "PD"},
    "rate_limits": {
        "football_data": 100,  # Más permisivo en tests
        "rapidapi": 1000,
        "odds_api": 5000,
    },
}


class TestDatabase:
    """Gestor de bases de datos temporales para tests"""

    def __init__(self):
        self.temp_dbs = {}
        self.temp_files = []

    def create_temp_db(self, db_name: str = "test") -> str:
        """Crea una base de datos temporal"""
        if db_name in self.temp_dbs:
            return self.temp_dbs[db_name]

        # Crear archivo temporal
        temp_file = tempfile.NamedTemporaryFile(
            suffix=f"_{db_name}.db", delete=False
        )
        temp_file.close()

        db_path = temp_file.name
        self.temp_dbs[db_name] = db_path
        self.temp_files.append(db_path)

        return db_path

    def init_test_data(self, db_path: str):
        """Inicializa datos de prueba en la base de datos"""
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Tabla de partidos de prueba
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS test_matches (
                    match_id TEXT PRIMARY KEY,
                    home_team TEXT,
                    away_team TEXT,
                    home_goals INTEGER,
                    away_goals INTEGER,
                    match_date TEXT,
                    league TEXT
                )
            """
            )

            # Datos de prueba
            test_matches = [
                (
                    "test_1",
                    "Real Madrid",
                    "Barcelona",
                    2,
                    1,
                    "2024-01-15",
                    "La Liga",
                ),
                (
                    "test_2",
                    "Manchester City",
                    "Liverpool",
                    3,
                    1,
                    "2024-01-16",
                    "Premier League",
                ),
                (
                    "test_3",
                    "Bayern Munich",
                    "Borussia Dortmund",
                    1,
                    1,
                    "2024-01-17",
                    "Bundesliga",
                ),
                ("test_4", "PSG", "Marseille", 2, 0, "2024-01-18", "Ligue 1"),
                (
                    "test_5",
                    "Juventus",
                    "Inter Milan",
                    0,
                    2,
                    "2024-01-19",
                    "Serie A",
                ),
            ]

            cursor.executemany(
                "INSERT OR REPLACE INTO test_matches VALUES (?, ?, ?, ?, ?, ?, ?)",
                test_matches,
            )

            conn.commit()

    def cleanup(self):
        """Limpia todas las bases de datos temporales"""
        for db_path in self.temp_files:
            try:
                os.unlink(db_path)
            except FileNotFoundError:
                pass

        self.temp_dbs.clear()
        self.temp_files.clear()


class MockDataProvider:
    """Proveedor de datos mock para tests"""

    @staticmethod
    def get_mock_match_data() -> List[Dict[str, Any]]:
        """Retorna datos de partidos mock"""
        return [
            {
                "match_id": "mock_001",
                "home_team": "Real Madrid",
                "away_team": "Barcelona",
                "home_goals": 2,
                "away_goals": 1,
                "date": datetime.now() - timedelta(days=1),
                "league": "La Liga",
                "status": "finished",
            },
            {
                "match_id": "mock_002",
                "home_team": "Manchester City",
                "away_team": "Liverpool",
                "home_goals": None,
                "away_goals": None,
                "date": datetime.now() + timedelta(days=1),
                "league": "Premier League",
                "status": "scheduled",
            },
            {
                "match_id": "mock_003",
                "home_team": "Bayern Munich",
                "away_team": "Borussia Dortmund",
                "home_goals": 1,
                "away_goals": 1,
                "date": datetime.now(),
                "league": "Bundesliga",
                "status": "live",
            },
        ]

    @staticmethod
    def get_mock_team_data() -> List[Dict[str, Any]]:
        """Retorna datos de equipos mock"""
        return [
            {
                "team_id": "team_001",
                "name": "Real Madrid",
                "league": "La Liga",
                "country": "Spain",
                "matches_played": 20,
                "wins": 15,
                "draws": 3,
                "losses": 2,
                "goals_for": 45,
                "goals_against": 20,
                "points": 48,
            },
            {
                "team_id": "team_002",
                "name": "Barcelona",
                "league": "La Liga",
                "country": "Spain",
                "matches_played": 20,
                "wins": 13,
                "draws": 4,
                "losses": 3,
                "goals_for": 42,
                "goals_against": 25,
                "points": 43,
            },
            {
                "team_id": "team_003",
                "name": "Manchester City",
                "league": "Premier League",
                "country": "England",
                "matches_played": 22,
                "wins": 17,
                "draws": 3,
                "losses": 2,
                "goals_for": 55,
                "goals_against": 18,
                "points": 54,
            },
        ]

    @staticmethod
    def get_mock_odds_data() -> List[Dict[str, Any]]:
        """Retorna datos de cuotas mock"""
        return [
            {
                "match_id": "mock_001",
                "bookmaker": "Test Bookmaker",
                "home_win": 2.10,
                "draw": 3.40,
                "away_win": 3.80,
                "over_2_5": 1.85,
                "under_2_5": 1.95,
                "timestamp": datetime.now(),
            },
            {
                "match_id": "mock_002",
                "bookmaker": "Another Bookmaker",
                "home_win": 1.75,
                "draw": 3.60,
                "away_win": 4.50,
                "over_2_5": 1.90,
                "under_2_5": 1.90,
                "timestamp": datetime.now(),
            },
        ]

    @staticmethod
    def get_mock_prediction_data() -> Dict[str, Any]:
        """Retorna datos de predicción mock"""
        return {
            "match_id": "mock_pred_001",
            "home_team": "Real Madrid",
            "away_team": "Barcelona",
            "result_probabilities": {
                "home": 0.485,
                "draw": 0.287,
                "away": 0.228,
            },
            "expected_goals_home": 1.65,
            "expected_goals_away": 1.15,
            "confidence_score": 0.78,
            "most_likely_result": "home",
            "most_likely_score": (2, 1),
        }


class AsyncTestCase:
    """Clase base para tests asíncronos"""

    def setup_method(self):
        """Setup antes de cada test"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def teardown_method(self):
        """Cleanup después de cada test"""
        self.loop.close()

    async def run_async_test(self, coro):
        """Ejecuta un test asíncrono"""
        return await coro


class MockAPIResponses:
    """Respuestas mock para APIs externas"""

    @staticmethod
    def football_data_matches_response():
        """Mock response de Football-Data.org matches"""
        return {
            "matches": [
                {
                    "id": 12345,
                    "utcDate": "2024-06-15T15:00:00Z",
                    "status": "FINISHED",
                    "homeTeam": {"id": 86, "name": "Real Madrid"},
                    "awayTeam": {"id": 81, "name": "Barcelona"},
                    "score": {
                        "fullTime": {"home": 2, "away": 1},
                        "halfTime": {"home": 1, "away": 0},
                    },
                    "competition": {"code": "PD"},
                    "season": {"startDate": "2024-08-01"},
                    "matchday": 15,
                }
            ]
        }

    @staticmethod
    def football_data_teams_response():
        """Mock response de Football-Data.org teams"""
        return {
            "teams": [
                {
                    "id": 86,
                    "name": "Real Madrid",
                    "shortName": "Madrid",
                    "area": {"name": "Spain"},
                    "founded": 1902,
                    "venue": "Santiago Bernabéu Stadium",
                    "crest": "https://example.com/madrid.png",
                },
                {
                    "id": 81,
                    "name": "FC Barcelona",
                    "shortName": "Barça",
                    "area": {"name": "Spain"},
                    "founded": 1899,
                    "venue": "Camp Nou",
                    "crest": "https://example.com/barca.png",
                },
            ]
        }

    @staticmethod
    def odds_api_response():
        """Mock response de The Odds API"""
        return [
            {
                "id": "odds_123",
                "home_team": "Real Madrid",
                "away_team": "Barcelona",
                "bookmakers": [
                    {
                        "title": "Test Bookmaker",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Real Madrid", "price": 2.10},
                                    {"name": "Draw", "price": 3.40},
                                    {"name": "Barcelona", "price": 3.80},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]


# Fixtures y utilidades comunes
def create_test_config(overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """Crea configuración de test con sobrescrituras opcionales"""
    config = TEST_APP_CONFIG.copy()
    if overrides:
        config.update(overrides)
    return config


def setup_test_environment():
    """Configura el entorno de pruebas"""
    # Crear directorios temporales
    temp_dirs = ["/tmp/test_logs", "/tmp/test_backups", "/tmp/test_data"]
    for temp_dir in temp_dirs:
        os.makedirs(temp_dir, exist_ok=True)

    # Configurar variables de entorno para tests
    os.environ["TESTING"] = "true"
    os.environ["FOOTBALL_DATA_API_KEY"] = TEST_APP_CONFIG[
        "football_data_api_key"
    ]


def cleanup_test_environment():
    """Limpia el entorno después de las pruebas"""
    if TEST_CONFIG["cleanup_after_tests"]:
        # Limpiar archivos temporales
        import shutil

        temp_dirs = ["/tmp/test_logs", "/tmp/test_backups", "/tmp/test_data"]
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)


# Decoradores para tests
def requires_real_api(test_func):
    """Decorador para tests que requieren APIs reales"""

    def wrapper(*args, **kwargs):
        if not TEST_CONFIG["use_real_apis"]:
            pytest.skip("Test requiere APIs reales - usar flag --real-apis")
        return test_func(*args, **kwargs)

    return wrapper


def async_test(test_func):
    """Decorador para tests asíncronos"""

    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(test_func(*args, **kwargs))
        finally:
            loop.close()

    return wrapper


# Instancias globales para tests
test_db = TestDatabase()
mock_data = MockDataProvider()
mock_api = MockAPIResponses()


# pytest fixtures
@pytest.fixture(scope="session")
def test_config():
    """Configuración global para tests"""
    return create_test_config()


@pytest.fixture(scope="session")
def test_database():
    """Base de datos temporal para tests"""
    db_path = test_db.create_temp_db("main_test")
    test_db.init_test_data(db_path)
    yield db_path
    test_db.cleanup()


@pytest.fixture
def mock_match_data():
    """Datos de partidos mock"""
    return mock_data.get_mock_match_data()


@pytest.fixture
def mock_team_data():
    """Datos de equipos mock"""
    return mock_data.get_mock_team_data()


@pytest.fixture
def mock_odds_data():
    """Datos de cuotas mock"""
    return mock_data.get_mock_odds_data()


@pytest.fixture
def mock_prediction_data():
    """Datos de predicción mock"""
    return mock_data.get_mock_prediction_data()


@pytest.fixture(autouse=True)
def setup_teardown():
    """Setup y teardown automático para cada test"""
    setup_test_environment()
    yield
    # Cleanup se ejecuta solo al final si está configurado


# Configuración de pytest
def pytest_configure(config):
    """Configuración de pytest"""
    # Marcadores personalizados
    config.addinivalue_line("markers", "unit: marca tests unitarios")
    config.addinivalue_line(
        "markers", "integration: marca tests de integración"
    )
    config.addinivalue_line("markers", "e2e: marca tests end-to-end")
    config.addinivalue_line("markers", "slow: marca tests lentos")
    config.addinivalue_line(
        "markers", "api: marca tests que usan APIs externas"
    )


def pytest_addoption(parser):
    """Opciones adicionales para pytest"""
    parser.addoption(
        "--real-apis",
        action="store_true",
        default=False,
        help="Usar APIs reales en lugar de mocks",
    )
    parser.addoption(
        "--coverage-threshold",
        type=int,
        default=80,
        help="Umbral mínimo de cobertura de código",
    )


def pytest_collection_modifyitems(config, items):
    """Modifica la colección de tests"""
    if config.getoption("--real-apis"):
        TEST_CONFIG["use_real_apis"] = True

    # Marcar tests según su ubicación
    for item in items:
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)


# Funciones de utilidad para tests
def assert_prediction_valid(prediction):
    """Valida que una predicción tenga la estructura correcta"""
    assert hasattr(prediction, "match_id")
    assert hasattr(prediction, "home_team")
    assert hasattr(prediction, "away_team")
    assert hasattr(prediction, "result_probabilities")
    assert sum(prediction.result_probabilities.values()) == pytest.approx(
        1.0, abs=0.01
    )


def assert_odds_valid(odds_data):
    """Valida que los datos de cuotas sean correctos"""
    assert "home_win" in odds_data
    assert "draw" in odds_data
    assert "away_win" in odds_data
    assert all(
        odd > 1.0
        for odd in [
            odds_data["home_win"],
            odds_data["draw"],
            odds_data["away_win"],
        ]
    )


def create_mock_async_response(data, status=200):
    """Crea una respuesta async mock"""
    mock_response = AsyncMock()
    mock_response.status = status
    mock_response.json = AsyncMock(return_value=data)
    return mock_response


# Logging para tests
test_logger = logging.getLogger("football_analytics.tests")
test_logger.setLevel(logging.INFO)

# Información del paquete de tests
__all__ = [
    "TEST_CONFIG",
    "TEST_APP_CONFIG",
    "TestDatabase",
    "MockDataProvider",
    "MockAPIResponses",
    "AsyncTestCase",
    "create_test_config",
    "setup_test_environment",
    "cleanup_test_environment",
    "requires_real_api",
    "async_test",
    "assert_prediction_valid",
    "assert_odds_valid",
    "create_mock_async_response",
    "test_db",
    "mock_data",
    "mock_api",
]

if __name__ == "__main__":
    print("🧪 Football Analytics - Test Suite")
    print(f"📦 Tests version: {__version__}")
    print(f"🔧 Configuration loaded: {len(TEST_CONFIG)} options")
    print(f"📊 Mock data providers: {len(__all__)} utilities")
    print("\n✅ Test environment ready")
    print("🚀 Run tests with: pytest app/tests/")
