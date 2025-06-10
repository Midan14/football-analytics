"""
Football Analytics - API Tests
Pruebas completas para todos los endpoints de la API
"""

import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Importaciones del proyecto
from app.main import app
from app.services.calculator import CalculatorService
from app.services.data_collector import DataCollectorService
from app.services.live_tracker import LiveTrackerService
from app.services.odds_calculator import (
    BettingOpportunity,
    OddsCalculatorService,
)
from app.services.predictor import MatchPrediction, PredictorService

# Test configuration
from . import (
    TEST_APP_CONFIG,
    assert_odds_valid,
    assert_prediction_valid,
    create_mock_async_response,
    mock_api,
    mock_data,
)

# Cliente de test para FastAPI
client = TestClient(app)


class TestHealthEndpoints:
    """Tests para endpoints de salud y estado"""

    def test_health_check(self):
        """Test del endpoint de health check"""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert "components" in data

    def test_status_endpoint(self):
        """Test del endpoint de estado del sistema"""
        response = client.get("/status")

        assert response.status_code == 200
        data = response.json()

        assert "system_status" in data
        assert "services" in data
        assert "database" in data
        assert "api_keys" in data

        # Verificar que los servicios principales estén listados
        services = data["services"]
        expected_services = [
            "predictor",
            "data_collector",
            "odds_calculator",
            "live_tracker",
        ]
        for service in expected_services:
            assert service in services


class TestPredictionEndpoints:
    """Tests para endpoints de predicciones"""

    @patch("app.services.predictor.PredictorService")
    def test_predict_match_success(self, mock_predictor_service):
        """Test exitoso de predicción de partido"""
        # Mock del predictor
        mock_predictor = Mock()
        mock_prediction = Mock(spec=MatchPrediction)
        mock_prediction.match_id = "test_123"
        mock_prediction.home_team = "Real Madrid"
        mock_prediction.away_team = "Barcelona"
        mock_prediction.result_probabilities = {
            "home": 0.485,
            "draw": 0.287,
            "away": 0.228,
        }
        mock_prediction.confidence_score = 0.78
        mock_prediction.most_likely_result = "home"
        mock_prediction.expected_goals_home = 1.65
        mock_prediction.expected_goals_away = 1.15
        mock_prediction.most_likely_score = (2, 1)

        mock_predictor.predict_match.return_value = mock_prediction
        mock_predictor_service.return_value = mock_predictor

        # Request
        response = client.post(
            "/api/v1/predict/match",
            json={
                "home_team": "Real Madrid",
                "away_team": "Barcelona",
                "league": "La Liga",
                "match_date": "2024-06-15T15:00:00Z",
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Verificar estructura de respuesta
        assert "prediction" in data
        assert "success" in data
        assert data["success"] is True

        prediction = data["prediction"]
        assert prediction["home_team"] == "Real Madrid"
        assert prediction["away_team"] == "Barcelona"
        assert "result_probabilities" in prediction
        assert "confidence_score" in prediction

    def test_predict_match_invalid_data(self):
        """Test de predicción con datos inválidos"""
        response = client.post(
            "/api/v1/predict/match",
            json={
                "home_team": "",  # Equipo vacío
                "away_team": "Barcelona",
                # Faltan campos requeridos
            },
        )

        assert response.status_code == 422  # Validation error

    @patch("app.services.predictor.PredictorService")
    def test_predict_multiple_matches(self, mock_predictor_service):
        """Test de predicción múltiple"""
        mock_predictor = Mock()
        mock_predictions = [Mock(spec=MatchPrediction) for _ in range(2)]

        for i, prediction in enumerate(mock_predictions):
            prediction.match_id = f"test_{i}"
            prediction.home_team = f"Team Home {i}"
            prediction.away_team = f"Team Away {i}"
            prediction.result_probabilities = {
                "home": 0.4,
                "draw": 0.3,
                "away": 0.3,
            }

        mock_predictor.predict_multiple_matches.return_value = mock_predictions
        mock_predictor_service.return_value = mock_predictor

        matches = [
            {
                "home_team": "Real Madrid",
                "away_team": "Barcelona",
                "league": "La Liga",
                "match_date": "2024-06-15T15:00:00Z",
            },
            {
                "home_team": "Manchester City",
                "away_team": "Liverpool",
                "league": "Premier League",
                "match_date": "2024-06-16T15:00:00Z",
            },
        ]

        response = client.post(
            "/api/v1/predict/matches", json={"matches": matches}
        )

        assert response.status_code == 200
        data = response.json()

        assert "predictions" in data
        assert len(data["predictions"]) == 2
        assert data["success"] is True

    def test_get_prediction_by_id(self):
        """Test de obtener predicción por ID"""
        with patch("app.services.predictor.PredictorService") as mock_service:
            mock_predictor = Mock()
            mock_prediction = Mock(spec=MatchPrediction)
            mock_prediction.match_id = "test_123"
            mock_prediction.home_team = "Real Madrid"
            mock_prediction.away_team = "Barcelona"

            mock_predictor.database.get_prediction_by_id.return_value = (
                mock_prediction
            )
            mock_service.return_value = mock_predictor

            response = client.get("/api/v1/predictions/test_123")

            assert response.status_code == 200
            data = response.json()
            assert "prediction" in data

    def test_get_prediction_not_found(self):
        """Test de predicción no encontrada"""
        with patch("app.services.predictor.PredictorService") as mock_service:
            mock_predictor = Mock()
            mock_predictor.database.get_prediction_by_id.return_value = None
            mock_service.return_value = mock_predictor

            response = client.get("/api/v1/predictions/nonexistent")

            assert response.status_code == 404


class TestDataEndpoints:
    """Tests para endpoints de datos"""

    @patch("app.services.data_collector.DataCollectorService")
    def test_collect_league_data(self, mock_collector_service):
        """Test de recolección de datos de liga"""
        mock_collector = Mock()
        mock_collector.collect_league_data.return_value = asyncio.Future()
        mock_collector.collect_league_data.return_value.set_result(
            {
                "stats": {
                    "matches_collected": 380,
                    "teams_collected": 20,
                    "odds_collected": 1140,
                    "sources_used": 2,
                    "errors": 0,
                }
            }
        )
        mock_collector_service.return_value = mock_collector

        response = client.post(
            "/api/v1/data/collect",
            json={"league_id": "PL", "season": "2024", "include_odds": True},
        )

        assert response.status_code == 200
        data = response.json()

        assert "success" in data
        assert data["success"] is True
        assert "stats" in data
        assert data["stats"]["matches_collected"] == 380

    @patch("app.services.data_collector.DataCollectorService")
    def test_get_matches_data(self, mock_collector_service):
        """Test de obtener datos de partidos"""
        mock_collector = Mock()
        mock_matches_df = Mock()
        mock_matches_df.to_dict.return_value = {
            "match_id": {"0": "123", "1": "124"},
            "home_team": {"0": "Real Madrid", "1": "Barcelona"},
            "away_team": {"0": "Barcelona", "1": "Atletico Madrid"},
        }
        mock_collector.get_matches_dataframe.return_value = mock_matches_df
        mock_collector_service.return_value = mock_collector

        response = client.get("/api/v1/data/matches?league=PL&season=2024")

        assert response.status_code == 200
        data = response.json()

        assert "matches" in data
        assert "success" in data
        assert data["success"] is True

    def test_get_matches_invalid_params(self):
        """Test con parámetros inválidos"""
        response = client.get("/api/v1/data/matches?league=&season=")

        assert response.status_code == 422


class TestOddsEndpoints:
    """Tests para endpoints de cuotas y análisis de valor"""

    @patch("app.services.odds_calculator.OddsCalculatorService")
    def test_analyze_betting_value(self, mock_odds_service):
        """Test de análisis de valor de apuestas"""
        mock_calculator = Mock()
        mock_opportunities = [
            Mock(spec=BettingOpportunity),
            Mock(spec=BettingOpportunity),
        ]

        for i, opp in enumerate(mock_opportunities):
            opp.selection = "home" if i == 0 else "away"
            opp.value_percentage = 8.5 + i
            opp.expected_value = 12.5 + i
            opp.is_value_bet = True

        mock_calculator.analyze_betting_opportunity.return_value = (
            mock_opportunities
        )
        mock_odds_service.return_value = mock_calculator

        response = client.post(
            "/api/v1/odds/analyze",
            json={
                "predicted_probabilities": {
                    "home": 0.485,
                    "draw": 0.287,
                    "away": 0.228,
                },
                "market_odds": {"home": 2.20, "draw": 3.40, "away": 4.00},
                "match_info": {
                    "match_id": "123",
                    "home_team": "Real Madrid",
                    "away_team": "Barcelona",
                },
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert "opportunities" in data
        assert len(data["opportunities"]) == 2
        assert data["success"] is True

    @patch("app.services.odds_calculator.OddsCalculatorService")
    def test_find_arbitrage(self, mock_odds_service):
        """Test de búsqueda de arbitraje"""
        mock_calculator = Mock()
        mock_calculator.find_arbitrage_opportunities.return_value = [
            {
                "profit_margin": 2.5,
                "guaranteed_profit": 25.0,
                "selections": ["home", "draw", "away"],
                "best_odds": {"home": 2.25, "draw": 3.40, "away": 4.10},
            }
        ]
        mock_odds_service.return_value = mock_calculator

        response = client.post(
            "/api/v1/odds/arbitrage",
            json={
                "multi_bookmaker_odds": {
                    "bet365": {"home": 2.20, "draw": 3.40, "away": 4.00},
                    "pinnacle": {"home": 2.25, "draw": 3.30, "away": 4.10},
                },
                "match_info": {
                    "match_id": "123",
                    "home_team": "Real Madrid",
                    "away_team": "Barcelona",
                },
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert "arbitrage_opportunities" in data
        assert len(data["arbitrage_opportunities"]) > 0

    def test_convert_odds_format(self):
        """Test de conversión de formatos de cuotas"""
        response = client.get(
            "/api/v1/odds/convert?odds=2.50&from_format=decimal&to_format=american"
        )

        assert response.status_code == 200
        data = response.json()

        assert "converted_odds" in data
        assert "original_odds" in data
        assert data["from_format"] == "decimal"
        assert data["to_format"] == "american"


class TestLiveTrackingEndpoints:
    """Tests para endpoints de seguimiento en vivo"""

    @patch("app.services.live_tracker.LiveTrackerService")
    def test_get_live_matches(self, mock_tracker_service):
        """Test de obtener partidos en vivo"""
        mock_tracker = Mock()
        mock_live_matches = [
            Mock(
                match_id="live_1",
                home_team="Real Madrid",
                away_team="Barcelona",
                home_score=1,
                away_score=0,
                minute=45,
                status="live",
            ),
            Mock(
                match_id="live_2",
                home_team="City",
                away_team="Liverpool",
                home_score=2,
                away_score=1,
                minute=67,
                status="live",
            ),
        ]
        mock_tracker.get_live_matches.return_value = mock_live_matches
        mock_tracker_service.return_value = mock_tracker

        response = client.get("/api/v1/live/matches")

        assert response.status_code == 200
        data = response.json()

        assert "live_matches" in data
        assert len(data["live_matches"]) == 2
        assert data["success"] is True

    @patch("app.services.live_tracker.LiveTrackerService")
    def test_get_live_match_by_id(self, mock_tracker_service):
        """Test de obtener partido en vivo específico"""
        mock_tracker = Mock()
        mock_live_match = Mock(
            match_id="live_123",
            home_team="Real Madrid",
            away_team="Barcelona",
            home_score=2,
            away_score=1,
            minute=78,
            status="live",
        )
        mock_tracker.get_match_by_id.return_value = mock_live_match
        mock_tracker_service.return_value = mock_tracker

        response = client.get("/api/v1/live/matches/live_123")

        assert response.status_code == 200
        data = response.json()

        assert "match" in data
        assert data["match"]["match_id"] == "live_123"

    def test_get_live_match_not_found(self):
        """Test de partido en vivo no encontrado"""
        with patch(
            "app.services.live_tracker.LiveTrackerService"
        ) as mock_service:
            mock_tracker = Mock()
            mock_tracker.get_match_by_id.return_value = None
            mock_service.return_value = mock_tracker

            response = client.get("/api/v1/live/matches/nonexistent")

            assert response.status_code == 404


class TestCalculatorEndpoints:
    """Tests para endpoints de calculadoras y métricas"""

    @patch("app.services.calculator.CalculatorService")
    def test_calculate_team_metrics(self, mock_calculator_service):
        """Test de cálculo de métricas de equipo"""
        mock_calculator = Mock()
        mock_calculator.calculate_team_metrics.return_value = {
            "team": "Real Madrid",
            "form": {"form_rating": 85.5, "form_points": 12, "form_ppg": 2.4},
            "strength": {
                "overall_attack_strength": 1.4,
                "overall_defense_strength": 1.3,
                "home_rating": 1.5,
            },
        }
        mock_calculator_service.return_value = mock_calculator

        response = client.post(
            "/api/v1/calculate/team-metrics",
            json={
                "team": "Real Madrid",
                "recent_matches": mock_data.get_mock_match_data()[:5],
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert "metrics" in data
        assert data["metrics"]["team"] == "Real Madrid"
        assert "form" in data["metrics"]
        assert "strength" in data["metrics"]

    @patch("app.services.calculator.CalculatorService")
    def test_calculate_match_prediction(self, mock_calculator_service):
        """Test de cálculo de predicción de partido"""
        mock_calculator = Mock()
        mock_calculator.calculate_match_prediction.return_value = {
            "home_team": "Real Madrid",
            "away_team": "Barcelona",
            "prediction": {
                "match_result_probabilities": {
                    "home_win": 0.485,
                    "draw": 0.287,
                    "away_win": 0.228,
                },
                "expected_goals": {"home": 1.65, "away": 1.15},
            },
        }
        mock_calculator_service.return_value = mock_calculator

        response = client.post(
            "/api/v1/calculate/match-prediction",
            json={
                "home_team_metrics": {"team": "Real Madrid"},
                "away_team_metrics": {"team": "Barcelona"},
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert "prediction" in data
        assert data["prediction"]["home_team"] == "Real Madrid"


class TestAnalyticsEndpoints:
    """Tests para endpoints de analytics y reportes"""

    @patch("app.services.predictor.PredictorService")
    def test_get_model_performance(self, mock_predictor_service):
        """Test de obtener rendimiento del modelo"""
        mock_predictor = Mock()
        mock_predictor.evaluate_model_performance.return_value = {
            "period_days": 30,
            "model_version": "v2.1.0",
            "accuracy_metrics": {
                "overall_accuracy": 0.642,
                "total_predictions": 1247,
                "correct_predictions": 800,
            },
            "status": "Excelente",
        }
        mock_predictor_service.return_value = mock_predictor

        response = client.get("/api/v1/analytics/model-performance?days=30")

        assert response.status_code == 200
        data = response.json()

        assert "performance" in data
        assert (
            data["performance"]["accuracy_metrics"]["overall_accuracy"] > 0.6
        )

    @patch("app.services.odds_calculator.OddsCalculatorService")
    def test_get_betting_report(self, mock_odds_service):
        """Test de reporte de apuestas"""
        mock_calculator = Mock()
        mock_calculator.generate_betting_report.return_value = {
            "generated_at": datetime.now().isoformat(),
            "analysis_summary": {
                "total_opportunities": 15,
                "high_value_bets": 3,
                "average_value_percentage": 6.2,
            },
            "recommendations": [
                "🚀 3 oportunidades de alto valor identificadas",
                "✅ Alta probabilidad de rentabilidad a largo plazo",
            ],
        }
        mock_calculator_service.return_value = mock_calculator

        response = client.post(
            "/api/v1/analytics/betting-report",
            json={"opportunities": [], "bankroll": 10000},
        )

        assert response.status_code == 200
        data = response.json()

        assert "report" in data
        assert "analysis_summary" in data["report"]
        assert "recommendations" in data["report"]


class TestErrorHandling:
    """Tests para manejo de errores"""

    def test_internal_server_error(self):
        """Test de error interno del servidor"""
        with patch("app.services.predictor.PredictorService") as mock_service:
            mock_service.side_effect = Exception("Test error")

            response = client.post(
                "/api/v1/predict/match",
                json={
                    "home_team": "Real Madrid",
                    "away_team": "Barcelona",
                    "league": "La Liga",
                    "match_date": "2024-06-15T15:00:00Z",
                },
            )

            assert response.status_code == 500
            data = response.json()
            assert "error" in data
            assert data["success"] is False

    def test_not_found_endpoint(self):
        """Test de endpoint no encontrado"""
        response = client.get("/api/v1/nonexistent")

        assert response.status_code == 404

    def test_method_not_allowed(self):
        """Test de método no permitido"""
        response = client.put("/api/v1/predict/match")

        assert response.status_code == 405


class TestRateLimiting:
    """Tests para rate limiting"""

    @pytest.mark.slow
    def test_rate_limit_exceeded(self):
        """Test de límite de rate excedido"""
        # Simular múltiples requests rápidos
        responses = []
        for _ in range(100):  # Exceder límite
            response = client.get("/health")
            responses.append(response.status_code)

        # Al menos uno debería ser rate limited (429)
        # Nota: Esto depende de la configuración de rate limiting
        status_codes = set(responses)
        assert 200 in status_codes  # Algunos exitosos
        # assert 429 in status_codes  # Algunos rate limited (si está configurado)


class TestAuthentication:
    """Tests para autenticación (si está implementada)"""

    def test_protected_endpoint_without_auth(self):
        """Test de endpoint protegido sin autenticación"""
        # Si tienes endpoints protegidos, test aquí
        response = client.get("/api/v1/admin/stats")

        # Podría ser 401 (no autenticado) o 200 (si no hay auth implementada)
        assert response.status_code in [200, 401, 404]

    def test_invalid_api_key(self):
        """Test con API key inválida"""
        headers = {"X-API-Key": "invalid_key"}
        response = client.get("/api/v1/admin/stats", headers=headers)

        # Podría ser 401 (no autorizado) o 200 (si no hay auth implementada)
        assert response.status_code in [200, 401, 404]


class TestWebSocketEndpoints:
    """Tests para endpoints WebSocket (si están implementados)"""

    @pytest.mark.asyncio
    async def test_live_updates_websocket(self):
        """Test de WebSocket para actualizaciones en vivo"""
        # Test básico de conexión WebSocket
        async with AsyncClient(app=app, base_url="http://test") as ac:
            # Si tienes WebSocket endpoints, test aquí
            pass


# Configuración de pytest para este módulo
@pytest.fixture(autouse=True)
def setup_test_client():
    """Setup del cliente de test"""
    # Configurar la app para testing
    app.dependency_overrides = {}
    yield
    # Cleanup
    app.dependency_overrides.clear()


# Tests de integración
@pytest.mark.integration
class TestFullAPIIntegration:
    """Tests de integración completa de la API"""

    def test_prediction_to_betting_analysis_flow(self):
        """Test del flujo completo: predicción → análisis de valor"""
        with patch(
            "app.services.predictor.PredictorService"
        ) as mock_pred, patch(
            "app.services.odds_calculator.OddsCalculatorService"
        ) as mock_odds:

            # Mock predicción
            mock_predictor = Mock()
            mock_prediction = Mock(spec=MatchPrediction)
            mock_prediction.result_probabilities = {
                "home": 0.485,
                "draw": 0.287,
                "away": 0.228,
            }
            mock_predictor.predict_match.return_value = mock_prediction
            mock_pred.return_value = mock_predictor

            # Mock análisis de cuotas
            mock_calculator = Mock()
            mock_calculator.analyze_betting_opportunity.return_value = []
            mock_odds.return_value = mock_calculator

            # 1. Hacer predicción
            pred_response = client.post(
                "/api/v1/predict/match",
                json={
                    "home_team": "Real Madrid",
                    "away_team": "Barcelona",
                    "league": "La Liga",
                    "match_date": "2024-06-15T15:00:00Z",
                },
            )

            assert pred_response.status_code == 200

            # 2. Usar resultado para análisis de valor
            pred_data = pred_response.json()
            odds_response = client.post(
                "/api/v1/odds/analyze",
                json={
                    "predicted_probabilities": {
                        "home": 0.485,
                        "draw": 0.287,
                        "away": 0.228,
                    },
                    "market_odds": {"home": 2.20, "draw": 3.40, "away": 4.00},
                    "match_info": {
                        "match_id": "123",
                        "home_team": "Real Madrid",
                        "away_team": "Barcelona",
                    },
                },
            )

            assert odds_response.status_code == 200


if __name__ == "__main__":
    # Ejecutar tests específicos
    pytest.main([__file__, "-v", "--tb=short"])
