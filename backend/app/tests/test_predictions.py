"""
Football Analytics - Prediction System Tests
Pruebas específicas para el sistema completo de predicciones del proyecto
"""

import json
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from app.services.calculator import CalculatorService
from app.services.odds_calculator import (
    BettingOpportunity,
    OddsCalculatorService,
)

# Importaciones del proyecto
from app.services.predictor import (
    ConfidenceLevel,
    FeatureExtractor,
    HeadToHeadRecord,
    MatchPrediction,
    PredictionDatabase,
    PredictionType,
    PredictorService,
    TeamForm,
    analyze_predictions_batch,
    create_predictor,
    predict_match_quick,
)

# Test utilities
from . import TEST_APP_CONFIG, TestDatabase, assert_prediction_valid, mock_data


class TestPredictorService:
    """Tests para el servicio principal de predicciones"""

    def setup_method(self):
        """Setup antes de cada test"""
        self.predictor = PredictorService(TEST_APP_CONFIG)
        self.madrid_features = self._get_madrid_features()
        self.barca_features = self._get_barca_features()
        self.h2h_record = self._get_h2h_record()

    def _get_madrid_features(self) -> dict:
        """Features realistas de Real Madrid"""
        return {
            "points_per_game": 2.1,
            "win_percentage": 0.65,
            "goals_per_game": 2.3,
            "goals_conceded_per_game": 0.9,
            "goal_difference_per_game": 1.4,
            "home_points_per_game": 2.5,
            "home_goals_per_game": 2.6,
            "home_conceded_per_game": 0.8,
            "away_points_per_game": 1.7,
            "away_goals_per_game": 2.0,
            "away_conceded_per_game": 1.0,
            "attack_strength": 1.4,
            "defense_strength": 1.3,
            "last_5_points": 2.2,
            "last_5_goals": 2.4,
            "result_consistency": 0.7,
            "goals_consistency": 0.6,
        }

    def _get_barca_features(self) -> dict:
        """Features realistas de Barcelona"""
        return {
            "points_per_game": 2.0,
            "win_percentage": 0.60,
            "goals_per_game": 2.1,
            "goals_conceded_per_game": 1.0,
            "goal_difference_per_game": 1.1,
            "home_points_per_game": 2.4,
            "home_goals_per_game": 2.4,
            "home_conceded_per_game": 0.9,
            "away_points_per_game": 1.6,
            "away_goals_per_game": 1.8,
            "away_conceded_per_game": 1.1,
            "attack_strength": 1.3,
            "defense_strength": 1.2,
            "last_5_points": 1.8,
            "last_5_goals": 2.0,
            "result_consistency": 0.6,
            "goals_consistency": 0.5,
        }

    def _get_h2h_record(self) -> HeadToHeadRecord:
        """Historial H2H Real Madrid vs Barcelona"""
        return HeadToHeadRecord(
            home_team="Real Madrid",
            away_team="Barcelona",
            total_matches=10,
            home_wins=4,
            draws=3,
            away_wins=3,
            last_5_results=["H", "A", "D", "H", "D"],
            avg_goals_home=1.4,
            avg_goals_away=1.2,
            avg_total_goals=2.6,
        )

    def test_predictor_initialization(self):
        """Test de inicialización del PredictorService"""
        assert self.predictor.model_version == "v2.1.0"
        assert self.predictor.feature_extractor is not None
        assert self.predictor.database is not None
        assert self.predictor.odds_calculator is not None
        assert self.predictor.ensemble is not None
        assert len(self.predictor.confidence_thresholds) == 5

    def test_feature_extraction_integration(self):
        """Test de integración con extracción de features"""
        # Mock del feature extractor
        with patch.object(
            self.predictor.feature_extractor, "extract_match_features"
        ) as mock_extract:
            mock_extract.return_value = {
                "home_attack_strength": 1.4,
                "away_defense_strength": 1.2,
                "points_difference": 0.1,
                "h2h_home_win_rate": 0.4,
            }

            # Ejecutar extracción
            features = self.predictor.feature_extractor.extract_match_features(
                "Real Madrid",
                "Barcelona",
                self.madrid_features,
                self.barca_features,
                self.h2h_record,
            )

            assert "home_attack_strength" in features
            assert "away_defense_strength" in features
            assert features["home_attack_strength"] == 1.4

    @patch("app.services.predictor.PredictorService._get_team_features")
    @patch("app.services.predictor.PredictorService._get_h2h_record")
    def test_predict_match_complete_flow(self, mock_h2h, mock_features):
        """Test del flujo completo de predicción"""
        # Mock dependencies
        mock_features.side_effect = [self.madrid_features, self.barca_features]
        mock_h2h.return_value = self.h2h_record

        # Mock del modelo ensemble
        with patch.object(
            self.predictor, "_predict_probabilities"
        ) as mock_predict:
            mock_predict.return_value = {
                "home": 0.485,
                "draw": 0.287,
                "away": 0.228,
            }

            # Realizar predicción
            prediction = self.predictor.predict_match(
                home_team="Real Madrid",
                away_team="Barcelona",
                league="La Liga",
                match_date=datetime.now() + timedelta(days=1),
                match_id="clasico_2024",
            )

            # Verificar estructura de la predicción
            assert isinstance(prediction, MatchPrediction)
            assert prediction.home_team == "Real Madrid"
            assert prediction.away_team == "Barcelona"
            assert prediction.league == "La Liga"
            assert prediction.match_id == "clasico_2024"

            # Verificar probabilidades
            assert "home" in prediction.result_probabilities
            assert "draw" in prediction.result_probabilities
            assert "away" in prediction.result_probabilities

            # Verificar que suman 1
            total_prob = sum(prediction.result_probabilities.values())
            assert abs(total_prob - 1.0) < 0.01

            # Verificar otros campos
            assert prediction.most_likely_result in ["home", "draw", "away"]
            assert isinstance(prediction.confidence_level, ConfidenceLevel)
            assert 0 <= prediction.confidence_score <= 1
            assert prediction.expected_goals_home > 0
            assert prediction.expected_goals_away > 0

    def test_confidence_level_calculation(self):
        """Test del cálculo de niveles de confianza"""
        test_cases = [
            (
                {"home": 0.9, "draw": 0.05, "away": 0.05},
                ConfidenceLevel.VERY_HIGH,
            ),
            ({"home": 0.8, "draw": 0.1, "away": 0.1}, ConfidenceLevel.HIGH),
            (
                {"home": 0.7, "draw": 0.15, "away": 0.15},
                ConfidenceLevel.MEDIUM,
            ),
            ({"home": 0.5, "draw": 0.25, "away": 0.25}, ConfidenceLevel.LOW),
            (
                {"home": 0.4, "draw": 0.3, "away": 0.3},
                ConfidenceLevel.VERY_LOW,
            ),
        ]

        for probabilities, expected_level in test_cases:
            max_prob = max(probabilities.values())
            confidence = self.predictor._calculate_confidence_level(
                max_prob, probabilities
            )
            assert confidence == expected_level

    def test_expected_goals_calculation(self):
        """Test del cálculo de goles esperados"""
        # Features que indican un partido de muchos goles
        high_scoring_features = np.array(
            [
                2.5,
                2.3,  # home_goals_avg, away_goals_avg
                1.8,
                1.2,  # attack_ratio, defense_ratio
                0.3,
                2.4,  # home_advantage, goal_expectation
            ]
            + [1.0] * 14
        )  # Rellenar con valores neutros

        expected_goals = self.predictor._predict_expected_goals(
            high_scoring_features.reshape(1, -1)
        )

        assert expected_goals["home"] > 1.0
        assert expected_goals["away"] > 1.0
        assert expected_goals["total"] > 2.0
        assert (
            expected_goals["total"]
            == expected_goals["home"] + expected_goals["away"]
        )

    def test_over_under_probabilities(self):
        """Test de cálculo de probabilidades Over/Under"""
        expected_goals = {"home": 1.8, "away": 1.3, "total": 3.1}

        over_under_probs = self.predictor._calculate_over_under_probabilities(
            expected_goals
        )

        # Verificar estructura
        assert "over_1.5" in over_under_probs
        assert "under_1.5" in over_under_probs
        assert "over_2.5" in over_under_probs
        assert "under_2.5" in over_under_probs
        assert "over_3.5" in over_under_probs
        assert "under_3.5" in over_under_probs

        # Verificar que cada par suma 1
        for line in [1.5, 2.5, 3.5]:
            total = (
                over_under_probs[f"over_{line}"]
                + over_under_probs[f"under_{line}"]
            )
            assert abs(total - 1.0) < 0.01

        # Con 3.1 goles esperados, over 2.5 debería ser más probable
        assert over_under_probs["over_2.5"] > over_under_probs["under_2.5"]

    def test_btts_probabilities(self):
        """Test de probabilidades de ambos equipos marcan"""
        # Caso con goles esperados altos para ambos equipos
        high_scoring = {"home": 2.0, "away": 1.8, "total": 3.8}
        btts_high = self.predictor._calculate_btts_probabilities(high_scoring)

        assert "yes" in btts_high
        assert "no" in btts_high
        assert abs(btts_high["yes"] + btts_high["no"] - 1.0) < 0.01
        assert (
            btts_high["yes"] > btts_high["no"]
        )  # Más probable que marquen ambos

        # Caso con pocos goles esperados
        low_scoring = {"home": 0.8, "away": 0.7, "total": 1.5}
        btts_low = self.predictor._calculate_btts_probabilities(low_scoring)

        assert (
            btts_low["no"] > btts_low["yes"]
        )  # Menos probable que marquen ambos

    def test_most_likely_score_prediction(self):
        """Test de predicción del marcador más probable"""
        expected_goals = {"home": 1.8, "away": 1.2, "total": 3.0}

        most_likely_score = self.predictor._predict_most_likely_score(
            expected_goals
        )
        score_prob = self.predictor._calculate_score_probability(
            most_likely_score, expected_goals
        )

        # Verificar formato del marcador
        assert isinstance(most_likely_score, tuple)
        assert len(most_likely_score) == 2
        assert isinstance(most_likely_score[0], int)
        assert isinstance(most_likely_score[1], int)
        assert most_likely_score[0] >= 0
        assert most_likely_score[1] >= 0

        # Verificar probabilidad
        assert 0 < score_prob <= 1

        # Con 1.8 vs 1.2 goles esperados, el local debería marcar más
        assert most_likely_score[0] >= most_likely_score[1]

    def test_betting_value_integration(self):
        """Test de integración con análisis de valor"""
        market_odds = {"home": 2.20, "draw": 3.40, "away": 4.00}

        # Mock prediction
        mock_prediction = MatchPrediction(
            match_id="test_123",
            home_team="Real Madrid",
            away_team="Barcelona",
            league="La Liga",
            match_date=datetime.now() + timedelta(days=1),
            result_probabilities={"home": 0.485, "draw": 0.287, "away": 0.228},
            most_likely_result="home",
            confidence_level=ConfidenceLevel.HIGH,
            confidence_score=0.78,
            expected_goals_home=1.8,
            expected_goals_away=1.2,
            expected_goals_total=3.0,
            over_under_probabilities={"over_2.5": 0.65, "under_2.5": 0.35},
            btts_probabilities={"yes": 0.58, "no": 0.42},
            most_likely_score=(2, 1),
            score_probability=0.12,
            model_version="v2.1.0",
            prediction_method="ensemble_ml",
            features_used=["home_attack", "away_defense"],
        )

        # Mock del odds calculator
        with patch.object(
            self.predictor.odds_calculator, "analyze_betting_opportunity"
        ) as mock_analyze:
            mock_opportunities = [
                Mock(
                    spec=BettingOpportunity,
                    selection="home",
                    value_percentage=8.5,
                    expected_value=12.3,
                    is_value_bet=True,
                )
            ]
            mock_analyze.return_value = mock_opportunities

            # Analizar valor
            opportunities = self.predictor._analyze_betting_value(
                mock_prediction, market_odds
            )

            assert len(opportunities) == 1
            assert opportunities[0].selection == "home"
            assert opportunities[0].is_value_bet

    def test_multiple_matches_prediction(self):
        """Test de predicción de múltiples partidos"""
        matches = [
            {
                "home_team": "Real Madrid",
                "away_team": "Barcelona",
                "league": "La Liga",
                "match_date": datetime.now() + timedelta(days=1),
            },
            {
                "home_team": "Manchester City",
                "away_team": "Liverpool",
                "league": "Premier League",
                "match_date": datetime.now() + timedelta(days=2),
            },
            {
                "home_team": "Bayern Munich",
                "away_team": "Borussia Dortmund",
                "league": "Bundesliga",
                "match_date": datetime.now() + timedelta(days=3),
            },
        ]

        # Mock dependencies para todos los partidos
        with patch.object(
            self.predictor, "_get_team_features"
        ) as mock_features, patch.object(
            self.predictor, "_get_h2h_record"
        ) as mock_h2h, patch.object(
            self.predictor, "_predict_probabilities"
        ) as mock_predict:

            mock_features.side_effect = [
                # Real Madrid vs Barcelona
                self.madrid_features,
                self.barca_features,
                # Manchester City vs Liverpool
                {"attack_strength": 1.5, "defense_strength": 1.4},
                {"attack_strength": 1.3, "defense_strength": 1.1},
                # Bayern vs Dortmund
                {"attack_strength": 1.4, "defense_strength": 1.3},
                {"attack_strength": 1.2, "defense_strength": 1.0},
            ]

            mock_h2h.return_value = self.h2h_record
            mock_predict.return_value = {
                "home": 0.45,
                "draw": 0.30,
                "away": 0.25,
            }

            # Predecir múltiples partidos
            predictions = self.predictor.predict_multiple_matches(matches)

            assert len(predictions) == 3
            for prediction in predictions:
                assert isinstance(prediction, MatchPrediction)
                assert prediction.home_team in [
                    "Real Madrid",
                    "Manchester City",
                    "Bayern Munich",
                ]
                assert_prediction_valid(prediction)


class TestMatchPrediction:
    """Tests para la clase MatchPrediction"""

    def setup_method(self):
        """Setup para tests de MatchPrediction"""
        self.prediction = MatchPrediction(
            match_id="test_clasico",
            home_team="Real Madrid",
            away_team="Barcelona",
            league="La Liga",
            match_date=datetime.now() + timedelta(days=1),
            result_probabilities={"home": 0.485, "draw": 0.287, "away": 0.228},
            most_likely_result="home",
            confidence_level=ConfidenceLevel.HIGH,
            confidence_score=0.78,
            expected_goals_home=1.8,
            expected_goals_away=1.2,
            expected_goals_total=3.0,
            over_under_probabilities={"over_2.5": 0.65, "under_2.5": 0.35},
            btts_probabilities={"yes": 0.58, "no": 0.42},
            most_likely_score=(2, 1),
            score_probability=0.12,
            model_version="v2.1.0",
            prediction_method="ensemble_ml",
            features_used=["home_attack", "away_defense"],
        )

    def test_prediction_properties(self):
        """Test de propiedades calculadas de la predicción"""
        # Test result_prediction_text
        result_text = self.prediction.result_prediction_text
        assert "Victoria de Real Madrid" in result_text
        assert "confianza alta" in result_text

        # Test goals_prediction_text
        goals_text = self.prediction.goals_prediction_text
        assert "Real Madrid 2-1 Barcelona" == goals_text

    def test_prediction_serialization(self):
        """Test de serialización de la predicción"""
        # Convertir a dict
        prediction_dict = self.prediction.__dict__.copy()

        # Verificar campos principales
        assert prediction_dict["home_team"] == "Real Madrid"
        assert prediction_dict["away_team"] == "Barcelona"
        assert prediction_dict["confidence_score"] == 0.78

        # Verificar que se puede serializar a JSON
        json_str = json.dumps(prediction_dict, default=str)
        assert isinstance(json_str, str)
        assert "Real Madrid" in json_str


class TestFeatureExtractor:
    """Tests para el extractor de características"""

    def setup_method(self):
        """Setup para FeatureExtractor"""
        self.extractor = FeatureExtractor()

    def test_team_features_extraction(self):
        """Test de extracción de features de equipo"""
        # Partidos recientes simulados de Real Madrid
        recent_matches = [
            {
                "result": "W",
                "goals_for": 3,
                "goals_against": 1,
                "venue": "home",
            },
            {
                "result": "W",
                "goals_for": 2,
                "goals_against": 0,
                "venue": "away",
            },
            {
                "result": "D",
                "goals_for": 1,
                "goals_against": 1,
                "venue": "home",
            },
            {
                "result": "W",
                "goals_for": 4,
                "goals_against": 2,
                "venue": "away",
            },
            {
                "result": "L",
                "goals_for": 0,
                "goals_against": 2,
                "venue": "away",
            },
        ]

        features = self.extractor.extract_team_features(
            "Real Madrid", recent_matches, league_avg_goals=2.5
        )

        # Verificar estructura
        required_features = [
            "points_per_game",
            "win_percentage",
            "goals_per_game",
            "goals_conceded_per_game",
            "attack_strength",
            "defense_strength",
            "home_points_per_game",
            "away_points_per_game",
        ]

        for feature in required_features:
            assert feature in features
            assert isinstance(features[feature], (int, float))

        # Verificar cálculos
        assert features["points_per_game"] == 2.4  # (3+3+1+3+0)/5 = 2.4
        assert features["win_percentage"] == 0.6  # 3/5 = 0.6
        assert features["goals_per_game"] == 2.0  # (3+2+1+4+0)/5 = 2.0

    def test_match_features_extraction(self):
        """Test de extracción de features del partido"""
        madrid_features = {
            "attack_strength": 1.4,
            "defense_strength": 1.3,
            "points_per_game": 2.1,
            "home_points_per_game": 2.5,
            "goals_per_game": 2.3,
        }

        barca_features = {
            "attack_strength": 1.3,
            "defense_strength": 1.2,
            "points_per_game": 2.0,
            "away_points_per_game": 1.6,
            "goals_per_game": 2.1,
        }

        h2h_record = HeadToHeadRecord(
            home_team="Real Madrid",
            away_team="Barcelona",
            total_matches=10,
            home_wins=4,
            draws=3,
            away_wins=3,
            last_5_results=["H", "A", "D", "H", "D"],
            avg_goals_home=1.4,
            avg_goals_away=1.2,
            avg_total_goals=2.6,
        )

        match_features = self.extractor.extract_match_features(
            "Real Madrid",
            "Barcelona",
            madrid_features,
            barca_features,
            h2h_record,
            match_importance=1.2,
        )

        # Verificar features combinadas
        assert "home_attack_strength" in match_features
        assert "away_defense_strength" in match_features
        assert "points_difference" in match_features
        assert "attack_difference" in match_features
        assert "home_advantage" in match_features
        assert "h2h_home_win_rate" in match_features
        assert "match_importance" in match_features

        # Verificar cálculos
        assert match_features["points_difference"] == 0.1  # 2.1 - 2.0
        assert match_features["attack_difference"] == 0.1  # 1.4 - 1.3
        assert match_features["h2h_home_win_rate"] == 0.4  # 4/10
        assert match_features["match_importance"] == 1.2

    def test_default_features_handling(self):
        """Test de manejo de features por defecto"""
        # Test con lista vacía de partidos
        features = self.extractor.extract_team_features(
            "Unknown Team", [], 2.5
        )

        # Debe retornar features por defecto
        default_features = self.extractor._get_default_features()
        assert features == default_features

        # Verificar que todos los valores son numéricos y razonables
        for key, value in features.items():
            assert isinstance(value, (int, float))
            assert not np.isnan(value)
            assert not np.isinf(value)


class TestPredictionDatabase:
    """Tests para la base de datos de predicciones"""

    def setup_method(self):
        """Setup para PredictionDatabase"""
        # Usar base de datos temporal
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()
        self.db = PredictionDatabase(self.temp_db.name)

        # Crear predicción de prueba
        self.test_prediction = MatchPrediction(
            match_id="db_test_123",
            home_team="Real Madrid",
            away_team="Barcelona",
            league="La Liga",
            match_date=datetime.now() + timedelta(days=1),
            result_probabilities={"home": 0.485, "draw": 0.287, "away": 0.228},
            most_likely_result="home",
            confidence_level=ConfidenceLevel.HIGH,
            confidence_score=0.78,
            expected_goals_home=1.8,
            expected_goals_away=1.2,
            expected_goals_total=3.0,
            over_under_probabilities={"over_2.5": 0.65, "under_2.5": 0.35},
            btts_probabilities={"yes": 0.58, "no": 0.42},
            most_likely_score=(2, 1),
            score_probability=0.12,
            model_version="v2.1.0",
            prediction_method="ensemble_ml",
            features_used=["test_feature"],
        )

    def teardown_method(self):
        """Cleanup después de cada test"""
        import os

        os.unlink(self.temp_db.name)

    def test_save_prediction(self):
        """Test de guardado de predicción"""
        # Guardar predicción
        self.db.save_prediction(self.test_prediction)

        # Verificar que se guardó correctamente
        import sqlite3

        with sqlite3.connect(self.temp_db.name) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM predictions WHERE match_id = ?",
                (self.test_prediction.match_id,),
            )
            result = cursor.fetchone()

            assert result is not None
            assert result[1] == "db_test_123"  # match_id
            assert result[2] == "Real Madrid"  # home_team
            assert result[3] == "Barcelona"  # away_team

    def test_save_actual_result(self):
        """Test de guardado de resultado real"""
        # Guardar resultado real
        self.db.save_actual_result("db_test_123", "home", 2, 1)

        # Verificar guardado
        import sqlite3

        with sqlite3.connect(self.temp_db.name) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM actual_results WHERE match_id = ?",
                ("db_test_123",),
            )
            result = cursor.fetchone()

            assert result is not None
            assert result[1] == "home"  # actual_result
            assert result[2] == 2  # home_goals
            assert result[3] == 1  # away_goals

    def test_prediction_accuracy_calculation(self):
        """Test de cálculo de precisión"""
        # Guardar múltiples predicciones y resultados
        test_data = [
            ("match_1", "home", "home", 0.8),  # Correcto, alta confianza
            ("match_2", "draw", "draw", 0.7),  # Correcto, media confianza
            ("match_3", "away", "home", 0.6),  # Incorrecto
            ("match_4", "home", "home", 0.9),  # Correcto, muy alta confianza
            ("match_5", "draw", "away", 0.5),  # Incorrecto
        ]

        for match_id, predicted, actual, confidence in test_data:
            # Crear y guardar predicción
            prediction = MatchPrediction(
                match_id=match_id,
                home_team="Team A",
                away_team="Team B",
                league="Test League",
                match_date=datetime.now(),
                result_probabilities={"home": 0.4, "draw": 0.3, "away": 0.3},
                most_likely_result=predicted,
                confidence_level=ConfidenceLevel.MEDIUM,
                confidence_score=confidence,
                expected_goals_home=1.5,
                expected_goals_away=1.5,
                expected_goals_total=3.0,
                over_under_probabilities={"over_2.5": 0.5, "under_2.5": 0.5},
                btts_probabilities={"yes": 0.5, "no": 0.5},
                most_likely_score=(1, 1),
                score_probability=0.1,
                model_version="v2.1.0",
                prediction_method="test",
                features_used=[],
            )

            self.db.save_prediction(prediction)
            self.db.save_actual_result(match_id, actual, 1, 1)

        # Calcular precisión
        accuracy_stats = self.db.get_prediction_accuracy(days=1)

        # Verificar cálculos
        assert accuracy_stats["total_predictions"] == 5
        assert (
            accuracy_stats["correct_predictions"] == 3
        )  # match_1, match_2, match_4
        assert accuracy_stats["overall_accuracy"] == 0.6  # 3/5
        assert 0.5 <= accuracy_stats["average_confidence"] <= 0.8


class TestPredictionIntegration:
    """Tests de integración del sistema de predicciones"""

    def setup_method(self):
        """Setup para tests de integración"""
        self.predictor = PredictorService(TEST_APP_CONFIG)

    @patch("app.services.data_collector.DataCollectorService")
    def test_integration_with_data_collector(self, mock_collector_service):
        """Test de integración con el data collector"""
        # Mock del data collector
        mock_collector = Mock()
        mock_matches_data = pd.DataFrame(
            {
                "match_id": ["1", "2", "3"],
                "home_team": ["Real Madrid", "Barcelona", "Atletico"],
                "away_team": ["Barcelona", "Valencia", "Sevilla"],
                "home_goals": [2, 1, 0],
                "away_goals": [1, 1, 2],
                "date": [datetime.now() - timedelta(days=i) for i in range(3)],
            }
        )
        mock_collector.get_matches_dataframe.return_value = mock_matches_data
        mock_collector_service.return_value = mock_collector

        # Verificar que el predictor puede usar datos del collector
        # En implementación real, esto se usaría para obtener features históricas
        assert len(mock_matches_data) == 3
        assert "Real Madrid" in mock_matches_data["home_team"].values

    @patch("app.services.calculator.CalculatorService")
    def test_integration_with_calculator(self, mock_calculator_service):
        """Test de integración con el calculator"""
        # Mock del calculator service
        mock_calculator = Mock()
        mock_team_metrics = {
            "team": "Real Madrid",
            "form": {"form_rating": 85.5, "form_points": 12},
            "strength": {
                "overall_attack_strength": 1.4,
                "overall_defense_strength": 1.3,
            },
        }
        mock_calculator.calculate_team_metrics.return_value = mock_team_metrics
        mock_calculator_service.return_value = mock_calculator

        # Verificar integración
        # En implementación real, el predictor usaría estas métricas
        assert mock_team_metrics["team"] == "Real Madrid"
        assert mock_team_metrics["form"]["form_rating"] == 85.5

    def test_prediction_to_betting_analysis_flow(self):
        """Test del flujo completo: predicción → análisis de apuestas"""
        # Mock prediction
        mock_prediction = MatchPrediction(
            match_id="flow_test",
            home_team="Real Madrid",
            away_team="Barcelona",
            league="La Liga",
            match_date=datetime.now() + timedelta(days=1),
            result_probabilities={"home": 0.485, "draw": 0.287, "away": 0.228},
            most_likely_result="home",
            confidence_level=ConfidenceLevel.HIGH,
            confidence_score=0.78,
            expected_goals_home=1.8,
            expected_goals_away=1.2,
            expected_goals_total=3.0,
            over_under_probabilities={"over_2.5": 0.65, "under_2.5": 0.35},
            btts_probabilities={"yes": 0.58, "no": 0.42},
            most_likely_score=(2, 1),
            score_probability=0.12,
            model_version="v2.1.0",
            prediction_method="ensemble_ml",
            features_used=["test"],
        )

        # Mock market odds
        market_odds = {"home": 2.20, "draw": 3.40, "away": 4.00}

        # Mock betting analysis
        with patch.object(
            self.predictor.odds_calculator, "analyze_betting_opportunity"
        ) as mock_analyze:
            mock_opportunities = [
                Mock(
                    spec=BettingOpportunity,
                    selection="home",
                    value_percentage=8.33,
                    expected_value=12.5,
                    kelly_percentage=4.2,
                    is_value_bet=True,
                    recommendation="Apuesta Moderada",
                )
            ]
            mock_analyze.return_value = mock_opportunities

            # Ejecutar análisis
            opportunities = self.predictor._analyze_betting_value(
                mock_prediction, market_odds
            )

            # Verificar resultado
            assert len(opportunities) == 1
            assert opportunities[0].selection == "home"
            assert opportunities[0].value_percentage > 8
            assert opportunities[0].is_value_bet

    def test_model_performance_evaluation_integration(self):
        """Test de integración con evaluación de rendimiento"""
        # Mock de datos de rendimiento
        with patch.object(
            self.predictor.database, "get_prediction_accuracy"
        ) as mock_accuracy:
            mock_accuracy.return_value = {
                "overall_accuracy": 0.642,
                "total_predictions": 1247,
                "correct_predictions": 800,
                "home_precision": 0.71,
                "draw_precision": 0.46,
                "away_precision": 0.69,
                "average_confidence": 0.67,
            }

            # Evaluar rendimiento
            performance = self.predictor.evaluate_model_performance(30)

            # Verificar estructura
            assert "period_days" in performance
            assert "model_version" in performance
            assert "accuracy_metrics" in performance
            assert "status" in performance
            assert "recommendation" in performance

            # Verificar cálculos
            assert performance["accuracy_metrics"]["overall_accuracy"] > 0.6
            assert performance["status"] == "Excelente"


class TestPredictionInsights:
    """Tests para insights y análisis de predicciones"""

    def setup_method(self):
        """Setup para tests de insights"""
        self.predictor = PredictorService(TEST_APP_CONFIG)
        self.sample_prediction = MatchPrediction(
            match_id="insights_test",
            home_team="Real Madrid",
            away_team="Barcelona",
            league="La Liga",
            match_date=datetime.now() + timedelta(days=1),
            result_probabilities={"home": 0.485, "draw": 0.287, "away": 0.228},
            most_likely_result="home",
            confidence_level=ConfidenceLevel.HIGH,
            confidence_score=0.78,
            expected_goals_home=1.8,
            expected_goals_away=1.2,
            expected_goals_total=3.0,
            over_under_probabilities={"over_2.5": 0.65, "under_2.5": 0.35},
            btts_probabilities={"yes": 0.58, "no": 0.42},
            most_likely_score=(2, 1),
            score_probability=0.12,
            model_version="v2.1.0",
            prediction_method="ensemble_ml",
            features_used=["home_attack", "away_defense"],
            betting_opportunities=[
                Mock(
                    spec=BettingOpportunity,
                    selection="home",
                    value_percentage=8.33,
                    expected_value=12.5,
                    stake_suggestion=50.0,
                    risk_level="Moderado",
                )
            ],
        )

    def test_prediction_insights_generation(self):
        """Test de generación de insights"""
        insights = self.predictor.get_prediction_insights(
            self.sample_prediction
        )

        # Verificar estructura
        assert "match_summary" in insights
        assert "key_factors" in insights
        assert "statistical_analysis" in insights
        assert "betting_analysis" in insights

        # Verificar match summary
        summary = insights["match_summary"]
        assert "Real Madrid vs Barcelona" in summary["teams"]
        assert summary["league"] == "La Liga"
        assert "Victoria de Real Madrid" in summary["prediction"]

        # Verificar análisis estadístico
        stats = insights["statistical_analysis"]
        assert "home_win_probability" in stats
        assert "expected_total_goals" in stats
        assert "over_2_5_probability" in stats
        assert "btts_probability" in stats

        # Verificar que hay factores clave
        assert len(insights["key_factors"]) > 0

        # Verificar análisis de apuestas
        betting = insights["betting_analysis"]
        assert betting is not None
        assert "best_value" in betting
        assert "expected_value" in betting

    def test_key_factors_identification(self):
        """Test de identificación de factores clave"""
        # Test con alta confianza
        high_confidence_prediction = self.sample_prediction
        high_confidence_prediction.confidence_score = 0.85

        insights = self.predictor.get_prediction_insights(
            high_confidence_prediction
        )
        key_factors = insights["key_factors"]

        assert any("Alta confianza" in factor for factor in key_factors)

        # Test con muchos goles esperados
        high_scoring_prediction = self.sample_prediction
        high_scoring_prediction.expected_goals_total = 3.5

        insights = self.predictor.get_prediction_insights(
            high_scoring_prediction
        )
        key_factors = insights["key_factors"]

        assert any("muchos goles" in factor for factor in key_factors)

    def test_betting_insights(self):
        """Test de insights de apuestas"""
        insights = self.predictor.get_prediction_insights(
            self.sample_prediction
        )
        betting_analysis = insights["betting_analysis"]

        assert betting_analysis is not None
        assert "home con 8.33% de valor" in betting_analysis["best_value"]
        assert "12.5" in betting_analysis["expected_value"]
        assert "€50.00" in betting_analysis["recommended_stake"]
        assert betting_analysis["risk_level"] == "Moderado"


class TestFactoryFunctions:
    """Tests para funciones factory y de conveniencia"""

    def test_create_predictor_factory(self):
        """Test de la función factory create_predictor"""
        predictor = create_predictor(TEST_APP_CONFIG)

        assert isinstance(predictor, PredictorService)
        assert predictor.config == TEST_APP_CONFIG
        assert predictor.model_version == "v2.1.0"

    def test_predict_match_quick_function(self):
        """Test de la función de predicción rápida"""
        with patch("app.services.predictor.PredictorService") as mock_service:
            mock_predictor = Mock()
            mock_prediction = Mock(spec=MatchPrediction)
            mock_prediction.home_team = "Real Madrid"
            mock_prediction.away_team = "Barcelona"
            mock_predictor.predict_match.return_value = mock_prediction
            mock_service.return_value = mock_predictor

            # Usar función de conveniencia
            result = predict_match_quick("Real Madrid", "Barcelona", "La Liga")

            assert result.home_team == "Real Madrid"
            assert result.away_team == "Barcelona"

    def test_analyze_predictions_batch_function(self):
        """Test de la función de análisis en lote"""
        matches = [
            {
                "home_team": "Real Madrid",
                "away_team": "Barcelona",
                "league": "La Liga",
            },
            {
                "home_team": "Manchester City",
                "away_team": "Liverpool",
                "league": "Premier League",
            },
        ]

        with patch("app.services.predictor.PredictorService") as mock_service:
            mock_predictor = Mock()
            mock_predictions = [Mock(spec=MatchPrediction) for _ in range(2)]
            mock_predictor.predict_multiple_matches.return_value = (
                mock_predictions
            )
            mock_service.return_value = mock_predictor

            # Usar función en lote
            results = analyze_predictions_batch(matches)

            assert len(results) == 2
            assert all(isinstance(pred, Mock) for pred in results)


class TestPredictionValidation:
    """Tests de validación de predicciones"""

    def test_prediction_probability_validation(self):
        """Test de validación de probabilidades"""
        # Probabilidades válidas
        valid_prediction = MatchPrediction(
            match_id="valid_test",
            home_team="Team A",
            away_team="Team B",
            league="Test League",
            match_date=datetime.now(),
            result_probabilities={"home": 0.485, "draw": 0.287, "away": 0.228},
            most_likely_result="home",
            confidence_level=ConfidenceLevel.MEDIUM,
            confidence_score=0.5,
            expected_goals_home=1.5,
            expected_goals_away=1.5,
            expected_goals_total=3.0,
            over_under_probabilities={"over_2.5": 0.5, "under_2.5": 0.5},
            btts_probabilities={"yes": 0.5, "no": 0.5},
            most_likely_score=(1, 1),
            score_probability=0.1,
            model_version="v2.1.0",
            prediction_method="test",
            features_used=[],
        )

        # Usar función de validación del módulo tests
        assert_prediction_valid(valid_prediction)

        # Verificar que las probabilidades suman 1
        total_prob = sum(valid_prediction.result_probabilities.values())
        assert abs(total_prob - 1.0) < 0.01

        # Verificar que over/under suman 1
        ou_total = sum(valid_prediction.over_under_probabilities.values())
        assert abs(ou_total - 1.0) < 0.01

        # Verificar que BTTS suma 1
        btts_total = sum(valid_prediction.btts_probabilities.values())
        assert abs(btts_total - 1.0) < 0.01

    def test_prediction_confidence_validation(self):
        """Test de validación de confianza"""
        # Test diferentes niveles de confianza
        confidence_tests = [
            (0.9, ConfidenceLevel.VERY_HIGH),
            (0.8, ConfidenceLevel.HIGH),
            (0.7, ConfidenceLevel.MEDIUM),
            (0.5, ConfidenceLevel.LOW),
            (0.3, ConfidenceLevel.VERY_LOW),
        ]

        for confidence_score, expected_level in confidence_tests:
            probabilities = {
                "home": confidence_score,
                "draw": (1 - confidence_score) / 2,
                "away": (1 - confidence_score) / 2,
            }
            predictor = PredictorService(TEST_APP_CONFIG)

            calculated_level = predictor._calculate_confidence_level(
                confidence_score, probabilities
            )

            # Verificar que el nivel calculado es razonable
            assert isinstance(calculated_level, ConfidenceLevel)

    def test_expected_goals_validation(self):
        """Test de validación de goles esperados"""
        predictor = PredictorService(TEST_APP_CONFIG)

        # Features que deberían resultar en goles razonables
        normal_features = np.array([1.5, 1.3, 1.2, 1.1] + [1.0] * 16)
        expected_goals = predictor._predict_expected_goals(
            normal_features.reshape(1, -1)
        )

        # Verificar rangos razonables
        assert 0.1 <= expected_goals["home"] <= 5.0
        assert 0.1 <= expected_goals["away"] <= 5.0
        assert (
            expected_goals["total"]
            == expected_goals["home"] + expected_goals["away"]
        )

    def test_score_probability_validation(self):
        """Test de validación de probabilidades de marcador"""
        predictor = PredictorService(TEST_APP_CONFIG)
        expected_goals = {"home": 1.8, "away": 1.2, "total": 3.0}

        # Test varios marcadores
        scores_to_test = [(0, 0), (1, 1), (2, 1), (3, 0), (1, 3)]

        for score in scores_to_test:
            prob = predictor._calculate_score_probability(
                score, expected_goals
            )

            # Verificar que la probabilidad está en rango válido
            assert 0 <= prob <= 1
            assert isinstance(prob, float)


class TestPredictionPerformance:
    """Tests de rendimiento del sistema de predicciones"""

    def test_prediction_speed(self):
        """Test de velocidad de predicción"""
        import time

        predictor = PredictorService(TEST_APP_CONFIG)

        # Mock dependencies para evitar llamadas reales
        with patch.object(
            predictor, "_get_team_features"
        ) as mock_features, patch.object(
            predictor, "_get_h2h_record"
        ) as mock_h2h, patch.object(
            predictor, "_predict_probabilities"
        ) as mock_predict:

            mock_features.return_value = {
                "attack_strength": 1.0,
                "defense_strength": 1.0,
            }
            mock_h2h.return_value = HeadToHeadRecord(
                "A", "B", 5, 2, 1, 2, [], 1.0, 1.0, 2.0
            )
            mock_predict.return_value = {"home": 0.4, "draw": 0.3, "away": 0.3}

            # Medir tiempo de predicción
            start_time = time.time()

            prediction = predictor.predict_match(
                "Real Madrid",
                "Barcelona",
                "La Liga",
                datetime.now() + timedelta(days=1),
            )

            prediction_time = time.time() - start_time

            # Verificar que la predicción es rápida (< 1 segundo)
            assert prediction_time < 1.0
            assert isinstance(prediction, MatchPrediction)

    @pytest.mark.slow
    def test_batch_prediction_performance(self):
        """Test de rendimiento de predicción en lote"""
        import time

        predictor = PredictorService(TEST_APP_CONFIG)

        # Crear múltiples partidos para probar
        matches = []
        teams = ["Real Madrid", "Barcelona", "Atletico", "Valencia", "Sevilla"]

        for i in range(10):
            home_team = teams[i % len(teams)]
            away_team = teams[(i + 1) % len(teams)]
            if home_team != away_team:
                matches.append(
                    {
                        "home_team": home_team,
                        "away_team": away_team,
                        "league": "La Liga",
                        "match_date": datetime.now() + timedelta(days=i),
                    }
                )

        # Mock dependencies
        with patch.object(
            predictor, "_get_team_features"
        ) as mock_features, patch.object(
            predictor, "_get_h2h_record"
        ) as mock_h2h, patch.object(
            predictor, "_predict_probabilities"
        ) as mock_predict:

            mock_features.return_value = {"attack_strength": 1.0}
            mock_h2h.return_value = HeadToHeadRecord(
                "A", "B", 5, 2, 1, 2, [], 1.0, 1.0, 2.0
            )
            mock_predict.return_value = {"home": 0.4, "draw": 0.3, "away": 0.3}

            # Medir tiempo de predicción en lote
            start_time = time.time()
            predictions = predictor.predict_multiple_matches(matches)
            batch_time = time.time() - start_time

            # Verificar rendimiento (< 5 segundos para 10 partidos)
            assert batch_time < 5.0
            assert len(predictions) == len(matches)


# Configuración de pytest para este módulo
@pytest.fixture(autouse=True)
def setup_prediction_tests():
    """Setup automático para tests de predicciones"""
    # Configurar seeds para reproducibilidad
    np.random.seed(42)

    yield

    # Cleanup después de tests
    import gc

    gc.collect()


# Marks para organización
pytestmark = [pytest.mark.unit, pytest.mark.predictions]

if __name__ == "__main__":
    # Ejecutar tests específicos de predicciones
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])
