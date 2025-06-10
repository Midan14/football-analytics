"""
Football Analytics - Calculator Service
Calculadoras especializadas para métricas y análisis de fútbol
"""

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Resultado de un partido"""

    home_goals: int
    away_goals: int
    home_team: str
    away_team: str
    date: datetime

    @property
    def result(self) -> str:
        """Resultado del partido (H/D/A)"""
        if self.home_goals > self.away_goals:
            return "H"
        elif self.home_goals < self.away_goals:
            return "A"
        else:
            return "D"

    @property
    def total_goals(self) -> int:
        """Total de goles en el partido"""
        return self.home_goals + self.away_goals


@dataclass
class TeamStats:
    """Estadísticas de un equipo"""

    team_name: str
    matches_played: int
    wins: int
    draws: int
    losses: int
    goals_for: int
    goals_against: int
    points: int

    @property
    def goal_difference(self) -> int:
        """Diferencia de goles"""
        return self.goals_for - self.goals_against

    @property
    def win_percentage(self) -> float:
        """Porcentaje de victorias"""
        return (
            (self.wins / self.matches_played) * 100
            if self.matches_played > 0
            else 0.0
        )

    @property
    def points_per_game(self) -> float:
        """Puntos por partido"""
        return (
            self.points / self.matches_played
            if self.matches_played > 0
            else 0.0
        )


class BaseCalculator(ABC):
    """Clase base para todas las calculadoras"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def calculate(self, *args, **kwargs) -> Any:
        """Método principal de cálculo"""
        pass


class FormCalculator(BaseCalculator):
    """Calculadora de forma de equipos"""

    def calculate_team_form(
        self, matches: List[MatchResult], team: str, num_matches: int = 5
    ) -> Dict[str, float]:
        """
        Calcula la forma reciente de un equipo

        Args:
            matches: Lista de partidos recientes
            team: Nombre del equipo
            num_matches: Número de partidos a considerar

        Returns:
            Diccionario con métricas de forma
        """
        # Filtrar partidos del equipo
        team_matches = [
            m for m in matches if team in [m.home_team, m.away_team]
        ]
        recent_matches = sorted(
            team_matches, key=lambda x: x.date, reverse=True
        )[:num_matches]

        if not recent_matches:
            return self._empty_form()

        points = 0
        goals_for = 0
        goals_against = 0
        wins = draws = losses = 0

        for match in recent_matches:
            is_home = match.home_team == team
            team_goals = match.home_goals if is_home else match.away_goals
            opponent_goals = match.away_goals if is_home else match.home_goals

            goals_for += team_goals
            goals_against += opponent_goals

            if team_goals > opponent_goals:
                wins += 1
                points += 3
            elif team_goals == opponent_goals:
                draws += 1
                points += 1
            else:
                losses += 1

        matches_played = len(recent_matches)

        return {
            "form_points": points,
            "form_ppg": points / matches_played if matches_played > 0 else 0,
            "form_wins": wins,
            "form_draws": draws,
            "form_losses": losses,
            "form_goals_for": goals_for,
            "form_goals_against": goals_against,
            "form_goal_difference": goals_for - goals_against,
            "form_win_percentage": (
                (wins / matches_played) * 100 if matches_played > 0 else 0
            ),
            "matches_considered": matches_played,
            "form_rating": self._calculate_form_rating(
                points, matches_played, goals_for - goals_against
            ),
        }

    def _empty_form(self) -> Dict[str, float]:
        """Retorna forma vacía"""
        return {
            "form_points": 0,
            "form_ppg": 0,
            "form_wins": 0,
            "form_draws": 0,
            "form_losses": 0,
            "form_goals_for": 0,
            "form_goals_against": 0,
            "form_goal_difference": 0,
            "form_win_percentage": 0,
            "matches_considered": 0,
            "form_rating": 0.0,
        }

    def _calculate_form_rating(
        self, points: int, matches: int, goal_diff: int
    ) -> float:
        """Calcula rating de forma (0-100)"""
        if matches == 0:
            return 0.0

        # Base rating basado en puntos
        base_rating = (points / (matches * 3)) * 70  # Máximo 70 puntos

        # Bonus por diferencia de goles
        goal_bonus = (
            min(goal_diff * 2, 30)
            if goal_diff > 0
            else max(goal_diff * 1.5, -30)
        )

        rating = base_rating + goal_bonus
        return max(0, min(100, rating))

    def calculate(
        self, matches: List[MatchResult], team: str, num_matches: int = 5
    ) -> Dict[str, float]:
        """Implementación del método abstracto"""
        return self.calculate_team_form(matches, team, num_matches)


class StrengthCalculator(BaseCalculator):
    """Calculadora de fortaleza ofensiva y defensiva"""

    def calculate_team_strength(
        self,
        matches: List[MatchResult],
        team: str,
        league_average_goals: float = 2.5,
    ) -> Dict[str, float]:
        """
        Calcula la fortaleza ofensiva y defensiva de un equipo

        Args:
            matches: Lista de partidos
            team: Nombre del equipo
            league_average_goals: Promedio de goles de la liga

        Returns:
            Diccionario con métricas de fortaleza
        """
        home_matches = [m for m in matches if m.home_team == team]
        away_matches = [m for m in matches if m.away_team == team]

        # Goles marcados y recibidos en casa
        home_goals_for = sum(m.home_goals for m in home_matches)
        home_goals_against = sum(m.away_goals for m in home_matches)

        # Goles marcados y recibidos fuera
        away_goals_for = sum(m.away_goals for m in away_matches)
        away_goals_against = sum(m.home_goals for m in away_matches)

        home_matches_count = len(home_matches)
        away_matches_count = len(away_matches)
        total_matches = home_matches_count + away_matches_count

        if total_matches == 0:
            return self._empty_strength()

        # Promedios
        home_attack = (
            home_goals_for / home_matches_count
            if home_matches_count > 0
            else 0
        )
        home_defense = (
            home_goals_against / home_matches_count
            if home_matches_count > 0
            else 0
        )
        away_attack = (
            away_goals_for / away_matches_count
            if away_matches_count > 0
            else 0
        )
        away_defense = (
            away_goals_against / away_matches_count
            if away_matches_count > 0
            else 0
        )

        overall_attack = (home_goals_for + away_goals_for) / total_matches
        overall_defense = (
            home_goals_against + away_goals_against
        ) / total_matches

        # Fortalezas relativas (comparado con promedio de liga)
        home_attack_strength = (
            home_attack / league_average_goals
            if league_average_goals > 0
            else 0
        )
        home_defense_strength = (
            league_average_goals / home_defense if home_defense > 0 else 0
        )
        away_attack_strength = (
            away_attack / league_average_goals
            if league_average_goals > 0
            else 0
        )
        away_defense_strength = (
            league_average_goals / away_defense if away_defense > 0 else 0
        )

        overall_attack_strength = (
            overall_attack / league_average_goals
            if league_average_goals > 0
            else 0
        )
        overall_defense_strength = (
            league_average_goals / overall_defense
            if overall_defense > 0
            else 0
        )

        return {
            # Promedios absolutos
            "home_attack_avg": home_attack,
            "home_defense_avg": home_defense,
            "away_attack_avg": away_attack,
            "away_defense_avg": away_defense,
            "overall_attack_avg": overall_attack,
            "overall_defense_avg": overall_defense,
            # Fortalezas relativas
            "home_attack_strength": home_attack_strength,
            "home_defense_strength": home_defense_strength,
            "away_attack_strength": away_attack_strength,
            "away_defense_strength": away_defense_strength,
            "overall_attack_strength": overall_attack_strength,
            "overall_defense_strength": overall_defense_strength,
            # Ratings combinados
            "home_rating": (home_attack_strength + home_defense_strength) / 2,
            "away_rating": (away_attack_strength + away_defense_strength) / 2,
            "overall_rating": (
                overall_attack_strength + overall_defense_strength
            )
            / 2,
            # Estadísticas adicionales
            "total_matches": total_matches,
            "home_matches": home_matches_count,
            "away_matches": away_matches_count,
        }

    def _empty_strength(self) -> Dict[str, float]:
        """Retorna fortaleza vacía"""
        return {
            k: 0.0
            for k in [
                "home_attack_avg",
                "home_defense_avg",
                "away_attack_avg",
                "away_defense_avg",
                "overall_attack_avg",
                "overall_defense_avg",
                "home_attack_strength",
                "home_defense_strength",
                "away_attack_strength",
                "away_defense_strength",
                "overall_attack_strength",
                "overall_defense_strength",
                "home_rating",
                "away_rating",
                "overall_rating",
                "total_matches",
                "home_matches",
                "away_matches",
            ]
        }

    def calculate(
        self, matches: List[MatchResult], team: str, league_avg: float = 2.5
    ) -> Dict[str, float]:
        """Implementación del método abstracto"""
        return self.calculate_team_strength(matches, team, league_avg)


class PoissonCalculator(BaseCalculator):
    """Calculadora de probabilidades usando distribución de Poisson"""

    def calculate_match_probabilities(
        self,
        home_attack: float,
        home_defense: float,
        away_attack: float,
        away_defense: float,
        home_advantage: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Calcula probabilidades de resultado usando modelo de Poisson

        Args:
            home_attack: Fortaleza ofensiva local
            home_defense: Fortaleza defensiva local
            away_attack: Fortaleza ofensiva visitante
            away_defense: Fortaleza defensiva visitante
            home_advantage: Ventaja de local (factor multiplicativo)

        Returns:
            Diccionario con probabilidades y predicciones
        """
        # Calcular goles esperados
        home_expected = home_attack * away_defense * (1 + home_advantage)
        away_expected = away_attack * home_defense

        # Probabilidades de resultados exactos (0-5 goles cada equipo)
        max_goals = 6
        probabilities = np.zeros((max_goals, max_goals))

        for home_goals in range(max_goals):
            for away_goals in range(max_goals):
                prob_home = self._poisson_probability(
                    home_goals, home_expected
                )
                prob_away = self._poisson_probability(
                    away_goals, away_expected
                )
                probabilities[home_goals, away_goals] = prob_home * prob_away

        # Calcular probabilidades de resultado final
        prob_home_win = np.sum(
            [
                probabilities[i, j]
                for i in range(max_goals)
                for j in range(max_goals)
                if i > j
            ]
        )
        prob_draw = np.sum([probabilities[i, i] for i in range(max_goals)])
        prob_away_win = np.sum(
            [
                probabilities[i, j]
                for i in range(max_goals)
                for j in range(max_goals)
                if i < j
            ]
        )

        # Probabilidades de goles totales
        prob_over_2_5 = np.sum(
            [
                probabilities[i, j]
                for i in range(max_goals)
                for j in range(max_goals)
                if i + j > 2.5
            ]
        )
        prob_under_2_5 = 1 - prob_over_2_5

        prob_over_1_5 = np.sum(
            [
                probabilities[i, j]
                for i in range(max_goals)
                for j in range(max_goals)
                if i + j > 1.5
            ]
        )
        prob_under_1_5 = 1 - prob_over_1_5

        # Probabilidades de ambos equipos marcan
        prob_btts_yes = (
            1
            - probabilities[0, :].sum()
            - probabilities[:, 0].sum()
            + probabilities[0, 0]
        )
        prob_btts_no = 1 - prob_btts_yes

        # Resultado más probable
        most_likely_score = np.unravel_index(
            probabilities.argmax(), probabilities.shape
        )

        return {
            "expected_goals": {
                "home": home_expected,
                "away": away_expected,
                "total": home_expected + away_expected,
            },
            "match_result_probabilities": {
                "home_win": prob_home_win,
                "draw": prob_draw,
                "away_win": prob_away_win,
            },
            "total_goals_probabilities": {
                "over_2_5": prob_over_2_5,
                "under_2_5": prob_under_2_5,
                "over_1_5": prob_over_1_5,
                "under_1_5": prob_under_1_5,
            },
            "btts_probabilities": {"yes": prob_btts_yes, "no": prob_btts_no},
            "most_likely_score": {
                "home_goals": most_likely_score[0],
                "away_goals": most_likely_score[1],
                "probability": probabilities[most_likely_score],
            },
            "score_probabilities": probabilities.tolist(),
        }

    def _poisson_probability(self, k: int, lambda_val: float) -> float:
        """Calcula probabilidad de Poisson"""
        return (math.exp(-lambda_val) * (lambda_val**k)) / math.factorial(k)

    def calculate(
        self,
        home_attack: float,
        home_defense: float,
        away_attack: float,
        away_defense: float,
    ) -> Dict[str, Any]:
        """Implementación del método abstracto"""
        return self.calculate_match_probabilities(
            home_attack, home_defense, away_attack, away_defense
        )


class ValueCalculator(BaseCalculator):
    """Calculadora de valor en apuestas"""

    def calculate_betting_value(
        self,
        predicted_probability: float,
        bookmaker_odds: float,
        stake: float = 100,
    ) -> Dict[str, float]:
        """
        Calcula el valor esperado de una apuesta

        Args:
            predicted_probability: Probabilidad predicha por nuestro modelo
            bookmaker_odds: Cuotas de la casa de apuestas
            stake: Cantidad apostada

        Returns:
            Análisis de valor de la apuesta
        """
        # Probabilidad implícita de las cuotas
        implied_probability = 1 / bookmaker_odds

        # Valor esperado
        expected_value = (
            predicted_probability * (bookmaker_odds - 1) * stake
        ) - ((1 - predicted_probability) * stake)

        # Porcentaje de valor esperado
        ev_percentage = (expected_value / stake) * 100

        # Edge (ventaja)
        edge = predicted_probability - implied_probability
        edge_percentage = edge * 100

        # Kelly Criterion para tamaño óptimo de apuesta
        kelly_fraction = (
            edge / (bookmaker_odds - 1) if bookmaker_odds > 1 else 0
        )
        kelly_percentage = max(
            0, kelly_fraction * 100
        )  # No apostar si es negativo

        # Clasificación de valor
        if ev_percentage > 5:
            value_rating = "Excelente"
        elif ev_percentage > 2:
            value_rating = "Bueno"
        elif ev_percentage > 0:
            value_rating = "Positivo"
        elif ev_percentage > -2:
            value_rating = "Neutral"
        else:
            value_rating = "Negativo"

        return {
            "expected_value": expected_value,
            "ev_percentage": ev_percentage,
            "edge": edge,
            "edge_percentage": edge_percentage,
            "implied_probability": implied_probability,
            "predicted_probability": predicted_probability,
            "kelly_fraction": kelly_fraction,
            "kelly_percentage": kelly_percentage,
            "value_rating": value_rating,
            "is_value_bet": ev_percentage > 0,
            "recommended_stake": (
                stake * kelly_fraction if kelly_fraction > 0 else 0
            ),
        }

    def calculate_portfolio_value(self, bets: List[Dict]) -> Dict[str, float]:
        """
        Calcula valor de un portafolio de apuestas

        Args:
            bets: Lista de diccionarios con datos de apuestas

        Returns:
            Análisis del portafolio
        """
        total_stake = sum(bet["stake"] for bet in bets)
        total_expected_value = sum(
            self.calculate_betting_value(
                bet["probability"], bet["odds"], bet["stake"]
            )["expected_value"]
            for bet in bets
        )

        portfolio_ev_percentage = (
            (total_expected_value / total_stake) * 100
            if total_stake > 0
            else 0
        )

        positive_ev_bets = len(
            [
                bet
                for bet in bets
                if self.calculate_betting_value(
                    bet["probability"], bet["odds"], bet["stake"]
                )["ev_percentage"]
                > 0
            ]
        )

        return {
            "total_stake": total_stake,
            "total_expected_value": total_expected_value,
            "portfolio_ev_percentage": portfolio_ev_percentage,
            "number_of_bets": len(bets),
            "positive_ev_bets": positive_ev_bets,
            "positive_ev_percentage": (
                (positive_ev_bets / len(bets)) * 100 if bets else 0
            ),
        }

    def calculate(
        self, probability: float, odds: float, stake: float = 100
    ) -> Dict[str, float]:
        """Implementación del método abstracto"""
        return self.calculate_betting_value(probability, odds, stake)


class PerformanceCalculator(BaseCalculator):
    """Calculadora de métricas de rendimiento"""

    def calculate_prediction_accuracy(
        self, predictions: List[str], actual_results: List[str]
    ) -> Dict[str, float]:
        """
        Calcula la precisión de las predicciones

        Args:
            predictions: Lista de predicciones (H/D/A)
            actual_results: Lista de resultados reales (H/D/A)

        Returns:
            Métricas de precisión
        """
        if len(predictions) != len(actual_results):
            raise ValueError("Las listas deben tener la misma longitud")

        total_predictions = len(predictions)
        if total_predictions == 0:
            return self._empty_accuracy()

        correct_predictions = sum(
            1 for p, a in zip(predictions, actual_results) if p == a
        )
        accuracy = (correct_predictions / total_predictions) * 100

        # Precisión por tipo de resultado
        home_predictions = [i for i, p in enumerate(predictions) if p == "H"]
        draw_predictions = [i for i, p in enumerate(predictions) if p == "D"]
        away_predictions = [i for i, p in enumerate(predictions) if p == "A"]

        home_accuracy = self._calculate_type_accuracy(
            home_predictions, actual_results, "H"
        )
        draw_accuracy = self._calculate_type_accuracy(
            draw_predictions, actual_results, "D"
        )
        away_accuracy = self._calculate_type_accuracy(
            away_predictions, actual_results, "A"
        )

        return {
            "overall_accuracy": accuracy,
            "correct_predictions": correct_predictions,
            "total_predictions": total_predictions,
            "home_accuracy": home_accuracy,
            "draw_accuracy": draw_accuracy,
            "away_accuracy": away_accuracy,
            "home_predictions_count": len(home_predictions),
            "draw_predictions_count": len(draw_predictions),
            "away_predictions_count": len(away_predictions),
        }

    def _calculate_type_accuracy(
        self, indices: List[int], actual: List[str], result_type: str
    ) -> float:
        """Calcula precisión para un tipo específico de resultado"""
        if not indices:
            return 0.0

        correct = sum(1 for i in indices if actual[i] == result_type)
        return (correct / len(indices)) * 100

    def _empty_accuracy(self) -> Dict[str, float]:
        """Retorna métricas vacías"""
        return {
            "overall_accuracy": 0.0,
            "correct_predictions": 0,
            "total_predictions": 0,
            "home_accuracy": 0.0,
            "draw_accuracy": 0.0,
            "away_accuracy": 0.0,
            "home_predictions_count": 0,
            "draw_predictions_count": 0,
            "away_predictions_count": 0,
        }

    def calculate_roi(self, bets: List[Dict]) -> Dict[str, float]:
        """
        Calcula ROI de apuestas

        Args:
            bets: Lista con datos de apuestas realizadas

        Returns:
            Métricas de ROI
        """
        total_stake = sum(bet["stake"] for bet in bets)
        total_return = sum(bet["return"] for bet in bets)
        total_profit = total_return - total_stake

        roi_percentage = (
            (total_profit / total_stake) * 100 if total_stake > 0 else 0
        )

        winning_bets = [bet for bet in bets if bet["return"] > bet["stake"]]
        win_rate = (len(winning_bets) / len(bets)) * 100 if bets else 0

        average_odds = np.mean([bet["odds"] for bet in bets]) if bets else 0
        average_stake = total_stake / len(bets) if bets else 0

        return {
            "total_stake": total_stake,
            "total_return": total_return,
            "total_profit": total_profit,
            "roi_percentage": roi_percentage,
            "win_rate": win_rate,
            "number_of_bets": len(bets),
            "winning_bets": len(winning_bets),
            "losing_bets": len(bets) - len(winning_bets),
            "average_odds": average_odds,
            "average_stake": average_stake,
        }

    def calculate(
        self, predictions: List[str], actual: List[str]
    ) -> Dict[str, float]:
        """Implementación del método abstracto"""
        return self.calculate_prediction_accuracy(predictions, actual)


class CalculatorService:
    """Servicio principal que coordina todas las calculadoras"""

    def __init__(self):
        self.form_calculator = FormCalculator()
        self.strength_calculator = StrengthCalculator()
        self.poisson_calculator = PoissonCalculator()
        self.value_calculator = ValueCalculator()
        self.performance_calculator = PerformanceCalculator()

        self.logger = logging.getLogger(__name__)

    def calculate_team_metrics(
        self, matches: List[MatchResult], team: str
    ) -> Dict[str, Any]:
        """
        Calcula todas las métricas para un equipo

        Args:
            matches: Lista de partidos
            team: Nombre del equipo

        Returns:
            Métricas completas del equipo
        """
        try:
            form_metrics = self.form_calculator.calculate_team_form(
                matches, team
            )
            strength_metrics = (
                self.strength_calculator.calculate_team_strength(matches, team)
            )

            return {
                "team": team,
                "form": form_metrics,
                "strength": strength_metrics,
                "calculated_at": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error calculando métricas para {team}: {e}")
            raise

    def calculate_match_prediction(
        self, home_team_metrics: Dict, away_team_metrics: Dict
    ) -> Dict[str, Any]:
        """
        Calcula predicción completa para un partido

        Args:
            home_team_metrics: Métricas del equipo local
            away_team_metrics: Métricas del equipo visitante

        Returns:
            Predicción completa del partido
        """
        try:
            home_attack = home_team_metrics["strength"]["home_attack_strength"]
            home_defense = home_team_metrics["strength"][
                "home_defense_strength"
            ]
            away_attack = away_team_metrics["strength"]["away_attack_strength"]
            away_defense = away_team_metrics["strength"][
                "away_defense_strength"
            ]

            poisson_prediction = (
                self.poisson_calculator.calculate_match_probabilities(
                    home_attack, home_defense, away_attack, away_defense
                )
            )

            return {
                "home_team": home_team_metrics["team"],
                "away_team": away_team_metrics["team"],
                "prediction": poisson_prediction,
                "team_metrics": {
                    "home": home_team_metrics,
                    "away": away_team_metrics,
                },
                "calculated_at": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error en predicción de partido: {e}")
            raise

    def calculate_betting_analysis(
        self, prediction: Dict, odds: Dict
    ) -> Dict[str, Any]:
        """
        Analiza valor de apuestas para un partido

        Args:
            prediction: Predicción del partido
            odds: Cuotas disponibles

        Returns:
            Análisis de valor completo
        """
        try:
            probabilities = prediction["prediction"][
                "match_result_probabilities"
            ]

            home_value = self.value_calculator.calculate_betting_value(
                probabilities["home_win"], odds.get("home_win", 2.0)
            )
            draw_value = self.value_calculator.calculate_betting_value(
                probabilities["draw"], odds.get("draw", 3.0)
            )
            away_value = self.value_calculator.calculate_betting_value(
                probabilities["away_win"], odds.get("away_win", 4.0)
            )

            return {
                "match": f"{prediction['home_team']} vs {prediction['away_team']}",
                "value_analysis": {
                    "home_win": home_value,
                    "draw": draw_value,
                    "away_win": away_value,
                },
                "best_bet": max(
                    [
                        ("home_win", home_value["ev_percentage"]),
                        ("draw", draw_value["ev_percentage"]),
                        ("away_win", away_value["ev_percentage"]),
                    ],
                    key=lambda x: x[1],
                ),
                "calculated_at": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error en análisis de apuestas: {e}")
            raise


# Funciones de conveniencia
def create_match_result(
    home_team: str,
    away_team: str,
    home_goals: int,
    away_goals: int,
    date: datetime = None,
) -> MatchResult:
    """Crea un resultado de partido"""
    if date is None:
        date = datetime.now()
    return MatchResult(home_goals, away_goals, home_team, away_team, date)


def create_team_stats(
    team_name: str,
    matches_played: int,
    wins: int,
    draws: int,
    losses: int,
    goals_for: int,
    goals_against: int,
) -> TeamStats:
    """Crea estadísticas de equipo"""
    points = (wins * 3) + draws
    return TeamStats(
        team_name,
        matches_played,
        wins,
        draws,
        losses,
        goals_for,
        goals_against,
        points,
    )


if __name__ == "__main__":
    # Ejemplo de uso
    print("🧮 Football Analytics - Calculator Service")
    print("Calculadoras especializadas para métricas deportivas")

    # Crear servicio
    calculator = CalculatorService()

    # Ejemplo con datos ficticios
    matches = [
        create_match_result("Real Madrid", "Barcelona", 2, 1),
        create_match_result("Barcelona", "Atletico", 3, 0),
        create_match_result("Valencia", "Real Madrid", 1, 2),
    ]

    # Calcular métricas
    madrid_metrics = calculator.calculate_team_metrics(matches, "Real Madrid")
    print(f"✅ Métricas calculadas para Real Madrid")
    print(f"📊 Forma: {madrid_metrics['form']['form_rating']:.2f}")
    print(
        f"⚔️ Fortaleza ofensiva: {madrid_metrics['strength']['overall_attack_strength']:.2f}"
    )

    barca_metrics = calculator.calculate_team_metrics(matches, "Barcelona")
    print(f"✅ Métricas calculadas para Barcelona")

    # Calcular predicción
    prediction = calculator.calculate_match_prediction(
        madrid_metrics, barca_metrics
    )
    probs = prediction["prediction"]["match_result_probabilities"]
    print(f"🔮 Predicción Real Madrid vs Barcelona:")
    print(f"   Local: {probs['home_win']:.1%}")
    print(f"   Empate: {probs['draw']:.1%}")
    print(f"   Visitante: {probs['away_win']:.1%}")

    # Análisis de valor
    odds = {"home_win": 2.10, "draw": 3.20, "away_win": 3.50}
    betting_analysis = calculator.calculate_betting_analysis(prediction, odds)
    best_bet = betting_analysis["best_bet"]
    print(f"💰 Mejor apuesta: {best_bet[0]} (EV: {best_bet[1]:.2f}%)")
