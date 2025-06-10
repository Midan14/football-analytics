"""
Football Analytics - Odds Calculator Service
Sistema completo para cálculo, análisis y optimización de cuotas deportivas
"""

import json
import logging
import math
import sqlite3
import warnings
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Importaciones para cálculos estadísticos
from scipy import stats
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class BetType(Enum):
    """Tipos de apuestas disponibles"""

    MATCH_RESULT = "1x2"  # Victoria local/empate/visitante
    OVER_UNDER = "over_under"  # Más/menos goles
    BOTH_TEAMS_SCORE = "btts"  # Ambos equipos marcan
    ASIAN_HANDICAP = "asian_handicap"
    EUROPEAN_HANDICAP = "european_handicap"
    CORRECT_SCORE = "correct_score"
    HALF_TIME_RESULT = "ht_result"
    DOUBLE_CHANCE = "double_chance"
    DRAW_NO_BET = "draw_no_bet"


class OddsFormat(Enum):
    """Formatos de cuotas"""

    DECIMAL = "decimal"  # 2.50
    FRACTIONAL = "fractional"  # 3/2
    AMERICAN = "american"  # +150, -200
    IMPLIED = "implied"  # 40% (probabilidad implícita)


@dataclass
class OddsData:
    """Estructura para datos de cuotas"""

    bookmaker: str
    bet_type: BetType
    selection: str  # "home", "draw", "away", "over", "under", etc.
    odds_decimal: float
    odds_fractional: Optional[str] = None
    odds_american: Optional[int] = None
    implied_probability: Optional[float] = None
    margin: Optional[float] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

        # Calcular otros formatos automáticamente
        if self.odds_decimal:
            self.implied_probability = 1 / self.odds_decimal
            self.odds_american = self._decimal_to_american(self.odds_decimal)
            self.odds_fractional = self._decimal_to_fractional(
                self.odds_decimal
            )

    def _decimal_to_american(self, decimal: float) -> int:
        """Convierte cuotas decimales a americanas"""
        if decimal >= 2.0:
            return int((decimal - 1) * 100)
        else:
            return int(-100 / (decimal - 1))

    def _decimal_to_fractional(self, decimal: float) -> str:
        """Convierte cuotas decimales a fraccionarias"""
        if decimal < 1:
            return "0/1"

        # Convertir a fracción
        numerator = decimal - 1
        denominator = 1

        # Simplificar la fracción
        for i in range(2, 100):
            if abs(numerator * i - round(numerator * i)) < 0.01:
                numerator = round(numerator * i)
                denominator = i
                break

        return f"{int(numerator)}/{int(denominator)}"


@dataclass
class BettingOpportunity:
    """Oportunidad de apuesta identificada"""

    match_id: str
    home_team: str
    away_team: str
    bet_type: BetType
    selection: str
    bookmaker: str
    market_odds: float
    fair_odds: float
    predicted_probability: float
    implied_probability: float
    value_percentage: float
    expected_value: float
    kelly_percentage: float
    confidence_score: float
    risk_level: str
    recommendation: str
    stake_suggestion: float
    potential_profit: float

    @property
    def is_value_bet(self) -> bool:
        """Determina si es una apuesta de valor"""
        return self.value_percentage > 0 and self.expected_value > 0

    @property
    def risk_category(self) -> str:
        """Categoría de riesgo basada en múltiples factores"""
        if self.confidence_score >= 0.8 and self.value_percentage >= 10:
            return "Bajo Riesgo"
        elif self.confidence_score >= 0.6 and self.value_percentage >= 5:
            return "Riesgo Moderado"
        elif self.confidence_score >= 0.4 and self.value_percentage >= 2:
            return "Riesgo Alto"
        else:
            return "Muy Alto Riesgo"


class OddsConverter:
    """Conversor entre diferentes formatos de cuotas"""

    @staticmethod
    def decimal_to_fractional(decimal_odds: float) -> str:
        """Convierte cuotas decimales a fraccionarias"""
        if decimal_odds < 1:
            return "0/1"

        whole = int(decimal_odds)
        fraction = decimal_odds - whole

        if fraction == 0:
            return f"{whole-1}/1"

        # Encontrar la fracción más simple
        for denominator in range(1, 100):
            numerator = fraction * denominator
            if abs(numerator - round(numerator)) < 0.001:
                total_numerator = (whole - 1) * denominator + round(numerator)
                return f"{int(total_numerator)}/{denominator}"

        return f"{decimal_odds-1:.2f}/1"

    @staticmethod
    def decimal_to_american(decimal_odds: float) -> int:
        """Convierte cuotas decimales a americanas"""
        if decimal_odds >= 2.0:
            return int((decimal_odds - 1) * 100)
        else:
            return int(-100 / (decimal_odds - 1))

    @staticmethod
    def american_to_decimal(american_odds: int) -> float:
        """Convierte cuotas americanas a decimales"""
        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1

    @staticmethod
    def fractional_to_decimal(fractional: str) -> float:
        """Convierte cuotas fraccionarias a decimales"""
        try:
            parts = fractional.split("/")
            numerator = float(parts[0])
            denominator = float(parts[1])
            return (numerator / denominator) + 1
        except:
            return 1.0

    @staticmethod
    def probability_to_decimal(probability: float) -> float:
        """Convierte probabilidad a cuotas decimales"""
        if probability <= 0 or probability >= 1:
            raise ValueError("La probabilidad debe estar entre 0 y 1")
        return 1 / probability

    @staticmethod
    def decimal_to_probability(decimal_odds: float) -> float:
        """Convierte cuotas decimales a probabilidad implícita"""
        return 1 / decimal_odds


class ValueCalculator:
    """Calculadora de valor en apuestas"""

    @staticmethod
    def calculate_expected_value(
        predicted_prob: float, odds: float, stake: float = 100
    ) -> float:
        """Calcula el valor esperado de una apuesta"""
        return (predicted_prob * (odds - 1) * stake) - (
            (1 - predicted_prob) * stake
        )

    @staticmethod
    def calculate_value_percentage(
        predicted_prob: float, implied_prob: float
    ) -> float:
        """Calcula el porcentaje de valor"""
        return ((predicted_prob - implied_prob) / implied_prob) * 100

    @staticmethod
    def calculate_kelly_criterion(predicted_prob: float, odds: float) -> float:
        """Calcula el criterio de Kelly para tamaño óptimo de apuesta"""
        if odds <= 1:
            return 0

        edge = predicted_prob - (1 / odds)
        if edge <= 0:
            return 0

        kelly_fraction = edge / (odds - 1)
        return max(0, min(kelly_fraction, 0.25))  # Máximo 25% del bankroll

    @staticmethod
    def calculate_roi_potential(
        predicted_prob: float, odds: float, num_bets: int = 100
    ) -> Dict[str, float]:
        """Calcula el ROI potencial a largo plazo"""
        expected_wins = predicted_prob * num_bets
        expected_losses = num_bets - expected_wins

        total_winnings = expected_wins * (odds - 1)
        total_stakes = num_bets
        net_profit = total_winnings - total_stakes

        roi_percentage = (net_profit / total_stakes) * 100

        return {
            "expected_wins": expected_wins,
            "expected_losses": expected_losses,
            "total_winnings": total_winnings,
            "net_profit": net_profit,
            "roi_percentage": roi_percentage,
            "break_even_percentage": (1 / odds) * 100,
        }


class ArbitrageCalculator:
    """Calculadora de arbitraje (sure bets)"""

    @staticmethod
    def find_arbitrage_opportunity(
        odds_list: List[float],
    ) -> Optional[Dict[str, Any]]:
        """Encuentra oportunidades de arbitraje"""
        if len(odds_list) < 2:
            return None

        # Calcular suma de probabilidades implícitas
        implied_probs = [1 / odds for odds in odds_list]
        total_implied_prob = sum(implied_probs)

        # Si la suma es menor a 1, hay arbitraje
        if total_implied_prob < 1:
            profit_margin = (1 - total_implied_prob) * 100

            # Calcular stakes óptimos para cada resultado
            stakes = []
            total_stake = 1000  # Stake base de 1000

            for prob in implied_probs:
                stake = (prob / total_implied_prob) * total_stake
                stakes.append(stake)

            # Calcular profit garantizado
            profits = []
            for i, (odds, stake) in enumerate(zip(odds_list, stakes)):
                profit = (odds * stake) - total_stake
                profits.append(profit)

            return {
                "arbitrage_exists": True,
                "profit_margin": profit_margin,
                "total_stake": total_stake,
                "stakes": stakes,
                "guaranteed_profit": min(profits),
                "roi_percentage": (min(profits) / total_stake) * 100,
            }

        return {
            "arbitrage_exists": False,
            "overround": (total_implied_prob - 1) * 100,
        }

    @staticmethod
    def calculate_sure_bet_stakes(
        odds: List[float], total_investment: float
    ) -> List[float]:
        """Calcula las cantidades exactas para sure bet"""
        implied_probs = [1 / odd for odd in odds]
        total_prob = sum(implied_probs)

        if total_prob >= 1:
            return []  # No hay arbitraje

        stakes = []
        for prob in implied_probs:
            stake = (prob / total_prob) * total_investment
            stakes.append(round(stake, 2))

        return stakes


class OddsAnalyzer:
    """Analizador avanzado de cuotas"""

    def __init__(self):
        self.historical_odds = []
        self.market_efficiency_threshold = 0.05  # 5% de margen de ineficiencia

    def analyze_odds_movement(
        self, odds_history: List[Tuple[datetime, float]]
    ) -> Dict[str, Any]:
        """Analiza el movimiento de cuotas en el tiempo"""
        if len(odds_history) < 2:
            return {}

        # Extraer valores
        timestamps = [item[0] for item in odds_history]
        odds_values = [item[1] for item in odds_history]

        # Calcular tendencia
        time_numeric = [
            (ts - timestamps[0]).total_seconds() for ts in timestamps
        ]
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            time_numeric, odds_values
        )

        # Determinar dirección del movimiento
        direction = (
            "Subiendo" if slope > 0 else "Bajando" if slope < 0 else "Estable"
        )

        # Calcular volatilidad
        volatility = np.std(odds_values)

        # Detectar cambios bruscos
        changes = []
        for i in range(1, len(odds_values)):
            change_pct = (
                (odds_values[i] - odds_values[i - 1]) / odds_values[i - 1]
            ) * 100
            if abs(change_pct) > 5:  # Cambio mayor al 5%
                changes.append(
                    {
                        "timestamp": timestamps[i],
                        "change_percentage": change_pct,
                        "from_odds": odds_values[i - 1],
                        "to_odds": odds_values[i],
                    }
                )

        return {
            "direction": direction,
            "slope": slope,
            "correlation": r_value,
            "volatility": volatility,
            "initial_odds": odds_values[0],
            "final_odds": odds_values[-1],
            "total_change_pct": (
                (odds_values[-1] - odds_values[0]) / odds_values[0]
            )
            * 100,
            "significant_changes": changes,
            "trend_strength": abs(r_value),
        }

    def detect_steam_moves(
        self, odds_history: List[Tuple[datetime, float]]
    ) -> List[Dict[str, Any]]:
        """Detecta movimientos de vapor (steam moves)"""
        steam_moves = []

        for i in range(1, len(odds_history)):
            prev_time, prev_odds = odds_history[i - 1]
            curr_time, curr_odds = odds_history[i]

            # Calcular el cambio porcentual
            change_pct = ((curr_odds - prev_odds) / prev_odds) * 100
            time_diff = (curr_time - prev_time).total_seconds() / 60  # minutos

            # Steam move: cambio > 10% en menos de 30 minutos
            if abs(change_pct) > 10 and time_diff < 30:
                steam_moves.append(
                    {
                        "timestamp": curr_time,
                        "change_percentage": change_pct,
                        "time_span_minutes": time_diff,
                        "from_odds": prev_odds,
                        "to_odds": curr_odds,
                        "direction": "Bajada" if change_pct < 0 else "Subida",
                        "intensity": (
                            "Alta" if abs(change_pct) > 20 else "Media"
                        ),
                    }
                )

        return steam_moves

    def calculate_market_efficiency(
        self, bookmaker_odds: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """Calcula la eficiencia del mercado comparando bookmakers"""
        efficiency_scores = {}

        for bet_type, odds_list in bookmaker_odds.items():
            if len(odds_list) < 2:
                continue

            # Calcular coeficiente de variación
            mean_odds = np.mean(odds_list)
            std_odds = np.std(odds_list)
            cv = std_odds / mean_odds if mean_odds > 0 else 0

            # Eficiencia: menor variación = mayor eficiencia
            efficiency = max(0, 1 - cv)
            efficiency_scores[bet_type] = efficiency

        return efficiency_scores


class OddsDatabase:
    """Base de datos para almacenar y analizar cuotas históricas"""

    def __init__(self, db_path: str = "odds_data.db"):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Inicializa la base de datos de cuotas"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Tabla principal de cuotas
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS odds_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT,
                    bookmaker TEXT,
                    bet_type TEXT,
                    selection TEXT,
                    odds_decimal REAL,
                    implied_probability REAL,
                    timestamp TIMESTAMP,
                    source TEXT,
                    INDEX(match_id, bookmaker, bet_type)
                )
            """
            )

            # Tabla de oportunidades de valor
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS value_bets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT,
                    home_team TEXT,
                    away_team TEXT,
                    bet_type TEXT,
                    selection TEXT,
                    bookmaker TEXT,
                    market_odds REAL,
                    fair_odds REAL,
                    value_percentage REAL,
                    expected_value REAL,
                    kelly_percentage REAL,
                    confidence_score REAL,
                    risk_level TEXT,
                    identified_at TIMESTAMP
                )
            """
            )

            # Tabla de resultados de apuestas
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS bet_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bet_id TEXT UNIQUE,
                    match_id TEXT,
                    bet_type TEXT,
                    selection TEXT,
                    odds REAL,
                    stake REAL,
                    result TEXT,  -- 'won', 'lost', 'void'
                    profit_loss REAL,
                    placed_at TIMESTAMP,
                    settled_at TIMESTAMP
                )
            """
            )

            conn.commit()

    def save_odds(self, odds_data: OddsData, match_id: str):
        """Guarda datos de cuotas"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO odds_history 
                (match_id, bookmaker, bet_type, selection, odds_decimal, 
                 implied_probability, timestamp, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    match_id,
                    odds_data.bookmaker,
                    odds_data.bet_type.value,
                    odds_data.selection,
                    odds_data.odds_decimal,
                    odds_data.implied_probability,
                    odds_data.timestamp,
                    "api",
                ),
            )

            conn.commit()

    def save_value_bet(self, opportunity: BettingOpportunity):
        """Guarda oportunidad de apuesta de valor"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO value_bets
                (match_id, home_team, away_team, bet_type, selection, bookmaker,
                 market_odds, fair_odds, value_percentage, expected_value,
                 kelly_percentage, confidence_score, risk_level, identified_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    opportunity.match_id,
                    opportunity.home_team,
                    opportunity.away_team,
                    opportunity.bet_type.value,
                    opportunity.selection,
                    opportunity.bookmaker,
                    opportunity.market_odds,
                    opportunity.fair_odds,
                    opportunity.value_percentage,
                    opportunity.expected_value,
                    opportunity.kelly_percentage,
                    opportunity.confidence_score,
                    opportunity.risk_level,
                    datetime.now(),
                ),
            )

            conn.commit()

    def get_odds_history(
        self, match_id: str, bet_type: str
    ) -> List[Tuple[datetime, float]]:
        """Obtiene historial de cuotas para un partido"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT timestamp, odds_decimal 
                FROM odds_history 
                WHERE match_id = ? AND bet_type = ?
                ORDER BY timestamp
            """,
                (match_id, bet_type),
            )

            return [
                (datetime.fromisoformat(row[0]), row[1])
                for row in cursor.fetchall()
            ]

    def get_performance_stats(self, days: int = 30) -> Dict[str, Any]:
        """Obtiene estadísticas de rendimiento de las apuestas"""
        cutoff_date = datetime.now() - timedelta(days=days)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Total de apuestas
            cursor.execute(
                """
                SELECT COUNT(*), SUM(stake), SUM(profit_loss)
                FROM bet_results 
                WHERE placed_at >= ? AND result != 'void'
            """,
                (cutoff_date,),
            )

            total_bets, total_stake, total_profit = cursor.fetchone()

            if not total_bets:
                return {}

            # Apuestas ganadas
            cursor.execute(
                """
                SELECT COUNT(*) FROM bet_results 
                WHERE placed_at >= ? AND result = 'won'
            """,
                (cutoff_date,),
            )

            won_bets = cursor.fetchone()[0]

            # ROI y win rate
            roi = (total_profit / total_stake * 100) if total_stake else 0
            win_rate = (won_bets / total_bets * 100) if total_bets else 0

            return {
                "total_bets": total_bets,
                "won_bets": won_bets,
                "lost_bets": total_bets - won_bets,
                "total_stake": total_stake,
                "total_profit": total_profit,
                "roi_percentage": roi,
                "win_rate": win_rate,
                "average_odds": total_stake / total_bets if total_bets else 0,
                "period_days": days,
            }


class OddsCalculatorService:
    """Servicio principal para cálculo y análisis de cuotas"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.database = OddsDatabase()
        self.analyzer = OddsAnalyzer()
        self.value_calculator = ValueCalculator()
        self.arbitrage_calculator = ArbitrageCalculator()
        self.odds_converter = OddsConverter()

        # Configuración de análisis
        self.min_value_percentage = self.config.get(
            "min_value_percentage", 2.0
        )
        self.max_kelly_percentage = self.config.get(
            "max_kelly_percentage", 25.0
        )
        self.confidence_threshold = self.config.get(
            "confidence_threshold", 0.6
        )

        logger.info("💰 Odds Calculator Service inicializado")

    def analyze_betting_opportunity(
        self,
        predicted_probabilities: Dict[str, float],
        market_odds: Dict[str, float],
        match_info: Dict[str, str],
        confidence_scores: Dict[str, float] = None,
    ) -> List[BettingOpportunity]:
        """
        Analiza oportunidades de apuesta comparando probabilidades predichas vs mercado

        Args:
            predicted_probabilities: {'home': 0.45, 'draw': 0.30, 'away': 0.25}
            market_odds: {'home': 2.20, 'draw': 3.40, 'away': 4.00}
            match_info: {'match_id': '123', 'home_team': 'Madrid', 'away_team': 'Barca'}
            confidence_scores: {'home': 0.8, 'draw': 0.6, 'away': 0.7}

        Returns:
            Lista de oportunidades de apuesta identificadas
        """
        opportunities = []

        for selection in predicted_probabilities.keys():
            if selection not in market_odds:
                continue

            predicted_prob = predicted_probabilities[selection]
            market_odd = market_odds[selection]
            implied_prob = 1 / market_odd
            confidence = (
                confidence_scores.get(selection, 0.5)
                if confidence_scores
                else 0.5
            )

            # Calcular métricas de valor
            value_pct = self.value_calculator.calculate_value_percentage(
                predicted_prob, implied_prob
            )
            expected_value = self.value_calculator.calculate_expected_value(
                predicted_prob, market_odd
            )
            kelly_pct = (
                self.value_calculator.calculate_kelly_criterion(
                    predicted_prob, market_odd
                )
                * 100
            )

            # Solo incluir si hay valor positivo
            if value_pct > self.min_value_percentage and expected_value > 0:
                # Calcular fair odds (sin margen)
                fair_odds = 1 / predicted_prob

                # Determinar nivel de riesgo
                if confidence >= 0.8 and value_pct >= 10:
                    risk_level = "Bajo"
                    recommendation = "Apuesta Fuerte"
                elif confidence >= 0.6 and value_pct >= 5:
                    risk_level = "Moderado"
                    recommendation = "Apuesta Moderada"
                elif confidence >= 0.4 and value_pct >= 2:
                    risk_level = "Alto"
                    recommendation = "Apuesta Pequeña"
                else:
                    risk_level = "Muy Alto"
                    recommendation = "Evitar"

                # Calcular stake sugerido (Kelly limitado)
                kelly_limited = min(kelly_pct, self.max_kelly_percentage)
                base_bankroll = 1000  # Bankroll base para cálculos
                stake_suggestion = (kelly_limited / 100) * base_bankroll
                potential_profit = stake_suggestion * (market_odd - 1)

                opportunity = BettingOpportunity(
                    match_id=match_info["match_id"],
                    home_team=match_info["home_team"],
                    away_team=match_info["away_team"],
                    bet_type=BetType.MATCH_RESULT,
                    selection=selection,
                    bookmaker="mercado",
                    market_odds=market_odd,
                    fair_odds=fair_odds,
                    predicted_probability=predicted_prob,
                    implied_probability=implied_prob,
                    value_percentage=value_pct,
                    expected_value=expected_value,
                    kelly_percentage=kelly_pct,
                    confidence_score=confidence,
                    risk_level=risk_level,
                    recommendation=recommendation,
                    stake_suggestion=stake_suggestion,
                    potential_profit=potential_profit,
                )

                opportunities.append(opportunity)

                # Guardar en base de datos
                self.database.save_value_bet(opportunity)

        # Ordenar por valor esperado descendente
        opportunities.sort(key=lambda x: x.expected_value, reverse=True)

        logger.info(
            f"💎 Encontradas {len(opportunities)} oportunidades de valor"
        )

        return opportunities

    def find_arbitrage_opportunities(
        self,
        multi_bookmaker_odds: Dict[str, Dict[str, float]],
        match_info: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """
        Busca oportunidades de arbitraje entre múltiples bookmakers

        Args:
            multi_bookmaker_odds: {
                'bet365': {'home': 2.20, 'draw': 3.40, 'away': 4.00},
                'pinnacle': {'home': 2.25, 'draw': 3.30, 'away': 3.90}
            }
            match_info: Información del partido

        Returns:
            Lista de oportunidades de arbitraje
        """
        arbitrage_opportunities = []

        # Obtener todas las selecciones disponibles
        all_selections = set()
        for bookmaker_odds in multi_bookmaker_odds.values():
            all_selections.update(bookmaker_odds.keys())

        # Para cada selección, encontrar las mejores cuotas
        best_odds = {}
        best_bookmakers = {}

        for selection in all_selections:
            best_odd = 0
            best_bookmaker = None

            for bookmaker, odds_dict in multi_bookmaker_odds.items():
                if selection in odds_dict and odds_dict[selection] > best_odd:
                    best_odd = odds_dict[selection]
                    best_bookmaker = bookmaker

            if best_bookmaker:
                best_odds[selection] = best_odd
                best_bookmakers[selection] = best_bookmaker

        # Verificar si hay arbitraje
        if len(best_odds) >= 2:
            odds_list = list(best_odds.values())
            arbitrage_result = (
                self.arbitrage_calculator.find_arbitrage_opportunity(odds_list)
            )

            if arbitrage_result and arbitrage_result.get("arbitrage_exists"):
                opportunity = {
                    "match_id": match_info["match_id"],
                    "home_team": match_info["home_team"],
                    "away_team": match_info["away_team"],
                    "selections": list(best_odds.keys()),
                    "best_odds": best_odds,
                    "best_bookmakers": best_bookmakers,
                    "profit_margin": arbitrage_result["profit_margin"],
                    "guaranteed_profit": arbitrage_result["guaranteed_profit"],
                    "roi_percentage": arbitrage_result["roi_percentage"],
                    "suggested_stakes": dict(
                        zip(best_odds.keys(), arbitrage_result["stakes"])
                    ),
                    "total_investment": arbitrage_result["total_stake"],
                    "identified_at": datetime.now(),
                }

                arbitrage_opportunities.append(opportunity)
                logger.info(
                    f"🎯 Arbitraje encontrado: {opportunity['profit_margin']:.2f}% profit"
                )

        return arbitrage_opportunities

    def calculate_optimal_portfolio(
        self,
        opportunities: List[BettingOpportunity],
        total_bankroll: float,
        max_bet_percentage: float = 0.05,
        diversification_factor: float = 0.8,
    ) -> Dict[str, Any]:
        """
        Calcula el portafolio óptimo de apuestas usando teoría moderna de portafolios

        Args:
            opportunities: Lista de oportunidades identificadas
            total_bankroll: Bankroll total disponible
            max_bet_percentage: Máximo % del bankroll por apuesta individual
            diversification_factor: Factor de diversificación (0.5-1.0)

        Returns:
            Portafolio optimizado con distribución de stakes
        """
        if not opportunities:
            return {"error": "No hay oportunidades disponibles"}

        # Filtrar oportunidades válidas
        valid_opportunities = [
            opp
            for opp in opportunities
            if opp.is_value_bet
            and opp.confidence_score >= self.confidence_threshold
        ]

        if not valid_opportunities:
            return {
                "error": "No hay oportunidades válidas con suficiente confianza"
            }

        # Calcular Kelly óptimo ajustado por diversificación
        portfolio = []
        total_kelly_allocation = 0

        for opp in valid_opportunities:
            # Kelly ajustado por diversificación y límites
            kelly_raw = opp.kelly_percentage / 100
            kelly_adjusted = kelly_raw * diversification_factor
            kelly_limited = min(kelly_adjusted, max_bet_percentage)

            # Stake absoluto
            stake = kelly_limited * total_bankroll

            # Solo incluir si el stake es significativo (mínimo 0.1% del bankroll)
            if stake >= total_bankroll * 0.001:
                portfolio_item = {
                    "match": f"{opp.home_team} vs {opp.away_team}",
                    "selection": opp.selection,
                    "bookmaker": opp.bookmaker,
                    "odds": opp.market_odds,
                    "kelly_percentage": kelly_limited * 100,
                    "stake": round(stake, 2),
                    "potential_profit": round(
                        stake * (opp.market_odds - 1), 2
                    ),
                    "expected_value": round(
                        opp.expected_value * (stake / 100), 2
                    ),
                    "confidence": opp.confidence_score,
                    "risk_level": opp.risk_level,
                }
                portfolio.append(portfolio_item)
                total_kelly_allocation += kelly_limited

        # Calcular métricas del portafolio
        total_stake = sum(item["stake"] for item in portfolio)
        total_potential_profit = sum(
            item["potential_profit"] for item in portfolio
        )
        total_expected_value = sum(
            item["expected_value"] for item in portfolio
        )

        # ROI esperado del portafolio
        portfolio_roi = (
            (total_expected_value / total_stake * 100)
            if total_stake > 0
            else 0
        )

        # Distribución por nivel de riesgo
        risk_distribution = {}
        for item in portfolio:
            risk = item["risk_level"]
            if risk not in risk_distribution:
                risk_distribution[risk] = {
                    "count": 0,
                    "stake": 0,
                    "potential_profit": 0,
                }
            risk_distribution[risk]["count"] += 1
            risk_distribution[risk]["stake"] += item["stake"]
            risk_distribution[risk]["potential_profit"] += item[
                "potential_profit"
            ]

        return {
            "portfolio": portfolio,
            "summary": {
                "total_opportunities": len(portfolio),
                "total_stake": round(total_stake, 2),
                "bankroll_utilization": round(
                    (total_stake / total_bankroll) * 100, 2
                ),
                "total_potential_profit": round(total_potential_profit, 2),
                "total_expected_value": round(total_expected_value, 2),
                "portfolio_roi": round(portfolio_roi, 2),
                "avg_confidence": round(
                    np.mean([item["confidence"] for item in portfolio]), 3
                ),
                "kelly_allocation": round(total_kelly_allocation * 100, 2),
            },
            "risk_distribution": risk_distribution,
            "recommendations": self._generate_portfolio_recommendations(
                portfolio, total_bankroll
            ),
        }

    def _generate_portfolio_recommendations(
        self, portfolio: List[Dict], bankroll: float
    ) -> List[str]:
        """Genera recomendaciones para el portafolio"""
        recommendations = []

        total_stake = sum(item["stake"] for item in portfolio)
        utilization = (total_stake / bankroll) * 100

        if utilization > 15:
            recommendations.append(
                "⚠️ Alta utilización del bankroll - considera reducir stakes"
            )
        elif utilization < 2:
            recommendations.append(
                "💡 Baja utilización del bankroll - podrías ser más agresivo"
            )

        # Análisis de diversificación
        high_risk_count = len(
            [item for item in portfolio if item["risk_level"] == "Alto"]
        )
        if high_risk_count > len(portfolio) * 0.5:
            recommendations.append(
                "🔴 Demasiadas apuestas de alto riesgo - diversifica más"
            )

        # Análisis de confianza
        low_confidence = [
            item for item in portfolio if item["confidence"] < 0.6
        ]
        if low_confidence:
            recommendations.append(
                f"⚡ {len(low_confidence)} apuestas con baja confianza - considera eliminarlas"
            )

        # ROI esperado
        avg_roi = np.mean(
            [
                item["expected_value"] / item["stake"] * 100
                for item in portfolio
            ]
        )
        if avg_roi > 10:
            recommendations.append(
                "🚀 Excelente ROI esperado - portafolio muy prometedor"
            )
        elif avg_roi > 5:
            recommendations.append("✅ Buen ROI esperado - portafolio sólido")
        else:
            recommendations.append(
                "📊 ROI moderado - busca oportunidades con más valor"
            )

        return recommendations

    def analyze_closing_line_value(
        self, opening_odds: float, closing_odds: float, bet_odds: float
    ) -> Dict[str, float]:
        """
        Analiza el valor de línea de cierre (CLV - Closing Line Value)
        Métrica clave para evaluar la calidad de las apuestas
        """
        # CLV = (Cuotas de apuesta / Cuotas de cierre) - 1
        clv = (bet_odds / closing_odds) - 1
        clv_percentage = clv * 100

        # Comparar con apertura
        opening_movement = ((closing_odds - opening_odds) / opening_odds) * 100

        # Evaluación de la apuesta
        if clv_percentage > 5:
            evaluation = "Excelente"
        elif clv_percentage > 2:
            evaluation = "Buena"
        elif clv_percentage > 0:
            evaluation = "Positiva"
        elif clv_percentage > -2:
            evaluation = "Aceptable"
        else:
            evaluation = "Mala"

        return {
            "clv_percentage": round(clv_percentage, 2),
            "opening_odds": opening_odds,
            "closing_odds": closing_odds,
            "bet_odds": bet_odds,
            "opening_movement_pct": round(opening_movement, 2),
            "evaluation": evaluation,
            "beat_closing_line": clv_percentage > 0,
        }

    def simulate_betting_outcomes(
        self,
        opportunities: List[BettingOpportunity],
        num_simulations: int = 10000,
    ) -> Dict[str, Any]:
        """
        Simula resultados de apuestas usando Monte Carlo
        """
        if not opportunities:
            return {}

        simulation_results = []

        for _ in range(num_simulations):
            total_profit = 0
            total_stake = 0
            wins = 0

            for opp in opportunities:
                stake = opp.stake_suggestion
                total_stake += stake

                # Simular resultado basado en probabilidad predicha
                if np.random.random() < opp.predicted_probability:
                    # Apuesta ganada
                    profit = stake * (opp.market_odds - 1)
                    total_profit += profit
                    wins += 1
                else:
                    # Apuesta perdida
                    total_profit -= stake

            roi = (total_profit / total_stake * 100) if total_stake > 0 else 0
            win_rate = (
                (wins / len(opportunities) * 100) if opportunities else 0
            )

            simulation_results.append(
                {
                    "total_profit": total_profit,
                    "roi": roi,
                    "win_rate": win_rate,
                    "wins": wins,
                }
            )

        # Análisis estadístico de resultados
        profits = [result["total_profit"] for result in simulation_results]
        rois = [result["roi"] for result in simulation_results]
        win_rates = [result["win_rate"] for result in simulation_results]

        return {
            "simulations": num_simulations,
            "profit_stats": {
                "mean": round(np.mean(profits), 2),
                "median": round(np.median(profits), 2),
                "std": round(np.std(profits), 2),
                "min": round(np.min(profits), 2),
                "max": round(np.max(profits), 2),
                "percentile_5": round(np.percentile(profits, 5), 2),
                "percentile_95": round(np.percentile(profits, 95), 2),
            },
            "roi_stats": {
                "mean": round(np.mean(rois), 2),
                "median": round(np.median(rois), 2),
                "std": round(np.std(rois), 2),
                "positive_scenarios": len([roi for roi in rois if roi > 0])
                / num_simulations
                * 100,
            },
            "win_rate_stats": {
                "mean": round(np.mean(win_rates), 2),
                "median": round(np.median(win_rates), 2),
                "std": round(np.std(win_rates), 2),
            },
            "risk_metrics": {
                "probability_of_loss": len([p for p in profits if p < 0])
                / num_simulations
                * 100,
                "maximum_drawdown": round(np.min(profits), 2),
                "sharpe_ratio": (
                    round(np.mean(rois) / np.std(rois), 3)
                    if np.std(rois) > 0
                    else 0
                ),
            },
        }

    def generate_betting_report(
        self, opportunities: List[BettingOpportunity], bankroll: float = 10000
    ) -> Dict[str, Any]:
        """
        Genera un reporte completo de análisis de apuestas
        """
        if not opportunities:
            return {"error": "No hay oportunidades para analizar"}

        # Calcular portafolio óptimo
        portfolio = self.calculate_optimal_portfolio(opportunities, bankroll)

        # Simulación Monte Carlo
        simulation = self.simulate_betting_outcomes(opportunities)

        # Análisis de arbitraje si hay múltiples oportunidades del mismo partido
        arbitrage_opportunities = []
        # Agrupar por partido para buscar arbitraje
        matches = {}
        for opp in opportunities:
            match_key = f"{opp.home_team}_{opp.away_team}"
            if match_key not in matches:
                matches[match_key] = []
            matches[match_key].append(opp)

        # Estadísticas generales
        total_value = sum(opp.value_percentage for opp in opportunities)
        avg_confidence = np.mean(
            [opp.confidence_score for opp in opportunities]
        )
        high_value_bets = len(
            [opp for opp in opportunities if opp.value_percentage > 10]
        )

        # Distribución por tipo de apuesta
        bet_type_distribution = {}
        for opp in opportunities:
            bet_type = opp.bet_type.value
            if bet_type not in bet_type_distribution:
                bet_type_distribution[bet_type] = 0
            bet_type_distribution[bet_type] += 1

        report = {
            "generated_at": datetime.now().isoformat(),
            "bankroll": bankroll,
            "analysis_summary": {
                "total_opportunities": len(opportunities),
                "high_value_bets": high_value_bets,
                "average_value_percentage": round(
                    total_value / len(opportunities), 2
                ),
                "average_confidence": round(avg_confidence, 3),
                "bet_type_distribution": bet_type_distribution,
            },
            "portfolio_optimization": portfolio,
            "monte_carlo_simulation": simulation,
            "top_opportunities": sorted(
                opportunities, key=lambda x: x.expected_value, reverse=True
            )[:5],
            "risk_analysis": {
                "low_risk_bets": len(
                    [opp for opp in opportunities if opp.risk_level == "Bajo"]
                ),
                "medium_risk_bets": len(
                    [
                        opp
                        for opp in opportunities
                        if opp.risk_level == "Moderado"
                    ]
                ),
                "high_risk_bets": len(
                    [opp for opp in opportunities if opp.risk_level == "Alto"]
                ),
                "avg_kelly_percentage": round(
                    np.mean([opp.kelly_percentage for opp in opportunities]), 2
                ),
            },
            "recommendations": self._generate_report_recommendations(
                opportunities, portfolio, simulation
            ),
        }

        return report

    def _generate_report_recommendations(
        self,
        opportunities: List[BettingOpportunity],
        portfolio: Dict,
        simulation: Dict,
    ) -> List[str]:
        """Genera recomendaciones basadas en el análisis completo"""
        recommendations = []

        # Análisis de oportunidades
        if len(opportunities) == 0:
            recommendations.append(
                "❌ No se encontraron oportunidades de valor"
            )
            return recommendations

        high_value = len(
            [opp for opp in opportunities if opp.value_percentage > 10]
        )
        if high_value > 0:
            recommendations.append(
                f"🚀 {high_value} oportunidades de alto valor identificadas"
            )

        # Análisis de confianza
        low_confidence = len(
            [opp for opp in opportunities if opp.confidence_score < 0.6]
        )
        if low_confidence > len(opportunities) * 0.3:
            recommendations.append(
                "⚠️ Muchas apuestas con baja confianza - procede con cautela"
            )

        # Análisis de simulación
        if simulation and "roi_stats" in simulation:
            positive_scenarios = simulation["roi_stats"].get(
                "positive_scenarios", 0
            )
            if positive_scenarios > 70:
                recommendations.append(
                    "✅ Alta probabilidad de rentabilidad a largo plazo"
                )
            elif positive_scenarios > 50:
                recommendations.append(
                    "📊 Probabilidad moderada de rentabilidad"
                )
            else:
                recommendations.append(
                    "🔴 Baja probabilidad de rentabilidad - revisa estrategia"
                )

        # Análisis de diversificación
        if len(opportunities) < 3:
            recommendations.append(
                "💡 Considera diversificar con más oportunidades"
            )

        # Kelly criterion
        avg_kelly = np.mean([opp.kelly_percentage for opp in opportunities])
        if avg_kelly > 10:
            recommendations.append(
                "⚠️ Stakes muy altos según Kelly - reduce para minimizar riesgo"
            )

        return recommendations


# Funciones de conveniencia
def create_odds_calculator(
    config: Dict[str, Any] = None,
) -> OddsCalculatorService:
    """Factory function para crear el calculador de cuotas"""
    return OddsCalculatorService(config)


def analyze_match_value(
    predicted_probs: Dict[str, float],
    market_odds: Dict[str, float],
    match_info: Dict[str, str],
) -> List[BettingOpportunity]:
    """Función de conveniencia para analizar valor de un partido"""
    calculator = create_odds_calculator()
    return calculator.analyze_betting_opportunity(
        predicted_probs, market_odds, match_info
    )


def convert_odds_format(
    odds: float, from_format: OddsFormat, to_format: OddsFormat
) -> Union[float, str, int]:
    """Convierte entre diferentes formatos de cuotas"""
    converter = OddsConverter()

    # Convertir a decimal primero
    if from_format == OddsFormat.DECIMAL:
        decimal_odds = odds
    elif from_format == OddsFormat.AMERICAN:
        decimal_odds = converter.american_to_decimal(int(odds))
    elif from_format == OddsFormat.FRACTIONAL:
        decimal_odds = converter.fractional_to_decimal(str(odds))
    else:
        decimal_odds = odds

    # Convertir al formato destino
    if to_format == OddsFormat.DECIMAL:
        return decimal_odds
    elif to_format == OddsFormat.AMERICAN:
        return converter.decimal_to_american(decimal_odds)
    elif to_format == OddsFormat.FRACTIONAL:
        return converter.decimal_to_fractional(decimal_odds)
    elif to_format == OddsFormat.IMPLIED:
        return converter.decimal_to_probability(decimal_odds) * 100

    return decimal_odds


if __name__ == "__main__":
    # Sistema de análisis de cuotas en producción
    async def main():
        print("💰 Football Analytics - Odds Calculator")
        print("Sistema completo de análisis de cuotas y valor")
        print("=" * 60)

        # Crear calculador
        calculator = create_odds_calculator(
            {
                "min_value_percentage": 2.0,
                "max_kelly_percentage": 20.0,
                "confidence_threshold": 0.6,
            }
        )

        # Ejemplo de análisis con datos reales
        print("📊 Analizando oportunidades de valor...")

        # Datos de ejemplo de un partido real
        predicted_probabilities = {
            "home": 0.45,  # 45% probabilidad victoria local
            "draw": 0.30,  # 30% probabilidad empate
            "away": 0.25,  # 25% probabilidad victoria visitante
        }

        market_odds = {
            "home": 2.20,
            "draw": 3.40,
            "away": 4.00,
        }  # Cuotas del mercado

        match_info = {
            "match_id": "12345",
            "home_team": "Real Madrid",
            "away_team": "Barcelona",
        }

        confidence_scores = {"home": 0.85, "draw": 0.65, "away": 0.70}

        # Analizar oportunidades
        opportunities = calculator.analyze_betting_opportunity(
            predicted_probabilities, market_odds, match_info, confidence_scores
        )

        print(f"\n🎯 Oportunidades encontradas: {len(opportunities)}")

        for opp in opportunities:
            print(
                f"\n💎 {opp.selection.upper()}: {opp.home_team} vs {opp.away_team}"
            )
            print(f"   📈 Valor: {opp.value_percentage:.2f}%")
            print(f"   💰 EV: {opp.expected_value:.2f}")
            print(f"   🎲 Kelly: {opp.kelly_percentage:.2f}%")
            print(f"   ⭐ Confianza: {opp.confidence_score:.2f}")
            print(f"   🔥 Recomendación: {opp.recommendation}")

        # Calcular portafolio óptimo
        if opportunities:
            print(f"\n📊 Calculando portafolio óptimo...")
            portfolio = calculator.calculate_optimal_portfolio(
                opportunities, 10000
            )

            if "portfolio" in portfolio:
                print(
                    f"   💼 Apuestas recomendadas: {len(portfolio['portfolio'])}"
                )
                print(
                    f"   💵 Stake total: €{portfolio['summary']['total_stake']}"
                )
                print(
                    f"   📈 ROI esperado: {portfolio['summary']['portfolio_roi']:.2f}%"
                )
                print(
                    f"   🎯 Utilización bankroll: {portfolio['summary']['bankroll_utilization']:.1f}%"
                )

        # Generar reporte completo
        print(f"\n📋 Generando reporte completo...")
        report = calculator.generate_betting_report(opportunities, 10000)

        if "analysis_summary" in report:
            summary = report["analysis_summary"]
            print(
                f"   🔍 Total oportunidades: {summary['total_opportunities']}"
            )
            print(f"   🚀 Alto valor: {summary['high_value_bets']}")
            print(
                f"   📊 Valor promedio: {summary['average_value_percentage']:.2f}%"
            )
            print(
                f"   ⭐ Confianza promedio: {summary['average_confidence']:.2f}"
            )

        # Mostrar recomendaciones
        if "recommendations" in report:
            print(f"\n💡 RECOMENDACIONES:")
            for rec in report["recommendations"]:
                print(f"   {rec}")

        print(f"\n✅ Análisis completado exitosamente!")

    # Ejecutar análisis
    import asyncio

    asyncio.run(main())
