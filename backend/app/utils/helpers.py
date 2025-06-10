#!/usr/bin/env python3
"""
Football Analytics - Helper Functions
Funciones auxiliares y utilidades para el sistema Football Analytics

Autor: Sistema Football Analytics
Versión: 2.1.0
Fecha: 2024-06-02
"""

import asyncio
import hashlib
import json
import logging
import math
import os
import re
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache, wraps
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from .constants import (
    ALL_KNOWN_TEAMS,
    BOOKMAKERS,
    CONFIDENCE_LEVELS,
    DATA_VALIDATION_LIMITS,
    FOOTBALL_DISTRIBUTIONS,
    LEAGUE_CODES,
    MatchResult,
    OddsFormat,
)

# =============================================================================
# CONFIGURACIÓN DE LOGGING
# =============================================================================

logger = logging.getLogger("football_analytics.helpers")

# =============================================================================
# DECORADORES ÚTILES
# =============================================================================


def timing_decorator(func):
    """
    Decorador para medir tiempo de ejecución de funciones.

    Args:
        func: Función a decorar

    Returns:
        Función decorada que mide tiempo
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            execution_time = (datetime.now() - start_time).total_seconds()
            if execution_time > 1.0:
                logger.warning(
                    f"⚠️ Función lenta: {func.__name__} tomó {execution_time:.2f}s"
                )
            else:
                logger.debug(f"⏱️ {func.__name__}: {execution_time:.3f}s")

    return wrapper


def async_timing_decorator(func):
    """
    Decorador para medir tiempo de ejecución de funciones asíncronas.

    Args:
        func: Función asíncrona a decorar

    Returns:
        Función decorada que mide tiempo
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = datetime.now()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            execution_time = (datetime.now() - start_time).total_seconds()
            if execution_time > 2.0:
                logger.warning(
                    f"⚠️ Función async lenta: {func.__name__} tomó {execution_time:.2f}s"
                )
            else:
                logger.debug(f"⏱️ {func.__name__}: {execution_time:.3f}s")

    return wrapper


def safe_execution(default_return=None, log_errors=True):
    """
    Decorador para ejecución segura que maneja errores.

    Args:
        default_return: Valor por defecto en caso de error
        log_errors: Si registrar errores en log

    Returns:
        Decorador que maneja errores
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(f"❌ Error en {func.__name__}: {str(e)}")
                return default_return

        return wrapper

    return decorator


def memoize_with_ttl(ttl_seconds: int = 300):
    """
    Decorador para memoización con TTL (Time To Live).

    Args:
        ttl_seconds: Tiempo de vida del cache en segundos

    Returns:
        Decorador que cachea resultados
    """

    def decorator(func):
        cache = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Crear clave del cache
            key = str(args) + str(sorted(kwargs.items()))
            current_time = datetime.now()

            # Verificar si está en cache y no ha expirado
            if key in cache:
                result, timestamp = cache[key]
                if (current_time - timestamp).total_seconds() < ttl_seconds:
                    return result

            # Ejecutar función y cachear resultado
            result = func(*args, **kwargs)
            cache[key] = (result, current_time)

            # Limpiar cache viejo
            expired_keys = [
                k
                for k, (_, ts) in cache.items()
                if (current_time - ts).total_seconds() >= ttl_seconds
            ]
            for k in expired_keys:
                del cache[k]

            return result

        return wrapper

    return decorator


# =============================================================================
# UTILIDADES DE CADENAS Y TEXTO
# =============================================================================


def normalize_team_name(team_name: str) -> str:
    """
    Normaliza nombre de equipo para comparaciones consistentes.

    Args:
        team_name: Nombre original del equipo

    Returns:
        Nombre normalizado
    """
    if not team_name:
        return ""

    # Remover espacios extra y convertir a título
    normalized = " ".join(team_name.strip().split()).title()

    # Mapeos específicos de nombres comunes
    name_mappings = {
        "Fc Barcelona": "Barcelona",
        "Real Madrid Cf": "Real Madrid",
        "Atletico De Madrid": "Atletico Madrid",
        "Manchester United Fc": "Manchester United",
        "Liverpool Fc": "Liverpool",
        "Chelsea Fc": "Chelsea",
        "Arsenal Fc": "Arsenal",
        "Tottenham Hotspur Fc": "Tottenham Hotspur",
        "Manchester City Fc": "Manchester City",
        "Bayern Munich": "Bayern Munich",
        "Borussia Dortmund": "Borussia Dortmund",
        "Juventus Fc": "Juventus",
        "Ac Milan": "AC Milan",
        "Inter Milan": "Inter Milan",
        "Paris Saint-Germain": "Paris Saint-Germain",
    }

    return name_mappings.get(normalized, normalized)


def extract_team_abbreviation(team_name: str) -> str:
    """
    Extrae abreviación de 3 letras de un nombre de equipo.

    Args:
        team_name: Nombre del equipo

    Returns:
        Abreviación de 3 letras
    """
    if not team_name:
        return "UNK"

    # Casos especiales conocidos
    abbreviations = {
        "Real Madrid": "RMA",
        "Barcelona": "BAR",
        "Atletico Madrid": "ATM",
        "Manchester United": "MUN",
        "Manchester City": "MCI",
        "Liverpool": "LIV",
        "Chelsea": "CHE",
        "Arsenal": "ARS",
        "Tottenham Hotspur": "TOT",
        "Bayern Munich": "BAY",
        "Borussia Dortmund": "BVB",
        "Juventus": "JUV",
        "AC Milan": "MIL",
        "Inter Milan": "INT",
        "Paris Saint-Germain": "PSG",
    }

    normalized_name = normalize_team_name(team_name)
    if normalized_name in abbreviations:
        return abbreviations[normalized_name]

    # Generar abreviación automáticamente
    words = normalized_name.split()
    if len(words) == 1:
        return words[0][:3].upper()
    elif len(words) == 2:
        return (words[0][0] + words[1][:2]).upper()
    else:
        return "".join(word[0] for word in words[:3]).upper()


def sanitize_filename(filename: str) -> str:
    """
    Sanitiza un nombre de archivo removiendo caracteres problemáticos.

    Args:
        filename: Nombre de archivo original

    Returns:
        Nombre de archivo sanitizado
    """
    if not filename:
        return "unnamed"

    # Remover caracteres problemáticos
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", filename)

    # Remover espacios múltiples y convertir a guiones bajos
    sanitized = re.sub(r"\s+", "_", sanitized)

    # Truncar si es muy largo
    if len(sanitized) > 100:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:95] + ext

    return sanitized


def format_currency(
    amount: float, currency: str = "EUR", decimals: int = 2
) -> str:
    """
    Formatea cantidad como moneda.

    Args:
        amount: Cantidad a formatear
        currency: Código de moneda (EUR, USD, GBP)
        decimals: Número de decimales

    Returns:
        Cantidad formateada como string
    """
    symbols = {"EUR": "€", "USD": "$", "GBP": "£"}
    symbol = symbols.get(currency, currency)

    # Usar Decimal para precisión
    decimal_amount = Decimal(str(amount)).quantize(
        Decimal("0." + "0" * decimals), rounding=ROUND_HALF_UP
    )

    return f"{symbol}{decimal_amount:,.{decimals}f}"


# =============================================================================
# UTILIDADES DE FECHA Y TIEMPO
# =============================================================================


def parse_flexible_date(
    date_input: Union[str, datetime],
) -> Optional[datetime]:
    """
    Parsea fecha de múltiples formatos flexiblemente.

    Args:
        date_input: String de fecha o datetime

    Returns:
        Objeto datetime o None si no se puede parsear
    """
    if isinstance(date_input, datetime):
        return date_input

    if not isinstance(date_input, str):
        return None

    # Formatos soportados
    formats = [
        "%Y-%m-%d %H:%M:%S",  # 2024-06-02 15:30:00
        "%Y-%m-%d",  # 2024-06-02
        "%Y/%m/%d",  # 2024/06/02
        "%d/%m/%Y",  # 02/06/2024
        "%d-%m-%Y",  # 02-06-2024
        "%Y-%m-%dT%H:%M:%S",  # ISO format sin Z
        "%Y-%m-%dT%H:%M:%SZ",  # ISO format con Z
        "%Y-%m-%dT%H:%M:%S.%f",  # ISO con microsegundos
        "%Y-%m-%dT%H:%M:%S.%fZ",  # ISO con microsegundos y Z
        "%d %b %Y",  # 02 Jun 2024
        "%d %B %Y",  # 02 June 2024
        "%b %d, %Y",  # Jun 02, 2024
        "%B %d, %Y",  # June 02, 2024
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_input.strip(), fmt)
        except ValueError:
            continue

    logger.warning(f"⚠️ No se pudo parsear fecha: {date_input}")
    return None


def get_season_from_date(date: datetime) -> str:
    """
    Determina la temporada de fútbol basada en la fecha.

    Args:
        date: Fecha del partido

    Returns:
        String de temporada (ej: "2023/24")
    """
    if date.month >= 8:  # Agosto o después
        return f"{date.year}/{str(date.year + 1)[-2:]}"
    else:  # Antes de agosto
        return f"{date.year - 1}/{str(date.year)[-2:]}"


def is_weekend(date: datetime) -> bool:
    """
    Verifica si una fecha es fin de semana.

    Args:
        date: Fecha a verificar

    Returns:
        True si es sábado o domingo
    """
    return date.weekday() >= 5  # 5=sábado, 6=domingo


def get_match_time_category(date: datetime) -> str:
    """
    Categoriza un partido por horario.

    Args:
        date: Fecha y hora del partido

    Returns:
        Categoría de horario
    """
    hour = date.hour

    if 12 <= hour < 15:
        return "lunch_time"  # Medio día
    elif 15 <= hour < 18:
        return "afternoon"  # Tarde
    elif 18 <= hour < 21:
        return "evening"  # Noche
    elif 21 <= hour <= 23:
        return "night"  # Noche tardía
    else:
        return "unusual_time"  # Horario inusual


def days_until_match(match_date: datetime) -> int:
    """
    Calcula días hasta un partido.

    Args:
        match_date: Fecha del partido

    Returns:
        Número de días (negativo si ya pasó)
    """
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    match_day = match_date.replace(hour=0, minute=0, second=0, microsecond=0)

    return (match_day - today).days


# =============================================================================
# UTILIDADES MATEMÁTICAS Y ESTADÍSTICAS
# =============================================================================


def safe_divide(
    numerator: float, denominator: float, default: float = 0.0
) -> float:
    """
    División segura que maneja división por cero.

    Args:
        numerator: Numerador
        denominator: Denominador
        default: Valor por defecto si denominador es 0

    Returns:
        Resultado de la división o valor por defecto
    """
    if denominator == 0 or math.isnan(denominator):
        return default

    result = numerator / denominator
    return result if not math.isnan(result) else default


def clamp_value(value: float, min_val: float, max_val: float) -> float:
    """
    Limita un valor entre un mínimo y máximo.

    Args:
        value: Valor a limitar
        min_val: Valor mínimo
        max_val: Valor máximo

    Returns:
        Valor limitado
    """
    if math.isnan(value):
        return min_val

    return max(min_val, min(value, max_val))


def normalize_probability_distribution(
    probabilities: List[float],
) -> List[float]:
    """
    Normaliza una lista de probabilidades para que sumen 1.0.

    Args:
        probabilities: Lista de probabilidades

    Returns:
        Lista normalizada
    """
    if not probabilities or all(p <= 0 for p in probabilities):
        # Si todas son 0 o negativas, distribuir uniformemente
        return [1.0 / len(probabilities)] * len(probabilities)

    total = sum(p for p in probabilities if p > 0)
    if total == 0:
        return [1.0 / len(probabilities)] * len(probabilities)

    return [max(0, p) / total for p in probabilities]


def calculate_percentage(
    part: float, total: float, decimals: int = 1
) -> float:
    """
    Calcula porcentaje de manera segura.

    Args:
        part: Parte del total
        total: Total
        decimals: Decimales para redondeo

    Returns:
        Porcentaje redondeado
    """
    if total == 0:
        return 0.0

    percentage = (part / total) * 100
    return round(percentage, decimals)


def moving_average(values: List[float], window: int = 5) -> List[float]:
    """
    Calcula media móvil de una serie de valores.

    Args:
        values: Lista de valores
        window: Tamaño de la ventana

    Returns:
        Lista con medias móviles
    """
    if len(values) < window:
        return values

    averages = []
    for i in range(len(values)):
        if i < window - 1:
            # Para los primeros valores, usar todos los disponibles
            avg = sum(values[: i + 1]) / (i + 1)
        else:
            # Media móvil normal
            avg = sum(values[i - window + 1 : i + 1]) / window

        averages.append(avg)

    return averages


def calculate_z_score(value: float, mean: float, std: float) -> float:
    """
    Calcula z-score de un valor.

    Args:
        value: Valor a evaluar
        mean: Media de la distribución
        std: Desviación estándar

    Returns:
        Z-score
    """
    if std == 0:
        return 0.0

    return (value - mean) / std


def is_outlier(
    value: float, values: List[float], threshold: float = 2.0
) -> bool:
    """
    Determina si un valor es atípico usando z-score.

    Args:
        value: Valor a evaluar
        values: Lista de valores de referencia
        threshold: Umbral de z-score para considerar outlier

    Returns:
        True si es outlier
    """
    if len(values) < 3:
        return False

    mean_val = sum(values) / len(values)
    variance = sum((x - mean_val) ** 2 for x in values) / len(values)
    std_val = math.sqrt(variance)

    z_score = calculate_z_score(value, mean_val, std_val)
    return abs(z_score) > threshold


# =============================================================================
# UTILIDADES DE CUOTAS Y APUESTAS
# =============================================================================


def convert_odds_format(
    odds: float, from_format: OddsFormat, to_format: OddsFormat
) -> float:
    """
    Convierte cuotas entre diferentes formatos.

    Args:
        odds: Cuotas originales
        from_format: Formato original
        to_format: Formato destino

    Returns:
        Cuotas convertidas
    """
    # Primero convertir a decimal (formato base)
    if from_format == OddsFormat.DECIMAL:
        decimal_odds = odds
    elif from_format == OddsFormat.FRACTIONAL:
        decimal_odds = odds + 1.0
    elif from_format == OddsFormat.AMERICAN:
        if odds > 0:
            decimal_odds = (odds / 100) + 1
        else:
            decimal_odds = (100 / abs(odds)) + 1
    elif from_format == OddsFormat.IMPLIED:
        decimal_odds = 1 / (odds / 100)
    else:
        raise ValueError(f"Formato no soportado: {from_format}")

    # Luego convertir a formato destino
    if to_format == OddsFormat.DECIMAL:
        return decimal_odds
    elif to_format == OddsFormat.FRACTIONAL:
        return decimal_odds - 1.0
    elif to_format == OddsFormat.AMERICAN:
        if decimal_odds >= 2.0:
            return (decimal_odds - 1) * 100
        else:
            return -100 / (decimal_odds - 1)
    elif to_format == OddsFormat.IMPLIED:
        return (1 / decimal_odds) * 100
    else:
        raise ValueError(f"Formato no soportado: {to_format}")


def calculate_implied_probability(
    odds: float, format: OddsFormat = OddsFormat.DECIMAL
) -> float:
    """
    Calcula probabilidad implícita de unas cuotas.

    Args:
        odds: Cuotas
        format: Formato de las cuotas

    Returns:
        Probabilidad implícita (0-1)
    """
    decimal_odds = convert_odds_format(odds, format, OddsFormat.DECIMAL)
    return 1.0 / decimal_odds


def calculate_overround(odds_dict: Dict[str, float]) -> float:
    """
    Calcula el overround (margen) de un conjunto de cuotas.

    Args:
        odds_dict: Diccionario con cuotas por resultado

    Returns:
        Overround como porcentaje
    """
    if not odds_dict:
        return 0.0

    total_implied_prob = sum(
        calculate_implied_probability(odds)
        for odds in odds_dict.values()
        if odds > 0
    )

    return max(0.0, (total_implied_prob - 1.0) * 100)


def calculate_fair_odds(probability: float) -> float:
    """
    Calcula cuotas justas (sin margen) para una probabilidad.

    Args:
        probability: Probabilidad (0-1)

    Returns:
        Cuotas decimales justas
    """
    if probability <= 0 or probability >= 1:
        return 1.0

    return 1.0 / probability


def find_best_odds(
    odds_by_bookmaker: Dict[str, Dict[str, float]],
) -> Dict[str, Tuple[str, float]]:
    """
    Encuentra las mejores cuotas para cada resultado entre múltiples bookmakers.

    Args:
        odds_by_bookmaker: Cuotas organizadas por bookmaker

    Returns:
        Mejores cuotas por resultado con bookmaker
    """
    best_odds = {}

    # Obtener todos los posibles resultados
    all_outcomes = set()
    for bookmaker_odds in odds_by_bookmaker.values():
        all_outcomes.update(bookmaker_odds.keys())

    # Encontrar mejores cuotas para cada resultado
    for outcome in all_outcomes:
        best_odd = 0.0
        best_bookmaker = ""

        for bookmaker, odds in odds_by_bookmaker.items():
            if outcome in odds and odds[outcome] > best_odd:
                best_odd = odds[outcome]
                best_bookmaker = bookmaker

        if best_odd > 0:
            best_odds[outcome] = (best_bookmaker, best_odd)

    return best_odds


# =============================================================================
# UTILIDADES DE ARCHIVOS Y DATOS
# =============================================================================


def safe_json_load(file_path: str) -> Optional[Dict]:
    """
    Carga archivo JSON de manera segura.

    Args:
        file_path: Ruta al archivo JSON

    Returns:
        Diccionario con datos o None si error
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
        logger.error(f"❌ Error cargando JSON {file_path}: {str(e)}")
        return None


def safe_json_save(data: Any, file_path: str) -> bool:
    """
    Guarda datos en archivo JSON de manera segura.

    Args:
        data: Datos a guardar
        file_path: Ruta del archivo destino

    Returns:
        True si guardado exitoso
    """
    try:
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Convertir dataclasses a dict si es necesario
        if hasattr(data, "__dataclass_fields__"):
            data = asdict(data)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        return True
    except Exception as e:
        logger.error(f"❌ Error guardando JSON {file_path}: {str(e)}")
        return False


def create_backup_filename(original_path: str) -> str:
    """
    Crea nombre de archivo de respaldo con timestamp.

    Args:
        original_path: Ruta del archivo original

    Returns:
        Ruta del archivo de respaldo
    """
    dir_path = os.path.dirname(original_path)
    filename = os.path.basename(original_path)
    name, ext = os.path.splitext(filename)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"{name}_backup_{timestamp}{ext}"

    return os.path.join(dir_path, "backups", backup_filename)


def clean_old_files(
    directory: str, max_age_days: int = 30, pattern: str = "*"
) -> int:
    """
    Limpia archivos antiguos de un directorio.

    Args:
        directory: Directorio a limpiar
        max_age_days: Edad máxima en días
        pattern: Patrón de archivos a limpiar

    Returns:
        Número de archivos eliminados
    """
    if not os.path.exists(directory):
        return 0

    cutoff_date = datetime.now() - timedelta(days=max_age_days)
    deleted_count = 0

    try:
        for filename in os.listdir(directory):
            if pattern != "*" and pattern not in filename:
                continue

            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))

                if file_time < cutoff_date:
                    os.remove(file_path)
                    deleted_count += 1
                    logger.info(f"🗑️ Archivo eliminado: {filename}")

    except Exception as e:
        logger.error(f"❌ Error limpiando directorio {directory}: {str(e)}")

    return deleted_count


# =============================================================================
# UTILIDADES DE FÚTBOL ESPECÍFICAS
# =============================================================================


def calculate_goal_difference(
    home_goals: int, away_goals: int, perspective: str = "home"
) -> int:
    """
    Calcula diferencia de goles desde perspectiva específica.

    Args:
        home_goals: Goles del equipo local
        away_goals: Goles del equipo visitante
        perspective: 'home' o 'away'

    Returns:
        Diferencia de goles (positivo = favorable)
    """
    if perspective == "home":
        return home_goals - away_goals
    else:
        return away_goals - home_goals


def determine_match_result(home_goals: int, away_goals: int) -> MatchResult:
    """
    Determina el resultado de un partido.

    Args:
        home_goals: Goles del equipo local
        away_goals: Goles del equipo visitante

    Returns:
        Resultado del partido
    """
    if home_goals > away_goals:
        return MatchResult.HOME_WIN
    elif home_goals < away_goals:
        return MatchResult.AWAY_WIN
    else:
        return MatchResult.DRAW


def is_high_scoring_match(total_goals: int, threshold: int = 3) -> bool:
    """
    Determina si un partido fue de muchos goles.

    Args:
        total_goals: Total de goles en el partido
        threshold: Umbral para considerar "muchos goles"

    Returns:
        True si fue partido de muchos goles
    """
    return total_goals >= threshold


def calculate_team_form_rating(
    recent_results: List[MatchResult], weights: Optional[List[float]] = None
) -> float:
    """
    Calcula rating de forma de equipo basado en resultados recientes.

    Args:
        recent_results: Lista de resultados recientes (más reciente primero)
        weights: Pesos opcionales para cada partido (más reciente = mayor peso)

    Returns:
        Rating de forma (0-100)
    """
    if not recent_results:
        return 50.0  # Neutral si no hay datos

    # Pesos por defecto (más reciente = mayor peso)
    if weights is None:
        weights = [1.0 / (i + 1) for i in range(len(recent_results))]
    else:
        weights = weights[: len(recent_results)]

    # Puntos por resultado
    points_map = {
        MatchResult.HOME_WIN: 3,
        MatchResult.AWAY_WIN: 3,
        MatchResult.DRAW: 1,
    }

    total_weighted_points = 0
    total_weight = 0

    for result, weight in zip(recent_results, weights):
        points = points_map.get(result, 0)
        total_weighted_points += points * weight
        total_weight += weight * 3  # 3 es el máximo de puntos posible

    if total_weight == 0:
        return 50.0

    form_rating = (total_weighted_points / total_weight) * 100
    return clamp_value(form_rating, 0.0, 100.0)


def estimate_team_strength(stats: Dict[str, float]) -> Dict[str, float]:
    """
    Estima fortalezas de equipo basado en estadísticas.

    Args:
        stats: Diccionario con estadísticas del equipo

    Returns:
        Diccionario con fortalezas estimadas
    """
    # Valores por defecto
    defaults = {
        "goals_per_game": 1.5,
        "goals_conceded_per_game": 1.5,
        "win_percentage": 0.33,
        "points_per_game": 1.0,
    }

    # Usar valores proporcionados o valores por defecto
    goals_for = stats.get("goals_per_game", defaults["goals_per_game"])
    goals_against = stats.get(
        "goals_conceded_per_game", defaults["goals_conceded_per_game"]
    )
    win_rate = stats.get("win_percentage", defaults["win_percentage"])
    ppg = stats.get("points_per_game", defaults["points_per_game"])

    # Promedios de liga para normalización
    league_avg_goals = (
        FOOTBALL_DISTRIBUTIONS["goals_per_match"]["mean"] / 2
    )  # Por equipo

    # Calcular fortalezas relativas
    attack_strength = safe_divide(goals_for, league_avg_goals, 1.0)
    defense_strength = safe_divide(league_avg_goals, goals_against, 1.0)

    # Fortaleza general basada en múltiples métricas
    overall_strength = (
        attack_strength * 0.3
        + defense_strength * 0.3
        + (win_rate / 0.5) * 0.2  # Normalizado a 50% win rate promedio
        + (ppg / 1.5) * 0.2  # Normalizado a 1.5 PPG promedio
    )

    return {
        "attack_strength": clamp_value(attack_strength, 0.1, 3.0),
        "defense_strength": clamp_value(defense_strength, 0.1, 3.0),
        "overall_strength": clamp_value(overall_strength, 0.1, 3.0),
        "consistency": calculate_consistency_score(stats),
    }


def calculate_consistency_score(stats: Dict[str, float]) -> float:
    """
    Calcula score de consistencia del equipo.

    Args:
        stats: Estadísticas del equipo

    Returns:
        Score de consistencia (0-1)
    """
    # Métricas de consistencia
    win_rate = stats.get("win_percentage", 0.33)
    draw_rate = stats.get("draw_percentage", 0.33)
    loss_rate = stats.get("loss_percentage", 0.33)

    # Menor varianza en resultados = mayor consistencia
    variance = (
        (win_rate - 0.33) ** 2
        + (draw_rate - 0.33) ** 2
        + (loss_rate - 0.33) ** 2
    ) / 3

    # Convertir varianza a score de consistencia (inverso)
    consistency = 1.0 / (1.0 + variance * 10)

    return clamp_value(consistency, 0.0, 1.0)


def calculate_h2h_advantage(
    h2h_record: Dict[str, int], perspective: str = "home"
) -> float:
    """
    Calcula ventaja en historial directo.

    Args:
        h2h_record: Registro H2H {'wins': X, 'draws': Y, 'losses': Z}
        perspective: 'home' o 'away'

    Returns:
        Score de ventaja (-1 a 1, positivo = ventaja)
    """
    wins = h2h_record.get("wins", 0)
    draws = h2h_record.get("draws", 0)
    losses = h2h_record.get("losses", 0)

    total_matches = wins + draws + losses
    if total_matches == 0:
        return 0.0  # Sin datos, neutral

    # Calcular score basado en resultados
    win_rate = wins / total_matches
    loss_rate = losses / total_matches

    advantage = win_rate - loss_rate

    # Ajustar por número de partidos (más partidos = más confiable)
    confidence_factor = min(
        total_matches / 10, 1.0
    )  # Máximo confidence con 10+ partidos

    return clamp_value(advantage * confidence_factor, -1.0, 1.0)


def is_derby_match(home_team: str, away_team: str) -> bool:
    """
    Determina si un partido es un derby basado en los equipos.

    Args:
        home_team: Equipo local
        away_team: Equipo visitante

    Returns:
        True si es un derby
    """
    # Definir derbies conocidos
    derbies = {
        ("Real Madrid", "Barcelona"),  # El Clásico
        ("Real Madrid", "Atletico Madrid"),  # Derby de Madrid
        ("Barcelona", "Espanyol"),  # Derby Barcelonés
        ("Manchester United", "Manchester City"),  # Derby de Manchester
        ("Manchester United", "Liverpool"),  # North West Derby
        ("Liverpool", "Everton"),  # Merseyside Derby
        ("Arsenal", "Tottenham Hotspur"),  # North London Derby
        ("Chelsea", "Arsenal"),  # London Derby
        ("AC Milan", "Inter Milan"),  # Derby della Madonnina
        ("Juventus", "Torino"),  # Derby della Mole
        ("Roma", "Lazio"),  # Derby della Capitale
        ("Bayern Munich", "Borussia Dortmund"),  # Der Klassiker
        ("Bayern Munich", "1860 Munich"),  # Münchner Stadtderby
    }

    # Normalizar nombres
    home_normalized = normalize_team_name(home_team)
    away_normalized = normalize_team_name(away_team)

    # Verificar en ambas direcciones
    return (home_normalized, away_normalized) in derbies or (
        away_normalized,
        home_normalized,
    ) in derbies


# =============================================================================
# UTILIDADES DE HASH Y CACHE
# =============================================================================


def generate_match_id(home_team: str, away_team: str, date: datetime) -> str:
    """
    Genera ID único para un partido.

    Args:
        home_team: Equipo local
        away_team: Equipo visitante
        date: Fecha del partido

    Returns:
        ID único del partido
    """
    # Normalizar nombres de equipos
    home_normalized = normalize_team_name(home_team)
    away_normalized = normalize_team_name(away_team)

    # Crear string único
    match_string = (
        f"{home_normalized}vs{away_normalized}_{date.strftime('%Y%m%d')}"
    )

    # Generar hash MD5
    return hashlib.md5(match_string.encode("utf-8")).hexdigest()[:12]


def generate_cache_key(*args, **kwargs) -> str:
    """
    Genera clave de cache basada en argumentos.

    Args:
        *args: Argumentos posicionales
        **kwargs: Argumentos con nombre

    Returns:
        Clave de cache
    """
    # Convertir argumentos a string
    args_str = "_".join(str(arg) for arg in args)
    kwargs_str = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))

    cache_string = f"{args_str}_{kwargs_str}"

    # Generar hash para clave corta y consistente
    return hashlib.sha256(cache_string.encode("utf-8")).hexdigest()[:16]


def hash_object(obj: Any) -> str:
    """
    Genera hash de un objeto para comparaciones.

    Args:
        obj: Objeto a hashear

    Returns:
        Hash del objeto
    """
    if hasattr(obj, "__dataclass_fields__"):
        obj = asdict(obj)

    obj_str = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(obj_str.encode("utf-8")).hexdigest()


# =============================================================================
# UTILIDADES DE VALIDACIÓN AVANZADA
# =============================================================================


def validate_team_in_league(team: str, league: str) -> bool:
    """
    Valida que un equipo pertenezca a una liga específica.

    Args:
        team: Nombre del equipo
        league: Código de la liga

    Returns:
        True si el equipo está en la liga
    """
    if league not in ALL_KNOWN_TEAMS:
        return False

    normalized_team = normalize_team_name(team)
    league_teams = {normalize_team_name(t) for t in ALL_KNOWN_TEAMS[league]}

    return normalized_team in league_teams


def validate_match_data_consistency(match_data: Dict[str, Any]) -> List[str]:
    """
    Valida consistencia de datos de partido.

    Args:
        match_data: Datos del partido

    Returns:
        Lista de errores encontrados
    """
    errors = []

    # Validar equipos diferentes
    home_team = match_data.get("home_team", "")
    away_team = match_data.get("away_team", "")

    if normalize_team_name(home_team) == normalize_team_name(away_team):
        errors.append("El equipo local y visitante no pueden ser el mismo")

    # Validar goles
    home_goals = match_data.get("home_goals")
    away_goals = match_data.get("away_goals")

    if home_goals is not None and away_goals is not None:
        max_goals = DATA_VALIDATION_LIMITS["max_goals_per_team"]

        if home_goals < 0 or away_goals < 0:
            errors.append("Los goles no pueden ser negativos")

        if home_goals > max_goals or away_goals > max_goals:
            errors.append(f"Demasiados goles (máximo {max_goals} por equipo)")

        if (home_goals + away_goals) > max_goals:
            errors.append(
                f"Total de goles excede límite razonable ({max_goals})"
            )

    # Validar fecha
    match_date = match_data.get("date")
    if match_date:
        if isinstance(match_date, str):
            parsed_date = parse_flexible_date(match_date)
            if not parsed_date:
                errors.append("Formato de fecha inválido")
            else:
                match_date = parsed_date

        if isinstance(match_date, datetime):
            # Verificar que no sea demasiado en el pasado o futuro
            now = datetime.now()

            if match_date < (now - timedelta(days=365 * 5)):
                errors.append("Fecha demasiado antigua (más de 5 años)")

            if match_date > (now + timedelta(days=365)):
                errors.append("Fecha demasiado en el futuro (más de 1 año)")

    # Validar liga
    league = match_data.get("league")
    if league and league not in LEAGUE_CODES:
        errors.append(f"Liga no reconocida: {league}")

    return errors


def validate_prediction_consistency(
    prediction_data: Dict[str, Any],
) -> List[str]:
    """
    Valida consistencia de datos de predicción.

    Args:
        prediction_data: Datos de predicción

    Returns:
        Lista de errores encontrados
    """
    errors = []

    # Validar probabilidades
    probs = prediction_data.get("probabilities", {})
    if probs:
        total_prob = sum(probs.values())

        if abs(total_prob - 1.0) > 0.02:  # Tolerancia de 2%
            errors.append(
                f"Probabilidades no suman 1.0 (suman {total_prob:.3f})"
            )

        for outcome, prob in probs.items():
            if prob < 0 or prob > 1:
                errors.append(f"Probabilidad inválida para {outcome}: {prob}")

    # Validar goles esperados
    expected_goals_home = prediction_data.get("expected_goals_home")
    expected_goals_away = prediction_data.get("expected_goals_away")

    max_expected = DATA_VALIDATION_LIMITS["max_expected_goals"]

    if expected_goals_home is not None:
        if expected_goals_home < 0 or expected_goals_home > max_expected:
            errors.append(
                f"Goles esperados local inválidos: {expected_goals_home}"
            )

    if expected_goals_away is not None:
        if expected_goals_away < 0 or expected_goals_away > max_expected:
            errors.append(
                f"Goles esperados visitante inválidos: {expected_goals_away}"
            )

    # Validar confidence score
    confidence = prediction_data.get("confidence_score")
    if confidence is not None:
        if confidence < 0 or confidence > 1:
            errors.append(f"Score de confianza inválido: {confidence}")

    return errors


# =============================================================================
# UTILIDADES ASÍNCRONAS
# =============================================================================


async def async_retry(
    func, max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0
):
    """
    Reintenta función asíncrona con backoff exponencial.

    Args:
        func: Función asíncrona a ejecutar
        max_attempts: Número máximo de intentos
        delay: Delay inicial en segundos
        backoff: Factor de backoff

    Returns:
        Resultado de la función
    """
    last_exception = None

    for attempt in range(max_attempts):
        try:
            return await func()
        except Exception as e:
            last_exception = e

            if attempt < max_attempts - 1:
                wait_time = delay * (backoff**attempt)
                logger.warning(
                    f"⚠️ Intento {attempt + 1} falló, reintentando en {wait_time}s: {str(e)}"
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"❌ Todos los intentos fallaron: {str(e)}")

    raise last_exception


async def async_timeout(coro, timeout_seconds: float):
    """
    Ejecuta corrutina con timeout.

    Args:
        coro: Corrutina a ejecutar
        timeout_seconds: Timeout en segundos

    Returns:
        Resultado de la corrutina
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.error(f"❌ Timeout después de {timeout_seconds}s")
        raise


async def batch_process_async(
    items: List[Any], process_func, batch_size: int = 10, delay: float = 0.1
):
    """
    Procesa lista de items en lotes de manera asíncrona.

    Args:
        items: Lista de items a procesar
        process_func: Función async que procesa cada item
        batch_size: Tamaño del lote
        delay: Delay entre lotes

    Returns:
        Lista de resultados
    """
    results = []

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]

        # Procesar lote en paralelo
        batch_tasks = [process_func(item) for item in batch]
        batch_results = await asyncio.gather(
            *batch_tasks, return_exceptions=True
        )

        results.extend(batch_results)

        # Delay entre lotes
        if i + batch_size < len(items):
            await asyncio.sleep(delay)

    return results


# =============================================================================
# UTILIDADES DE FORMATEO Y DISPLAY
# =============================================================================


def format_match_display(
    home_team: str,
    away_team: str,
    home_goals: Optional[int] = None,
    away_goals: Optional[int] = None,
    status: str = "SCHEDULED",
) -> str:
    """
    Formatea display de partido.

    Args:
        home_team: Equipo local
        away_team: Equipo visitante
        home_goals: Goles local (opcional)
        away_goals: Goles visitante (opcional)
        status: Estado del partido

    Returns:
        String formateado del partido
    """
    home_abbr = extract_team_abbreviation(home_team)
    away_abbr = extract_team_abbreviation(away_team)

    if home_goals is not None and away_goals is not None:
        return f"{home_abbr} {home_goals}-{away_goals} {away_abbr}"
    else:
        return f"{home_abbr} vs {away_abbr}"


def format_probability_display(probabilities: Dict[str, float]) -> str:
    """
    Formatea display de probabilidades.

    Args:
        probabilities: Diccionario de probabilidades

    Returns:
        String formateado de probabilidades
    """
    formatted = []

    for outcome, prob in probabilities.items():
        percentage = prob * 100
        formatted.append(f"{outcome.upper()}: {percentage:.1f}%")

    return " | ".join(formatted)


def format_odds_display(
    odds: Dict[str, float], format: OddsFormat = OddsFormat.DECIMAL
) -> str:
    """
    Formatea display de cuotas.

    Args:
        odds: Diccionario de cuotas
        format: Formato de display

    Returns:
        String formateado de cuotas
    """
    formatted = []

    for outcome, odd in odds.items():
        if format == OddsFormat.DECIMAL:
            formatted.append(f"{outcome.upper()}: {odd:.2f}")
        elif format == OddsFormat.AMERICAN:
            american_odd = convert_odds_format(
                odd, OddsFormat.DECIMAL, OddsFormat.AMERICAN
            )
            sign = "+" if american_odd > 0 else ""
            formatted.append(f"{outcome.upper()}: {sign}{american_odd:.0f}")
        elif format == OddsFormat.FRACTIONAL:
            fractional_odd = convert_odds_format(
                odd, OddsFormat.DECIMAL, OddsFormat.FRACTIONAL
            )
            formatted.append(f"{outcome.upper()}: {fractional_odd:.2f}/1")

    return " | ".join(formatted)


def format_confidence_display(confidence: float) -> str:
    """
    Formatea display de nivel de confianza.

    Args:
        confidence: Score de confianza (0-1)

    Returns:
        String formateado de confianza
    """
    percentage = confidence * 100

    # Determinar nivel
    for level, (min_val, max_val) in CONFIDENCE_LEVELS.items():
        if min_val <= confidence < max_val:
            level_text = level.replace("_", " ").title()
            return f"{percentage:.1f}% ({level_text})"

    return f"{percentage:.1f}%"


def create_ascii_table(
    data: List[Dict], headers: Optional[List[str]] = None
) -> str:
    """
    Crea tabla ASCII simple para display en consola.

    Args:
        data: Lista de diccionarios con datos
        headers: Headers opcionales

    Returns:
        String con tabla ASCII
    """
    if not data:
        return "No data available"

    # Usar headers proporcionados o keys del primer item
    if headers is None:
        headers = list(data[0].keys())

    # Calcular ancho de columnas
    col_widths = {}
    for header in headers:
        col_widths[header] = len(str(header))

        for row in data:
            value = str(row.get(header, ""))
            col_widths[header] = max(col_widths[header], len(value))

    # Construir tabla
    table_lines = []

    # Header
    header_line = " | ".join(
        header.ljust(col_widths[header]) for header in headers
    )
    separator_line = "-" * len(header_line)

    table_lines.append(header_line)
    table_lines.append(separator_line)

    # Filas de datos
    for row in data:
        row_line = " | ".join(
            str(row.get(header, "")).ljust(col_widths[header])
            for header in headers
        )
        table_lines.append(row_line)

    return "\n".join(table_lines)


# =============================================================================
# UTILIDADES DE PERFORMANCE Y MONITORING
# =============================================================================


class PerformanceMonitor:
    """Monitor de performance para funciones críticas."""

    def __init__(self):
        self.metrics = {}

    def start_timer(self, operation: str) -> str:
        """Inicia timer para una operación."""
        timer_id = f"{operation}_{datetime.now().timestamp()}"
        self.metrics[timer_id] = {
            "operation": operation,
            "start_time": datetime.now(),
            "end_time": None,
            "duration": None,
        }
        return timer_id

    def end_timer(self, timer_id: str) -> float:
        """Termina timer y calcula duración."""
        if timer_id not in self.metrics:
            return 0.0

        end_time = datetime.now()
        self.metrics[timer_id]["end_time"] = end_time

        duration = (
            end_time - self.metrics[timer_id]["start_time"]
        ).total_seconds()
        self.metrics[timer_id]["duration"] = duration

        return duration

    def get_operation_stats(self, operation: str) -> Dict[str, float]:
        """Obtiene estadísticas de una operación."""
        operation_times = [
            metric["duration"]
            for metric in self.metrics.values()
            if metric["operation"] == operation
            and metric["duration"] is not None
        ]

        if not operation_times:
            return {}

        return {
            "count": len(operation_times),
            "avg_duration": sum(operation_times) / len(operation_times),
            "min_duration": min(operation_times),
            "max_duration": max(operation_times),
            "total_duration": sum(operation_times),
        }

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Obtiene estadísticas de todas las operaciones."""
        operations = set(
            metric["operation"] for metric in self.metrics.values()
        )

        return {
            operation: self.get_operation_stats(operation)
            for operation in operations
        }


# Instancia global del monitor
performance_monitor = PerformanceMonitor()

# =============================================================================
# FUNCIONES DE UTILIDAD PARA TESTING
# =============================================================================


def create_mock_match_data(
    home_team: str = "Real Madrid",
    away_team: str = "Barcelona",
    date: Optional[datetime] = None,
    league: str = "PD",
) -> Dict[str, Any]:
    """
    Crea datos mock de partido para testing.

    Args:
        home_team: Equipo local
        away_team: Equipo visitante
        date: Fecha del partido
        league: Liga

    Returns:
        Diccionario con datos mock
    """
    if date is None:
        date = datetime.now() + timedelta(days=7)

    return {
        "match_id": generate_match_id(home_team, away_team, date),
        "home_team": home_team,
        "away_team": away_team,
        "date": date,
        "league": league,
        "season": get_season_from_date(date),
        "status": "SCHEDULED",
        "home_goals": None,
        "away_goals": None,
    }


def create_mock_prediction_data(match_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Crea datos mock de predicción para testing.

    Args:
        match_data: Datos del partido

    Returns:
        Diccionario con predicción mock
    """
    return {
        "match_id": match_data["match_id"],
        "probabilities": {"home": 0.485, "draw": 0.287, "away": 0.228},
        "expected_goals_home": 1.8,
        "expected_goals_away": 1.2,
        "confidence_score": 0.78,
        "model_version": "2.1.0",
        "created_at": datetime.now(),
    }


def create_mock_odds_data(match_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Crea datos mock de cuotas para testing.

    Args:
        match_data: Datos del partido

    Returns:
        Diccionario con cuotas mock
    """
    return {
        "match_id": match_data["match_id"],
        "bookmaker": "bet365",
        "odds": {"home": 2.20, "draw": 3.40, "away": 4.00},
        "market_type": "1x2",
        "timestamp": datetime.now(),
    }


# =============================================================================
# EXPORTACIONES
# =============================================================================

__all__ = [
    # Decoradores
    "timing_decorator",
    "async_timing_decorator",
    "safe_execution",
    "memoize_with_ttl",
    # Utilidades de texto
    "normalize_team_name",
    "extract_team_abbreviation",
    "sanitize_filename",
    "format_currency",
    # Utilidades de fecha
    "parse_flexible_date",
    "get_season_from_date",
    "is_weekend",
    "get_match_time_category",
    "days_until_match",
    # Utilidades matemáticas
    "safe_divide",
    "clamp_value",
    "normalize_probability_distribution",
    "calculate_percentage",
    "moving_average",
    "calculate_z_score",
    "is_outlier",
    # Utilidades de cuotas
    "convert_odds_format",
    "calculate_implied_probability",
    "calculate_overround",
    "calculate_fair_odds",
    "find_best_odds",
    # Utilidades de archivos
    "safe_json_load",
    "safe_json_save",
    "create_backup_filename",
    "clean_old_files",
    # Utilidades específicas de fútbol
    "calculate_goal_difference",
    "determine_match_result",
    "is_high_scoring_match",
    "calculate_team_form_rating",
    "estimate_team_strength",
    "calculate_consistency_score",
    "calculate_h2h_advantage",
    "is_derby_match",
    # Utilidades de hash y cache
    "generate_match_id",
    "generate_cache_key",
    "hash_object",
    # Utilidades de validación
    "validate_team_in_league",
    "validate_match_data_consistency",
    "validate_prediction_consistency",
    # Utilidades asíncronas
    "async_retry",
    "async_timeout",
    "batch_process_async",
    # Utilidades de formateo
    "format_match_display",
    "format_probability_display",
    "format_odds_display",
    "format_confidence_display",
    "create_ascii_table",
    # Performance monitoring
    "PerformanceMonitor",
    "performance_monitor",
    # Utilidades para testing
    "create_mock_match_data",
    "create_mock_prediction_data",
    "create_mock_odds_data",
]
