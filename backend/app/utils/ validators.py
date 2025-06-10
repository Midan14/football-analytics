"""
Football Analytics - Validators
Sistema completo de validación para todos los componentes del proyecto
"""

import re
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Error de validación personalizado"""
    pass

class ValidationLevel(Enum):
    """Niveles de validación"""
    STRICT = "strict"      # Validación estricta
    NORMAL = "normal"      # Validación normal
    LENIENT = "lenient"    # Validación permisiva

@dataclass
class ValidationResult:
    """Resultado de una validación"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    field: Optional[str] = None
    value: Optional[Any] = None
    
    def add_error(self, message: str):
        """Agrega un error"""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str):
        """Agrega una advertencia"""
        self.warnings.append(message)
    
    @property
    def has_errors(self) -> bool:
        """Verifica si hay errores"""
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        """Verifica si hay advertencias"""
        return len(self.warnings) > 0

class BaseValidator:
    """Validador base para todos los validadores específicos"""
    
    def __init__(self, level: ValidationLevel = ValidationLevel.NORMAL):
        self.level = level
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate(self, value: Any, **kwargs) -> ValidationResult:
        """Método principal de validación - debe ser implementado por subclases"""
        raise NotImplementedError("Subclases deben implementar el método validate")
    
    def _create_result(self, field: str = None, value: Any = None) -> ValidationResult:
        """Crea un resultado de validación inicializado"""
        return ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            field=field,
            value=value
        )

class TeamValidator(BaseValidator):
    """Validador para datos de equipos"""
    
    # Equipos conocidos por liga
    KNOWN_TEAMS = {
        'La Liga': [
            'Real Madrid', 'Barcelona', 'Atletico Madrid', 'Valencia', 'Sevilla',
            'Athletic Bilbao', 'Real Sociedad', 'Villarreal', 'Real Betis', 'Celta Vigo',
            'Osasuna', 'Granada', 'Getafe', 'Levante', 'Alaves', 'Cadiz',
            'Elche', 'Mallorca', 'Espanyol', 'Rayo Vallecano'
        ],
        'Premier League': [
            'Manchester City', 'Liverpool', 'Chelsea', 'Arsenal', 'Manchester United',
            'Tottenham', 'Newcastle', 'Brighton', 'Aston Villa', 'West Ham',
            'Crystal Palace', 'Fulham', 'Wolves', 'Everton', 'Brentford',
            'Southampton', 'Leicester', 'Leeds United', 'Nottingham Forest', 'Bournemouth'
        ],
        'Serie A': [
            'Juventus', 'Inter Milan', 'AC Milan', 'Napoli', 'Roma', 'Lazio',
            'Atalanta', 'Fiorentina', 'Bologna', 'Torino', 'Sassuolo', 'Udinese',
            'Genoa', 'Empoli', 'Verona', 'Spezia', 'Sampdoria', 'Cagliari',
            'Venezia', 'Salernitana'
        ],
        'Bundesliga': [
            'Bayern Munich', 'Borussia Dortmund', 'RB Leipzig', 'Bayer Leverkusen',
            'Union Berlin', 'Freiburg', 'Cologne', 'Mainz', 'Hoffenheim',
            'Wolfsburg', 'Augsburg', 'Stuttgart', 'Hertha Berlin', 'Arminia Bielefeld',
            'Greuther Furth', 'Eintracht Frankfurt', 'Borussia Monchengladbach', 'Bochum'
        ]
    }
    
    def validate_team_name(self, team_name: str, league: str = None) -> ValidationResult:
        """Valida el nombre de un equipo"""
        result = self._create_result("team_name", team_name)
        
        if not team_name or not isinstance(team_name, str):
            result.add_error("Nombre de equipo requerido y debe ser string")
            return result
        
        # Verificar longitud
        if len(team_name.strip()) < 2:
            result.add_error("Nombre de equipo debe tener al menos 2 caracteres")
        
        if len(team_name) > 50:
            result.add_error("Nombre de equipo no puede exceder 50 caracteres")
        
        # Verificar caracteres válidos
        if not re.match(r'^[a-zA-Z0-9\s\-\.\']+$', team_name):
            result.add_error("Nombre de equipo contiene caracteres inválidos")
        
        # Verificar si el equipo es conocido en la liga
        if league and league in self.KNOWN_TEAMS:
            if team_name not in self.KNOWN_TEAMS[league]:
                if self.level == ValidationLevel.STRICT:
                    result.add_error(f"Equipo '{team_name}' no reconocido en {league}")
                else:
                    result.add_warning(f"Equipo '{team_name}' no está en la lista conocida de {league}")
        
        return result
    
    def validate_team_stats(self, stats: Dict[str, Any]) -> ValidationResult:
        """Valida estadísticas de un equipo"""
        result = self._create_result("team_stats", stats)
        
        if not isinstance(stats, dict):
            result.add_error("Estadísticas deben ser un diccionario")
            return result
        
        required_fields = ['matches_played', 'wins', 'draws', 'losses', 'goals_for', 'goals_against']
        
        for field in required_fields:
            if field not in stats:
                result.add_error(f"Campo requerido faltante: {field}")
                continue
            
            value = stats[field]
            if not isinstance(value, (int, float)) or value < 0:
                result.add_error(f"{field} debe ser un número no negativo")
        
        # Validaciones lógicas
        if all(field in stats for field in required_fields):
            matches_played = stats['matches_played']
            wins = stats['wins']
            draws = stats['draws']
            losses = stats['losses']
            
            # Verificar que wins + draws + losses = matches_played
            total_matches = wins + draws + losses
            if total_matches != matches_played:
                result.add_error(f"Wins + Draws + Losses ({total_matches}) debe igual Matches Played ({matches_played})")
            
            # Verificar rangos razonables
            if matches_played > 60:  # Una temporada típica
                result.add_warning(f"Partidos jugados ({matches_played}) parece muy alto")
            
            if stats['goals_for'] > matches_played * 10:  # Más de 10 goles por partido
                result.add_warning("Goles a favor parecen excesivamente altos")
            
            if stats['goals_against'] > matches_played * 10:
                result.add_warning("Goles en contra parecen excesivamente altos")
        
        return result

class MatchValidator(BaseValidator):
    """Validador para datos de partidos"""
    
    def validate_match_data(self, match_data: Dict[str, Any]) -> ValidationResult:
        """Valida datos completos de un partido"""
        result = self._create_result("match_data", match_data)
        
        if not isinstance(match_data, dict):
            result.add_error("Datos de partido deben ser un diccionario")
            return result
        
        required_fields = ['home_team', 'away_team', 'match_date']
        
        for field in required_fields:
            if field not in match_data:
                result.add_error(f"Campo requerido faltante: {field}")
        
        # Validar equipos
        if 'home_team' in match_data and 'away_team' in match_data:
            if match_data['home_team'] == match_data['away_team']:
                result.add_error("Equipo local y visitante no pueden ser el mismo")
        
        # Validar fecha
        if 'match_date' in match_data:
            date_result = self.validate_match_date(match_data['match_date'])
            result.errors.extend(date_result.errors)
            result.warnings.extend(date_result.warnings)
        
        # Validar goles si están presentes
        if 'home_goals' in match_data and 'away_goals' in match_data:
            goals_result = self.validate_match_goals(
                match_data['home_goals'], 
                match_data['away_goals']
            )
            result.errors.extend(goals_result.errors)
            result.warnings.extend(goals_result.warnings)
        
        # Validar liga
        if 'league' in match_data:
            league_result = self.validate_league(match_data['league'])
            result.errors.extend(league_result.errors)
            result.warnings.extend(league_result.warnings)
        
        return result
    
    def validate_match_date(self, date_value: Any) -> ValidationResult:
        """Valida fecha de un partido"""
        result = self._create_result("match_date", date_value)
        
        # Convertir a datetime si es string
        if isinstance(date_value, str):
            try:
                date_value = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
            except ValueError:
                result.add_error("Formato de fecha inválido. Use ISO format (YYYY-MM-DDTHH:MM:SS)")
                return result
        
        if not isinstance(date_value, datetime):
            result.add_error("Fecha debe ser datetime o string en formato ISO")
            return result
        
        # Verificar rangos razonables
        now = datetime.now()
        min_date = datetime(1900, 1, 1)
        max_date = now + timedelta(days=365)  # Máximo 1 año en el futuro
        
        if date_value < min_date:
            result.add_error(f"Fecha demasiado antigua: {date_value}")
        
        if date_value > max_date:
            result.add_error(f"Fecha demasiado lejana en el futuro: {date_value}")
        
        # Advertencias para fechas sospechosas
        if date_value > now + timedelta(days=30):
            result.add_warning("Fecha está más de 30 días en el futuro")
        
        return result
    
    def validate_match_goals(self, home_goals: Any, away_goals: Any) -> ValidationResult:
        """Valida goles de un partido"""
        result = self._create_result("match_goals", (home_goals, away_goals))
        
        # Validar tipos
        if not isinstance(home_goals, int) or not isinstance(away_goals, int):
            result.add_error("Goles deben ser números enteros")
            return result
        
        # Validar rangos
        if home_goals < 0 or away_goals < 0:
            result.add_error("Goles no pueden ser negativos")
        
        if home_goals > 20 or away_goals > 20:
            result.add_error("Más de 20 goles por equipo es poco realista")
        
        # Advertencias
        total_goals = home_goals + away_goals
        if total_goals > 10:
            result.add_warning(f"Total de goles ({total_goals}) es inusualmente alto")
        
        if total_goals == 0:
            result.add_warning("Partido sin goles (0-0)")
        
        return result
    
    def validate_league(self, league: str) -> ValidationResult:
        """Valida nombre de liga"""
        result = self._create_result("league", league)
        
        if not league or not isinstance(league, str):
            result.add_error("Liga requerida y debe ser string")
            return result
        
        known_leagues = [
            'La Liga', 'Premier League', 'Serie A', 'Bundesliga', 'Ligue 1',
            'Eredivisie', 'Primeira Liga', 'Champions League', 'Europa League',
            'Copa del Rey', 'FA Cup', 'Coppa Italia', 'DFB-Pokal'
        ]
        
        if league not in known_leagues:
            if self.level == ValidationLevel.STRICT:
                result.add_error(f"Liga '{league}' no reconocida")
            else:
                result.add_warning(f"Liga '{league}' no está en la lista conocida")
        
        return result

class PredictionValidator(BaseValidator):
    """Validador para predicciones"""
    
    def validate_prediction_probabilities(self, probabilities: Dict[str, float]) -> ValidationResult:
        """Valida probabilidades de predicción"""
        result = self._create_result("probabilities", probabilities)
        
        if not isinstance(probabilities, dict):
            result.add_error("Probabilidades deben ser un diccionario")
            return result
        
        required_outcomes = ['home', 'draw', 'away']
        
        # Verificar que todas las probabilidades están presentes
        for outcome in required_outcomes:
            if outcome not in probabilities:
                result.add_error(f"Probabilidad faltante: {outcome}")
                continue
            
            prob = probabilities[outcome]
            
            # Verificar tipo y rango
            if not isinstance(prob, (int, float)):
                result.add_error(f"Probabilidad '{outcome}' debe ser numérica")
                continue
            
            if prob < 0 or prob > 1:
                result.add_error(f"Probabilidad '{outcome}' debe estar entre 0 y 1, got {prob}")
            
            # Advertencias para probabilidades extremas
            if prob < 0.01:
                result.add_warning(f"Probabilidad '{outcome}' muy baja: {prob}")
            
            if prob > 0.95:
                result.add_warning(f"Probabilidad '{outcome}' muy alta: {prob}")
        
        # Verificar que suman 1
        if all(outcome in probabilities for outcome in required_outcomes):
            total = sum(probabilities[outcome] for outcome in required_outcomes)
            
            if abs(total - 1.0) > 0.01:
                result.add_error(f"Probabilidades deben sumar 1.0, suman {total:.4f}")
            elif abs(total - 1.0) > 0.001:
                result.add_warning(f"Probabilidades suman {total:.4f}, idealmente 1.0")
        
        return result
    
    def validate_expected_goals(self, home_goals: float, away_goals: float) -> ValidationResult:
        """Valida goles esperados"""
        result = self._create_result("expected_goals", (home_goals, away_goals))
        
        # Verificar tipos
        if not isinstance(home_goals, (int, float)) or not isinstance(away_goals, (int, float)):
            result.add_error("Goles esperados deben ser numéricos")
            return result
        
        # Verificar rangos razonables
        if home_goals < 0 or away_goals < 0:
            result.add_error("Goles esperados no pueden ser negativos")
        
        if home_goals > 8 or away_goals > 8:
            result.add_error("Más de 8 goles esperados por equipo es poco realista")
        
        # Advertencias
        if home_goals > 5 or away_goals > 5:
            result.add_warning("Más de 5 goles esperados por equipo es alto")
        
        total_goals = home_goals + away_goals
        if total_goals > 6:
            result.add_warning(f"Total de goles esperados ({total_goals:.1f}) es muy alto")
        
        if total_goals < 1:
            result.add_warning(f"Total de goles esperados ({total_goals:.1f}) es muy bajo")
        
        return result
    
    def validate_confidence_score(self, confidence: float) -> ValidationResult:
        """Valida score de confianza"""
        result = self._create_result("confidence", confidence)
        
        if not isinstance(confidence, (int, float)):
            result.add_error("Confianza debe ser numérica")
            return result
        
        if confidence < 0 or confidence > 1:
            result.add_error(f"Confianza debe estar entre 0 y 1, got {confidence}")
        
        # Advertencias
        if confidence < 0.3:
            result.add_warning("Confianza muy baja para hacer predicciones útiles")
        
        if confidence > 0.95:
            result.add_warning("Confianza extremadamente alta, verificar overfitting")
        
        return result

class OddsValidator(BaseValidator):
    """Validador para cuotas de apuestas"""
    
    def validate_odds_format(self, odds: Dict[str, float]) -> ValidationResult:
        """Valida formato de cuotas"""
        result = self._create_result("odds", odds)
        
        if not isinstance(odds, dict):
            result.add_error("Cuotas deben ser un diccionario")
            return result
        
        required_markets = ['home', 'draw', 'away']
        
        for market in required_markets:
            if market not in odds:
                result.add_error(f"Cuota faltante: {market}")
                continue
            
            odd_value = odds[market]
            
            # Verificar tipo
            if not isinstance(odd_value, (int, float)):
                result.add_error(f"Cuota '{market}' debe ser numérica")
                continue
            
            # Verificar rango razonable
            if odd_value < 1.01:
                result.add_error(f"Cuota '{market}' demasiado baja: {odd_value}")
            
            if odd_value > 1000:
                result.add_error(f"Cuota '{market}' demasiado alta: {odd_value}")
            
            # Advertencias
            if odd_value < 1.1:
                result.add_warning(f"Cuota '{market}' muy baja: {odd_value}")
            
            if odd_value > 50:
                result.add_warning(f"Cuota '{market}' muy alta: {odd_value}")
        
        return result
    
    def validate_odds_consistency(self, odds: Dict[str, float]) -> ValidationResult:
        """Valida consistencia de cuotas"""
        result = self._create_result("odds_consistency", odds)
        
        required_markets = ['home', 'draw', 'away']
        
        if not all(market in odds for market in required_markets):
            result.add_error("Faltan cuotas para validar consistencia")
            return result
        
        # Calcular probabilidades implícitas
        implied_probs = {market: 1/odds[market] for market in required_markets}
        total_implied_prob = sum(implied_probs.values())
        
        # Calcular overround (margen de la casa)
        overround = (total_implied_prob - 1) * 100
        
        if overround < -5:  # Arbitraje
            result.add_warning(f"Posible oportunidad de arbitraje detectada (overround: {overround:.2f}%)")
        elif overround > 20:  # Margen muy alto
            result.add_warning(f"Overround muy alto: {overround:.2f}%")
        
        # Verificar balance entre cuotas
        min_odd = min(odds.values())
        max_odd = max(odds.values())
        
        if max_odd / min_odd > 20:  # Ratio muy alto
            result.add_warning("Gran diferencia entre cuotas mínima y máxima")
        
        return result

class FeatureValidator(BaseValidator):
    """Validador para features de ML"""
    
    def validate_feature_dataframe(self, df: pd.DataFrame) -> ValidationResult:
        """Valida DataFrame de features"""
        result = self._create_result("feature_dataframe", df)
        
        if not isinstance(df, pd.DataFrame):
            result.add_error("Datos deben ser un pandas DataFrame")
            return result
        
        if df.empty:
            result.add_error("DataFrame no puede estar vacío")
            return result
        
        # Verificar tipos de datos
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            if self.level == ValidationLevel.STRICT:
                result.add_error(f"Columnas no numéricas encontradas: {non_numeric_cols}")
            else:
                result.add_warning(f"Columnas no numéricas: {non_numeric_cols}")
        
        # Verificar valores faltantes
        missing_counts = df.isnull().sum()
        cols_with_missing = missing_counts[missing_counts > 0]
        
        if len(cols_with_missing) > 0:
            if self.level == ValidationLevel.STRICT:
                result.add_error(f"Valores faltantes en columnas: {cols_with_missing.to_dict()}")
            else:
                result.add_warning(f"Valores faltantes encontrados: {cols_with_missing.to_dict()}")
        
        # Verificar valores infinitos
        numeric_df = df.select_dtypes(include=[np.number])
        inf_counts = np.isinf(numeric_df).sum()
        cols_with_inf = inf_counts[inf_counts > 0]
        
        if len(cols_with_inf) > 0:
            result.add_error(f"Valores infinitos en columnas: {cols_with_inf.to_dict()}")
        
        # Verificar varianza
        zero_var_cols = numeric_df.columns[numeric_df.var() == 0].tolist()
        if zero_var_cols:
            result.add_warning(f"Columnas con varianza cero: {zero_var_cols}")
        
        return result
    
    def validate_feature_values(self, features: Dict[str, float]) -> ValidationResult:
        """Valida valores específicos de features"""
        result = self._create_result("feature_values", features)
        
        if not isinstance(features, dict):
            result.add_error("Features deben ser un diccionario")
            return result
        
        for feature_name, value in features.items():
            # Verificar tipo
            if not isinstance(value, (int, float)):
                result.add_error(f"Feature '{feature_name}' debe ser numérica, got {type(value)}")
                continue
            
            # Verificar valores problemáticos
            if math.isnan(value):
                result.add_error(f"Feature '{feature_name}' es NaN")
                continue
            
            if math.isinf(value):
                result.add_error(f"Feature '{feature_name}' es infinito")
                continue
            
            # Validaciones específicas por tipo de feature
            self._validate_specific_feature(feature_name, value, result)
        
        return result
    
    def _validate_specific_feature(self, feature_name: str, value: float, result: ValidationResult):
        """Valida features específicos según su nombre"""
        
        # Features de fuerza (deben ser positivos)
        if 'strength' in feature_name.lower():
            if value <= 0:
                result.add_warning(f"Feature de fuerza '{feature_name}' debería ser positivo: {value}")
        
        # Features de porcentaje (deben estar entre 0 y 1)
        if any(keyword in feature_name.lower() for keyword in ['percentage', 'rate', 'ratio']):
            if value < 0 or value > 1:
                result.add_warning(f"Feature de porcentaje '{feature_name}' fuera de rango [0,1]: {value}")
        
        # Features de goles (deben ser razonables)
        if 'goals' in feature_name.lower():
            if value < 0:
                result.add_error(f"Feature de goles '{feature_name}' no puede ser negativo: {value}")
            elif value > 10:
                result.add_warning(f"Feature de goles '{feature_name}' muy alto: {value}")
        
        # Features de puntos por partido
        if 'points_per_game' in feature_name.lower() or 'ppg' in feature_name.lower():
            if value < 0 or value > 3:
                result.add_warning(f"Puntos por partido '{feature_name}' fuera de rango [0,3]: {value}")

class APIValidator(BaseValidator):
    """Validador para requests de API"""
    
    def validate_prediction_request(self, request_data: Dict[str, Any]) -> ValidationResult:
        """Valida request de predicción"""
        result = self._create_result("prediction_request", request_data)
        
        if not isinstance(request_data, dict):
            result.add_error("Request debe ser un diccionario")
            return result
        
        required_fields = ['home_team', 'away_team', 'league', 'match_date']
        
        for field in required_fields:
            if field not in request_data:
                result.add_error(f"Campo requerido faltante: {field}")
        
        # Validar campos específicos
        if 'home_team' in request_data:
            team_result = TeamValidator().validate_team_name(request_data['home_team'])
            result.errors.extend(team_result.errors)
            result.warnings.extend(team_result.warnings)
        
        if 'away_team' in request_data:
            team_result = TeamValidator().validate_team_name(request_data['away_team'])
            result.errors.extend(team_result.errors)
            result.warnings.extend(team_result.warnings)
        
        if 'match_date' in request_data:
            match_validator = MatchValidator()
            date_result = match_validator.validate_match_date(request_data['match_date'])
            result.errors.extend(date_result.errors)
            result.warnings.extend(date_result.warnings)
        
        # Validar cuotas si están presentes
        if 'market_odds' in request_data:
            odds_result = OddsValidator().validate_odds_format(request_data['market_odds'])
            result.errors.extend(odds_result.errors)
            result.warnings.extend(odds_result.warnings)
        
        return result
    
    def validate_pagination_params(self, page: int, per_page: int) -> ValidationResult:
        """Valida parámetros de paginación"""
        result = self._create_result("pagination", (page, per_page))
        
        if not isinstance(page, int) or not isinstance(per_page, int):
            result.add_error("Page y per_page deben ser enteros")
            return result
        
        if page < 1:
            result.add_error("Page debe ser mayor a 0")
        
        if per_page < 1:
            result.add_error("Per_page debe ser mayor a 0")
        
        if per_page > 1000:
            result.add_error("Per_page no puede exceder 1000")
        
        if per_page > 100:
            result.add_warning("Per_page mayor a 100 puede afectar performance")
        
        return result

class ConfigValidator(BaseValidator):
    """Validador para configuración del sistema"""
    
    def validate_api_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Valida configuración de APIs"""
        result = self._create_result("api_config", config)
        
        if not isinstance(config, dict):
            result.add_error("Configuración debe ser un diccionario")
            return result
        
        # Validar API keys
        api_keys = ['football_data_api_key', 'rapidapi_key', 'odds_api_key']
        
        for key_name in api_keys:
            if key_name in config:
                key_value = config[key_name]
                
                if not isinstance(key_value, str):
                    result.add_error(f"{key_name} debe ser string")
                    continue
                
                if len(key_value) < 10:
                    result.add_warning(f"{key_name} parece muy corta")
                
                if len(key_value) > 100:
                    result.add_warning(f"{key_name} parece muy larga")
                
                # Verificar formato hexadecimal para ciertas keys
                if 'football_data' in key_name:
                    if not re.match(r'^[a-f0-9]{32}$', key_value):
                        result.add_warning(f"{key_name} no tiene formato hexadecimal esperado")
        
        # Validar rate limits
        if 'rate_limits' in config:
            rate_limits = config['rate_limits']
            
            if not isinstance(rate_limits, dict):
                result.add_error("Rate limits debe ser un diccionario")
            else:
                for service, limit in rate_limits.items():
                    if not isinstance(limit, int) or limit <= 0:
                        result.add_error(f"Rate limit para {service} debe ser entero positivo")
        
        return result
    
    def validate_database_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Valida configuración de base de datos"""
        result = self._create_result("database_config", config)
        
        if not isinstance(config, dict):
            result.add_error("Configuración de DB debe ser un diccionario")
            return result
        
        # Validar tipo de base de datos
        if 'type' in config:
            db_type = config['type']
            if db_type not in ['sqlite', 'postgresql', 'mysql']:
                result.add_warning(f"Tipo de DB no reconocido: {db_type}")
        
        # Validar path para SQLite
        if config.get('type') == 'sqlite':
            if 'path' not in config:
                result.add_error("Path requerido para SQLite")
            else:
                path = config['path']
                if path != ':memory:' and not isinstance(path, str):
                    result.add_error("Path de SQLite debe ser string")
        
        # Validar conexiones máximas
        if 'max_connections' in config:
            max_conn = config['max_connections']
            if not isinstance(max_conn, int) or max_conn <= 0:
                result.add_error("max_connections debe ser entero positivo")
            elif max_conn > 1000:
                result.add_warning("max_connections muy alto, puede causar problemas")
        
        return result

class ComprehensiveValidator:
    """Validador integral que combina todos los validadores específicos"""
    
    def __init__(self, level: ValidationLevel = ValidationLevel.NORMAL):
        self.level = level
        self.team_validator = TeamValidator(level)
        self.match_validator = MatchValidator(level)
        self.prediction_validator = PredictionValidator(level)
        self.odds_validator = OddsValidator(level)
        self.feature_validator = FeatureValidator(level)
        self.api_validator = APIValidator(level)
        self.config_validator = ConfigValidator(level)
        
        self.logger = logging.getLogger(__name__)
    
    def validate_full_prediction_pipeline(self, data: Dict[str, Any]) -> ValidationResult:
        """Valida el pipeline completo de predicción"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], field="prediction_pipeline")
        
        # 1. Validar request inicial
        if 'request' in data:
            request_result = self.api_validator.validate_prediction_request(data['request'])
            result.errors.extend(request_result.errors)
            result.warnings.extend(request_result.warnings)
        
        # 2. Validar features
        if 'features' in data:
            if isinstance(data['features'], pd.DataFrame):
                features_result = self.feature_validator.validate_feature_dataframe(data['features'])
            else:
                features_result = self.feature_validator.validate_feature_values(data['features'])
            result.errors.extend(features_result.errors)
            result.warnings.extend(features_result.warnings)
        
        # 3. Validar predicción resultante
        if 'prediction' in data:
            pred_data = data['prediction']
            
            if 'result_probabilities' in pred_data:
                prob_result = self.prediction_validator.validate_prediction_probabilities(
                    pred_data['result_probabilities']
                )
                result.errors.extend(prob_result.errors)
                result.warnings.extend(prob_result.warnings)
            
            if 'expected_goals_home' in pred_data and 'expected_goals_away' in pred_data:
                goals_result = self.prediction_validator.validate_expected_goals(
                    pred_data['expected_goals_home'],
                    pred_data['expected_goals_away']
                )
                result.errors.extend(goals_result.errors)
                result.warnings.extend(goals_result.warnings)
            
            if 'confidence_score' in pred_data:
                conf_result = self.prediction_validator.validate_confidence_score(
                    pred_data['confidence_score']
                )
                result.errors.extend(conf_result.errors)
                result.warnings.extend(conf_result.warnings)
        
        # 4. Validar cuotas si están presentes
        if 'market_odds' in data:
            odds_result = self.odds_validator.validate_odds_format(data['market_odds'])
            result.errors.extend(odds_result.errors)
            result.warnings.extend(odds_result.warnings)
            
            # Validar consistencia
            consistency_result = self.odds_validator.validate_odds_consistency(data['market_odds'])
            result.errors.extend(consistency_result.errors)
            result.warnings.extend(consistency_result.warnings)
        
        # Determinar si es válido
        result.is_valid = len(result.errors) == 0
        
        return result
    
    def validate_system_health(self, system_data: Dict[str, Any]) -> ValidationResult:
        """Valida la salud general del sistema"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], field="system_health")
        
        # Validar configuración
        if 'config' in system_data:
            config_result = self.config_validator.validate_api_config(system_data['config'])
            result.errors.extend(config_result.errors)
            result.warnings.extend(config_result.warnings)
        
        # Validar métricas de rendimiento
        if 'performance_metrics' in system_data:
            perf_result = self._validate_performance_metrics(system_data['performance_metrics'])
            result.errors.extend(perf_result.errors)
            result.warnings.extend(perf_result.warnings)
        
        # Validar estado de servicios
        if 'services' in system_data:
            services_result = self._validate_services_status(system_data['services'])
            result.errors.extend(services_result.errors)
            result.warnings.extend(services_result.warnings)
        
        result.is_valid = len(result.errors) == 0
        return result
    
    def _validate_performance_metrics(self, metrics: Dict[str, Any]) -> ValidationResult:
        """Valida métricas de rendimiento"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], field="performance_metrics")
        
        # Validar accuracy del modelo
        if 'model_accuracy' in metrics:
            accuracy = metrics['model_accuracy']
            if not isinstance(accuracy, (int, float)):
                result.add_error("Model accuracy debe ser numérica")
            elif accuracy < 0 or accuracy > 1:
                result.add_error("Model accuracy debe estar entre 0 y 1")
            elif accuracy < 0.4:
                result.add_warning("Model accuracy muy baja")
        
        # Validar tiempo de respuesta
        if 'response_time_ms' in metrics:
            response_time = metrics['response_time_ms']
            if not isinstance(response_time, (int, float)):
                result.add_error("Response time debe ser numérico")
            elif response_time < 0:
                result.add_error("Response time no puede ser negativo")
            elif response_time > 5000:  # 5 segundos
                result.add_warning("Response time muy alto")
        
        # Validar uso de memoria
        if 'memory_usage_mb' in metrics:
            memory = metrics['memory_usage_mb']
            if not isinstance(memory, (int, float)):
                result.add_error("Memory usage debe ser numérica")
            elif memory < 0:
                result.add_error("Memory usage no puede ser negativa")
            elif memory > 8192:  # 8GB
                result.add_warning("Memory usage muy alta")
        
        return result
    
    def _validate_services_status(self, services: Dict[str, Any]) -> ValidationResult:
        """Valida estado de servicios"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], field="services_status")
        
        required_services = ['predictor', 'data_collector', 'odds_calculator', 'live_tracker']
        
        for service_name in required_services:
            if service_name not in services:
                result.add_warning(f"Servicio {service_name} no reportado")
                continue
            
            service_status = services[service_name]
            
            if not isinstance(service_status, dict):
                result.add_error(f"Estado de {service_name} debe ser diccionario")
                continue
            
            # Verificar status
            if 'status' in service_status:
                status = service_status['status']
                if status not in ['healthy', 'unhealthy', 'degraded']:
                    result.add_warning(f"Status desconocido para {service_name}: {status}")
                elif status == 'unhealthy':
                    result.add_error(f"Servicio {service_name} no saludable")
                elif status == 'degraded':
                    result.add_warning(f"Servicio {service_name} degradado")
        
        return result

# Funciones de conveniencia
def validate_team_name(team_name: str, league: str = None, level: ValidationLevel = ValidationLevel.NORMAL) -> ValidationResult:
    """Función de conveniencia para validar nombre de equipo"""
    validator = TeamValidator(level)
    return validator.validate_team_name(team_name, league)

def validate_match_data(match_data: Dict[str, Any], level: ValidationLevel = ValidationLevel.NORMAL) -> ValidationResult:
    """Función de conveniencia para validar datos de partido"""
    validator = MatchValidator(level)
    return validator.validate_match_data(match_data)

def validate_prediction_probabilities(probabilities: Dict[str, float], level: ValidationLevel = ValidationLevel.NORMAL) -> ValidationResult:
    """Función de conveniencia para validar probabilidades"""
    validator = PredictionValidator(level)
    return validator.validate_prediction_probabilities(probabilities)

def validate_odds(odds: Dict[str, float], level: ValidationLevel = ValidationLevel.NORMAL) -> ValidationResult:
    """Función de conveniencia para validar cuotas"""
    validator = OddsValidator(level)
    format_result = validator.validate_odds_format(odds)
    
    if format_result.is_valid:
        consistency_result = validator.validate_odds_consistency(odds)
        format_result.errors.extend(consistency_result.errors)
        format_result.warnings.extend(consistency_result.warnings)
        format_result.is_valid = len(format_result.errors) == 0
    
    return format_result

def validate_features(features: Union[pd.DataFrame, Dict[str, float]], level: ValidationLevel = ValidationLevel.NORMAL) -> ValidationResult:
    """Función de conveniencia para validar features"""
    validator = FeatureValidator(level)
    
    if isinstance(features, pd.DataFrame):
        return validator.validate_feature_dataframe(features)
    else:
        return validator.validate_feature_values(features)

def validate_api_request(request_data: Dict[str, Any], level: ValidationLevel = ValidationLevel.NORMAL) -> ValidationResult:
    """Función de conveniencia para validar request de API"""
    validator = APIValidator(level)
    return validator.validate_prediction_request(request_data)

def validate_full_pipeline(pipeline_data: Dict[str, Any], level: ValidationLevel = ValidationLevel.NORMAL) -> ValidationResult:
    """Función de conveniencia para validar pipeline completo"""
    validator = ComprehensiveValidator(level)
    return validator.validate_full_prediction_pipeline(pipeline_data)

# Decoradores para validación automática
def validate_input(validator_func):
    """Decorador para validación automática de inputs"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Validar argumentos usando la función de validación proporcionada
            for arg in args[1:]:  # Saltar 'self'
                if isinstance(arg, dict):
                    result = validator_func(arg)
                    if not result.is_valid:
                        raise ValidationError(f"Validación falló: {result.errors}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Excepciones personalizadas
class TeamValidationError(ValidationError):
    """Error específico de validación de equipos"""
    pass

class MatchValidationError(ValidationError):
    """Error específico de validación de partidos"""
    pass

class PredictionValidationError(ValidationError):
    """Error específico de validación de predicciones"""
    pass

class OddsValidationError(ValidationError):
    """Error específico de validación de cuotas"""
    pass

class FeatureValidationError(ValidationError):
    """Error específico de validación de features"""
    pass

# Utilidades de validación
class ValidationUtils:
    """Utilidades para validación"""
    
    @staticmethod
    def is_valid_email(email: str) -> bool:
        """Valida formato de email"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}
        return re.match(pattern, email) is not None
    
    @staticmethod
    def is_valid_url(url: str) -> bool:
        """Valida formato de URL"""
        pattern = r'^https?://[^\s/$.?#].[^\s]*
        return re.match(pattern, url) is not None
    
    @staticmethod
    def is_probability_distribution(values: List[float], tolerance: float = 0.01) -> bool:
        """Verifica si los valores forman una distribución de probabilidad válida"""
        if not all(0 <= v <= 1 for v in values):
            return False
        return abs(sum(values) - 1.0) <= tolerance
    
    @staticmethod
    def normalize_team_name(team_name: str) -> str:
        """Normaliza nombre de equipo para comparaciones"""
        return re.sub(r'\s+', ' ', team_name.strip().title())
    
    @staticmethod
    def sanitize_string(text: str, max_length: int = 100) -> str:
        """Sanitiza string removiendo caracteres peligrosos"""
        # Remover caracteres especiales peligrosos
        text = re.sub(r'[<>"\';\\]', '', text)
        # Limitar longitud
        return text[:max_length].strip()

# Configuración de logging
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # Tests de validadores
    print("🔍 Football Analytics - Validators")
    print("Sistema de validación completo")
    print("=" * 50)
    
    # Test de validación de equipo
    print("\n1. Validando nombre de equipo:")
    result = validate_team_name("Real Madrid", "La Liga")
    print(f"   Real Madrid: {'✅ Válido' if result.is_valid else '❌ Inválido'}")
    if result.warnings:
        print(f"   Advertencias: {result.warnings}")
    
    # Test de validación de probabilidades
    print("\n2. Validando probabilidades:")
    probs = {'home': 0.485, 'draw': 0.287, 'away': 0.228}
    result = validate_prediction_probabilities(probs)
    print(f"   Probabilidades: {'✅ Válidas' if result.is_valid else '❌ Inválidas'}")
    if result.errors:
        print(f"   Errores: {result.errors}")
    
    # Test de validación de cuotas
    print("\n3. Validando cuotas:")
    odds = {'home': 2.20, 'draw': 3.40, 'away': 4.00}
    result = validate_odds(odds)
    print(f"   Cuotas: {'✅ Válidas' if result.is_valid else '❌ Inválidas'}")
    if result.warnings:
        print(f"   Advertencias: {result.warnings}")
    
    # Test de pipeline completo
    print("\n4. Validando pipeline completo:")
    pipeline_data = {
        'request': {
            'home_team': 'Real Madrid',
            'away_team': 'Barcelona',
            'league': 'La Liga',
            'match_date': '2024-06-15T15:00:00Z'
        },
        'prediction': {
            'result_probabilities': probs,
            'expected_goals_home': 1.8,
            'expected_goals_away': 1.2,
            'confidence_score': 0.78
        },
        'market_odds': odds
    }
    
    result = validate_full_pipeline(pipeline_data)
    print(f"   Pipeline: {'✅ Válido' if result.is_valid else '❌ Inválido'}")
    if result.errors:
        print(f"   Errores: {result.errors}")
    if result.warnings:
        print(f"   Advertencias: {result.warnings}")
    
    print(f"\n✅ Sistema de validación funcionando correctamente!")
    print(f"📊 Validadores disponibles: 7 especializados + 1 integral")
    print(f"🔧 Niveles de validación: STRICT, NORMAL, LENIENT")
    print(f"⚡ Funciones de conveniencia y decoradores incluidos")