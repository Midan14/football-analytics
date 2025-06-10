# backend/app/api/routes/matches.py
"""
Endpoints para gestión completa de partidos de fútbol.
Incluye CRUD, filtros avanzados, estadísticas, análisis y predicciones.
"""

import logging
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy import and_, asc, desc, extract, func, or_
from sqlalchemy.orm import Session, joinedload

# Imports internos
from ...database.connection import get_db
from ...database.models import (
    League,
    Match,
    MatchEvent,
    MatchPlayerStats,
    MatchTeamStats,
    Player,
    Season,
    Stadium,
    Team,
)
from ...services.calculator import StatisticalCalculator
from ...services.data_collector import DataCollector
from ...services.predictor import PredictionEngine

# Crear router
router = APIRouter()

# Configurar logging
logger = logging.getLogger(__name__)

# Instanciar servicios
prediction_engine = PredictionEngine()
stats_calculator = StatisticalCalculator()
data_collector = DataCollector()

# =====================================================
# ENUMS Y SCHEMAS
# =====================================================


class MatchStatus(str, Enum):
    """Estados posibles de un partido."""

    SCHEDULED = "scheduled"
    LIVE = "live"
    FINISHED = "finished"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"
    SUSPENDED = "suspended"


class SortBy(str, Enum):
    """Opciones de ordenamiento."""

    DATE_ASC = "date_asc"
    DATE_DESC = "date_desc"
    LEAGUE = "league"
    IMPORTANCE = "importance"


from pydantic import BaseModel, Field


class MatchResponse(BaseModel):
    """Schema de respuesta para partidos."""

    id: int
    league: Dict[str, Any]
    season: Optional[Dict[str, Any]] = None
    home_team: Dict[str, Any]
    away_team: Dict[str, Any]
    match_date: datetime
    status: str
    score: Dict[str, int]
    venue: Optional[Dict[str, Any]] = None
    referee: Optional[str] = None
    weather: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class MatchStatsResponse(BaseModel):
    """Schema de respuesta para estadísticas de partido."""

    match_id: int
    team_stats: Dict[str, Dict]
    player_stats: List[Dict]
    events: List[Dict]
    key_moments: List[Dict]


# =====================================================
# ENDPOINTS PRINCIPALES
# =====================================================


@router.get("/", response_model=Dict, summary="Obtener todos los partidos")
async def get_matches(
    # Filtros básicos
    skip: int = Query(0, ge=0, description="Número de registros a saltar"),
    limit: int = Query(50, ge=1, le=500, description="Límite de registros"),
    status: Optional[MatchStatus] = Query(
        None, description="Filtrar por estado"
    ),
    # Filtros por entidades
    league_id: Optional[int] = Query(None, description="Filtrar por liga"),
    season_id: Optional[int] = Query(
        None, description="Filtrar por temporada"
    ),
    team_id: Optional[int] = Query(
        None, description="Filtrar por equipo (local o visitante)"
    ),
    stadium_id: Optional[int] = Query(None, description="Filtrar por estadio"),
    # Filtros de fecha
    date_from: Optional[date] = Query(
        None, description="Fecha desde (YYYY-MM-DD)"
    ),
    date_to: Optional[date] = Query(
        None, description="Fecha hasta (YYYY-MM-DD)"
    ),
    today: Optional[bool] = Query(None, description="Solo partidos de hoy"),
    this_week: Optional[bool] = Query(
        None, description="Solo partidos de esta semana"
    ),
    # Filtros avanzados
    min_goals: Optional[int] = Query(
        None, description="Mínimo total de goles"
    ),
    max_goals: Optional[int] = Query(
        None, description="Máximo total de goles"
    ),
    has_events: Optional[bool] = Query(
        None, description="Solo partidos con eventos"
    ),
    importance: Optional[str] = Query(
        None, description="Importancia: high, medium, low"
    ),
    # Ordenamiento
    sort_by: SortBy = Query(SortBy.DATE_DESC, description="Ordenar por"),
    # Incluir datos adicionales
    include_stats: bool = Query(
        False, description="Incluir estadísticas básicas"
    ),
    include_events: bool = Query(
        False, description="Incluir eventos del partido"
    ),
    include_lineups: bool = Query(False, description="Incluir alineaciones"),
    include_predictions: bool = Query(
        False, description="Incluir predicciones"
    ),
    db: Session = Depends(get_db),
):
    """
    Obtener lista de partidos con filtros avanzados y datos opcionales.

    **Filtros disponibles:**
    - Por estado, liga, temporada, equipo, estadio
    - Por rangos de fecha y fechas especiales
    - Por características del partido (goles, eventos)
    - Ordenamiento flexible

    **Datos opcionales:**
    - Estadísticas del partido
    - Eventos y momentos clave
    - Alineaciones
    - Predicciones de ML
    """
    try:
        # Construir query base con joins optimizados
        query = db.query(Match).options(
            joinedload(Match.league),
            joinedload(Match.season),
            joinedload(Match.home_team),
            joinedload(Match.away_team),
            joinedload(Match.stadium),
        )

        # === APLICAR FILTROS ===

        # Filtro por estado
        if status:
            query = query.filter(Match.status == status.value)

        # Filtro por liga
        if league_id:
            query = query.filter(Match.league_id == league_id)

        # Filtro por temporada
        if season_id:
            query = query.filter(Match.season_id == season_id)

        # Filtro por equipo (local o visitante)
        if team_id:
            query = query.filter(
                or_(
                    Match.home_team_id == team_id,
                    Match.away_team_id == team_id,
                )
            )

        # Filtro por estadio
        if stadium_id:
            query = query.filter(Match.stadium_id == stadium_id)

        # === FILTROS DE FECHA ===

        if today:
            today_date = date.today()
            query = query.filter(func.date(Match.match_date) == today_date)

        elif this_week:
            today_date = date.today()
            week_start = today_date - timedelta(days=today_date.weekday())
            week_end = week_start + timedelta(days=6)
            query = query.filter(
                and_(
                    func.date(Match.match_date) >= week_start,
                    func.date(Match.match_date) <= week_end,
                )
            )

        else:
            if date_from:
                query = query.filter(func.date(Match.match_date) >= date_from)

            if date_to:
                query = query.filter(func.date(Match.match_date) <= date_to)

        # === FILTROS AVANZADOS ===

        # Filtro por total de goles
        if min_goals is not None:
            query = query.filter(
                (Match.home_score + Match.away_score) >= min_goals
            )

        if max_goals is not None:
            query = query.filter(
                (Match.home_score + Match.away_score) <= max_goals
            )

        # Filtro por eventos
        if has_events:
            query = query.join(MatchEvent).filter(
                MatchEvent.match_id == Match.id
            )

        # Filtro por importancia (simulado - implementar lógica real)
        if importance:
            if importance == "high":
                # Partidos importantes: derbis, playoffs, etc.
                query = query.filter(
                    or_(
                        Match.round_name.ilike("%final%"),
                        Match.round_name.ilike("%derby%"),
                        Match.round_name.ilike("%clasico%"),
                    )
                )

        # === ORDENAMIENTO ===

        if sort_by == SortBy.DATE_ASC:
            query = query.order_by(asc(Match.match_date))
        elif sort_by == SortBy.DATE_DESC:
            query = query.order_by(desc(Match.match_date))
        elif sort_by == SortBy.LEAGUE:
            query = query.join(League).order_by(League.name, Match.match_date)
        elif sort_by == SortBy.IMPORTANCE:
            # Ordenar por importancia (simulado)
            query = query.order_by(
                desc(Match.round_name.ilike("%final%")), desc(Match.match_date)
            )

        # Obtener total antes de paginación
        total_matches = query.count()

        # Aplicar paginación
        matches = query.offset(skip).limit(limit).all()

        # === CONSTRUIR RESPUESTA ===

        matches_data = []

        for match in matches:
            # Datos básicos del partido
            match_data = {
                "id": match.id,
                "league": {
                    "id": match.league.id if match.league else None,
                    "name": match.league.name if match.league else "Unknown",
                    "logo_url": (
                        match.league.logo_url if match.league else None
                    ),
                },
                "season": (
                    {
                        "id": match.season.id if match.season else None,
                        "name": match.season.name if match.season else None,
                        "year": (
                            f"{match.season.year_start}/{match.season.year_end}"
                            if match.season
                            else None
                        ),
                    }
                    if match.season
                    else None
                ),
                "home_team": {
                    "id": match.home_team.id if match.home_team else None,
                    "name": match.home_team.name if match.home_team else "TBD",
                    "short_name": (
                        match.home_team.short_name if match.home_team else None
                    ),
                    "logo_url": (
                        match.home_team.logo_url if match.home_team else None
                    ),
                },
                "away_team": {
                    "id": match.away_team.id if match.away_team else None,
                    "name": match.away_team.name if match.away_team else "TBD",
                    "short_name": (
                        match.away_team.short_name if match.away_team else None
                    ),
                    "logo_url": (
                        match.away_team.logo_url if match.away_team else None
                    ),
                },
                "match_date": match.match_date.isoformat(),
                "status": match.status,
                "round": {
                    "number": match.round_number,
                    "name": match.round_name,
                },
                "score": {
                    "home": match.home_score,
                    "away": match.away_score,
                    "half_time": (
                        {
                            "home": match.home_score_ht,
                            "away": match.away_score_ht,
                        }
                        if match.home_score_ht is not None
                        else None
                    ),
                },
                "venue": (
                    {
                        "id": match.stadium.id if match.stadium else None,
                        "name": match.stadium.name if match.stadium else None,
                        "city": match.stadium.city if match.stadium else None,
                        "capacity": (
                            match.stadium.capacity if match.stadium else None
                        ),
                    }
                    if match.stadium
                    else None
                ),
                "officials": {"referee": match.referee},
                "weather": (
                    {
                        "condition": match.weather_condition,
                        "temperature": match.temperature,
                        "humidity": match.humidity,
                    }
                    if match.weather_condition
                    else None
                ),
                "time": {
                    "minutes_played": match.minutes_played,
                    "added_time_first_half": match.added_time_first_half,
                    "added_time_second_half": match.added_time_second_half,
                },
                "api_id": match.api_id,
                "created_at": (
                    match.created_at.isoformat() if match.created_at else None
                ),
                "updated_at": (
                    match.updated_at.isoformat() if match.updated_at else None
                ),
            }

            # === DATOS OPCIONALES ===

            # Incluir estadísticas básicas
            if include_stats:
                match_data["basic_stats"] = await get_match_basic_stats(
                    match.id, db
                )

            # Incluir eventos
            if include_events:
                match_data["events"] = await get_match_events_summary(
                    match.id, db
                )

            # Incluir alineaciones
            if include_lineups:
                match_data["lineups"] = await get_match_lineups_summary(
                    match.id, db
                )

            # Incluir predicciones
            if include_predictions:
                match_data["predictions"] = (
                    await get_match_predictions_summary(match.id)
                )

            matches_data.append(match_data)

        # Metadatos de la respuesta
        metadata = {
            "total_matches": total_matches,
            "returned_matches": len(matches_data),
            "page_info": {
                "skip": skip,
                "limit": limit,
                "has_more": (skip + limit) < total_matches,
                "total_pages": (total_matches + limit - 1) // limit,
                "current_page": (skip // limit) + 1,
            },
            "filters_applied": {
                "status": status.value if status else None,
                "league_id": league_id,
                "season_id": season_id,
                "team_id": team_id,
                "date_range": {
                    "from": date_from.isoformat() if date_from else None,
                    "to": date_to.isoformat() if date_to else None,
                },
                "special_filters": {
                    "today": today,
                    "this_week": this_week,
                    "min_goals": min_goals,
                    "max_goals": max_goals,
                    "has_events": has_events,
                    "importance": importance,
                },
            },
            "sorting": sort_by.value,
            "includes": {
                "stats": include_stats,
                "events": include_events,
                "lineups": include_lineups,
                "predictions": include_predictions,
            },
        }

        return {
            "success": True,
            "data": matches_data,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"❌ Error obteniendo partidos: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error obteniendo partidos: {str(e)}"
        )


@router.get(
    "/{match_id}", response_model=Dict, summary="Obtener partido específico"
)
async def get_match(
    match_id: int,
    include_full_stats: bool = Query(
        False, description="Incluir estadísticas completas"
    ),
    include_timeline: bool = Query(
        False, description="Incluir timeline completo"
    ),
    include_lineups: bool = Query(
        False, description="Incluir alineaciones completas"
    ),
    include_head_to_head: bool = Query(
        False, description="Incluir historial cara a cara"
    ),
    include_predictions: bool = Query(
        False, description="Incluir análisis predictivo"
    ),
    include_betting_odds: bool = Query(
        False, description="Incluir cuotas de apuestas"
    ),
    db: Session = Depends(get_db),
):
    """
    Obtener información completa y detallada de un partido específico.

    **Datos incluidos:**
    - Información básica del partido
    - Equipos y estadios
    - Score y tiempo
    - Datos opcionales según parámetros

    **Datos opcionales:**
    - Estadísticas completas por equipo y jugador
    - Timeline de eventos minuto a minuto
    - Alineaciones y formaciones
    - Historial cara a cara entre equipos
    - Predicciones de ML y análisis
    - Cuotas de casas de apuestas
    """
    try:
        # Buscar partido con todas las relaciones
        match = (
            db.query(Match)
            .options(
                joinedload(Match.league),
                joinedload(Match.season),
                joinedload(Match.home_team),
                joinedload(Match.away_team),
                joinedload(Match.stadium),
            )
            .filter(Match.id == match_id)
            .first()
        )

        if not match:
            raise HTTPException(
                status_code=404,
                detail=f"Partido con ID {match_id} no encontrado",
            )

        # Construir respuesta base
        match_data = {
            "id": match.id,
            "basic_info": {
                "league": {
                    "id": match.league.id if match.league else None,
                    "name": match.league.name if match.league else "Unknown",
                    "country": (
                        match.league.country.name
                        if match.league and match.league.country
                        else None
                    ),
                    "logo_url": (
                        match.league.logo_url if match.league else None
                    ),
                },
                "season": (
                    {
                        "id": match.season.id if match.season else None,
                        "name": match.season.name if match.season else None,
                        "year_range": (
                            f"{match.season.year_start}/{match.season.year_end}"
                            if match.season
                            else None
                        ),
                        "is_current": (
                            match.season.is_current if match.season else False
                        ),
                    }
                    if match.season
                    else None
                ),
                "round": {
                    "number": match.round_number,
                    "name": match.round_name,
                },
                "match_date": match.match_date.isoformat(),
                "status": match.status,
                "api_id": match.api_id,
            },
            "teams": {
                "home": {
                    "id": match.home_team.id if match.home_team else None,
                    "name": match.home_team.name if match.home_team else "TBD",
                    "short_name": (
                        match.home_team.short_name if match.home_team else None
                    ),
                    "logo_url": (
                        match.home_team.logo_url if match.home_team else None
                    ),
                    "country": (
                        match.home_team.country.name
                        if match.home_team and match.home_team.country
                        else None
                    ),
                    "founded": (
                        match.home_team.founded_year
                        if match.home_team
                        else None
                    ),
                },
                "away": {
                    "id": match.away_team.id if match.away_team else None,
                    "name": match.away_team.name if match.away_team else "TBD",
                    "short_name": (
                        match.away_team.short_name if match.away_team else None
                    ),
                    "logo_url": (
                        match.away_team.logo_url if match.away_team else None
                    ),
                    "country": (
                        match.away_team.country.name
                        if match.away_team and match.away_team.country
                        else None
                    ),
                    "founded": (
                        match.away_team.founded_year
                        if match.away_team
                        else None
                    ),
                },
            },
            "venue": {
                "stadium": (
                    {
                        "id": match.stadium.id if match.stadium else None,
                        "name": match.stadium.name if match.stadium else None,
                        "city": match.stadium.city if match.stadium else None,
                        "capacity": (
                            match.stadium.capacity if match.stadium else None
                        ),
                        "surface": (
                            match.stadium.surface if match.stadium else None
                        ),
                    }
                    if match.stadium
                    else None
                ),
                "weather": (
                    {
                        "condition": match.weather_condition,
                        "temperature": match.temperature,
                        "humidity": match.humidity,
                    }
                    if match.weather_condition
                    else None
                ),
            },
            "score": {
                "current": {
                    "home": match.home_score,
                    "away": match.away_score,
                },
                "half_time": (
                    {"home": match.home_score_ht, "away": match.away_score_ht}
                    if match.home_score_ht is not None
                    else None
                ),
                "full_time": (
                    {"home": match.home_score, "away": match.away_score}
                    if match.status == "finished"
                    else None
                ),
                "extra_time": (
                    {"home": match.home_score_et, "away": match.away_score_et}
                    if match.home_score_et is not None
                    else None
                ),
                "penalties": (
                    {
                        "home": match.home_score_pen,
                        "away": match.away_score_pen,
                    }
                    if match.home_score_pen is not None
                    else None
                ),
            },
            "time": {
                "minutes_played": match.minutes_played,
                "added_time": {
                    "first_half": match.added_time_first_half,
                    "second_half": match.added_time_second_half,
                },
            },
            "officials": {"referee": match.referee},
        }

        # === DATOS OPCIONALES ===

        # Estadísticas completas
        if include_full_stats:
            match_data["statistics"] = await get_full_match_statistics(
                match_id, db
            )

        # Timeline de eventos
        if include_timeline:
            match_data["timeline"] = await get_match_timeline(match_id, db)

        # Alineaciones completas
        if include_lineups:
            match_data["lineups"] = await get_full_match_lineups(match_id, db)

        # Historial cara a cara
        if include_head_to_head:
            if match.home_team and match.away_team:
                match_data["head_to_head"] = await get_head_to_head_stats(
                    match.home_team_id, match.away_team_id, db
                )

        # Predicciones y análisis
        if include_predictions:
            match_data["predictions"] = await get_detailed_match_predictions(
                match_id
            )

        # Cuotas de apuestas
        if include_betting_odds:
            match_data["betting_odds"] = await get_match_betting_odds(
                match_id, db
            )

        # Metadatos de la respuesta
        match_data["metadata"] = {
            "last_updated": (
                match.updated_at.isoformat() if match.updated_at else None
            ),
            "data_completeness": {
                "basic_info": True,
                "full_stats": include_full_stats,
                "timeline": include_timeline,
                "lineups": include_lineups,
                "head_to_head": include_head_to_head,
                "predictions": include_predictions,
                "betting_odds": include_betting_odds,
            },
            "generated_at": datetime.now().isoformat(),
        }

        return {"success": True, "data": match_data}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error obteniendo partido {match_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error obteniendo partido: {str(e)}"
        )


@router.get("/live", summary="Obtener partidos en vivo")
async def get_live_matches(
    league_id: Optional[int] = Query(None, description="Filtrar por liga"),
    include_events: bool = Query(
        True, description="Incluir eventos recientes"
    ),
    include_stats: bool = Query(
        True, description="Incluir estadísticas en vivo"
    ),
    auto_refresh: bool = Query(False, description="Datos para auto-refresh"),
    db: Session = Depends(get_db),
):
    """
    Obtener todos los partidos que están siendo jugados actualmente.

    **Optimizado para:**
    - Actualizaciones en tiempo real
    - Datos mínimos para rendimiento
    - Auto-refresh de interfaces

    **Incluye:**
    - Estado actual del partido
    - Marcador en tiempo real
    - Eventos recientes
    - Estadísticas básicas en vivo
    """
    try:
        # Query optimizada para partidos en vivo
        query = (
            db.query(Match)
            .options(
                joinedload(Match.league),
                joinedload(Match.home_team),
                joinedload(Match.away_team),
            )
            .filter(Match.status == "live")
        )

        # Filtro por liga si se especifica
        if league_id:
            query = query.filter(Match.league_id == league_id)

        # Ordenar por importancia y fecha
        live_matches = query.order_by(desc(Match.match_date)).all()

        if not live_matches:
            return {
                "success": True,
                "data": {
                    "live_matches": [],
                    "total_live": 0,
                    "message": "No hay partidos en vivo en este momento",
                },
                "timestamp": datetime.now().isoformat(),
            }

        matches_data = []

        for match in live_matches:
            match_data = {
                "id": match.id,
                "league": {
                    "id": match.league.id if match.league else None,
                    "name": match.league.name if match.league else "Unknown",
                    "logo_url": (
                        match.league.logo_url if match.league else None
                    ),
                },
                "teams": {
                    "home": {
                        "id": match.home_team.id if match.home_team else None,
                        "name": (
                            match.home_team.name if match.home_team else "TBD"
                        ),
                        "short_name": (
                            match.home_team.short_name
                            if match.home_team
                            else None
                        ),
                        "logo_url": (
                            match.home_team.logo_url
                            if match.home_team
                            else None
                        ),
                    },
                    "away": {
                        "id": match.away_team.id if match.away_team else None,
                        "name": (
                            match.away_team.name if match.away_team else "TBD"
                        ),
                        "short_name": (
                            match.away_team.short_name
                            if match.away_team
                            else None
                        ),
                        "logo_url": (
                            match.away_team.logo_url
                            if match.away_team
                            else None
                        ),
                    },
                },
                "score": {"home": match.home_score, "away": match.away_score},
                "time": {
                    "minute": match.minutes_played,
                    "added_time": max(
                        match.added_time_first_half or 0,
                        match.added_time_second_half or 0,
                    ),
                    "display": f"{match.minutes_played}'"
                    + (
                        f"+{max(match.added_time_first_half or 0, match.added_time_second_half or 0)}"
                        if max(
                            match.added_time_first_half or 0,
                            match.added_time_second_half or 0,
                        )
                        > 0
                        else ""
                    ),
                },
                "status": match.status,
                "last_updated": (
                    match.updated_at.isoformat() if match.updated_at else None
                ),
            }

            # Incluir eventos recientes si se solicita
            if include_events:
                recent_events = (
                    db.query(MatchEvent)
                    .filter(MatchEvent.match_id == match.id)
                    .order_by(desc(MatchEvent.minute))
                    .limit(5)
                    .all()
                )

                match_data["recent_events"] = [
                    {
                        "type": event.event_type,
                        "minute": event.minute,
                        "team": event.team.short_name if event.team else None,
                        "player": event.player.name if event.player else None,
                        "description": event.description,
                    }
                    for event in recent_events
                ]

            # Incluir estadísticas básicas si se solicita
            if include_stats:
                match_data["live_stats"] = await get_live_match_stats(
                    match.id, db
                )

            matches_data.append(match_data)

        return {
            "success": True,
            "data": {
                "live_matches": matches_data,
                "total_live": len(matches_data),
                "filters": {"league_id": league_id},
                "refresh_interval": 30 if auto_refresh else None,  # segundos
                "websocket_available": f"/api/v1/live/ws",
            },
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"❌ Error obteniendo partidos en vivo: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo partidos en vivo: {str(e)}",
        )


@router.get("/upcoming", summary="Obtener próximos partidos")
async def get_upcoming_matches(
    limit: int = Query(20, ge=1, le=100, description="Límite de partidos"),
    hours_ahead: int = Query(
        24, ge=1, le=168, description="Horas hacia adelante"
    ),
    league_id: Optional[int] = Query(None, description="Filtrar por liga"),
    team_id: Optional[int] = Query(None, description="Filtrar por equipo"),
    include_predictions: bool = Query(
        False, description="Incluir predicciones"
    ),
    db: Session = Depends(get_db),
):
    """
    Obtener próximos partidos programados.

    **Filtros:**
    - Próximas X horas
    - Por liga o equipo específico
    - Límite de resultados

    **Datos opcionales:**
    - Predicciones de resultado
    - Análisis de forma reciente
    """
    try:
        # Calcular rango de fechas
        now = datetime.now()
        future_limit = now + timedelta(hours=hours_ahead)

        # Query para próximos partidos
        query = (
            db.query(Match)
            .options(
                joinedload(Match.league),
                joinedload(Match.home_team),
                joinedload(Match.away_team),
                joinedload(Match.stadium),
            )
            .filter(
                and_(
                    Match.status == "scheduled",
                    Match.match_date >= now,
                    Match.match_date <= future_limit,
                )
            )
        )

        # Aplicar filtros
        if league_id:
            query = query.filter(Match.league_id == league_id)

        if team_id:
            query = query.filter(
                or_(
                    Match.home_team_id == team_id,
                    Match.away_team_id == team_id,
                )
            )

        # Ordenar por fecha y limitar
        upcoming_matches = (
            query.order_by(asc(Match.match_date)).limit(limit).all()
        )

        matches_data = []

        for match in upcoming_matches:
            # Calcular tiempo hasta el partido
            time_until_match = match.match_date - now
            hours_until = int(time_until_match.total_seconds() / 3600)

            match_data = {
                "id": match.id,
                "league": {
                    "id": match.league.id if match.league else None,
                    "name": match.league.name if match.league else "Unknown",
                    "logo_url": (
                        match.league.logo_url if match.league else None
                    ),
                },
                "teams": {
                    "home": {
                        "id": match.home_team.id if match.home_team else None,
                        "name": (
                            match.home_team.name if match.home_team else "TBD"
                        ),
                        "logo_url": (
                            match.home_team.logo_url
                            if match.home_team
                            else None
                        ),
                    },
                    "away": {
                        "id": match.away_team.id if match.away_team else None,
                        "name": (
                            match.away_team.name if match.away_team else "TBD"
                        ),
                        "logo_url": (
                            match.away_team.logo_url
                            if match.away_team
                            else None
                        ),
                    },
                },
                "schedule": {
                    "match_date": match.match_date.isoformat(),
                    "hours_until_kickoff": hours_until,
                    "countdown": {
                        "days": time_until_match.days,
                        "hours": hours_until % 24,
                        "minutes": int((time_until_match.seconds % 3600) / 60),
                    },
                },
                "venue": (
                    {
                        "stadium": (
                            match.stadium.name if match.stadium else None
                        ),
                        "city": match.stadium.city if match.stadium else None,
                    }
                    if match.stadium
                    else None
                ),
                "round": {
                    "number": match.round_number,
                    "name": match.round_name,
                },
                "importance": await calculate_match_importance(match, db),
            }

            # Incluir predicciones si se solicita
            if include_predictions:
                match_data["predictions"] = (
                    await get_match_predictions_summary(match.id)
                )

            matches_data.append(match_data)

        return {
            "success": True,
            "data": {
                "upcoming_matches": matches_data,
                "total_found": len(matches_data),
                "search_criteria": {
                    "hours_ahead": hours_ahead,
                    "limit": limit,
                    "league_id": league_id,
                    "team_id": team_id,
                },
                "time_range": {
                    "from": now.isoformat(),
                    "to": future_limit.isoformat(),
                },
            },
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"❌ Error obteniendo próximos partidos: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo próximos partidos: {str(e)}",
        )


@router.get("/by-date/{match_date}", summary="Obtener partidos por fecha")
async def get_matches_by_date(
    match_date: date,
    league_id: Optional[int] = Query(None, description="Filtrar por liga"),
    status: Optional[MatchStatus] = Query(
        None, description="Filtrar por estado"
    ),
    include_stats: bool = Query(False, description="Incluir estadísticas"),
    db: Session = Depends(get_db),
):
    """
    Obtener todos los partidos de una fecha específica.

    **Formato de fecha:** YYYY-MM-DD

    **Útil para:**
    - Calendarios de partidos
    - Resúmenes diarios
    - Análisis por jornada
    """
    try:
        # Query para partidos de la fecha específica
        query = (
            db.query(Match)
            .options(
                joinedload(Match.league),
                joinedload(Match.home_team),
                joinedload(Match.away_team),
            )
            .filter(func.date(Match.match_date) == match_date)
        )

        # Aplicar filtros opcionales
        if league_id:
            query = query.filter(Match.league_id == league_id)

        if status:
            query = query.filter(Match.status == status.value)

        # Ordenar por hora del partido
        matches = query.order_by(asc(Match.match_date)).all()

        matches_data = []

        for match in matches:
            match_data = {
                "id": match.id,
                "league": {
                    "id": match.league.id if match.league else None,
                    "name": match.league.name if match.league else "Unknown",
                },
                "teams": {
                    "home": {
                        "id": match.home_team.id if match.home_team else None,
                        "name": (
                            match.home_team.name if match.home_team else "TBD"
                        ),
                    },
                    "away": {
                        "id": match.away_team.id if match.away_team else None,
                        "name": (
                            match.away_team.name if match.away_team else "TBD"
                        ),
                    },
                },
                "kickoff_time": match.match_date.strftime("%H:%M"),
                "status": match.status,
                "score": (
                    {"home": match.home_score, "away": match.away_score}
                    if match.status in ["live", "finished"]
                    else None
                ),
            }

            # Incluir estadísticas si se solicita
            if include_stats and match.status == "finished":
                match_data["basic_stats"] = await get_match_basic_stats(
                    match.id, db
                )

            matches_data.append(match_data)

        # Agrupar por liga para mejor presentación
        matches_by_league = {}
        for match in matches_data:
            league_name = match["league"]["name"]
            if league_name not in matches_by_league:
                matches_by_league[league_name] = []
            matches_by_league[league_name].append(match)

        return {
            "success": True,
            "data": {
                "date": match_date.isoformat(),
                "total_matches": len(matches_data),
                "matches_by_league": matches_by_league,
                "all_matches": matches_data,
                "filters_applied": {
                    "league_id": league_id,
                    "status": status.value if status else None,
                },
            },
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"❌ Error obteniendo partidos por fecha: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo partidos por fecha: {str(e)}",
        )


@router.get("/{match_id}/events", summary="Obtener eventos del partido")
async def get_match_events(
    match_id: int,
    event_type: Optional[str] = Query(
        None, description="Filtrar por tipo de evento"
    ),
    team_id: Optional[int] = Query(None, description="Filtrar por equipo"),
    db: Session = Depends(get_db),
):
    """
    Obtener todos los eventos de un partido específico.

    **Eventos incluidos:**
    - Goles
    - Tarjetas (amarillas y rojas)
    - Sustituciones
    - Penales
    - Otros eventos importantes
    """
    try:
        # Verificar que el partido existe
        match = db.query(Match).filter(Match.id == match_id).first()
        if not match:
            raise HTTPException(
                status_code=404, detail="Partido no encontrado"
            )

        # Query para eventos
        query = (
            db.query(MatchEvent)
            .options(
                joinedload(MatchEvent.team), joinedload(MatchEvent.player)
            )
            .filter(MatchEvent.match_id == match_id)
        )

        # Aplicar filtros
        if event_type:
            query = query.filter(MatchEvent.event_type == event_type)

        if team_id:
            query = query.filter(MatchEvent.team_id == team_id)

        # Ordenar por minuto
        events = query.order_by(asc(MatchEvent.minute)).all()

        events_data = []

        for event in events:
            event_data = {
                "id": event.id,
                "type": event.event_type,
                "detail": event.event_detail,
                "minute": event.minute,
                "added_time": event.added_time,
                "display_time": f"{event.minute}'"
                + (
                    f"+{event.added_time}"
                    if event.added_time and event.added_time > 0
                    else ""
                ),
                "team": {
                    "id": event.team.id if event.team else None,
                    "name": event.team.name if event.team else None,
                },
                "player": {
                    "id": event.player.id if event.player else None,
                    "name": event.player.name if event.player else None,
                },
                "description": event.description,
            }

            events_data.append(event_data)

        # Estadísticas de eventos
        event_stats = {
            "total_events": len(events_data),
            "by_type": {},
            "by_team": {},
            "by_half": {
                "first_half": len([e for e in events if e.minute <= 45]),
                "second_half": len([e for e in events if e.minute > 45]),
            },
        }

        # Contar por tipo
        for event in events:
            event_type = event.event_type
            if event_type not in event_stats["by_type"]:
                event_stats["by_type"][event_type] = 0
            event_stats["by_type"][event_type] += 1

        # Contar por equipo
        for event in events:
            if event.team:
                team_name = event.team.name
                if team_name not in event_stats["by_team"]:
                    event_stats["by_team"][team_name] = 0
                event_stats["by_team"][team_name] += 1

        return {
            "success": True,
            "data": {
                "match_id": match_id,
                "events": events_data,
                "statistics": event_stats,
                "filters_applied": {
                    "event_type": event_type,
                    "team_id": team_id,
                },
            },
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error obteniendo eventos del partido: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error obteniendo eventos: {str(e)}"
        )


@router.get("/{match_id}/stats", summary="Obtener estadísticas del partido")
async def get_match_stats(
    match_id: int,
    include_player_stats: bool = Query(
        False, description="Incluir estadísticas de jugadores"
    ),
    include_advanced_stats: bool = Query(
        False, description="Incluir estadísticas avanzadas"
    ),
    db: Session = Depends(get_db),
):
    """
    Obtener estadísticas completas de un partido.

    **Estadísticas incluidas:**
    - Estadísticas por equipo
    - Estadísticas por jugador (opcional)
    - Métricas avanzadas (opcional)
    """
    try:
        # Verificar que el partido existe
        match = db.query(Match).filter(Match.id == match_id).first()
        if not match:
            raise HTTPException(
                status_code=404, detail="Partido no encontrado"
            )

        # Obtener estadísticas por equipo
        team_stats = (
            db.query(MatchTeamStats)
            .options(joinedload(MatchTeamStats.team))
            .filter(MatchTeamStats.match_id == match_id)
            .all()
        )

        team_stats_data = {}
        for stat in team_stats:
            team_name = stat.team.name if stat.team else "Unknown"
            team_stats_data[team_name] = {
                "possession_percentage": stat.possession_percentage,
                "shots": {
                    "total": stat.shots_total,
                    "on_target": stat.shots_on_target,
                    "off_target": stat.shots_off_target,
                    "blocked": stat.shots_blocked,
                },
                "passes": {
                    "total": stat.passes_total,
                    "accurate": stat.passes_accurate,
                    "accuracy_percentage": stat.passes_accuracy_percentage,
                },
                "attacking": {
                    "corners": stat.corners,
                    "offsides": stat.offsides,
                },
                "defending": {
                    "tackles_total": stat.tackles_total,
                    "tackles_successful": stat.tackles_successful,
                    "clearances": stat.clearances,
                    "interceptions": stat.interceptions,
                },
                "discipline": {
                    "fouls_committed": stat.fouls_committed,
                    "fouls_received": stat.fouls_received,
                    "yellow_cards": stat.yellow_cards,
                    "red_cards": stat.red_cards,
                },
                "physical": {
                    "distance_covered": stat.distance_covered,
                    "sprints": stat.sprints,
                },
            }

        response_data = {
            "match_id": match_id,
            "team_statistics": team_stats_data,
        }

        # Incluir estadísticas de jugadores si se solicita
        if include_player_stats:
            player_stats = (
                db.query(MatchPlayerStats)
                .options(
                    joinedload(MatchPlayerStats.player),
                    joinedload(MatchPlayerStats.team),
                )
                .filter(MatchPlayerStats.match_id == match_id)
                .order_by(MatchPlayerStats.rating.desc())
                .all()
            )

            player_stats_data = []
            for stat in player_stats:
                player_data = {
                    "player": {
                        "id": stat.player.id if stat.player else None,
                        "name": stat.player.name if stat.player else "Unknown",
                        "position": (
                            stat.player.position if stat.player else None
                        ),
                    },
                    "team": {
                        "id": stat.team.id if stat.team else None,
                        "name": stat.team.name if stat.team else "Unknown",
                    },
                    "minutes_played": stat.minutes_played,
                    "performance": {
                        "rating": stat.rating,
                        "goals": stat.goals,
                        "assists": stat.assists,
                    },
                    "passing": {
                        "total": stat.passes_total,
                        "accurate": stat.passes_accurate,
                        "key_passes": stat.key_passes,
                    },
                    "attacking": {
                        "shots_total": stat.shots_total,
                        "shots_on_target": stat.shots_on_target,
                        "dribbles_attempted": stat.dribbles_attempted,
                        "dribbles_successful": stat.dribbles_successful,
                    },
                    "defending": {
                        "tackles": stat.tackles,
                        "interceptions": stat.interceptions,
                        "clearances": stat.clearances,
                    },
                    "discipline": {
                        "fouls_committed": stat.fouls_committed,
                        "fouls_received": stat.fouls_received,
                        "yellow_cards": stat.yellow_cards,
                        "red_cards": stat.red_cards,
                    },
                    "physical": {
                        "distance_covered": stat.distance_covered,
                        "sprints": stat.sprints,
                    },
                }
                player_stats_data.append(player_data)

            response_data["player_statistics"] = player_stats_data

        # Incluir estadísticas avanzadas si se solicita
        if include_advanced_stats:
            response_data["advanced_statistics"] = (
                await calculate_advanced_match_stats(match_id, db)
            )

        return {
            "success": True,
            "data": response_data,
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error obteniendo estadísticas del partido: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error obteniendo estadísticas: {str(e)}"
        )


# =====================================================
# FUNCIONES AUXILIARES
# =====================================================


async def get_match_basic_stats(match_id: int, db: Session) -> Dict:
    """Obtener estadísticas básicas de un partido."""
    try:
        # Contar eventos básicos
        events = (
            db.query(MatchEvent).filter(MatchEvent.match_id == match_id).all()
        )

        stats = {
            "total_events": len(events),
            "goals": len([e for e in events if e.event_type == "goal"]),
            "yellow_cards": len(
                [e for e in events if e.event_type == "yellow_card"]
            ),
            "red_cards": len(
                [e for e in events if e.event_type == "red_card"]
            ),
            "substitutions": len(
                [e for e in events if e.event_type == "substitution"]
            ),
        }

        return stats

    except Exception as e:
        logger.error(f"❌ Error calculando estadísticas básicas: {e}")
        return {}


async def get_match_events_summary(match_id: int, db: Session) -> List[Dict]:
    """Obtener resumen de eventos del partido."""
    try:
        events = (
            db.query(MatchEvent)
            .filter(MatchEvent.match_id == match_id)
            .order_by(desc(MatchEvent.minute))
            .limit(5)
            .all()
        )

        return [
            {
                "type": event.event_type,
                "minute": event.minute,
                "description": event.description,
            }
            for event in events
        ]

    except Exception as e:
        logger.error(f"❌ Error obteniendo resumen de eventos: {e}")
        return []


async def get_match_lineups_summary(match_id: int, db: Session) -> Dict:
    """Obtener resumen de alineaciones."""
    try:
        # Placeholder - implementar con modelo Lineup real
        return {
            "home_formation": "4-3-3",
            "away_formation": "4-4-2",
            "total_players": 22,
        }

    except Exception as e:
        logger.error(f"❌ Error obteniendo alineaciones: {e}")
        return {}


async def get_match_predictions_summary(match_id: int) -> Dict:
    """Obtener resumen de predicciones para un partido."""
    try:
        # Usar el motor de predicciones
        predictions = prediction_engine.predict_match_outcome(
            1, 2
        )  # Placeholder

        return {
            "home_win_probability": predictions.get(
                "home_win_probability", 0.33
            ),
            "draw_probability": predictions.get("draw_probability", 0.33),
            "away_win_probability": predictions.get(
                "away_win_probability", 0.33
            ),
            "confidence": predictions.get("confidence", 0.7),
        }

    except Exception as e:
        logger.error(f"❌ Error obteniendo predicciones: {e}")
        return {}


async def calculate_match_importance(match: Match, db: Session) -> str:
    """Calcular la importancia de un partido."""
    try:
        importance_score = 0

        # Factores que aumentan la importancia
        if match.round_name and any(
            keyword in match.round_name.lower()
            for keyword in ["final", "semi", "derby", "clasico"]
        ):
            importance_score += 3

        # Liga principal
        if match.league and match.league.name in [
            "Premier League",
            "La Liga",
            "Serie A",
            "Bundesliga",
        ]:
            importance_score += 2

        # Determinar nivel
        if importance_score >= 4:
            return "high"
        elif importance_score >= 2:
            return "medium"
        else:
            return "low"

    except Exception as e:
        logger.error(f"❌ Error calculando importancia: {e}")
        return "low"


async def get_live_match_stats(match_id: int, db: Session) -> Dict:
    """Obtener estadísticas básicas para partido en vivo."""
    try:
        # Simular estadísticas en vivo (implementar con datos reales)
        return {
            "possession": {"home": 55, "away": 45},
            "shots": {"home": 8, "away": 4},
            "corners": {"home": 3, "away": 1},
        }

    except Exception as e:
        logger.error(f"❌ Error obteniendo stats en vivo: {e}")
        return {}


async def get_full_match_statistics(match_id: int, db: Session) -> Dict:
    """Obtener estadísticas completas del partido."""
    # Implementar lógica completa
    return await get_match_basic_stats(match_id, db)


async def get_match_timeline(match_id: int, db: Session) -> List[Dict]:
    """Obtener timeline completo del partido."""
    return await get_match_events_summary(match_id, db)


async def get_full_match_lineups(match_id: int, db: Session) -> Dict:
    """Obtener alineaciones completas."""
    return await get_match_lineups_summary(match_id, db)


async def get_head_to_head_stats(
    home_team_id: int, away_team_id: int, db: Session
) -> Dict:
    """Obtener estadísticas cara a cara entre equipos."""
    try:
        # Obtener últimos 10 partidos entre estos equipos
        h2h_matches = (
            db.query(Match)
            .filter(
                or_(
                    and_(
                        Match.home_team_id == home_team_id,
                        Match.away_team_id == away_team_id,
                    ),
                    and_(
                        Match.home_team_id == away_team_id,
                        Match.away_team_id == home_team_id,
                    ),
                ),
                Match.status == "finished",
            )
            .order_by(desc(Match.match_date))
            .limit(10)
            .all()
        )

        if not h2h_matches:
            return {"message": "No hay historial entre estos equipos"}

        # Calcular estadísticas
        home_wins = 0
        away_wins = 0
        draws = 0
        total_goals = 0

        for match in h2h_matches:
            total_goals += match.home_score + match.away_score

            if match.home_score > match.away_score:
                if match.home_team_id == home_team_id:
                    home_wins += 1
                else:
                    away_wins += 1
            elif match.home_score < match.away_score:
                if match.away_team_id == home_team_id:
                    home_wins += 1
                else:
                    away_wins += 1
            else:
                draws += 1

        return {
            "total_matches": len(h2h_matches),
            "home_wins": home_wins,
            "away_wins": away_wins,
            "draws": draws,
            "average_goals": round(total_goals / len(h2h_matches), 2),
            "recent_matches": [
                {
                    "date": match.match_date.date().isoformat(),
                    "home_team": (
                        match.home_team.name if match.home_team else "Unknown"
                    ),
                    "away_team": (
                        match.away_team.name if match.away_team else "Unknown"
                    ),
                    "score": f"{match.home_score}-{match.away_score}",
                }
                for match in h2h_matches[:5]
            ],
        }

    except Exception as e:
        logger.error(f"❌ Error obteniendo historial cara a cara: {e}")
        return {}


async def get_detailed_match_predictions(match_id: int) -> Dict:
    """Obtener predicciones detalladas."""
    return await get_match_predictions_summary(match_id)


async def get_match_betting_odds(match_id: int, db: Session) -> Dict:
    """Obtener cuotas de apuestas."""
    try:
        # Placeholder - implementar con modelo Odds real
        return {
            "home_win": 2.10,
            "draw": 3.20,
            "away_win": 3.50,
            "over_2_5": 1.85,
            "under_2_5": 1.95,
            "btts_yes": 1.75,
            "btts_no": 2.05,
        }

    except Exception as e:
        logger.error(f"❌ Error obteniendo cuotas: {e}")
        return {}


async def calculate_advanced_match_stats(match_id: int, db: Session) -> Dict:
    """Calcular estadísticas avanzadas."""
    try:
        return {
            "expected_goals": {"home": 1.8, "away": 1.2},
            "big_chances": {"home": 3, "away": 2},
            "heat_map": "available",
            "pass_network": "available",
        }

    except Exception as e:
        logger.error(f"❌ Error calculando stats avanzadas: {e}")
        return {}
