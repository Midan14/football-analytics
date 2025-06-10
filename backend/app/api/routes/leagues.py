"""
API Routes para gestión de ligas de fútbol.
Incluye operaciones CRUD, clasificaciones, estadísticas y análisis.
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

# Imports internos
from ...database.connection import get_db
from ...database.models import Country, League, Match, Season, Team

# Crear router
router = APIRouter()

# =====================================================
# SCHEMAS/MODELS PYDANTIC (respuestas de la API)
# =====================================================


class LeagueResponse(BaseModel):
    id: int
    name: str
    country_name: Optional[str] = None
    logo_url: Optional[str] = None
    type: str
    is_active: bool
    season_start_month: int
    season_end_month: int
    total_teams: Optional[int] = 0

    class Config:
        from_attributes = True


class LeagueStats(BaseModel):
    total_matches: int
    completed_matches: int
    live_matches: int
    scheduled_matches: int
    total_goals: int
    average_goals_per_match: float
    top_scorer_team: Optional[str] = None


class StandingsEntry(BaseModel):
    position: int
    team_name: str
    team_logo: Optional[str] = None
    matches_played: int
    wins: int
    draws: int
    losses: int
    goals_for: int
    goals_against: int
    goal_difference: int
    points: int
    form: Optional[str] = None  # Últimos 5 partidos: "WWDLW"


# =====================================================
# ENDPOINTS PRINCIPALES
# =====================================================


@router.get("/", response_model=dict, summary="Obtener todas las ligas")
async def get_leagues(
    skip: int = Query(0, ge=0, description="Número de registros a saltar"),
    limit: int = Query(100, ge=1, le=1000, description="Límite de registros"),
    country_id: Optional[int] = Query(None, description="Filtrar por país"),
    is_active: Optional[bool] = Query(None, description="Filtrar por estado activo"),
    search: Optional[str] = Query(None, description="Buscar por nombre"),
    db: Session = Depends(get_db),
):
    """
    Obtener lista de todas las ligas de fútbol con filtros opcionales.

    **Filtros disponibles:**
    - `country_id`: Filtrar por ID de país
    - `is_active`: Solo ligas activas (true) o inactivas (false)
    - `search`: Buscar por nombre de liga
    - `skip` y `limit`: Paginación

    **Retorna:**
    - Lista de ligas con información básica
    - Total de registros
    - Metadatos de paginación
    """
    try:
        # Construir query base
        query = db.query(League).join(Country)

        # Aplicar filtros
        if country_id:
            query = query.filter(League.country_id == country_id)

        if is_active is not None:
            query = query.filter(League.is_active == is_active)

        if search:
            query = query.filter(League.name.ilike(f"%{search}%"))

        # Obtener total y aplicar paginación
        total = query.count()
        leagues = query.offset(skip).limit(limit).all()

        # Enriquecer datos
        leagues_data = []
        for league in leagues:
            # Contar equipos en la liga
            teams_count = db.query(Team).filter(Team.league_id == league.id).count()

            league_data = {
                "id": league.id,
                "name": league.name,
                "country_name": (league.country.name if league.country else None),
                "logo_url": league.logo_url,
                "type": league.type,
                "is_active": league.is_active,
                "season_start_month": league.season_start_month,
                "season_end_month": league.season_end_month,
                "total_teams": teams_count,
            }
            leagues_data.append(league_data)

        return {
            "success": True,
            "data": leagues_data,
            "pagination": {
                "total": total,
                "skip": skip,
                "limit": limit,
                "pages": (total + limit - 1) // limit,
            },
            "filters_applied": {
                "country_id": country_id,
                "is_active": is_active,
                "search": search,
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error obteniendo ligas: {str(e)}"
        ) from e


@router.get("/{league_id}", response_model=dict, summary="Obtener liga por ID")
async def get_league(
    league_id: int,
    include_stats: bool = Query(False, description="Incluir estadísticas de la liga"),
    db: Session = Depends(get_db),
):
    """
    Obtener información detallada de una liga específica.

    **Parámetros:**
    - `league_id`: ID único de la liga
    - `include_stats`: Si incluir estadísticas detalladas

    **Retorna:**
    - Información completa de la liga
    - Estadísticas opcionales
    - Lista de equipos participantes
    """
    try:
        # Buscar liga con relaciones
        league = db.query(League).join(Country).filter(League.id == league_id).first()

        if not league:
            raise HTTPException(
                status_code=404,
                detail=f"Liga con ID {league_id} no encontrada",
            )

        # Obtener equipos de la liga
        teams = db.query(Team).filter(Team.league_id == league_id).all()
        teams_data = [
            {
                "id": team.id,
                "name": team.name,
                "short_name": team.short_name,
                "logo_url": team.logo_url,
                "founded_year": team.founded_year,
            }
            for team in teams
        ]

        league_data = {
            "id": league.id,
            "name": league.name,
            "country": {
                "id": league.country.id,
                "name": league.country.name,
                "code": league.country.code,
                "flag_url": league.country.flag_url,
            },
            "logo_url": league.logo_url,
            "type": league.type,
            "is_active": league.is_active,
            "season_start_month": league.season_start_month,
            "season_end_month": league.season_end_month,
            "api_id": league.api_id,
            "teams": teams_data,
            "total_teams": len(teams_data),
            "created_at": league.created_at,
            "updated_at": league.updated_at,
        }

        # Incluir estadísticas si se solicita
        if include_stats:
            stats = await get_league_statistics(league_id, db)
            league_data["statistics"] = stats

        return {"success": True, "data": league_data}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error obteniendo liga: {str(e)}"
        ) from e


@router.get(
    "/{league_id}/standings",
    response_model=dict,
    summary="Obtener clasificación de liga",
)
async def get_league_standings(
    league_id: int,
    season_id: Optional[int] = Query(None, description="ID de temporada específica"),
    db: Session = Depends(get_db),
):
    """
    Obtener tabla de posiciones/clasificación de una liga.

    **Parámetros:**
    - `league_id`: ID de la liga
    - `season_id`: ID de temporada (opcional, usa actual si no se especifica)

    **Retorna:**
    - Tabla de posiciones completa
    - Información de temporada
    - Estadísticas por equipo
    """
    try:
        # Verificar que la liga existe
        league = db.query(League).filter(League.id == league_id).first()
        if not league:
            raise HTTPException(
                status_code=404,
                detail=f"Liga con ID {league_id} no encontrada",
            )

        # Obtener temporada (actual si no se especifica)
        if season_id:
            season = (
                db.query(Season)
                .filter(Season.id == season_id, Season.league_id == league_id)
                .first()
            )
        else:
            season = (
                db.query(Season)
                .filter(Season.league_id == league_id, Season.is_current.is_(True))
                .first()
            )

        if not season:
            # Si no hay temporada actual, tomar la más reciente
            season = (
                db.query(Season)
                .filter(Season.league_id == league_id)
                .order_by(Season.year_start.desc())
                .first()
            )

        if not season:
            return {
                "success": True,
                "data": {
                    "league_name": league.name,
                    "season": None,
                    "standings": [],
                    "message": "No hay temporadas disponibles para esta liga",
                },
            }

        # Calcular clasificación basada en partidos
        teams = db.query(Team).filter(Team.league_id == league_id).all()
        standings = []

        for team in teams:
            # Obtener partidos del equipo en la temporada
            home_matches = (
                db.query(Match)
                .filter(
                    Match.home_team_id == team.id,
                    Match.league_id == league_id,
                    Match.status == "finished",
                )
                .all()
            )

            away_matches = (
                db.query(Match)
                .filter(
                    Match.away_team_id == team.id,
                    Match.league_id == league_id,
                    Match.status == "finished",
                )
                .all()
            )

            # Calcular estadísticas
            wins = draws = losses = 0
            goals_for = goals_against = 0

            # Partidos como local
            for match in home_matches:
                # Extraer valores de columnas SQLAlchemy de forma segura
                home_score_val = getattr(match, "home_score", None)
                away_score_val = getattr(match, "away_score", None)

                if home_score_val is not None and away_score_val is not None:
                    goals_for += home_score_val
                    goals_against += away_score_val

                    if home_score_val > away_score_val:
                        wins += 1
                    elif home_score_val == away_score_val:
                        draws += 1
                    else:
                        losses += 1

            # Partidos como visitante
            for match in away_matches:
                # Extraer valores de columnas SQLAlchemy de forma segura
                away_score_val = getattr(match, "away_score", None)
                home_score_val = getattr(match, "home_score", None)

                if away_score_val is not None and home_score_val is not None:
                    goals_for += away_score_val
                    goals_against += home_score_val

                    if away_score_val > home_score_val:
                        wins += 1
                    elif away_score_val == home_score_val:
                        draws += 1
                    else:
                        losses += 1

            matches_played = wins + draws + losses
            points = wins * 3 + draws
            goal_difference = goals_for - goals_against

            # Calcular forma reciente (últimos 5 partidos)
            recent_matches = (home_matches + away_matches)[-5:]
            form = ""
            for match in recent_matches:
                home_score_val = getattr(match, "home_score", None)
                away_score_val = getattr(match, "away_score", None)
                team_id_val = getattr(team, "id", None)
                home_team_id_val = getattr(match, "home_team_id", None)

                if (
                    home_score_val is not None
                    and away_score_val is not None
                    and team_id_val is not None
                    and home_team_id_val is not None
                ):

                    if team_id_val == home_team_id_val:
                        if home_score_val > away_score_val:
                            form += "W"
                        elif home_score_val == away_score_val:
                            form += "D"
                        else:
                            form += "L"
                    else:
                        if away_score_val > home_score_val:
                            form += "W"
                        elif away_score_val == home_score_val:
                            form += "D"
                        else:
                            form += "L"

            standings.append(
                {
                    "team_id": team.id,
                    "team_name": team.name,
                    "team_logo": team.logo_url,
                    "matches_played": matches_played,
                    "wins": wins,
                    "draws": draws,
                    "losses": losses,
                    "goals_for": goals_for,
                    "goals_against": goals_against,
                    "goal_difference": goal_difference,
                    "points": points,
                    "form": form[-5:] if form else "",  # Últimos 5
                }
            )

        # Ordenar por puntos, diferencia de goles y goles a favor
        standings.sort(
            key=lambda x: (
                -x["points"],
                -x["goal_difference"],
                -x["goals_for"],
            )
        )

        # Agregar posiciones
        for i, team in enumerate(standings, 1):
            team["position"] = i

        return {
            "success": True,
            "data": {
                "league": {"id": league.id, "name": league.name},
                "season": {
                    "id": season.id,
                    "name": season.name,
                    "year_start": season.year_start,
                    "year_end": season.year_end,
                    "is_current": season.is_current,
                },
                "standings": standings,
                "last_updated": datetime.now().isoformat(),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error obteniendo clasificación: {str(e)}"
        ) from e


@router.get(
    "/{league_id}/stats",
    response_model=dict,
    summary="Obtener estadísticas de liga",
)
async def get_league_stats(
    league_id: int,
    season_id: Optional[int] = Query(None, description="ID de temporada"),
    db: Session = Depends(get_db),
):
    """
    Obtener estadísticas detalladas de una liga.

    **Incluye:**
    - Estadísticas generales de partidos
    - Promedios de goles
    - Equipos más exitosos
    - Tendencias de la temporada
    """
    try:
        league = db.query(League).filter(League.id == league_id).first()
        if not league:
            raise HTTPException(status_code=404, detail="Liga no encontrada")

        stats = await get_league_statistics(league_id, db, season_id)

        return {
            "success": True,
            "data": {
                "league_name": league.name,
                "statistics": stats,
                "generated_at": datetime.now().isoformat(),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error obteniendo estadísticas: {str(e)}"
        ) from e


@router.post("/{league_id}/sync", summary="Sincronizar datos de liga")
async def sync_league_data(
    league_id: int,
    force_update: bool = Query(False, description="Forzar actualización completa"),
    db: Session = Depends(get_db),
):
    """
    Sincronizar datos de la liga desde fuentes externas.

    **Funcionalidad:**
    - Actualiza equipos y jugadores
    - Sincroniza partidos recientes
    - Actualiza estadísticas
    """
    try:
        league = db.query(League).filter(League.id == league_id).first()
        if not league:
            raise HTTPException(status_code=404, detail="Liga no encontrada")

        # Simular sincronización (implementar con API real)
        result = {
            "league_id": league_id,
            "teams_updated": 0,
            "matches_updated": 0,
            "players_updated": 0,
            "last_sync": datetime.now().isoformat(),
            "status": "completed",
        }

        if force_update:
            result["message"] = "Actualización completa realizada"
        else:
            result["message"] = "Actualización incremental realizada"

        return {"success": True, "data": result}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error sincronizando liga: {str(e)}"
        ) from e


# =====================================================
# FUNCIONES AUXILIARES
# =====================================================


async def get_league_statistics(
    league_id: int, db: Session, season_id: Optional[int] = None
) -> dict:
    """
    Calcular estadísticas detalladas de una liga.
    """
    try:
        # Query base para partidos
        matches_query = db.query(Match).filter(Match.league_id == league_id)

        if season_id:
            matches_query = matches_query.filter(Match.season_id == season_id)

        all_matches = matches_query.all()

        # Filtrar partidos por estado de forma segura
        completed_matches = []
        live_matches = []
        scheduled_matches = []

        for m in all_matches:
            status_val = getattr(m, "status", None)
            if status_val == "finished":
                completed_matches.append(m)
            elif status_val == "live":
                live_matches.append(m)
            elif status_val == "scheduled":
                scheduled_matches.append(m)

        # Calcular estadísticas de goles de forma segura
        total_goals = 0
        for m in completed_matches:
            home_score_val = getattr(m, "home_score", 0) or 0
            away_score_val = getattr(m, "away_score", 0) or 0
            total_goals += home_score_val + away_score_val

        avg_goals = total_goals / len(completed_matches) if completed_matches else 0

        # Equipo con más goles
        team_goals = {}
        for match in completed_matches:
            home_score_val = getattr(match, "home_score", None)
            away_score_val = getattr(match, "away_score", None)

            if home_score_val is not None and away_score_val is not None:
                home_team = getattr(match, "home_team_id", None)
                away_team = getattr(match, "away_team_id", None)

                if home_team:
                    team_goals[home_team] = (
                        team_goals.get(home_team, 0) + home_score_val
                    )
                if away_team:
                    team_goals[away_team] = (
                        team_goals.get(away_team, 0) + away_score_val
                    )

        top_scorer_team_id = None
        if team_goals:
            top_scorer_team_id = max(team_goals.keys(), key=lambda k: team_goals[k])

        top_scorer_team = None
        if top_scorer_team_id:
            team = db.query(Team).filter(Team.id == top_scorer_team_id).first()
            top_scorer_team = team.name if team else None

        return {
            "total_matches": len(all_matches),
            "completed_matches": len(completed_matches),
            "live_matches": len(live_matches),
            "scheduled_matches": len(scheduled_matches),
            "total_goals": total_goals,
            "average_goals_per_match": round(avg_goals, 2),
            "top_scorer_team": top_scorer_team,
            "completion_percentage": round(
                (
                    (len(completed_matches) / len(all_matches) * 100)
                    if all_matches
                    else 0
                ),
                2,
            ),
        }

    except Exception as e:
        return {"error": f"Error calculando estadísticas: {str(e)}"}
