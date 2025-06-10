"""
API Routes para gestión de equipos de fútbol
Incluye CRUD, estadísticas, análisis de rendimiento, plantillas y comparaciones
"""

import json
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel
from sqlalchemy import and_, asc, desc, func, or_
from sqlalchemy.orm import Session, joinedload

from app.database.connection import get_db
from app.database.models import League, Match, Player, Team, TeamStats, TeamTransfer
from app.services.form_calculator import FormCalculator
from app.services.team_analyzer import TeamAnalyzer

# from app.core.redis_client import redis_client  # TODO: implement redis client

router = APIRouter()

# =====================================================
# MODELOS PYDANTIC (SCHEMAS)
# =====================================================


class TeamResponse(BaseModel):
    id: int
    name: str
    short_name: Optional[str] = None
    founded_year: Optional[int] = None
    logo_url: Optional[str] = None
    stadium_name: Optional[str] = None
    stadium_capacity: Optional[int] = None
    league_name: Optional[str] = None
    league_id: Optional[int] = None
    country: Optional[str] = None
    coach_name: Optional[str] = None
    market_value: Optional[float] = None
    squad_size: Optional[int] = None


class TeamStatsResponse(BaseModel):
    team_id: int
    season: str
    matches_played: int
    wins: int
    draws: int
    losses: int
    goals_for: int
    goals_against: int
    goal_difference: int
    points: int
    position: Optional[int] = None
    home_wins: int
    home_draws: int
    home_losses: int
    away_wins: int
    away_draws: int
    away_losses: int


class TeamFormResponse(BaseModel):
    team_id: int
    last_5_results: List[str]
    last_5_points: int
    form_rating: float
    recent_goals_for: int
    recent_goals_against: int
    trend: str  # "improving", "stable", "declining"


# =====================================================
# ENDPOINTS PRINCIPALES
# =====================================================


@router.get("/", response_model=dict, summary="Obtener lista de equipos")
async def get_teams(
    # Filtros básicos
    league_id: Optional[int] = Query(None, description="ID de la liga"),
    country: Optional[str] = Query(None, description="País"),
    # Filtros de rendimiento
    min_market_value: Optional[float] = Query(
        None, ge=0, description="Valor mínimo de mercado"
    ),
    max_market_value: Optional[float] = Query(
        None, ge=0, description="Valor máximo de mercado"
    ),
    # Búsqueda
    search: Optional[str] = Query(None, min_length=2, description="Buscar por nombre"),
    # Paginación
    skip: int = Query(0, ge=0, description="Número de registros a saltar"),
    limit: int = Query(50, ge=1, le=200, description="Límite de registros"),
    # Ordenamiento
    sort_by: str = Query("name", description="Campo para ordenar"),
    sort_order: str = Query("asc", regex="^(asc|desc)$", description="Orden"),
    # Datos adicionales
    include_stats: bool = Query(False, description="Incluir estadísticas"),
    include_players_count: bool = Query(
        True, description="Incluir número de jugadores"
    ),
    include_form: bool = Query(False, description="Incluir forma reciente"),
    db: Session = Depends(get_db),
):
    """
    Obtener lista de equipos con filtros avanzados, búsqueda y paginación.

    Características:
    - Filtros por liga, país, valor de mercado
    - Búsqueda por nombre con coincidencias parciales
    - Ordenamiento por múltiples campos
    - Datos opcionales (estadísticas, forma reciente)
    - Cache Redis para mejor performance
    """
    try:
        # Crear clave de cache
        cache_key = f"teams:list:{hash(str({'league_id': league_id, 'country': country, 'search': search, 'skip': skip, 'limit': limit, 'sort_by': sort_by, 'sort_order': sort_order, 'include_stats': include_stats, 'include_form': include_form}))}"

        # Verificar cache (deshabilitado temporalmente)
        # cached_result = await redis_client.get(cache_key)
        # if cached_result and not include_form:
        #     return json.loads(cached_result)

        # Construir query base con joins optimizados
        query = (
            db.query(Team)
            .options(
                joinedload(Team.league),
                joinedload(Team.stadium),
                joinedload(Team.coach),
                joinedload(Team.players) if include_players_count else None,
                joinedload(Team.stats) if include_stats else None,
            )
            .filter(Team.is_active == True)
        )

        # Aplicar filtros
        if league_id:
            query = query.filter(Team.league_id == league_id)

        if country:
            query = query.join(League).filter(League.country.ilike(f"%{country}%"))

        if min_market_value is not None:
            query = query.filter(Team.market_value >= min_market_value)

        if max_market_value is not None:
            query = query.filter(Team.market_value <= max_market_value)

        if search:
            search_term = f"%{search}%"
            query = query.filter(
                or_(Team.name.ilike(search_term), Team.short_name.ilike(search_term))
            )

        # Contar total antes de paginación
        total_teams = query.count()

        # Aplicar ordenamiento
        if sort_by == "name":
            order_by = Team.name
        elif sort_by == "founded_year":
            order_by = Team.founded_year
        elif sort_by == "market_value":
            order_by = Team.market_value
        else:
            order_by = Team.name

        if sort_order == "desc":
            query = query.order_by(desc(order_by))
        else:
            query = query.order_by(asc(order_by))

        # Aplicar paginación
        teams = query.offset(skip).limit(limit).all()

        # Procesar resultados
        teams_data = []
        form_calculator = FormCalculator() if include_form else None

        for team in teams:
            team_data = {
                "id": team.id,
                "name": team.name,
                "short_name": team.short_name,
                "founded_year": team.founded_year,
                "logo_url": team.logo_url,
                "stadium_name": team.stadium.name if team.stadium else None,
                "stadium_capacity": team.stadium.capacity if team.stadium else None,
                "league_name": team.league.name if team.league else None,
                "league_id": team.league_id,
                "country": team.league.country if team.league else None,
                "coach_name": team.coach.name if team.coach else None,
                "market_value": team.market_value,
            }

            # Número de jugadores
            if include_players_count:
                if hasattr(team, "players") and team.players:
                    team_data["squad_size"] = len(
                        [p for p in team.players if p.is_active]
                    )
                else:
                    # Consulta separada si no se cargó la relación
                    squad_count = (
                        db.query(Player)
                        .filter(Player.team_id == team.id, Player.is_active == True)
                        .count()
                    )
                    team_data["squad_size"] = squad_count

            # Estadísticas de temporada actual
            if include_stats and team.stats:
                latest_stats = sorted(team.stats, key=lambda x: x.season, reverse=True)
                if latest_stats:
                    stat = latest_stats[0]
                    team_data["current_season_stats"] = {
                        "matches_played": stat.matches_played,
                        "wins": stat.wins,
                        "draws": stat.draws,
                        "losses": stat.losses,
                        "goals_for": stat.goals_for,
                        "goals_against": stat.goals_against,
                        "goal_difference": stat.goal_difference,
                        "points": stat.points,
                        "position": stat.position,
                        "form_percentage": (
                            round(
                                (stat.wins * 3 + stat.draws)
                                / (stat.matches_played * 3)
                                * 100,
                                1,
                            )
                            if stat.matches_played > 0
                            else 0
                        ),
                    }

            # Forma reciente (últimos 5 partidos)
            if include_form and form_calculator:
                try:
                    form_data = await form_calculator.calculate_team_form(team.id, db)
                    team_data["recent_form"] = {
                        "last_5_results": form_data["results"],
                        "form_points": form_data["points"],
                        "form_rating": form_data["rating"],
                        "trend": form_data["trend"],
                    }
                except Exception:
                    team_data["recent_form"] = None

            teams_data.append(team_data)

        # Preparar respuesta
        result = {
            "success": True,
            "data": teams_data,
            "pagination": {
                "skip": skip,
                "limit": limit,
                "total": total_teams,
                "has_more": (skip + limit) < total_teams,
                "total_pages": (total_teams + limit - 1) // limit,
                "current_page": (skip // limit) + 1,
            },
            "filters_applied": {
                "league_id": league_id,
                "country": country,
                "market_value_range": (
                    f"{min_market_value}-{max_market_value}"
                    if min_market_value or max_market_value
                    else None
                ),
                "search": search,
            },
            "metadata": {
                "sort_by": sort_by,
                "sort_order": sort_order,
                "include_stats": include_stats,
                "include_players_count": include_players_count,
                "include_form": include_form,
            },
            "timestamp": datetime.now().isoformat(),
        }

        # Guardar en cache (5 minutos para datos sin forma)
        if not include_form:
            await redis_client.set(cache_key, json.dumps(result), ex=300)

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error al obtener equipos: {str(e)}"
        )


@router.get("/{team_id}", response_model=dict, summary="Obtener equipo específico")
async def get_team(
    team_id: int = Path(..., gt=0, description="ID del equipo"),
    include_stats: bool = Query(True, description="Incluir estadísticas detalladas"),
    include_players: bool = Query(True, description="Incluir plantilla de jugadores"),
    include_recent_matches: bool = Query(True, description="Incluir últimos partidos"),
    include_transfers: bool = Query(
        False, description="Incluir historial de transferencias"
    ),
    include_head_to_head: bool = Query(
        False, description="Incluir estadísticas H2H principales"
    ),
    season: Optional[str] = Query(
        None, description="Temporada específica para estadísticas"
    ),
    db: Session = Depends(get_db),
):
    """
    Obtener información detallada de un equipo específico.

    Incluye datos básicos, estadísticas, plantilla, partidos recientes
    y análisis de rendimiento.
    """
    try:
        # Cache key
        cache_key = f"team:detail:{team_id}:{include_stats}:{include_players}:{season}"

        # Verificar cache
        cached_result = await redis_client.get(cache_key)
        if (
            cached_result and not include_recent_matches
        ):  # No cache para partidos recientes
            return json.loads(cached_result)

        # Query base con joins optimizados
        query = (
            db.query(Team)
            .options(
                joinedload(Team.league),
                joinedload(Team.stadium),
                joinedload(Team.coach),
                joinedload(Team.stats) if include_stats else None,
                (
                    joinedload(Team.players).joinedload(Player.injuries)
                    if include_players
                    else None
                ),
                joinedload(Team.transfers) if include_transfers else None,
            )
            .filter(Team.id == team_id)
        )

        team = query.first()
        if not team:
            raise HTTPException(
                status_code=404, detail=f"Equipo con ID {team_id} no encontrado"
            )

        # Datos básicos del equipo
        result = {
            "success": True,
            "data": {
                "id": team.id,
                "name": team.name,
                "short_name": team.short_name,
                "founded_year": team.founded_year,
                "logo_url": team.logo_url,
                "website": team.website,
                "colors": team.colors,
                "market_value": team.market_value,
                "league": (
                    {
                        "id": team.league.id if team.league else None,
                        "name": team.league.name if team.league else None,
                        "country": team.league.country if team.league else None,
                        "tier": team.league.tier if team.league else None,
                    }
                    if team.league
                    else None
                ),
                "stadium": (
                    {
                        "id": team.stadium.id if team.stadium else None,
                        "name": team.stadium.name if team.stadium else None,
                        "capacity": team.stadium.capacity if team.stadium else None,
                        "city": team.stadium.city if team.stadium else None,
                        "surface": team.stadium.surface if team.stadium else None,
                    }
                    if team.stadium
                    else None
                ),
                "coach": (
                    {
                        "id": team.coach.id if team.coach else None,
                        "name": team.coach.name if team.coach else None,
                        "nationality": team.coach.nationality if team.coach else None,
                        "appointed_date": (
                            team.coach.appointed_date if team.coach else None
                        ),
                    }
                    if team.coach
                    else None
                ),
            },
        }

        # Estadísticas
        if include_stats and team.stats:
            stats_query = db.query(TeamStats).filter(TeamStats.team_id == team_id)
            if season:
                stats_query = stats_query.filter(TeamStats.season == season)

            stats = stats_query.all()

            if stats:
                # Estadísticas por temporada
                stats_by_season = {}
                for stat in stats:
                    stats_by_season[stat.season] = {
                        "matches_played": stat.matches_played,
                        "wins": stat.wins,
                        "draws": stat.draws,
                        "losses": stat.losses,
                        "goals_for": stat.goals_for,
                        "goals_against": stat.goals_against,
                        "goal_difference": stat.goal_difference,
                        "points": stat.points,
                        "position": stat.position,
                        "home_record": {
                            "wins": stat.home_wins,
                            "draws": stat.home_draws,
                            "losses": stat.home_losses,
                        },
                        "away_record": {
                            "wins": stat.away_wins,
                            "draws": stat.away_draws,
                            "losses": stat.away_losses,
                        },
                        "win_percentage": round(
                            stat.wins / max(stat.matches_played, 1) * 100, 1
                        ),
                        "goals_per_game": round(
                            stat.goals_for / max(stat.matches_played, 1), 2
                        ),
                        "goals_conceded_per_game": round(
                            stat.goals_against / max(stat.matches_played, 1), 2
                        ),
                    }

                result["data"]["statistics"] = {
                    "by_season": stats_by_season,
                    "all_time_totals": {
                        "total_matches": sum(s.matches_played for s in stats),
                        "total_wins": sum(s.wins for s in stats),
                        "total_draws": sum(s.draws for s in stats),
                        "total_losses": sum(s.losses for s in stats),
                        "total_goals_for": sum(s.goals_for for s in stats),
                        "total_goals_against": sum(s.goals_against for s in stats),
                    },
                }

        # Plantilla de jugadores
        if include_players:
            players_query = (
                db.query(Player)
                .filter(Player.team_id == team_id, Player.is_active == True)
                .order_by(Player.jersey_number.asc().nullslast(), Player.name)
                .all()
            )

            # Organizar por posiciones
            players_by_position = {
                "Goalkeeper": [],
                "Defender": [],
                "Midfielder": [],
                "Forward": [],
                "Other": [],
            }

            total_market_value = 0
            injured_players = 0

            for player in players_query:
                # Calcular edad
                age = None
                if player.birth_date:
                    age = (date.today() - player.birth_date).days // 365

                # Verificar lesiones activas
                is_injured = False
                if hasattr(player, "injuries") and player.injuries:
                    for injury in player.injuries:
                        if (
                            injury.injury_date <= datetime.now()
                            and (
                                injury.expected_return_date is None
                                or injury.expected_return_date > datetime.now().date()
                            )
                            and injury.actual_return_date is None
                        ):
                            is_injured = True
                            injured_players += 1
                            break

                player_data = {
                    "id": player.id,
                    "name": player.name,
                    "jersey_number": player.jersey_number,
                    "position": player.position,
                    "age": age,
                    "nationality": player.nationality,
                    "market_value": player.market_value,
                    "is_injured": is_injured,
                }

                if player.market_value:
                    total_market_value += player.market_value

                # Clasificar por posición
                position_key = "Other"
                if "Goalkeeper" in player.position or "GK" in player.position:
                    position_key = "Goalkeeper"
                elif any(
                    pos in player.position
                    for pos in ["Defender", "Back", "CB", "LB", "RB"]
                ):
                    position_key = "Defender"
                elif any(
                    pos in player.position
                    for pos in ["Midfielder", "Mid", "CM", "DM", "AM"]
                ):
                    position_key = "Midfielder"
                elif any(
                    pos in player.position
                    for pos in ["Forward", "Striker", "Wing", "CF", "LW", "RW"]
                ):
                    position_key = "Forward"

                players_by_position[position_key].append(player_data)

            result["data"]["squad"] = {
                "players_by_position": players_by_position,
                "squad_summary": {
                    "total_players": len(players_query),
                    "injured_players": injured_players,
                    "available_players": len(players_query) - injured_players,
                    "total_market_value": total_market_value,
                    "average_age": (
                        round(
                            sum(
                                (date.today() - p.birth_date).days // 365
                                for p in players_query
                                if p.birth_date
                            )
                            / len([p for p in players_query if p.birth_date]),
                            1,
                        )
                        if any(p.birth_date for p in players_query)
                        else None
                    ),
                },
            }

        # Partidos recientes
        if include_recent_matches:
            recent_matches = (
                db.query(Match)
                .options(
                    joinedload(Match.home_team),
                    joinedload(Match.away_team),
                    joinedload(Match.league),
                )
                .filter(
                    or_(Match.home_team_id == team_id, Match.away_team_id == team_id),
                    Match.status == "finished",
                )
                .order_by(desc(Match.match_date))
                .limit(10)
                .all()
            )

            matches_data = []
            recent_form = []

            for match in recent_matches:
                is_home = match.home_team_id == team_id
                opponent = match.away_team if is_home else match.home_team
                team_score = match.home_score if is_home else match.away_score
                opponent_score = match.away_score if is_home else match.home_score

                # Determinar resultado
                if team_score > opponent_score:
                    result_char = "W"
                    result_text = "Victoria"
                elif team_score < opponent_score:
                    result_char = "L"
                    result_text = "Derrota"
                else:
                    result_char = "D"
                    result_text = "Empate"

                recent_form.append(result_char)

                matches_data.append(
                    {
                        "match_id": match.id,
                        "date": match.match_date,
                        "opponent": opponent.name,
                        "opponent_id": opponent.id,
                        "is_home": is_home,
                        "score": f"{team_score}-{opponent_score}",
                        "result": result_text,
                        "league": match.league.name if match.league else None,
                    }
                )

            # Calcular forma reciente (últimos 5)
            form_points = 0
            for result_char in recent_form[:5]:
                if result_char == "W":
                    form_points += 3
                elif result_char == "D":
                    form_points += 1

            result["data"]["recent_matches"] = {
                "matches": matches_data,
                "form": {
                    "last_5_results": recent_form[:5],
                    "last_5_points": form_points,
                    "form_rating": round(
                        form_points / 15 * 100, 1
                    ),  # Sobre 15 puntos posibles
                    "current_streak": calculate_current_streak(recent_form),
                },
            }

        # Transferencias
        if include_transfers:
            transfers = (
                db.query(TeamTransfer)
                .filter(
                    or_(
                        TeamTransfer.from_team_id == team_id,
                        TeamTransfer.to_team_id == team_id,
                    )
                )
                .order_by(desc(TeamTransfer.transfer_date))
                .limit(20)
                .all()
            )

            result["data"]["transfers"] = {
                "incoming": [
                    {
                        "player_name": t.player_name,
                        "from_team": t.from_team,
                        "transfer_fee": t.transfer_fee,
                        "transfer_date": t.transfer_date,
                        "transfer_type": t.transfer_type,
                    }
                    for t in transfers
                    if t.to_team_id == team_id
                ],
                "outgoing": [
                    {
                        "player_name": t.player_name,
                        "to_team": t.to_team,
                        "transfer_fee": t.transfer_fee,
                        "transfer_date": t.transfer_date,
                        "transfer_type": t.transfer_type,
                    }
                    for t in transfers
                    if t.from_team_id == team_id
                ],
            }

        result["metadata"] = {
            "include_stats": include_stats,
            "include_players": include_players,
            "include_recent_matches": include_recent_matches,
            "include_transfers": include_transfers,
            "season_filter": season,
        }
        result["timestamp"] = datetime.now().isoformat()

        # Cache por 15 minutos (datos relativamente estáticos)
        if not include_recent_matches:
            await redis_client.set(cache_key, json.dumps(result), ex=900)

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error al obtener equipo: {str(e)}"
        )


@router.get(
    "/{team_id}/stats",
    response_model=dict,
    summary="Estadísticas detalladas del equipo",
)
async def get_team_stats(
    team_id: int = Path(..., gt=0, description="ID del equipo"),
    season: Optional[str] = Query(None, description="Temporada específica"),
    home_away: Optional[str] = Query(
        None, regex="^(home|away|all)$", description="Filtrar por local/visitante"
    ),
    compare_seasons: bool = Query(False, description="Comparar entre temporadas"),
    include_advanced_metrics: bool = Query(
        False, description="Incluir métricas avanzadas"
    ),
    db: Session = Depends(get_db),
):
    """
    Obtener estadísticas detalladas de un equipo con opciones de filtro y comparación.
    """
    try:
        # Verificar que el equipo existe
        team = db.query(Team).filter(Team.id == team_id).first()
        if not team:
            raise HTTPException(
                status_code=404, detail=f"Equipo con ID {team_id} no encontrado"
            )

        # Query base de estadísticas
        stats_query = db.query(TeamStats).filter(TeamStats.team_id == team_id)

        if season:
            stats_query = stats_query.filter(TeamStats.season == season)

        stats = stats_query.order_by(desc(TeamStats.season)).all()

        if not stats:
            return {
                "success": True,
                "data": {
                    "team_id": team_id,
                    "team_name": team.name,
                    "message": "No hay estadísticas disponibles para este equipo",
                },
                "timestamp": datetime.now().isoformat(),
            }

        result = {
            "success": True,
            "data": {
                "team_id": team_id,
                "team_name": team.name,
                "league": team.league.name if team.league else None,
            },
        }

        # Estadísticas por temporada
        seasons_stats = {}
        for stat in stats:
            season_data = {
                "matches_played": stat.matches_played,
                "wins": stat.wins,
                "draws": stat.draws,
                "losses": stat.losses,
                "goals_for": stat.goals_for,
                "goals_against": stat.goals_against,
                "goal_difference": stat.goal_difference,
                "points": stat.points,
                "position": stat.position,
            }

            # Filtrar por local/visitante
            if home_away == "home":
                season_data.update(
                    {
                        "home_matches": stat.matches_played // 2,  # Aproximado
                        "home_wins": stat.home_wins,
                        "home_draws": stat.home_draws,
                        "home_losses": stat.home_losses,
                        "home_win_percentage": round(
                            stat.home_wins
                            / max(
                                stat.home_wins + stat.home_draws + stat.home_losses, 1
                            )
                            * 100,
                            1,
                        ),
                    }
                )
            elif home_away == "away":
                season_data.update(
                    {
                        "away_matches": stat.matches_played // 2,  # Aproximado
                        "away_wins": stat.away_wins,
                        "away_draws": stat.away_draws,
                        "away_losses": stat.away_losses,
                        "away_win_percentage": round(
                            stat.away_wins
                            / max(
                                stat.away_wins + stat.away_draws + stat.away_losses, 1
                            )
                            * 100,
                            1,
                        ),
                    }
                )
            else:
                # Estadísticas completas
                season_data.update(
                    {
                        "win_percentage": round(
                            stat.wins / max(stat.matches_played, 1) * 100, 1
                        ),
                        "goals_per_game": round(
                            stat.goals_for / max(stat.matches_played, 1), 2
                        ),
                        "goals_conceded_per_game": round(
                            stat.goals_against / max(stat.matches_played, 1), 2
                        ),
                        "points_per_game": round(
                            stat.points / max(stat.matches_played, 1), 2
                        ),
                        "home_record": f"{stat.home_wins}-{stat.home_draws}-{stat.home_losses}",
                        "away_record": f"{stat.away_wins}-{stat.away_draws}-{stat.away_losses}",
                    }
                )

            seasons_stats[stat.season] = season_data

        result["data"]["seasons"] = seasons_stats

        # Totales históricos
        if len(stats) > 1:
            historical_totals = {
                "total_seasons": len(stats),
                "total_matches": sum(s.matches_played for s in stats),
                "total_wins": sum(s.wins for s in stats),
                "total_draws": sum(s.draws for s in stats),
                "total_losses": sum(s.losses for s in stats),
                "total_goals_for": sum(s.goals_for for s in stats),
                "total_goals_against": sum(s.goals_against for s in stats),
                "historical_win_percentage": round(
                    sum(s.wins for s in stats)
                    / max(sum(s.matches_played for s in stats), 1)
                    * 100,
                    1,
                ),
                "best_season_points": max(s.points for s in stats),
                "best_season_position": min(s.position for s in stats if s.position),
                "worst_season_position": max(s.position for s in stats if s.position),
            }
            result["data"]["historical_totals"] = historical_totals

        # Comparación entre temporadas
        if compare_seasons and len(stats) >= 2:
            # Comparar última temporada vs anterior
            latest_season = stats[0]
            previous_season = stats[1]

            comparison = {
                "latest_season": latest_season.season,
                "previous_season": previous_season.season,
                "points_change": latest_season.points - previous_season.points,
                "position_change": (
                    previous_season.position - latest_season.position
                    if latest_season.position and previous_season.position
                    else None
                ),
                "goals_for_change": latest_season.goals_for - previous_season.goals_for,
                "goals_against_change": latest_season.goals_against
                - previous_season.goals_against,
                "performance_trend": (
                    "improving"
                    if latest_season.points > previous_season.points
                    else "declining"
                ),
            }
            result["data"]["season_comparison"] = comparison

        # Métricas avanzadas
        if include_advanced_metrics:
            try:
                analyzer = TeamAnalyzer()
                advanced_metrics = await analyzer.calculate_advanced_metrics(
                    team_id, season, db
                )
                result["data"]["advanced_metrics"] = advanced_metrics
            except Exception as e:
                result["data"]["advanced_metrics"] = {
                    "error": f"No se pudieron calcular métricas avanzadas: {str(e)}"
                }

        result["metadata"] = {
            "season_filter": season,
            "home_away_filter": home_away,
            "compare_seasons": compare_seasons,
            "include_advanced_metrics": include_advanced_metrics,
            "seasons_available": len(stats),
        }
        result["timestamp"] = datetime.now().isoformat()

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error al obtener estadísticas: {str(e)}"
        )


@router.get("/{team_id}/form", response_model=dict, summary="Forma reciente del equipo")
async def get_team_form(
    team_id: int = Path(..., gt=0, description="ID del equipo"),
    matches_count: int = Query(
        5, ge=3, le=20, description="Número de partidos a analizar"
    ),
    home_away: Optional[str] = Query(
        None, regex="^(home|away|all)$", description="Solo local, visitante o todos"
    ),
    include_analysis: bool = Query(True, description="Incluir análisis detallado"),
    db: Session = Depends(get_db),
):
    """
    Analizar la forma reciente de un equipo basada en sus últimos partidos.
    """
    try:
        # Verificar que el equipo existe
        team = db.query(Team).filter(Team.id == team_id).first()
        if not team:
            raise HTTPException(
                status_code=404, detail=f"Equipo con ID {team_id} no encontrado"
            )

        # Construir query de partidos recientes
        query = (
            db.query(Match)
            .options(
                joinedload(Match.home_team),
                joinedload(Match.away_team),
                joinedload(Match.league),
            )
            .filter(
                or_(Match.home_team_id == team_id, Match.away_team_id == team_id),
                Match.status == "finished",
            )
        )

        # Filtrar por local/visitante
        if home_away == "home":
            query = query.filter(Match.home_team_id == team_id)
        elif home_away == "away":
            query = query.filter(Match.away_team_id == team_id)

        # Obtener partidos recientes
        recent_matches = (
            query.order_by(desc(Match.match_date)).limit(matches_count).all()
        )

        if not recent_matches:
            return {
                "success": True,
                "data": {
                    "team_id": team_id,
                    "team_name": team.name,
                    "message": "No hay partidos recientes disponibles",
                },
                "timestamp": datetime.now().isoformat(),
            }

        # Analizar forma
        form_calculator = FormCalculator()
        matches_analysis = []
        results = []
        points = 0
        goals_for = 0
        goals_against = 0

        for match in recent_matches:
            is_home = match.home_team_id == team_id
            opponent = match.away_team if is_home else match.home_team
            team_score = match.home_score if is_home else match.away_score
            opponent_score = match.away_score if is_home else match.home_score

            # Determinar resultado
            if team_score > opponent_score:
                result = "W"
                match_points = 3
                result_text = "Victoria"
            elif team_score < opponent_score:
                result = "L"
                match_points = 0
                result_text = "Derrota"
            else:
                result = "D"
                match_points = 1
                result_text = "Empate"

            results.append(result)
            points += match_points
            goals_for += team_score
            goals_against += opponent_score

            matches_analysis.append(
                {
                    "match_id": match.id,
                    "date": match.match_date,
                    "opponent": opponent.name,
                    "opponent_id": opponent.id,
                    "is_home": is_home,
                    "venue": "Local" if is_home else "Visitante",
                    "score": f"{team_score}-{opponent_score}",
                    "result": result_text,
                    "result_code": result,
                    "points": match_points,
                    "league": match.league.name if match.league else None,
                }
            )

        # Calcular métricas de forma
        total_possible_points = len(recent_matches) * 3
        form_percentage = round(points / total_possible_points * 100, 1)

        # Determinar tendencia
        if len(results) >= 3:
            recent_3 = results[:3]
            recent_3_points = sum(
                3 if r == "W" else 1 if r == "D" else 0 for r in recent_3
            )
            if recent_3_points >= 7:
                trend = "improving"
            elif recent_3_points <= 3:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        # Racha actual
        current_streak = calculate_current_streak(results)

        form_data = {
            "team_id": team_id,
            "team_name": team.name,
            "analysis_period": {
                "matches_analyzed": len(recent_matches),
                "home_away_filter": home_away,
                "date_range": {
                    "from": recent_matches[-1].match_date if recent_matches else None,
                    "to": recent_matches[0].match_date if recent_matches else None,
                },
            },
            "form_summary": {
                "results": results,
                "points": points,
                "max_possible_points": total_possible_points,
                "form_percentage": form_percentage,
                "wins": results.count("W"),
                "draws": results.count("D"),
                "losses": results.count("L"),
                "goals_for": goals_for,
                "goals_against": goals_against,
                "goal_difference": goals_for - goals_against,
                "goals_per_game": round(goals_for / len(recent_matches), 2),
                "goals_conceded_per_game": round(
                    goals_against / len(recent_matches), 2
                ),
                "trend": trend,
                "current_streak": current_streak,
            },
            "matches_detail": matches_analysis,
        }

        # Análisis detallado
        if include_analysis:
            try:
                analyzer = TeamAnalyzer()
                detailed_analysis = await analyzer.analyze_form_patterns(
                    team_id, recent_matches, db
                )
                form_data["detailed_analysis"] = detailed_analysis
            except Exception as e:
                form_data["detailed_analysis"] = {
                    "error": f"No se pudo generar análisis detallado: {str(e)}"
                }

        result = {
            "success": True,
            "data": form_data,
            "metadata": {
                "matches_count": matches_count,
                "home_away": home_away,
                "include_analysis": include_analysis,
            },
            "timestamp": datetime.now().isoformat(),
        }

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error al analizar forma: {str(e)}"
        )


@router.get(
    "/{team_id}/head-to-head/{opponent_id}",
    response_model=dict,
    summary="Estadísticas cara a cara",
)
async def get_head_to_head(
    team_id: int = Path(..., gt=0, description="ID del equipo"),
    opponent_id: int = Path(..., gt=0, description="ID del equipo rival"),
    limit: int = Query(10, ge=5, le=50, description="Número de partidos a analizar"),
    venue: Optional[str] = Query(
        None, regex="^(home|away|neutral|all)$", description="Filtrar por sede"
    ),
    include_predictions: bool = Query(
        False, description="Incluir predicciones para próximo enfrentamiento"
    ),
    db: Session = Depends(get_db),
):
    """
    Análisis de estadísticas cara a cara entre dos equipos.
    """
    try:
        # Verificar que ambos equipos existen
        team = db.query(Team).filter(Team.id == team_id).first()
        opponent = db.query(Team).filter(Team.id == opponent_id).first()

        if not team or not opponent:
            raise HTTPException(
                status_code=404, detail="Uno o ambos equipos no encontrados"
            )

        if team_id == opponent_id:
            raise HTTPException(
                status_code=400, detail="No se puede comparar un equipo consigo mismo"
            )

        # Construir query de enfrentamientos
        query = (
            db.query(Match)
            .options(
                joinedload(Match.home_team),
                joinedload(Match.away_team),
                joinedload(Match.league),
            )
            .filter(
                or_(
                    and_(
                        Match.home_team_id == team_id, Match.away_team_id == opponent_id
                    ),
                    and_(
                        Match.home_team_id == opponent_id, Match.away_team_id == team_id
                    ),
                ),
                Match.status == "finished",
            )
        )

        # Filtrar por sede
        if venue == "home":
            query = query.filter(Match.home_team_id == team_id)
        elif venue == "away":
            query = query.filter(Match.away_team_id == team_id)
        elif venue == "neutral":
            # Asumiendo que partidos neutrales tienen un flag especial
            query = query.filter(getattr(Match, "is_neutral", False) == True)

        # Obtener enfrentamientos
        h2h_matches = query.order_by(desc(Match.match_date)).limit(limit).all()

        if not h2h_matches:
            return {
                "success": True,
                "data": {
                    "team": {"id": team_id, "name": team.name},
                    "opponent": {"id": opponent_id, "name": opponent.name},
                    "message": "No hay enfrentamientos previos entre estos equipos",
                },
                "timestamp": datetime.now().isoformat(),
            }

        # Analizar estadísticas
        team_wins = 0
        opponent_wins = 0
        draws = 0
        team_goals = 0
        opponent_goals = 0
        matches_analysis = []

        for match in h2h_matches:
            team_is_home = match.home_team_id == team_id
            team_score = match.home_score if team_is_home else match.away_score
            opponent_score = match.away_score if team_is_home else match.home_score

            # Contabilizar resultado
            if team_score > opponent_score:
                team_wins += 1
                result_for_team = "W"
            elif team_score < opponent_score:
                opponent_wins += 1
                result_for_team = "L"
            else:
                draws += 1
                result_for_team = "D"

            team_goals += team_score
            opponent_goals += opponent_score

            matches_analysis.append(
                {
                    "match_id": match.id,
                    "date": match.match_date,
                    "venue": "Local" if team_is_home else "Visitante",
                    "score": f"{team_score}-{opponent_score}",
                    "result": result_for_team,
                    "league": match.league.name if match.league else None,
                    "season": getattr(match, "season", "Unknown"),
                }
            )

        # Calcular estadísticas generales
        total_matches = len(h2h_matches)
        team_win_percentage = round(team_wins / total_matches * 100, 1)

        # Estadísticas por sede
        home_matches = [m for m in h2h_matches if m.home_team_id == team_id]
        away_matches = [m for m in h2h_matches if m.away_team_id == team_id]

        home_wins = len([m for m in home_matches if m.home_score > m.away_score])
        away_wins = len([m for m in away_matches if m.away_score > m.home_score])

        # Tendencias recientes (últimos 5)
        recent_5 = matches_analysis[:5]
        recent_form = [m["result"] for m in recent_5]
        recent_wins = recent_form.count("W")

        h2h_data = {
            "teams": {
                "team": {"id": team_id, "name": team.name},
                "opponent": {"id": opponent_id, "name": opponent.name},
            },
            "overall_statistics": {
                "total_matches": total_matches,
                "team_wins": team_wins,
                "opponent_wins": opponent_wins,
                "draws": draws,
                "team_win_percentage": team_win_percentage,
                "opponent_win_percentage": round(
                    opponent_wins / total_matches * 100, 1
                ),
                "draw_percentage": round(draws / total_matches * 100, 1),
            },
            "goals_statistics": {
                "team_goals_total": team_goals,
                "opponent_goals_total": opponent_goals,
                "team_goals_per_game": round(team_goals / total_matches, 2),
                "opponent_goals_per_game": round(opponent_goals / total_matches, 2),
                "total_goals_per_game": round(
                    (team_goals + opponent_goals) / total_matches, 2
                ),
                "highest_scoring_match": max(
                    [(m["score"], m["date"]) for m in matches_analysis],
                    key=lambda x: sum(int(s) for s in x[0].split("-")),
                ),
            },
            "venue_breakdown": {
                "when_team_at_home": {
                    "matches": len(home_matches),
                    "wins": home_wins,
                    "win_percentage": round(
                        home_wins / max(len(home_matches), 1) * 100, 1
                    ),
                },
                "when_team_away": {
                    "matches": len(away_matches),
                    "wins": away_wins,
                    "win_percentage": round(
                        away_wins / max(len(away_matches), 1) * 100, 1
                    ),
                },
            },
            "recent_form": {
                "last_5_matches": recent_5,
                "last_5_wins": recent_wins,
                "last_meeting": matches_analysis[0] if matches_analysis else None,
                "recent_trend": (
                    "favorable"
                    if recent_wins >= 3
                    else "unfavorable" if recent_wins <= 1 else "balanced"
                ),
            },
            "matches_history": matches_analysis,
        }

        # Predicciones para próximo enfrentamiento
        if include_predictions:
            try:
                # Buscar próximo partido programado entre estos equipos
                next_match = (
                    db.query(Match)
                    .filter(
                        or_(
                            and_(
                                Match.home_team_id == team_id,
                                Match.away_team_id == opponent_id,
                            ),
                            and_(
                                Match.home_team_id == opponent_id,
                                Match.away_team_id == team_id,
                            ),
                        ),
                        Match.status == "scheduled",
                        Match.match_date > datetime.now(),
                    )
                    .order_by(Match.match_date)
                    .first()
                )

                if next_match:
                    from app.services.advanced_predictor import (
                        AdvancedFootballPredictor,
                    )

                    predictor = AdvancedFootballPredictor()

                    match_data = {
                        "home_team_id": next_match.home_team_id,
                        "away_team_id": next_match.away_team_id,
                        "match_date": next_match.match_date,
                        "league_id": next_match.league_id,
                    }

                    prediction = await predictor.predict_match_result(match_data, db)
                    h2h_data["next_match_prediction"] = {
                        "match_date": next_match.match_date,
                        "prediction": prediction,
                        "h2h_influence": (
                            "favorable" if team_win_percentage > 50 else "unfavorable"
                        ),
                    }
                else:
                    h2h_data["next_match_prediction"] = {
                        "message": "No hay próximos enfrentamientos programados"
                    }

            except Exception as e:
                h2h_data["next_match_prediction"] = {
                    "error": f"Error generando predicción: {str(e)}"
                }

        result = {
            "success": True,
            "data": h2h_data,
            "filters": {
                "limit": limit,
                "venue": venue,
                "include_predictions": include_predictions,
            },
            "timestamp": datetime.now().isoformat(),
        }

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error en análisis cara a cara: {str(e)}"
        )


@router.get("/search", response_model=dict, summary="Búsqueda avanzada de equipos")
async def search_teams(
    q: str = Query(..., min_length=2, description="Término de búsqueda"),
    league_id: Optional[int] = Query(None, description="Filtrar por liga"),
    country: Optional[str] = Query(None, description="Filtrar por país"),
    limit: int = Query(20, ge=1, le=100, description="Límite de resultados"),
    include_suggestions: bool = Query(True, description="Incluir sugerencias"),
    db: Session = Depends(get_db),
):
    """
    Búsqueda avanzada de equipos con filtros y sugerencias.
    """
    try:
        # Query base
        query = (
            db.query(Team)
            .options(joinedload(Team.league), joinedload(Team.stadium))
            .filter(Team.is_active == True)
        )

        # Búsqueda por texto
        search_term = f"%{q.lower()}%"
        text_filter = or_(
            func.lower(Team.name).like(search_term),
            func.lower(Team.short_name).like(search_term),
        )

        query = query.filter(text_filter)

        # Aplicar filtros adicionales
        if league_id:
            query = query.filter(Team.league_id == league_id)

        if country:
            query = query.join(League).filter(League.country.ilike(f"%{country}%"))

        # Ordenar por relevancia (nombre exacto primero)
        query = query.order_by(
            func.lower(Team.name).like(f"{q.lower()}%").desc(),
            Team.market_value.desc().nullslast(),
            Team.name,
        )

        # Limitar resultados
        teams = query.limit(limit).all()

        # Procesar resultados
        results = []
        for team in teams:
            results.append(
                {
                    "id": team.id,
                    "name": team.name,
                    "short_name": team.short_name,
                    "logo_url": team.logo_url,
                    "league": (
                        {
                            "id": team.league.id if team.league else None,
                            "name": team.league.name if team.league else None,
                            "country": team.league.country if team.league else None,
                        }
                        if team.league
                        else None
                    ),
                    "stadium_name": team.stadium.name if team.stadium else None,
                    "market_value": team.market_value,
                    "relevance_score": (
                        100 if team.name.lower().startswith(q.lower()) else 50
                    ),
                }
            )

        result = {
            "success": True,
            "data": {
                "query": q,
                "results": results,
                "total_found": len(results),
                "filters_applied": {"league_id": league_id, "country": country},
            },
        }

        # Sugerencias si no hay muchos resultados
        if include_suggestions and len(results) < 5:
            # Búsqueda más amplia para sugerencias
            suggestion_query = (
                db.query(Team)
                .options(joinedload(Team.league))
                .filter(
                    Team.is_active == True,
                    or_(
                        func.lower(Team.name).like(f"%{q.lower()}%"),
                        Team.league.has(League.country.ilike(f"%{q.lower()}%")),
                    ),
                )
                .limit(10)
                .all()
            )

            suggestions = [
                {
                    "id": t.id,
                    "name": t.name,
                    "league_name": t.league.name if t.league else None,
                    "country": t.league.country if t.league else None,
                }
                for t in suggestion_query
                if t.id not in [r["id"] for r in results]
            ]

            result["data"]["suggestions"] = suggestions

        result["timestamp"] = datetime.now().isoformat()

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en búsqueda: {str(e)}")


# =====================================================
# FUNCIONES AUXILIARES
# =====================================================


def calculate_current_streak(results: List[str]) -> Dict[str, Any]:
    """
    Calcular la racha actual del equipo.
    """
    if not results:
        return {"type": "none", "count": 0, "description": "Sin datos"}

    current_result = results[0]
    streak_count = 1

    for result in results[1:]:
        if result == current_result:
            streak_count += 1
        else:
            break

    streak_descriptions = {
        "W": f"{streak_count} victoria{'s' if streak_count > 1 else ''} consecutiva{'s' if streak_count > 1 else ''}",
        "L": f"{streak_count} derrota{'s' if streak_count > 1 else ''} consecutiva{'s' if streak_count > 1 else ''}",
        "D": f"{streak_count} empate{'s' if streak_count > 1 else ''} consecutivo{'s' if streak_count > 1 else ''}",
    }

    return {
        "type": current_result,
        "count": streak_count,
        "description": streak_descriptions.get(current_result, "Racha desconocida"),
    }
