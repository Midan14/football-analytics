"""
API Routes para gestión de jugadores de fútbol
Incluye CRUD, estadísticas, rendimiento, lesiones y análisis predictivo
"""

import json
from datetime import date, datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel
from sqlalchemy import and_, asc, desc, func, or_
from sqlalchemy.orm import Session, joinedload

from app.database.connection import get_db
from app.database.models import (
    Match,
    MatchEvent,
    Player,
    PlayerInjury,
    PlayerStats,
    PlayerTransfer,
    Team,
)
from app.services.player_analyzer import PlayerAnalyzer

# from app.core.redis_client import redis_client  # TODO: implement redis client

router = APIRouter()

# =====================================================
# MODELOS PYDANTIC (SCHEMAS)
# =====================================================

class PlayerResponse(BaseModel):
    id: int
    name: str
    full_name: Optional[str] = None
    birth_date: Optional[date] = None
    age: Optional[int] = None
    nationality: Optional[str] = None
    position: str
    jersey_number: Optional[int] = None
    market_value: Optional[float] = None
    team_name: Optional[str] = None
    team_id: Optional[int] = None
    league_name: Optional[str] = None
    is_injured: bool = False
    photo_url: Optional[str] = None

class PlayerStatsResponse(BaseModel):
    player_id: int
    season: str
    matches_played: int
    goals: int
    assists: int
    yellow_cards: int
    red_cards: int
    minutes_played: int
    pass_accuracy: Optional[float] = None
    shots_on_target: Optional[int] = None
    key_passes: Optional[int] = None
    rating: Optional[float] = None

class PlayerSearchFilters(BaseModel):
    name: Optional[str] = None
    team_id: Optional[int] = None
    league_id: Optional[int] = None
    position: Optional[str] = None
    nationality: Optional[str] = None
    age_min: Optional[int] = None
    age_max: Optional[int] = None
    market_value_min: Optional[float] = None
    market_value_max: Optional[float] = None
    is_injured: Optional[bool] = None

# =====================================================
# ENDPOINTS PRINCIPALES
# =====================================================

@router.get("/", response_model=dict, summary="Obtener lista de jugadores")
async def get_players(
    # Filtros básicos
    team_id: Optional[int] = Query(None, description="ID del equipo"),
    league_id: Optional[int] = Query(None, description="ID de la liga"),
    position: Optional[str] = Query(None, description="Posición del jugador"),
    nationality: Optional[str] = Query(None, description="Nacionalidad"),
    
    # Filtros de edad y valor
    age_min: Optional[int] = Query(None, ge=16, le=50, description="Edad mínima"),
    age_max: Optional[int] = Query(None, ge=16, le=50, description="Edad máxima"),
    market_value_min: Optional[float] = Query(None, ge=0, description="Valor mínimo de mercado"),
    market_value_max: Optional[float] = Query(None, ge=0, description="Valor máximo de mercado"),
    
    # Filtros de estado
    is_injured: Optional[bool] = Query(None, description="Filtrar por lesionados"),
    available_only: bool = Query(False, description="Solo jugadores disponibles"),
    
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
    include_injuries: bool = Query(False, description="Incluir historial de lesiones"),
    
    db: Session = Depends(get_db)
):
    """
    Obtener lista de jugadores con filtros avanzados, búsqueda y paginación.
    
    Características:
    - Filtros por equipo, liga, posición, nacionalidad, edad, valor
    - Búsqueda por nombre con coincidencias parciales
    - Ordenamiento por múltiples campos
    - Datos opcionales (estadísticas, lesiones)
    - Cache Redis para mejor performance
    """
    try:
        # Crear clave de cache
        cache_key = f"players:list:{hash(str({
            'team_id': team_id, 'league_id': league_id, 'position': position,
            'nationality': nationality, 'search': search, 'skip': skip, 'limit': limit,
            'sort_by': sort_by, 'sort_order': sort_order, 'include_stats': include_stats
        }))}"
        
        # Verificar cache
        cached_result = await redis_client.get(cache_key)
        if cached_result and not include_stats:  # No cache si incluye stats
            return json.loads(cached_result)
        
        # Construir query base con joins optimizados
        query = (
            db.query(Player)
            .options(
                joinedload(Player.team).joinedload(Team.league),
                joinedload(Player.injuries) if include_injuries else None,
                joinedload(Player.stats) if include_stats else None
            )
            .filter(Player.is_active == True)
        )
        
        # Aplicar filtros
        if team_id:
            query = query.filter(Player.team_id == team_id)
        
        if league_id:
            query = query.join(Team).filter(Team.league_id == league_id)
        
        if position:
            query = query.filter(Player.position.ilike(f"%{position}%"))
        
        if nationality:
            query = query.filter(Player.nationality.ilike(f"%{nationality}%"))
        
        if age_min is not None or age_max is not None:
            current_date = date.today()
            if age_min is not None:
                max_birth_date = current_date - timedelta(days=age_min * 365)
                query = query.filter(Player.birth_date <= max_birth_date)
            if age_max is not None:
                min_birth_date = current_date - timedelta(days=age_max * 365)
                query = query.filter(Player.birth_date >= min_birth_date)
        
        if market_value_min is not None:
            query = query.filter(Player.market_value >= market_value_min)
        
        if market_value_max is not None:
            query = query.filter(Player.market_value <= market_value_max)
        
        if is_injured is not None:
            if is_injured:
                query = query.join(PlayerInjury).filter(
                    and_(
                        PlayerInjury.injury_date <= datetime.now(),
                        or_(
                            PlayerInjury.expected_return_date == None,
                            PlayerInjury.expected_return_date > datetime.now()
                        )
                    )
                )
            else:
                # Jugadores sin lesiones activas
                subquery = (
                    db.query(PlayerInjury.player_id)
                    .filter(
                        and_(
                            PlayerInjury.injury_date <= datetime.now(),
                            or_(
                                PlayerInjury.expected_return_date == None,
                                PlayerInjury.expected_return_date > datetime.now()
                            )
                        )
                    )
                )
                query = query.filter(~Player.id.in_(subquery))
        
        if available_only:
            # Solo jugadores disponibles (no lesionados, no suspendidos)
            today = datetime.now().date()
            query = query.filter(
                and_(
                    Player.is_active == True,
                    ~Player.id.in_(
                        db.query(PlayerInjury.player_id)
                        .filter(
                            and_(
                                PlayerInjury.injury_date <= today,
                                or_(
                                    PlayerInjury.expected_return_date == None,
                                    PlayerInjury.expected_return_date > today
                                )
                            )
                        )
                    )
                )
            )
        
        if search:
            search_term = f"%{search}%"
            query = query.filter(
                or_(
                    Player.name.ilike(search_term),
                    Player.full_name.ilike(search_term)
                )
            )
        
        # Contar total antes de paginación
        total_players = query.count()
        
        # Aplicar ordenamiento
        if sort_by == "name":
            order_by = Player.name
        elif sort_by == "age":
            order_by = Player.birth_date  # Más joven = fecha más reciente
        elif sort_by == "market_value":
            order_by = Player.market_value
        elif sort_by == "jersey_number":
            order_by = Player.jersey_number
        else:
            order_by = Player.name
        
        if sort_order == "desc":
            query = query.order_by(desc(order_by))
        else:
            query = query.order_by(asc(order_by))
        
        # Aplicar paginación
        players = query.offset(skip).limit(limit).all()
        
        # Procesar resultados
        players_data = []
        for player in players:
            # Calcular edad
            age = None
            if player.birth_date:
                age = (date.today() - player.birth_date).days // 365
            
            # Verificar si está lesionado
            is_injured = False
            if include_injuries and player.injuries:
                for injury in player.injuries:
                    if (injury.injury_date <= datetime.now() and 
                        (injury.expected_return_date is None or 
                         injury.expected_return_date > datetime.now())):
                        is_injured = True
                        break
            
            player_data = {
                "id": player.id,
                "name": player.name,
                "full_name": player.full_name,
                "birth_date": player.birth_date,
                "age": age,
                "nationality": player.nationality,
                "position": player.position,
                "jersey_number": player.jersey_number,
                "market_value": player.market_value,
                "team_name": player.team.name if player.team else None,
                "team_id": player.team_id,
                "league_name": player.team.league.name if player.team and player.team.league else None,
                "is_injured": is_injured,
                "photo_url": player.photo_url
            }
            
            # Agregar estadísticas si se solicitan
            if include_stats and player.stats:
                latest_stats = sorted(player.stats, key=lambda x: x.season, reverse=True)
                if latest_stats:
                    stat = latest_stats[0]
                    player_data["current_season_stats"] = {
                        "matches_played": stat.matches_played,
                        "goals": stat.goals,
                        "assists": stat.assists,
                        "yellow_cards": stat.yellow_cards,
                        "red_cards": stat.red_cards,
                        "minutes_played": stat.minutes_played,
                        "rating": stat.rating
                    }
            
            players_data.append(player_data)
        
        # Preparar respuesta
        result = {
            "success": True,
            "data": players_data,
            "pagination": {
                "skip": skip,
                "limit": limit,
                "total": total_players,
                "has_more": (skip + limit) < total_players,
                "total_pages": (total_players + limit - 1) // limit,
                "current_page": (skip // limit) + 1
            },
            "filters_applied": {
                "team_id": team_id,
                "league_id": league_id,
                "position": position,
                "nationality": nationality,
                "age_range": f"{age_min}-{age_max}" if age_min or age_max else None,
                "market_value_range": f"{market_value_min}-{market_value_max}" if market_value_min or market_value_max else None,
                "search": search,
                "is_injured": is_injured,
                "available_only": available_only
            },
            "metadata": {
                "sort_by": sort_by,
                "sort_order": sort_order,
                "include_stats": include_stats,
                "include_injuries": include_injuries
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Guardar en cache (5 minutos para datos sin estadísticas)
        if not include_stats:
            await redis_client.set(cache_key, json.dumps(result), ex=300)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener jugadores: {str(e)}")


@router.get("/{player_id}", response_model=dict, summary="Obtener jugador específico")
async def get_player(
    player_id: int = Path(..., gt=0, description="ID del jugador"),
    include_stats: bool = Query(True, description="Incluir estadísticas detalladas"),
    include_injuries: bool = Query(True, description="Incluir historial de lesiones"),
    include_transfers: bool = Query(False, description="Incluir historial de transferencias"),
    include_recent_matches: bool = Query(True, description="Incluir últimos partidos"),
    include_performance_analysis: bool = Query(False, description="Incluir análisis de rendimiento"),
    season: Optional[str] = Query(None, description="Temporada específica para estadísticas"),
    db: Session = Depends(get_db)
):
    """
    Obtener información detallada de un jugador específico.
    
    Incluye datos personales, estadísticas, lesiones, transferencias
    y análisis de rendimiento opcional.
    """
    try:
        # Cache key
        cache_key = f"player:detail:{player_id}:{include_stats}:{season}"
        
        # Verificar cache
        cached_result = await redis_client.get(cache_key)
        if cached_result and not include_performance_analysis:
            return json.loads(cached_result)
        
        # Query base con joins optimizados
        query = (
            db.query(Player)
            .options(
                joinedload(Player.team).joinedload(Team.league),
                joinedload(Player.stats) if include_stats else None,
                joinedload(Player.injuries) if include_injuries else None,
                joinedload(Player.transfers) if include_transfers else None
            )
            .filter(Player.id == player_id)
        )
        
        player = query.first()
        if not player:
            raise HTTPException(status_code=404, detail=f"Jugador con ID {player_id} no encontrado")
        
        # Datos básicos del jugador
        age = None
        if player.birth_date:
            age = (date.today() - player.birth_date).days // 365
        
        result = {
            "success": True,
            "data": {
                "id": player.id,
                "name": player.name,
                "full_name": player.full_name,
                "birth_date": player.birth_date,
                "age": age,
                "nationality": player.nationality,
                "position": player.position,
                "jersey_number": player.jersey_number,
                "height": player.height,
                "weight": player.weight,
                "preferred_foot": player.preferred_foot,
                "market_value": player.market_value,
                "contract_expires": player.contract_expires,
                "photo_url": player.photo_url,
                "team": {
                    "id": player.team.id if player.team else None,
                    "name": player.team.name if player.team else None,
                    "logo_url": player.team.logo_url if player.team else None,
                    "league": {
                        "id": player.team.league.id if player.team and player.team.league else None,
                        "name": player.team.league.name if player.team and player.team.league else None
                    } if player.team else None
                } if player.team else None
            }
        }
        
        # Estadísticas
        if include_stats and player.stats:
            stats_query = db.query(PlayerStats).filter(PlayerStats.player_id == player_id)
            if season:
                stats_query = stats_query.filter(PlayerStats.season == season)
            
            stats = stats_query.all()
            
            if stats:
                # Estadísticas por temporada
                stats_by_season = {}
                for stat in stats:
                    stats_by_season[stat.season] = {
                        "matches_played": stat.matches_played,
                        "goals": stat.goals,
                        "assists": stat.assists,
                        "yellow_cards": stat.yellow_cards,
                        "red_cards": stat.red_cards,
                        "minutes_played": stat.minutes_played,
                        "pass_accuracy": stat.pass_accuracy,
                        "shots_on_target": stat.shots_on_target,
                        "key_passes": stat.key_passes,
                        "rating": stat.rating,
                        "goals_per_game": round(stat.goals / max(stat.matches_played, 1), 2),
                        "assists_per_game": round(stat.assists / max(stat.matches_played, 1), 2)
                    }
                
                result["data"]["statistics"] = {
                    "by_season": stats_by_season,
                    "career_totals": {
                        "matches_played": sum(s.matches_played for s in stats),
                        "goals": sum(s.goals for s in stats),
                        "assists": sum(s.assists for s in stats),
                        "yellow_cards": sum(s.yellow_cards for s in stats),
                        "red_cards": sum(s.red_cards for s in stats),
                        "minutes_played": sum(s.minutes_played for s in stats)
                    }
                }
        
        # Lesiones
        if include_injuries:
            injuries = (
                db.query(PlayerInjury)
                .filter(PlayerInjury.player_id == player_id)
                .order_by(desc(PlayerInjury.injury_date))
                .limit(10)
                .all()
            )
            
            result["data"]["injuries"] = []
            current_injury = None
            
            for injury in injuries:
                injury_data = {
                    "id": injury.id,
                    "injury_type": injury.injury_type,
                    "injury_date": injury.injury_date,
                    "expected_return_date": injury.expected_return_date,
                    "actual_return_date": injury.actual_return_date,
                    "severity": injury.severity,
                    "description": injury.description,
                    "is_active": (
                        injury.injury_date <= datetime.now() and
                        (injury.expected_return_date is None or 
                         injury.expected_return_date > datetime.now()) and
                        injury.actual_return_date is None
                    )
                }
                
                if injury_data["is_active"]:
                    current_injury = injury_data
                
                result["data"]["injuries"].append(injury_data)
            
            result["data"]["current_injury"] = current_injury
        
        # Transferencias
        if include_transfers:
            transfers = (
                db.query(PlayerTransfer)
                .filter(PlayerTransfer.player_id == player_id)
                .order_by(desc(PlayerTransfer.transfer_date))
                .all()
            )
            
            result["data"]["transfers"] = [
                {
                    "id": transfer.id,
                    "from_team": transfer.from_team,
                    "to_team": transfer.to_team,
                    "transfer_date": transfer.transfer_date,
                    "transfer_fee": transfer.transfer_fee,
                    "transfer_type": transfer.transfer_type,
                    "contract_duration": transfer.contract_duration
                }
                for transfer in transfers
            ]
        
        # Partidos recientes
        if include_recent_matches:
            recent_matches = (
                db.query(Match)
                .join(MatchEvent, Match.id == MatchEvent.match_id)
                .filter(
                    MatchEvent.player_id == player_id,
                    Match.status == "finished"
                )
                .order_by(desc(Match.match_date))
                .limit(5)
                .all()
            )
            
            result["data"]["recent_matches"] = [
                {
                    "match_id": match.id,
                    "match_date": match.match_date,
                    "home_team": match.home_team.name,
                    "away_team": match.away_team.name,
                    "score": f"{match.home_score}-{match.away_score}",
                    "league": match.league.name
                }
                for match in recent_matches
            ]
        
        # Análisis de rendimiento (opcional, computacionalmente costoso)
        if include_performance_analysis:
            try:
                analyzer = PlayerAnalyzer()
                performance_data = await analyzer.analyze_player_performance(player_id, db)
                result["data"]["performance_analysis"] = performance_data
            except Exception as e:
                result["data"]["performance_analysis"] = {
                    "error": f"No se pudo generar análisis: {str(e)}"
                }
        
        result["metadata"] = {
            "include_stats": include_stats,
            "include_injuries": include_injuries,
            "include_transfers": include_transfers,
            "include_recent_matches": include_recent_matches,
            "include_performance_analysis": include_performance_analysis,
            "season_filter": season
        }
        result["timestamp"] = datetime.now().isoformat()
        
        # Cache por 10 minutos
        if not include_performance_analysis:
            await redis_client.set(cache_key, json.dumps(result), ex=600)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener jugador: {str(e)}")


@router.get("/{player_id}/stats", response_model=dict, summary="Estadísticas detalladas del jugador")
async def get_player_stats(
    player_id: int = Path(..., gt=0, description="ID del jugador"),
    season: Optional[str] = Query(None, description="Temporada específica"),
    stat_type: str = Query("all", regex="^(all|offensive|defensive|general)$", description="Tipo de estadísticas"),
    compare_seasons: bool = Query(False, description="Comparar entre temporadas"),
    db: Session = Depends(get_db)
):
    """
    Obtener estadísticas detalladas de un jugador con opciones de filtro y comparación.
    """
    try:
        # Verificar que el jugador existe
        player = db.query(Player).filter(Player.id == player_id).first()
        if not player:
            raise HTTPException(status_code=404, detail=f"Jugador con ID {player_id} no encontrado")
        
        # Query base de estadísticas
        stats_query = db.query(PlayerStats).filter(PlayerStats.player_id == player_id)
        
        if season:
            stats_query = stats_query.filter(PlayerStats.season == season)
        
        stats = stats_query.order_by(desc(PlayerStats.season)).all()
        
        if not stats:
            return {
                "success": True,
                "data": {
                    "player_id": player_id,
                    "player_name": player.name,
                    "message": "No hay estadísticas disponibles para este jugador"
                },
                "timestamp": datetime.now().isoformat()
            }
        
        result = {
            "success": True,
            "data": {
                "player_id": player_id,
                "player_name": player.name,
                "position": player.position
            }
        }
        
        # Estadísticas por temporada
        seasons_stats = {}
        for stat in stats:
            season_data = {
                "matches_played": stat.matches_played,
                "minutes_played": stat.minutes_played
            }
            
            # Estadísticas ofensivas
            if stat_type in ["all", "offensive"]:
                season_data.update({
                    "goals": stat.goals,
                    "assists": stat.assists,
                    "shots_on_target": stat.shots_on_target,
                    "key_passes": stat.key_passes,
                    "goals_per_game": round(stat.goals / max(stat.matches_played, 1), 3),
                    "assists_per_game": round(stat.assists / max(stat.matches_played, 1), 3),
                    "minutes_per_goal": round(stat.minutes_played / max(stat.goals, 1), 1) if stat.goals > 0 else None
                })
            
            # Estadísticas defensivas
            if stat_type in ["all", "defensive"]:
                season_data.update({
                    "tackles": getattr(stat, 'tackles', 0),
                    "interceptions": getattr(stat, 'interceptions', 0),
                    "clearances": getattr(stat, 'clearances', 0),
                    "blocks": getattr(stat, 'blocks', 0)
                })
            
            # Estadísticas generales
            if stat_type in ["all", "general"]:
                season_data.update({
                    "yellow_cards": stat.yellow_cards,
                    "red_cards": stat.red_cards,
                    "pass_accuracy": stat.pass_accuracy,
                    "rating": stat.rating
                })
            
            seasons_stats[stat.season] = season_data
        
        result["data"]["seasons"] = seasons_stats
        
        # Totales de carrera
        if len(stats) > 1:
            career_totals = {
                "total_seasons": len(stats),
                "total_matches": sum(s.matches_played for s in stats),
                "total_minutes": sum(s.minutes_played for s in stats),
                "total_goals": sum(s.goals for s in stats),
                "total_assists": sum(s.assists for s in stats),
                "total_yellow_cards": sum(s.yellow_cards for s in stats),
                "total_red_cards": sum(s.red_cards for s in stats),
                "career_goals_per_game": round(sum(s.goals for s in stats) / max(sum(s.matches_played for s in stats), 1), 3),
                "career_assists_per_game": round(sum(s.assists for s in stats) / max(sum(s.matches_played for s in stats), 1), 3)
            }
            result["data"]["career_totals"] = career_totals
        
        # Comparación entre temporadas
        if compare_seasons and len(stats) >= 2:
            # Comparar última temporada vs anterior
            latest_season = stats[0]
            previous_season = stats[1]
            
            comparison = {
                "latest_season": latest_season.season,
                "previous_season": previous_season.season,
                "goals_change": latest_season.goals - previous_season.goals,
                "assists_change": latest_season.assists - previous_season.assists,
                "matches_change": latest_season.matches_played - previous_season.matches_played,
                "rating_change": round((latest_season.rating or 0) - (previous_season.rating or 0), 2),
                "performance_trend": "improving" if (latest_season.rating or 0) > (previous_season.rating or 0) else "declining"
            }
            result["data"]["season_comparison"] = comparison
        
        result["metadata"] = {
            "season_filter": season,
            "stat_type": stat_type,
            "compare_seasons": compare_seasons,
            "seasons_available": len(stats)
        }
        result["timestamp"] = datetime.now().isoformat()
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener estadísticas: {str(e)}")


@router.get("/{player_id}/injuries", response_model=dict, summary="Historial de lesiones del jugador")
async def get_player_injuries(
    player_id: int = Path(..., gt=0, description="ID del jugador"),
    active_only: bool = Query(False, description="Solo lesiones activas"),
    include_recovery_analysis: bool = Query(False, description="Incluir análisis de recuperación"),
    db: Session = Depends(get_db)
):
    """
    Obtener historial completo de lesiones de un jugador.
    """
    try:
        # Verificar que el jugador existe
        player = db.query(Player).filter(Player.id == player_id).first()
        if not player:
            raise HTTPException(status_code=404, detail=f"Jugador con ID {player_id} no encontrado")
        
        # Query de lesiones
        injuries_query = db.query(PlayerInjury).filter(PlayerInjury.player_id == player_id)
        
        if active_only:
            injuries_query = injuries_query.filter(
                and_(
                    PlayerInjury.injury_date <= datetime.now(),
                    or_(
                        PlayerInjury.expected_return_date == None,
                        PlayerInjury.expected_return_date > datetime.now()
                    ),
                    PlayerInjury.actual_return_date == None
                )
            )
        
        injuries = injuries_query.order_by(desc(PlayerInjury.injury_date)).all()
        
        injuries_data = []
        current_injury = None
        total_days_injured = 0
        
        for injury in injuries:
            # Calcular días de lesión
            if injury.actual_return_date:
                days_injured = (injury.actual_return_date - injury.injury_date.date()).days
            elif injury.expected_return_date:
                days_injured = (injury.expected_return_date - injury.injury_date.date()).days
            else:
                days_injured = (date.today() - injury.injury_date.date()).days
            
            total_days_injured += days_injured
            
            is_active = (
                injury.injury_date <= datetime.now() and
                (injury.expected_return_date is None or injury.expected_return_date > datetime.now().date()) and
                injury.actual_return_date is None
            )
            
            injury_data = {
                "id": injury.id,
                "injury_type": injury.injury_type,
                "injury_date": injury.injury_date,
                "expected_return_date": injury.expected_return_date,
                "actual_return_date": injury.actual_return_date,
                "severity": injury.severity,
                "description": injury.description,
                "days_injured": days_injured,
                "is_active": is_active,
                "recovery_status": "active" if is_active else "recovered"
            }
            
            if is_active:
                current_injury = injury_data
            
            injuries_data.append(injury_data)
        
        # Análisis de lesiones
        injury_analysis = {
            "total_injuries": len(injuries),
            "active_injuries": len([i for i in injuries_data if i["is_active"]]),
            "total_days_injured": total_days_injured,
            "average_injury_duration": round(total_days_injured / max(len(injuries), 1), 1),
            "most_common_injury": None,
            "injury_frequency": {
                "per_year": round(len(injuries) / max((date.today() - min(i.injury_date.date() for i in injuries)).days / 365, 1), 2) if injuries else 0
            }
        }
        
        # Tipo de lesión más común
        if injuries:
            injury_types = {}
            for injury in injuries:
                injury_type = injury.injury_type
                injury_types[injury_type] = injury_types.get(injury_type, 0) + 1
            
            if injury_types:
                most_common = max(injury_types.items(), key=lambda x: x[1])
                injury_analysis["most_common_injury"] = {
                    "type": most_common[0],
                    "count": most_common[1]
                }
                injury_analysis["injury_types_breakdown"] = injury_types
        
        result = {
            "success": True,
            "data": {
                "player_id": player_id,
                "player_name": player.name,
                "current_injury": current_injury,
                "injuries": injuries_data,
                "analysis": injury_analysis
            },
            "metadata": {
                "active_only": active_only,
                "include_recovery_analysis": include_recovery_analysis
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Análisis de recuperación (opcional)
        if include_recovery_analysis and injuries:
            try:
                analyzer = PlayerAnalyzer()
                recovery_data = await analyzer.analyze_injury_recovery_patterns(player_id, db)
                result["data"]["recovery_analysis"] = recovery_data
            except Exception as e:
                result["data"]["recovery_analysis"] = {
                    "error": f"No se pudo generar análisis de recuperación: {str(e)}"
                }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener lesiones: {str(e)}")


@router.get("/{player_id}/performance", response_model=dict, summary="Análisis de rendimiento del jugador")
async def get_player_performance(
    player_id: int = Path(..., gt=0, description="ID del jugador"),
    period: str = Query("season", regex="^(season|last_10|last_5|month)$", description="Período de análisis"),
    include_predictions: bool = Query(False, description="Incluir predicciones de rendimiento"),
    compare_with_team: bool = Query(False, description="Comparar con promedio del equipo"),
    db: Session = Depends(get_db)
):
    """
    Análisis detallado de rendimiento del jugador con métricas avanzadas.
    """
    try:
        # Verificar que el jugador existe
        player = db.query(Player).filter(Player.id == player_id).first()
        if not player:
            raise HTTPException(status_code=404, detail=f"Jugador con ID {player_id} no encontrado")
        
        # Usar el analizador de rendimiento
        analyzer = PlayerAnalyzer()
        
        # Obtener datos de rendimiento según el período
        if period == "season":
            performance_data = await analyzer.analyze_season_performance(player_id, db)
        elif period == "last_10":
            performance_data = await analyzer.analyze_recent_performance(player_id, 10, db)
        elif period == "last_5":
            performance_data = await analyzer.analyze_recent_performance(player_id, 5, db)
        elif period == "month":
            performance_data = await analyzer.analyze_monthly_performance(player_id, db)
        
        result = {
            "success": True,
            "data": {
                "player_id": player_id,
                "player_name": player.name,
                "period": period,
                "performance": performance_data
            }
        }
        
        # Predicciones de rendimiento
        if include_predictions:
            try:
                predictions = await analyzer.predict_future_performance(player_id, db)
                result["data"]["predictions"] = predictions
            except Exception as e:
                result["data"]["predictions"] = {
                    "error": f"No se pudieron generar predicciones: {str(e)}"
                }
        
        # Comparación con equipo
        if compare_with_team and player.team_id:
            try:
                team_comparison = await analyzer.compare_with_team_average(player_id, player.team_id, db)
                result["data"]["team_comparison"] = team_comparison
            except Exception as e:
                result["data"]["team_comparison"] = {
                    "error": f"No se pudo comparar con el equipo: {str(e)}"
                }
        
        result["metadata"] = {
            "period": period,
            "include_predictions": include_predictions,
            "compare_with_team": compare_with_team
        }
        result["timestamp"] = datetime.now().isoformat()
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al analizar rendimiento: {str(e)}")


@router.get("/search", response_model=dict, summary="Búsqueda avanzada de jugadores")
async def search_players(
    q: str = Query(..., min_length=2, description="Término de búsqueda"),
    filters: str = Query("", description="Filtros JSON adicionales"),
    limit: int = Query(20, ge=1, le=100, description="Límite de resultados"),
    include_suggestions: bool = Query(True, description="Incluir sugerencias"),
    db: Session = Depends(get_db)
):
    """
    Búsqueda avanzada de jugadores con filtros dinámicos y sugerencias.
    """
    try:
        # Parsear filtros adicionales
        additional_filters = {}
        if filters:
            try:
                additional_filters = json.loads(filters)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Filtros JSON inválidos")
        
        # Query base
        query = (
            db.query(Player)
            .options(
                joinedload(Player.team).joinedload(Team.league)
            )
            .filter(Player.is_active == True)
        )
        
        # Búsqueda por texto
        search_term = f"%{q.lower()}%"
        text_filter = or_(
            func.lower(Player.name).like(search_term),
            func.lower(Player.full_name).like(search_term),
            func.lower(Player.nationality).like(search_term)
        )
        
        query = query.filter(text_filter)
        
        # Aplicar filtros adicionales
        for key, value in additional_filters.items():
            if key == "position" and value:
                query = query.filter(Player.position.ilike(f"%{value}%"))
            elif key == "team_id" and value:
                query = query.filter(Player.team_id == value)
            elif key == "league_id" and value:
                query = query.join(Team).filter(Team.league_id == value)
            elif key == "nationality" and value:
                query = query.filter(Player.nationality.ilike(f"%{value}%"))
            elif key == "age_min" and value:
                max_birth_date = date.today() - timedelta(days=int(value) * 365)
                query = query.filter(Player.birth_date <= max_birth_date)
            elif key == "age_max" and value:
                min_birth_date = date.today() - timedelta(days=int(value) * 365)
                query = query.filter(Player.birth_date >= min_birth_date)
        
        # Ordenar por relevancia (nombre exacto primero)
        query = query.order_by(
            func.lower(Player.name).like(f"{q.lower()}%").desc(),
            Player.market_value.desc().nullslast(),
            Player.name
        )
        
        # Limitar resultados
        players = query.limit(limit).all()
        
        # Procesar resultados
        results = []
        for player in players:
            age = None
            if player.birth_date:
                age = (date.today() - player.birth_date).days // 365
            
            results.append({
                "id": player.id,
                "name": player.name,
                "full_name": player.full_name,
                "age": age,
                "nationality": player.nationality,
                "position": player.position,
                "jersey_number": player.jersey_number,
                "market_value": player.market_value,
                "team": {
                    "id": player.team.id if player.team else None,
                    "name": player.team.name if player.team else None,
                    "league_name": player.team.league.name if player.team and player.team.league else None
                } if player.team else None,
                "photo_url": player.photo_url,
                "relevance_score": 100 if player.name.lower().startswith(q.lower()) else 50
            })
        
        result = {
            "success": True,
            "data": {
                "query": q,
                "results": results,
                "total_found": len(results),
                "filters_applied": additional_filters
            }
        }
        
        # Sugerencias si no hay muchos resultados
        if include_suggestions and len(results) < 5:
            # Búsqueda más amplia para sugerencias
            suggestion_query = (
                db.query(Player)
                .options(joinedload(Player.team))
                .filter(
                    Player.is_active == True,
                    or_(
                        func.lower(Player.name).like(f"%{q.lower()}%"),
                        func.lower(Player.nationality).like(f"%{q.lower()}%"),
                        func.lower(Player.position).like(f"%{q.lower()}%")
                    )
                )
                .limit(10)
                .all()
            )
            
            suggestions = [
                {
                    "id": p.id,
                    "name": p.name,
                    "team_name": p.team.name if p.team else None,
                    "position": p.position
                }
                for p in suggestion_query if p.id not in [r["id"] for r in results]
            ]
            
            result["data"]["suggestions"] = suggestions
        
        result["timestamp"] = datetime.now().isoformat()
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en búsqueda: {str(e)}")


@router.get("/top", response_model=dict, summary="Top jugadores por categoría")
async def get_top_players(
    category: str = Query("rating", regex="^(rating|goals|assists|market_value|young_talents)$", description="Categoría de ranking"),
    position: Optional[str] = Query(None, description="Filtrar por posición"),
    league_id: Optional[int] = Query(None, description="Filtrar por liga"),
    season: Optional[str] = Query(None, description="Temporada específica"),
    limit: int = Query(20, ge=1, le=100, description="Número de jugadores"),
    db: Session = Depends(get_db)
):
    """
    Obtener ranking de top jugadores por diferentes categorías.
    """
    try:
        # Cache key
        cache_key = f"top_players:{category}:{position}:{league_id}:{season}:{limit}"
        
        # Verificar cache
        cached_result = await redis_client.get(cache_key)
        if cached_result:
            return json.loads(cached_result)
        
        if category == "young_talents":
            # Jugadores jóvenes prometedores (menores de 23 años)
            max_birth_date = date.today() - timedelta(days=23 * 365)
            query = (
                db.query(Player)
                .options(
                    joinedload(Player.team).joinedload(Team.league),
                    joinedload(Player.stats)
                )
                .filter(
                    Player.is_active == True,
                    Player.birth_date >= max_birth_date,
                    Player.market_value.isnot(None)
                )
                .order_by(desc(Player.market_value))
            )
        else:
            # Query base para estadísticas
            query = (
                db.query(Player)
                .join(PlayerStats)
                .options(
                    joinedload(Player.team).joinedload(Team.league)
                )
                .filter(Player.is_active == True)
            )
            
            # Filtro de temporada
            if season:
                query = query.filter(PlayerStats.season == season)
            else:
                # Última temporada disponible
                latest_season = db.query(func.max(PlayerStats.season)).scalar()
                if latest_season:
                    query = query.filter(PlayerStats.season == latest_season)
            
            # Ordenamiento por categoría
            if category == "rating":
                query = query.filter(PlayerStats.rating.isnot(None)).order_by(desc(PlayerStats.rating))
            elif category == "goals":
                query = query.order_by(desc(PlayerStats.goals))
            elif category == "assists":
                query = query.order_by(desc(PlayerStats.assists))
            elif category == "market_value":
                query = query.filter(Player.market_value.isnot(None)).order_by(desc(Player.market_value))
        
        # Filtros adicionales
        if position:
            query = query.filter(Player.position.ilike(f"%{position}%"))
        
        if league_id:
            query = query.join(Team).filter(Team.league_id == league_id)
        
        # Obtener jugadores
        players = query.limit(limit).all()
        
        # Procesar resultados
        top_players = []
        for rank, player in enumerate(players, 1):
            age = None
            if player.birth_date:
                age = (date.today() - player.birth_date).days // 365
            
            player_data = {
                "rank": rank,
                "id": player.id,
                "name": player.name,
                "age": age,
                "nationality": player.nationality,
                "position": player.position,
                "team": {
                    "id": player.team.id if player.team else None,
                    "name": player.team.name if player.team else None,
                    "league_name": player.team.league.name if player.team and player.team.league else None
                } if player.team else None,
                "market_value": player.market_value,
                "photo_url": player.photo_url
            }
            
            # Agregar estadística específica de la categoría
            if hasattr(player, 'stats') and player.stats:
                latest_stat = sorted(player.stats, key=lambda x: x.season, reverse=True)[0]
                if category == "rating":
                    player_data["rating"] = latest_stat.rating
                elif category == "goals":
                    player_data["goals"] = latest_stat.goals
                    player_data["goals_per_game"] = round(latest_stat.goals / max(latest_stat.matches_played, 1), 2)
                elif category == "assists":
                    player_data["assists"] = latest_stat.assists
                    player_data["assists_per_game"] = round(latest_stat.assists / max(latest_stat.matches_played, 1), 2)
            
            top_players.append(player_data)
        
        result = {
            "success": True,
            "data": {
                "category": category,
                "top_players": top_players,
                "filters": {
                    "position": position,
                    "league_id": league_id,
                    "season": season,
                    "limit": limit
                }
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache por 1 hora
        await redis_client.set(cache_key, json.dumps(result), ex=3600)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener top jugadores: {str(e)}")


# =====================================================
# ENDPOINTS DE COMPARACIÓN Y ANÁLISIS
# =====================================================

@router.post("/compare", response_model=dict, summary="Comparar jugadores")
async def compare_players(
    player_ids: List[int] = Query(..., description="IDs de jugadores a comparar"),
    comparison_type: str = Query("stats", regex="^(stats|performance|market_value)$", description="Tipo de comparación"),
    season: Optional[str] = Query(None, description="Temporada para comparar"),
    db: Session = Depends(get_db)
):
    """
    Comparar estadísticas y rendimiento entre múltiples jugadores.
    """
    try:
        if len(player_ids) < 2:
            raise HTTPException(status_code=400, detail="Se necesitan al menos 2 jugadores para comparar")
        
        if len(player_ids) > 5:
            raise HTTPException(status_code=400, detail="Máximo 5 jugadores para comparar")
        
        # Obtener jugadores
        players = (
            db.query(Player)
            .options(
                joinedload(Player.team).joinedload(Team.league),
                joinedload(Player.stats)
            )
            .filter(Player.id.in_(player_ids))
            .all()
        )
        
        if len(players) != len(player_ids):
            missing_ids = set(player_ids) - {p.id for p in players}
            raise HTTPException(status_code=404, detail=f"Jugadores no encontrados: {missing_ids}")
        
        # Preparar datos de comparación
        comparison_data = {
            "players": [],
            "comparison_metrics": {}
        }
        
        for player in players:
            age = None
            if player.birth_date:
                age = (date.today() - player.birth_date).days // 365
            
            player_data = {
                "id": player.id,
                "name": player.name,
                "age": age,
                "position": player.position,
                "nationality": player.nationality,
                "team": player.team.name if player.team else None,
                "league": player.team.league.name if player.team and player.team.league else None,
                "market_value": player.market_value
            }
            
            # Agregar estadísticas según el tipo de comparación
            if comparison_type == "stats" and player.stats:
                stats_query = [s for s in player.stats]
                if season:
                    stats_query = [s for s in stats_query if s.season == season]
                
                if stats_query:
                    latest_stat = sorted(stats_query, key=lambda x: x.season, reverse=True)[0]
                    player_data["stats"] = {
                        "season": latest_stat.season,
                        "matches_played": latest_stat.matches_played,
                        "goals": latest_stat.goals,
                        "assists": latest_stat.assists,
                        "rating": latest_stat.rating,
                        "goals_per_game": round(latest_stat.goals / max(latest_stat.matches_played, 1), 3),
                        "assists_per_game": round(latest_stat.assists / max(latest_stat.matches_played, 1), 3)
                    }
            
            comparison_data["players"].append(player_data)
        
        # Calcular métricas de comparación
        if comparison_type == "stats":
            # Encontrar mejor en cada categoría
            metrics = ["goals", "assists", "rating", "goals_per_game", "assists_per_game"]
            for metric in metrics:
                values = [(p["name"], p["stats"].get(metric, 0)) for p in comparison_data["players"] if "stats" in p and p["stats"].get(metric) is not None]
                if values:
                    best = max(values, key=lambda x: x[1])
                    comparison_data["comparison_metrics"][f"best_{metric}"] = {
                        "player": best[0],
                        "value": best[1]
                    }
        
        elif comparison_type == "market_value":
            values = [(p["name"], p["market_value"]) for p in comparison_data["players"] if p["market_value"]]
            if values:
                best = max(values, key=lambda x: x[1])
                comparison_data["comparison_metrics"]["highest_market_value"] = {
                    "player": best[0],
                    "value": best[1]
                }
        
        result = {
            "success": True,
            "data": comparison_data,
            "metadata": {
                "comparison_type": comparison_type,
                "season": season,
                "players_compared": len(players)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al comparar jugadores: {str(e)}")