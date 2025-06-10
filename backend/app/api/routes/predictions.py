"""
API Routes para predicciones de fútbol usando Machine Learning
Incluye predicciones de resultados, goles, mercados de apuestas y análisis probabilístico
"""

import json
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Path,
    Query,
)
from pydantic import BaseModel, Field
from sqlalchemy import or_
from sqlalchemy.orm import Session, joinedload

# from app.core.celery_app import celery_app  # TODO: implement celery
# from app.core.redis_client import redis_client  # TODO: implement redis client
from app.database.connection import get_db
from app.database.models import Match, PredictionHistory, Team
from app.services.advanced_predictor import AdvancedFootballPredictor
from app.services.betting_calculator import BettingCalculator
from app.services.ml_analyzer import MLAnalyzer

router = APIRouter()

# =====================================================
# MODELOS PYDANTIC (SCHEMAS)
# =====================================================


class PredictionType(str, Enum):
    MATCH_RESULT = "match_result"  # 1X2
    OVER_UNDER = "over_under"  # Over/Under 2.5 goles
    BOTH_TEAMS_SCORE = "both_teams_score"  # BTTS
    CORRECT_SCORE = "correct_score"
    TOTAL_GOALS = "total_goals"
    FIRST_HALF = "first_half"
    PLAYER_GOALS = "player_goals"
    CARDS = "cards"


class MatchPredictionRequest(BaseModel):
    home_team_id: int = Field(..., gt=0, description="ID del equipo local")
    away_team_id: int = Field(..., gt=0, description="ID del equipo visitante")
    match_date: Optional[datetime] = Field(
        None, description="Fecha del partido"
    )
    league_id: Optional[int] = Field(None, description="ID de la liga")
    stadium_id: Optional[int] = Field(None, description="ID del estadio")
    prediction_types: List[PredictionType] = Field(
        default=[PredictionType.MATCH_RESULT, PredictionType.OVER_UNDER],
        description="Tipos de predicciones a generar",
    )
    include_explanation: bool = Field(
        True, description="Incluir explicación del modelo"
    )
    confidence_threshold: float = Field(
        0.6, ge=0.0, le=1.0, description="Umbral de confianza"
    )


class PredictionResponse(BaseModel):
    prediction_id: str
    match_info: Dict[str, Any]
    predictions: Dict[str, Any]
    confidence_scores: Dict[str, float]
    explanation: Optional[Dict[str, Any]] = None
    betting_recommendations: Optional[Dict[str, Any]] = None
    timestamp: datetime


class BulkPredictionRequest(BaseModel):
    match_ids: List[int] = Field(
        ..., max_items=50, description="IDs de partidos"
    )
    prediction_types: List[PredictionType] = Field(
        default=[PredictionType.MATCH_RESULT],
        description="Tipos de predicciones",
    )
    include_odds_analysis: bool = Field(
        False, description="Incluir análisis de cuotas"
    )


# =====================================================
# ENDPOINTS PRINCIPALES DE PREDICCIONES
# =====================================================


@router.post(
    "/match", response_model=dict, summary="Predicción para partido específico"
)
async def predict_match(
    request: MatchPredictionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Generar predicciones ML para un partido específico.

    Características:
    - Múltiples tipos de predicciones (1X2, Over/Under, BTTS, etc.)
    - Análisis probabilístico avanzado
    - Explicabilidad con SHAP/LIME
    - Recomendaciones de apuestas
    - Cálculo de confianza del modelo
    """
    try:
        # Validar que los equipos existen
        home_team = (
            db.query(Team).filter(Team.id == request.home_team_id).first()
        )
        away_team = (
            db.query(Team).filter(Team.id == request.away_team_id).first()
        )

        if not home_team or not away_team:
            raise HTTPException(
                status_code=404, detail="Uno o ambos equipos no encontrados"
            )

        if home_team.id == away_team.id:
            raise HTTPException(
                status_code=400,
                detail="No se puede hacer predicción del mismo equipo",
            )

        # Cache key para la predicción
        cache_key = f"prediction:{request.home_team_id}:{request.away_team_id}:{request.match_date}"

        # Verificar cache
        cached_prediction = await redis_client.get(cache_key)
        if cached_prediction:
            return json.loads(cached_prediction)

        # Inicializar predictor avanzado
        predictor = AdvancedFootballPredictor()

        # Preparar datos del partido
        match_data = {
            "home_team_id": request.home_team_id,
            "away_team_id": request.away_team_id,
            "home_team_name": home_team.name,
            "away_team_name": away_team.name,
            "league_id": request.league_id or home_team.league_id,
            "match_date": request.match_date or datetime.now(),
            "stadium_id": request.stadium_id,
        }

        # Generar predicciones para cada tipo solicitado
        predictions = {}
        confidence_scores = {}
        explanations = {}

        for pred_type in request.prediction_types:
            try:
                # Generar predicción específica
                if pred_type == PredictionType.MATCH_RESULT:
                    result = await predictor.predict_match_result(
                        match_data, db
                    )
                    predictions["match_result"] = {
                        "home_win": result["probabilities"]["home_win"],
                        "draw": result["probabilities"]["draw"],
                        "away_win": result["probabilities"]["away_win"],
                        "predicted_outcome": result["predicted_outcome"],
                        "expected_goals_home": result.get(
                            "expected_goals_home"
                        ),
                        "expected_goals_away": result.get(
                            "expected_goals_away"
                        ),
                    }
                    confidence_scores["match_result"] = result["confidence"]

                elif pred_type == PredictionType.OVER_UNDER:
                    result = await predictor.predict_over_under(
                        match_data, 2.5, db
                    )
                    predictions["over_under_2_5"] = {
                        "over_probability": result["over_probability"],
                        "under_probability": result["under_probability"],
                        "predicted_total_goals": result[
                            "predicted_total_goals"
                        ],
                        "recommendation": (
                            "over"
                            if result["over_probability"] > 0.5
                            else "under"
                        ),
                    }
                    confidence_scores["over_under_2_5"] = result["confidence"]

                elif pred_type == PredictionType.BOTH_TEAMS_SCORE:
                    result = await predictor.predict_both_teams_score(
                        match_data, db
                    )
                    predictions["both_teams_score"] = {
                        "btts_yes_probability": result["btts_probability"],
                        "btts_no_probability": 1 - result["btts_probability"],
                        "recommendation": (
                            "yes" if result["btts_probability"] > 0.5 else "no"
                        ),
                    }
                    confidence_scores["both_teams_score"] = result[
                        "confidence"
                    ]

                elif pred_type == PredictionType.CORRECT_SCORE:
                    result = await predictor.predict_correct_score(
                        match_data, db
                    )
                    predictions["correct_score"] = {
                        "most_likely_scores": result["top_scores"],
                        "score_probabilities": result["score_matrix"],
                    }
                    confidence_scores["correct_score"] = result["confidence"]

                elif pred_type == PredictionType.TOTAL_GOALS:
                    result = await predictor.predict_total_goals_distribution(
                        match_data, db
                    )
                    predictions["total_goals"] = {
                        "goals_distribution": result["distribution"],
                        "expected_goals": result["expected_total"],
                        "most_likely_total": result["most_likely"],
                    }
                    confidence_scores["total_goals"] = result["confidence"]

                elif pred_type == PredictionType.FIRST_HALF:
                    result = await predictor.predict_first_half(match_data, db)
                    predictions["first_half"] = {
                        "ht_result_probabilities": result["ht_probabilities"],
                        "ht_goals_over_0_5": result["ht_goals_over_0_5"],
                        "ht_goals_over_1_5": result["ht_goals_over_1_5"],
                    }
                    confidence_scores["first_half"] = result["confidence"]

                # Explicación del modelo (si se solicita)
                if request.include_explanation:
                    explanation = await predictor.explain_prediction(
                        pred_type.value, match_data, db
                    )
                    explanations[pred_type.value] = explanation

            except Exception as e:
                # Si falla una predicción específica, continuar con las demás
                predictions[pred_type.value] = {
                    "error": f"No se pudo generar predicción: {str(e)}"
                }
                confidence_scores[pred_type.value] = 0.0

        # Calcular confianza general
        valid_confidences = [
            score for score in confidence_scores.values() if score > 0
        ]
        overall_confidence = (
            sum(valid_confidences) / len(valid_confidences)
            if valid_confidences
            else 0.0
        )

        # Generar recomendaciones de apuestas
        betting_recommendations = None
        if overall_confidence >= request.confidence_threshold:
            try:
                calculator = BettingCalculator()
                betting_recommendations = (
                    await calculator.generate_betting_recommendations(
                        predictions, confidence_scores, match_data, db
                    )
                )
            except Exception as e:
                betting_recommendations = {
                    "error": f"No se pudieron generar recomendaciones: {str(e)}"
                }

        # ID único para la predicción
        prediction_id = f"pred_{request.home_team_id}_{request.away_team_id}_{int(datetime.now().timestamp())}"

        # Construir respuesta
        result = {
            "success": True,
            "data": {
                "prediction_id": prediction_id,
                "match_info": match_data,
                "predictions": predictions,
                "confidence_scores": confidence_scores,
                "overall_confidence": round(overall_confidence, 3),
                "explanation": (
                    explanations if request.include_explanation else None
                ),
                "betting_recommendations": betting_recommendations,
                "model_version": predictor.get_model_version(),
                "factors_considered": [
                    "form_últimos_5_partidos",
                    "head_to_head_histórico",
                    "estadísticas_casa_visitante",
                    "lesiones_suspensiones",
                    "importancia_partido",
                    "condiciones_climáticas",
                    "factor_estadio",
                ],
            },
            "metadata": {
                "prediction_types": [
                    pt.value for pt in request.prediction_types
                ],
                "confidence_threshold": request.confidence_threshold,
                "processing_time_ms": None,  # Se calculará al final
            },
            "timestamp": datetime.now().isoformat(),
        }

        # Guardar predicción en base de datos (task en background)
        background_tasks.add_task(
            save_prediction_to_db,
            prediction_id,
            match_data,
            predictions,
            confidence_scores,
            db,
        )

        # Guardar en cache (30 minutos)
        await redis_client.set(cache_key, json.dumps(result), ex=1800)

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error al generar predicción: {str(e)}"
        )


@router.post("/bulk", response_model=dict, summary="Predicciones en lote")
async def predict_bulk_matches(
    request: BulkPredictionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Generar predicciones para múltiples partidos de forma eficiente.
    """
    try:
        if len(request.match_ids) == 0:
            raise HTTPException(
                status_code=400, detail="Lista de partidos vacía"
            )

        # Obtener partidos
        matches = (
            db.query(Match)
            .options(
                joinedload(Match.home_team),
                joinedload(Match.away_team),
                joinedload(Match.league),
            )
            .filter(Match.id.in_(request.match_ids))
            .all()
        )

        if len(matches) != len(request.match_ids):
            found_ids = {m.id for m in matches}
            missing_ids = set(request.match_ids) - found_ids
            raise HTTPException(
                status_code=404,
                detail=f"Partidos no encontrados: {list(missing_ids)}",
            )

        # Procesar predicciones en lotes para eficiencia
        predictor = AdvancedFootballPredictor()
        batch_results = []

        # Procesar en lotes de 10
        for i in range(0, len(matches), 10):
            batch = matches[i : i + 10]
            batch_predictions = await predictor.predict_batch(
                batch, request.prediction_types, db
            )
            batch_results.extend(batch_predictions)

        # Análisis de cuotas (si se solicita)
        odds_analysis = None
        if request.include_odds_analysis:
            try:
                calculator = BettingCalculator()
                odds_analysis = await calculator.analyze_bulk_odds(
                    batch_results, db
                )
            except Exception as e:
                odds_analysis = {
                    "error": f"Error en análisis de cuotas: {str(e)}"
                }

        # Estadísticas del lote
        total_predictions = len(batch_results)
        high_confidence = len(
            [r for r in batch_results if r.get("overall_confidence", 0) > 0.7]
        )
        medium_confidence = len(
            [
                r
                for r in batch_results
                if 0.5 <= r.get("overall_confidence", 0) <= 0.7
            ]
        )

        result = {
            "success": True,
            "data": {
                "predictions": batch_results,
                "batch_statistics": {
                    "total_matches": total_predictions,
                    "high_confidence_predictions": high_confidence,
                    "medium_confidence_predictions": medium_confidence,
                    "low_confidence_predictions": total_predictions
                    - high_confidence
                    - medium_confidence,
                },
                "odds_analysis": odds_analysis,
            },
            "metadata": {
                "requested_matches": len(request.match_ids),
                "processed_matches": total_predictions,
                "prediction_types": [
                    pt.value for pt in request.prediction_types
                ],
            },
            "timestamp": datetime.now().isoformat(),
        }

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error en predicciones en lote: {str(e)}"
        )


@router.get(
    "/match/{match_id}",
    response_model=dict,
    summary="Predicción para partido existente",
)
async def get_match_prediction(
    match_id: int = Path(..., gt=0, description="ID del partido"),
    prediction_types: List[PredictionType] = Query(
        default=[PredictionType.MATCH_RESULT, PredictionType.OVER_UNDER],
        description="Tipos de predicciones",
    ),
    include_explanation: bool = Query(True, description="Incluir explicación"),
    force_refresh: bool = Query(False, description="Forzar recálculo"),
    db: Session = Depends(get_db),
):
    """
    Obtener predicción para un partido que ya existe en la base de datos.
    """
    try:
        # Obtener partido
        match = (
            db.query(Match)
            .options(
                joinedload(Match.home_team),
                joinedload(Match.away_team),
                joinedload(Match.league),
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

        # Cache key
        cache_key = f"match_prediction:{match_id}:{','.join([pt.value for pt in prediction_types])}"

        # Verificar cache (a menos que se fuerce refresh)
        if not force_refresh:
            cached_result = await redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)

        # Preparar datos del partido
        match_data = {
            "match_id": match.id,
            "home_team_id": match.home_team_id,
            "away_team_id": match.away_team_id,
            "home_team_name": match.home_team.name,
            "away_team_name": match.away_team.name,
            "league_id": match.league_id,
            "league_name": match.league.name if match.league else None,
            "match_date": match.match_date,
            "stadium_id": match.stadium_id,
            "stadium_name": match.stadium.name if match.stadium else None,
            "status": match.status,
        }

        # Si el partido ya terminó, incluir resultado real
        if match.status == "finished":
            match_data.update(
                {
                    "actual_result": {
                        "home_score": match.home_score,
                        "away_score": match.away_score,
                        "total_goals": match.home_score + match.away_score,
                        "winner": (
                            "home"
                            if match.home_score > match.away_score
                            else (
                                "away"
                                if match.away_score > match.home_score
                                else "draw"
                            )
                        ),
                    }
                }
            )

        # Generar predicciones
        predictor = AdvancedFootballPredictor()
        predictions_result = await predictor.predict_multiple_types(
            match_data, prediction_types, include_explanation, db
        )

        # Si el partido ya terminó, calcular precisión
        accuracy_analysis = None
        if match.status == "finished":
            try:
                accuracy_analysis = (
                    await predictor.calculate_prediction_accuracy(
                        predictions_result["predictions"],
                        match_data["actual_result"],
                    )
                )
            except Exception as e:
                accuracy_analysis = {
                    "error": f"Error calculando precisión: {str(e)}"
                }

        result = {
            "success": True,
            "data": {
                "match_info": match_data,
                "predictions": predictions_result["predictions"],
                "confidence_scores": predictions_result["confidence_scores"],
                "overall_confidence": predictions_result["overall_confidence"],
                "explanation": (
                    predictions_result.get("explanation")
                    if include_explanation
                    else None
                ),
                "accuracy_analysis": accuracy_analysis,
                "model_version": predictor.get_model_version(),
            },
            "metadata": {
                "match_id": match_id,
                "prediction_types": [pt.value for pt in prediction_types],
                "include_explanation": include_explanation,
                "force_refresh": force_refresh,
                "is_finished": match.status == "finished",
            },
            "timestamp": datetime.now().isoformat(),
        }

        # Cache por 1 hora (más tiempo si el partido ya terminó)
        cache_time = 21600 if match.status == "finished" else 3600  # 6h vs 1h
        await redis_client.set(cache_key, json.dumps(result), ex=cache_time)

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener predicción del partido: {str(e)}",
        )


@router.get(
    "/upcoming",
    response_model=dict,
    summary="Predicciones para próximos partidos",
)
async def get_upcoming_predictions(
    league_id: Optional[int] = Query(None, description="Filtrar por liga"),
    team_id: Optional[int] = Query(None, description="Filtrar por equipo"),
    days_ahead: int = Query(7, ge=1, le=30, description="Días hacia adelante"),
    min_confidence: float = Query(
        0.6, ge=0.0, le=1.0, description="Confianza mínima"
    ),
    limit: int = Query(20, ge=1, le=100, description="Límite de partidos"),
    include_betting_tips: bool = Query(
        False, description="Incluir tips de apuestas"
    ),
    db: Session = Depends(get_db),
):
    """
    Obtener predicciones para próximos partidos con filtros.
    """
    try:
        # Fechas para filtrar
        start_date = datetime.now()
        end_date = start_date + timedelta(days=days_ahead)

        # Query base de partidos próximos
        query = (
            db.query(Match)
            .options(
                joinedload(Match.home_team),
                joinedload(Match.away_team),
                joinedload(Match.league),
            )
            .filter(
                Match.match_date.between(start_date, end_date),
                Match.status == "scheduled",
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
        matches = query.order_by(Match.match_date).limit(limit).all()

        if not matches:
            return {
                "success": True,
                "data": {
                    "message": "No hay partidos próximos que coincidan con los filtros",
                    "matches": [],
                },
                "timestamp": datetime.now().isoformat(),
            }

        # Generar predicciones para todos los partidos
        predictor = AdvancedFootballPredictor()
        predictions_with_confidence = []

        for match in matches:
            try:
                match_data = {
                    "match_id": match.id,
                    "home_team_id": match.home_team_id,
                    "away_team_id": match.away_team_id,
                    "home_team_name": match.home_team.name,
                    "away_team_name": match.away_team.name,
                    "league_id": match.league_id,
                    "match_date": match.match_date,
                }

                # Predicciones básicas (rápidas)
                prediction_result = await predictor.predict_multiple_types(
                    match_data,
                    [PredictionType.MATCH_RESULT, PredictionType.OVER_UNDER],
                    include_explanation=False,
                    db=db,
                )

                # Solo incluir si cumple el umbral de confianza
                if prediction_result["overall_confidence"] >= min_confidence:
                    match_prediction = {
                        "match": {
                            "id": match.id,
                            "home_team": match.home_team.name,
                            "away_team": match.away_team.name,
                            "league": (
                                match.league.name if match.league else None
                            ),
                            "match_date": match.match_date,
                            "importance": getattr(
                                match, "importance", "medium"
                            ),
                        },
                        "predictions": prediction_result["predictions"],
                        "confidence": prediction_result["overall_confidence"],
                        "top_pick": None,
                    }

                    # Determinar la mejor apuesta
                    best_confidence = 0
                    best_pick = None

                    for pred_type, pred_data in prediction_result[
                        "predictions"
                    ].items():
                        if pred_type == "match_result":
                            probs = pred_data
                            max_prob = max(
                                probs["home_win"],
                                probs["draw"],
                                probs["away_win"],
                            )
                            if max_prob > best_confidence:
                                best_confidence = max_prob
                                if max_prob == probs["home_win"]:
                                    best_pick = (
                                        f"Victoria {match.home_team.name}"
                                    )
                                elif max_prob == probs["away_win"]:
                                    best_pick = (
                                        f"Victoria {match.away_team.name}"
                                    )
                                else:
                                    best_pick = "Empate"

                        elif pred_type == "over_under_2_5":
                            if pred_data["over_probability"] > best_confidence:
                                best_confidence = pred_data["over_probability"]
                                best_pick = "Over 2.5 goles"
                            elif (
                                pred_data["under_probability"]
                                > best_confidence
                            ):
                                best_confidence = pred_data[
                                    "under_probability"
                                ]
                                best_pick = "Under 2.5 goles"

                    match_prediction["top_pick"] = {
                        "prediction": best_pick,
                        "confidence": round(best_confidence, 3),
                    }

                    predictions_with_confidence.append(match_prediction)

            except Exception:
                # Si falla una predicción, continuar con las demás
                continue

        # Ordenar por confianza descendente
        predictions_with_confidence.sort(
            key=lambda x: x["confidence"], reverse=True
        )

        # Tips de apuestas (si se solicitan)
        betting_tips = None
        if include_betting_tips and predictions_with_confidence:
            try:
                calculator = BettingCalculator()
                betting_tips = await calculator.generate_daily_tips(
                    predictions_with_confidence, min_confidence, db
                )
            except Exception as e:
                betting_tips = {"error": f"Error generando tips: {str(e)}"}

        # Estadísticas del día
        high_confidence_count = len(
            [p for p in predictions_with_confidence if p["confidence"] > 0.8]
        )
        medium_confidence_count = len(
            [
                p
                for p in predictions_with_confidence
                if 0.6 <= p["confidence"] <= 0.8
            ]
        )

        result = {
            "success": True,
            "data": {
                "upcoming_predictions": predictions_with_confidence,
                "summary": {
                    "total_matches_analyzed": len(matches),
                    "matches_with_good_predictions": len(
                        predictions_with_confidence
                    ),
                    "high_confidence_matches": high_confidence_count,
                    "medium_confidence_matches": medium_confidence_count,
                    "average_confidence": (
                        round(
                            sum(
                                p["confidence"]
                                for p in predictions_with_confidence
                            )
                            / len(predictions_with_confidence),
                            3,
                        )
                        if predictions_with_confidence
                        else 0
                    ),
                },
                "betting_tips": betting_tips,
                "best_bets_today": predictions_with_confidence[:5],  # Top 5
            },
            "filters": {
                "league_id": league_id,
                "team_id": team_id,
                "days_ahead": days_ahead,
                "min_confidence": min_confidence,
                "limit": limit,
            },
            "timestamp": datetime.now().isoformat(),
        }

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener predicciones próximas: {str(e)}",
        )


# =====================================================
# ENDPOINTS DE ANÁLISIS Y EVALUACIÓN
# =====================================================


@router.get("/accuracy", response_model=dict, summary="Precisión del modelo")
async def get_model_accuracy(
    days_back: int = Query(
        30, ge=1, le=365, description="Días hacia atrás para análisis"
    ),
    league_id: Optional[int] = Query(None, description="Filtrar por liga"),
    prediction_type: Optional[PredictionType] = Query(
        None, description="Tipo de predicción"
    ),
    min_confidence: float = Query(
        0.0, ge=0.0, le=1.0, description="Confianza mínima"
    ),
    db: Session = Depends(get_db),
):
    """
    Analizar la precisión del modelo en predicciones pasadas.
    """
    try:
        # Fecha de corte
        cutoff_date = datetime.now() - timedelta(days=days_back)

        # Obtener predicciones históricas
        query = (
            db.query(PredictionHistory)
            .join(Match)
            .filter(
                PredictionHistory.created_at >= cutoff_date,
                Match.status == "finished",
                PredictionHistory.confidence >= min_confidence,
            )
        )

        if league_id:
            query = query.filter(Match.league_id == league_id)

        if prediction_type:
            query = query.filter(
                PredictionHistory.prediction_type == prediction_type.value
            )

        predictions = query.all()

        if not predictions:
            return {
                "success": True,
                "data": {
                    "message": "No hay predicciones históricas suficientes para el análisis",
                    "accuracy_metrics": {},
                },
                "timestamp": datetime.now().isoformat(),
            }

        # Calcular métricas de precisión
        analyzer = MLAnalyzer()
        accuracy_metrics = await analyzer.calculate_historical_accuracy(
            predictions, prediction_type, db
        )

        result = {
            "success": True,
            "data": {
                "accuracy_metrics": accuracy_metrics,
                "analysis_period": {
                    "days_analyzed": days_back,
                    "start_date": cutoff_date.isoformat(),
                    "end_date": datetime.now().isoformat(),
                    "total_predictions": len(predictions),
                },
                "performance_by_confidence": await analyzer.analyze_by_confidence_levels(
                    predictions
                ),
                "performance_by_league": (
                    await analyzer.analyze_by_league(predictions, db)
                    if not league_id
                    else None
                ),
                "model_reliability": {
                    "overall_accuracy": accuracy_metrics.get(
                        "overall_accuracy", 0
                    ),
                    "high_confidence_accuracy": accuracy_metrics.get(
                        "high_confidence_accuracy", 0
                    ),
                    "roi_simulation": accuracy_metrics.get(
                        "roi_simulation", {}
                    ),
                    "recommendation": (
                        "reliable"
                        if accuracy_metrics.get("overall_accuracy", 0) > 0.65
                        else "needs_improvement"
                    ),
                },
            },
            "filters": {
                "days_back": days_back,
                "league_id": league_id,
                "prediction_type": (
                    prediction_type.value if prediction_type else None
                ),
                "min_confidence": min_confidence,
            },
            "timestamp": datetime.now().isoformat(),
        }

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error al analizar precisión: {str(e)}"
        )


@router.get(
    "/model/performance",
    response_model=dict,
    summary="Rendimiento detallado del modelo",
)
async def get_model_performance(
    metric_type: str = Query(
        "all", regex="^(all|accuracy|calibration|profit|feature_importance)$"
    ),
    time_period: str = Query("month", regex="^(week|month|quarter|year)$"),
    include_comparison: bool = Query(
        False, description="Comparar con modelos anteriores"
    ),
    db: Session = Depends(get_db),
):
    """
    Análisis detallado del rendimiento del modelo ML.
    """
    try:
        analyzer = MLAnalyzer()
        predictor = AdvancedFootballPredictor()

        # Definir período de tiempo
        if time_period == "week":
            days_back = 7
        elif time_period == "month":
            days_back = 30
        elif time_period == "quarter":
            days_back = 90
        else:  # year
            days_back = 365

        performance_data = {}

        # Métricas de precisión
        if metric_type in ["all", "accuracy"]:
            accuracy_data = await analyzer.comprehensive_accuracy_analysis(
                days_back, db
            )
            performance_data["accuracy"] = accuracy_data

        # Calibración del modelo
        if metric_type in ["all", "calibration"]:
            calibration_data = await analyzer.model_calibration_analysis(
                days_back, db
            )
            performance_data["calibration"] = calibration_data

        # Análisis de rentabilidad
        if metric_type in ["all", "profit"]:
            profit_data = await analyzer.profit_loss_analysis(days_back, db)
            performance_data["profit"] = profit_data

        # Importancia de características
        if metric_type in ["all", "feature_importance"]:
            feature_data = await predictor.get_feature_importance()
            performance_data["feature_importance"] = feature_data

        # Comparación con versiones anteriores
        comparison_data = None
        if include_comparison:
            try:
                comparison_data = await analyzer.compare_model_versions(
                    days_back, db
                )
            except Exception as e:
                comparison_data = {"error": f"Error en comparación: {str(e)}"}

        # Recomendaciones de mejora
        recommendations = await analyzer.generate_improvement_recommendations(
            performance_data, days_back, db
        )

        result = {
            "success": True,
            "data": {
                "model_info": {
                    "version": predictor.get_model_version(),
                    "last_trained": predictor.get_last_training_date(),
                    "features_count": predictor.get_features_count(),
                    "training_samples": predictor.get_training_samples_count(),
                },
                "performance_metrics": performance_data,
                "model_comparison": comparison_data,
                "recommendations": recommendations,
                "analysis_summary": {
                    "period_analyzed": time_period,
                    "days_back": days_back,
                    "metric_types": metric_type,
                    "overall_health": (
                        "good"
                        if performance_data.get("accuracy", {}).get(
                            "overall_accuracy", 0
                        )
                        > 0.65
                        else "needs_attention"
                    ),
                },
            },
            "metadata": {
                "metric_type": metric_type,
                "time_period": time_period,
                "include_comparison": include_comparison,
            },
            "timestamp": datetime.now().isoformat(),
        }

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al analizar rendimiento del modelo: {str(e)}",
        )


@router.post("/retrain", response_model=dict, summary="Reentrenar modelo")
async def retrain_model(
    background_tasks: BackgroundTasks,
    force_retrain: bool = Query(
        False, description="Forzar reentrenamiento completo"
    ),
    include_latest_data: bool = Query(
        True, description="Incluir datos más recientes"
    ),
    optimize_hyperparameters: bool = Query(
        False, description="Optimizar hiperparámetros"
    ),
    db: Session = Depends(get_db),
):
    """
    Reentrenar el modelo ML con datos actualizados.
    """
    try:
        # Verificar si ya hay un entrenamiento en progreso
        training_status = await redis_client.get("model_training_status")
        if training_status and training_status.decode() == "in_progress":
            return {
                "success": False,
                "message": "Ya hay un entrenamiento en progreso",
                "data": {
                    "status": "training_in_progress",
                    "estimated_completion": "Unknown",
                },
                "timestamp": datetime.now().isoformat(),
            }

        # Marcar entrenamiento en progreso
        await redis_client.set(
            "model_training_status", "in_progress", ex=7200
        )  # 2 horas

        # Crear tarea de entrenamiento en background
        task_id = f"retrain_{int(datetime.now().timestamp())}"

        background_tasks.add_task(
            retrain_model_task,
            task_id,
            force_retrain,
            include_latest_data,
            optimize_hyperparameters,
            db,
        )

        return {
            "success": True,
            "message": "Reentrenamiento iniciado en background",
            "data": {
                "task_id": task_id,
                "status": "training_started",
                "estimated_duration_minutes": (
                    60 if optimize_hyperparameters else 30
                ),
                "options": {
                    "force_retrain": force_retrain,
                    "include_latest_data": include_latest_data,
                    "optimize_hyperparameters": optimize_hyperparameters,
                },
            },
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        # Limpiar estado de entrenamiento en caso de error
        await redis_client.delete("model_training_status")
        raise HTTPException(
            status_code=500,
            detail=f"Error al iniciar reentrenamiento: {str(e)}",
        )


@router.get(
    "/training/status", response_model=dict, summary="Estado del entrenamiento"
)
async def get_training_status():
    """
    Obtener el estado actual del entrenamiento del modelo.
    """
    try:
        # Verificar estado en Redis
        training_status = await redis_client.get("model_training_status")
        training_progress = await redis_client.get("model_training_progress")

        if not training_status:
            return {
                "success": True,
                "data": {
                    "status": "idle",
                    "message": "No hay entrenamiento en progreso",
                },
                "timestamp": datetime.now().isoformat(),
            }

        status = training_status.decode()
        progress_data = {}

        if training_progress:
            try:
                progress_data = json.loads(training_progress.decode())
            except:
                progress_data = {}

        return {
            "success": True,
            "data": {
                "status": status,
                "progress": progress_data,
                "message": {
                    "in_progress": "Entrenamiento en progreso...",
                    "completed": "Entrenamiento completado exitosamente",
                    "failed": "Entrenamiento falló",
                }.get(status, "Estado desconocido"),
            },
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener estado de entrenamiento: {str(e)}",
        )


@router.get(
    "/history/{prediction_id}",
    response_model=dict,
    summary="Historial de predicción específica",
)
async def get_prediction_history(
    prediction_id: str = Path(..., description="ID de la predicción"),
    include_accuracy: bool = Query(
        True, description="Incluir análisis de precisión"
    ),
    db: Session = Depends(get_db),
):
    """
    Obtener historial y análisis de una predicción específica.
    """
    try:
        # Buscar predicción en historial
        prediction = (
            db.query(PredictionHistory)
            .filter(PredictionHistory.prediction_id == prediction_id)
            .first()
        )

        if not prediction:
            raise HTTPException(
                status_code=404,
                detail=f"Predicción con ID {prediction_id} no encontrada",
            )

        # Obtener partido asociado
        match = (
            db.query(Match)
            .options(
                joinedload(Match.home_team),
                joinedload(Match.away_team),
                joinedload(Match.league),
            )
            .filter(Match.id == prediction.match_id)
            .first()
        )

        prediction_data = {
            "prediction_id": prediction.prediction_id,
            "match_info": {
                "id": match.id if match else None,
                "home_team": (
                    match.home_team.name if match and match.home_team else None
                ),
                "away_team": (
                    match.away_team.name if match and match.away_team else None
                ),
                "league": (
                    match.league.name if match and match.league else None
                ),
                "match_date": match.match_date if match else None,
                "status": match.status if match else None,
            },
            "prediction_details": {
                "type": prediction.prediction_type,
                "predicted_value": prediction.predicted_value,
                "confidence": prediction.confidence,
                "created_at": prediction.created_at,
                "model_version": prediction.model_version,
            },
        }

        # Análisis de precisión (si el partido ya terminó)
        accuracy_analysis = None
        if include_accuracy and match and match.status == "finished":
            try:
                analyzer = MLAnalyzer()
                accuracy_analysis = (
                    await analyzer.analyze_single_prediction_accuracy(
                        prediction, match, db
                    )
                )
            except Exception as e:
                accuracy_analysis = {
                    "error": f"Error calculando precisión: {str(e)}"
                }

        result = {
            "success": True,
            "data": {
                **prediction_data,
                "accuracy_analysis": accuracy_analysis,
                "actual_result": (
                    {
                        "home_score": (
                            match.home_score
                            if match and match.status == "finished"
                            else None
                        ),
                        "away_score": (
                            match.away_score
                            if match and match.status == "finished"
                            else None
                        ),
                        "winner": (
                            None
                            if not match or match.status != "finished"
                            else (
                                "home"
                                if match.home_score > match.away_score
                                else (
                                    "away"
                                    if match.away_score > match.home_score
                                    else "draw"
                                )
                            )
                        ),
                    }
                    if match
                    else None
                ),
            },
            "metadata": {
                "include_accuracy": include_accuracy,
                "match_finished": (
                    match.status == "finished" if match else False
                ),
            },
            "timestamp": datetime.now().isoformat(),
        }

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener historial de predicción: {str(e)}",
        )


# =====================================================
# FUNCIONES AUXILIARES (BACKGROUND TASKS)
# =====================================================


async def save_prediction_to_db(
    prediction_id: str,
    match_data: Dict[str, Any],
    predictions: Dict[str, Any],
    confidence_scores: Dict[str, float],
    db: Session,
):
    """
    Guardar predicción en base de datos (tarea en background).
    """
    try:
        # Guardar cada tipo de predicción
        for pred_type, pred_data in predictions.items():
            if "error" not in pred_data:
                prediction_record = PredictionHistory(
                    prediction_id=prediction_id,
                    match_id=match_data.get("match_id"),
                    home_team_id=match_data["home_team_id"],
                    away_team_id=match_data["away_team_id"],
                    prediction_type=pred_type,
                    predicted_value=json.dumps(pred_data),
                    confidence=confidence_scores.get(pred_type, 0.0),
                    model_version="v2.0",  # Obtener versión real del predictor
                    created_at=datetime.now(),
                )
                db.add(prediction_record)

        db.commit()

    except Exception as e:
        db.rollback()
        # Log error pero no fallar la respuesta principal
        print(f"Error guardando predicción en DB: {str(e)}")


async def retrain_model_task(
    task_id: str,
    force_retrain: bool,
    include_latest_data: bool,
    optimize_hyperparameters: bool,
    db: Session,
):
    """
    Tarea de reentrenamiento del modelo (background).
    """
    try:
        # Actualizar progreso inicial
        await redis_client.set(
            "model_training_progress",
            json.dumps(
                {
                    "task_id": task_id,
                    "stage": "initializing",
                    "progress_percent": 0,
                    "message": "Inicializando reentrenamiento...",
                }
            ),
            ex=7200,
        )

        predictor = AdvancedFootballPredictor()

        # Etapa 1: Preparación de datos
        await redis_client.set(
            "model_training_progress",
            json.dumps(
                {
                    "task_id": task_id,
                    "stage": "data_preparation",
                    "progress_percent": 20,
                    "message": "Preparando datos de entrenamiento...",
                }
            ),
            ex=7200,
        )

        # Obtener datos actualizados
        training_data = await predictor.prepare_training_data(
            include_latest=include_latest_data, db=db
        )

        # Etapa 2: Entrenamiento
        await redis_client.set(
            "model_training_progress",
            json.dumps(
                {
                    "task_id": task_id,
                    "stage": "training",
                    "progress_percent": 50,
                    "message": "Entrenando modelos...",
                }
            ),
            ex=7200,
        )

        # Entrenar modelo
        if optimize_hyperparameters:
            await predictor.train_with_hyperparameter_optimization(
                training_data
            )
        else:
            await predictor.retrain_models(training_data, force_retrain)

        # Etapa 3: Validación
        await redis_client.set(
            "model_training_progress",
            json.dumps(
                {
                    "task_id": task_id,
                    "stage": "validation",
                    "progress_percent": 80,
                    "message": "Validando modelo...",
                }
            ),
            ex=7200,
        )

        # Validar modelo
        validation_results = await predictor.validate_model(
            training_data["test"]
        )

        # Etapa 4: Finalización
        await redis_client.set(
            "model_training_progress",
            json.dumps(
                {
                    "task_id": task_id,
                    "stage": "completed",
                    "progress_percent": 100,
                    "message": "Entrenamiento completado exitosamente",
                    "validation_results": validation_results,
                }
            ),
            ex=7200,
        )

        # Actualizar estado
        await redis_client.set("model_training_status", "completed", ex=3600)

    except Exception as e:
        # Marcar como fallido
        await redis_client.set("model_training_status", "failed", ex=3600)
        await redis_client.set(
            "model_training_progress",
            json.dumps(
                {
                    "task_id": task_id,
                    "stage": "failed",
                    "progress_percent": 0,
                    "message": f"Error en entrenamiento: {str(e)}",
                }
            ),
            ex=7200,
        )
        raise e
