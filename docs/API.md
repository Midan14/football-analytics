# 🏈 Football Analytics API Documentation

> **Sistema avanzado de análisis y predicción deportiva para fútbol**  
> Versión: 2.1.0 | Fecha: Junio 2025

## 📋 Tabla de Contenido

- [Información General](#información-general)
- [Autenticación](#autenticación)
- [Endpoints Principales](#endpoints-principales)
  - [Predicciones](#predicciones)
  - [Datos](#datos)
  - [Cuotas y Value Bets](#cuotas-y-value-bets)
  - [Partidos en Vivo](#partidos-en-vivo)
  - [Análisis](#análisis)
  - [Sistema](#sistema)
- [WebSocket](#websocket)
- [Códigos de Respuesta](#códigos-de-respuesta)
- [Ejemplos de Uso](#ejemplos-de-uso)
- [Rate Limits](#rate-limits)
- [Errores Comunes](#errores-comunes)

## 📡 Información General

### Base URLs

- **Desarrollo**: `http://localhost:8000`
- **Producción**: `https://api.football-analytics.com`
- **WebSocket**: `ws://localhost:8765` (desarrollo)

### Formatos Soportados

- **Request**: JSON
- **Response**: JSON
- **Encoding**: UTF-8

### Headers Requeridos

```http
Content-Type: application/json
Accept: application/json
User-Agent: Football-Analytics-Client/2.1.0
```

## 🔐 Autenticación

### API Key (Desarrollo)

```http
X-API-Key: tu_api_key_aqui
```

### JWT Token (Producción)

```http
Authorization: Bearer <jwt_token>
```

**Obtener Token:**

```bash
POST /auth/login
{
  "username": "tu_usuario",
  "password": "tu_password"
}
```

## 🔮 Predicciones

### Predecir Partido Individual

**Endpoint:** `POST /predict/match`

**Descripción:** Genera predicción para un partido específico utilizando modelos ML.

**Request Body:**

```json
{
  "home_team": "Real Madrid",
  "away_team": "Barcelona",
  "league": "PD",
  "match_date": "2024-06-15T20:00:00Z",
  "model": "xgboost" // opcional, default: xgboost
}
```

**Response:**

```json
{
  "success": true,
  "message": "Predicción creada exitosamente",
  "data": {
    "match": "Real Madrid vs Barcelona",
    "league": "La Liga",
    "match_date": "2024-06-15T20:00:00Z",
    "predictions": {
      "home_win": 0.45,
      "draw": 0.30,
      "away_win": 0.25
    },
    "confidence": 0.78,
    "model_used": "xgboost",
    "expected_goals": {
      "home": 1.8,
      "away": 1.2
    },
    "key_factors": [
      "Forma reciente del equipo local",
      "Historial head-to-head",
      "Estadísticas defensivas"
    ]
  },
  "timestamp": "2024-06-02T15:30:00Z"
}
```

### Predicciones Múltiples

**Endpoint:** `POST /predict/multiple`

**Descripción:** Genera predicciones para múltiples partidos en lote.

**Request Body:**

```json
{
  "matches": [
    {
      "home_team": "Arsenal",
      "away_team": "Chelsea",
      "league": "PL",
      "match_date": "2024-06-15T15:00:00Z"
    },
    {
      "home_team": "Manchester City",
      "away_team": "Liverpool",
      "league": "PL",
      "match_date": "2024-06-16T17:30:00Z"
    }
  ],
  "model": "xgboost"
}
```

**Response:**

```json
{
  "success": true,
  "message": "2 predicciones creadas exitosamente",
  "data": {
    "predictions": [
      {
        "match": "Arsenal vs Chelsea",
        "predictions": { "home_win": 0.52, "draw": 0.28, "away_win": 0.20 },
        "confidence": 0.75
      },
      {
        "match": "Manchester City vs Liverpool",
        "predictions": { "home_win": 0.65, "draw": 0.22, "away_win": 0.13 },
        "confidence": 0.82
      }
    ],
    "summary": {
      "total_matches": 2,
      "average_confidence": 0.785,
      "processing_time_ms": 1250
    }
  },
  "timestamp": "2024-06-02T15:30:00Z"
}
```

### Obtener Predicción por ID

**Endpoint:** `GET /predict/{prediction_id}`

**Response:**

```json
{
  "success": true,
  "data": {
    "id": 123,
    "match": "Real Madrid vs Barcelona",
    "predictions": { "home_win": 0.45, "draw": 0.30, "away_win": 0.25 },
    "confidence": 0.78,
    "model_used": "xgboost",
    "created_at": "2024-06-02T15:30:00Z",
    "is_correct": null,
    "actual_result": null
  }
}
```

## 📊 Datos

### Obtener Equipos por Liga

**Endpoint:** `GET /data/teams/{league_code}`

**Parámetros de Query:**

- `include_stats` (boolean): Incluir estadísticas del equipo
- `season` (string): Temporada específica (ej: "2024-25")

**Ejemplo:** `GET /data/teams/PL?include_stats=true`

**Response:**

```json
{
  "success": true,
  "data": {
    "league": {
      "code": "PL",
      "name": "Premier League",
      "country": "England"
    },
    "teams": [
      {
        "id": 1,
        "name": "Arsenal",
        "short_name": "ARS",
        "country": "England",
        "founded_year": 1886,
        "stadium": {
          "name": "Emirates Stadium",
          "capacity": 60704
        },
        "stats": {
          "matches_played": 38,
          "wins": 26,
          "draws": 6,
          "losses": 6,
          "goals_for": 78,
          "goals_against": 32,
          "points": 84,
          "position": 2,
          "form": "WWDWW"
        }
      }
    ]
  }
}
```

### Obtener Partidos Recientes

**Endpoint:** `GET /data/matches/recent`

**Parámetros de Query:**

- `league` (string): Código de liga (ej: "PL", "PD")
- `days` (integer): Días hacia atrás (default: 7)
- `include_stats` (boolean): Incluir estadísticas detalladas

**Response:**

```json
{
  "success": true,
  "data": {
    "matches": [
      {
        "id": 1001,
        "league": "Premier League",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "match_date": "2024-06-15T15:00:00Z",
        "status": "FINISHED",
        "score": {
          "home": 2,
          "away": 1,
          "half_time": {
            "home": 1,
            "away": 0
          }
        },
        "stats": {
          "possession": { "home": 58, "away": 42 },
          "shots": { "home": 12, "away": 8 },
          "shots_on_target": { "home": 5, "away": 3 },
          "corners": { "home": 7, "away": 4 }
        }
      }
    ],
    "pagination": {
      "total": 45,
      "page": 1,
      "per_page": 20,
      "pages": 3
    }
  }
}
```

### Recolectar Datos

**Endpoint:** `POST /data/collect`

**Descripción:** Inicia proceso de recolección de datos desde APIs externas.

**Request Body:**

```json
{
  "sources": ["football_data", "rapidapi"],
  "leagues": ["PL", "PD", "SA", "BL1"],
  "data_types": ["matches", "teams", "players"],
  "force_update": false
}
```

**Response:**

```json
{
  "success": true,
  "message": "Recolección de datos iniciada",
  "data": {
    "job_id": "collect_001",
    "status": "RUNNING",
    "estimated_completion": "2024-06-02T16:00:00Z",
    "sources_count": 2,
    "leagues_count": 4
  }
}
```

## 💰 Cuotas y Value Bets

### Obtener Cuotas de Partido

**Endpoint:** `GET /odds/{home_team}/{away_team}`

**Parámetros de Query:**

- `match_date` (string): Fecha del partido (ISO 8601)
- `bookmakers` (string): Lista de bookmakers separados por coma

**Response:**

```json
{
  "success": true,
  "data": {
    "match": "Real Madrid vs Barcelona",
    "match_date": "2024-06-15T20:00:00Z",
    "odds": [
      {
        "bookmaker": "Bet365",
        "home_odds": 2.10,
        "draw_odds": 3.20,
        "away_odds": 3.80,
        "is_closing": false,
        "updated_at": "2024-06-02T15:30:00Z"
      },
      {
        "bookmaker": "William Hill",
        "home_odds": 2.05,
        "draw_odds": 3.30,
        "away_odds": 3.90,
        "is_closing": false,
        "updated_at": "2024-06-02T15:25:00Z"
      }
    ],
    "best_odds": {
      "home": { "odds": 2.10, "bookmaker": "Bet365" },
      "draw": { "odds": 3.30, "bookmaker": "William Hill" },
      "away": { "odds": 3.90, "bookmaker": "William Hill" }
    }
  }
}
```

### Analizar Value Bets

**Endpoint:** `POST /odds/analyze`

**Request Body:**

```json
{
  "matches": [
    {
      "home_team": "Arsenal",
      "away_team": "Chelsea",
      "match_date": "2024-06-15T15:00:00Z"
    }
  ],
  "min_value_percentage": 2.0,
  "confidence_threshold": 0.70
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "value_bets": [
      {
        "match": "Arsenal vs Chelsea",
        "bet_type": "H",
        "bet_description": "Arsenal to Win",
        "our_probability": 0.52,
        "best_odds": 1.95,
        "bookmaker": "Bet365",
        "value_percentage": 3.2,
        "kelly_fraction": 0.028,
        "recommended_stake": 2.5,
        "confidence_level": "MEDIUM",
        "expected_profit": 6.4
      }
    ],
    "summary": {
      "total_matches_analyzed": 1,
      "value_bets_found": 1,
      "average_value": 3.2,
      "total_recommended_stake": 2.5
    }
  }
}
```

## 🔴 Partidos en Vivo

### Obtener Partidos en Vivo

**Endpoint:** `GET /live/matches`

**Response:**

```json
{
  "success": true,
  "data": {
    "live_matches": [
      {
        "id": 2001,
        "league": "Premier League",
        "home_team": "Manchester United",
        "away_team": "Liverpool",
        "status": "LIVE",
        "minute": 67,
        "score": {
          "home": 1,
          "away": 2
        },
        "events": [
          {
            "minute": 15,
            "type": "GOAL",
            "team": "home",
            "player": "Marcus Rashford"
          },
          {
            "minute": 34,
            "type": "GOAL", 
            "team": "away",
            "player": "Mohamed Salah"
          }
        ],
        "live_odds": {
          "home_win": 2.8,
          "draw": 3.1,
          "away_win": 2.4
        }
      }
    ]
  }
}
```

### Iniciar Tracking de Partido

**Endpoint:** `POST /live/track/{match_id}`

**Request Body:**

```json
{
  "track_events": true,
  "track_odds": true,
  "notification_preferences": {
    "goals": true,
    "cards": false,
    "odds_changes": true
  }
}
```

**Response:**

```json
{
  "success": true,
  "message": "Tracking iniciado para el partido",
  "data": {
    "match_id": 2001,
    "tracking_id": "track_001",
    "status": "ACTIVE",
    "features": ["events", "odds", "notifications"]
  }
}
```

## 📈 Análisis

### Análisis de Equipo

**Endpoint:** `GET /analysis/team/{team_id}`

**Parámetros de Query:**

- `matches` (integer): Número de partidos a analizar (default: 10)
- `include_opponents` (boolean): Incluir análisis de oponentes

**Response:**

```json
{
  "success": true,
  "data": {
    "team": {
      "id": 1,
      "name": "Arsenal",
      "league": "Premier League"
    },
    "form_analysis": {
      "recent_form": "WWDWW",
      "points_per_game": 2.1,
      "goals_per_game": 2.3,
      "goals_conceded_per_game": 0.8,
      "clean_sheets_percentage": 60,
      "matches_analyzed": 10
    },
    "performance_metrics": {
      "attacking_strength": 1.15,
      "defensive_strength": 0.72,
      "home_advantage": 1.23,
      "away_performance": 0.89
    },
    "trends": {
      "scoring_trend": "IMPROVING",
      "defensive_trend": "STABLE",
      "overall_trend": "POSITIVE"
    }
  }
}
```

### Análisis Head-to-Head

**Endpoint:** `GET /analysis/h2h/{team1_id}/{team2_id}`

**Response:**

```json
{
  "success": true,
  "data": {
    "teams": {
      "team1": "Arsenal",
      "team2": "Chelsea"
    },
    "historical_record": {
      "total_matches": 58,
      "team1_wins": 25,
      "draws": 13,
      "team2_wins": 20,
      "team1_win_percentage": 43.1
    },
    "recent_matches": [
      {
        "date": "2024-04-20",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "score": "3-1",
        "venue": "Emirates Stadium"
      }
    ],
    "trends": {
      "goals_per_match": 2.8,
      "both_teams_score_percentage": 72,
      "over_2_5_goals_percentage": 65
    }
  }
}
```

## ⚙️ Sistema

### Health Check

**Endpoint:** `GET /health`

**Response:**

```json
{
  "overall_status": "healthy",
  "timestamp": "2024-06-02T15:30:00Z",
  "version": "2.1.0",
  "environment": "development",
  "system": {
    "initialized": true,
    "uptime_seconds": 3600,
    "memory_usage_mb": 512,
    "cpu_usage_percentage": 15.5
  },
  "components": {
    "database": {
      "status": "healthy",
      "connection_pool": "8/10",
      "last_query_ms": 45
    },
    "ml_models": {
      "status": "healthy",
      "models_loaded": 3,
      "last_prediction_ms": 120
    },
    "external_apis": {
      "status": "healthy",
      "football_data_api": "CONNECTED",
      "odds_api": "CONNECTED"
    },
    "cache": {
      "status": "healthy",
      "redis_connected": true,
      "cache_hit_rate": 85.2
    }
  }
}
```

### Configuración del Sistema

**Endpoint:** `GET /config`

**Response:**

```json
{
  "success": true,
  "data": {
    "application": {
      "name": "Football Analytics",
      "version": "2.1.0",
      "environment": "development"
    },
    "ml": {
      "default_model": "xgboost",
      "confidence_threshold": 0.65,
      "available_models": ["xgboost", "lightgbm", "catboost"]
    },
    "api": {
      "rate_limits": {
        "predictions": "100/hour",
        "data_collection": "50/hour"
      }
    },
    "features": {
      "live_tracking": true,
      "value_betting": true,
      "multi_league": true
    }
  }
}
```

### Métricas del Sistema

**Endpoint:** `GET /metrics`

**Response:**

```json
{
  "success": true,
  "data": {
    "predictions": {
      "total_predictions": 15420,
      "predictions_today": 45,
      "accuracy_last_30_days": 72.5,
      "average_confidence": 0.74
    },
    "api_usage": {
      "requests_today": 1250,
      "requests_last_hour": 85,
      "average_response_time_ms": 245
    },
    "data": {
      "leagues_monitored": 25,
      "teams_tracked": 580,
      "matches_in_database": 25600
    },
    "performance": {
      "uptime_percentage": 99.8,
      "average_prediction_time_ms": 120,
      "cache_hit_rate": 85.2
    }
  }
}
```

## 📡 WebSocket

### Conexión

**URL:** `ws://localhost:8765`

**Conexión inicial:**

```javascript
const ws = new WebSocket('ws://localhost:8765');

ws.onopen = function() {
    console.log('🔌 Conectado a Football Analytics');
    
    // Suscribirse a eventos
    ws.send(JSON.stringify({
        type: 'subscribe',
        channels: ['predictions', 'live_matches', 'value_bets']
    }));
};
```

### Mensajes Disponibles

#### Suscribirse a Predicciones

```json
{
  "type": "subscribe_predictions",
  "data": {
    "leagues": ["PL", "PD", "SA"],
    "min_confidence": 0.70
  }
}
```

#### Obtener Partidos en Vivo

```json
{
  "type": "get_live_matches",
  "data": {}
}
```

#### Ping/Pong

```json
{
  "type": "ping",
  "data": {}
}
```

### Respuestas WebSocket

#### Nueva Predicción

```json
{
  "type": "new_prediction",
  "data": {
    "match": "Real Madrid vs Barcelona",
    "predictions": { "home_win": 0.45, "draw": 0.30, "away_win": 0.25 },
    "confidence": 0.78,
    "created_at": "2024-06-02T15:30:00Z"
  },
  "timestamp": "2024-06-02T15:30:00Z"
}
```

#### Partidos en Vivo

```json
{
  "type": "live_matches",
  "data": [
    {
      "match": "Manchester United vs Liverpool",
      "minute": 67,
      "score": { "home": 1, "away": 2 },
      "status": "LIVE"
    }
  ],
  "timestamp": "2024-06-02T15:30:00Z"
}
```

#### Value Bet Detectado

```json
{
  "type": "value_bet_alert",
  "data": {
    "match": "Arsenal vs Chelsea",
    "bet_type": "H",
    "value_percentage": 5.2,
    "recommended_stake": 3.0,
    "confidence": "HIGH"
  },
  "timestamp": "2024-06-02T15:30:00Z"
}
```

## 📋 Códigos de Respuesta

| Código | Descripción | Uso |
|--------|-------------|-----|
| 200 | OK | Solicitud exitosa |
| 201 | Created | Recurso creado exitosamente |
| 400 | Bad Request | Error en parámetros de solicitud |
| 401 | Unauthorized | Autenticación requerida |
| 403 | Forbidden | Sin permisos suficientes |
| 404 | Not Found | Recurso no encontrado |
| 429 | Too Many Requests | Rate limit excedido |
| 500 | Internal Server Error | Error interno del servidor |
| 503 | Service Unavailable | Servicio temporalmente no disponible |

## 📖 Ejemplos de Uso

### Ejemplo 1: Predicción Completa

```bash
# 1. Hacer predicción
curl -X POST "http://localhost:8000/predict/match" \
  -H "Content-Type: application/json" \
  -d '{
    "home_team": "Real Madrid",
    "away_team": "Barcelona",
    "league": "PD",
    "match_date": "2024-06-15T20:00:00Z"
  }'

# 2. Obtener cuotas
curl "http://localhost:8000/odds/Real%20Madrid/Barcelona?match_date=2024-06-15T20:00:00Z"

# 3. Analizar value bets
curl -X POST "http://localhost:8000/odds/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "matches": [
      {
        "home_team": "Real Madrid",
        "away_team": "Barcelona",
        "match_date": "2024-06-15T20:00:00Z"
      }
    ],
    "min_value_percentage": 2.0
  }'
```

### Ejemplo 2: Análisis de Liga

```bash
# 1. Obtener equipos de Premier League
curl "http://localhost:8000/data/teams/PL?include_stats=true"

# 2. Obtener partidos recientes
curl "http://localhost:8000/data/matches/recent?league=PL&days=7"

# 3. Analizar equipo específico
curl "http://localhost:8000/analysis/team/1?matches=10"
```

### Ejemplo 3: Cliente WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8765');

ws.onopen = function() {
    // Suscribirse a predicciones de Premier League
    ws.send(JSON.stringify({
        type: 'subscribe_predictions',
        data: {
            leagues: ['PL'],
            min_confidence: 0.75
        }
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    switch(data.type) {
        case 'new_prediction':
            console.log('🔮 Nueva predicción:', data.data);
            break;
        case 'value_bet_alert':
            console.log('💰 Value bet detectado:', data.data);
            break;
        case 'live_matches':
            console.log('🔴 Partidos en vivo:', data.data);
            break;
    }
};
```

## ⏱️ Rate Limits

| Endpoint | Límite | Ventana |
|----------|--------|---------|
| `/predict/*` | 100 requests | por hora |
| `/data/*` | 200 requests | por hora |
| `/odds/*` | 150 requests | por hora |
| `/live/*` | 300 requests | por hora |
| `/analysis/*` | 50 requests | por hora |
| **Global** | 500 requests | por hora |

### Headers de Rate Limit

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1623456789
```

## ❌ Errores Comunes

### Error 400: Bad Request

```json
{
  "success": false,
  "error": "BAD_REQUEST",
  "message": "Parámetros inválidos",
  "details": {
    "field": "match_date",
    "issue": "Formato de fecha inválido. Use ISO 8601."
  },
  "timestamp": "2024-06-02T15:30:00Z"
}
```

### Error 404: Not Found

```json
{
  "success": false,
  "error": "NOT_FOUND",
  "message": "Equipo no encontrado",
  "details": {
    "team": "Real Madri",
    "suggestion": "¿Quisiste decir 'Real Madrid'?"
  },
  "timestamp": "2024-06-02T15:30:00Z"
}
```

### Error 429: Rate Limit

```json
{
  "success": false,
  "error": "RATE_LIMIT_EXCEEDED",
  "message": "Límite de requests excedido",
  "details": {
    "limit": 100,
    "window": "1 hour",
    "retry_after": 1800
  },
  "timestamp": "2024-06-02T15:30:00Z"
}
```

### Error 500: Internal Error

```json
{
  "success": false,
  "error": "INTERNAL_SERVER_ERROR",
  "message": "Error interno del servidor",
  "details": {
    "error_id": "err_001",
    "support_contact": "support@football-analytics.com"
  },
  "timestamp": "2024-06-02T15:30:00Z"
}
```

## 🚀 SDK y Librerías

### Python SDK

```python
from football_analytics import FootballAPI

client = FootballAPI(api_key="tu_api_key")

# Hacer predicción
prediction = client.predict_match(
    home_team="Real Madrid",
    away_team="Barcelona",
    league="PD"
)

# Obtener value bets
value_bets = client.get_value_bets(
    min_value=2.0,
    leagues=["PL", "PD"]
)
```

### JavaScript SDK

```javascript
import { FootballAnalytics } from 'football-analytics-js';

const client = new FootballAnalytics({
    apiKey: 'tu_api_key',
    baseURL: 'http://localhost:8000'
});

// Hacer predicción
const prediction = await client.predictMatch({
    homeTeam: 'Real Madrid',
    awayTeam: 'Barcelona',
    league: 'PD'
});

// WebSocket
const ws = client.connectWebSocket();
ws.subscribe('predictions', (data) => {
    console.log('Nueva predicción:', data);
});
```

## 📞 Soporte

- **Email**: <support@football-analytics.com>
- **Discord**: [Football Analytics Community](https://discord.gg/football-analytics)
- **GitHub**: [Issues](https://github.com/football-analytics/api/issues)
- **Documentación**: [docs.football-analytics.com](https://docs.football-analytics.com)

## 📝 Changelog

### v2.1.0 (Junio 2025)

- ✅ WebSocket en tiempo real
- ✅ Value betting automático
- ✅ 265+ ligas soportadas
- ✅ Modelos ML mejorados

### v2.0.0 (Mayo 2025)

- ✅ API REST completa
- ✅ Predicciones ML
- ✅ Sistema de cuotas
- ✅ Análisis de equipos

---

**🏈 Football Analytics API - Conquista el mundo del fútbol con datos** ⚽🚀
