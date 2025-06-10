# 🤝 Contributing to Football Analytics

> **¡Bienvenido al proyecto Football Analytics!**  
> Gracias por tu interés en contribuir a nuestro sistema de análisis y predicción deportiva. Esta guía te ayudará a comenzar.

## 📋 Tabla de Contenido

- [Código de Conducta](#código-de-conducta)
- [Cómo Contribuir](#cómo-contribuir)
- [Configuración del Entorno](#configuración-del-entorno)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Estándares de Código](#estándares-de-código)
- [Testing](#testing)
- [Pull Requests](#pull-requests)
- [Reportar Issues](#reportar-issues)
- [Tipos de Contribuciones](#tipos-de-contribuciones)
- [Documentación](#documentación)
- [Comunidad](#comunidad)

## 📜 Código de Conducta

### Nuestros Valores

- **Respeto**: Tratamos a todos con respeto y profesionalismo
- **Inclusión**: Valoramos la diversidad y perspectivas diferentes
- **Colaboración**: Trabajamos juntos hacia objetivos comunes
- **Aprendizaje**: Fomentamos el crecimiento y conocimiento compartido
- **Excelencia**: Nos esforzamos por la calidad en todo lo que hacemos

### Comportamiento Esperado

✅ Usar lenguaje inclusivo y profesional  
✅ Respetar diferentes puntos de vista y experiencias  
✅ Aceptar críticas constructivas de manera positiva  
✅ Centrarse en lo que es mejor para la comunidad  
✅ Mostrar empatía hacia otros miembros  

### Comportamiento Inaceptable

❌ Lenguaje o imágenes sexualizadas  
❌ Trolling, comentarios insultantes o ataques personales  
❌ Acoso público o privado  
❌ Publicar información privada sin autorización  
❌ Otra conducta que podría considerarse inapropiada  

## 🚀 Cómo Contribuir

### 1. Fork del Repositorio

```bash
# Fork en GitHub, luego clona tu fork
git clone https://github.com/TU_USUARIO/football-analytics.git
cd football-analytics

# Agrega el repositorio original como upstream
git remote add upstream https://github.com/ORIGINAL_OWNER/football-analytics.git
```

### 2. Crea una Rama de Trabajo

```bash
# Sincroniza con upstream
git fetch upstream
git checkout main
git merge upstream/main

# Crea tu rama de feature
git checkout -b feature/nombre-descriptivo
# o para bugfixes
git checkout -b fix/descripcion-del-bug
# o para documentación
git checkout -b docs/mejora-documentacion
```

### 3. Realiza tus Cambios

```bash
# Haz tus cambios siguiendo nuestros estándares
# Asegúrate de ejecutar tests
pytest

# Verifica el código
flake8
black --check .
mypy app/
```

### 4. Commit y Push

```bash
# Commits descriptivos siguiendo Conventional Commits
git add .
git commit -m "feat: agregar predicción para ligas asiáticas"

# Push a tu fork
git push origin feature/nombre-descriptivo
```

### 5. Crea un Pull Request

- Ve a GitHub y crea un Pull Request
- Llena la plantilla de PR completamente
- Asigna reviewers apropiados
- Añade labels relevantes

## 🛠️ Configuración del Entorno

### Prerrequisitos

- Python 3.9+
- Git
- Docker (opcional pero recomendado)
- Node.js 16+ (para frontend)

### Setup Local

#### 1. Clonar y Configurar

```bash
git clone https://github.com/TU_USUARIO/football-analytics.git
cd football-analytics

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias de desarrollo
pip install -e .[dev,test,lint]
```

#### 2. Configurar Base de Datos

```bash
# Copiar archivo de configuración
cp .env.example .env

# Editar .env con tus configuraciones
# Especialmente DATABASE_URL y API keys

# Inicializar base de datos
cd database
sqlite3 ../data/football_analytics.db < 01-create-tables.sql
sqlite3 ../data/football_analytics.db < 02-insert-initial-data.sql
```

#### 3. Verificar Instalación

```bash
# Ejecutar diagnóstico
python diagnose.py

# Iniciar servidor de desarrollo
python app/main.py

# En otra terminal, ejecutar tests
pytest
```

#### 4. Setup con Docker (Alternativo)

```bash
# Construir y ejecutar
docker-compose up -d

# Verificar que todo funciona
curl http://localhost:8000/health
```

### Variables de Entorno Requeridas

```bash
# .env para desarrollo
ENVIRONMENT=development
DEBUG=true
FOOTBALL_DATA_API_KEY=tu_api_key_aqui
DATABASE_URL=sqlite:///data/football_analytics.db
SECRET_KEY=tu_secret_key_para_desarrollo
```

## 📁 Estructura del Proyecto

```
football-analytics/
├── 📁 backend/                 # Aplicación principal
│   ├── 📁 app/
│   │   ├── 📁 api/            # Endpoints REST
│   │   ├── 📁 ml_models/      # Modelos ML
│   │   ├── 📁 services/       # Lógica de negocio
│   │   ├── 📁 utils/          # Utilidades
│   │   ├── config.py          # Configuración
│   │   └── main.py            # Punto de entrada
│   ├── 📁 tests/              # Tests automatizados
│   ├── pyproject.toml         # Configuración del proyecto
│   └── requirements.txt       # Dependencias
├── 📁 database/               # Esquemas y datos
│   ├── __init__.py           # Módulo de DB
│   ├── 01-create-tables.sql  # Esquema de DB
│   └── 02-insert-initial-data.sql  # Datos iniciales
├── 📁 docs/                   # Documentación
│   ├── API.md                 # Documentación de API
│   ├── CONTRIBUTING.md        # Esta guía
│   └── README.md              # Documentación principal
├── 📁 frontend/               # Interfaz web (futuro)
├── 📁 scripts/                # Scripts de utilidad
└── 📁 data/                   # Datos locales
```

### Convenciones de Nomenclatura

#### Archivos y Directorios

- **snake_case** para archivos Python: `team_analyzer.py`
- **kebab-case** para archivos de configuración: `docker-compose.yml`
- **PascalCase** para clases: `PredictorService`
- **UPPER_CASE** para constantes: `API_BASE_URL`

#### Git Branches

- `feature/descripcion-corta` - Nuevas funcionalidades
- `fix/descripcion-del-bug` - Corrección de bugs
- `docs/mejora-especifica` - Mejoras de documentación
- `refactor/componente-afectado` - Refactoring
- `perf/mejora-performance` - Optimizaciones

## 🎨 Estándares de Código

### Python

#### Formateo y Linting

```bash
# Formateo automático
black .
isort .

# Linting
flake8
mypy app/

# Security check
bandit -r app/
```

#### Configuración incluida

- **Black**: Formateo de código (línea 88 caracteres)
- **isort**: Ordenamiento de imports
- **flake8**: Linting con configuración personalizada
- **mypy**: Type checking estático
- **pytest**: Framework de testing

#### Ejemplo de Código Bien Formateado

```python
from typing import Dict, List, Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.utils.constants import PREMIER_LEAGUE_TEAMS
from app.services.base import BaseService


class TeamAnalyzer(BaseService):
    """Analiza el rendimiento y estadísticas de equipos."""
    
    def __init__(self, db_session: Session) -> None:
        super().__init__(db_session)
        self.logger = self._setup_logger()
    
    def calculate_form(
        self, 
        team_id: int, 
        matches: int = 5
    ) -> Dict[str, float]:
        """Calcula la forma reciente de un equipo.
        
        Args:
            team_id: ID del equipo a analizar
            matches: Número de partidos a considerar
            
        Returns:
            Diccionario con métricas de forma
            
        Raises:
            HTTPException: Si el equipo no existe
        """
        if not self._team_exists(team_id):
            raise HTTPException(
                status_code=404, 
                detail=f"Team with ID {team_id} not found"
            )
        
        # Implementación aquí...
        return {
            "points_per_game": 2.1,
            "goals_per_game": 1.8,
            "clean_sheets_percentage": 60.0
        }
```

### Documentación de Código

- **Docstrings**: Usar formato Google style
- **Type hints**: Obligatorio para funciones públicas
- **Comentarios**: Solo cuando la lógica no es obvia
- **README**: Mantener actualizado para cada módulo

### SQL

```sql
-- Comentarios claros y descriptivos
-- Usar UPPER CASE para palabras clave SQL
-- Usar snake_case para nombres de tablas y columnas
-- Indentar consultas complejas

SELECT 
    t.name AS team_name,
    COUNT(m.id) AS matches_played,
    AVG(CASE 
        WHEN m.home_team_id = t.id THEN m.home_score_ft
        ELSE m.away_score_ft 
    END) AS avg_goals_scored
FROM teams t
JOIN matches m ON (m.home_team_id = t.id OR m.away_team_id = t.id)
WHERE m.match_date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY t.id, t.name
ORDER BY avg_goals_scored DESC;
```

## 🧪 Testing

### Estructura de Tests

```
tests/
├── conftest.py              # Configuración de pytest
├── test_api/               # Tests de endpoints
│   ├── test_predictions.py
│   ├── test_data.py
│   └── test_odds.py
├── test_services/          # Tests de servicios
│   ├── test_predictor.py
│   ├── test_data_collector.py
│   └── test_odds_analyzer.py
├── test_ml_models/         # Tests de modelos ML
│   ├── test_xgboost_model.py
│   └── test_model_evaluation.py
└── test_utils/             # Tests de utilidades
    ├── test_helpers.py
    └── test_constants.py
```

### Ejecutar Tests

```bash
# Todos los tests
pytest

# Tests específicos
pytest tests/test_api/test_predictions.py

# Tests con coverage
pytest --cov=app --cov-report=html

# Tests de performance (marcar como slow)
pytest -m "not slow"

# Tests de integración
pytest -m integration
```

### Escribir Tests

#### Ejemplo de Test de API

```python
import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.factories import TeamFactory, MatchFactory


class TestPredictionsAPI:
    """Tests para endpoints de predicciones."""
    
    def setup_method(self):
        """Setup ejecutado antes de cada test."""
        self.client = TestClient(app)
        self.team_home = TeamFactory(name="Arsenal")
        self.team_away = TeamFactory(name="Chelsea")
    
    def test_predict_match_success(self):
        """Test predicción exitosa de partido."""
        payload = {
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "league": "PL",
            "match_date": "2024-06-15T15:00:00Z"
        }
        
        response = self.client.post("/predict/match", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "predictions" in data["data"]
        assert "confidence" in data["data"]
        assert data["data"]["confidence"] > 0.5
    
    def test_predict_match_invalid_team(self):
        """Test con equipo inexistente."""
        payload = {
            "home_team": "Equipo Inexistente",
            "away_team": "Chelsea",
            "league": "PL"
        }
        
        response = self.client.post("/predict/match", json=payload)
        
        assert response.status_code == 404
        data = response.json()
        assert data["success"] is False
        assert "not found" in data["message"].lower()
```

#### Ejemplo de Test de Servicio

```python
import pytest
from unittest.mock import Mock, patch

from app.services.predictor import PredictorService
from app.ml_models.match_outcome import MatchOutcomeModel


class TestPredictorService:
    """Tests para el servicio de predicciones."""
    
    def setup_method(self):
        """Setup ejecutado antes de cada test."""
        self.db_session = Mock()
        self.predictor = PredictorService(self.db_session)
    
    @patch('app.services.predictor.MatchOutcomeModel')
    def test_predict_match_outcome(self, mock_model):
        """Test predicción de resultado de partido."""
        # Arrange
        mock_model.return_value.predict.return_value = {
            "home_win": 0.45,
            "draw": 0.30,
            "away_win": 0.25
        }
        
        # Act
        result = self.predictor.predict_match_outcome(1, 2)
        
        # Assert
        assert result["home_win"] == 0.45
        assert result["draw"] == 0.30
        assert result["away_win"] == 0.25
        mock_model.return_value.predict.assert_called_once()
```

### Coverage Requirements

- **Minimum coverage**: 80%
- **Critical components**: 90%+ (servicios, modelos ML)
- **API endpoints**: 85%+

## 📝 Pull Requests

### Plantilla de PR

```markdown
## 📋 Descripción
Breve descripción de los cambios realizados.

## 🔄 Tipo de Cambio
- [ ] 🐛 Bug fix (cambio que soluciona un issue)
- [ ] ✨ Nueva feature (cambio que añade funcionalidad)
- [ ] 💥 Breaking change (fix o feature que causa cambios incompatibles)
- [ ] 📚 Documentación (cambios solo en documentación)
- [ ] 🔧 Refactoring (cambio que no arregla bug ni añade feature)
- [ ] ⚡ Performance (cambio que mejora performance)
- [ ] 🧪 Tests (añadir tests faltantes o corregir existentes)

## 🧪 Testing
- [ ] Tests unitarios añadidos/actualizados
- [ ] Tests de integración añadidos/actualizados
- [ ] Todos los tests pasan
- [ ] Coverage mantenido/mejorado

## 📋 Checklist
- [ ] Mi código sigue los estándares del proyecto
- [ ] He realizado auto-review de mi código
- [ ] He comentado áreas complejas de mi código
- [ ] He actualizado la documentación correspondiente
- [ ] Mis cambios no generan nuevos warnings
- [ ] He añadido tests que prueban mis cambios
- [ ] Tests nuevos y existentes pasan localmente

## 📊 Impacto
Describe el impacto de estos cambios en el sistema.

## 📸 Screenshots (si aplica)
Añadir screenshots para cambios visuales.

## 🔗 Issues Relacionados
Closes #123
Related to #456
```

### Proceso de Review

1. **Automated checks**: CI/CD debe pasar
2. **Peer review**: Al menos 1 aprobación requerida
3. **Maintainer review**: Para cambios críticos
4. **Testing**: Verificar en entorno de desarrollo

### Criterios de Aprobación

✅ Código sigue estándares establecidos  
✅ Tests adecuados incluidos  
✅ Documentación actualizada  
✅ Sin conflictos de merge  
✅ CI/CD pasa exitosamente  

## 🐛 Reportar Issues

### Antes de Reportar

1. **Busca issues existentes** para evitar duplicados
2. **Verifica en la última versión** del proyecto
3. **Reproduce el bug** de manera consistente
4. **Revisa la documentación** por si es un malentendido

### Plantilla de Bug Report

```markdown
## 🐛 Descripción del Bug
Una descripción clara y concisa del bug.

## 🔄 Pasos para Reproducir
1. Ve a '...'
2. Haz click en '....'
3. Desplázate hacia '....'
4. Ver error

## ✅ Comportamiento Esperado
Descripción clara de lo que esperabas que pasara.

## 🚫 Comportamiento Actual
Descripción clara de lo que está pasando actualmente.

## 📸 Screenshots
Si aplica, añade screenshots para explicar el problema.

## 🖥️ Entorno
- OS: [ej. macOS 12.0]
- Python: [ej. 3.9.7]
- Versión del proyecto: [ej. 2.1.0]
- Docker: [Si/No, versión]

## 📋 Información Adicional
Cualquier otro contexto sobre el problema.

## 🔧 Solución Propuesta (opcional)
Si tienes ideas sobre cómo solucionarlo.
```

### Plantilla de Feature Request

```markdown
## 🚀 Feature Request

## 📋 Resumen
Breve descripción de la feature que solicitas.

## 🎯 Problema que Resuelve
¿Qué problema resuelve esta feature? ¿Por qué es necesaria?

## 💡 Solución Propuesta
Descripción detallada de cómo te gustaría que funcionara.

## 🔄 Alternativas Consideradas
Otras soluciones que has considerado.

## 📊 Casos de Uso
Ejemplos específicos de cómo se usaría esta feature.

## 📋 Información Adicional
Cualquier contexto adicional, mockups, etc.
```

## 🎯 Tipos de Contribuciones

### 🔮 Machine Learning y Algoritmos

- Nuevos modelos de predicción
- Mejoras en algoritmos existentes
- Optimización de hyperparámetros
- Features engineering innovadoras

**Áreas prioritarias:**

- Modelos para ligas específicas
- Predicciones de goles exactos
- Análisis de jugadores individuales
- Detección de value bets mejorada

### 📊 Análisis de Datos

- Nuevas métricas y estadísticas
- Visualizaciones avanzadas
- Análisis comparativos
- Dashboards interactivos

### 🌐 API y Backend

- Nuevos endpoints
- Optimizaciones de performance
- Mejoras en caching
- Monitoreo y logging

### 📱 Frontend y UX

- Interfaces de usuario
- Dashboards interactivos
- Mobile responsiveness
- Experiencia de usuario

### 🗄️ Base de Datos

- Optimización de queries
- Nuevas estructuras de datos
- Migraciones
- Índices y performance

### 📚 Documentación

- Guías de usuario
- Tutoriales técnicos
- Ejemplos de código
- Traducciones

### 🧪 Testing y QA

- Tests automatizados
- Performance testing
- Security testing
- End-to-end testing

## 📖 Documentación

### Estructura de Documentación

```
docs/
├── README.md              # Introducción general
├── API.md                 # Documentación de API
├── CONTRIBUTING.md        # Esta guía
├── DEPLOYMENT.md          # Guía de deployment
├── ARCHITECTURE.md        # Arquitectura del sistema
├── ML_MODELS.md          # Documentación de modelos
├── DATABASE.md           # Esquemas y datos
└── tutorials/            # Tutoriales específicos
    ├── getting_started.md
    ├── adding_new_league.md
    └── custom_predictions.md
```

### Estilo de Documentación

- **Markdown** para toda la documentación
- **Emojis** para hacer más visual y amigable
- **Ejemplos prácticos** en cada sección
- **Links internos** para navegación fácil
- **Código comentado** en ejemplos

### Actualizar Documentación

```bash
# Siempre actualizar documentación con cambios
git add docs/API.md
git commit -m "docs: actualizar endpoint de predicciones"

# Verificar links rotos
markdown-link-check docs/*.md

# Generar documentación automática
python scripts/generate_api_docs.py
```

## 👥 Comunidad

### Canales de Comunicación

- **GitHub Issues**: Para bugs y feature requests
- **GitHub Discussions**: Para preguntas generales y ideas
- **Discord**: [Football Analytics Community](https://discord.gg/football-analytics)
- **Twitter**: [@FootballAnalytics](https://twitter.com/footballanalytics)

### Reuniones de la Comunidad

- **Weekly Dev Meeting**: Martes 19:00 UTC
- **Monthly Planning**: Primer viernes de cada mes
- **Quarterly Review**: Revisión trimestral de roadmap

### Reconocimientos

Reconocemos las contribuciones en:

- **README.md**: Lista de contributors
- **Release notes**: Mention en changelogs
- **Hall of Fame**: Contributors destacados
- **Swag**: Camisetas y stickers para contributors activos

### Niveles de Contributors

- 🌱 **Newcomer**: Primera contribución
- 🚀 **Regular**: 5+ contribuciones
- ⭐ **Core**: 20+ contribuciones + review access
- 👑 **Maintainer**: Acceso completo al repositorio

## 🎉 Empezando

### Tu Primera Contribución

#### 1. Issues para Principiantes

Busca issues etiquetados con:

- `good-first-issue`: Ideal para principiantes
- `help-wanted`: Necesitamos ayuda
- `documentation`: Mejoras en docs
- `easy`: Nivel de dificultad bajo

#### 2. Ideas de Contribución Fáciles

- Añadir nuevos equipos a ligas existentes
- Mejorar mensajes de error en la API
- Escribir tests para funciones existentes
- Corregir typos en documentación
- Añadir ejemplos de uso

#### 3. Contribuciones Avanzadas

- Implementar nuevos modelos ML
- Optimizar performance de predicciones
- Añadir soporte para nuevas ligas
- Crear dashboards interactivos

### Mentorship Program

¿Nuevo en el proyecto? ¡Te asignamos un mentor!

- Guía personalizada para tu primera contribución
- Revisión de código 1-on-1
- Ayuda con configuración del entorno
- Introducción a la comunidad

Solicita un mentor abriendo un issue con el tag `mentor-request`.

## 📞 Obtener Ayuda

### ❓ Tengo una Pregunta

- **GitHub Discussions**: Para preguntas generales
- **Discord**: Para chat en tiempo real
- **Stack Overflow**: Tag `football-analytics`

### 🐛 Encontré un Bug

- **GitHub Issues**: Reporta usando la plantilla
- **Discord #bugs**: Para discusión rápida
- **Email**: <critical-bugs@football-analytics.com> (solo críticos)

### 💡 Tengo una Idea

- **GitHub Discussions**: Comparte tu idea
- **Discord #ideas**: Brainstorming con la comunidad
- **Feature Request**: Issue formal si está bien definida

### 🚀 Quiero Contribuir

- **Esta guía**: Lee completamente este documento
- **Discord #contributors**: Canal para nuevos contributors
- **Mentorship**: Solicita un mentor si eres nuevo

---

## 🎯 Conclusión

¡Gracias por considerar contribuir a Football Analytics! Tu participación hace que este proyecto sea mejor para toda la comunidad deportiva.

Recuerda:

- **Lee esta guía completamente** antes de tu primera contribución
- **Sigue nuestros estándares** de código y documentación
- **Sé respetuoso** en todas las interacciones
- **Pide ayuda** cuando la necesites
- **Diviértete** contribuyendo al futuro del análisis deportivo

### Primeros Pasos Rápidos

1. 🍴 Fork el repositorio
2. 🛠️ Configura tu entorno local
3. 🔍 Encuentra un `good-first-issue`
4. 💻 Haz tu contribución
5. 📝 Abre tu primer Pull Request
6. 🎉 ¡Celebra ser parte de la comunidad!

**¡Esperamos ver tus contribuciones pronto!** ⚽🚀

---

*Football Analytics - Construyendo el futuro del análisis deportivo, juntos.*
