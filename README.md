# ⚽ Football Analytics Platform

Una plataforma completa de análisis de fútbol con predicciones de machine learning, seguimiento en vivo y estadísticas avanzadas.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg)](https://reactjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg)](https://fastapi.tiangolo.com)

## 🌟 Características Principales

### 🤖 Machine Learning
- **Predicciones de resultados** de partidos con alta precisión
- **Análisis de rendimiento** de jugadores
- **Predicción de lesiones** basada en datos históricos
- **Cálculo de probabilidades** de goles y tarjetas

### 📊 Analytics Avanzados
- Dashboard interactivo con métricas en tiempo real
- Comparación de equipos y jugadores
- Estadísticas históricas y tendencias
- Visualizaciones dinámicas con gráficos

### 🔴 Seguimiento en Vivo
- Actualizaciones de partidos en tiempo real
- Notificaciones de eventos importantes
- Tracking de estadísticas durante el juego
- WebSocket para comunicación bidireccional

## 🛠️ Stack Tecnológico

### Backend
- **FastAPI** - API REST moderna y rápida
- **Python 3.10+** - Lenguaje principal
- **SQLAlchemy** - ORM para base de datos
- **PostgreSQL** - Base de datos principal
- **Redis** - Cache y sesiones
- **Celery** - Tareas asíncronas

### Frontend
- **React 18** - Interfaz de usuario
- **Tailwind CSS** - Estilos y diseño
- **Context API** - Gestión de estado
- **WebSocket** - Comunicación en tiempo real
- **Chart.js** - Visualizaciones

### Machine Learning
- **scikit-learn** - Algoritmos de ML
- **pandas** - Manipulación de datos
- **numpy** - Computación numérica
- **matplotlib** - Visualizaciones
- **joblib** - Serialización de modelos

### DevOps
- **Docker** - Contenedores
- **Docker Compose** - Orquestación
- **GitHub Actions** - CI/CD
- **Nginx** - Proxy reverso

## 🚀 Instalación y Configuración

### Prerrequisitos
- Docker y Docker Compose
- Python 3.10+
- Node.js 16+
- Git

### 1. Clonar el repositorio
```bash
git clone https://github.com/Midan14/football-analytics.git
cd football-analytics
```

### 2. Configurar variables de entorno
```bash
cp .env.example .env
# Editar .env con tus configuraciones
```

### 3. Levantar con Docker
```bash
# Desarrollo
docker-compose up -d

# Producción
docker-compose -f docker-compose.prod.yml up -d
```

### 4. Desarrollo local

#### Backend
```bash
cd backend
python -m venv football_env
source football_env/bin/activate  # Linux/Mac
# football_env\Scripts\activate  # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload
```

#### Frontend
```bash
cd frontend
npm install
npm start
```

## 📁 Estructura del Proyecto

```
football-analytics/
├── backend/                 # API Backend
│   ├── app/
│   │   ├── api/            # Rutas de la API
│   │   ├── core/           # Configuración principal
│   │   ├── database/       # Modelos y conexión DB
│   │   ├── ml_models/      # Modelos de ML
│   │   ├── services/       # Lógica de negocio
│   │   ├── tests/          # Pruebas unitarias
│   │   └── utils/          # Utilidades
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/               # Aplicación React
│   ├── src/
│   │   ├── components/     # Componentes React
│   │   ├── hooks/          # Custom hooks
│   │   ├── services/       # APIs y servicios
│   │   └── utils/          # Utilidades
│   ├── Dockerfile
│   └── package.json
├── database/               # Scripts SQL
├── docs/                   # Documentación
├── scripts/               # Scripts de utilidad
└── docker-compose.yml
```

## 🔧 API Endpoints

### Autenticación
- `POST /auth/login` - Iniciar sesión
- `POST /auth/register` - Registrar usuario
- `POST /auth/refresh` - Renovar token

### Partidos
- `GET /matches` - Lista de partidos
- `GET /matches/{id}` - Detalles del partido
- `GET /matches/live` - Partidos en vivo
- `POST /matches/{id}/predict` - Predicción del partido

### Equipos
- `GET /teams` - Lista de equipos
- `GET /teams/{id}` - Detalles del equipo
- `GET /teams/{id}/stats` - Estadísticas del equipo

### Jugadores
- `GET /players` - Lista de jugadores
- `GET /players/{id}` - Detalles del jugador
- `GET /players/{id}/injuries` - Historial de lesiones

### Analytics
- `GET /analytics/dashboard` - Datos del dashboard
- `POST /analytics/compare` - Comparar equipos/jugadores
- `GET /analytics/predictions` - Predicciones disponibles

## 🧪 Testing

```bash
# Backend
cd backend
pytest

# Frontend
cd frontend
npm test

# E2E
npm run test:e2e
```

## 📊 Métricas y Monitoreo

- **Prometheus** - Métricas de aplicación
- **Grafana** - Dashboards de monitoreo
- **Sentry** - Tracking de errores
- **Health checks** - Estado de servicios

## 🔐 Seguridad

- Autenticación JWT
- Rate limiting
- Validación de datos
- Sanitización de inputs
- CORS configurado
- Variables de entorno seguras

## 🤝 Contribución

1. Fork el proyecto
2. Crear una rama feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit los cambios (`git commit -m 'Add: nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abrir un Pull Request

### Convenciones de Commits
- `feat:` Nueva funcionalidad
- `fix:` Corrección de bug
- `docs:` Cambios en documentación
- `style:` Cambios de formato
- `refactor:` Refactorización de código
- `test:` Añadir o modificar tests

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 👥 Equipo

- **Miguel Antonio** - *Desarrollador Principal* - [@Midan14](https://github.com/Midan14)

## 🙏 Agradecimientos

- [Football-Data.org](https://www.football-data.org/) - API de datos de fútbol
- [FastAPI](https://fastapi.tiangolo.com/) - Framework web
- [React](https://reactjs.org/) - Biblioteca de UI
- [scikit-learn](https://scikit-learn.org/) - Machine Learning

## 📞 Soporte

Si tienes preguntas o necesitas ayuda:

- 📧 Email: [tu-email@ejemplo.com]
- 💬 Discord: [Enlace al servidor]
- 🐛 Issues: [GitHub Issues](https://github.com/Midan14/football-analytics/issues)

---

⭐ ¡No olvides dar una estrella al proyecto si te ha sido útil!
