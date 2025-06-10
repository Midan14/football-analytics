# =============================================================================
# FOOTBALL ANALYTICS - MAKEFILE
# =============================================================================
# Automatización de tareas para la plataforma de análisis de fútbol
# Incluye: desarrollo, testing, deployment, Docker, base de datos, etc.

# =============================================================================
# VARIABLES DE CONFIGURACIÓN
# =============================================================================
# Colores para output
RED=\033[0;31m
GREEN=\033[0;32m
YELLOW=\033[1;33m
BLUE=\033[0;34m
PURPLE=\033[0;35m
CYAN=\033[0;36m
WHITE=\033[1;37m
NC=\033[0m # No Color

# Configuración del proyecto
PROJECT_NAME=football-analytics
VERSION=$(shell grep '"version"' frontend/package.json | cut -d '"' -f 4)
BUILD_TIME=$(shell date -u +"%Y-%m-%dT%H:%M:%SZ")
GIT_COMMIT=$(shell git rev-parse HEAD 2>/dev/null || echo "unknown")

# Puertos por defecto
FRONTEND_PORT=3000
BACKEND_PORT=3001
DB_PORT=5432
REDIS_PORT=6379

# Docker configuración
DOCKER_COMPOSE=docker-compose
DOCKER_BUILD_ARGS=--build-arg BUILD_TIME=$(BUILD_TIME) --build-arg GIT_COMMIT=$(GIT_COMMIT) --build-arg VERSION=$(VERSION)

# Python/Node configuración
PYTHON=python3
NODE=node
NPM=npm
PIP=pip3

# =============================================================================
# AYUDA - Comando por defecto
# =============================================================================
.DEFAULT_GOAL := help

help: ## 📋 Mostrar esta ayuda
	@echo "$(CYAN)⚽ Football Analytics - Comandos Disponibles$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(YELLOW)🔧 Comandos de desarrollo:$(NC)"
	@echo "  make setup      - Configuración inicial completa"
	@echo "  make dev        - Iniciar desarrollo"
	@echo "  make test       - Ejecutar todos los tests"
	@echo ""
	@echo "$(BLUE)🐳 Comandos de Docker:$(NC)"
	@echo "  make docker-dev - Desarrollo con Docker"
	@echo "  make docker-prod - Producción con Docker"
	@echo ""
	@echo "$(PURPLE)📊 Comandos de base de datos:$(NC)"
	@echo "  make db-setup   - Configurar base de datos"
	@echo "  make db-seed    - Poblar con datos de ejemplo"

# =============================================================================
# CONFIGURACIÓN INICIAL
# =============================================================================
setup: ## 🚀 Configuración inicial completa del proyecto
	@echo "$(CYAN)🚀 Configurando Football Analytics...$(NC)"
	$(MAKE) check-dependencies
	$(MAKE) setup-env
	$(MAKE) setup-frontend
	$(MAKE) setup-backend
	$(MAKE) setup-database
	@echo "$(GREEN)✅ Configuración completada!$(NC)"

check-dependencies: ## 🔍 Verificar dependencias del sistema
	@echo "$(YELLOW)🔍 Verificando dependencias...$(NC)"
	@command -v $(NODE) >/dev/null 2>&1 || { echo "$(RED)❌ Node.js no encontrado. Instalar desde https://nodejs.org$(NC)"; exit 1; }
	@command -v $(PYTHON) >/dev/null 2>&1 || { echo "$(RED)❌ Python3 no encontrado$(NC)"; exit 1; }
	@command -v docker >/dev/null 2>&1 || { echo "$(RED)❌ Docker no encontrado$(NC)"; exit 1; }
	@command -v docker-compose >/dev/null 2>&1 || { echo "$(RED)❌ Docker Compose no encontrado$(NC)"; exit 1; }
	@echo "$(GREEN)✅ Todas las dependencias están instaladas$(NC)"

setup-env: ## 📝 Configurar variables de entorno
	@echo "$(YELLOW)📝 Configurando variables de entorno...$(NC)"
	@if [ ! -f .env ]; then \
		echo "$(YELLOW)⚠️  Creando .env desde .env.example...$(NC)"; \
		cp .env.example .env 2>/dev/null || echo "$(RED)❌ .env.example no encontrado$(NC)"; \
	fi
	@echo "$(GREEN)✅ Variables de entorno configuradas$(NC)"
	@echo "$(YELLOW)⚠️  Revisa y actualiza el archivo .env con tus configuraciones$(NC)"

setup-frontend: ## 🎨 Configurar frontend React
	@echo "$(YELLOW)🎨 Configurando frontend...$(NC)"
	@if [ -d "frontend" ]; then \
		cd frontend && $(NPM) install; \
		echo "$(GREEN)✅ Frontend configurado$(NC)"; \
	else \
		echo "$(RED)❌ Directorio frontend no encontrado$(NC)"; \
	fi

setup-backend: ## ⚙️ Configurar backend y entorno virtual Python
	@echo "$(YELLOW)⚙️ Configurando backend...$(NC)"
	@if [ ! -d "venv" ]; then \
		$(PYTHON) -m venv venv; \
		echo "$(GREEN)✅ Entorno virtual creado$(NC)"; \
	fi
	@. venv/bin/activate && $(PIP) install -r requirements.txt 2>/dev/null || echo "$(YELLOW)⚠️  requirements.txt no encontrado$(NC)"
	@echo "$(GREEN)✅ Backend configurado$(NC)"

setup-database: ## 🗄️ Configurar base de datos
	@echo "$(YELLOW)🗄️ Configurando base de datos...$(NC)"
	@if [ -f "database/init.sql" ]; then \
		echo "$(GREEN)✅ Scripts de base de datos encontrados$(NC)"; \
	else \
		mkdir -p database; \
		echo "$(YELLOW)⚠️  Crear scripts de inicialización en database/$(NC)"; \
	fi

# =============================================================================
# DESARROLLO
# =============================================================================
dev: ## 🔥 Iniciar desarrollo (frontend + backend)
	@echo "$(CYAN)🔥 Iniciando modo desarrollo...$(NC)"
	$(MAKE) -j2 dev-frontend dev-backend

dev-frontend: ## 🎨 Iniciar solo frontend
	@echo "$(YELLOW)🎨 Iniciando frontend en puerto $(FRONTEND_PORT)...$(NC)"
	@cd frontend && $(NPM) start

dev-backend: ## ⚙️ Iniciar solo backend
	@echo "$(YELLOW)⚙️ Iniciando backend en puerto $(BACKEND_PORT)...$(NC)"
	@. venv/bin/activate && cd backend && $(PYTHON) app.py

dev-full: ## 🚀 Desarrollo completo con servicios
	@echo "$(CYAN)🚀 Iniciando desarrollo completo...$(NC)"
	$(DOCKER_COMPOSE) up frontend backend postgres redis

# =============================================================================
# TESTING
# =============================================================================
test: ## 🧪 Ejecutar todos los tests
	@echo "$(CYAN)🧪 Ejecutando tests...$(NC)"
	$(MAKE) test-frontend
	$(MAKE) test-backend

test-frontend: ## 🎨 Tests del frontend
	@echo "$(YELLOW)🎨 Testing frontend...$(NC)"
	@cd frontend && $(NPM) test -- --coverage --watchAll=false

test-backend: ## ⚙️ Tests del backend
	@echo "$(YELLOW)⚙️ Testing backend...$(NC)"
	@. venv/bin/activate && $(PYTHON) -m pytest tests/ -v

test-e2e: ## 🔄 Tests end-to-end
	@echo "$(YELLOW)🔄 Testing E2E...$(NC)"
	@cd frontend && $(NPM) run test:e2e

lint: ## 📏 Linting y formateo
	@echo "$(CYAN)📏 Ejecutando linting...$(NC)"
	$(MAKE) lint-frontend
	$(MAKE) lint-backend

lint-frontend: ## 🎨 Lint frontend
	@echo "$(YELLOW)🎨 Linting frontend...$(NC)"
	@cd frontend && $(NPM) run lint
	@cd frontend && $(NPM) run format:check

lint-backend: ## ⚙️ Lint backend
	@echo "$(YELLOW)⚙️ Linting backend...$(NC)"
	@. venv/bin/activate && flake8 backend/
	@. venv/bin/activate && black --check backend/

fix-lint: ## 🔧 Corregir problemas de linting
	@echo "$(CYAN)🔧 Corrigiendo linting...$(NC)"
	@cd frontend && $(NPM) run lint:fix
	@cd frontend && $(NPM) run format
	@. venv/bin/activate && black backend/

# =============================================================================
# BUILD Y DEPLOYMENT
# =============================================================================
build: ## 🏗️ Build para producción
	@echo "$(CYAN)🏗️ Building para producción...$(NC)"
	$(MAKE) build-frontend
	$(MAKE) build-backend

build-frontend: ## 🎨 Build frontend
	@echo "$(YELLOW)🎨 Building frontend...$(NC)"
	@cd frontend && $(NPM) run build

build-backend: ## ⚙️ Build backend
	@echo "$(YELLOW)⚙️ Building backend...$(NC)"
	@. venv/bin/activate && $(PYTHON) -m py_compile backend/*.py

deploy-staging: ## 🚀 Deploy a staging
	@echo "$(YELLOW)🚀 Deploying to staging...$(NC)"
	$(MAKE) build
	$(MAKE) docker-build
	@echo "$(GREEN)✅ Deploy a staging completado$(NC)"

deploy-production: ## 🌟 Deploy a producción
	@echo "$(RED)🌟 Deploying to production...$(NC)"
	@read -p "¿Estás seguro? (y/N): " confirm && [ "$$confirm" = "y" ]
	$(MAKE) build
	$(MAKE) docker-build
	$(MAKE) docker-prod
	@echo "$(GREEN)✅ Deploy a producción completado$(NC)"

# =============================================================================
# DOCKER
# =============================================================================
docker-build: ## 🐳 Build imágenes Docker
	@echo "$(CYAN)🐳 Building imágenes Docker...$(NC)"
	$(DOCKER_COMPOSE) build $(DOCKER_BUILD_ARGS)

docker-dev: ## 🔥 Desarrollo con Docker
	@echo "$(YELLOW)🔥 Iniciando desarrollo con Docker...$(NC)"
	$(DOCKER_COMPOSE) up --build frontend backend postgres redis

docker-prod: ## 🌟 Producción con Docker
	@echo "$(RED)🌟 Iniciando producción con Docker...$(NC)"
	$(DOCKER_COMPOSE) --profile production up -d

docker-ml: ## 🤖 Servicios con Machine Learning
	@echo "$(PURPLE)🤖 Iniciando con ML...$(NC)"
	$(DOCKER_COMPOSE) --profile ml up --build

docker-stop: ## ⏹️ Detener servicios Docker
	@echo "$(YELLOW)⏹️ Deteniendo servicios...$(NC)"
	$(DOCKER_COMPOSE) down

docker-clean: ## 🧹 Limpiar Docker
	@echo "$(RED)🧹 Limpiando Docker...$(NC)"
	$(DOCKER_COMPOSE) down -v --remove-orphans
	docker system prune -f

docker-logs: ## 📋 Ver logs de Docker
	@echo "$(CYAN)📋 Mostrando logs...$(NC)"
	$(DOCKER_COMPOSE) logs -f

# =============================================================================
# BASE DE DATOS
# =============================================================================
db-setup: ## 🗄️ Configurar base de datos
	@echo "$(CYAN)🗄️ Configurando base de datos...$(NC)"
	$(DOCKER_COMPOSE) up -d postgres
	sleep 5
	$(MAKE) db-migrate

db-migrate: ## 📦 Ejecutar migraciones
	@echo "$(YELLOW)📦 Ejecutando migraciones...$(NC)"
	@. venv/bin/activate && cd backend && $(PYTHON) manage.py migrate

db-seed: ## 🌱 Poblar base de datos con datos de ejemplo
	@echo "$(GREEN)🌱 Poblando base de datos...$(NC)"
	@. venv/bin/activate && cd backend && $(PYTHON) manage.py seed

db-reset: ## 🔄 Resetear base de datos
	@echo "$(RED)🔄 Reseteando base de datos...$(NC)"
	@read -p "¿Estás seguro? Esto eliminará todos los datos (y/N): " confirm && [ "$$confirm" = "y" ]
	$(DOCKER_COMPOSE) down -v postgres
	$(DOCKER_COMPOSE) up -d postgres
	sleep 5
	$(MAKE) db-migrate
	$(MAKE) db-seed

db-backup: ## 💾 Backup de base de datos
	@echo "$(YELLOW)💾 Creando backup...$(NC)"
	mkdir -p database/backups
	$(DOCKER_COMPOSE) exec postgres pg_dump -U football_user football_analytics_db > database/backups/backup-$(shell date +%Y%m%d-%H%M%S).sql
	@echo "$(GREEN)✅ Backup creado en database/backups/$(NC)"

db-shell: ## 🐚 Conectar a base de datos
	@echo "$(CYAN)🐚 Conectando a base de datos...$(NC)"
	$(DOCKER_COMPOSE) exec postgres psql -U football_user -d football_analytics_db

# =============================================================================
# API Y DATOS
# =============================================================================
api-test: ## 🔌 Probar APIs externas
	@echo "$(CYAN)🔌 Probando APIs...$(NC)"
	@. venv/bin/activate && cd backend && $(PYTHON) test_apis.py

data-fetch: ## 📥 Obtener datos de APIs de fútbol
	@echo "$(GREEN)📥 Obteniendo datos de fútbol...$(NC)"
	@. venv/bin/activate && cd backend && $(PYTHON) fetch_data.py

data-update: ## 🔄 Actualizar datos en tiempo real
	@echo "$(YELLOW)🔄 Actualizando datos...$(NC)"
	@. venv/bin/activate && cd backend && $(PYTHON) update_live_data.py

# =============================================================================
# MACHINE LEARNING
# =============================================================================
ml-train: ## 🤖 Entrenar modelos ML
	@echo "$(PURPLE)🤖 Entrenando modelos...$(NC)"
	@. venv/bin/activate && cd ml-service && $(PYTHON) train_models.py

ml-predict: ## 🔮 Generar predicciones
	@echo "$(PURPLE)🔮 Generando predicciones...$(NC)"
	@. venv/bin/activate && cd ml-service && $(PYTHON) generate_predictions.py

ml-evaluate: ## 📊 Evaluar modelos
	@echo "$(PURPLE)📊 Evaluando modelos...$(NC)"
	@. venv/bin/activate && cd ml-service && $(PYTHON) evaluate_models.py

# =============================================================================
# MONITOREO Y LOGS
# =============================================================================
logs: ## 📋 Ver logs de la aplicación
	@echo "$(CYAN)📋 Mostrando logs...$(NC)"
	tail -f logs/*.log 2>/dev/null || echo "$(YELLOW)⚠️  No hay archivos de log$(NC)"

logs-frontend: ## 🎨 Logs del frontend
	@echo "$(YELLOW)🎨 Logs del frontend...$(NC)"
	$(DOCKER_COMPOSE) logs -f frontend

logs-backend: ## ⚙️ Logs del backend
	@echo "$(YELLOW)⚙️ Logs del backend...$(NC)"
	$(DOCKER_COMPOSE) logs -f backend

monitor: ## 📊 Iniciar monitoreo
	@echo "$(CYAN)📊 Iniciando monitoreo...$(NC)"
	$(DOCKER_COMPOSE) --profile monitoring up -d prometheus grafana

# =============================================================================
# LIMPIEZA Y MANTENIMIENTO
# =============================================================================
clean: ## 🧹 Limpiar archivos temporales
	@echo "$(RED)🧹 Limpiando archivos temporales...$(NC)"
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name ".DS_Store" -delete 2>/dev/null || true
	cd frontend && $(NPM) run clean 2>/dev/null || true
	@echo "$(GREEN)✅ Limpieza completada$(NC)"

clean-all: ## 🗑️ Limpieza completa (incluye dependencias)
	@echo "$(RED)🗑️ Limpieza completa...$(NC)"
	$(MAKE) clean
	$(MAKE) docker-clean
	rm -rf frontend/node_modules
	rm -rf venv
	@echo "$(GREEN)✅ Limpieza completa terminada$(NC)"

reset: ## 🔄 Reset completo del proyecto
	@echo "$(RED)🔄 Reset completo del proyecto...$(NC)"
	@read -p "¿Estás seguro? Esto eliminará todo (y/N): " confirm && [ "$$confirm" = "y" ]
	$(MAKE) clean-all
	$(MAKE) setup
	@echo "$(GREEN)✅ Reset completado$(NC)"

# =============================================================================
# SEGURIDAD
# =============================================================================
security-check: ## 🔒 Verificar seguridad
	@echo "$(CYAN)🔒 Verificando seguridad...$(NC)"
	@cd frontend && $(NPM) audit
	@. venv/bin/activate && safety check

update-deps: ## ⬆️ Actualizar dependencias
	@echo "$(YELLOW)⬆️ Actualizando dependencias...$(NC)"
	@cd frontend && $(NPM) update
	@. venv/bin/activate && $(PIP) list --outdated

# =============================================================================
# INFORMACIÓN Y DIAGNÓSTICO
# =============================================================================
info: ## ℹ️ Información del proyecto
	@echo "$(CYAN)⚽ Football Analytics - Información del Proyecto$(NC)"
	@echo "$(WHITE)Nombre:$(NC) $(PROJECT_NAME)"
	@echo "$(WHITE)Versión:$(NC) $(VERSION)"
	@echo "$(WHITE)Build Time:$(NC) $(BUILD_TIME)"
	@echo "$(WHITE)Git Commit:$(NC) $(GIT_COMMIT)"
	@echo ""
	@echo "$(WHITE)Puertos:$(NC)"
	@echo "  Frontend: http://localhost:$(FRONTEND_PORT)"
	@echo "  Backend:  http://localhost:$(BACKEND_PORT)"
	@echo "  Database: localhost:$(DB_PORT)"
	@echo "  Redis:    localhost:$(REDIS_PORT)"
	@echo ""
	@echo "$(WHITE)Estado de servicios:$(NC)"
	@$(DOCKER_COMPOSE) ps 2>/dev/null || echo "  Docker no iniciado"

status: ## 📊 Estado de servicios
	@echo "$(CYAN)📊 Estado de servicios...$(NC)"
	@$(DOCKER_COMPOSE) ps

health: ## 🏥 Health check de servicios
	@echo "$(CYAN)🏥 Verificando salud de servicios...$(NC)"
	@curl -f http://localhost:$(FRONTEND_PORT)/health 2>/dev/null && echo "$(GREEN)✅ Frontend OK$(NC)" || echo "$(RED)❌ Frontend DOWN$(NC)"
	@curl -f http://localhost:$(BACKEND_PORT)/health 2>/dev/null && echo "$(GREEN)✅ Backend OK$(NC)" || echo "$(RED)❌ Backend DOWN$(NC)"

# =============================================================================
# SHORTCUTS Y ALIASES
# =============================================================================
start: dev ## 🚀 Alias para 'make dev'
stop: docker-stop ## ⏹️ Alias para 'make docker-stop'
restart: ## 🔄 Reiniciar servicios
	$(MAKE) stop
	$(MAKE) start

install: setup ## 📦 Alias para 'make setup'
run: dev ## ▶️ Alias para 'make dev'

# =============================================================================
# CONFIGURACIÓN ESPECIAL
# =============================================================================
.PHONY: help setup check-dependencies setup-env setup-frontend setup-backend setup-database \
        dev dev-frontend dev-backend dev-full \
        test test-frontend test-backend test-e2e lint lint-frontend lint-backend fix-lint \
        build build-frontend build-backend deploy-staging deploy-production \
        docker-build docker-dev docker-prod docker-ml docker-stop docker-clean docker-logs \
        db-setup db-migrate db-seed db-reset db-backup db-shell \
        api-test data-fetch data-update \
        ml-train ml-predict ml-evaluate \
        logs logs-frontend logs-backend monitor \
        clean clean-all reset \
        security-check update-deps \
        info status health \
        start stop restart install run

# Silenciar comandos por defecto
.SILENT: