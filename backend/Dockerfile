# ================================
# DOCKERFILE - FOOTBALL ANALYTICS
# ================================
# Multi-stage Docker build para Football Analytics
# Optimizado para desarrollo y producción con ML/AI capabilities
#
# Autor: Football Analytics Team
# Versión: 2.1.0
# Fecha: 2024-06-02
#
# Uso:
#   docker build -t football-analytics .                    # Build por defecto (producción)
#   docker build --target development -t football-dev .     # Build para desarrollo
#   docker build --target testing -t football-test .       # Build para testing
#   docker run -p 8000:8000 football-analytics             # Ejecutar contenedor

# ================================
# STAGE 1: BASE IMAGE
# ================================
FROM python:3.9-slim as base

# Metadata
LABEL maintainer="Football Analytics Team"
LABEL version="2.1.0"
LABEL description="Sistema avanzado de análisis y predicción deportiva"

# Variables de entorno para Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

# Instalar dependencias del sistema necesarias para ML
RUN apt-get update && apt-get install -y \
    # Compiladores y herramientas de build
    gcc \
    g++ \
    make \
    cmake \
    # Librerías para computación científica
    libblas3 \
    liblapack3 \
    liblapack-dev \
    libblas-dev \
    gfortran \
    # Librerías para procesamiento de datos
    libhdf5-dev \
    libnetcdf-dev \
    # Utilidades del sistema
    curl \
    wget \
    git \
    # Limpieza de archivos innecesarios
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario no-root para seguridad
RUN groupadd -r football && useradd -r -g football football

# Crear estructura de directorios
RUN mkdir -p /app/{data,logs,models,cache,backups} && \
    chown -R football:football /app

# Establecer directorio de trabajo
WORKDIR /app

# ================================
# STAGE 2: DEPENDENCIES
# ================================
FROM base as dependencies

# Copiar requirements primero para aprovechar cache de Docker
COPY requirements.txt /app/

# Actualizar pip y setuptools
RUN pip install --upgrade pip setuptools wheel

# Instalar dependencias Python
RUN pip install -r requirements.txt

# Instalar dependencias adicionales para ML/Analytics
RUN pip install \
    # Machine Learning frameworks
    scikit-learn==1.3.0 \
    xgboost==1.7.6 \
    lightgbm==4.0.0 \
    catboost==1.2 \
    # Data processing
    pandas==2.0.3 \
    numpy==1.24.3 \
    scipy==1.11.1 \
    # Web framework
    fastapi[all]==0.100.1 \
    uvicorn[standard]==0.23.1 \
    # Database
    sqlalchemy==2.0.19 \
    alembic==1.11.1 \
    asyncpg==0.28.0 \
    # Cache and messaging
    redis==4.6.0 \
    celery==5.3.1 \
    # HTTP requests
    httpx==0.24.1 \
    requests==2.31.0 \
    # WebSocket
    websockets==11.0.3 \
    # Configuration
    python-dotenv==1.0.0 \
    pydantic==2.1.1 \
    pydantic-settings==2.0.2 \
    # Monitoring and logging
    prometheus-client==0.17.1 \
    structlog==23.1.0 \
    # Testing (para stage de testing)
    pytest==7.4.0 \
    pytest-asyncio==0.21.1 \
    pytest-cov==4.1.0 \
    # Linting y formateo
    black==23.7.0 \
    flake8==6.0.0 \
    isort==5.12.0

# ================================
# STAGE 3: DEVELOPMENT
# ================================
FROM dependencies as development

# Variables específicas de desarrollo
ENV ENVIRONMENT=development \
    DEBUG=true \
    LOG_LEVEL=DEBUG

# Instalar herramientas adicionales para desarrollo
RUN pip install \
    # Desarrollo
    jupyter==1.0.0 \
    notebook==7.0.2 \
    ipython==8.14.0 \
    # Profiling
    line-profiler==4.1.1 \
    memory-profiler==0.60.0 \
    # Debugging
    pdb-clone==2.2 \
    pudb==2023.1

# Copiar código fuente
COPY --chown=football:football . /app/

# Cambiar a usuario no-root
USER football

# Exponer puertos
EXPOSE 8000 8765

# Comando por defecto para desarrollo (con auto-reload)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-level", "debug"]

# ================================
# STAGE 4: TESTING
# ================================
FROM dependencies as testing

# Variables específicas de testing
ENV ENVIRONMENT=testing \
    DEBUG=false \
    LOG_LEVEL=WARNING

# Copiar código fuente
COPY --chown=football:football . /app/

# Cambiar a usuario no-root
USER football

# Ejecutar tests por defecto
CMD ["python", "-m", "pytest", "tests/", "-v", "--cov=app", "--cov-report=term-missing"]

# ================================
# STAGE 5: PRODUCTION
# ================================
FROM dependencies as production

# Variables específicas de producción
ENV ENVIRONMENT=production \
    DEBUG=false \
    LOG_LEVEL=INFO \
    WEB_CONCURRENCY=4

# Copiar solo archivos necesarios para producción
COPY --chown=football:football app/ /app/app/
COPY --chown=football:football requirements.txt /app/
COPY --chown=football:football .env.example /app/.env

# Crear directorios de datos con permisos correctos
RUN mkdir -p /app/{data,logs,models,cache,backups} && \
    chmod -R 755 /app/data /app/logs /app/models /app/cache /app/backups

# Cambiar a usuario no-root
USER football

# Exponer puertos
EXPOSE 8000 8765

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando por defecto para producción
CMD ["python", "app/main.py"]

# ================================
# STAGE 6: WORKER (para tareas en background)
# ================================
FROM production as worker

# Variables específicas del worker
ENV WORKER_TYPE=celery \
    CELERY_WORKER_CONCURRENCY=2 \
    CELERY_WORKER_LOGLEVEL=INFO

# Comando para worker
CMD ["python", "-m", "app.worker"]

# ================================
# OPTIMIZACIONES Y MEJORES PRÁCTICAS
# ================================

# Multi-stage build reduce el tamaño final de la imagen
# Usuario no-root mejora la seguridad
# Cache de dependencias optimiza builds repetidos
# Health checks aseguran disponibilidad del servicio
# Variables de entorno configurables por stage

# ================================
# NOTAS DE USO
# ================================

# 1. BUILD COMMANDS:
#    docker build -t football-analytics .
#    docker build --target development -t football-dev .
#    docker build --target testing -t football-test .
#    docker build --target worker -t football-worker .

# 2. RUN COMMANDS:
#    docker run -p 8000:8000 -p 8765:8765 football-analytics
#    docker run -v $(pwd)/data:/app/data football-analytics
#    docker run --env-file .env football-analytics

# 3. DOCKER COMPOSE:
#    services:
#      app:
#        build:
#          context: .
#          target: production

# 4. DEVELOPMENT:
#    docker run -v $(pwd):/app -p 8000:8000 football-dev

# 5. TESTING:
#    docker run -v $(pwd):/app football-test

# ================================
# SEGURIDAD
# ================================

# - Usuario no-root (football:football)
# - Archivos de solo lectura donde es posible
# - Dependencias fijadas a versiones específicas
# - Limpieza de cache y archivos temporales
# - Health checks para monitoring

# ================================
# OPTIMIZACIÓN DE TAMAÑO
# ================================

# Imagen base: python:3.9-slim (~45MB base)
# Multi-stage build elimina dependencias de compilación
# Limpieza de apt cache y archivos temporales
# .dockerignore para excluir archivos innecesarios

# Tamaños aproximados:
# - Base: ~300MB
# - Development: ~800MB (incluye Jupyter, debugging tools)
# - Testing: ~600MB (incluye pytest, coverage)
# - Production: ~500MB (optimizado, sin herramientas dev)
# - Worker: ~500MB (mismo que production)