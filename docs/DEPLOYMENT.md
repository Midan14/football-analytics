# 🚀 Football Analytics Deployment Guide

> **Guía completa para el deployment de Football Analytics**  
> Versión: 2.1.0 | Última actualización: Junio 2025

## 📋 Tabla de Contenido

- [Visión General](#visión-general)
- [Prerrequisitos](#prerrequisitos)
- [Entornos](#entornos)
- [Configuración Local](#configuración-local)
- [Staging Environment](#staging-environment)
- [Production Deployment](#production-deployment)
- [Docker y Containerización](#docker-y-containerización)
- [CI/CD Pipeline](#cicd-pipeline)
- [Monitoreo y Logging](#monitoreo-y-logging)
- [Backup y Recuperación](#backup-y-recuperación)
- [Escalabilidad](#escalabilidad)
- [Seguridad](#seguridad)
- [Troubleshooting](#troubleshooting)
- [Maintenance](#maintenance)

## 🎯 Visión General

### Arquitectura de Deployment

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   DEVELOPMENT   │    │     STAGING     │    │   PRODUCTION    │
│                 │    │                 │    │                 │
│ • Local SQLite  │    │ • PostgreSQL    │    │ • PostgreSQL HA │
│ • Single Node   │    │ • Redis Single  │    │ • Redis Cluster │
│ • Hot Reload    │    │ • Docker        │    │ • Kubernetes    │
│ • Debug Mode    │    │ • SSL Optional  │    │ • SSL Required  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   MONITORING    │
                    │                 │
                    │ • Prometheus    │
                    │ • Grafana       │
                    │ • ELK Stack     │
                    │ • Alerting      │
                    └─────────────────┘
```

### Tecnologías de Deployment

| Componente | Desarrollo | Staging | Producción |
|------------|------------|---------|------------|
| **App Server** | Uvicorn | Gunicorn + Uvicorn | Gunicorn + Uvicorn |
| **Database** | SQLite | PostgreSQL 15 | PostgreSQL 15 HA |
| **Cache** | Memory | Redis | Redis Cluster |
| **Reverse Proxy** | None | Nginx | Nginx + Load Balancer |
| **Orchestration** | Direct | Docker Compose | Kubernetes |
| **Monitoring** | Basic | Prometheus | Full Stack |

## 📋 Prerrequisitos

### Hardware Mínimo

#### Desarrollo
- **CPU**: 4 cores
- **RAM**: 8 GB
- **Storage**: 50 GB SSD
- **Network**: 10 Mbps

#### Staging
- **CPU**: 4 cores (2.4 GHz+)
- **RAM**: 16 GB
- **Storage**: 100 GB SSD
- **Network**: 100 Mbps

#### Producción
- **CPU**: 8+ cores (3.0 GHz+)
- **RAM**: 32+ GB
- **Storage**: 500+ GB SSD (NVMe)
- **Network**: 1 Gbps+

### Software

#### Todos los Entornos
```bash
# Sistema Operativo
Ubuntu 22.04 LTS (recomendado)
CentOS 8+ / RHEL 8+
Debian 11+

# Contenedores
Docker 24.0+
Docker Compose 2.20+

# Opcional para Producción
Kubernetes 1.27+
Helm 3.12+
```

#### Herramientas de CI/CD
```bash
# Version Control
Git 2.34+

# CI/CD
GitHub Actions
GitLab CI/CD
Jenkins 2.400+

# Infrastructure as Code
Terraform 1.5+
Ansible 2.15+
```

## 🌍 Entornos

### Variables de Entorno por Ambiente

#### Development (.env.dev)
```bash
# Aplicación
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
SECRET_KEY=dev-secret-key-change-in-production

# Base de Datos
DATABASE_URL=sqlite:///data/football_analytics.db
ASYNC_DATABASE_URL=sqlite+aiosqlite:///data/football_analytics.db

# Cache
CACHE_BACKEND=memory
REDIS_URL=redis://localhost:6379/0

# APIs
FOOTBALL_DATA_API_KEY=9c9a42cbff2e8eb387eac2755c5e1e97
RAPIDAPI_KEY=your_rapidapi_key_here
ODDS_API_KEY=your_odds_api_key_here

# Servicios
WEB_HOST=0.0.0.0
WEB_PORT=8000
WEBSOCKET_PORT=8765
WORKERS=1

# Features
ENABLE_WEBSOCKET=true
ENABLE_LIVE_TRACKING=true
ENABLE_VALUE_BETTING=true
```

#### Staging (.env.staging)
```bash
# Aplicación
ENVIRONMENT=staging
DEBUG=false
LOG_LEVEL=INFO
SECRET_KEY=${SECRET_KEY}

# Base de Datos
DATABASE_URL=postgresql://football_user:${DB_PASSWORD}@postgres:5432/football_analytics_staging
ASYNC_DATABASE_URL=postgresql+asyncpg://football_user:${DB_PASSWORD}@postgres:5432/football_analytics_staging

# Cache
CACHE_BACKEND=redis
REDIS_URL=redis://redis:6379/0
REDIS_PASSWORD=${REDIS_PASSWORD}

# APIs
FOOTBALL_DATA_API_KEY=${FOOTBALL_DATA_API_KEY}
RAPIDAPI_KEY=${RAPIDAPI_KEY}
ODDS_API_KEY=${ODDS_API_KEY}

# Servicios
WEB_HOST=0.0.0.0
WEB_PORT=8000
WEBSOCKET_PORT=8765
WORKERS=2

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true

# SSL
SSL_REDIRECT=false
```

#### Production (.env.production)
```bash
# Aplicación
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING
SECRET_KEY=${SECRET_KEY}

# Base de Datos (HA Setup)
DATABASE_URL=postgresql://football_user:${DB_PASSWORD}@postgres-primary:5432/football_analytics
ASYNC_DATABASE_URL=postgresql+asyncpg://football_user:${DB_PASSWORD}@postgres-primary:5432/football_analytics
DATABASE_REPLICA_URL=postgresql://football_user:${DB_PASSWORD}@postgres-replica:5432/football_analytics

# Cache (Cluster)
CACHE_BACKEND=redis_cluster
REDIS_CLUSTER_NODES=redis-1:7000,redis-2:7001,redis-3:7002
REDIS_PASSWORD=${REDIS_PASSWORD}

# APIs
FOOTBALL_DATA_API_KEY=${FOOTBALL_DATA_API_KEY}
RAPIDAPI_KEY=${RAPIDAPI_KEY}
ODDS_API_KEY=${ODDS_API_KEY}

# Servicios
WEB_HOST=0.0.0.0
WEB_PORT=8000
WEBSOCKET_PORT=8765
WORKERS=4

# SSL y Seguridad
SSL_REDIRECT=true
FORCE_HTTPS=true
HSTS_MAX_AGE=31536000

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
SENTRY_DSN=${SENTRY_DSN}

# Alerting
ALERT_EMAIL=${ALERT_EMAIL}
SLACK_WEBHOOK=${SLACK_WEBHOOK}
```

## 💻 Configuración Local

### Setup Rápido

#### 1. Clonar Repositorio
```bash
git clone https://github.com/your-org/football-analytics.git
cd football-analytics
```

#### 2. Configurar Entorno
```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -e .[dev,test,lint]

# Configurar variables de entorno
cp .env.example .env.dev
# Editar .env.dev con tus configuraciones
```

#### 3. Inicializar Base de Datos
```bash
# Crear directorios
mkdir -p data logs models cache backups

# Inicializar base de datos
cd database
sqlite3 ../data/football_analytics.db < 01-create-tables.sql
sqlite3 ../data/football_analytics.db < 02-insert-initial-data.sql
cd ..

# Verificar configuración
python diagnose.py
```

#### 4. Ejecutar Aplicación
```bash
# Opción 1: Directamente
python app/main.py

# Opción 2: Con Uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Opción 3: Con Docker
docker-compose -f docker-compose.dev.yml up
```

### Docker para Desarrollo

#### docker-compose.dev.yml
```yaml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
      target: development
    container_name: football-analytics-dev
    ports:
      - "8000:8000"
      - "8765:8765"
    volumes:
      - .:/app
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - ENVIRONMENT=development
      - DEBUG=true
    env_file:
      - .env.dev
    restart: unless-stopped
    command: ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

  postgres-dev:
    image: postgres:15-alpine
    container_name: postgres-dev
    ports:
      - "5432:5432"
    environment:
      POSTGRES_DB: football_analytics_dev
      POSTGRES_USER: football_user
      POSTGRES_PASSWORD: dev_password
    volumes:
      - postgres_dev_data:/var/lib/postgresql/data
      - ./database:/docker-entrypoint-initdb.d
    restart: unless-stopped

  redis-dev:
    image: redis:7-alpine
    container_name: redis-dev
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_dev_data:/data
    restart: unless-stopped

volumes:
  postgres_dev_data:
  redis_dev_data:

networks:
  default:
    name: football-analytics-dev
```

## 🧪 Staging Environment

### Configuración de Staging

#### docker-compose.staging.yml
```yaml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
      target: production
    container_name: football-analytics-staging
    ports:
      - "8000:8000"
      - "8765:8765"
    environment:
      - ENVIRONMENT=staging
      - DEBUG=false
    env_file:
      - .env.staging
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./models:/app/models
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  postgres:
    image: postgres:15-alpine
    container_name: postgres-staging
    environment:
      POSTGRES_DB: football_analytics_staging
      POSTGRES_USER: football_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_staging_data:/var/lib/postgresql/data
      - ./database:/docker-entrypoint-initdb.d
      - ./backups:/backups
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U football_user -d football_analytics_staging"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: redis-staging
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_staging_data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "--no-auth-warning", "-a", "${REDIS_PASSWORD}", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3

  nginx:
    image: nginx:alpine
    container_name: nginx-staging
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/staging.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl/certs
    depends_on:
      - app
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus-staging
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_staging_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: grafana-staging
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_staging_data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
    restart: unless-stopped

volumes:
  postgres_staging_data:
  redis_staging_data:
  prometheus_staging_data:
  grafana_staging_data:

networks:
  default:
    name: football-analytics-staging
```

### Deployment a Staging

#### Script de Deployment
```bash
#!/bin/bash
# deploy-staging.sh

set -e

echo "🚀 Deploying Football Analytics to Staging..."

# Variables
REPO_URL="https://github.com/your-org/football-analytics.git"
BRANCH="develop"
DEPLOY_DIR="/opt/football-analytics-staging"
BACKUP_DIR="/opt/backups/staging"

# Crear backup
echo "📦 Creating backup..."
mkdir -p $BACKUP_DIR
timestamp=$(date +%Y%m%d_%H%M%S)

# Backup de base de datos
docker exec postgres-staging pg_dump -U football_user football_analytics_staging > "$BACKUP_DIR/db_backup_$timestamp.sql"

# Backup de aplicación
tar -czf "$BACKUP_DIR/app_backup_$timestamp.tar.gz" -C $DEPLOY_DIR .

# Actualizar código
echo "📥 Updating code..."
cd $DEPLOY_DIR
git fetch origin
git checkout $BRANCH
git pull origin $BRANCH

# Construir nueva imagen
echo "🔨 Building application..."
docker-compose -f docker-compose.staging.yml build app

# Ejecutar tests
echo "🧪 Running tests..."
docker-compose -f docker-compose.staging.yml run --rm app pytest tests/ -x

# Ejecutar migraciones
echo "🗄️ Running database migrations..."
docker-compose -f docker-compose.staging.yml run --rm app python scripts/migrate.py

# Deployment con zero-downtime
echo "🔄 Deploying with zero downtime..."

# Iniciar nuevos contenedores
docker-compose -f docker-compose.staging.yml up -d --no-deps app

# Esperar que la aplicación esté lista
echo "⏳ Waiting for application to be ready..."
for i in {1..30}; do
    if curl -f http://localhost:8000/health; then
        echo "✅ Application is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Application failed to start"
        exit 1
    fi
    sleep 10
done

# Actualizar servicios restantes
docker-compose -f docker-compose.staging.yml up -d

# Verificar deployment
echo "🔍 Verifying deployment..."
sleep 30

# Health checks
if ! curl -f http://localhost:8000/health; then
    echo "❌ Health check failed"
    exit 1
fi

# Verificar WebSocket
if ! curl -f --http1.1 --include --no-buffer --header "Connection: Upgrade" --header "Upgrade: websocket" --header "Sec-WebSocket-Key: SGVsbG8sIHdvcmxkIQ==" --header "Sec-WebSocket-Version: 13" http://localhost:8765/; then
    echo "⚠️ WebSocket check failed (might be normal)"
fi

# Limpiar imágenes antigas
echo "🧹 Cleaning up old images..."
docker image prune -f

echo "🎉 Staging deployment completed successfully!"
echo "🌐 Application: http://staging.football-analytics.com"
echo "📊 Metrics: http://staging.football-analytics.com:9090"
echo "📈 Grafana: http://staging.football-analytics.com:3000"
```

## 🏭 Production Deployment

### Arquitectura de Producción

#### Kubernetes Deployment

##### namespace.yaml
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: football-analytics
  labels:
    name: football-analytics
    environment: production
```

##### configmap.yaml
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: football-analytics-config
  namespace: football-analytics
data:
  ENVIRONMENT: "production"
  DEBUG: "false"
  LOG_LEVEL: "WARNING"
  WEB_HOST: "0.0.0.0"
  WEB_PORT: "8000"
  WEBSOCKET_PORT: "8765"
  WORKERS: "4"
  CACHE_BACKEND: "redis_cluster"
  SSL_REDIRECT: "true"
  PROMETHEUS_ENABLED: "true"
```

##### secrets.yaml
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: football-analytics-secrets
  namespace: football-analytics
type: Opaque
stringData:
  SECRET_KEY: "your-super-secret-key-for-production"
  DB_PASSWORD: "your-secure-database-password"
  REDIS_PASSWORD: "your-redis-password"
  FOOTBALL_DATA_API_KEY: "9c9a42cbff2e8eb387eac2755c5e1e97"
  RAPIDAPI_KEY: "your-rapidapi-key"
  ODDS_API_KEY: "your-odds-api-key"
  SENTRY_DSN: "your-sentry-dsn"
  GRAFANA_PASSWORD: "your-grafana-password"
```

##### deployment.yaml
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: football-analytics-app
  namespace: football-analytics
  labels:
    app: football-analytics
    component: app
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: football-analytics
      component: app
  template:
    metadata:
      labels:
        app: football-analytics
        component: app
    spec:
      containers:
      - name: app
        image: football-analytics:latest
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 8765
          name: websocket
        env:
        - name: DATABASE_URL
          value: "postgresql://football_user:$(DB_PASSWORD)@postgres-service:5432/football_analytics"
        - name: REDIS_CLUSTER_NODES
          value: "redis-service:7000"
        envFrom:
        - configMapRef:
            name: football-analytics-config
        - secretRef:
            name: football-analytics-secrets
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        volumeMounts:
        - name: app-data
          mountPath: /app/data
        - name: app-logs
          mountPath: /app/logs
        - name: app-models
          mountPath: /app/models
      volumes:
      - name: app-data
        persistentVolumeClaim:
          claimName: football-analytics-data-pvc
      - name: app-logs
        persistentVolumeClaim:
          claimName: football-analytics-logs-pvc
      - name: app-models
        persistentVolumeClaim:
          claimName: football-analytics-models-pvc
```

##### service.yaml
```yaml
apiVersion: v1
kind: Service
metadata:
  name: football-analytics-service
  namespace: football-analytics
  labels:
    app: football-analytics
    component: app
spec:
  type: ClusterIP
  ports:
  - port: 8000
    targetPort: 8000
    name: http
  - port: 8765
    targetPort: 8765
    name: websocket
  selector:
    app: football-analytics
    component: app
```

##### ingress.yaml
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: football-analytics-ingress
  namespace: football-analytics
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
spec:
  tls:
  - hosts:
    - api.football-analytics.com
    - ws.football-analytics.com
    secretName: football-analytics-tls
  rules:
  - host: api.football-analytics.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: football-analytics-service
            port:
              number: 8000
  - host: ws.football-analytics.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: football-analytics-service
            port:
              number: 8765
```

##### hpa.yaml
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: football-analytics-hpa
  namespace: football-analytics
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: football-analytics-app
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
```

### PostgreSQL High Availability

#### postgresql-primary.yaml
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres-primary
  namespace: football-analytics
spec:
  serviceName: postgres-primary-service
  replicas: 1
  selector:
    matchLabels:
      app: postgres
      role: primary
  template:
    metadata:
      labels:
        app: postgres
        role: primary
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        env:
        - name: POSTGRES_DB
          value: football_analytics
        - name: POSTGRES_USER
          value: football_user
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: football-analytics-secrets
              key: DB_PASSWORD
        - name: POSTGRES_REPLICATION_USER
          value: replicator
        - name: POSTGRES_REPLICATION_PASSWORD
          valueFrom:
            secretKeyRef:
              name: football-analytics-secrets
              key: DB_PASSWORD
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-data
          mountPath: /var/lib/postgresql/data
        - name: postgres-config
          mountPath: /etc/postgresql/postgresql.conf
          subPath: postgresql.conf
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
      volumes:
      - name: postgres-config
        configMap:
          name: postgres-config
  volumeClaimTemplates:
  - metadata:
      name: postgres-data
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 100Gi
```

#### postgresql-replica.yaml
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres-replica
  namespace: football-analytics
spec:
  serviceName: postgres-replica-service
  replicas: 2
  selector:
    matchLabels:
      app: postgres
      role: replica
  template:
    metadata:
      labels:
        app: postgres
        role: replica
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        env:
        - name: PGUSER
          value: football_user
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: football-analytics-secrets
              key: DB_PASSWORD
        - name: POSTGRES_MASTER_SERVICE
          value: postgres-primary-service
        command:
        - /bin/bash
        - -c
        - |
          # Setup replica
          pg_basebackup -h $POSTGRES_MASTER_SERVICE -D /var/lib/postgresql/data -U replicator -v -P -R
          echo "standby_mode = 'on'" >> /var/lib/postgresql/data/recovery.conf
          echo "primary_conninfo = 'host=$POSTGRES_MASTER_SERVICE port=5432 user=replicator'" >> /var/lib/postgresql/data/recovery.conf
          postgres
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-replica-data
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
  volumeClaimTemplates:
  - metadata:
      name: postgres-replica-data
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 100Gi
```

### Redis Cluster

#### redis-cluster.yaml
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis-cluster
  namespace: football-analytics
spec:
  serviceName: redis-cluster-service
  replicas: 6
  selector:
    matchLabels:
      app: redis-cluster
  template:
    metadata:
      labels:
        app: redis-cluster
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        command:
        - redis-server
        - /etc/redis/redis.conf
        - --cluster-enabled
        - "yes"
        - --cluster-config-file
        - /var/lib/redis/nodes.conf
        - --cluster-node-timeout
        - "5000"
        - --appendonly
        - "yes"
        - --requirepass
        - $(REDIS_PASSWORD)
        env:
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: football-analytics-secrets
              key: REDIS_PASSWORD
        ports:
        - containerPort: 6379
          name: client
        - containerPort: 16379
          name: gossip
        volumeMounts:
        - name: redis-data
          mountPath: /var/lib/redis
        - name: redis-config
          mountPath: /etc/redis
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
      volumes:
      - name: redis-config
        configMap:
          name: redis-config
  volumeClaimTemplates:
  - metadata:
      name: redis-data
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 10Gi
```

## 🔄 CI/CD Pipeline

### GitHub Actions

#### .github/workflows/ci.yml
```yaml
name: CI Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test_password
          POSTGRES_USER: test_user
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
        cache: 'pip'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev,test,lint]

    - name: Lint with flake8
      run: |
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 . --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics

    - name: Format check with black
      run: black --check --diff .

    - name: Type check with mypy
      run: mypy app/ --ignore-missing-imports

    - name: Security check with bandit
      run: bandit -r app/ -f json -o bandit-report.json
      continue-on-error: true

    - name: Test with pytest
      env:
        DATABASE_URL: postgresql://test_user:test_password@localhost:5432/test_db
        REDIS_URL: redis://localhost:6379/0
        ENVIRONMENT: testing
      run: |
        pytest tests/ --cov=app --cov-report=xml --cov-report=html -v

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
        fail_ci_if_error: true

  build:
    needs: test
    runs-on: ubuntu-latest
    outputs:
      image-tag: ${{ steps.meta.outputs.tags }}
      image-digest: ${{ steps.build.outputs.digest }}

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3

    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha,prefix={{branch}}-
          type=raw,value=latest,enable={{is_default_branch}}

    - name: Build and push Docker image
      id: build
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
        target: production

  security-scan:
    needs: build
    runs-on: ubuntu-latest
    
    steps:
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: ${{ needs.build.outputs.image-tag }}
        format: 'sarif'
        output: 'trivy-results.sarif'

    - name: Upload Trivy scan results to GitHub Security tab
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'
```

#### .github/workflows/deploy-staging.yml
```yaml
name: Deploy to Staging

on:
  push:
    branches: [ develop ]
  workflow_dispatch:

jobs:
  deploy-staging:
    runs-on: ubuntu-latest
    environment: staging
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Setup SSH
      uses: webfactory/ssh-agent@v0.7.0
      with:
        ssh-private-key: ${{ secrets.STAGING_SSH_PRIVATE_KEY }}

    - name: Deploy to staging server
      run: |
        ssh -o StrictHostKeyChecking=no ${{ secrets.STAGING_USER }}@${{ secrets.STAGING_HOST }} << 'EOF'
          cd /opt/football-analytics-staging
          git pull origin develop
          docker-compose -f docker-compose.staging.yml pull
          docker-compose -f docker-compose.staging.yml up -d
          
          # Wait for health check
          for i in {1..30}; do
            if curl -f http://localhost:8000/health; then
              echo "✅ Staging deployment successful!"
              break
            fi
            sleep 10
          done
        EOF

    - name: Notify deployment
      uses: 8398a7/action-slack@v3
      with:
        status: ${{ job.status }}
        channel: '#deployments'
        webhook_url: ${{ secrets.SLACK_WEBHOOK }}
        message: |
          Staging deployment ${{ job.status }}!
          Branch: ${{ github.ref }}
          Commit: ${{ github.sha }}
          URL: https://staging.football-analytics.com
```

#### .github/workflows/deploy-production.yml
```yaml
name: Deploy to Production

on:
  release:
    types: [published]
  workflow_dispatch:
    inputs:
      version:
        description: 'Version to deploy'
        required: true
        type: string

jobs:
  deploy-production:
    runs-on: ubuntu-latest
    environment: production
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      with:
        ref: ${{ github.event.inputs.version || github.event.release.tag_name }}

    - name: Setup kubectl
      uses: azure/setup-kubectl@v3
      with:
        version: 'latest'

    - name: Configure kubectl
      run: |
        echo "${{ secrets.KUBE_CONFIG }}" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig

    - name: Update image tags
      run: |
        VERSION=${{ github.event.inputs.version || github.event.release.tag_name }}
        sed -i "s|image: football-analytics:.*|image: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${VERSION}|g" k8s/deployment.yaml

    - name: Apply Kubernetes manifests
      run: |
        kubectl apply -f k8s/namespace.yaml
        kubectl apply -f k8s/configmap.yaml
        kubectl apply -f k8s/secrets.yaml
        kubectl apply -f k8s/pvc.yaml
        kubectl apply -f k8s/postgresql.yaml
        kubectl apply -f k8s/redis.yaml
        kubectl apply -f k8s/deployment.yaml
        kubectl apply -f k8s/service.yaml
        kubectl apply -f k8s/ingress.yaml
        kubectl apply -f k8s/hpa.yaml

    - name: Wait for rollout
      run: |
        kubectl rollout status deployment/football-analytics-app -n football-analytics --timeout=600s

    - name: Verify deployment
      run: |
        kubectl get pods -n football-analytics
        kubectl get services -n football-analytics
        
        # Health check
        sleep 30
        if kubectl exec -n football-analytics deployment/football-analytics-app -- curl -f http://localhost:8000/health; then
          echo "✅ Production deployment successful!"
        else
          echo "❌ Production deployment failed!"
          exit 1
        fi

    - name: Notify deployment
      uses: 8398a7/action-slack@v3
      with:
        status: ${{ job.status }}
        channel: '#deployments'
        webhook_url: ${{ secrets.SLACK_WEBHOOK }}
        message: |
          🚀 Production deployment ${{ job.status }}!
          Version: ${{ github.event.inputs.version || github.event.release.tag_name }}
          URL: https://api.football-analytics.com
```

## 📊 Monitoreo y Logging

### Prometheus Configuration

#### monitoring/prometheus.yml
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'football-analytics'
    static_configs:
      - targets: ['app:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx-exporter:9113']

  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
    - role: pod
    relabel_configs:
    - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
      action: keep
      regex: true
    - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
      action: replace
      target_label: __metrics_path__
      regex: (.+)
```

#### monitoring/rules/football-analytics.yml
```yaml
groups:
- name: football-analytics
  rules:
  
  # Application Health
  - alert: ApplicationDown
    expr: up{job="football-analytics"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Football Analytics application is down"
      description: "Application has been down for more than 1 minute"

  - alert: HighResponseTime
    expr: histogram_quantile(0.95, http_request_duration_seconds_bucket{job="football-analytics"}) > 2
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High response time detected"
      description: "95th percentile response time is {{ $value }}s"

  - alert: HighErrorRate
    expr: rate(http_requests_total{job="football-analytics",status=~"5.."}[5m]) > 0.1
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }} requests/second"

  # Database
  - alert: DatabaseConnectionsHigh
    expr: postgresql_connections{} / postgresql_max_connections{} > 0.8
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "PostgreSQL connection usage is high"
      description: "Connection usage is {{ $value | humanizePercentage }}"

  - alert: DatabaseReplicationLag
    expr: postgresql_replication_lag_seconds{} > 30
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "PostgreSQL replication lag is high"
      description: "Replication lag is {{ $value }}s"

  # Redis
  - alert: RedisMemoryHigh
    expr: redis_memory_used_bytes{} / redis_memory_max_bytes{} > 0.9
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Redis memory usage is high"
      description: "Memory usage is {{ $value | humanizePercentage }}"

  # ML Models
  - alert: PredictionAccuracyLow
    expr: prediction_accuracy_ratio{} < 0.6
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "ML prediction accuracy is low"
      description: "Prediction accuracy is {{ $value | humanizePercentage }}"

  - alert: ValueBetVolumeHigh
    expr: increase(value_bets_detected_total{}[1h]) > 100
    for: 0m
    labels:
      severity: info
    annotations:
      summary: "High volume of value bets detected"
      description: "{{ $value }} value bets detected in the last hour"
```

### Grafana Dashboards

#### monitoring/grafana/dashboards/football-analytics-overview.json
```json
{
  "dashboard": {
    "id": null,
    "title": "Football Analytics - Overview",
    "tags": ["football-analytics"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Application Status",
        "type": "stat",
        "targets": [
          {
            "expr": "up{job=\"football-analytics\"}",
            "legendFormat": "Status"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "color": {
              "mode": "thresholds"
            },
            "thresholds": {
              "steps": [
                {"color": "red", "value": 0},
                {"color": "green", "value": 1}
              ]
            },
            "mappings": [
              {"options": {"0": {"text": "DOWN"}}, "type": "value"},
              {"options": {"1": {"text": "UP"}}, "type": "value"}
            ]
          }
        }
      },
      {
        "id": 2,
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{job=\"football-analytics\"}[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "id": 3,
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, http_request_duration_seconds_bucket{job=\"football-analytics\"})",
            "legendFormat": "50th percentile"
          },
          {
            "expr": "histogram_quantile(0.95, http_request_duration_seconds_bucket{job=\"football-analytics\"})",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "id": 4,
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{job=\"football-analytics\",status=~\"4..|5..\"}[5m])",
            "legendFormat": "Error Rate"
          }
        ]
      },
      {
        "id": 5,
        "title": "Predictions per Hour",
        "type": "stat",
        "targets": [
          {
            "expr": "increase(predictions_total{}[1h])",
            "legendFormat": "Predictions"
          }
        ]
      },
      {
        "id": 6,
        "title": "Model Accuracy",
        "type": "gauge",
        "targets": [
          {
            "expr": "prediction_accuracy_ratio{}",
            "legendFormat": "{{model}}"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "min": 0,
            "max": 1,
            "thresholds": {
              "steps": [
                {"color": "red", "value": 0},
                {"color": "yellow", "value": 0.6},
                {"color": "green", "value": 0.75}
              ]
            }
          }
        }
      },
      {
        "id": 7,
        "title": "Value Bets Detected",
        "type": "graph",
        "targets": [
          {
            "expr": "increase(value_bets_detected_total{}[1h])",
            "legendFormat": "Value Bets/Hour"
          }
        ]
      },
      {
        "id": 8,
        "title": "Database Connections",
        "type": "graph",
        "targets": [
          {
            "expr": "postgresql_connections{}",
            "legendFormat": "Active Connections"
          },
          {
            "expr": "postgresql_max_connections{}",
            "legendFormat": "Max Connections"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "5s"
  }
}
```

### ELK Stack Configuration

#### docker-compose.monitoring.yml
```yaml
version: '3.8'

services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.8.0
    container_name: elasticsearch
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - "ES_JAVA_OPTS=-Xms1g -Xmx1g"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    ports:
      - "9200:9200"
    restart: unless-stopped

  logstash:
    image: docker.elastic.co/logstash/logstash:8.8.0
    container_name: logstash
    volumes:
      - ./monitoring/logstash/pipeline:/usr/share/logstash/pipeline
      - ./monitoring/logstash/config:/usr/share/logstash/config
      - ./logs:/var/log/football-analytics
    ports:
      - "5044:5044"
      - "5000:5000/tcp"
      - "5000:5000/udp"
      - "9600:9600"
    environment:
      LS_JAVA_OPTS: "-Xmx512m -Xms512m"
    depends_on:
      - elasticsearch
    restart: unless-stopped

  kibana:
    image: docker.elastic.co/kibana/kibana:8.8.0
    container_name: kibana
    ports:
      - "5601:5601"
    environment:
      ELASTICSEARCH_HOSTS: http://elasticsearch:9200
    depends_on:
      - elasticsearch
    restart: unless-stopped

  filebeat:
    image: docker.elastic.co/beats/filebeat:8.8.0
    container_name: filebeat
    user: root
    volumes:
      - ./monitoring/filebeat/filebeat.yml:/usr/share/filebeat/filebeat.yml:ro
      - ./logs:/var/log/football-analytics:ro
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
      - /var/run/docker.sock:/var/run/docker.sock:ro
    command: filebeat -e -strict.perms=false
    depends_on:
      - logstash
    restart: unless-stopped

volumes:
  elasticsearch_data:
```

## 💾 Backup y Recuperación

### Estrategia de Backup Automatizada

#### scripts/backup-production.sh
```bash
#!/bin/bash
# Production Backup Script

set -e

# Configuration
NAMESPACE="football-analytics"
BACKUP_BUCKET="s3://football-analytics-backups"
RETENTION_DAYS=30
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "🔄 Starting Football Analytics backup..."

# Database Backup
echo "📦 Backing up PostgreSQL database..."
kubectl exec -n $NAMESPACE postgres-primary-0 -- pg_dumpall -U football_user | \
  gzip > "db_backup_${TIMESTAMP}.sql.gz"

# Upload to S3
aws s3 cp "db_backup_${TIMESTAMP}.sql.gz" "${BACKUP_BUCKET}/database/"

# Application Data Backup
echo "💾 Backing up application data..."
kubectl exec -n $NAMESPACE deployment/football-analytics-app -- \
  tar -czf - /app/data /app/models | \
  aws s3 cp - "${BACKUP_BUCKET}/app-data/app_data_${TIMESTAMP}.tar.gz"

# Configuration Backup
echo "⚙️ Backing up Kubernetes configurations..."
kubectl get all,configmap,secret,pvc -n $NAMESPACE -o yaml | \
  gzip > "k8s_config_${TIMESTAMP}.yaml.gz"

aws s3 cp "k8s_config_${TIMESTAMP}.yaml.gz" "${BACKUP_BUCKET}/kubernetes/"

# Redis Backup
echo "💾 Backing up Redis data..."
kubectl exec -n $NAMESPACE redis-cluster-0 -- redis-cli --rdb /tmp/dump.rdb
kubectl cp $NAMESPACE/redis-cluster-0:/tmp/dump.rdb "redis_backup_${TIMESTAMP}.rdb"
aws s3 cp "redis_backup_${TIMESTAMP}.rdb" "${BACKUP_BUCKET}/redis/"

# Cleanup old backups
echo "🧹 Cleaning up old backups..."
aws s3api list-objects-v2 --bucket football-analytics-backups --query "Contents[?LastModified<=\`$(date -d "${RETENTION_DAYS} days ago" --iso-8601)\`].[Key]" --output text | \
  xargs -I {} aws s3 rm s3://football-analytics-backups/{}

# Cleanup local files
rm -f db_backup_*.sql.gz k8s_config_*.yaml.gz redis_backup_*.rdb

echo "✅ Backup completed successfully!"

# Send notification
curl -X POST -H 'Content-type: application/json' \
  --data "{\"text\":\"✅ Football Analytics backup completed: ${TIMESTAMP}\"}" \
  $SLACK_WEBHOOK_URL
```

### Disaster Recovery

#### scripts/disaster-recovery.sh
```bash
#!/bin/bash
# Disaster Recovery Script

set -e

BACKUP_DATE=$1
NAMESPACE="football-analytics"
BACKUP_BUCKET="s3://football-analytics-backups"

if [ -z "$BACKUP_DATE" ]; then
  echo "Usage: $0 <backup_date> (format: YYYYMMDD_HHMMSS)"
  exit 1
fi

echo "🚨 Starting disaster recovery for backup: $BACKUP_DATE"

# Confirm action
read -p "⚠️  This will restore from backup $BACKUP_DATE. Are you sure? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
  echo "❌ Recovery cancelled"
  exit 1
fi

# Scale down application
echo "📉 Scaling down application..."
kubectl scale deployment football-analytics-app --replicas=0 -n $NAMESPACE

# Download backups
echo "📥 Downloading backups..."
aws s3 cp "${BACKUP_BUCKET}/database/db_backup_${BACKUP_DATE}.sql.gz" .
aws s3 cp "${BACKUP_BUCKET}/app-data/app_data_${BACKUP_DATE}.tar.gz" .
aws s3 cp "${BACKUP_BUCKET}/redis/redis_backup_${BACKUP_DATE}.rdb" .

# Restore database
echo "🗄️ Restoring database..."
gunzip "db_backup_${BACKUP_DATE}.sql.gz"
kubectl exec -i -n $NAMESPACE postgres-primary-0 -- psql -U football_user < "db_backup_${BACKUP_DATE}.sql"

# Restore application data
echo "📁 Restoring application data..."
kubectl exec -i -n $NAMESPACE deployment/football-analytics-app -- \
  tar -xzf - -C / < "app_data_${BACKUP_DATE}.tar.gz"

# Restore Redis
echo "💾 Restoring Redis data..."
kubectl cp "redis_backup_${BACKUP_DATE}.rdb" $NAMESPACE/redis-cluster-0:/tmp/dump.rdb
kubectl exec -n $NAMESPACE redis-cluster-0 -- redis-cli --rdb /tmp/dump.rdb RESTORE

# Scale up application
echo "📈 Scaling up application..."
kubectl scale deployment football-analytics-app --replicas=3 -n $NAMESPACE

# Wait for rollout
kubectl rollout status deployment/football-analytics-app -n $NAMESPACE

# Verify restoration
echo "🔍 Verifying restoration..."
sleep 30

if kubectl exec -n $NAMESPACE deployment/football-analytics-app -- curl -f http://localhost:8000/health; then
  echo "✅ Disaster recovery completed successfully!"
else
  echo "❌ Disaster recovery failed - application health check failed"
  exit 1
fi

# Cleanup
rm -f "db_backup_${BACKUP_DATE}.sql" "app_data_${BACKUP_DATE}.tar.gz" "redis_backup_${BACKUP_DATE}.rdb"

# Send notification
curl -X POST -H 'Content-type: application/json' \
  --data "{\"text\":\"✅ Football Analytics disaster recovery completed for backup: ${BACKUP_DATE}\"}" \
  $SLACK_WEBHOOK_URL
```

## 📈 Escalabilidad

### Auto-scaling Configuration

#### Load Testing
```bash
#!/bin/bash
# load-test.sh

echo "🔥 Starting load test on Football Analytics..."

# Install k6 if not available
if ! command -v k6 &> /dev/null; then
  echo "Installing k6..."
  curl -s https://dl.k6.io/key.gpg | sudo apt-key add -
  echo "deb https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
  sudo apt-get update && sudo apt-get install k6
fi

# Run load test
k6 run --vus 100 --duration 10m - <<EOF
import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
  stages: [
    { duration: '2m', target: 20 },   // Ramp up
    { duration: '5m', target: 100 },  // Stay at 100 users
    { duration: '2m', target: 200 },  // Spike test
    { duration: '1m', target: 0 },    // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<2000'], // 95% of requests under 2s
    http_req_failed: ['rate<0.1'],     // Error rate under 10%
  },
};

export default function() {
  // Test predictions endpoint
  let payload = JSON.stringify({
    home_team: 'Real Madrid',
    away_team: 'Barcelona',
    league: 'PD'
  });

  let params = {
    headers: {
      'Content-Type': 'application/json',
    },
  };

  let response = http.post('https://api.football-analytics.com/predict/match', payload, params);
  
  check(response, {
    'status is 200': (r) => r.status === 200,
    'response time < 2000ms': (r) => r.timings.duration < 2000,
  });

  // Test health endpoint
  let health = http.get('https://api.football-analytics.com/health');
  check(health, {
    'health check status is 200': (r) => r.status === 200,
  });

  sleep(1);
}
EOF
```

### Resource Optimization

#### Vertical Pod Autoscaler
```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: football-analytics-vpa
  namespace: football-analytics
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: football-analytics-app
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: app
      maxAllowed:
        cpu: 2
        memory: 4Gi
      minAllowed:
        cpu: 100m
        memory: 128Mi
      controlledResources: ["cpu", "memory"]
```

### Multi-Region Deployment

#### Global Load Balancer (AWS)
```yaml
# aws-global-lb.yaml
apiVersion: v1
kind: Service
metadata:
  name: football-analytics-global
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
    service.beta.kubernetes.io/aws-load-balancer-cross-zone-load-balancing-enabled: "true"
    service.beta.kubernetes.io/aws-load-balancer-backend-protocol: "tcp"
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
  - port: 443
    targetPort: 8000
    protocol: TCP
  selector:
    app: football-analytics
```

## 🔒 Seguridad

### Network Policies

#### network-policy.yaml
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: football-analytics-network-policy
  namespace: football-analytics
spec:
  podSelector:
    matchLabels:
      app: football-analytics
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: nginx-ingress
    ports:
    - protocol: TCP
      port: 8000
    - protocol: TCP
      port: 8765
  - from:
    - namespaceSelector:
        matchLabels:
          name: monitoring
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: football-analytics
    ports:
    - protocol: TCP
      port: 5432  # PostgreSQL
    - protocol: TCP
      port: 6379  # Redis
  - to: []  # Allow external API calls
    ports:
    - protocol: TCP
      port: 443
    - protocol: TCP
      port: 80
```

### Pod Security Standards

#### pod-security-policy.yaml
```yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: football-analytics-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
  readOnlyRootFilesystem: true
```

### Secrets Management

#### External Secrets Operator
```yaml
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: vault-backend
  namespace: football-analytics
spec:
  provider:
    vault:
      server: "https://vault.company.com"
      path: "secret"
      version: "v2"
      auth:
        kubernetes:
          mountPath: "kubernetes"
          role: "football-analytics"

---
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: football-analytics-secrets
  namespace: football-analytics
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: vault-backend
    kind: SecretStore
  target:
    name: football-analytics-secrets
    creationPolicy: Owner
  data:
  - secretKey: SECRET_KEY
    remoteRef:
      key: football-analytics/production
      property: secret_key
  - secretKey: DB_PASSWORD
    remoteRef:
      key: football-analytics/production
      property: db_password
  - secretKey: FOOTBALL_DATA_API_KEY
    remoteRef:
      key: football-analytics/production
      property: football_data_api_key
```

## 🔧 Troubleshooting

### Common Issues

#### Application Won't Start
```bash
# Check pod status
kubectl get pods -n football-analytics

# Check pod logs
kubectl logs -f deployment/football-analytics-app -n football-analytics

# Check events
kubectl get events -n football-analytics --sort-by='.lastTimestamp'

# Check configuration
kubectl describe configmap football-analytics-config -n football-analytics
kubectl describe secret football-analytics-secrets -n football-analytics
```

#### Database Connection Issues
```bash
# Test database connectivity
kubectl exec -it postgres-primary-0 -n football-analytics -- psql -U football_user -d football_analytics -c "SELECT 1;"

# Check database logs
kubectl logs postgres-primary-0 -n football-analytics

# Check connection pool
kubectl exec -it deployment/football-analytics-app -n football-analytics -- python -c "
from app.database import health_check
print(health_check())
"
```

#### Performance Issues
```bash
# Check resource usage
kubectl top pods -n football-analytics
kubectl top nodes

# Check HPA status
kubectl get hpa -n football-analytics

# Check metrics
curl -s http://prometheus:9090/api/v1/query?query=up{job="football-analytics"}
```

### Debug Tools

#### debug-pod.yaml
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: debug-pod
  namespace: football-analytics
spec:
  containers:
  - name: debug
    image: nicolaka/netshoot
    command: ["/bin/bash"]
    args: ["-c", "while true; do ping localhost; sleep 60;done"]
    resources:
      requests:
        memory: "64Mi"
        cpu: "50m"
      limits:
        memory: "128Mi"
        cpu: "100m"
  restartPolicy: Always
```

## 🔧 Maintenance

### Regular Maintenance Tasks

#### Weekly Maintenance Script
```bash
#!/bin/bash
# weekly-maintenance.sh

echo "🔧 Starting weekly maintenance..."

# Database maintenance
echo "🗄️ Database maintenance..."
kubectl exec postgres-primary-0 -n football-analytics -- psql -U football_user -d football_analytics -c "
  VACUUM ANALYZE;
  REINDEX DATABASE football_analytics;
"

# Clean up old logs
echo "🧹 Cleaning up old logs..."
kubectl exec deployment/football-analytics-app -n football-analytics -- find /app/logs -name "*.log" -mtime +7 -delete

# Update statistics
echo "📊 Updating statistics..."
kubectl exec deployment/football-analytics-app -n football-analytics -- python scripts/update_team_stats.py

# Image cleanup
echo "🐳 Cleaning up old Docker images..."
kubectl get nodes -o json | jq -r '.items[].metadata.name' | xargs -I {} kubectl debug node/{} -it --image=alpine -- chroot /host docker system prune -f

# Certificate renewal check
echo "🔐 Checking SSL certificates..."
kubectl get certificates -n football-analytics

echo "✅ Weekly maintenance completed!"
```

### Update Procedures

#### Rolling Update
```bash
#!/bin/bash
# rolling-update.sh

VERSION=$1

if [ -z "$VERSION" ]; then
  echo "Usage: $0 <version>"
  exit 1
fi

echo "🔄 Starting rolling update to version $VERSION..."

# Update image in deployment
kubectl set image deployment/football-analytics-app app=football-analytics:$VERSION -n football-analytics

# Wait for rollout
kubectl rollout status deployment/football-analytics-app -n football-analytics --timeout=600s

# Verify deployment
sleep 30
if kubectl exec -n football-analytics deployment/football-analytics-app -- curl -f http://localhost:8000/health; then
  echo "✅ Rolling update completed successfully!"
else
  echo "❌ Rolling update failed, rolling back..."
  kubectl rollout undo deployment/football-analytics-app -n football-analytics
  exit 1
fi
```

---

## 🎯 Conclusión

Esta guía de deployment proporciona una estrategia completa para desplegar Football Analytics desde desarrollo hasta producción enterprise. Los componentes clave incluyen:

✅ **Multi-Environment**: Configuraciones específicas para cada entorno  
✅ **Containerización**: Docker y Kubernetes para escalabilidad  
✅ **CI/CD**: Pipelines automatizados con GitHub Actions  
✅ **Monitoreo**: Stack completo con Prometheus, Grafana y ELK  
✅ **Alta Disponibilidad**: PostgreSQL HA y Redis Cluster  
✅ **Seguridad**: Network policies, secrets management y SSL  
✅ **Backup**: Estrategias automatizadas y disaster recovery  
✅ **Escalabilidad**: Auto-scaling y load balancing  

### Próximos Pasos

1. **Implementar infrastructure as code** con Terraform
2. **Configurar multi-región** para disaster recovery
3. **Optimizar costos** con spot instances y resource optimization
4. **Implementar chaos engineering** para testing de resiliencia
5. **Agregar observabilidad** avanzada con tracing distribuido

---

**🚀 Football Analytics Deployment - Llevando el análisis deportivo a escala global** ⚽🌍