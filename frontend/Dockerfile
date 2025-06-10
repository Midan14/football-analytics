# =============================================================================
# FOOTBALL ANALYTICS - DOCKERFILE FRONTEND
# =============================================================================
# Dockerfile multi-stage optimizado para aplicación React de producción
# Genera una imagen minimalista y segura usando nginx

# =============================================================================
# ETAPA 1: BUILD - Construcción de la aplicación React
# =============================================================================
FROM node:18-alpine AS builder

# Metadatos de la imagen
LABEL maintainer="Football Analytics Team"
LABEL description="Frontend React para plataforma de análisis de fútbol"
LABEL version="1.0.0"

# Crear usuario no-root para seguridad durante el build
RUN addgroup -g 1001 -S nodejs && \
    adduser -S reactuser -u 1001

# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias para el build
RUN apk add --no-cache \
    python3 \
    make \
    g++ \
    git \
    && rm -rf /var/cache/apk/*

# Copiar archivos de configuración de dependencias
# Se copian primero para aprovechar el cache de Docker layers
COPY package.json package-lock.json ./

# Configurar npm para optimizar la instalación
RUN npm config set fund false && \
    npm config set audit-level moderate && \
    npm config set progress false && \
    npm config set loglevel warn

# Instalar dependencias con optimizaciones para producción
RUN npm ci \
    --only=production=false \
    --no-audit \
    --no-fund \
    --prefer-offline \
    --frozen-lockfile && \
    npm cache clean --force

# Copiar código fuente de la aplicación
COPY . .

# Cambiar propiedad de archivos al usuario no-root
RUN chown -R reactuser:nodejs /app
USER reactuser

# Variables de entorno para el build de producción
ENV NODE_ENV=production
ENV GENERATE_SOURCEMAP=false
ENV REACT_APP_BUILD_TIME=$BUILD_TIME
ENV REACT_APP_GIT_COMMIT=$GIT_COMMIT
ENV REACT_APP_VERSION=$VERSION

# Construir la aplicación React para producción
RUN npm run build

# Verificar que el build se completó correctamente
RUN test -d build || (echo "❌ Build failed - no build directory found" && exit 1)

# Optimizar archivos estáticos post-build
RUN find build -name "*.js" -exec gzip -k {} \; && \
    find build -name "*.css" -exec gzip -k {} \; && \
    find build -name "*.html" -exec gzip -k {} \;

# =============================================================================
# ETAPA 2: PRODUCTION - Servidor nginx optimizado
# =============================================================================
FROM nginx:1.24-alpine AS production

# Metadatos de la imagen final
LABEL maintainer="Football Analytics Team"
LABEL description="Football Analytics Frontend - Production Ready"
LABEL version="1.0.0"

# Instalar dependencias adicionales para seguridad y monitoreo
RUN apk add --no-cache \
    curl \
    tini \
    && rm -rf /var/cache/apk/*

# Crear usuario no-root para nginx
RUN addgroup -g 101 -S nginx-group && \
    adduser -S nginx-user -u 101 -G nginx-group

# Crear directorios necesarios con permisos correctos
RUN mkdir -p /var/cache/nginx/client_temp \
    /var/cache/nginx/proxy_temp \
    /var/cache/nginx/fastcgi_temp \
    /var/cache/nginx/uwsgi_temp \
    /var/cache/nginx/scgi_temp \
    /var/log/nginx \
    /var/run && \
    chown -R nginx-user:nginx-group /var/cache/nginx \
    /var/log/nginx \
    /var/run \
    /etc/nginx

# Remover configuración por defecto de nginx
RUN rm -rf /usr/share/nginx/html/* && \
    rm /etc/nginx/conf.d/default.conf

# Copiar archivos de build desde la etapa anterior
COPY --from=builder --chown=nginx-user:nginx-group /app/build /usr/share/nginx/html

# Copiar configuración personalizada de nginx
COPY --chown=nginx-user:nginx-group nginx.conf /etc/nginx/nginx.conf
COPY --chown=nginx-user:nginx-group nginx-default.conf /etc/nginx/conf.d/default.conf

# Copiar script de inicio personalizado
COPY --chown=nginx-user:nginx-group docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Variables de entorno para configuración en runtime
ENV NGINX_WORKER_PROCESSES=auto
ENV NGINX_WORKER_CONNECTIONS=1024
ENV NGINX_KEEPALIVE_TIMEOUT=65
ENV NGINX_CLIENT_MAX_BODY_SIZE=10m
ENV NGINX_GZIP_LEVEL=6

# Variables de entorno para la aplicación React
ENV REACT_APP_API_URL=http://localhost:3001/api
ENV REACT_APP_WS_URL=ws://localhost:3001
ENV REACT_APP_ENABLE_ANALYTICS=false
ENV REACT_APP_ENABLE_WEBSOCKETS=true
ENV REACT_APP_ENABLE_NOTIFICATIONS=true

# Exponer puerto 80 (HTTP) y 443 (HTTPS preparado para futuro)
EXPOSE 80 443

# Health check para verificar que la aplicación está funcionando
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:80/health || exit 1

# Cambiar a usuario no-root para mayor seguridad
USER nginx-user

# Usar tini como init para manejo correcto de señales
ENTRYPOINT ["tini", "--"]

# Comando por defecto: ejecutar script de inicio
CMD ["/docker-entrypoint.sh"]

# =============================================================================
# ETAPA 3: DEVELOPMENT - Para desarrollo local con hot reload
# =============================================================================
FROM node:18-alpine AS development

# Metadatos para imagen de desarrollo
LABEL maintainer="Football Analytics Team"
LABEL description="Football Analytics Frontend - Development Environment"
LABEL version="1.0.0-dev"

# Instalar dependencias adicionales para desarrollo
RUN apk add --no-cache \
    git \
    openssh-client \
    && rm -rf /var/cache/apk/*

# Crear usuario para desarrollo
RUN addgroup -g 1001 -S nodejs && \
    adduser -S reactuser -u 1001

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivos de configuración
COPY package.json package-lock.json ./

# Instalar todas las dependencias (incluyendo devDependencies)
RUN npm ci --include=dev && \
    npm cache clean --force

# Cambiar propiedad al usuario de desarrollo
RUN chown -R reactuser:nodejs /app
USER reactuser

# Variables de entorno para desarrollo
ENV NODE_ENV=development
ENV CHOKIDAR_USEPOLLING=true
ENV WATCHPACK_POLLING=true
ENV FAST_REFRESH=true

# Exponer puerto de desarrollo
EXPOSE 3000

# Health check para desarrollo
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:3000 || exit 1

# Comando por defecto para desarrollo
CMD ["npm", "start"]

# =============================================================================
# INFORMACIÓN ADICIONAL PARA BUILDS
# =============================================================================

# Para construir imagen de producción:
# docker build --target production -t football-analytics-frontend:latest .

# Para construir imagen de desarrollo:
# docker build --target development -t football-analytics-frontend:dev .

# Para construir con argumentos de build:
# docker build \
#   --build-arg BUILD_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
#   --build-arg GIT_COMMIT=$(git rev-parse HEAD) \
#   --build-arg VERSION=1.0.0 \
#   --target production \
#   -t football-analytics-frontend:latest .

# Variables de entorno disponibles en runtime:
# - REACT_APP_API_URL: URL del backend API
# - REACT_APP_WS_URL: URL del WebSocket server
# - REACT_APP_GA_TRACKING_ID: Google Analytics tracking ID
# - REACT_APP_SENTRY_DSN: Sentry DSN para error reporting
# - REACT_APP_ENABLE_ANALYTICS: Habilitar/deshabilitar analytics
# - REACT_APP_ENABLE_WEBSOCKETS: Habilitar/deshabilitar WebSockets
# - REACT_APP_ENABLE_NOTIFICATIONS: Habilitar/deshabilitar notificaciones

# Optimizaciones incluidas:
# ✅ Multi-stage build para imagen minimalista
# ✅ Usuario no-root para seguridad
# ✅ Compresión gzip de archivos estáticos
# ✅ Health checks integrados
# ✅ Configuración nginx optimizada
# ✅ Cache de dependencias eficiente
# ✅ Variables de entorno flexibles
# ✅ Soporte para desarrollo y producción
# ✅ Init system (tini) para manejo de señales
# ✅ Limpieza de cache para imagen mínima