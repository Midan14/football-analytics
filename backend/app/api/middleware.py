"""
Middleware personalizado para Football Analytics API

Este módulo contiene middleware para logging, rate limiting, CORS,
autenticación, monitoreo y seguridad de la API.
"""

import json
import logging
import time
import uuid
from datetime import datetime
from typing import Callable

from fastapi import Request, Response, status
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp

from app.config import settings

# from app.core.redis_client import redis_client  # TODO: implement redis client

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================
# MIDDLEWARE DE LOGGING Y MONITOREO
# =====================================================


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware para logging detallado de todas las requests.
    Registra información de timing, IP, user agent, etc.
    """

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        # Generar ID único para la request
        request_id = str(uuid.uuid4())
        start_time = time.time()

        # Información de la request
        client_ip = self.get_client_ip(request)
        user_agent = request.headers.get("user-agent", "Unknown")

        # Log de request entrante
        logger.info(
            f"REQUEST_START - ID: {request_id} | "
            f"Method: {request.method} | "
            f"URL: {str(request.url)} | "
            f"IP: {client_ip} | "
            f"User-Agent: {user_agent}"
        )

        # Agregar request_id a los headers para tracking
        request.state.request_id = request_id

        try:
            # Procesar request
            response = await call_next(request)

            # Calcular tiempo de procesamiento
            process_time = time.time() - start_time

            # Log de response
            logger.info(
                f"REQUEST_END - ID: {request_id} | "
                f"Status: {response.status_code} | "
                f"Duration: {process_time:.4f}s"
            )

            # Agregar headers de timing y tracking
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)

            # Guardar métricas en Redis para monitoreo
            await self.save_metrics(request, response, process_time, client_ip)

            return response

        except Exception as e:
            process_time = time.time() - start_time

            # Log de error
            logger.error(
                f"REQUEST_ERROR - ID: {request_id} | "
                f"Error: {str(e)} | "
                f"Duration: {process_time:.4f}s"
            )

            # Response de error
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "request_id": request_id,
                    "timestamp": datetime.now().isoformat(),
                },
                headers={"X-Request-ID": request_id},
            )

    def get_client_ip(self, request: Request) -> str:
        """Obtener IP real del cliente considerando proxies."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"

    async def save_metrics(
        self,
        request: Request,
        response: Response,
        process_time: float,
        client_ip: str,
    ):
        """Guardar métricas de la request en Redis."""
        try:
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "method": request.method,
                "path": str(request.url.path),
                "status_code": response.status_code,
                "process_time": process_time,
                "client_ip": client_ip,
                "user_agent": request.headers.get("user-agent", ""),
                "content_length": response.headers.get("content-length", "0"),
            }

            # Guardar en Redis con TTL de 24 horas
            await redis_client.lpush("api_metrics", json.dumps(metrics))
            await redis_client.expire("api_metrics", 86400)  # 24 horas

        except Exception as e:
            logger.warning(f"Error saving metrics: {e}")


# =====================================================
# MIDDLEWARE DE RATE LIMITING
# =====================================================


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """
    Middleware para limitar la cantidad de requests por IP/usuario.
    Implementa sliding window con Redis.
    """

    def __init__(
        self,
        app: ASGIApp,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_limit: int = 10,
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_limit = burst_limit

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        # Skip rate limiting para endpoints de sistema
        if request.url.path in ["/health", "/metrics", "/docs", "/redoc"]:
            return await call_next(request)

        client_ip = self.get_client_ip(request)

        # Verificar rate limits
        is_allowed, limit_info = await self.check_rate_limits(client_ip)

        if not is_allowed:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")

            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": "Too many requests. Please try again later.",
                    "retry_after": limit_info.get("retry_after", 60),
                    "limits": {
                        "requests_per_minute": self.requests_per_minute,
                        "requests_per_hour": self.requests_per_hour,
                    },
                },
                headers={
                    "Retry-After": str(limit_info.get("retry_after", 60)),
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": str(
                        limit_info.get("remaining", 0)
                    ),
                    "X-RateLimit-Reset": str(limit_info.get("reset_time", 0)),
                },
            )

        # Procesar request
        response = await call_next(request)

        # Agregar headers de rate limiting info
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(
            limit_info.get("remaining", 0)
        )

        return response

    def get_client_ip(self, request: Request) -> str:
        """Obtener IP real del cliente."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def check_rate_limits(self, client_ip: str) -> tuple[bool, dict]:
        """
        Verificar rate limits usando sliding window en Redis.

        Returns:
            tuple: (is_allowed, limit_info)
        """
        try:
            current_time = time.time()
            minute_key = f"rate_limit:{client_ip}:minute"
            hour_key = f"rate_limit:{client_ip}:hour"

            # Sliding window para minuto
            minute_count = await self.sliding_window_counter(
                minute_key, current_time, 60, self.requests_per_minute
            )

            # Sliding window para hora
            hour_count = await self.sliding_window_counter(
                hour_key, current_time, 3600, self.requests_per_hour
            )

            # Verificar límites
            if minute_count > self.requests_per_minute:
                return False, {
                    "retry_after": 60,
                    "remaining": 0,
                    "reset_time": int(current_time) + 60,
                }

            if hour_count > self.requests_per_hour:
                return False, {
                    "retry_after": 3600,
                    "remaining": 0,
                    "reset_time": int(current_time) + 3600,
                }

            return True, {
                "remaining": min(
                    self.requests_per_minute - minute_count,
                    self.requests_per_hour - hour_count,
                ),
                "reset_time": int(current_time) + 60,
            }

        except Exception as e:
            logger.error(f"Error checking rate limits: {e}")
            # En caso de error, permitir la request
            return True, {"remaining": self.requests_per_minute}

    async def sliding_window_counter(
        self, key: str, current_time: float, window_size: int, limit: int
    ) -> int:
        """Implementar sliding window counter con Redis."""
        try:
            # Limpiar entradas antiguas
            cutoff_time = current_time - window_size
            await redis_client.zremrangebyscore(key, 0, cutoff_time)

            # Agregar entrada actual
            await redis_client.zadd(key, {str(current_time): current_time})

            # Contar entradas en la ventana
            count = await redis_client.zcard(key)

            # Set TTL
            await redis_client.expire(key, window_size + 10)

            return count

        except Exception as e:
            logger.error(f"Error in sliding window counter: {e}")
            return 0


# =====================================================
# MIDDLEWARE DE AUTENTICACIÓN Y AUTORIZACIÓN
# =====================================================


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Middleware para autenticación de API keys y tokens JWT.
    """

    # Rutas públicas que no requieren autenticación
    PUBLIC_PATHS = {
        "/health",
        "/metrics",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/api/v1/health",
        "/api/v1/info",
    }

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        # Skip autenticación para rutas públicas
        if any(
            request.url.path.startswith(path) for path in self.PUBLIC_PATHS
        ):
            return await call_next(request)

        # Verificar API key o token
        auth_result = await self.authenticate_request(request)

        if not auth_result["valid"]:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "Authentication required",
                    "message": auth_result["message"],
                    "type": "authentication_error",
                },
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Agregar información del usuario a la request
        request.state.user = auth_result.get("user")
        request.state.auth_type = auth_result.get("auth_type")

        return await call_next(request)

    async def authenticate_request(self, request: Request) -> dict:
        """
        Autenticar request usando API key o JWT token.

        Returns:
            dict: {"valid": bool, "user": dict, "auth_type": str, "message": str}
        """
        # Verificar API key en headers
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return await self.validate_api_key(api_key)

        # Verificar Bearer token
        authorization = request.headers.get("Authorization")
        if authorization and authorization.startswith("Bearer "):
            token = authorization.split(" ")[1]
            return await self.validate_jwt_token(token)

        return {
            "valid": False,
            "message": "No authentication credentials provided",
            "auth_type": None,
        }

    async def validate_api_key(self, api_key: str) -> dict:
        """Validar API key contra Redis/base de datos."""
        try:
            # Verificar en Redis cache
            cached_key = await redis_client.get(f"api_key:{api_key}")
            if cached_key:
                user_data = json.loads(cached_key)
                return {
                    "valid": True,
                    "user": user_data,
                    "auth_type": "api_key",
                    "message": "Valid API key",
                }

            # Si no está en cache, verificar en base de datos
            # (implementar según tu modelo de usuarios)

            return {
                "valid": False,
                "message": "Invalid API key",
                "auth_type": "api_key",
            }

        except Exception as e:
            logger.error(f"Error validating API key: {e}")
            return {
                "valid": False,
                "message": "Authentication service error",
                "auth_type": "api_key",
            }

    async def validate_jwt_token(self, token: str) -> dict:
        """Validar JWT token."""
        try:
            # Implementar validación JWT
            # (usar libraries como python-jose)

            # Por ahora, validación básica
            if token == "valid_token_example":
                return {
                    "valid": True,
                    "user": {"id": 1, "username": "demo_user"},
                    "auth_type": "jwt",
                    "message": "Valid JWT token",
                }

            return {
                "valid": False,
                "message": "Invalid or expired token",
                "auth_type": "jwt",
            }

        except Exception as e:
            logger.error(f"Error validating JWT token: {e}")
            return {
                "valid": False,
                "message": "Token validation error",
                "auth_type": "jwt",
            }


# =====================================================
# MIDDLEWARE DE CORS PERSONALIZADO
# =====================================================


class CustomCORSMiddleware:
    """
    Configuración CORS personalizada para Football Analytics API.
    """

    @staticmethod
    def get_cors_middleware():
        """Obtener middleware CORS configurado."""

        allowed_origins = [
            "http://localhost:3000",  # React frontend
            "http://localhost:3001",  # Desarrollo alternativo
            "https://your-domain.com",  # Producción
            "https://www.your-domain.com",
        ]

        # En desarrollo, permitir todos los orígenes
        if settings.DEBUG:
            allowed_origins = ["*"]

        return CORSMiddleware, {
            "allow_origins": allowed_origins,
            "allow_credentials": True,
            "allow_methods": [
                "GET",
                "POST",
                "PUT",
                "DELETE",
                "OPTIONS",
                "PATCH",
            ],
            "allow_headers": [
                "Authorization",
                "Content-Type",
                "X-API-Key",
                "X-Request-ID",
                "X-Forwarded-For",
                "User-Agent",
                "Accept",
                "Origin",
                "DNT",
                "X-CustomHeader",
                "Keep-Alive",
                "X-Requested-With",
                "If-Modified-Since",
                "Cache-Control",
            ],
            "expose_headers": [
                "X-Request-ID",
                "X-Process-Time",
                "X-RateLimit-Limit",
                "X-RateLimit-Remaining",
                "X-RateLimit-Reset",
            ],
        }


# =====================================================
# MIDDLEWARE DE COMPRESIÓN Y OPTIMIZACIÓN
# =====================================================


class CompressionMiddleware(BaseHTTPMiddleware):
    """
    Middleware para compresión automática de responses grandes.
    """

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        response = await call_next(request)

        # Verificar si el cliente acepta compresión
        accept_encoding = request.headers.get("accept-encoding", "")

        if "gzip" in accept_encoding.lower():
            # Comprimir responses grandes (>1KB)
            if hasattr(response, "body") and len(response.body) > 1024:
                try:
                    import gzip

                    compressed_body = gzip.compress(response.body)

                    # Solo usar compresión si reduce el tamaño significativamente
                    if len(compressed_body) < len(response.body) * 0.9:
                        response.headers["content-encoding"] = "gzip"
                        response.headers["content-length"] = str(
                            len(compressed_body)
                        )
                        # Actualizar body comprimido
                        # (implementación específica depende del tipo de response)

                except Exception as e:
                    logger.warning(f"Compression error: {e}")

        return response


# =====================================================
# MIDDLEWARE DE SEGURIDAD
# =====================================================


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Middleware para headers de seguridad y protección básica.
    """

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        response = await call_next(request)

        # Headers de seguridad
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
        }

        # Agregar headers de seguridad
        for header, value in security_headers.items():
            response.headers[header] = value

        # Header personalizado de la API
        response.headers["X-API-Version"] = "1.0.0"
        response.headers["X-Powered-By"] = "Football Analytics"

        return response


# =====================================================
# FUNCIONES DE CONFIGURACIÓN
# =====================================================


def setup_middleware(app):
    """
    Configurar todos los middleware en el orden correcto.

    Args:
        app: Instancia de FastAPI
    """

    # 1. CORS (debe ir primero)
    cors_middleware, cors_config = CustomCORSMiddleware.get_cors_middleware()
    app.add_middleware(cors_middleware, **cors_config)

    # 2. Seguridad
    app.add_middleware(SecurityMiddleware)

    # 3. Compresión
    app.add_middleware(CompressionMiddleware)

    # 4. Rate Limiting
    app.add_middleware(
        RateLimitingMiddleware,
        requests_per_minute=getattr(settings, "RATE_LIMIT_PER_MINUTE", 60),
        requests_per_hour=getattr(settings, "RATE_LIMIT_PER_HOUR", 1000),
    )

    # 5. Autenticación
    if getattr(settings, "ENABLE_AUTHENTICATION", True):
        app.add_middleware(AuthenticationMiddleware)

    # 6. Logging (debe ir al final para capturar todo)
    app.add_middleware(RequestLoggingMiddleware)

    logger.info("All middleware configured successfully")


# =====================================================
# EXPORTACIONES
# =====================================================

__all__ = [
    "RequestLoggingMiddleware",
    "RateLimitingMiddleware",
    "AuthenticationMiddleware",
    "CustomCORSMiddleware",
    "CompressionMiddleware",
    "SecurityMiddleware",
    "setup_middleware",
]
