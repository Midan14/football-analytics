"""
=============================================================================
FOOTBALL ANALYTICS - CLIENTE MCP (MODEL CONTEXT PROTOCOL)
=============================================================================
Cliente para conectar con servidores MCP y obtener contexto adicional
para análisis de fútbol y predicciones mejoradas.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


class MCPFootballClient:
    """Cliente MCP especializado para datos de fútbol"""

    def __init__(self):
        self.sessions: Dict[str, ClientSession] = {}
        self.session_status: Dict[str, Dict[str, Any]] = {}
        self.active_connections = 0

    async def connect_to_server(
        self, server_name: str, server_params: StdioServerParameters
    ) -> bool:
        """Conectar a un servidor MCP específico"""
        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    # Store session and update status
                    self.sessions[server_name] = session
                    self.active_connections += 1
                    self.session_status[server_name] = {
                        "status": "connected",
                        "connected_at": datetime.now().isoformat(),
                        "last_activity": datetime.now().isoformat(),
                    }
                    logger.info(f"✅ Conectado a servidor MCP: {server_name}")
                    return True

        except Exception as e:
            logger.error(f"❌ Error conectando a {server_name}: {e}")
            return False

    async def get_football_context(self, query: str) -> Optional[Dict[str, Any]]:
        """Obtener contexto de fútbol desde servidores MCP"""
        if not self.sessions:
            logger.warning("⚠️ No hay sesiones MCP activas")
            return None

        try:
            # Buscar en todos los servidores conectados
            results = []
            for server_name, session in self.sessions.items():
                try:
                    # Realizar consulta al servidor MCP
                    response = await session.call_tool(
                        "football_analysis", {"query": query}
                    )

                    # Update last activity timestamp
                    if server_name in self.session_status:
                        self.session_status[server_name][
                            "last_activity"
                        ] = datetime.now().isoformat()
                    if response:
                        results.append({"server": server_name, "data": response})
                except Exception as e:
                    logger.error(f"Error consultando {server_name}: {e}")

            return {
                "query": query,
                "results": results,
                "servers_consulted": len(self.sessions),
                "successful_responses": len(results),
            }

        except Exception as e:
            logger.error(f"❌ Error obteniendo contexto: {e}")
            return None

    async def get_prediction_context(
        self, match_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Obtener contexto para predicciones desde MCP"""
        try:
            context_query = f"Análisis predictivo para: {match_data.get('teams', 'equipos desconocidos')}"
            return await self.get_football_context(context_query)
        except Exception as e:
            logger.error(f"❌ Error obteniendo contexto de predicción: {e}")
            return None

    def get_status(self) -> Dict[str, Any]:
        """Obtener estado de las conexiones MCP"""
        # Update status for sessions that might have disconnected
        current_sessions = list(self.sessions.keys())
        for server_name in list(self.session_status.keys()):
            if server_name not in current_sessions:
                self.session_status[server_name]["status"] = "disconnected"

        return {
            "active_connections": self.active_connections,
            "server_details": self.session_status,
            "status": "active" if self.active_connections > 0 else "inactive",
        }


# Instancia global del cliente MCP
mcp_client = MCPFootballClient()
