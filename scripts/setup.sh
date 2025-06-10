#!/bin/bash
echo "🚀 Configurando Football Analytics..."

# Configurar .env
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✅ Archivo .env creado"
fi

echo "✅ Setup completado"
echo "Para iniciar: docker-compose up -d"
