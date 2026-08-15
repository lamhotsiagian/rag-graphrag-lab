#!/bin/bash
# Start all services for RAG & GraphRAG Lab

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "🚀 Starting RAG & GraphRAG Lab Services..."

# Start Docker services
echo "📦 Starting Docker services (PostgreSQL + Neo4j)..."
cd "$PROJECT_DIR"
docker compose up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check PostgreSQL
until docker compose exec -T postgres pg_isready -U raguser -d ragdb > /dev/null 2>&1; do
    echo "  Waiting for PostgreSQL..."
    sleep 2
done
echo "✅ PostgreSQL is ready"

# Check Neo4j
until curl -s http://localhost:7474 > /dev/null 2>&1; do
    echo "  Waiting for Neo4j..."
    sleep 2
done
echo "✅ Neo4j is ready"

# Check Ollama
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama is running"
else
    echo "⚠️  Ollama is not running. Start it with: ollama serve"
fi

echo ""
echo "🎉 All services are ready!"
echo ""
echo "Start chapter apps individually:"
for i in $(seq 1 10); do
    port=$((8500 + i))
    chapter=$(printf "%02d" $i)
    echo "  Chapter $i: streamlit run chapter_${chapter}/app.py --server.port $port"
done
