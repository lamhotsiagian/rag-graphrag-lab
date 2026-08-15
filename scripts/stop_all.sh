#!/bin/bash
# Stop all services for RAG & GraphRAG Lab

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "🛑 Stopping RAG & GraphRAG Lab Services..."

# Kill any running Streamlit processes
echo "Stopping Streamlit apps..."
pkill -f "streamlit run chapter_" 2>/dev/null || true

# Stop Docker services
echo "Stopping Docker services..."
cd "$PROJECT_DIR"
docker compose down

echo "✅ All services stopped"
