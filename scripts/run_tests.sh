#!/usr/bin/env bash
# ============================================================================
#  run_tests.sh — test the RAG and GraphRAG lab
#
#    ./scripts/run_tests.sh            unit suite plus the book parity check
#    ./scripts/run_tests.sh unit       unit suite only, no services needed
#    ./scripts/run_tests.sh parity     assert the book and the lab agree
#    ./scripts/run_tests.sh e2e        Playwright suite, needs Docker + Ollama
#    ./scripts/run_tests.sh all        everything
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

GREEN=$'\033[0;32m'; BLUE=$'\033[0;34m'; NC=$'\033[0m'
info() { printf "%s==>%s %s\n" "$BLUE" "$NC" "$1"; }

run_unit() {
  info "unit suite (no Docker, no Ollama, no network)"
  python -m pytest tests/unit -q
}

run_parity() {
  info "book to lab parity"
  python scripts/check_book_parity.py --ebook ../ebook
}

run_e2e() {
  info "end to end Playwright suite (requires docker compose up -d and ollama)"
  python -m pytest tests/e2e -q
}

case "${1:-default}" in
  unit)    run_unit ;;
  parity)  run_parity ;;
  e2e)     run_e2e ;;
  all)     run_unit; run_parity; run_e2e ;;
  default) run_unit; run_parity
           printf "\n%sReady.%s Run './scripts/run_tests.sh e2e' for the browser suite.\n" "$GREEN" "$NC" ;;
  *)       echo "Usage: $0 [unit|parity|e2e|all]"; exit 1 ;;
esac
