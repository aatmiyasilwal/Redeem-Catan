#!/usr/bin/env bash
# play_gui.sh - Spin up catanatron GUI and run a game via catanatron-play
# This script cd's into the catanatron source repo (past_work/catanatron)
# where catanatron-play is registered as a CLI entry point.

set -euo pipefail

# ── Colours ────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC}    $*"; }
success() { echo -e "${GREEN}[OK]${NC}      $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}    $*"; }
error()   { echo -e "${RED}[ERROR]${NC}   $*"; }
step()    { echo -e "\n${CYAN}${BOLD}▶ $*${NC}"; }

CATANATRON_DIR="$(cd "$(dirname "$0")/../past_work/catanatron" && pwd)"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ── Defaults ───────────────────────────────────────────────────────────────
PLAYERS="W,W,W,W"   # Weighted random players
NUM_GAMES=1

# ── Parse optional flags ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--players) PLAYERS="$2"; shift 2;;
        -n|--num)     NUM_GAMES="$2"; shift 2;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "  -p, --players  Player types (default: W,W,W,W)"
            echo "                 W=WeightedRandom, R=Random, H=Human, F=FastGreedy"
            echo "  -n, --num      Number of games (default: 1)"
            echo "  -h, --help     Show this help"
            exit 0
            ;;
        *) error "Unknown option: $1"; exit 1;;
    esac
done

# ── Step 1: Start Docker Compose (DB + Server + React UI) ────────────────
step "Step 1: Starting catanatron services via Docker Compose"
info "  PostgreSQL → localhost:5432"
info "  Flask API  → localhost:5001"
info "  React UI   → localhost:3000"

(
    cd "$CATANATRON_DIR"
    docker compose up -d
)

success "Services started."

# ── Wait for services to be ready ─────────────────────────────────────────
step "Waiting for PostgreSQL and Flask server to be healthy"
MAX_WAIT=60
WAITED=0
while ! curl -s http://localhost:5001/ > /dev/null 2>&1; do
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then
        error "Server did not become ready within ${MAX_WAIT}s."
        error "Check logs: docker compose -C $CATANATRON_DIR logs"
        exit 1
    fi
    printf "."
    sleep 2
    WAITED=$((WAITED + 2))
done
echo ""
success "Server is ready after ${WAITED}s."

# ── Step 2: Run catanatron-play ──────────────────────────────────────────
step "Step 2: Running catanatron-play"
info "Players : ${PLAYERS}"
info "Games   : ${NUM_GAMES}"

OUTPUT=$(docker compose -f "${CATANATRON_DIR}/docker-compose.yml" exec -T server \
    catanatron-play --players="${PLAYERS}" --db --num="${NUM_GAMES}")
echo "$OUTPUT"

# Query PostgreSQL for the most recent game UUIDs and their final state indices.
# The terminal output wraps the LINK column, so we reconstruct URLs from DB.
GAME_LOG="${PROJECT_ROOT}/game_logs.txt"
touch "$GAME_LOG"

# Get the latest state_index per unique UUID, ordered by most recent state_index
UUIDS_AND_STATES=$(docker compose -f "${CATANATRON_DIR}/docker-compose.yml" exec -T db \
    psql -U catanatron -d catanatron_db -t -c \
    "SELECT uuid, MAX(state_index) AS max_state FROM game_states GROUP BY uuid ORDER BY max_state DESC LIMIT ${NUM_GAMES};" 2>/dev/null | sed 's/^ *//;s/ *$//' | tr -s ' ')

if [ -n "$UUIDS_AND_STATES" ]; then
    LINK_COUNT=0
    echo "$UUIDS_AND_STATES" | while IFS='|' read -r uid state_idx; do
        uid=$(echo "$uid" | tr -d ' ')
        state_idx=$(echo "$state_idx" | tr -d ' ')
        if [ -n "$uid" ] && [ -n "$state_idx" ]; then
            echo "http://localhost:3000/games/${uid}/states/${state_idx}" >> "$GAME_LOG"
            LINK_COUNT=$((LINK_COUNT + 1))
        fi
    done
    success "Appended ${NUM_GAMES} game link(s) to ${GAME_LOG}"
else
    warn "No game links found. DB query returned empty."
fi

success "Game(s) completed."

# ── Cleanup option ────────────────────────────────────────────────────────
echo ""
read -p "$(echo -e ${YELLOW}Keep services running for browser viewing at localhost:3000? [Y/n]:${NC} )" KEEP
KEEP="${KEEP:-Y}"
if [[ ! "$KEEP" =~ ^[Yy]$ ]]; then
    step "Shutting down services"
    docker compose -f "${CATANATRON_DIR}/docker-compose.yml" down
    success "Services stopped."
else
    info "Services remain running. Open http://localhost:3000 to view."
    info "To stop later: docker compose -C ${CATANATRON_DIR} down"
fi
