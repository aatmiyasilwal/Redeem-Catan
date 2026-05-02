#!/usr/bin/env bash
# eval_pipeline.sh - Orchestrates the Catan RL training/evaluation pipeline
# Assumes data preparation (scraper -> pipeliner -> profiler -> EDA) is already complete.

set -euo pipefail

# ── Colours ────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PYTHON="${PROJECT_ROOT}/src/.venv/bin/python"

# Clear any parent VIRTUAL_ENV to prevent uv conflicts
unset VIRTUAL_ENV
export VIRTUAL_ENV="${PROJECT_ROOT}/src/.venv"

UV_RUN="uv run"

info()    { echo -e "${BLUE}[INFO]${NC}    $*"; }
success() { echo -e "${GREEN}[OK]${NC}      $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}    $*"; }
error()   { echo -e "${RED}[ERROR]${NC}   $*"; }
step()    { echo -e "\n${CYAN}${BOLD}▶ $*${NC}"; }

# ── Pre-flight check ──────────────────────────────────────────────────────
step "Pre-flight Check"
echo -e "${BOLD}This pipeline assumes you have already run the following in order:${NC}"
echo "  1. src/scraper/boilerplate_scraper.js   (colonist.gg data scraping)"
echo "  2. src/eda/data_pipeliner.ipynb         (raw → consolidated parquet)"
echo "  3. src/eda/player_profiler.ipynb        (per-player profile generation)"
echo "  4. src/eda/eda.ipynb                    (exploratory analysis)"
echo ""
printf "${YELLOW}Have you completed all four steps above? [y/N]:${NC} "
read CONFIRM
CONFIRM="${CONFIRM:-N}"
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    error "Please complete the data preparation steps first. Exiting."
    exit 1
fi
success "Data preparation confirmed."

# ── Collect user preferences ──────────────────────────────────────────────
step "Configuration"
echo -e "${BOLD}Opponent selection${NC} (from player_index.json)"
echo -e "  ${CYAN}Default: 0,1,3${NC}"
read -p "Enter 3 comma-separated player indices (or press Enter for default): " PLAYERS_INPUT
PLAYERS="${PLAYERS_INPUT:-0,1,3}"

printf "${CYAN}Enable Axelrod (Tit-for-Tat) trading logic? [y/N]:${NC} "
read AXELROD_INPUT
AXELROD_INPUT="${AXELROD_INPUT:-N}"
if [[ "$AXELROD_INPUT" =~ ^[Yy]$ ]]; then
    AXELROD=1
    success "Axelrod trading enabled."
else
    AXELROD=0
    info "Axelrod trading disabled."
fi

echo ""
echo -e "${BOLD}PPO Training Mode${NC}"
echo "  b - baseline (fixed opponents)"
echo "  a - aware (profile-aware opponents)"
echo "  s - shuffled (randomised opponent pool)"
printf "${CYAN}Select mode [b/a/s] (default: b):${NC} "
read MODE_INPUT
MODE="${MODE_INPUT:-b}"
if [[ ! "$MODE" =~ ^[bas]$ ]]; then
    error "Invalid mode '${MODE}'. Must be b, a, or s. Exiting."
    exit 1
fi
success "Mode set to '${MODE}'."

# ── Step 1: Train ─────────────────────────────────────────────────────────
step "Step 1: Training PPO Agent"
TRAIN_ARGS=""
if [[ -n "$PLAYERS" ]]; then
    TRAIN_ARGS="-p ${PLAYERS}"
fi
TRAIN_ARGS="${TRAIN_ARGS} -m ${MODE}"
if [[ "$AXELROD" -eq 1 ]]; then
    TRAIN_ARGS="${TRAIN_ARGS} --axelrod 1"
fi

info "Running: uv run train.py ${TRAIN_ARGS}"
(
    cd "${PROJECT_ROOT}/src/rl"
    $UV_RUN python train.py $TRAIN_ARGS
)

if [[ $? -eq 0 ]]; then
    echo ""
    success "Training completed successfully."
else
    echo ""
    error "Training failed. Check logs above."
    exit 1
fi

# ── Step 2: Evaluate ──────────────────────────────────────────────────────
step "Step 2: Evaluating Trained Agent (500 games)"
EVAL_ARGS=""
if [[ -n "$PLAYERS" ]]; then
    EVAL_ARGS="-p ${PLAYERS}"
fi
EVAL_ARGS="${EVAL_ARGS} -m ${MODE}"
if [[ "$AXELROD" -eq 1 ]]; then
    EVAL_ARGS="${EVAL_ARGS} --axelrod 1"
fi

info "Running: uv run eval.py ${EVAL_ARGS}"
(
    cd "${PROJECT_ROOT}/src/rl"
    $UV_RUN python eval.py $EVAL_ARGS
)

if [[ $? -eq 0 ]]; then
    echo ""
    success "Evaluation completed successfully."
else
    echo ""
    error "Evaluation failed. Check logs above."
    exit 1
fi

# ── Step 3: Parse Logs ───────────────────────────────────────────────────
step "Step 3: Parsing Evaluation Logs"
(
    cd "${PROJECT_ROOT}/src/analysis"
    $VENV_PYTHON parse_logs.py \
        --input "${PROJECT_ROOT}/data/eval_logs" \
        --output "${PROJECT_ROOT}/data/player_profiles/parsed_eval_logs_profiles.parquet"
)

if [[ $? -eq 0 ]]; then
    echo ""
    success "Log parsing completed. Profiles saved to data/player_profiles/parsed_eval_logs_profiles.parquet"
else
    echo ""
    error "Log parsing failed. Check logs above."
    exit 1
fi

# ── Done ──────────────────────────────────────────────────────────────────
step "Pipeline Complete"
echo -e "  ${GREEN}✓${NC} Training  → src/rl/models/"
echo -e "  ${GREEN}✓${NC} Eval CSV → src/rl/results/"
echo -e "  ${GREEN}✓${NC} Profiles → data/player_profiles/parsed_eval_logs_profiles.parquet"
echo ""
success "All done. Run src/analysis/cluster_bots.ipynb for visual analysis."
