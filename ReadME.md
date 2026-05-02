# 🎲 Redeem-Catan: Opponent-Aware RL for Settlers of Catan

**Redeem-Catan** is a reinforcement learning project that investigates whether incorporating human behavioral profiling into the observation space of a PPO agent improves strategic decision-making in *Settlers of Catan*. Unlike traditional bots that optimize purely for game mechanics, this agent adapts its policy based on opponent tendencies (trading aggression, hoarding, robber placement) derived from historical gameplay data.

---

## 🏗️ Architecture

The project is built on a multi-stage pipeline:
1.  **Data Mining:** Scraping raw colonist.io replays to extract player action logs.
2.  **Behavioral Profiling:** Aggregating per-game statistics into 19-dimensional player profiles.
3.  **Opponent-Aware Training:** Injecting these profiles into the RL observation space via custom Gymnasium wrappers.
4.  **Evaluation:** Running 500-game sweeps against various bot configurations and analyzing performance via statistical clustering.

---

## 📂 Project Structure

```text
Redeem-Catan/
├── _Ideas/                  # Sprint plans, technical deep-dives, and design docs
├── data/
│   ├── raw_divs/            # Raw scraped data from colonist.io replays
│   ├── game_log_dfs/        # Consolidated per-game parquet files (4-player stats)
│   ├── player_profiles/     # ML-ready profile vectors and index mappings
│   └── eval_logs/           # 500-game textual logs generated during evaluation
├── scripts/
│   ├── eval_pipeline.sh     # Automated training, evaluation, and parsing pipeline
│   └── play_gui.sh          # Docker-based GUI launcher for visual inspection
├── src/
│   ├── scraper/             # JS scripts for colonist.io data extraction
│   ├── eda/                 # Data preparation and exploratory analysis
│   │   ├── data_pipeliner.ipynb
│   │   ├── player_profiler.ipynb
│   │   └── eda.ipynb
│   ├── rl/                  # Core RL logic
│   │   ├── train.py         # PPO training entry point
│   │   ├── eval.py          # Evaluation entry point
│   │   ├── make_env.py      # Environment wrappers (Profile/Axelrod)
│   │   ├── axelrod.py       # Tit-for-Tat logic wrapper
│   │   ├── profiles.py      # Profile vector manager
│   │   └── models/          # Saved .zip models
│   └── analysis/            # Post-eval clustering
│       ├── parse_logs.py
│       └── cluster_bots.ipynb
├── pyproject.toml           # Dependencies managed via uv
└── game_logs.txt            # Archive of unique game replays

```
---

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- [uv](https://docs.astral.sh/uv/) for dependency management
- Docker & Docker Compose (for GUI testing)

### 1. Installation
```bash
cd src
uv sync
```

### 2. Data Preparation (Strict Order)
Before running any training or evaluation, the data pipeline must be executed in the following order to generate valid profile vectors:

| Step | File | Description |
|:----:|:-----|:------------|
| **1** | `src/eda/data_pipeliner.ipynb` | Parses `raw_divs` into consolidated `game_log_dfs`. Validates regex patterns and cleans data. |
| **2** | `src/eda/player_profiler.ipynb` | Aggregates per-game stats into `data/player_profiles/profiles.npy`. Identifies stable player archetypes. |
| **3** | `src/eda/eda.ipynb` | Exploratory analysis. Generates correlation matrices, radar charts, and initial clustering proofs. |

> ⚠️ **Note:** Training and evaluation scripts rely on `profiles.npy` and `player_index.json` existing in `data/player_profiles/`.

---

## ⚙️ Training & Evaluation

### Automated Pipeline
For a seamless end-to-end execution (Train → Evaluate → Parse Logs), use the orchestration script:

```bash
./scripts/eval_pipeline.sh
```
This script will prompt you for:
- **Opponents:** Player indices to simulate against (default: `0,1,3`).
- **Axelrod Mode:** Enable Tit-for-Tat robber logic.
- **Training Mode:** Baseline, Aware, or Shuffled.

### Manual Execution
You can also run components individually from `src/rl`:

```bash
# Train
uv run python train.py --players 0,1,3 --mode a --axelrod 1

# Evaluate (500 games)
uv run python eval.py --players 0,1,3 --mode a --axelrod 1
```

### Configuration Modes

| Flag | Mode | Description |
|:----:|:-----|:------------|
| `-m b` | **Baseline** | Agent plays against 3 fixed opponents without profile injection. |
| `-m a` | **Aware** | Agent receives opponent profile vectors in observation space to adapt strategy. |
| `-m s` | **Shuffled** | **Ablation Control.** Profiles are randomly permuted to test if the agent learns from the signal or just noise. |
| `--axelrod 1` | **Tit-for-Tat** | Intercepts `MOVE_ROBBER` actions. Automatically targets the player who has targeted the bot most frequently in the current game. |

---

## 📊 Visualization & GUI

### GUI Inspection
Spin up the Catanatron web interface to visually inspect trained agents:

```bash
# Default: 4 Weighted-Random players
./scripts/play_gui.sh

# Custom: 1 Human, 2 Random, 1 RL Agent (PPO)
./scripts/play_gui.sh --players "H,R,R,W"
```
*Requires Docker. Generates a unique replay URL saved to `game_logs.txt`.*

### Clustering Analysis
To compare the RL bot's behavior against real human players, run:
```bash
jupyter notebook src/analysis/cluster_bots.ipynb
```
This notebook applies K-Means and PCA to show which human archetype the trained bot most closely resembles.

---

## 📝 Limitations & Future Work
- **Fixed Topology:** Agents are trained on the standard BASE map. Generalization to tournament maps may require Graph Neural Networks.
- **Statistical Profiles:** Profiles represent frequencies of actions, not semantic intent (e.g., chat logs).
- **No Live Deployment:** Due to platform ethics and TOS, no live colonist.io integration is included.

## 📄 License
BSD 3-Clause License

