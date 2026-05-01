# 🚀 48-Hour Final Sprint Plan

## **Overview**
This plan is designed to close out the remaining coding portion of the project in the next 48 hours. It drops the Selenium/LangGraph live bot execution (due to online platform limitations) and focuses purely on agent behavioral adjustments, rigorous evaluation, statistical profiling, clustering, and GUI visualization via `catanatron`.

---

## **Proposed Directory Structure**
To facilitate modularity and orchestration, update your directory structure as follows:

```text
Redeem-Catan/
├── _Ideas/
├── data/
│   ├── eval_logs/          <-- NEW: For storing 500 txt logs from evaluation
│   ├── game_log_dfs/
│   ├── player_profiles/
│   └── raw_divs/
├── scripts/                <-- NEW: Shell scripts for orchestration via `uv run`
│   ├── play_gui.sh         <-- Spin up catanatron-play with specific bot
│   └── eval_pipeline.sh    <-- Orchestrates eval games, logging, parsing, and clustering
├── src/
│   ├── pyproject.toml
│   ├── analysis/           <-- NEW: Scripts for parsing and clustering post-eval
│   │   ├── parse_logs.py
│   │   └── cluster_bots.py
│   ├── eda/
│   ├── rl/
│   │   ├── wrappers.py     <-- Opponent profile injection
│   │   ├── axelrod.py      <-- Tit-for-tat logic/tracker
│   │   ├── train.py
│   │   └── eval.py
│   └── scraper/
```

---

## **Phase 1: Agent Logic & Implementation (Hours 1 - 12)**

### **Step 1: Axelrod's "Tit-for-Tat" Logic**
- **Task**: Implement logic to track who targets your bot the most (e.g., Robber placements, stealing). 
- **Details**: 
  - Wrap the model's action selection or state observation to maintain a dictionary of `times_targeted_by[player_id]`.
  - When the bot must choose a target (e.g., moving the Robber), override the model's choice to target the player with the highest score in `times_targeted_by`.
  - Introduce a CLI flag `--axelrod=1` in your training/eval scripts. When active, this overriding logic takes precedence.
- **Output**: `src/rl/axelrod.py` or integrated into your Catanatron custom agent.

### **Step 2: Core RL Pipeline Finalization**
- **Task**: Ensure the `OpponentProfileWrapper` correctly injects your pre-calculated player profiles into the observation space.
- **Task**: Hook up the baseline and profile-aware training scripts.
- **Output**: `src/rl/train.py` functioning correctly and producing `models/baseline_ppo.zip` and `models/aware_ppo.zip`.

---

## **Phase 2: Evaluation Pipeline Architecture (Hours 12 - 24)**

### **Step 3: Generating the Eval Logs**
- **Task**: Modify `src/rl/eval.py` to run 500 games against specified opponents.
- **Details**: Ensure that for each game, a textual game log is generated (simulating the raw logs you originally parsed) and saved to `data/eval_logs/<bot_name>_game_<id>.txt`.
- **Output**: `data/eval_logs/` populated for each evaluated model.

### **Step 4: Parsing Stats from Logs**
- **Task**: Write `src/analysis/parse_logs.py`.
- **Details**: Reuse the logic from your EDA notebook (`player_profiler.ipynb`) to parse the 500 `.txt` logs.
- **Metrics to extract**: `num_games`, `avg_turns_before_first_trade`, `ratio_cards_given_to_taken`, `trade_success_rate`, `avg_counter_offers`, `avg_bank_trades`, `avg_dev_cards_bought`, `avg_roads_built`, `avg_cities_built`, `avg_players_targeted`, `avg_times_targeted`, `win_rate_largest_army`, `win_rate_longest_road`, `average_games_with_port_used`, `overall_avg_hand_size`, `top_3_starting_resources`, `most_traded_away_resource`, `most_received_resource`.
- **Output**: A final CSV or Parquet file summarizing the evaluated bot's characteristics.

### **Step 5: Clustering Analysis**
- **Task**: Write `src/analysis/cluster_bots.py`.
- **Details**: 
  - Load the stats of the 5 previously hardcoded profile players.
  - Load the newly generated stats for your RL bot(s).
  - Apply K-Means or Hierarchical clustering (with PCA for visualization) to compare them.
  - Determine and output which of the 5 hardcoded profiles your bot most closely resembles.
- **Output**: Visualizations (`bot_clusters.png`) and stdout metrics indicating similarity.

### **Step 6: Orchestration Script**
- **Task**: Write `scripts/eval_pipeline.sh`.
- **Details**: Chain the steps together utilizing `uv run`. 
  ```bash
  uv run src/rl/eval.py --model aware_ppo --games 500 --out_dir data/eval_logs
  uv run src/analysis/parse_logs.py --input data/eval_logs --output data/bot_stats.csv
  uv run src/analysis/cluster_bots.py --stats data/bot_stats.csv
  ```

---

## **Phase 3: Visualisation & Human Testing (Hours 24 - 36)**

### **Step 7: GUI Playback Script**
- **Task**: Write `scripts/play_gui.sh`.
- **Details**: A shell script utilizing `catanatron-play` to spin up the local graphical interface.
  - Accepts a bot name/path as an argument and starts the localhost:3000 server.
  ```bash
  # Example concept
  catanatron-play --port 3000 --players "Human,Human,Human,$1"
  ```
- **Output**: Easily testable command to view gameplay instantly.

### **Step 8: Local Human Testing**
- **Task**: Play testing games to manually assess the bot.
- **Details**: Using `scripts/play_gui.sh <your_bot>`, gather 3 friends around the laptop and play a few full games against your RL agent. 
- **Task**: If the `--axelrod=1` flag is implemented cleanly, test this visually by aggressively targeting the bot and confirming it retaliates.

---

## **Phase 4: Buffer, Debugging & Final Assets (Hours 36 - 48)**

### **Step 9: Run Full Sweeps**
- **Task**: Let the 500-game evaluations run for all configurations (Baseline, Profile-Aware, Axelrod-enabled).
- **Task**: Review the clustering results and ensure the logic holds up for the report.

### **Step 10: Clean Code & Export Plots**
- **Task**: Ensure all code is cleanly formatted and commented.
- **Task**: Export final PCA/Clustering plots and the comparison tables for your thesis/report.
- **Task**: Push final state to GitHub.