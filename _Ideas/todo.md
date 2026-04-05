# 📅 FYP Coding Sprint: Apr 4 - Apr 17 (14 days, 4-6 hrs/day)

## **Phase 1: Core RL Pipeline & Profiles (Apr 4 - Apr 6)**

### **Day 1: Apr 4 - Profile Vectorization (Today)**
```
4-6 hours:
[ ] You have completed the EDA! Now extract the ML-ready DataFrame (`df_ml_ready`).
[ ] Finalize vector schema (document index → feature mapping).
[ ] Implement & test `get_profile_vector(player_name) → np.array`.
[ ] Save `data/profiles.npy`, `data/profile_index.json`.
[ ] Quick sanity check: plot 2D PCA of generated profiles.
```

### **Day 2: Apr 5 - Catanatron Setup & Env Iteration**
```
4-6 hours:
[ ] Clone Catanatron repo, `pip install -e .[gym]`.
[ ] Test basic gym loop: `gym.make("catanatron/Catanatron-v0")`.
[ ] Write `src/rl/make_env.py` → returns configured env.
[ ] Design wrapper logic: Implement `OpponentProfileWrapper(gym.ObservationWrapper)` to inject your player profiles.
```

### **Day 3: Apr 6 - Baseline Training**
```
4-6 hours:
[ ] Train tiny PPO (50k steps) to validate pipeline.
[ ] Train full baseline PPO (500k-1M steps) → `models/baseline_ppo.zip`.
[ ] Implement basic `eval_agent(model, n_games=100)` → win rate, avg VP.
[ ] Run eval: 500 games vs built-in bots → `results/baseline.csv`.
```

***

## **Phase 2: Training + Evaluation (Apr 7 - Apr 11)**

### **Day 4: Apr 7 - Opponent-Aware Training**
```
4-6 hours:
[ ] Ensure your wrapper samples 3 opponents/episode from your profile dataset.
[ ] Train opponent-aware PPO (500k-1M steps) → `models/aware_ppo.zip`.
[ ] Eval opponent-aware vs same bots → `results/aware.csv`.
```

### **Day 5: Apr 8 - Ablation Experiments**
```
4-6 hours:
[ ] Implement 3rd config: "shuffled profiles" (control).
[ ] Train shuffled PPO (300k-500k steps).
[ ] Run full eval suite for all 3 configs (500 games each).
```

### **Day 6: Apr 9 - Results & Plotting**
```
4-6 hours:
[ ] Build comparison table in a notebook.
[ ] Add opponent-specific metrics (e.g., trade frequency vs trader opponents, Robber targeting patterns).
[ ] Export plots (win rate bars, VP boxplots, baseline vs aware).
```

### **Day 7: Apr 10 - Code Hardening**
```
4-6 hours:
[ ] Modularise: `src/rl/train.py`, `src/rl/eval.py`, `src/configs/`.
[ ] Config system (CLI flags: `--config baseline|aware|shuffled`).
[ ] Add docstrings and clean up unused code.
```

### **Day 8: Apr 11 - Buffer / Extract Game Highlights**
```
4-6 hours:
[ ] Increase eval size if possible (1000 games/config).
[ ] Extract 2-3 "representative" games showing different AI behaviour.
[ ] Push to GitHub with a solid README.
```

***

## **Phase 3: Live Bot via LangGraph (Apr 12 - Apr 15) [OPTIONAL WIN]**

### **Day 9: Apr 12 - Selenium & State Inference Pipeline**
```
4-6 hours:
[ ] `pip install selenium playwright langgraph`.
[ ] Test basic colonist.io navigation (create private game, join as bot).
[ ] Parse key states from DOM (resources, VP counters, valid action buttons).
```

### **Day 10: Apr 13 - RL Action Mapping**
```
4-6 hours:
[ ] Map Catanatron actions → colonist.io clicks.
[ ] Load your trained model → `model.predict(parsed_state)`.
[ ] Test executing predicted action (click button) end-to-end on 1-2 turns.
```

### **Day 11: Apr 14 - LangGraph Orchestration**
```
4-6 hours:
[ ] Implement LangGraph state machine: `observe_screen → parse_state → rl_decide → execute_action → loop`.
[ ] Add error handling, timeouts.
[ ] Test full loop (5-10 turns).
```

### **Day 12: Apr 15 - Live Game Testing**
```
4-6 hours:
[ ] Play a full private game (you + 2 friends + bot).
[ ] Debug edge cases (trades, robber, dev cards, blockages).
[ ] Record replay, analyse decisions.
```

***

## **Phase 4: Wrap Up (Apr 16 - Apr 17)**

### **Day 13: Apr 16 - Polish Bot & Safe Guards**
```
4-6 hours:
[ ] Add safeguards (detect game end, disconnects).
[ ] Write `bot.md` with setup/run instructions.
[ ] Screenshot key moments for report.
```

### **Day 14: Apr 17 - Buffer / Final Eval**
```
4-6 hours:
[ ] Rerun any missing evals.
[ ] Backup everything.
[ ] Prepare folder structure for report writing.

## 🎯 Success Criteria
**Must-haves** ✅
- 3 trained models + eval results proving opponent-awareness works.
- Clean, modular code and comparison plots/tables ready for thesis.

**Nice-to-haves** 🎁
- LangGraph-driven live colonist.io bot seamlessly completing a game.
```
