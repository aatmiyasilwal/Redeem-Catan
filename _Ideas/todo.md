# 📅 FYP Coding Sprint: Mar 27 - Apr 10 (14 days, 4-6 hrs/day)

## **Week 1: Core RL Pipeline (Mar 27-30)**

### **Day 1: Mar 27 - Profile finalisation**
```
4-6 hours:
[ ] Clean/normalize your per-player DataFrame (handle NaNs, scale continuous features)
[ ] Finalize vector schema (document index → feature mapping)
[ ] Implement & test `get_profile_vector(player_name) → np.array`
[ ] Save `data/profiles.npy`, `data/profile_index.json`
[ ] Quick sanity check: plot 2D PCA of profiles (friends vs you)
```

### **Day 2: Mar 28 - Catanatron setup**
```
4-6 hours:
[ ] Clone Catanatron repo, `pip install -e .[gym]`
[ ] Test basic gym loop: `gym.make("catanatron/Catanatron-v0")`
[ ] Write `src/rl/make_env.py` → returns configured env
[ ] Train tiny PPO (50k steps) to validate pipeline
[ ] Implement basic `eval_agent(model, n_games=100)` → win rate, avg VP
```

### **Day 3: Mar 29 - Baseline PPO**
```
4-6 hours:
[ ] Train full baseline PPO (500k-1M steps) → `models/baseline_ppo.zip`
[ ] Run eval: 500 games vs built-in bots → `results/baseline.csv`
[ ] Modularise: `src/rl/train.py`, `src/rl/eval.py`
[ ] Add simple logging (wandb or tensorboard optional)
```

### **Day 4: Mar 30 - Opponent wrapper**
```
4-6 hours:
[ ] Design wrapper logic (sample 3 opponents/episode)
[ ] Implement `OpponentProfileWrapper(gym.ObservationWrapper)`
[ ] Test shapes: `print(env.observation_space.shape)` before/after
[ ] Short training run (100k steps) to validate no NaNs/crashes
```

***

## **Week 2: Training + Evaluation (Mar 31-Apr 6)**

### **Day 5: Mar 31 - Opponent-aware training**
```
4-6 hours:
[ ] Train opponent-aware PPO (500k-1M steps) → `models/aware_ppo.zip`
[ ] Eval opponent-aware vs same bots → `results/aware.csv`
[ ] Quick comparison plot (baseline vs aware win rates)
```

### **Day 6: Apr 1 - Ablation experiments**
```
4-6 hours:
[ ] Implement 3rd config: "shuffled profiles" (control)
[ ] Train shuffled PPO (300k steps if compute tight)
[ ] Run full eval suite for all 3 configs (500 games each)
[ ] Build comparison table in notebook
```

### **Day 7: Apr 2 - Evaluation polish**
```
4-6 hours:
[ ] Increase eval size if possible (1000 games/config)
[ ] Add opponent-specific metrics:
  - Trade frequency vs trader opponents
  - Robber targeting patterns
[ ] Extract 2-3 "representative" games showing different behaviour
[ ] Export plots: win rate bars, VP boxplots
```

### **Day 8: Apr 3 - Code hardening**
```
4-6 hours:
[ ] Config system (CLI flags: `--config baseline|aware|shuffled`)
[ ] Modularise: `src/configs/`, `src/utils/`
[ ] Add docstrings, minimal tests
[ ] Push to GitHub with README
```

***

## **Week 3: Live colonist.io bot (Apr 4-10)**

### **Day 9: Apr 4 - Selenium basics**
```
4-6 hours:
[ ] `pip install selenium playwright langgraph`
[ ] Test basic colonist.io navigation:
  - Login, create private game, join as bot
[ ] Screenshot + basic DOM parsing (resources, VP, buttons)
[ ] Save screenshots of key states for manual inspection
```

### **Day 10: Apr 5 - State inference**
```
4-6 hours:
[ ] Parse key states from DOM:
  - My resources (img src → resource type)
  - VP counters
  - Valid action buttons (build, trade, etc.)
[ ] Test on a manual private game (you vs bots)
[ ] Log state to JSON for debugging
```

### **Day 11: Apr 6 - RL action mapping**
```
4-6 hours:
[ ] Map Catanatron actions → colonist.io clicks
[ ] Load your trained model → `model.predict(parsed_state)`
[ ] Execute predicted action (click button)
[ ] Test end-to-end on 1-2 turns
```

### **Day 12: Apr 7 - LangGraph orchestration**
```
4-6 hours:
[ ] Implement LangGraph state machine:
  ```
  observe_screen → parse_state → rl_decide → execute_action → loop
  ```
[ ] Add error handling, timeouts
[ ] Test full loop (5-10 turns)
```

### **Day 13: Apr 8 - Live game testing**
```
4-6 hours:
[ ] Play a full private game (you + 2 friends + bot)
[ ] Record replay, analyse decisions
[ ] Debug edge cases (trades, robber, dev cards)
[ ] Fix 1-2 critical bugs
```

### **Day 14: Apr 9 - Polish + documentation**
```
4-6 hours:
[ ] Add safeguards (detect game end, disconnects)
[ ] Record 1-2 demo games vs friends
[ ] Write `bot.md` with setup/run instructions
[ ] Screenshot key moments for report
```

### **Day 15: Apr 10 - Buffer / final eval**
```
4-6 hours:
[ ] Rerun any missing evals
[ ] Test bot on ranked game (if confident)
[ ] Backup everything
[ ] Prepare folder structure for report writing
```

***

## 🎯 Success criteria by Apr 10

**Must-haves** ✅
- 3 trained models + eval results showing opponent-awareness works
- Clean, modular code
- Comparison plots/tables ready for thesis

**Nice-to-haves** 🎁
- Live colonist.io bot that completes 1 game vs friends
- Demo replay showing smart opponent-specific decisions

**Timeline rationale**:
- **Days 1-4**: Get core RL pipeline solid (your supervisor cares most about this).
- **Days 5-8**: Evaluation (what proves your contribution).
- **Days 9-14**: Live bot (impressive demo, but secondary).

This gives you **ample buffer** if the bot proves tricky. You’ll have a complete FYP either way! 🚀
