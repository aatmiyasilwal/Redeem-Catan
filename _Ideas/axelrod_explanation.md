# ⚔️ Axelrod "Tit-for-Tat" Implementation in Catanatron

You asked: **"How can I truly be sure that the Axelrod theories are being implemented, and are we actually targeting the people who targeted us the most?"**

Yes! The logic strictly enforces Tit-for-Tat by overriding the environment's default target selection to explicitly penalize the player who has stolen from the bot the most. 

Here is exactly how it is being done under the hood, implemented in `src/rl/axelrod.py`.

---

### 1. The Interception
Whenever the RL agent decides to place a Robber on a specific hex coordinate, it outputs a generic integer action (e.g., `MOVE_ROBBER to hex 14`). In standard Catanatron, if there are multiple opponents on that hex, the environment simply picks the **first valid target** it finds. 

With `--axelrod 1`, the `AxelrodWrapper` intercepts this decision *before* the environment processes it.

```python
def step(self, action_int):
    # 1. Decode the intended action representation from the RL model
    action_type, value = ACTIONS_ARRAY[action_int]
    
    # 2. We only intervene if the agent is about to steal via a robber movement
    if action_type == ActionType.MOVE_ROBBER:
        game = self.env.unwrapped.game
        p0_color = self.env.unwrapped.p0.color
```

### 2. Tracking the History (The "Tat")
Before allowing the steal to happen, the wrapper dynamically scans the exact history of the current game. It counts how many times every single color has specifically targeted our bot (`P0`) with the Robber.

```python
        # Tally up who has targeted P0 using the game log so far
        times_targeted_by = {c: 0 for c in game.state.colors}
        
        for act in game.state.actions:
            if act.action_type == ActionType.MOVE_ROBBER:
                # Check the Robber payload: (coordinate, target_enemy_color, is_knight?)
                if isinstance(act.value, tuple) and len(act.value) >= 2:
                    target_color = act.value[1]
                    if target_color == p0_color:
                        # We were targeted by the player who took this action!
                        times_targeted_by[act.color] += 1
```
*Note: This guarantees perfect memory. The bot knows exactly who has been aggressive toward it.*

### 3. Forcing Revenge (The "Tit")
Now that the bot knows who its biggest enemies are, it needs to modify the game state so that the engine steals from the worst offender. 

It does this by taking the engine's internal list of `playable_actions` and **sorting** it. The sorting key applies a negative weight to the `times_targeted_by` score. The opponent who targeted us the most gets the most negative score, which shoots them to the absolute front of the line (index `0`).

```python
        def sort_key(catan_action):
            # Is this one of the steal actions matching the chosen hex coordinate?
            normalized = normalize_action(catan_action)
            if normalized.action_type == action_type and normalized.value == value:
                if isinstance(catan_action.value, tuple) and len(catan_action.value) >= 2:
                    target_color = catan_action.value[1]
                    if target_color is not None:
                        # A higher score -> more negative value -> shoots to index 0
                        return -times_targeted_by.get(target_color, 0)
            
            # Any other action receives a neutral priority
            return 0

        # Mutate state inline right before stepping
        game.state.playable_actions.sort(key=sort_key)
```

Because Catanatron's environment always blindly pops the **first** valid underlying game action that matches the RL agent's coordinate, sorting this array means Catanatron is forced to execute the steal against the player at index `0` — the player with the highest `times_targeted_by` score.

---

### Conclusion
By injecting this into the environment loop:
1. The RL agent figures out geographically *where* the most profitable hex to block is.
2. **The Axelrod wrapper steps in and dictates *who* to steal from on that hex, always prioritizing revenge.**

This ensures the agent maintains its geographical strategic intelligence while strictly enforcing Tit-for-Tat social diplomacy.
