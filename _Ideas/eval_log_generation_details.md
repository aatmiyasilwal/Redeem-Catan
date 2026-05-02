# 📝 Evaluation Log Generation Strategy

As part of **Step 3** of the 48-hour sprint, we needed a way to log all 500 games played during evaluation so we could later extract the 18 specific metrics required to profile the RL bot. 

Instead of dealing with complex HTML parsing or reverse-engineering Catanatron's visual replays, we implemented a highly-parseable custom plain-text format directly in `src/rl/eval.py`.

### Where are logs saved?
All generated logs are automatically dumped into the `data/eval_logs/` directory. The naming convention is `<bot_name>_game_<id>.txt` (e.g., `aware_axelrod_game_1.txt`).

### Log Structure
Each text file is split into three easy-to-read sections to make Step 4 (Parsing) as simple as possible.

#### 1. Header Details
A simple key-value header providing immediate context about the game.
```text
BOT_NAME: aware_axelrod
GAME_ID: 1
P0_COLOR: Color.RED
WINNER: Color.BLUE
```

#### 2. Actions Stream
A chronological dump of every action taken in the game, formatted as:
`[COLOR] | ACTION_TYPE | PAYLOAD`

*Example:*
```text
--- ACTIONS ---
[RED] | BUILD_SETTLEMENT | 14
[RED] | BUILD_ROAD | (14, 15)
[BLUE] | MOVE_ROBBER | (22, <Color.RED: 1>, None)
[RED] | MARITIME_TRADE | (WHEAT, WHEAT, WHEAT, WHEAT, ORE)
```
This sequential action log makes it trivial to calculate time-series metrics. By reading this block linearly, we can easily extract:
- `avg_turns_before_first_trade`
- `top_3_starting_resources` (derived from initial settlement placements)
- Trading behavior (`ratio_cards_given_to_taken`, `most_traded_away_resource`, `most_received_resource`)

#### 3. Final Player State JSON
At the end of every game, the Catanatron environment possesses a rich dictionary of stats. In `eval.py`, we intercept the state *specifically* for your agent (`P0`) and dump it directly as valid JSON.
```json
--- FINAL PLAYER STATE (P0) ---
{
  "P0_ACTUAL_VICTORY_POINTS": 8,
  "P0_CURRENT_LONGEST_ROAD": 6,
  "P0_HAS_LARGEST_ARMY": 0,
  "P0_ROADS_AVAILABLE": 9
  ...
}
```
By explicitly dumping the engine's internal dictionary, we can instantly pull exact values for metrics like:
- `win_rate_longest_road` / `win_rate_largest_army`
- `avg_roads_built` / `avg_cities_built`
- `avg_dev_cards_bought`
We get all of these stat arrays for free, completely bypassing the need to calculate them manually from the action stream!

#### 4. Board Layout
To ensure we can reliably map coordinates from the action logs (like `BUILD_SETTLEMENT | 14`) back to the actual resources, we also extract the randomized board topography at the end of the log. 

Extracting this required correctly accessing the internal mapping of Catanatron's board:
```python
--- BOARD LAYOUT ---
for coord, tile in game.state.board.map.land_tiles.items():
    if hasattr(tile, 'resource') and hasattr(tile, 'number'):
        res_name = tile.resource if tile.resource else "DESERT"
        num = tile.number if tile.number is not None else 0
        f.write(f"HEX {coord}: {res_name} {num}\n")
```

This dumps a mapping for each game so that metrics like `top_3_starting_resources` can accurately look up the hex resources surrounding the initial settlement nodes.
