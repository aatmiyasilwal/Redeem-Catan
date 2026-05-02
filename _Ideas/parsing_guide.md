# Log Parsing → Profiled Output: Implementation Guide

## Overview

`src/analysis/parse_logs.ipynb` parses 500 evaluation `.txt` logs from `data/eval_logs/` and produces a player profile in the same 19-column format as `src/eda/player_profiler.ipynb`. The parser reconstructs per-game metrics by cross-referencing each log's **board layout** (resource-to-coordinate mapping) with Catanatron's **static node geometry** (node_id → coordinates).

---

## Step 1: Build the Static Node-to-Tile Map

Catanatron's board geometry is fixed — there are 54 settlement/city nodes, and each node is adjacent to exactly 2–3 hex tiles. We create a fresh `Board()` once to get this mapping:

```python
from catanatron.models.board import Board

b = Board()
cm = b.map

# Build tile_id -> coord mapping
tile_id_to_coord = {}
for coord, tile in cm.land_tiles.items():
    tile_id_to_coord[tile.id] = coord

# Build node_id -> list of coord tuples mapping
node_tile_map = {}
for node_id, tiles in cm.adjacent_tiles.items():
    node_tile_map[node_id] = [tile_id_to_coord[t.id] for t in tiles]
```

`node_tile_map[41]` might return `[(-2, 2, 0)]` (a corner node touching one tile), while `node_tile_map[23]` returns `[(1, 0, -1), (2, 0, -2), (2, -1, -1)]` (a junction between three tiles).

**Why this matters**: Each game's `.txt` log randomly assigns resources to coordinates. The static map tells us *which* coordinates a settlement touches; the log tells us *what* resources those coordinates hold in this specific game.

---

## Step 2: Parse Each Log File

`parse_txt_log(filepath, node_tile_map)` performs four passes over each file:

### 2a. Header

```python
header_match = re.search(r'BOT_NAME:\s*(\w+)', content)
winner_match = re.search(r'WINNER:\s*Color\.(\w+)', content)
p0_color_match = re.search(r'P0_COLOR:\s*Color\.(\w+)', content)
```

Extracts the bot name, P0's color (e.g. `BLUE`), and the game winner.

### 2b. Board Layout → `board_map`

The log contains a `--- BOARD LAYOUT ---` section with lines like:

```
HEX (0, 0, 0): WOOD 5
HEX (1, -1, 0): WHEAT 11
```

Parsed into `board_map[coord] = {'resource': str, 'number': int}`. This is the per-game resource assignment.

### 2c. Actions → Per-Game Metrics

The `--- ACTIONS ---` section is iterated sequentially. Each action is tagged with the player color (e.g. `[BLUE]`).

#### Maritime Trades (Bank + Port)

```python
if 'MARITIME_TRADE' in action:
    trades_proposed += 1
    trades_completed += 1
    bank_trades += 1
```

The resource tuple format is `(given, given, given, given, received)` for 4:1 bank trades, and `(given, given, None, None, received)` for port trades (3:1 or 2:1):

```python
match = re.search(r"\('(.*)'\)", action)
if len(resources) == 5:
    if 'None' in resources:
        port_used += 1
        given_count = sum(1 for r in resources[:4] if r != 'None')
    else:
        given_count = 4
```

Traded-away and received resources are tracked per-type (`ore`, `wool`, `lumber`, `grain`, `brick`).

#### Build Actions

```python
elif 'BUILD_ROAD' in action:        roads_built += 1
elif 'BUILD_SETTLEMENT' in action:  settlements_built += 1; node_id captured
elif 'BUILD_CITY' in action:        cities_built += 1
elif 'BUY_DEVELOPMENT_CARD':        dev_cards_bought += 1
elif 'PLAY_KNIGHT_CARD':            knights_played += 1
```

Settlement node IDs are recorded for starting resource calculation.

#### Robber Targeting

```python
elif 'MOVE_ROBBER' in action:
    target_match = re.search(r'<Color\.(\w+):', action)
    if target_match:
        target_color = target_match.group(1)
        if target_color != bot_color:
            players_targeted += 1
            unique_players_stolen_from.add(target_color)
```

When a different player moves the robber and targets our bot's color, we increment `times_targeted`.

#### Opponent Targeting Us

```python
if f'[{bot_color}]' not in action:
    if 'MOVE_ROBBER' in action and f'<Color.{bot_color}' in action:
        times_targeted += 1
```

### 2d. Final Player State → End-of-Game Stats

The `--- FINAL PLAYER STATE (P0) ---` JSON section provides:

```python
hand_size = P0_WOOD + P0_BRICK + P0_SHEEP + P0_WHEAT + P0_ORE
longest_road_length = P0_LONGEST_ROAD_LENGTH
has_largest_army  = P0_HAS_ARMY
actual_vp         = P0_ACTUAL_VICTORY_POINTS
```

### 2e. Starting Resources (Key: Static Map + Per-Game Board)

```python
resource_map = {
    'WOOD': 'lumber', 'WHEAT': 'grain', 'ORE': 'ore',
    'SHEEP': 'wool', 'BRICK': 'brick', 'DESERT': None
}

for node_id in settlement_coords[:2]:  # First two settlements only
    if node_id in node_tile_map:
        for coord in node_tile_map[node_id]:  # Adjacent tile coords
            if coord in board_map:
                res = board_map[coord]['resource']
                mapped = resource_map.get(res)
                starting_{mapped} += 1
```

This is the critical step: the **static** `node_tile_map` gives coordinates; the **per-game** `board_map` gives resources at those coordinates. Only the first two settlements count as "starting" (placed during setup phase before the first roll).

---

## Step 3: Aggregate Per-Bot Profiles

All per-game metrics are collected into a Polars DataFrame and grouped by `bot_name`:

```python
player_profiles = df_all.group_by('bot_name').agg(
    num_games=pl.len(),
    avg_turns_before_first_trade=pl.col('turns_before_first_trade')
        .filter(pl.col('turns_before_first_trade') != -1).mean(),
    ratio_cards_given_to_taken=pl.col('cards_given_in_trades').sum()
        / pl.col('cards_gained_in_trades').sum().clip(lower_bound=1),
    trade_success_rate=pl.col('trades_completed').sum()
        / pl.col('trades_proposed').sum().clip(lower_bound=1),
    avg_counter_offers=pl.col('counter_offers').mean(),
    avg_bank_trades=pl.col('bank_trades').mean(),
    avg_dev_cards_bought=pl.col('dev_cards_bought').mean(),
    avg_roads_built=pl.col('roads_built').mean(),
    avg_cities_built=pl.col('cities_built').mean(),
    avg_players_targeted=pl.col('unique_players_stolen_from').mean(),
    avg_times_targeted=pl.col('times_targeted_by_others').mean(),
    win_rate_largest_army=(pl.col('largest_army_received') > 0).sum() / pl.len(),
    win_rate_longest_road=(pl.col('longest_road_received') > 0).sum() / pl.len(),
    average_games_with_port_used=pl.col('port_used').sum() / pl.len(),
    overall_avg_hand_size=pl.col('avg_hand_size').mean(),
    # Totals for derived metrics
    total_starting_ore=pl.col('starting_ore').sum(),
    # ... (all 5 resources, traded-away, received)
)
```

---

## Step 4: Derived Metrics

After aggregation, three qualitative metrics are computed in Python (not expressible as pure Polars expressions):

```python
# Top 3 starting resources by total count across all games
res_counts = {'ore': ..., 'wool': ..., 'lumber': ..., 'grain': ..., 'brick': ...}
top_3 = [res for res, count in sorted_res[:3]]

# Most traded-away resource
most_traded_away = max(traded_away, key=traded_away.get)

# Most received resource
most_received = max(received_trade, key=received_trade.get)
```

These are joined back onto the profile and the intermediate `total_*` columns are dropped.

---

## Step 5: Output

```python
output_path = Path('../../data/player_profiles/parsed_eval_logs_profiles.parquet')
player_profiles.write_parquet(output_path)
```

The final parquet has 19 columns, matching the schema from `player_profiler.ipynb`:

| Column | Type | Source |
|--------|------|--------|
| `bot_name` | str | Header |
| `num_games` | u32 | Count |
| `avg_turns_before_first_trade` | f64 | First MARITIME_TRADE index |
| `ratio_cards_given_to_taken` | f64 | Aggregate ratio |
| `trade_success_rate` | f64 | Aggregate ratio |
| `avg_counter_offers` | f64 | Mean per game |
| `avg_bank_trades` | f64 | Mean per game |
| `avg_dev_cards_bought` | f64 | Mean per game |
| `avg_roads_built` | f64 | Mean per game |
| `avg_cities_built` | f64 | Mean per game |
| `avg_players_targeted` | f64 | Unique players robbed, mean |
| `avg_times_targeted` | f64 | Robber placed on us, mean |
| `win_rate_largest_army` | f64 | P0_HAS_ARMY rate |
| `win_rate_longest_road` | f64 | longest_road >= 5 rate |
| `average_games_with_port_used` | f64 | Port trade rate |
| `overall_avg_hand_size` | f64 | Mean hand at game end |
| `top_3_starting_resources` | str | Derived from node→tile→resource |
| `most_traded_away_resource` | str | Derived from trade counts |
| `most_received_resource` | str | Derived from trade counts |

---

## Optional: Log Cleanup

Cell 8 contains commented-out code to delete the 500 `.txt` files after successful parsing:

```python
# for txt_file in txt_files:
#     txt_file.unlink()
```
