#!/usr/bin/env python3
"""
Parse evaluation .txt logs from data/eval_logs/ and produce player profile
parquet files compatible with the output of player_profiler.ipynb.

Uses Catanatron's static node-to-tile geometry to map each log's board
layout (resource-per-coordinate) back to the two initial settlement nodes,
thereby recovering starting resources for every game.
"""

import argparse
import json
import re
from pathlib import Path

import polars as pl
from catanatron.models.board import Board


def build_node_tile_map():
    """Build mapping from node_id -> list of adjacent tile coordinate tuples."""
    b = Board()
    cm = b.map

    tile_id_to_coord = {}
    for coord, tile in cm.land_tiles.items():
        tile_id_to_coord[tile.id] = coord

    node_tile_map = {}
    for node_id, tiles in cm.adjacent_tiles.items():
        node_tile_map[node_id] = [tile_id_to_coord[t.id] for t in tiles]

    return node_tile_map


def parse_txt_log(filepath, node_tile_map):
    """Parse a single .txt log file and extract all relevant metrics for P0."""
    with open(filepath, "r") as f:
        content = f.read()

    metrics = {}

    # Parse header
    header_match = re.search(r"BOT_NAME:\s*(\w+)", content)
    if header_match:
        metrics["bot_name"] = header_match.group(1)

    game_id_match = re.search(r"GAME_ID:\s*(\d+)", content)
    if game_id_match:
        metrics["game_id"] = int(game_id_match.group(1))

    winner_match = re.search(r"WINNER:\s*Color\.(\w+)", content)
    if winner_match:
        metrics["winner"] = winner_match.group(1)

    p0_color_match = re.search(r"P0_COLOR:\s*Color\.(\w+)", content)
    if p0_color_match:
        metrics["p0_color"] = p0_color_match.group(1)

    # Parse board layout into coordinate -> {resource, number}
    board_section = re.search(
        r"--- BOARD LAYOUT ---\n(.*?)--- ACTIONS ---", content, re.DOTALL
    )
    board_map = {}
    if board_section:
        for line in board_section.group(1).strip().split("\n"):
            m = re.search(r"HEX\s*\(([^)]+)\):\s*(\w+)\s*(\d+)", line)
            if m:
                coord = tuple(map(int, m.group(1).split(", ")))
                board_map[coord] = {"resource": m.group(2), "number": int(m.group(3))}

    # Parse actions
    actions_section = re.search(
        r"--- ACTIONS ---\n(.*?)--- FINAL PLAYER STATE", content, re.DOTALL
    )
    actions = []
    if actions_section:
        actions = [
            line.strip()
            for line in actions_section.group(1).split("\n")
            if line.strip() and not line.startswith("---")
        ]

    bot_color = metrics.get("p0_color", "").upper()

    # Counters
    trades_proposed = trades_completed = counter_offers = bank_trades = port_used = 0
    cards_given_in_trades = cards_taken_in_trades = 0
    turns_before_first_trade = -1
    first_trade_seen = False

    traded_away = {"ore": 0, "wool": 0, "lumber": 0, "grain": 0, "brick": 0}
    received_trade = {"ore": 0, "wool": 0, "lumber": 0, "grain": 0, "brick": 0}

    roads_built = settlements_built = cities_built = 0
    dev_cards_bought = knights_played = 0
    players_targeted = times_targeted = 0
    unique_players_stolen_from = set()
    settlement_coords = []

    for i, action in enumerate(actions):
        if f"[{bot_color}]" not in action:
            if "MOVE_ROBBER" in action and f"<Color.{bot_color}" in action:
                times_targeted += 1
            continue

        if "MARITIME_TRADE" in action:
            trades_proposed += 1
            trades_completed += 1
            bank_trades += 1
            if not first_trade_seen:
                turns_before_first_trade = i
                first_trade_seen = True

            m = re.search(r"\('(.*)'\)", action)
            if m:
                resources = [r.strip().strip("'") for r in m.group(1).split(", ")]
                if len(resources) == 5:
                    given_resource = resources[0]
                    received_resource = resources[4]

                    if "None" in resources:
                        port_used += 1
                        given_count = sum(1 for r in resources[:4] if r != "None")
                    else:
                        given_count = 4

                    if given_resource and given_resource != "None":
                        res_key = given_resource.lower()
                        if res_key in traded_away:
                            traded_away[res_key] += given_count
                            cards_given_in_trades += given_count

                    if received_resource and received_resource != "None":
                        res_key = received_resource.lower()
                        if res_key in received_trade:
                            received_trade[res_key] += 1
                            cards_taken_in_trades += 1

        elif "BUILD_ROAD" in action:
            roads_built += 1
        elif "BUILD_SETTLEMENT" in action:
            settlements_built += 1
            m = re.search(r"\|\s*BUILD_SETTLEMENT\s*\|\s*(\d+)", action)
            if m:
                settlement_coords.append(int(m.group(1)))
        elif "BUILD_CITY" in action:
            cities_built += 1
        elif "BUY_DEVELOPMENT_CARD" in action:
            dev_cards_bought += 1
        elif "PLAY_KNIGHT_CARD" in action:
            knights_played += 1
        elif "MOVE_ROBBER" in action:
            target_match = re.search(r"<Color\.(\w+):", action)
            if target_match:
                target_color = target_match.group(1)
                if target_color != bot_color:
                    players_targeted += 1
                    unique_players_stolen_from.add(target_color)

    # Parse final player state JSON
    player_state = {}
    json_match = re.search(
        r"--- FINAL PLAYER STATE \(P0\) ---\n(\{.*\})", content, re.DOTALL
    )
    if json_match:
        try:
            player_state = json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    hand_size = sum(
        player_state.get(f"P0_{r}_IN_HAND", 0)
        for r in ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
    )
    longest_road_length = player_state.get("P0_LONGEST_ROAD_LENGTH", 0)
    has_largest_army = player_state.get("P0_HAS_ARMY", False)
    actual_vp = player_state.get("P0_ACTUAL_VICTORY_POINTS", 0)

    # Starting resources from settlement nodes + per-game board map
    resource_map = {
        "WOOD": "lumber",
        "WHEAT": "grain",
        "ORE": "ore",
        "SHEEP": "wool",
        "BRICK": "brick",
        "DESERT": None,
    }

    starting = {"ore": 0, "wool": 0, "lumber": 0, "grain": 0, "brick": 0}
    for node_id in settlement_coords[:2]:
        if node_id in node_tile_map:
            for coord in node_tile_map[node_id]:
                if coord in board_map:
                    mapped = resource_map.get(board_map[coord]["resource"])
                    if mapped and mapped in starting:
                        starting[mapped] += 1

    won = metrics.get("winner", "").upper() == bot_color

    metrics.update(
        {
            "turns_before_first_trade": turns_before_first_trade,
            "cards_given_in_trades": cards_given_in_trades,
            "cards_gained_in_trades": cards_taken_in_trades,
            "trades_proposed": trades_proposed,
            "trades_completed": trades_completed,
            "counter_offers": counter_offers,
            "bank_trades": bank_trades,
            "port_used": 1 if port_used > 0 else 0,
            "dev_cards_bought": dev_cards_bought,
            "roads_built": roads_built,
            "settlements_built": settlements_built,
            "cities_built": cities_built,
            "unique_players_stolen_from": len(unique_players_stolen_from),
            "times_targeted_by_others": times_targeted,
            "largest_army_received": 1 if has_largest_army else 0,
            "longest_road_received": 1 if longest_road_length >= 5 else 0,
            "avg_hand_size": hand_size,
            "won": 1 if won else 0,
            "actual_vp": actual_vp,
            "longest_road_length": longest_road_length,
            "starting_ore": starting["ore"],
            "starting_wool": starting["wool"],
            "starting_lumber": starting["lumber"],
            "starting_grain": starting["grain"],
            "starting_brick": starting["brick"],
            "traded_away_ore": traded_away["ore"],
            "traded_away_wool": traded_away["wool"],
            "traded_away_lumber": traded_away["lumber"],
            "traded_away_grain": traded_away["grain"],
            "traded_away_brick": traded_away["brick"],
            "received_trade_ore": received_trade["ore"],
            "received_trade_wool": received_trade["wool"],
            "received_trade_lumber": received_trade["lumber"],
            "received_trade_grain": received_trade["grain"],
            "received_trade_brick": received_trade["brick"],
        }
    )

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Parse Catan eval logs into profiles")
    parser.add_argument(
        "--input",
        type=str,
        default="data/eval_logs",
        help="Directory containing .txt log files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/player_profiles/parsed_eval_logs_profiles.parquet",
        help="Output parquet path",
    )
    args = parser.parse_args()

    logs_dir = Path(args.input)
    txt_files = sorted(logs_dir.glob("*.txt"))
    print(f"Found {len(txt_files)} log files to parse")

    node_tile_map = build_node_tile_map()
    print(f"Built node-to-tile mapping for {len(node_tile_map)} nodes")

    all_metrics = []
    for txt_file in txt_files:
        try:
            metrics = parse_txt_log(txt_file, node_tile_map)
            all_metrics.append(metrics)
            if len(all_metrics) % 50 == 0:
                print(f"Parsed {len(all_metrics)} files...")
        except Exception as e:
            print(f"Error parsing {txt_file}: {e}")

    if not all_metrics:
        print("No metrics parsed!")
        return

    df_all = pl.DataFrame(all_metrics)
    print(f"Aggregated {len(all_metrics)} games.")

    player_profiles = df_all.group_by("bot_name").agg(
        num_games=pl.len(),
        avg_turns_before_first_trade=pl.col("turns_before_first_trade")
        .filter(pl.col("turns_before_first_trade") != -1)
        .mean(),
        ratio_cards_given_to_taken=(
            pl.col("cards_given_in_trades").sum()
            / pl.col("cards_gained_in_trades").sum().clip(lower_bound=1)
        ),
        trade_success_rate=(
            pl.col("trades_completed").sum()
            / pl.col("trades_proposed").sum().clip(lower_bound=1)
        ),
        avg_counter_offers=pl.col("counter_offers").mean(),
        avg_bank_trades=pl.col("bank_trades").mean(),
        avg_dev_cards_bought=pl.col("dev_cards_bought").mean(),
        avg_roads_built=pl.col("roads_built").mean(),
        avg_cities_built=pl.col("cities_built").mean(),
        avg_players_targeted=pl.col("unique_players_stolen_from").mean(),
        avg_times_targeted=pl.col("times_targeted_by_others").mean(),
        win_rate_largest_army=(pl.col("largest_army_received") > 0).sum() / pl.len(),
        win_rate_longest_road=(pl.col("longest_road_received") > 0).sum() / pl.len(),
        average_games_with_port_used=pl.col("port_used").sum() / pl.len(),
        overall_avg_hand_size=pl.col("avg_hand_size").mean(),
        total_starting_ore=pl.col("starting_ore").sum(),
        total_starting_wool=pl.col("starting_wool").sum(),
        total_starting_lumber=pl.col("starting_lumber").sum(),
        total_starting_grain=pl.col("starting_grain").sum(),
        total_starting_brick=pl.col("starting_brick").sum(),
        total_traded_away_ore=pl.col("traded_away_ore").sum(),
        total_traded_away_wool=pl.col("traded_away_wool").sum(),
        total_traded_away_lumber=pl.col("traded_away_lumber").sum(),
        total_traded_away_grain=pl.col("traded_away_grain").sum(),
        total_traded_away_brick=pl.col("traded_away_brick").sum(),
        total_received_trade_ore=pl.col("received_trade_ore").sum(),
        total_received_trade_wool=pl.col("received_trade_wool").sum(),
        total_received_trade_lumber=pl.col("received_trade_lumber").sum(),
        total_received_trade_grain=pl.col("received_trade_grain").sum(),
        total_received_trade_brick=pl.col("received_trade_brick").sum(),
    )

    # Derived metrics
    derived_metrics_list = []
    for row in player_profiles.iter_rows(named=True):
        bot_name = row["bot_name"]

        res_counts = {
            "ore": row["total_starting_ore"],
            "wool": row["total_starting_wool"],
            "lumber": row["total_starting_lumber"],
            "grain": row["total_starting_grain"],
            "brick": row["total_starting_brick"],
        }
        top_3 = [
            res
            for res, _ in sorted(res_counts.items(), key=lambda x: x[1], reverse=True)[
                :3
            ]
        ]

        traded_away = {
            "ore": row["total_traded_away_ore"],
            "wool": row["total_traded_away_wool"],
            "lumber": row["total_traded_away_lumber"],
            "grain": row["total_traded_away_grain"],
            "brick": row["total_traded_away_brick"],
        }
        most_traded_away = (
            max(traded_away, key=traded_away.get) # type: ignore
            if sum(traded_away.values()) > 0
            else "None"
        )

        received_trade = {
            "ore": row["total_received_trade_ore"],
            "wool": row["total_received_trade_wool"],
            "lumber": row["total_received_trade_lumber"],
            "grain": row["total_received_trade_grain"],
            "brick": row["total_received_trade_brick"],
        }
        most_received = (
            max(received_trade, key=received_trade.get) # type: ignore
            if sum(received_trade.values()) > 0
            else "None"
        )

        derived_metrics_list.append(
            {
                "bot_name": bot_name,
                "top_3_starting_resources": ", ".join(top_3),
                "most_traded_away_resource": most_traded_away,
                "most_received_resource": most_received,
            }
        )

    df_derived = pl.DataFrame(derived_metrics_list)
    player_profiles = player_profiles.join(df_derived, on="bot_name", how="left")

    columns_to_drop = [
        "total_starting_ore",
        "total_starting_wool",
        "total_starting_lumber",
        "total_starting_grain",
        "total_starting_brick",
        "total_traded_away_ore",
        "total_traded_away_wool",
        "total_traded_away_lumber",
        "total_traded_away_grain",
        "total_traded_away_brick",
        "total_received_trade_ore",
        "total_received_trade_wool",
        "total_received_trade_lumber",
        "total_received_trade_grain",
        "total_received_trade_brick",
    ]
    player_profiles = player_profiles.drop(columns_to_drop)
    player_profiles = player_profiles.sort("num_games", descending=True)

    print("\nPLAYER PROFILES:")
    print(player_profiles)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    player_profiles.write_parquet(output_path)
    print(f"\nPlayer profiles exported to {output_path.resolve()}")

    # Optional: uncomment to clean up txt logs after successful parsing
    # for txt_file in txt_files:
    #     txt_file.unlink()
    # print(f"Cleaned up {len(txt_files)} txt files from {logs_dir}")


if __name__ == "__main__":
    main()
