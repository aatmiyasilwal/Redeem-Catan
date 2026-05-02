# (012) 49% win rate, 7.41 avg VP against the default opponents in catanatron_gym:catanatron-v1 (unwrapped)

import argparse
import json
import polars as pl
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import set_random_seed
from train import make_create_masked_env


def eval_agent(model_path: str, opponents: list, mode: str = "baseline", axelrod=False, n_games: int = 500, out_filename: str = "baseline.csv", bot_name: str = "bot"):
    env_fn = make_create_masked_env(opponents, mode=mode, axelrod=axelrod)
    env = env_fn()

    print(f"Loading model from {model_path}...")
    model = MaskablePPO.load(model_path)

    results = []

    # Ensure log directory exists
    log_dir = Path(__file__).resolve().parent.parent.parent / \
        "data" / "eval_logs"
    log_dir.mkdir(exist_ok=True, parents=True)

    print(f"Evaluating {model_path} for {n_games} games...")
    for i in tqdm(range(n_games), desc="Games Played"):
        # explicitly control the seed for the initial game states to make comparison completely reproducible
        obs, info = env.reset(seed=100 + i)
        done = False

        while not done:
            # Fetch valid action masks for prediction at this specific state
            action_masks = env.action_masks()

            # Predict the best deterministic action using the mask
            action, _states = model.predict(
                obs, action_masks=action_masks, deterministic=True)

            # Take the action in the environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # Post-game extraction
        game = env.unwrapped.game  # type: ignore

        # In catanatron_gym, the RL agent is always mapped to Player 0 (P0)
        vp = game.state.player_state["P0_ACTUAL_VICTORY_POINTS"]
        won = (game.winning_color() == game.state.colors[0])

        results.append({
            "game_id": i + 1,
            "won": won,
            "vp": vp
        })

        # --- Write Contextual Game Log per Step 3 ---
        p0_color = game.state.colors[0]
        log_path = log_dir / f"{bot_name}_game_{i + 1}.txt"

        with open(log_path, "w") as f:
            f.write(f"BOT_NAME: {bot_name}\n")
            f.write(f"GAME_ID: {i + 1}\n")
            f.write(f"P0_COLOR: {p0_color}\n")
            f.write(f"WINNER: {game.winning_color()}\n")

            f.write("\n--- BOARD LAYOUT ---\n")
            # Dump the board layout: coordinate -> (resource string, number token)
            # board.map.land_tiles is a dictionary mapping tuple coordinate -> LandTile
            for coord, tile in game.state.board.map.land_tiles.items():
                if hasattr(tile, 'resource') and hasattr(tile, 'number'):
                    # tile.resource is already a string like "WHEAT", "ORE", etc.
                    res_name = tile.resource if tile.resource else "DESERT"
                    num = tile.number if tile.number is not None else 0
                    f.write(f"HEX {coord}: {res_name} {num}\n")

            f.write("\n--- ACTIONS ---\n")
            for act in game.state.actions:
                f.write(
                    f"[{act.color.name}] | {act.action_type.name} | {act.value}\n")

            f.write("\n--- FINAL PLAYER STATE (P0) ---\n")
            # Dump P0's final dictionary to make extracting stats (Largest Army, Longest Road, etc) trivial
            p0_stats = {k: v for k, v in game.state.player_state.items()
                        if k.startswith("P0_")}
            json.dump(p0_stats, f, indent=2)
            f.write("\n")

    env.close()

    # Save the results
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / out_filename

    df = pl.DataFrame(results)
    df.write_csv(out_path)

    # Print a summary
    win_rate = df["won"].mean() * 100  # type: ignore
    avg_vp = df["vp"].mean()
    print("\n" + "="*30)
    print(f"EVALUATION RESULTS")
    print("="*30)
    print(f"Model    : {model_path}")
    print(f"Games    : {n_games}")
    print(f"Win Rate : {win_rate:.1f}%")
    print(f"Avg VP   : {avg_vp:.2f}")
    print(f"Saved to : {out_path}")
    print("="*30 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Catanatron PPO Agent")
    parser.add_argument("-p", "--players", type=str, default="", metavar="P0,P1,P2",
                        help="Comma-separated list of 3 player indices, eg: 0,1,2 (Optional for aware/shuffled)")
    parser.add_argument("-m", "--mode", type=str, choices=["b", "a", "s"], default="b",
                        help="Mode: b (baseline), a (aware), s (shuffled)")
    parser.add_argument("--axelrod", type=int, choices=[0, 1], default=0,
                        help="Enable Axelrod's Tit-for-Tat logic (override target choice)")
    args = parser.parse_args()

    # Set random seeds for reproducibility
    seed = 100
    set_random_seed(seed)
    np.random.seed(seed)

    if args.mode == "b" and not args.players:
        raise ValueError("Must provide -p for baseline mode.")

    opponents = []
    indices = []
    if args.players:
        indices = sorted([int(x.strip()) for x in args.players.split(',')])
        if len(indices) != 3:
            raise ValueError("Must provide exactly 3 player indices.")

        index_path = Path(__file__).resolve().parent.parent.parent / \
            "data" / "player_profiles" / "player_index.json"
        with open(index_path, 'r') as f:
            player_map = json.load(f)

        reverse_map = {int(v): k for k, v in player_map.items()}
        for idx in indices:
            if idx not in reverse_map:
                raise ValueError(
                    f"Index {idx} not valid. Valid indices: {list(reverse_map.keys())}")
            opponents.append(reverse_map[idx])

    suffix = "".join(str(idx) for idx in indices) if indices else "all"

    if args.mode == "b":
        trained_prefix = "baseline"
    elif args.mode == "a":
        trained_prefix = "aware"
    elif args.mode == "s":
        trained_prefix = "shuffled"
    else:
        trained_prefix = "baseline"

    axelrod_flag = bool(args.axelrod)
    eval_prefix = trained_prefix + "_axelrod" if axelrod_flag else trained_prefix

    # Ensure the model exists before running
    # Make sure we search inside the relative models directory correctly
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / f"models/{eval_prefix}_ppo_{suffix}.zip"

    # Fallback to general model if evaluating specific players but only trained general
    if not model_path.exists() and suffix != "all":
        fallback_path = base_dir / f"models/{eval_prefix}_ppo_all.zip"
        if fallback_path.exists():
            print(
                f"Specific model {model_path} not found. Using generalized model: {fallback_path}")
            model_path = fallback_path

    if model_path.exists():
        # Generate timestamp in MMDD_HHMMSS format
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        export_name = f"{timestamp}_{eval_prefix}_{suffix}.csv"

        eval_agent(str(model_path), opponents=opponents, mode=trained_prefix, axelrod=axelrod_flag,
                   n_games=500, out_filename=export_name, bot_name=eval_prefix)
    else:
        print(
            f"Could not find {model_path} - did you finish training the {eval_prefix} model in train.py with these players?")
