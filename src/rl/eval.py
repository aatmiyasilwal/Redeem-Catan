# (012) 49% win rate, 7.41 avg VP against the default opponents in catanatron_gym:catanatron-v1 (unwrapped)

import argparse
import json
import polars as pl
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from sb3_contrib import MaskablePPO
from train import make_create_masked_env


def eval_agent(model_path: str, opponents: list, n_games: int = 500, out_filename: str = "baseline.csv"):
    env_fn = make_create_masked_env(opponents)
    env = env_fn()

    print(f"Loading model from {model_path}...")
    model = MaskablePPO.load(model_path)

    results = []

    print(f"Evaluating {model_path} for {n_games} games...")
    for i in tqdm(range(n_games), desc="Games Played"):
        obs, info = env.reset()
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
            "game_id": i,
            "won": won,
            "vp": vp
        })

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
    parser.add_argument("-p", "--players", type=str, required=True, metavar="P0,P1,P2",
                        help="Comma-separated list of 3 player indices, eg: 0,1,2")
    parser.add_argument("-m", "--mode", type=str, choices=["b", "a", "s"], default="b",
                        help="Mode: b (baseline), a (aware), s (shuffled)")
    args = parser.parse_args()

    indices = sorted([int(x.strip()) for x in args.players.split(',')])
    if len(indices) != 3:
        raise ValueError("Must provide exactly 3 player indices.")

    index_path = Path(__file__).resolve().parent.parent.parent / \
        "data" / "player_profiles" / "player_index.json"
    with open(index_path, 'r') as f:
        player_map = json.load(f)

    reverse_map = {int(v): k for k, v in player_map.items()}
    opponents = []
    for idx in indices:
        if idx not in reverse_map:
            raise ValueError(
                f"Index {idx} not valid. Valid indices: {list(reverse_map.keys())}")
        opponents.append(reverse_map[idx])
    
    suffix = "".join(str(idx) for idx in indices)    
    
    if args.mode == "b":
        prefix = "baseline"
    elif args.mode == "a":
        prefix = "aware"
    elif args.mode == "s":
        prefix = "shuffled"
    else:
        prefix = "baseline"


    # Ensure the model exists before running
    model_path = Path(f"models/{prefix}_ppo_{suffix}.zip")
    if model_path.exists():
        # Generate timestamp in MMDD_HHMMSS format
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        export_name = f"{timestamp}_{prefix}_{suffix}.csv"

        eval_agent(str(model_path), opponents=opponents,
                   n_games=500, out_filename=export_name)
    else:
        print(
            f"Could not find {model_path} - did you finish training the {prefix} model in train.py with these players?")
