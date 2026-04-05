import polars as pl
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from sb3_contrib import MaskablePPO
from train import create_masked_env


def eval_agent(model_path: str, n_games: int = 500, out_filename: str = "baseline.csv"):
    env = create_masked_env()

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
    # Ensure the model exists before running
    baseline_path = Path("models/baseline_ppo.zip")
    if baseline_path.exists():
        # Generate timestamp in MMDD_HHMMSS format
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        export_name = f"{timestamp}_baseline.csv"

        eval_agent(str(baseline_path), n_games=500, out_filename=export_name)
    else:
        print(
            f"Could not find {baseline_path} - did you finish training the baseline in train.py?")
