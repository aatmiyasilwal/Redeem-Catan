# 50.6% win rate, 7.57 avg VP against the default opponents in catanatron_gym:catanatron-v1 (unwrapped)

import gymnasium as gym
import catanatron_gym
import random
from tqdm import tqdm
import numpy as np


def evaluate_random_baseline(n_games: int = 500):
    """
    Evaluates a purely random agent taking valid actions 
    in the base Catanatron environment (without profile wrappers).
    """
    # Simply load the base Catanatron environment
    random.seed(100)
    np.random.seed(100)

    env = gym.make("catanatron_gym:catanatron-v1")

    results = []

    print(f"Evaluating Random Baseline Agent for {n_games} games...")
    for _ in tqdm(range(n_games), desc="Games Played"):
        obs, info = env.reset(seed=100)
        done = False

        while not done:
            # Fetch valid actions to avoid invalid move crashes
            valid_actions = env.unwrapped.get_valid_actions()  # type: ignore

            # The simplest baseline model: randomly picking a valid action
            action = random.choice(valid_actions)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # Post-game extraction
        game = env.unwrapped.game  # type: ignore

        vp = game.state.player_state["P0_ACTUAL_VICTORY_POINTS"]
        won = (game.winning_color() == game.state.colors[0])

        results.append({"won": won, "vp": vp})

    env.close()

    # Calculate summary metrics
    wins = sum(r["won"] for r in results)
    win_rate = (wins / n_games) * 100
    avg_vp = sum(r["vp"] for r in results) / n_games

    print("\n" + "="*40)
    print(" BASELINE RESULTS (Random Valid Actions)")
    print("="*40)
    print(f" Environment : catanatron_gym:catanatron-v1 (Unwrapped)")
    print(f" Games       : {n_games}")
    print(f" Win Rate    : {win_rate:.1f}%")
    print(f" Avg VP      : {avg_vp:.2f}")
    print("="*40 + "\n")


if __name__ == "__main__":
    evaluate_random_baseline(n_games=500)
