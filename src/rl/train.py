import gymnasium as gym
import numpy as np
import argparse
import json
from pathlib import Path
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from make_env import make_env


def mask_fn(env: gym.Env) -> np.ndarray:
    """Extract valid actions from the environment and format as a binary array."""
    valid_actions = env.unwrapped.get_valid_actions()  # type: ignore
    mask = np.zeros(env.action_space.n, dtype=np.float32)  # type: ignore
    mask[valid_actions] = 1
    return mask


def make_create_masked_env(opponents, mode="baseline"):
    """Factory builder for creating masked envs with specific opponents."""
    def _create_masked_env():
        env = make_env(opponents, mode=mode)
        env = ActionMasker(env, mask_fn)
        return env
    return _create_masked_env


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Catanatron PPO Agent")
    parser.add_argument("-p", "--players", type=str, default="", metavar="P0,P1,P2",
                        help="Comma-separated list of 3 player indices, eg: 0,1,2 (Optional for aware/shuffled)")
    parser.add_argument("-m", "--mode", type=str, choices=["b", "a", "s"], default="b",
                        help="Training mode: b (baseline), a (aware), s (shuffled)")
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

    # Create directory for saving models
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    suffix = "".join(str(idx) for idx in indices) if indices else "all"

    if args.mode == "b":
        prefix = "baseline"
    elif args.mode == "a":
        prefix = "aware"
    elif args.mode == "s":
        prefix = "shuffled"
    else:
        prefix = "baseline"

    print(
        f"Initializing environment with opponents: {opponents} (mode: {prefix})...")
    # Vectorize the environment (running 4 parallel environments speeds up training drastically)
    env_fn = make_create_masked_env(opponents, mode=prefix)
    vec_env = DummyVecEnv([env_fn for _ in range(4)])

    print("Loading Maskable PPO Model...")
    # Initialize Maskable PPO Model
    model = MaskablePPO("MlpPolicy", vec_env, verbose=1, seed=seed)

    # print("Starting 50k validation run...")
    # # Train Tiny PPO (Sanity Check)
    # model.learn(total_timesteps=50_000)
    # model.save(models_dir / f"tiny_ppo_{suffix}.zip")
    # print(f"Successfully saved {models_dir / 'tiny_ppo_{suffix}.zip'}!")

    # training the full baseline model for 500k steps
    model_name = f"{prefix}_ppo_{suffix}.zip"
    print(f"Starting 500k {prefix} training for {model_name}...")
    model.learn(total_timesteps=500_000)
    model.save(models_dir / model_name)
    print(f"Full {prefix} completed and saved to {models_dir / model_name}!")
