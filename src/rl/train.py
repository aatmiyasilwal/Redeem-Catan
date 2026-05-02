import gymnasium as gym
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from make_env import make_env


class LossTrackingCallback(BaseCallback):
    """
    Callback to track training loss at every update step.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.losses = []
        self.timesteps = []

    def _on_step(self) -> bool:
        # SB3 logs "train/loss" at the end of every update
        if "train/loss" in self.logger.name_to_value:
            self.losses.append(self.logger.name_to_value["train/loss"])
            self.timesteps.append(self.num_timesteps)
        return True


def mask_fn(env: gym.Env) -> np.ndarray:
    """Extract valid actions from the environment and format as a binary array."""
    valid_actions = env.unwrapped.get_valid_actions()  # type: ignore
    mask = np.zeros(env.action_space.n, dtype=np.float32)  # type: ignore
    mask[valid_actions] = 1
    return mask


def make_create_masked_env(opponents, mode="baseline", axelrod=False):
    """Factory builder for creating masked envs with specific opponents."""
    def _create_masked_env():
        env = make_env(opponents, mode=mode, axelrod=axelrod)
        env = ActionMasker(env, mask_fn)
        return env
    return _create_masked_env


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Catanatron PPO Agent")
    parser.add_argument("-p", "--players", type=str, default="0,1,3", metavar="P0,P1,P2",
                        help="Comma-separated list of 3 player indices, eg: 0,1,2 (Optional for aware/shuffled)")
    parser.add_argument("-m", "--mode", type=str, choices=["b", "a", "s"], default="b",
                        help="Training mode: b (baseline), a (aware), s (shuffled)")
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

    # Create directory for saving models (relative to script location)
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)

    # Create directory for saving plots
    plots_dir = Path(__file__).parent / "plots"
    plots_dir.mkdir(exist_ok=True)

    suffix = "".join(str(idx) for idx in indices) if indices else "all"

    if args.mode == "b":
        prefix = "baseline"
    elif args.mode == "a":
        prefix = "aware"
    elif args.mode == "s":
        prefix = "shuffled"
    else:
        prefix = "baseline"

    axelrod_flag = bool(args.axelrod)
    if axelrod_flag:
        prefix += "_axelrod"

    print(
        f"Initializing environment with opponents: {opponents} (mode: {prefix})...")
    
    # Vectorize the environment (running 4 parallel environments speeds up training drastically)
    env_fn = make_create_masked_env(
        opponents, mode=prefix.replace("_axelrod", ""), axelrod=axelrod_flag)
    vec_env = DummyVecEnv([env_fn for _ in range(4)])

    print("Loading Maskable PPO Model...")
    # Initialize Maskable PPO Model
    model = MaskablePPO("MlpPolicy", vec_env, verbose=1, seed=seed)

    # Initialize loss tracking callback
    loss_callback = LossTrackingCallback()

    # training the full baseline model for 500k steps
    model_name = f"{prefix}_ppo_{suffix}.zip"
    print(f"Starting 500k {prefix} training for {model_name}...")
    model.learn(total_timesteps=500_000, callback=loss_callback)
    model.save(models_dir / model_name)
    print(f"Full {prefix} completed and saved to {models_dir / model_name}!")

    # ── Generate Loss Plot ──────────────────────────────────────────────────
    if loss_callback.timesteps and loss_callback.losses:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(loss_callback.timesteps, loss_callback.losses, label="Train Loss", color="blue", linewidth=1.5)
        ax.set_title(f"Training Loss Over Time: {prefix.upper()} vs Players {suffix}")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Total Loss")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

        plot_filename = f"loss_chart_{suffix}_{prefix}.png"
        plot_path = plots_dir / plot_filename
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Loss chart saved to {plot_path}")
    else:
        print("Warning: No training data captured to generate plot.")
