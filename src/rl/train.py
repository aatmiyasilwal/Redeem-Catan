import gymnasium as gym
import numpy as np
from pathlib import Path
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from make_env import make_env

def mask_fn(env: gym.Env) -> np.ndarray:
    """Extract valid actions from the environment and format as a binary array."""
    valid_actions = env.unwrapped.get_valid_actions() # type: ignore
    mask = np.zeros(env.action_space.n, dtype=np.float32) # type: ignore
    mask[valid_actions] = 1
    return mask

def create_masked_env():
    """Environment factory combining the RL wrapper and the ActionMasker."""
    # Default baseline opponents
    opponents = ["AatNeverLose", "HomeofAD3005", "ZL24"] 
    env = make_env(opponents)
    env = ActionMasker(env, mask_fn)
    return env

if __name__ == "__main__":
    # Create directory for saving models
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    print("Initializing environment...")
    # Vectorize the environment (running 4 parallel environments speeds up training drastically)
    vec_env = DummyVecEnv([create_masked_env for _ in range(4)])
    
    print("Loading Maskable PPO Model...")
    # Initialize Maskable PPO Model
    model = MaskablePPO("MlpPolicy", vec_env, verbose=1)
    
    # print("Starting 50k validation run...")
    # # Train Tiny PPO (Sanity Check)
    # model.learn(total_timesteps=50_000)
    # model.save(models_dir / "tiny_ppo.zip")
    # print(f"Successfully saved {models_dir / 'tiny_ppo.zip'}!")
    
    # training the full baseline model for 500k steps
    print("Starting 500k baseline training...")
    model.learn(total_timesteps=500_000)
    model.save(models_dir / "baseline_ppo.zip")
    print("Full baseline completed!")
