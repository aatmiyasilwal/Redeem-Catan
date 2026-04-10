import gymnasium as gym
import catanatron_gym
import numpy as np
import json
from pathlib import Path
from profiles import get_profile_vector, get_profile_dim


class OpponentProfileWrapper(gym.ObservationWrapper):
    """
    Injects pre-computed opponent profiles into the RL agent's observation space.
    This lets the agent know *who* it is playing against.
    """

    def __init__(self, env, opponent_names, mode="baseline"):
        super().__init__(env)
        self.opponent_names = opponent_names
        self.mode = mode

        # Load the feature vectors for the initial/baseline opponents
        vectors = [get_profile_vector(name) for name in opponent_names]
        self.current_features = np.concatenate(vectors).astype(np.float32)

        # Expand the observation space dimension
        base_shape = env.observation_space.shape[0]
        new_dim = base_shape + len(self.current_features)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(new_dim,),
            dtype=np.float32
        )

        # Load all possible opponents for dynamic sampling
        if self.mode != "baseline":
            index_path = Path(__file__).resolve(
            ).parent.parent.parent / "data" / "player_profiles" / "player_index.json"
            if index_path.exists():
                with open(index_path, 'r') as f:
                    player_map = json.load(f)
                self.all_opponents = list(player_map.keys())
            else:
                self.all_opponents = opponent_names

    def reset(self, **kwargs):
        # Dynamically sample and build new features on every reset if not baseline
        if self.mode in ["aware", "shuffled"]:
            sampled_opponents = np.random.choice(
                self.all_opponents, size=3, replace=False)
            vectors = [get_profile_vector(name) for name in sampled_opponents]
            features = np.concatenate(vectors).astype(np.float32)

            # Ablation testing introduces random noise instead of usable signal
            if self.mode == "shuffled":
                np.random.shuffle(features)

            self.current_features = features

        return super().reset(**kwargs)

    def observation(self, obs):
        # Concatenate the dynamic base observation with static/dynamic opponent features
        return np.concatenate([obs, self.current_features])


def make_env(opponent_names, mode="baseline"):
    """
    Creates the Catanatron gym environment and wraps it with static opponent profiles.
    """
    # The environment ID is catanatron-v1, namespaced under catanatron_gym
    env = gym.make("catanatron_gym:catanatron-v1")
    env = OpponentProfileWrapper(env, opponent_names, mode=mode)
    return env
