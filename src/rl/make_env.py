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

        # Load player mapping (fixed indices for slot masking)
        index_path = Path(__file__).resolve().parent.parent.parent / \
            "data" / "player_profiles" / "player_index.json"
        with open(index_path, 'r') as f:
            self.player_map = json.load(f)
        self.all_opponents = list(self.player_map.keys())
        self.num_total_players = len(self.all_opponents)
        self.profile_dim = get_profile_dim()

        base_shape = env.observation_space.shape[0]

        if self.mode == "baseline":
            # Baseline: tight compact array of exactly 3 players
            vectors = [get_profile_vector(name)
                       for name in self.opponent_names]
            self.current_features = np.concatenate(vectors).astype(np.float32)
            new_dim = base_shape + len(self.current_features)
        else:
            # Aware/Shuffled: Fixed 5-player slot masking (5 * profile_dim)
            new_dim = base_shape + (self.num_total_players * self.profile_dim)
            self.current_features = np.zeros(
                self.num_total_players * self.profile_dim, dtype=np.float32)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(new_dim,),
            dtype=np.float32
        )

    def reset(self, **kwargs):
        # Dynamically sample and build new features on every reset if not baseline
        if self.mode in ["aware", "shuffled"]:
            # If explicit opponents were passed (for eval), lock them. Otherwise, random sample (for train).
            if self.opponent_names:
                active_opponents = self.opponent_names
            else:
                active_opponents = np.random.choice(
                    self.all_opponents, size=3, replace=False)

            # Start with an array of all zeros
            features = np.zeros(self.num_total_players *
                                self.profile_dim, dtype=np.float32)

            # Fill only the active players' respective slots based on their assigned index (0,1,2,3,4)
            for name in active_opponents:
                p_idx = self.player_map[name]
                start = p_idx * self.profile_dim
                end = start + self.profile_dim
                features[start:end] = get_profile_vector(name)

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
