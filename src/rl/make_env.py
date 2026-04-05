import gymnasium as gym
import catanatron_gym
import numpy as np
from profiles import get_profile_vector, get_profile_dim


class OpponentProfileWrapper(gym.ObservationWrapper):
    """
    Injects pre-computed opponent profiles into the RL agent's observation space.
    This lets the agent know *who* it is playing against.
    """

    def __init__(self, env, opponent_names):
        super().__init__(env)
        self.opponent_names = opponent_names

        # Load the feature vectors for the opponents using the API from profiles.py
        vectors = [get_profile_vector(name) for name in opponent_names]
        self.static_features = np.concatenate(vectors).astype(np.float32)

        # Expand the observation space dimension
        base_shape = env.observation_space.shape[0]
        new_dim = base_shape + len(self.static_features)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(new_dim,),
            dtype=np.float32
        )

    def observation(self, obs):
        # Concatenate the dynamic base observation with static opponent features
        return np.concatenate([obs, self.static_features])


def make_env(opponent_names):
    """
    Creates the Catanatron gym environment and wraps it with static opponent profiles.
    """
    # The environment ID is catanatron-v1, namespaced under catanatron_gym
    env = gym.make("catanatron_gym:catanatron-v1")
    env = OpponentProfileWrapper(env, opponent_names)
    return env
