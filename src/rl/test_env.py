import gymnasium as gym
import catanatron_gym
from make_env import make_env

# Pass opponent names matching your JSON index (3 opponents for a 4-player game)
opponents = ["AatNeverLose", "HomeofAD3005", "ZL24"]

# Use the factory function directly to get the wrapped environment
env = make_env(opponents)

# Test that it resets and prints the shape correctly
observation, info = env.reset(seed=100)
print("Wrapped Environment loaded successfully!")
print("Wrapped Observation shape:", observation.shape)

env.close()
