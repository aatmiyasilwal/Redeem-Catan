import json
from pathlib import Path
import numpy as np


class ProfileManager:
    """
    Singleton manager to ensure profiles.npy and player_index.json 
    are only loaded from disk once, keeping environment resets blazing fast.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ProfileManager, cls).__new__(cls)
            cls._instance._load_data()
        return cls._instance

    def _load_data(self):
        # Resolve data directory relative to this script: src/rl/profiles.py -> ../../data
        data_dir = Path(__file__).resolve().parent.parent.parent / "data"

        profiles_path = data_dir / "profiles.npy"
        index_path = data_dir / "player_index.json"

        if not profiles_path.exists() or not index_path.exists():
            raise FileNotFoundError(
                f"Profile data not found. Please ensure both 'profiles.npy' and "
                f"'player_index.json' exist in the {data_dir} directory."
            )

        # 1. Load the core numpy data into memory
        self.profiles_array = np.load(profiles_path)

        # 2. Load the player dictionary mapping
        with open(index_path, "r") as f:
            self.player_map = json.load(f)

        # 3. Precompute the median profile (column-wise median) to serve as a sensible
        # fallback for players without historical data.
        self.median_profile = np.median(self.profiles_array, axis=0)

        # Helpful attribute for your gym wrapper to dynamically get the shape
        self.feature_dim = self.profiles_array.shape[1]

    def get_vector(self, player_name: str) -> np.ndarray:
        """
        Returns the standard feature array for the given player_name.
        If the player is unknown, returns the precomputed median profile.
        """
        idx = self.player_map.get(player_name)
        if idx is not None:
            return self.profiles_array[idx]
        else:
            return self.median_profile



# public API
def get_profile_vector(player_name: str) -> np.ndarray:
    """
    Fetches the profile representation of a player.
    Loads data into memory automatically on the first call.
    """
    manager = ProfileManager()
    return manager.get_vector(player_name)


def get_profile_dim() -> int:
    """Returns the size of the feature vector."""
    manager = ProfileManager()
    return manager.feature_dim
