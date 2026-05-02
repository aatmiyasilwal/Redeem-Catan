import gymnasium as gym
from catanatron_gym.envs.catanatron_env import ACTIONS_ARRAY, normalize_action
from catanatron.models.enums import ActionType


class AxelrodWrapper(gym.Wrapper):
    """
    Axelrod 'Tit-for-Tat' Logic Wrapper.
    Intercepts RL actions and specifically overrides MOVE_ROBBER target selection
    to heavily penalise/target the player who has targeted P0 the most throughout the game.
    """

    def step(self, action_int):
        # 1. Decode the intended action representation from the RL model
        action_type, value = ACTIONS_ARRAY[action_int]

        # 2. We only intervene if the agent is about to steal via a robber movement
        if action_type == ActionType.MOVE_ROBBER:
            game = self.env.unwrapped.game # type: ignore
            p0_color = self.env.unwrapped.p0.color # type: ignore

            # Tally up who has targeted P0 using the game log so far
            times_targeted_by = {c: 0 for c in game.state.colors}
            for act in game.state.actions:
                if act.action_type == ActionType.MOVE_ROBBER:
                    # Inside Catanatron, Robber payload: (coordinate, target_enemy_color, is_knight?)
                    if isinstance(act.value, tuple) and len(act.value) >= 2:
                        target_color = act.value[1]
                        if target_color == p0_color:
                            # We were targeted by the player who took this action
                            times_targeted_by[act.color] += 1

            # The gym environment's 'from_action_space' maps our integer action to a concrete
            # game action by taking the *first* matching Action in 'playable_actions'.
            # To force it to pick our preferred target, we sort 'playable_actions' stably.
            def sort_key(catan_action):
                # Is this one of the steal actions matching the chosen hex coordinate?
                normalized = normalize_action(catan_action)
                if normalized.action_type == action_type and normalized.value == value:
                    if isinstance(catan_action.value, tuple) and len(catan_action.value) >= 2:
                        target_color = catan_action.value[1]
                        if target_color is not None:
                            # A higher score -> more negative value -> shoots to index 0
                            return -times_targeted_by.get(target_color, 0)

                # Any other action receives a neutral priority
                return 0

            # Mutate state inline right before stepping
            game.state.playable_actions.sort(key=sort_key)

        return super().step(action_int)
