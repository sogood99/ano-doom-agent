import time
import random
from vizdoom import *
import numpy as np
from stable_baselines.common.vec_env import DummyVecEnv
from env import DoomEnv, DoomWithBots
from game_actions import get_available_actions


def init_game(scenario, show_window=False):
    game = DoomGame()
    game.set_window_visible(show_window)
    game.load_config(f'../scenarios/{scenario}.cfg')
    game.set_window_visible(True)
    game.add_game_args('-host 1 -deathmatch +viz_nocheat 0 +cl_run 1 +name ANO +colorset 0' +
                       '+sv_forcerespawn 1 +sv_respawnprotect 1 +sv_nocrouch 1 +sv_noexit 1')
    game.init()

    possible_actions = get_available_actions(game.get_available_buttons())

    return game, possible_actions


def env_with_bots_shaped(scenario, **kwargs) -> DoomEnv:
    """Wraps a Doom game instance in an environment with shaped rewards."""
    game = init_game(scenario)
    return DoomWithBots(game, **kwargs)


def vec_env_with_bots_shaped(n_envs=1, **kwargs) -> DummyVecEnv:
    """Wraps a Doom game instance in a vectorized environment with shaped rewards."""
    return DummyVecEnv([lambda: env_with_bots_shaped(**kwargs)] * n_envs)


if __name__ == "__main__":
    game, possible_actions = init_game("deathmatch", True)
    for i in range(6):
        game.send_game_command("addbot")
    episodes = 1
    for i in range(episodes):
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            img = state.screen_buffer
            misc = state.game_variables
            reward = game.make_action(random.choice(possible_actions))
            print("\treward:", reward)
            if game.is_player_dead():
                game.respawn_player()
            time.sleep(0.02)
        time.sleep(2)
    game.close()
