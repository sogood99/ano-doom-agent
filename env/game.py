import time
import random
from vizdoom import *
import numpy as np
from stable_baselines.common.vec_env import DummyVecEnv
from .env import DoomEnv, DoomWithBots
from .game_actions import get_available_actions
from .config import EnvironmentConfig


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


def create_env(scenario, reward_type, env_param) -> DoomEnv:
    game, possible_actions = init_game(scenario)
    return DoomWithBots(game, possible_actions, reward_type, env_param)


def create_vec_env(scenario, reward_type, env_param, n_envs=1) -> DummyVecEnv:
    return DummyVecEnv([lambda: create_env(scenario, reward_type, env_param)] * n_envs)
