import time
import random
from vizdoom import *
import numpy as np
from stable_baselines.common.vec_env import DummyVecEnv
from .env import DoomEnv, DoomWithBots, DoomNavigateBattle
from .game_actions import get_available_actions


def init_game(scenario, show_window=False):
    game = DoomGame()
    game.load_config(f'./scenarios/{scenario}.cfg')
    game.set_window_visible(show_window)
    game.init()
    possible_actions = get_available_actions(game.get_available_buttons())
    return game, possible_actions


def create_env(scenario, reward_type, env_param, show_window=False) -> DoomEnv:
    game, possible_actions = init_game(scenario, show_window)
    return DoomWithBots(game, possible_actions, reward_type, env_param)


def create_vec_env(scenario, reward_type, env_param, show_window=False, n_envs=1) -> DummyVecEnv:
    return DummyVecEnv([lambda: create_env(scenario, reward_type, env_param, show_window)] * n_envs)


def create_env_combined(scenario, nav_agent, bat_agent, env_param, show_window=False) -> DoomEnv:
    game, possible_actions = init_game(scenario, show_window)
    return DoomNavigateBattle(game, possible_actions, nav_agent, bat_agent, env_param)


def create_vec_env_combined(scenario, nav_agent, bat_agent, env_param, show_window=False,
                            n_envs=1) -> DummyVecEnv:
    return DummyVecEnv(
        [lambda: create_env_combined(scenario, nav_agent, bat_agent, env_param, show_window)] * n_envs)
