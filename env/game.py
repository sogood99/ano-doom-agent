import time
import random
from utils import *
import numpy as np


def initGame(scenario):
    game = DoomGame()
    game.load_config(f'../scenarios/{scenario}.cfg')
    game.set_window_visible(True)
    game.init()
    return game


if __name__ == "__main__":
    game = initGame("deathmatch")
    episodes = 10
    for i in range(episodes):
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            img = state.screen_buffer
            misc = state.game_variables
            reward = game.make_action(np.eye(len(deathmatch_all_actions))[-1])
            print("\treward:", reward)
            time.sleep(0.02)
        time.sleep(2)
    game.close()
