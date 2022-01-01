import time
import random
from utils import *
import numpy as np


def initGame(scenario):
    game = DoomGame()
    game.load_config(f'../scenarios/{scenario}.cfg')
    game.set_window_visible(True)
    game.add_game_args('-host 1 -deathmatch +viz_nocheat 0 +cl_run 1 +name ANO +colorset 0' +
                       '+sv_forcerespawn 1 +sv_respawnprotect 1 +sv_nocrouch 1 +sv_noexit 1')
    game.init()
    return game


if __name__ == "__main__":
    game = initGame("deathmatch")
    for i in range(10):
        game.send_game_command("addbot")
    episodes = 10
    for i in range(episodes):
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            img = state.screen_buffer
            misc = state.game_variables
            reward = game.make_action(random.choice(np.eye(len(deathmatch_all_actions))[0:]))
            print("\treward:", reward)
            if game.is_player_dead():
                game.respawn_player()
            time.sleep(0.02)
        time.sleep(2)
    game.close()
