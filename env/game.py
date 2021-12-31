from vizdoom import *
from vizdoom.vizdoom import GameVariable
import matplotlib.pyplot as plt


def initGame(scenario):
    game = DoomGame()
    game.load_config(f'../scenarios/{scenario}.cfg')
    game.init()
    return game


if __name__ == "__main__":
    game = initGame("deathmatch")
    print(game.get_available_game_variables())
    plt.imshow(game.get_state().screen_buffer)
    plt.show()
    game.close()
