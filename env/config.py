import numpy as np
import typing as t

from vizdoom.vizdoom import ScreenFormat, GameVariable, Button, Mode
from vizdoom.vizdoom import ScreenResolution

# Mapping between the number of channels used in the input and the corresponding VizDoom screen format.
CHANNELS_TO_VZD = {1: ScreenFormat.GRAY8, 3: ScreenFormat.RGB24}

MODE_TO_VZD = {
    "PLAYER": Mode.PLAYER,
    "ASYNC_PLAYER": Mode.ASYNC_PLAYER,
    "SPECTATOR": Mode.SPECTATOR,
    "ASYNC_SPECTATOR": Mode.ASYNC_SPECTATOR,
}

# Mapping between tuple of (width, height) and the corresponding VizDoom screen resolution.
RESOLUTION_TO_VZD = {(320, 240): ScreenResolution.RES_320X240}

CHANNELS = 3
CROP_TOP = 40
OBS_SHAPE = (CHANNELS, 100, 160)
SCREEN_SHAPE = (CHANNELS, 240, 320)

AMMO_VARIABLES = [
    GameVariable.AMMO0, GameVariable.AMMO1, GameVariable.AMMO2, GameVariable.AMMO3, GameVariable.AMMO4,
    GameVariable.AMMO5, GameVariable.AMMO6, GameVariable.AMMO7, GameVariable.AMMO8, GameVariable.AMMO9
]

WEAPON_VARIABLES = [
    GameVariable.WEAPON0, GameVariable.WEAPON1, GameVariable.WEAPON2, GameVariable.WEAPON3, GameVariable.WEAPON4,
    GameVariable.WEAPON5, GameVariable.WEAPON6, GameVariable.WEAPON7, GameVariable.WEAPON8, GameVariable.WEAPON9
]


class EnvironmentConfig:
    """Class holding environment configuration information"""

    def __init__(self, params: t.Dict):
        self.scenario = params['scenario']

        # Doom env parameters
        self.n_envs = params['n_parallel']
        self.frame_stack = params['frame_stack']
        self.frame_skip = params['frame_skip']
        self.env_type = params['type']
        self.env_args = params['args']

        # Action space parameters
        self.action_combination = params['action_combination']
        self.action_noop = params['action_noop']

        # Observation space parameters
        self.raw_channels = params['obs_channels']
        self.raw_width = params['obs_width']
        self.raw_height = params['obs_height']
        self.crop = np.array(params['obs_crop'])
        self.resize = np.array(params['obs_resize'])

        self.game_mode = MODE_TO_VZD[params['vizdoom_mode']]
        self.screen_mode = CHANNELS_TO_VZD[self.raw_channels]
        self.screen_resolution = RESOLUTION_TO_VZD[(self.raw_width, self.raw_height)]

    def get_log_name(self):
        return '{}/nenvs={}_stack={}_skip={}'.format(
            self.scenario,
            self.n_envs,
            self.frame_stack,
            self.frame_skip,
        )

    def get_input_shape(self):
        width = self.resize[0] * (self.raw_width - sum(self.crop[[1, 3]]))
        height = self.resize[1] * (self.raw_height - sum(self.crop[[0, 2]]))

        return self.raw_channels, int(height), int(width)
