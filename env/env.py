import gym
import numpy as np
from gym import spaces
from vizdoom import *
import typing as t
import string
import random
from .game_actions import *
from .config import *
from .utils import FramePreprocessor


class DoomEnv(gym.Env):
    """Wrapper environment following OpenAI's gym interface for a Doom game instance."""

    metadata = {'video.frames_per_second': 35}

    def __init__(self, game: DoomGame, possible_actions: ActionList, environment_config: EnvironmentConfig):
        super().__init__()

        self.action_space = spaces.Discrete(len(possible_actions))
        print(environment_config.get_input_shape())
        self.observation_space = spaces.Box(low=0, high=255, shape=environment_config.get_input_shape(), dtype=np.uint8)

        self.game = game
        self.possible_actions = possible_actions
        self.frame_skip = environment_config.frame_skip
        self.frame_preprocessor = FramePreprocessor(scale=environment_config.resize, crop=environment_config.crop)
        self.empty_frame = np.zeros(self.observation_space.shape, dtype=np.uint8)
        self.state = self.empty_frame

    def step(self, action: int) -> t.Tuple[np.array, int, bool, t.Dict[str, int]]:
        """Apply an action to the environment.
        Args:
            action:
        Returns:
            A tuple containing:
                - A numpy ndarray containing the current environment state.
                - The reward obtained by applying the provided action.
                - A boolean flag indicating whether the episode has ended.
                - An empty dict.
        """
        reward = self.game.make_action(self.possible_actions[action], self.frame_skip)
        done = self.game.is_episode_finished()

        self.state = self.game_frame(done)

        return self.state, reward, done, {}

    def reset(self):
        """Resets the environment.
        Returns:
            The initial state of the new environment.
        """
        self.game.new_episode()
        self.state = self.game_frame()

        return self.state

    def render(self, **kwargs):
        return self.game_frame()

    def close(self):
        self.game.close()

    def game_frame(self, done: bool = False):
        return self.frame_preprocessor(self.game.get_state().screen_buffer) if not done else self.empty_frame


class DoomWithBots(DoomEnv):

    def __init__(self, doom_game, possible_actions, reward_type, environment_config: EnvironmentConfig):
        super().__init__(doom_game, possible_actions, environment_config)

        self.name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=2))

        self.total_rew = 0
        self.last_damage_dealt = 0

        self.living_time = 0

        self.last_kills = 0
        self.last_health = 100
        self.last_armor = 0
        self.last_x, self.last_y = self._get_player_pos()
        self.ammo_state = self._get_ammo_state()
        self.weapon_state = self._get_weapon_state()

        self.rewards_stats = {
            'kills': 0,
            'damage': 0,
            'ammo': 0,
            'health': 0,
            'armor': 0,
            'distance': 0,
            'living': 0,
            'death': 0,
        }

        self.tic_rate = environment_config.frame_skip

        # Rewards
        self.reward_type = reward_type
        if reward_type == "battle":
            # 1 per kill
            self.reward_factor_kill = 2
            self.reward_factor_damage = 0.02

            # Player can move at ~16.66 units per tick
            self.reward_factor_distance = 0.
            self.penalty_factor_distance = 0.
            self.reward_threshold_distance = 3.0

            self.reward_factor_ammo_increment = 0.001
            self.reward_factor_ammo_decrement = -0.05

            # Player starts at 100 health
            self.reward_factor_health_increment = 0.005
            self.reward_factor_health_decrement = -0.001
            self.reward_factor_armor_increment = 0.001

            # reward for living/penalty for dying
            self.reward_living = 0.0001

            self.penalty_death = -0.001
        elif reward_type == "navigate":
            self.reward_factor_kill = 0.1
            self.reward_factor_damage = 0.

            # Player can move at ~16.66 units per tick
            self.reward_factor_distance = 0.0008
            self.penalty_factor_distance = 0.0008
            self.reward_threshold_distance = 0.75

            self.reward_factor_ammo_increment = 0.02
            self.reward_factor_ammo_decrement = -0.05

            self.reward_factor_health_increment = 0.02
            self.reward_factor_health_decrement = -0.02
            self.reward_factor_armor_increment = 0.005

            # Reward for living
            # use (living_time/total_time)**2 to calculate reward_living
            self.reward_living = 60 / 700
            self.penalty_death = -10.0

        elif reward_type == "naive":
            # 1 per kill
            self.reward_factor_kill = 1.0
            self.reward_factor_damage = 0.

            self.reward_factor_distance = 0.
            self.penalty_factor_distance = 0.
            self.reward_threshold_distance = 0.

            # Pistol clips have 10 bullets
            self.reward_factor_ammo_increment = 0.
            self.reward_factor_ammo_decrement = 0.

            # Player starts at 100 health
            self.reward_factor_health_increment = 0.
            self.reward_factor_health_decrement = 0.
            self.reward_factor_armor_increment = 0.

            self.reward_living = 30 / 2100.
            self.penalty_death = -1.
        else:
            raise Exception("Unknown reward type")

        print(f'Logging with ID {self.name}')

    def shape_rewards(self, initial_reward: float):
        kill_reward = self._compute_kill_reward()
        damage_reward = self._compute_damage_reward()
        ammo_reward = self._compute_ammo_reward()
        health_reward = self._compute_health_reward()
        armor_reward = self._compute_armor_reward()
        distance_reward = self._compute_distance_reward(*self._get_player_pos())
        ld_reward = self._compute_life_death_reward()

        return initial_reward + kill_reward + damage_reward + ammo_reward + health_reward + armor_reward + distance_reward + ld_reward

    def _compute_distance_reward(self, x, y):
        dx = self.last_x - x
        dy = self.last_y - y

        self.last_x = x
        self.last_y = y

        distance = np.sqrt(dx ** 2 + dy ** 2)
        d = distance - self.reward_threshold_distance

        if d > 0:
            reward = self.reward_factor_distance * d
        else:
            reward = self.penalty_factor_distance * d

        self._log_reward_stat('distance', reward)

        return reward

    def _compute_kill_reward(self):
        kills = self.game.get_game_variable(GameVariable.KILLCOUNT)
        reward = self.reward_factor_kill * (kills - self.last_kills)

        self.last_kills = kills
        self._log_reward_stat('kills', reward)

        return reward

    def _compute_damage_reward(self):
        damage_dealt = self.game.get_game_variable(GameVariable.DAMAGECOUNT)
        reward = self.reward_factor_damage * (damage_dealt - self.last_damage_dealt)

        self.last_damage_dealt = damage_dealt
        self._log_reward_stat('damage', reward)

        return reward

    def _compute_health_reward(self):
        # When player is dead, the health game variable can be -999900
        health = max(self.game.get_game_variable(GameVariable.HEALTH), 0)

        health_reward = self.reward_factor_health_increment * max(0, health - self.last_health)
        health_penalty = self.reward_factor_health_decrement * min(0, health - self.last_health)
        reward = health_reward - health_penalty

        self.last_health = health
        self._log_reward_stat('health', reward)

        return reward

    def _compute_armor_reward(self):
        armor = self.game.get_game_variable(GameVariable.ARMOR)
        reward = self.reward_factor_armor_increment * max(0, armor - self.last_armor)
        self.last_armor = armor
        self._log_reward_stat('armor', reward)

        return reward

    def _compute_ammo_reward(self):
        self.weapon_state = self._get_weapon_state()

        new_ammo_state = self._get_ammo_state()
        ammo_diffs = (new_ammo_state - self.ammo_state) * self.weapon_state
        ammo_reward = self.reward_factor_ammo_increment * max(0, np.sum(ammo_diffs))
        ammo_penalty = self.reward_factor_ammo_decrement * min(0, np.sum(ammo_diffs))
        reward = ammo_reward - ammo_penalty
        self.ammo_state = new_ammo_state
        self._log_reward_stat('ammo', reward)

        return reward

    def _compute_life_death_reward(self):
        death_penalty = self.penalty_death if self.game.is_player_dead() else 0
        living_reward = self.reward_living if self.reward_type == "naive" else self.reward_living * (
                self.living_time / 2100.) ** 2

        self._log_reward_stat('death', death_penalty)
        self._log_reward_stat('living', living_reward)

        reward = death_penalty + living_reward
        return reward

    def _get_player_pos(self):
        return self.game.get_game_variable(GameVariable.POSITION_X), self.game.get_game_variable(
            GameVariable.POSITION_Y)

    def _get_ammo_state(self):
        ammo_state = np.zeros(10)

        for i in range(10):
            ammo_state[i] = self.game.get_game_variable(AMMO_VARIABLES[i])

        return ammo_state

    def _get_weapon_state(self):
        weapon_state = np.zeros(10)

        for i in range(10):
            weapon_state[i] = self.game.get_game_variable(WEAPON_VARIABLES[i])

        return weapon_state

    def _log_reward_stat(self, kind: str, reward: float):
        self.rewards_stats[kind] += reward

    def _reset_player(self):
        self.last_health = 100
        self.last_armor = 0
        self.living_time = 0
        self.last_x, self.last_y = self._get_player_pos()
        self.ammo_state = self._get_ammo_state()

    def step(self, action, array=False):
        # Apply action
        _ = self.game.make_action(self.possible_actions[action] if not array else action, self.frame_skip)

        self.living_time += 1

        reward = self.shape_rewards(initial_reward=0)

        done = self.game.is_episode_finished()
        self.state = self.game_frame(done)
        self.total_rew += reward

        return self.state, reward, done, {'kills': self.last_kills}

    def reset(self):
        self._print_state()

        state = super().reset()

        self.last_x, self.last_y = self._get_player_pos()
        self.last_armor = 0
        self.last_health = 100
        self.living_time = 0
        self.last_kills = 0
        self.last_damage_dealt = 0
        self.total_rew = 0

        # Reset reward stats
        for k in self.rewards_stats.keys():
            self.rewards_stats[k] = 0

        return state

    def _print_state(self):

        print('************************')
        print('Agent {} kills: {}, alive time: {}, total reward: {}'.format(
            self.name,
            self.last_kills,
            self.living_time,
            self.total_rew
        ))
        for k, v in self.rewards_stats.items():
            print(f'- {k}: {v:+.3f}')
        print('************************')
