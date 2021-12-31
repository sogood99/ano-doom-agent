# Modified from https://github.com/lkiel/rl-doom, many thanks!
import itertools
import typing as t
import numpy as np

from vizdoom import Button


MUTUALLY_EXCLUSIVE_GROUPS = [
    [Button.MOVE_RIGHT, Button.MOVE_LEFT],
    [Button.TURN_RIGHT, Button.TURN_LEFT],
    [Button.MOVE_FORWARD, Button.MOVE_BACKWARD],
    [Button.SPEED, Button.TURN_RIGHT],
    [Button.SPEED, Button.TURN_LEFT],
]

EXCLUSIVE_BUTTONS = [Button.ATTACK]

SELF_EXCLUSIVE = [Button.SPEED]


def has_exclusive_button(actions: np.ndarray, buttons: np.array) -> np.array:
    exclusion_mask = np.isin(buttons, EXCLUSIVE_BUTTONS)

    return (np.any(actions.astype(bool) & exclusion_mask, axis=-1)) & (np.sum(actions, axis=-1) > 1)


def has_excluded_pair(actions: np.ndarray, buttons: np.array) -> np.array:
    mutual_exclusion_mask = np.array(
        [np.isin(buttons, excluded_group)for excluded_group in MUTUALLY_EXCLUSIVE_GROUPS])

    return np.any(np.sum(
        (actions[:, np.newaxis, :] * mutual_exclusion_mask.astype(int)),
        axis=-1) > 1, axis=-1)


def has_self_exclusion(actions: np.ndarray, buttons: np.array) -> np.array:
    self_exclusion_mask = np.isin(buttons, SELF_EXCLUSIVE)

    return (np.any(actions.astype(bool) & self_exclusion_mask, axis=-1)) & (np.sum(actions, axis=-1) == 1)


def get_available_actions(buttons: np.array) -> t.List[t.List[float]]:
    action_combinations = np.array(
        [list(seq) for seq in itertools.product([0., 1.], repeat=len(buttons))])

    illegal_mask = (has_excluded_pair(action_combinations, buttons)
                    | has_exclusive_button(action_combinations, buttons) | has_self_exclusion(action_combinations, buttons))

    possible_actions = action_combinations[~illegal_mask]
    possible_actions = possible_actions[np.sum(possible_actions, axis=1) > 0]

    print('Built action space of size {} from buttons {}'.format(
        len(possible_actions), buttons))
    return possible_actions.tolist()


possible_actions = get_available_actions(np.array(
    [Button.ATTACK, Button.MOVE_FORWARD, Button.MOVE_LEFT, Button.MOVE_RIGHT, Button.TURN_LEFT, Button.TURN_RIGHT]))
