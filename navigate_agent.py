from stable_baselines import ppo2
from stable_baselines.common import callbacks, policies, schedules
from env.game import *
from env.config import EnvironmentConfig
import json

DEATHMATCH_CFG = "deathmatch"

with open("./env/deathmatch.json") as f:
    env_param_json = json.load(f)
DEATHMATCH_ENV = EnvironmentConfig(env_param_json)


def create_navigate_agent(timesteps, show_window=False) -> ppo2.PPO2:
    """
        Create a PPO agent with naive rewards and total training timestep = timestep
    """
    trainEnv = create_vec_env(scenario=DEATHMATCH_CFG, env_param=DEATHMATCH_ENV, reward_type="navigate",
                              show_window=show_window, n_envs=4)
    testEnv = create_vec_env(scenario=DEATHMATCH_CFG, env_param=DEATHMATCH_ENV, reward_type="navigate",
                             show_window=show_window)

    # linearLR = schedules.LinearSchedule(TIMESTEPS, 1e-1, 1e-4)
    constLR = schedules.ConstantSchedule(3e-4)

    agent = ppo2.PPO2(policy=policies.CnnLstmPolicy, env=trainEnv, nminibatches=4, learning_rate=constLR.value,
                      tensorboard_log="logs/tensorboard")

    evaluation_callback = callbacks.EvalCallback(testEnv, best_model_save_path='logs/models/navigate_agent')

    agent.learn(total_timesteps=timesteps, tb_log_name="navigate_agent", callback=evaluation_callback)
    trainEnv.close()
    testEnv.close()
    return agent


def train_existing_navigate_agent(timesteps, path, show_window=False) -> ppo2:
    trainEnv = create_vec_env(scenario=DEATHMATCH_CFG, env_param=DEATHMATCH_ENV, reward_type="navigate",
                              show_window=show_window, n_envs=4)
    testEnv = create_vec_env(scenario=DEATHMATCH_CFG, env_param=DEATHMATCH_ENV, reward_type="navigate",
                             show_window=show_window)

    # linearLR = schedules.LinearSchedule(TIMESTEPS, 1e-1, 1e-4)
    # constLR = schedules.ConstantSchedule(3e-4)

    agent = ppo2.PPO2.load(path, env=trainEnv, tensorboard_log="logs/tensorboard")

    evaluation_callback = callbacks.EvalCallback(testEnv, best_model_save_path='logs/models/navigate_agent')

    agent.learn(total_timesteps=timesteps, tb_log_name="navigate_agent", callback=evaluation_callback)
    trainEnv.close()
    testEnv.close()
    return agent


if __name__ == "__main__":
    TIMESTEPS = 1000000
    create_navigate_agent(TIMESTEPS, show_window=True)
