import numpy as np
from stable_baselines import ppo2
from stable_baselines.common import callbacks, policies, schedules
from env.game import *
from env.config import EnvironmentConfig
import matplotlib.pyplot as plt
import json

DEATHMATCH_CFG = "defend_the_center"
NAV_AGENT = "navigate_agent"
BAT_AGENT = "battle_agent"
COMB_AGENT = "combined_agent"

with open("./env/deathmatch.json") as f:
    env_param_json = json.load(f)
DEATHMATCH_ENV = EnvironmentConfig(env_param_json)


def create_combined_agent(timesteps, nav_agent, bat_agent, show_window=False) -> ppo2.PPO2:
    """
        Create a PPO agent with naive rewards and total training timestep = timestep
    """
    trainEnv = create_vec_env_combined(scenario=DEATHMATCH_CFG, env_param=DEATHMATCH_ENV,
                                       show_window=show_window, nav_agent=nav_agent, bat_agent=bat_agent, n_envs=4,
                                       reward_type="train")
    testEnv = create_env_combined(scenario=DEATHMATCH_CFG, env_param=DEATHMATCH_ENV, nav_agent=nav_agent,
                                  bat_agent=bat_agent, show_window=show_window, reward_type="train")

    # linearLR = schedules.LinearSchedule(TIMESTEPS, 1e-1, 1e-4)
    constLR = schedules.ConstantSchedule(3e-4)

    agent = ppo2.PPO2(policy=policies.CnnLstmPolicy, env=trainEnv, nminibatches=4, learning_rate=constLR.value,
                      tensorboard_log="logs/tensorboard")

    evaluation_callback = callbacks.EvalCallback(testEnv, best_model_save_path='logs/models/combined_agent_entropy')

    agent.learn(total_timesteps=timesteps, tb_log_name="combined_agent_entropy", callback=evaluation_callback)
    trainEnv.close()
    testEnv.close()
    return agent


def test_combined_agent(timesteps, nav_agent, bat_agent, combined_agent, show_window=False):
    """
        Create a PPO agent with naive rewards and total training timestep = timestep
    """
    testEnv = create_vec_env_combined(scenario=DEATHMATCH_CFG, env_param=DEATHMATCH_ENV, nav_agent=nav_agent,
                                      bat_agent=bat_agent, show_window=show_window, n_envs=4, reward_type="test")

    obs = testEnv.reset()

    state = None
    done = [False for _ in range(4)]
    current_step = [0 for _ in range(4)]

    bat_count = np.zeros(2100)
    nav_count = np.zeros(2100)

    for _ in range(timesteps):
        action, state = combined_agent.predict(obs, state=state, mask=done)
        obs, reward, done, info = testEnv.step(action)
        for i in range(4):
            if action[i] == 0:
                nav_count[current_step[i]] += 1
            else:
                bat_count[current_step[i]] += 1
            if done[i]:
                current_step[i] = 0
            else:
                current_step[i] += 1
    testEnv.close()
    return bat_count / (bat_count + nav_count)


if __name__ == "__main__":
    TIMESTEPS = 500000

    nav_agent, bat_agent = ppo2.PPO2.load(
        "logs/models/{}/best_model.zip".format(NAV_AGENT)), ppo2.PPO2.load(
        "logs/models/{}/best_model.zip".format(BAT_AGENT))
    create_combined_agent(TIMESTEPS, nav_agent, bat_agent)

    # comb_agent = ppo2.PPO2.load("logs/models/{}/best_model.zip".format(COMB_AGENT))
    #
    # percentage = test_combined_agent(TIMESTEPS, nav_agent, bat_agent, comb_agent)
    # plt.plot(percentage)
    # plt.show()
