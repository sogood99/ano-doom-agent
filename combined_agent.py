from stable_baselines import ppo2
from stable_baselines.common import callbacks, policies, schedules
from env.game import *
from env.config import EnvironmentConfig
import json

DEATHMATCH_CFG = "defend_the_center"
NAV_AGENT = "navigate_agent"
BAT_AGENT = "battle_agent"

with open("./env/deathmatch.json") as f:
    env_param_json = json.load(f)
DEATHMATCH_ENV = EnvironmentConfig(env_param_json)


def create_combined_agent(timesteps, nav_agent, bat_agent, show_window=False) -> ppo2.PPO2:
    """
        Create a PPO agent with naive rewards and total training timestep = timestep
    """
    trainEnv = create_vec_env_combined(scenario=DEATHMATCH_CFG, env_param=DEATHMATCH_ENV,
                                       show_window=show_window, nav_agent=nav_agent, bat_agent=bat_agent, n_envs=4)
    testEnv = create_env_combined(scenario=DEATHMATCH_CFG, env_param=DEATHMATCH_ENV, nav_agent=nav_agent,
                                  bat_agent=bat_agent, show_window=show_window)

    # linearLR = schedules.LinearSchedule(TIMESTEPS, 1e-1, 1e-4)
    constLR = schedules.ConstantSchedule(3e-4)

    agent = ppo2.PPO2(policy=policies.CnnLstmPolicy, env=trainEnv, nminibatches=4, learning_rate=constLR.value,
                      tensorboard_log="logs/tensorboard")

    evaluation_callback = callbacks.EvalCallback(testEnv, best_model_save_path='logs/models/combined_agent')

    agent.learn(total_timesteps=timesteps, tb_log_name="combined_agent", callback=evaluation_callback)
    trainEnv.close()
    testEnv.close()
    return agent


if __name__ == "__main__":
    TIMESTEPS = 500000
    nav_agent, bat_agent = ppo2.PPO2.load("logs/models/{}/best_model.zip".format(NAV_AGENT)), ppo2.PPO2.load(
        "logs/models/{}/best_model.zip".format(BAT_AGENT))
    create_combined_agent(TIMESTEPS, nav_agent, bat_agent)
