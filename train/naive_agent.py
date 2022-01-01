from stable_baselines import ppo2
from stable_baselines.common import callbacks, policies, schedules
from env.game import *
from env.config import EnvironmentConfig
import json

DEATHMATCH_CFG = "deathmatch"

with open("../env/deathmatch.json") as f:
    env_param_json = json.load(f)
DEATHMATCH_ENV = EnvironmentConfig(env_param_json)


def create_naive_agent() -> ppo2.PPO2:
    trainEnv = create_vec_env(scenario=DEATHMATCH_CFG, env_param=DEATHMATCH_ENV, reward_type="naive", n_envs=4)
    testEnv = create_vec_env(scenario=DEATHMATCH_CFG, env_param=DEATHMATCH_ENV, reward_type="naive")

    TIMESTEPS = 10000
    # linearLR = schedules.LinearSchedule(TIMESTEPS, 1e-1, 1e-4)
    constLR = schedules.ConstantSchedule(3e-4)

    agent = ppo2.PPO2(policy=policies.CnnLstmPolicy, env=trainEnv, nminibatches=4, learning_rate=constLR.value,
                      tensorboard_log="logs/tensorboard")

    evaluation_callback = callbacks.EvalCallback(testEnv,
                                                 n_eval_episodes=15,
                                                 eval_freq=2500,
                                                 best_model_save_path='logs/models/baseline_naive')

    agent.learn(total_timesteps=TIMESTEPS, tb_log_name="baseline_naive", callback=evaluation_callback)
    trainEnv.close()
    testEnv.close()
    return agent


def train_existing_naive_agent(path) -> ppo2:
    trainEnv = create_vec_env(scenario=DEATHMATCH_CFG, env_param=DEATHMATCH_ENV, reward_type="naive", n_envs=4)
    testEnv = create_vec_env(scenario=DEATHMATCH_CFG, env_param=DEATHMATCH_ENV, reward_type="naive")

    TIMESTEPS = 10000
    # linearLR = schedules.LinearSchedule(TIMESTEPS, 1e-1, 1e-4)
    # constLR = schedules.ConstantSchedule(3e-4)

    agent = ppo2.PPO2.load(path, env=trainEnv, tensorboard_log="logs/tensorboard")

    evaluation_callback = callbacks.EvalCallback(testEnv,
                                                 n_eval_episodes=15,
                                                 eval_freq=2500,
                                                 best_model_save_path='logs/models/baseline_naive')

    agent.learn(total_timesteps=TIMESTEPS, tb_log_name="baseline_naive", callback=evaluation_callback)
    trainEnv.close()
    testEnv.close()
    return agent


if __name__ == "__main__":
    create_naive_agent()
