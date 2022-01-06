import imageio
import numpy as np
from env.game import *
from env.config import EnvironmentConfig
import json
from stable_baselines import ppo2

DEATHMATCH_CFG = "defend_the_center"

with open("./env/deathmatch.json") as f:
    env_param_json = json.load(f)
DEATHMATCH_ENV = EnvironmentConfig(env_param_json)

bat_agent = ppo2.PPO2.load("./logs/models/battle_agent/best_model.zip")
nav_agent = ppo2.PPO2.load("./logs/models/navigate_agent/best_model.zip")
model = ppo2.PPO2.load("./logs/models/combined_agent/best_model.zip")
env = create_vec_env_combined(scenario=DEATHMATCH_CFG, env_param=DEATHMATCH_ENV, nav_agent=nav_agent,
                              bat_agent=bat_agent, reward_type="test", n_envs=4)

images = []
obs = env.reset()
state = None
done = [False for _ in range(4)]
img = env.render(mode='rgb_array')
for i in range(2100):
    images.append(img)
    action, state = model.predict(obs, state=state, mask=done)
    obs, reward, done, _ = env.step(action)
    img = env.render(mode='rgb_array')

imageio.mimsave('./fig/combined_agent.gif', [np.array(img) for i, img in enumerate(images) if i % 2 == 0], fps=60)
