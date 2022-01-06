# ano-doom-agent

A Doom game playing Deep Reinforcement Learning agent. Combines battle agent and navigator agent using a combining agent all implemented using PPO with CnnLstm Policy.

## Model Framework

<img src="./fig/framework.png" alt="Framework" width="650"/>

Tested on Defend The Center Map with kill count and survival time as reward metrics.

## Battle Agent

![Battle Agent](./fig/battle_agent.gif)

## Navigator Agent

![Navigator Agent](./fig/nav_agent.gif)

## Combined Agent

![Combined Agent](./fig/combined_agent.gif)

## Training Results

![Episode Reward](./fig/episode_reward.png)

Grey is final result, Blue is naive agent, Red is combined agent with entropy reward (entropy reward didn't work as expected).
