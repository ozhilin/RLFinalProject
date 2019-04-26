from gym.envs.registration import register

import continuous_cartpole.envs as e

register(
    id='continuous-cartpole-v0',
    entry_point='envs.continuous_cartpole:ContinuousCartPoleEnv',
    max_episode_steps=1000
)

register(
    id='continuous-cartpole-v99',
    entry_point='envs.continuous_cartpole2:ContinuousCartPoleEnv2',
    max_episode_steps=1000
)
