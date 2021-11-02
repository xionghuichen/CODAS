from gym.envs.registration import registry, register, make, spec

register(
    id='Walker2d-v4',
    max_episode_steps=1000,
    entry_point='codas.reset_able_mj_env.walker2d_v4:Walker2dEnv',
)

register(
    id='Hopper-v4',
    max_episode_steps=1000,
    entry_point='codas.reset_able_mj_env.hopper_v4:HopperEnv',
)

register(
    id='Swimmer-v4',
    max_episode_steps=1000,
    entry_point='codas.reset_able_mj_env.swimmer_v4:SwimmerEnv',
)

register(
    id='Swimmer-v5',
    max_episode_steps=1000,
    entry_point='codas.reset_able_mj_env.swimmer_v5:SwimmerEnv',
)


register(
    id='InvertedDouble-v4',
    max_episode_steps=1000,
    entry_point='codas.reset_able_mj_env.inverted_double_pendulum_v4:InvertedDoublePendulumEnv',
)
register(
    id='InvertedDouble-v5',
    max_episode_steps=1000,
    entry_point='codas.reset_able_mj_env.inverted_double_pendulum_v5:InvertedDoublePendulumEnv',
)