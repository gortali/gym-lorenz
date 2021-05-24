from gym.envs.registration import register

register(
    id='lorenz-v0',
    entry_point='gym_lorenz.envs:LorenzEnv',
)
#register(
#    id='lorenz-extrahard-v0',
#    entry_point='gym_lorenz.envs:LorenzExtraHardEnv',
#)
