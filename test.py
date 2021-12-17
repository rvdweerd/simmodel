# import gym
# from environments import GraphWorld
# import simdata_utils as su
# configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
# conf=configs['Manhattan3']

# env=GraphWorld(conf,optimization_method='static', fixed_initial_positions=None, state_representation='et', state_encoding='tensor')
# s=env.reset()
# a,b,c,d = env.step()

# from spinup import ppo_pytorch as ppo

# from spinup import ppo_tf1 as ppo
# import tensorflow as tf
# import gym

# env_fn = lambda : gym.make('LunarLander-v2')

# ac_kwargs = dict(hidden_sizes=[64,64], activation=tf.nn.relu)

# logger_kwargs = dict(output_dir='path/to/output_dir', exp_name='experiment_name')

# ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=5, logger_kwargs=logger_kwargs)

import gym
from PIL import Image
env = gym.make('LunarLander-v2')
env.reset()
ima = Image.fromarray(env.render(mode='rgb_array'))
print('done')