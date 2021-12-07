from environments import GraphWorld
import numpy as np
import simdata_utils as su
import matplotlib.pyplot as plt
import torch
from dqn_utils import seed_everything, QNetwork, FastReplayMemory, EpsilonGreedyPolicyDQN, train, run_episodes
from rl_utils import EvaluatePolicy, SelectTrainset
from rl_policy import MinIndegreePolicy
import random
import time
import os

#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# Select comp device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('Device = ',device)

# Select graph world
configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
conf=configs['Manhattan5']
#conf=configs['Manhattan11']
#conf=configs['CircGraph']
conf['direction_north']=False
fixed_init=conf['fixed_initial_positions']

# Define qnet 
dims_hidden_layers = [256, 128]

env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=fixed_init, state_representation='etUte0U0', state_encoding='tensor')
env.world_pool = env.all_worlds

dim_in = env.state_encoding_dim
dim_out = env.max_outdegree
qnet = QNetwork(dim_in, dim_out, dims_hidden_layers).to(device)
qnet.load_state_dict(torch.load("models/dqn_[256,128]_best_model_1.94.pt"))

policy = EpsilonGreedyPolicyDQN(qnet, env)
policy.epsilon=0.
EvaluatePolicy(env, policy, random.sample(env.world_pool,10), print_runs=False, save_plots=True)


