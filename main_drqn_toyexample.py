from environments import GraphWorld
import numpy as np
import simdata_utils as su
import matplotlib.pyplot as plt
import torch
from dqn_utils import seed_everything, SeqReplayMemory, train, run_episodes
from rl_utils import EvaluatePolicy, CreateDuplicatesTrainsets
from rl_models import RecurrentQNetwork
from rl_policy import EpsilonGreedyPolicyRDQN
import time
from copy import deepcopy
#import os

#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device = ',device)

def plot_traindata(episode_returns,losses):
    plt.plot(episode_returns)
    plt.savefig('testplots_returns_curve.png')
    plt.clf()
    plt.plot(losses)
    plt.savefig('testplots_loss_curve.png')
    plt.clf()

# Select graph world
configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
conf=configs['Manhattan3']
conf['direction_north']=False
fixed_init=conf['fixed_initial_positions']
conf['make_reflexive']=True

# Define qnet 
dims_hidden_layers = [128]          # [128]
lstm_hidden_size = 18               # 18
# Select hyperparameters
seed = 42                           # 42            41             
batch_size      = 64                # 64                
mem_size        = 300               # 300           230                
discount_factor = .9                # .9
learn_rate      = 5e-6              # 5e-6          3e-6
num_episodes    = 3500              # 3500          2500
eps_0           = 1.                # 1.0
eps_min         = 0.1               # 0.1
cutoff          = 0.8*num_episodes  # 0.8
state_noise     = False             # Cut-off = lower plateau reached and maintained from this point onward

# Initialize
seed_everything(seed)
env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=fixed_init, state_representation='et', state_encoding='tensor')
# env.register['coords']={((1,0),(0,2),(1,2),(2,2)):0}
# env.register['labels']={(1,6,7,8):0}
# env.databank['coords']=[{'start_escape_route':(1,0), 'start_units':[(0,2),(1,2),(2,2)], 'paths':[[(0,2),(0,1),(0,0),(0,1),(0,0)],[(1,2),(1,2),(1,2),(0,2),(0,1)],[(2,2),(2,1)]]}]
# env.databank['labels']=[{'start_escape_route':1, 'start_units':[6,7,8], 'paths':[[6,3,0,3,0],[7,7,7,6,3],[8,5]]}]
# env.iratios=[1.]
# env.all_worlds=[0]
# env.world_pool=[0]
env.register['coords']={((1,0),(0,2),(1,2),(2,2)):0}
env.register['labels']={(1,6,7,8):0}
env.databank['coords']=[{'start_escape_route':(1,0), 'start_units':[(0,2),(1,2),(2,2)], 'paths':[[(0,2),(0,1)],[(1,2),(1,2),(1,2),(0,2)],[(2,2),(2,1)]]}]
env.databank['labels']=[{'start_escape_route':1, 'start_units':[6,7,8], 'paths':[[6,3],[7,7,7,6],[8,5]]}]
env.iratios=[1.]
env.all_worlds=[0]
env.world_pool=[0]

dim_in = env.state_encoding_dim
dim_out = env.max_outdegree
memory = SeqReplayMemory(mem_size)
qnet = RecurrentQNetwork(dim_in, lstm_hidden_size, dim_out, dims_hidden_layers).to(device)
policy = EpsilonGreedyPolicyRDQN(qnet, env, eps_0=eps_0, eps_min=eps_min, eps_cutoff=cutoff)

# Run DQN
start_time = time.time()
episode_durations, episode_returns, losses, best_model_path, best_model = run_episodes(train, qnet, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate, print_every=100, noise=state_noise)
duration = time.time() - start_time
print('run time in seconds: ', duration)
plot_traindata(episode_returns,losses)

qnet_best = best_model#qnet
if best_model_path is not None:
    qnet_best.load_state_dict(torch.load(best_model_path))
policy.Q = qnet_best
print('evaluation of learned policy on trainset')
policy.epsilon=0.
EvaluatePolicy(env, policy, env.world_pool, print_runs=False, save_plots=True)

