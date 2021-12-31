from environments import GraphWorld
import numpy as np
import simdata_utils as su
import matplotlib.pyplot as plt
import torch
from dqn_utils import seed_everything, SeqReplayMemory, train, run_episodes
from rl_utils import EvaluatePolicy, CreateDuplicatesTrainsets
from rl_models import RecurrentQNetwork
from rl_policy import EpsilonGreedyPolicyDRQN
from rl_custom_worlds import GetCustomWorld
from rl_plotting import plot_traindata
import time
from copy import deepcopy

#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device = ',device)

# Select graph world
env=GetCustomWorld('Manhattan3x3_PauseFreezeWorld',make_reflexive=True, state_repr='et' ,state_enc='tensors')


# Define qnet                       # PauseFreezeWorld best: 
dims_hidden_layers = [128]          # [128]
lstm_hidden_size = 18               # 18
# Select hyperparameters
seed = 421                           # 42            41             
batch_size      = 64                # 64                
mem_size        = 300                # 300           230                
discount_factor = .9                # .9
learn_rate      = 5e-6             # 5e-6          3e-6
num_episodes    = 3500              # 3500          2500
eps_0           = 1.                # 1.0
eps_min         = 0.1               # 0.1
cutoff          = 0.8*num_episodes  # 0.8
state_noise     = False             # False

# Initialize
seed_everything(seed)
dim_in = env.state_encoding_dim
dim_out = env.max_outdegree
memory = SeqReplayMemory(mem_size)
qnet = RecurrentQNetwork(dim_in, lstm_hidden_size, dim_out, dims_hidden_layers).to(device)
policy = EpsilonGreedyPolicyDRQN(qnet, env, eps_0=eps_0, eps_min=eps_min, eps_cutoff=cutoff)

# Run DRQN
start_time = time.time()
episode_durations, episode_returns, losses, best_model = run_episodes(train, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate, print_every=100, noise=state_noise)
duration = time.time() - start_time
print('run time in seconds: ', duration)
plot_traindata(episode_returns,losses)

# Evaluate
qnet_best = best_model
# if best_model_path is not None:
#     qnet_best.load_state_dict(torch.load(best_model_path))
policy.Q = qnet_best
print('evaluation of learned policy on trainset')
policy.epsilon=0.
EvaluatePolicy(env, policy, env.world_pool, print_runs=False, save_plots=True)

