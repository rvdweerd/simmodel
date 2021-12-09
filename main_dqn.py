from environments import GraphWorld
import numpy as np
import simdata_utils as su
import matplotlib.pyplot as plt
import torch
from dqn_utils import seed_everything, FastReplayMemory, train, run_episodes
from rl_models import QNetwork
from rl_utils import EvaluatePolicy, CreateDuplicatesTrainsets
from rl_policy import MinIndegreePolicy, EpsilonGreedyPolicyDQN
import time
import os

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
conf=configs['Manhattan5']
#conf=configs['Manhattan11']
#conf=configs['CircGraph']
conf['direction_north']=False
fixed_init=conf['fixed_initial_positions']

# Define qnet 
dims_hidden_layers = [256,128]

# Select hyperparameters
seed = 42  # This is not randomly chosen
batch_size      = 64
mem_size        = 15000
discount_factor = .9#1.#0.8
learn_rate      = 1e-4
num_episodes    = 25000
eps_0           = 1.
eps_min         = 0.05
cutoff          = 0.8*num_episodes # lower plateau reached and maintained from this point onward
state_noise     = False

# Initialize
seed_everything(seed)
env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=fixed_init, state_representation='et', state_encoding='tensor')
# Select specific trainset, set0 has identical states with different rollouts, set1 has identical states with identical rollouts
init_pos_trainset_indices0, init_pos_trainset_indices1 = CreateDuplicatesTrainsets(env, min_y_coord=env.sp.N-1, min_num_same_positions=env.sp.U, min_num_worlds=4, print_selection=False)
#env.world_pool = init_pos_trainset_indices0 # limit the training set to the selected entries
# Select full world pool
env.world_pool = env.all_worlds

dim_in = env.state_encoding_dim
dim_out = env.max_outdegree
memory = FastReplayMemory(mem_size,dim_in)
qnet = QNetwork(dim_in, dim_out, dims_hidden_layers).to(device)
policy = EpsilonGreedyPolicyDQN(qnet, env, eps_0=eps_0, eps_min=eps_min, eps_cutoff=cutoff)

# Run DQN
start_time = time.time()
episode_durations, episode_returns, losses, best_model_path = run_episodes(train, qnet, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate, print_every=100, noise=state_noise)
duration = time.time() - start_time
print('run time in seconds: ', duration)
plot_traindata(episode_returns,losses)

qnet_best = qnet
if best_model_path is not None:
    qnet_best.load_state_dict(torch.load(best_model_path))
policy.Q = qnet_best
print('evaluation of learned policy on trainset')
policy.epsilon=0.
EvaluatePolicy(env, policy, env.world_pool, print_runs=False, save_plots=False)
