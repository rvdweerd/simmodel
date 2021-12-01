from environments import GraphWorld
from rl_policy import EpsilonGreedyPolicy
from rl_algorithms import q_learning, sarsa, expected_sarsa
from rl_plotting import PlotPerformanceCharts, PlotGridValues, PlotNodeValues
from rl_utils import EvaluatePolicy, SelectTrainset
import numpy as np
import simdata_utils as su
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm as _tqdm
from dqn_utils import QNetwork, ReplayMemory, FastReplayMemory, EpsilonGreedyPolicyDQN,\
     compute_q_vals, compute_targets, train, run_episodes
from dqn_tests import MemTest, FastMemTest, TorchTest, TensorTest, PolicyTest, TrainTest
import random
import os
import time

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # if you are using multi-GPU.
    np.random.seed(seed)
    # Numpy module.
    random.seed(seed)
    # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer
device = 'cuda' if torch.cuda.is_available() else 'cpu'


configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
conf=configs['Manhattan5']
conf['direction_north']=False

#env = GraphWorld(conf, optimization_method='dynamic', fixed_initial_positions=(2,15,19,22),state_representation='etUt', state_encoding='tensor')
#MemTest(env)
#FastMemTest(env)
#TorchTest()
#TensorTest(env)
#PolicyTest(env)
#TrainTest(env)

seed = 42  # This is not randomly chosen
batch_size = 64
discount_factor = .9#1.#0.8
learn_rate = 1e-3#1e-3
#num_seeds   = 10
eps_0       = 1.#.5
eps_min     = 0.05
cutoff      = 3500
num_episodes = 900

# We will seed the algorithm (before initializing QNetwork!) for reproducibility
#random.seed(seed)
#torch.manual_seed(seed)
#env.seed(seed)
seed_everything(seed)
print('Device = ',device)

env = GraphWorld(conf, optimization_method='dynamic', fixed_initial_positions=(2,15,19,22),state_representation='etUt', state_encoding='tensor')

dim_in=(1+env.sp.U)*env.sp.V
dim_out=4 #(max out-degree)
dim_hidden=128

memory = FastReplayMemory(10000,dim_in)
qnet = QNetwork(dim_in,dim_out,dim_hidden).to(device)
policy = EpsilonGreedyPolicyDQN(qnet, env, eps_0=eps_0, eps_min=eps_min, eps_cutoff=cutoff)

start_time = time.time()
episode_durations, episode_returns, losses = run_episodes(train, qnet, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate, noise=False)
duration = time.time() - start_time
print('run time in seconds: ', duration)

plt.plot(episode_returns)
plt.savefig('testplots_returns_curve.png')
plt.clf()
plt.plot(losses)
plt.savefig('testplots_loss_curve.png')

s=env.reset()
policy.epsilon=0.
plt.clf()
done=False
while True:
    env.render()
    if done:
        break
    action_idx, action = policy.sample_action(s,env._availableActionsInCurrentState())
    s_next, r, done, info = env.step(action_idx)
    s=s_next
