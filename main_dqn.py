from environments import GraphWorld
from rl_policy import EpsilonGreedyPolicy
from rl_algorithms import q_learning, sarsa, expected_sarsa
from rl_plotting import PlotPerformanceCharts, PlotGridValues, PlotNodeValues
from rl_utils import EvaluatePolicy, SelectTrainset
import numpy as np
import simdata_utils as su

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm as _tqdm
from dqn_utils import QNetwork, MemTest, TorchTest, TensorTest, PolicyTest, EpsilonGreedyPolicyDQN

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer
device = 'cuda' if torch.cuda.is_available() else 'cpu'


num_seeds   = 10
eps_0       = .2
eps_min     = 0.
cutoff      = 100#200
num_iter    = 500
gamma       = 1.
alpha_0     = .5
alpha_decay = 0.
initial_Q_values = 0.

configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
conf=configs['Manhattan5']
conf['direction_north']=False
env = GraphWorld(conf, optimization_method='dynamic', fixed_initial_positions=(2,15,19,22),state_representation='etUt', state_encoding='tensor')

print('Device = ',device)
MemTest(env)
TorchTest()
TensorTest(env)
PolicyTest(env)


