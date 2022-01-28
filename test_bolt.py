import warnings
warnings.filterwarnings("ignore")
# Demo of State2vec + DQN to solve a single world
import tqdm
from tkinter import W
import matplotlib.pyplot as plt
from modules.dqn.dqn_utils import seed_everything
from modules.gnn.comb_opt import QNet, QFunction, init_model, checkpoint_model, Memory
from modules.rl.rl_policy import GNN_s2v_Policy
from modules.rl.rl_custom_worlds import GetCustomWorld
from modules.rl.rl_utils import EvaluatePolicy, EvalArgs2
from modules.sim.simdata_utils import SimulateInteractiveMode
from modules.sim.graph_factory import GetPartialGraphEnvironments_Manh3x3
from modules.rl.environments import GraphWorld, GraphWorldFromDatabank
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from collections import namedtuple
from pytorch_lightning import Trainer
from modules.sim.graph_factory import LoadData
import gym
import modules.rl.environments

databank_full, register_full, solvable, reachable = LoadData()
state_repr='etUt'
state_enc='nodes'
config={
        'graph_type': "Manhattan",
        'make_reflexive': True,
        'N': 3,    # number of nodes along one side
        'U': 2,    # number of pursuer units
        'L': 4,    # Time steps
        'T': 7,
        'R': 100,  # Number of escape routes sampled 
        'direction_north': False,       # Directional preference of escaper
        'loadAllStartingPositions': False
    }
reg = register_full[2] # list of all (W,hashstr,hashint) combinations for 2 edge removals
entry=0
hashint=reg[entry][1]
env_data = databank_full['U=2'][hashint] # dict contains  'register':{(e0,U0):index}, 'databank':[], 'iratios':[]
env_data['W'] = reg[0][0]
env = GraphWorldFromDatabank(config,env_data,optimization_method='static',state_representation=state_repr,state_encoding=state_enc)


#env2=gym.make('GraphWorldFromDB-v0', config=config, env_data=env_data)
from pl_bolts.models.rl import DQN
dqn = DQN('GraphWorldFromDB-v0', config=config, env_data=env_data)
trainer = Trainer()
trainer.fit(dqn)

