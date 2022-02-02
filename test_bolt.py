import warnings

from modules.gnn.nfm_gen import NFM_ev_ec_t_um_us_xW
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
#from modules.sim.graph_factory import GetPartialGraphEnvironments_Manh3x3
#from modules.gnn.nfm_gen import NFM_ev_ec_t_um_us_xW
from modules.rl.environments import GraphWorld#, GraphWorldFromDatabank
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from collections import namedtuple
from pytorch_lightning import Trainer
from modules.sim.graph_factory import LoadData
import gym
#import modules.rl.environments

databank_full, register_full, solvable = LoadData()

#world_name='SparseManhattan5x5'
#env = GetCustomWorld(world_name, make_reflexive=True, state_repr=state_repr, state_enc=state_enc)

import modules.sim.simdata_utils as su
configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
conf=configs['SparseManhattan5x5']
state_repr='etUt'
state_enc='nfm'
conf['direction_north']=False
conf['loadAllStartingPositions']=False
conf['make_reflexive']=True
kwargs = {'config':conf, 'optimization_method':'static', 'fixed_initial_positions':None, 'state_representation':state_repr, 'state_encoding':state_enc}
#env = gym.make('GraphWorld-v0', **kwargs)
#nfm_func=NFM_ev_ec_t_um_us_xW()
#env.redefine_nfm(nfm_func)
#env.reset()
#env.redefine_goal_nodes([24])
#env._remove_world_pool()
#SimulateInteractiveMode(env)
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pl_bolts.models.rl import DQN

tblogger = TensorBoardLogger("tb_logs", name="my_model")
wblogger = WandbLogger(project="SPath", log_model="all")


dqn = DQN(env ='GraphWorld-v0', **kwargs)
trainer = Trainer(logger=[wblogger,tblogger],gpus=1,accelerator='gpu')
trainer.fit(dqn)

