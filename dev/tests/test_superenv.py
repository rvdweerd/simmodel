import copy
import random
from modules.gnn.comb_opt import train, evaluate, evaluate_spath_heuristic, evaluate_tabular
from modules.rl.rl_custom_worlds import GetCustomWorld
from modules.rl.environments import SuperEnv
from modules.gnn.nfm_gen import NFM_ec_t, NFM_ev_t, NFM_ev_ec_t, NFM_ev_ec_t_um_us
from modules.sim.graph_factory import GetWorldSet, LoadData
from modules.sim.simdata_utils import SimulateInteractiveMode
import numpy as np
import torch
import argparse
import gym
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#from Phase2_generate_partial_graphs import print_world_properties

def get_all_env(Etrain=[4],Utrain=[1]):
    scenario_name='test'
    #scenario_name = 'Train_U2E45'
    world_name = 'SubGraphsManhattan3x3'
    state_repr = 'etUte0U0'
    state_enc  = 'nfm'
    nfm_funcs = {'NFM_ev_ec_t':NFM_ev_ec_t(),'NFM_ec_t':NFM_ec_t(),'NFM_ev_t':NFM_ev_t(),'NFM_ev_ec_t_um_us':NFM_ev_ec_t_um_us()}
    nfm_func=nfm_funcs['NFM_ev_ec_t']
    edge_blocking = True
    solve_select = 'solvable'# only solvable worlds (so best achievable performance is 100%)
    reject_u_duplicates = False
    Etrain=[4]
    Utrain=[1]

    databank_full, register_full, solvable = LoadData(edge_blocking = True)
    env_all_train, hashint2env, env2hashint, env2hashstr = GetWorldSet(state_repr, state_enc, U=Utrain, E=Etrain, edge_blocking=edge_blocking, solve_select=solve_select, reject_duplicates=reject_u_duplicates, nfm_func=nfm_func)
    super_env = SuperEnv(env_all_train)
    #SimulateInteractiveMode(super_env)
    return super_env


