import copy
from modules.gnn.comb_opt import train, evaluate, evaluate_spath_heuristic, evaluate_tabular
from modules.rl.rl_custom_worlds import GetCustomWorld
from modules.gnn.nfm_gen import NFM_ec_t, NFM_ev_t, NFM_ev_ec_t, NFM_ev_ec_t_um_us
from modules.sim.graph_factory import GetWorldSet, LoadData
from modules.sim.simdata_utils import SimulateInteractiveMode
from modules.ppo.ppo_wrappers import PPO_ActWrapper, PPO_ObsFlatWrapper, PPO_ObsDictWrapper, VarTargetWrapper, PPO_ObsBasicDictWrapper, PPO_ObsBasicDictWrapperCRE
import numpy as np
import torch
import argparse
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from Phase2_generate_partial_graphs import print_world_properties

world_name = 'SubGraphsManhattan3x3'
state_repr = 'etUte0U0'
state_enc  = 'nfm'
nfm_funcs = {'NFM_ev_ec_t':NFM_ev_ec_t(),'NFM_ec_t':NFM_ec_t(),'NFM_ev_t':NFM_ev_t(),'NFM_ev_ec_t_um_us':NFM_ev_ec_t_um_us()}
nfm_func = nfm_funcs['NFM_ev_ec_t_um_us']
obs_mask='None'
obs_rate=1.0
edge_blocking = True
solve_select = 'solvable'
reject_u_duplicates = False
Etrain=[3]
Utrain=[1]

databank_full, register_full, solvable = LoadData(edge_blocking = True)
evalenv, hashint2env, env2hashint, env2hashstr = GetWorldSet(state_repr, state_enc, U=Utrain, E=Etrain, edge_blocking=edge_blocking, solve_select=solve_select, reject_duplicates=reject_u_duplicates, nfm_func=nfm_func)
for i in range(len(evalenv)):
    evalenv[i]=PPO_ObsBasicDictWrapper(evalenv[i], obs_mask=obs_mask, obs_rate=obs_rate)
    evalenv[i]=PPO_ActWrapper(evalenv[i])
    assert len(evalenv[i].world_pool) == len(evalenv[i].all_worlds)

hashint=4007
env_idx=hashint2env[hashint]
env=evalenv[env_idx]
entry=0
#hashint=env2hashint[env_idx]
hashstr=env2hashstr[env_idx]
u=env.sp.U
s= solvable['U='+str(u)][hashint]
print_world_properties(env, env_idx, entry, hashint, hashstr, edge_blocking, solve_select, reject_u_duplicates, solvable_=s)    
while True:
    SimulateInteractiveMode(env, filesave_with_time_suffix=False, entry=entry)
