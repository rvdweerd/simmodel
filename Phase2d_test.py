from http.client import NOT_IMPLEMENTED
#import random
#from matplotlib.pyplot import get
import torch
#import torch.nn as nn 
#import torch.nn.functional as F
from modules.ppo.helpfuncs import get_super_env, CreateEnv, eval_simple, evaluate_ppo, check_custom_position_probs
from modules.rl.environments import SuperEnv
from modules.rl.rl_policy import ActionMaskedPolicySB3_PPO
from modules.ppo.models_sb3 import s2v_ActorCriticPolicy, Struc2Vec, DeployablePPOPolicy
#from modules.ppo.ppo_wrappers import VarTargetWrapper
from sb3_contrib import MaskablePPO
from modules.gnn.construct_trainsets import ConstructTrainSet, get_train_configs
from modules.sim.simdata_utils import SimulateInteractiveMode_PPO, SimulateAutomaticMode_PPO
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Path directions to best model from ppo experiment
runname='RunC' # bestrun
#runname = 'SuperSet_noU'
train_configs=get_train_configs(runname, load_trainset=False)
seed = 0 
config = train_configs[runname]
logdir = config['logdir']
print(logdir)
# OPTIONS TO LOAD WORLDS:
# 1. 3x3 graph permutations
#config['solve_select']='solvable'
#env, _ = get_super_env(Uselected=[2,3], Eselected=[4], config=config)

## 2. Set of specific worlds
#global_env=[]
# world_names=[
    #'Manhattan5x5_FixedEscapeInit',
    #'Manhattan5x5_VariableEscapeInit',
    #'MetroU3_e17tborder_FixedEscapeInit',
    #'MetroU3_e1t31_FixedEscapeInit',
    #'SparseManhattan5x5' ]
#env = CreateEnv('MetroU3_e17tborder_FixedEscapeInit',max_nodes=config['max_nodes'],var_targets=None)#[4,4])
# for w in world_names:
#     env = CreateEnv(w,max_nodes=config['max_nodes'],var_targets=[4,4])
#     global_env.append(env)
# env=SuperEnv(global_env,hashint2env=None,max_possible_num_nodes=33)#,probs=[1,10,1,1,1,1,1,1])

## 3. Individual environment
#env = CreateEnv('MetroU3_e17tborder_FixedEscapeInit',max_nodes=config['max_nodes'],var_targets=None, remove_world_pool=False)
#env = CreateEnv('MetroU3_e1t31_FixedEscapeInit',max_nodes=33,nfm_func_name =config['nfm_func'],var_targets=[1,1], remove_world_pool=True)
#env = CreateEnv('Manhattan5x5_FixedEscapeInit',max_nodes=config['max_nodes'],var_targets=[3,3], remove_world_pool=False)
env = CreateEnv('NWB_test_VariableEscapeInit',nfm_func_name =config['nfm_func_name'],max_nodes=975,var_targets=None, remove_world_pool=False)
#env = CreateEnv('NWB_test_VariableEscapeInit',nfm_func_name =config['nfm_func'],max_nodes=975,var_targets=None, remove_world_pool=True)

## 4. Pre-defined training set for ppo experiments
#env, _ = ConstructTrainSet(config)

## Load pre-saved model
saved_model = MaskablePPO.load(logdir+'/SEED'+str(seed)+"/saved_models/model_last")
saved_policy = s2v_ActorCriticPolicy.load(logdir+'/SEED'+str(seed)+"/saved_models/policy_last")




saved_policy_deployable=DeployablePPOPolicy(env, saved_policy)


#ppo_policy = ActionMaskedPolicySB3_PPO(saved_policy, deterministic=True)
ppo_policy = ActionMaskedPolicySB3_PPO(saved_policy_deployable, deterministic=True)


# OPTIONS TO PERFORM TESTS

## 1. Evaluate a specific constellation on the graph
## Metro example, left turn or right turn
#check_custom_position_probs(env,saved_model.policy,hashint=None,entry=None,targetnodes=[13,22,23,29],epath=[17],upaths=[[23,22],[30,27],[32,7]],max_nodes=33,logdir=logdir)
#check_custom_position_probs(env,saved_model.policy,hashint=None,entry=None,targetnodes=[13,22,23,29],epath=[17],upaths=[[12,13],[30,27],[32,7]],max_nodes=33,logdir=logdir)
## Metro example, long range shortest path
#check_custom_position_probs(env,saved_model.policy,hashint=None,entry=None,targetnodes=[31],epath=[1],upaths=[[14]],max_nodes=33,logdir=logdir)
#check_custom_position_probs(env,saved_model.policy,hashint=None,entry=None,targetnodes=[31],epath=[1],upaths=[[14,17]],max_nodes=33,logdir=logdir)

## Metro example, long range shortest path with one pursuer
# epath=[1,5,6,7,14,18,19,25,29,31]
# upaths=[]#[[9,8,7]]
# check_custom_position_probs(env,saved_model.policy,hashint=None,entry=None,targetnodes=[31],epath=epath,upaths=upaths,max_nodes=33,logdir=logdir)

## 2. Run Interactive simulation 
# plots are updated in the results folder
# while True:
#     a = SimulateInteractiveMode_PPO(env, model = saved_model, t_suffix=True)
#     if a == 'Q': break

## 3. Run automated simulation (stepping)
while True:
    entries=None#[5012,218,3903]
    a = SimulateAutomaticMode_PPO(env, ppo_policy, t_suffix=False, entries=entries)
    if a == 'Q': break
    

