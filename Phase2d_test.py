import random
from matplotlib.pyplot import get
import torch
import torch.nn as nn 
import torch.nn.functional as F
from modules.ppo.helpfuncs import get_super_env, CreateEnv, eval_simple, evaluate_ppo, check_custom_position_probs
from modules.rl.environments import SuperEnv
from modules.rl.rl_policy import ActionMaskedPolicySB3_PPO
from modules.ppo.models_sb3 import s2v_ActorCriticPolicy, Struc2Vec
from modules.ppo.ppo_wrappers import VarTargetWrapper
from sb3_contrib import MaskablePPO
from Phase2d_construct_sets import ConstructTrainSet, get_train_configs
from modules.sim.simdata_utils import SimulateInteractiveMode_PPO, SimulateAutomaticMode_PPO
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Path directions to best model from ppo experiment
runname='RunC'
train_configs=get_train_configs(runname, load_trainset=False)
seed = 0 
config = train_configs[runname]
logdir = config['logdir']

# OPTIONS TO LOAD WORLDS:
# 1. 3x3 graph permutations
#config['solve_select']='solvable'
#env, _ = get_super_env(Uselected=[0,1,2,3], Eselected=[0,1,2,3,4,5,6,7,8,9], config=config)

## 2. Set of specific worlds
global_env=[]
# world_names=[
    #'Manhattan5x5_FixedEscapeInit',
    #'Manhattan5x5_VariableEscapeInit',
    #'MetroU3_e17tborder_FixedEscapeInit',
    #'MetroU3_e1t31_FixedEscapeInit',
    #'SparseManhattan5x5' ]
env = CreateEnv('MetroU3_e1t31_FixedEscapeInit',max_nodes=config['max_nodes'],var_targets=[4,4])
# for w in world_names:
#     env = CreateEnv(w,max_nodes=config['max_nodes'],var_targets=[4,4])
#     global_env.append(env)
# env=SuperEnv(global_env,hashint2env=None,max_possible_num_nodes=33)#,probs=[1,10,1,1,1,1,1,1])

## 3. Individual environment
#env = CreateEnv('MetroU3_e1t31_FixedEscapeInit',max_nodes=config['max_nodes'],var_targets=[3,3], remove_world_pool=False)

## 4. Pre-defined training set for ppo experiments
#env = ConstructTrainSet(config)

## Load pre-saved model
saved_model = MaskablePPO.load(logdir+'/SEED'+str(seed)+"/saved_models/model_last")
saved_policy = s2v_ActorCriticPolicy.load(logdir+'/SEED'+str(seed)+"/saved_models/policy_last")
ppo_policy = ActionMaskedPolicySB3_PPO(saved_policy, deterministic=True)

# OPTIONS TO PERFORM TESTS

## 1. Evaluate a specific constellation on the graph
check_custom_position_probs(env,saved_model.policy,hashint=None,entry=None,targetnodes=[13,22,23,29],epath=[17],upaths=[[23,22],[30,27],[32,7]],max_nodes=33,logdir=logdir)
check_custom_position_probs(env,saved_model.policy,hashint=None,entry=None,targetnodes=[13,22,23,29],epath=[17],upaths=[[12,13],[30,27],[32,7]],max_nodes=33,logdir=logdir)

## 2. Run Interactive simulation 
# plots are updated in the results folder
while True:
    a = SimulateInteractiveMode_PPO(env, model = saved_model, t_suffix=False)
    if a == 'Q': break

## 3. Run automated simulation (stepping)
while True:
    entries=[5012,218,3903]
    a = SimulateAutomaticMode_PPO(env, ppo_policy, entries)
    if a == 'Q': break
    
