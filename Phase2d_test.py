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
from modules.sim.simdata_utils import SimulateInteractiveMode_PPO
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_configs=get_train_configs()
#  RunA: Constructed trainset, n=500k, sol=both https://github.com/rvdweerd/simmodel/blob/a39cda4ce18a33e843e89673ef24b064b7907003/Phase2d_construct_sets.py#L20
#  RunB:            ,,       , n=300k, sol=both https://github.com/rvdweerd/simmodel/blob/5e8b82f7ed9427588d72b6b2dd2f90e61402c3ba/Phase2d_construct_sets.py#L20
# *RunC:            ,,       , n=300k, sol=solvable https://github.com/rvdweerd/simmodel/blob/5e8b82f7ed9427588d72b6b2dd2f90e61402c3ba/Phase2d_construct_sets.py#L20
#  RunD:            ,,       , n=500k, sol=solvable emb=128
#  RunE:                     , n=500k, idem + iterT=7                    
config=train_configs['RunC']

#train_on ='SubGraphsManhattan3x3'
# train_on ='Manhattan5x5_VariableEscapeInit'
# train_on='ContructedSuperSet'
# #scenario_name   = 'Train_U123E0123456789'
# scenario_name = ''
# config['solve_select'] = 'both'
# config['edge_blocking'] = True
# config['nfm_func_name'] = 'NFM_ev_ec_t_um_us'
# config['emb_dim'] = 64
# config['emb_iter_T'] = 5
# config['max_nodes'] = 33
# #config['num_step'] = 300000
# config['num_step'] = 500000
# rootdir='results/results_Phase2/Pathfinding/ppo/'+train_on+'/'+config['solve_select']+'_edgeblock'+str(config['edge_blocking'])#+'/'+scenario_name
# logdir=rootdir+'/'+\
#         config['nfm_func_name']+'/'+ \
#         'emb'+str(config['emb_dim']) + \
#         '_itT'+str(config['emb_iter_T']) + \
#         '_nstep'+str(config['num_step'])
# config['rootdir']=rootdir
# config['logdir']=logdir
logdir=config['logdir']
seed=config['seed0']

#global_env=[]
# #config['solve_select']='solvable'
# #env, _ = get_super_env(Uselected=[0,1,2,3], Eselected=[0,1,2,3,4,5,6,7,8,9], config=config)
# #for u in [0,1,2,3]:
# #    env, _ = get_super_env(Uselected=[u], Eselected=[0,1,2,3,4,5,6,7,8,9], config=config)
# #    global_env.append(env)
# world_names=[
#     #'Manhattan5x5_FixedEscapeInit',
#     #'Manhattan5x5_VariableEscapeInit',
#     #'MetroU3_e17tborder_FixedEscapeInit',
#     'MetroU3_e1t31_FixedEscapeInit',
#     #'SparseManhattan5x5',
# ]
# for w in world_names:
#     env = CreateEnv(w,max_nodes=config['max_nodes'],var_targets=[4,4])
#     global_env.append(env)
# env=SuperEnv(global_env,hashint2env=None,max_possible_num_nodes=33)#,probs=[1,10,1,1,1,1,1,1])
saved_model = MaskablePPO.load(logdir+'/SEED'+str(seed)+"/saved_models/model_last")

env = CreateEnv('MetroU3_e17tborder_FixedEscapeInit',max_nodes=config['max_nodes'],var_targets=None, remove_world_pool=False)

#env=ConstructTrainSet(config)

#check_custom_position_probs(env,saved_model.policy,hashint=None,entry=None,targetnodes=[22],epath=[14],upaths=[[17]],max_nodes=33,logdir=logdir)
#check_custom_position_probs(env,saved_model.policy,hashint=None,entry=None,targetnodes=[22],epath=[14],upaths=[[18]],max_nodes=33,logdir=logdir)
while True:
    SimulateInteractiveMode_PPO(env, saved_model, t_suffix=False)

saved_policy = s2v_ActorCriticPolicy.load(logdir+'/SEED'+str(seed)+"/saved_models/policy_last")
ppo_policy = ActionMaskedPolicySB3_PPO(saved_policy, deterministic=True)

import random
while True:
    #env=random.choice(global_env)
    obs=env.reset()#.to(device)
    done=False
    #action_masks=torch.tensor(env.action_masks())#.to(device)
    while not done:
        action, _state = ppo_policy.sample_greedy_action(obs,None,printing=True)
        env.render(fname='test',t_suffix=False)
        
        obs,r,done,i = env.step(action)
        a=input('[q]-stop current, [enter]-take step')
        if a=='q': break
    env.render(fname='test',t_suffix=False)
    env.render_eupaths(fname='final',t_suffix=False)
    input('')
    k=0

