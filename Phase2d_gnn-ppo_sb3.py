#import argparse
#mport gym
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
from stable_baselines3.common.env_checker import check_env
#from stable_baselines3.common.vec_env import DummyVecEnv
#from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from modules.rl.environments import GraphWorld
from modules.rl.rl_custom_worlds import GetCustomWorld
from modules.rl.rl_utils import EvaluatePolicy, print_parameters, GetFullCoverageSample, NpWrapper
from modules.rl.environments import SuperEnv
#from modules.sim.simdata_utils import SimulateInteractiveMode
#import modules.sim.simdata_utils as su
from modules.sim.graph_factory import GetWorldSet, LoadData
from modules.gnn.nfm_gen import NFM_ec_t, NFM_ev_t, NFM_ev_ec_t, NFM_ev_ec_t_um_us, NFM_ev_ec_t_u
from modules.ppo.ppo_wrappers import PPO_ActWrapper, PPO_ObsWrapper
from modules.ppo.callbacks_sb3 import SimpleCallback, TestCallBack
from modules.ppo.models_sb3 import s2v_ActorCriticPolicy, Struc2Vec
#from typing import Callable, Dict, List, Optional, Tuple, Type, Union
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_super_env(Utrain=[1], Etrain=[4], max_nodes=9):
    scenario_name='test'
    #scenario_name = 'Train_U2E45'
    world_name = 'SubGraphsManhattan3x3'
    state_repr = 'etUte0U0'
    state_enc  = 'nfm'
    nfm_funcs = {'NFM_ev_ec_t':NFM_ev_ec_t(),'NFM_ec_t':NFM_ec_t(),'NFM_ev_t':NFM_ev_t(),'NFM_ev_ec_t_um_us':NFM_ev_ec_t_um_us()}
    nfm_func=nfm_funcs['NFM_ev_ec_t_um_us']
    edge_blocking = True
    solve_select = 'solvable'# only solvable worlds (so best achievable performance is 100%)
    reject_u_duplicates = False

    #databank_full, register_full, solvable = LoadData(edge_blocking = True)
    env_all_train, hashint2env, env2hashint, env2hashstr = GetWorldSet(state_repr, state_enc, U=Utrain, E=Etrain, edge_blocking=edge_blocking, solve_select=solve_select, reject_duplicates=reject_u_duplicates, nfm_func=nfm_func)
    for i in range(len(env_all_train)):
        env_all_train[i]=PPO_ObsWrapper(env_all_train[i], max_possible_num_nodes = max_nodes)        
        env_all_train[i]=PPO_ActWrapper(env_all_train[i])        
    super_env = SuperEnv(env_all_train, hashint2env, max_possible_num_nodes = max_nodes)
    #SimulateInteractiveMode(super_env)
    return super_env

nfm_funcs = {
    'NFM_ev_ec_t'       : NFM_ev_ec_t(),
    'NFM_ec_t'          : NFM_ec_t(),
    'NFM_ev_t'          : NFM_ev_t(),
    'NFM_ev_ec_t_u'     : NFM_ev_ec_t_u(),
    'NFM_ev_ec_t_um_us' : NFM_ev_ec_t_um_us(),
    }


def CreateEnv(world_name, max_nodes=9):
    #world_name='Manhattan3x3_WalkAround'
    state_repr='etUte0U0'
    state_enc='nfm'
    nfm_func_name = 'NFM_ev_ec_t_um_us'
    edge_blocking = True
    remove_world_pool = False
    env = GetCustomWorld(world_name, make_reflexive=True, state_repr=state_repr, state_enc=state_enc)
    env.redefine_nfm(nfm_funcs[nfm_func_name])
    env.capture_on_edges = edge_blocking
    if remove_world_pool:
        env._remove_world_pool()
    #SimulateInteractiveMode(env,filesave_with_time_suffix=False)
    MAX_NODES=9
    env = PPO_ObsWrapper(env, max_possible_num_nodes = max_nodes)        
    env = PPO_ActWrapper(env)        
    return env

train=True
eval=True
if train:
    MAX_NODES=33
    EMB_DIM = 64
    EMB_ITER_T = 5
    env=CreateEnv(world_name='Manhattan3x3_WalkAround', max_nodes=MAX_NODES)
    #env=get_super_env(Utrain=[1], Etrain=[0],max_nodes=MAX_NODES)
    #env=get_super_env(Utrain=[1,2,3], Etrain=[0,1,2,3,4,5,6,7,8,9],max_nodes=MAX_NODES)
    obs=env.reset()
    NODE_DIM = env.F

    policy_kwargs = dict(
        features_extractor_class=Struc2Vec,
        features_extractor_kwargs=dict(emb_dim=EMB_DIM, emb_iter_T=EMB_ITER_T, node_dim=NODE_DIM),#, num_nodes=MAX_NODES),
        #net_arch=dict(max_num_nodes=MAX_NODES, emb_dim=EMB_DIM, num_nodes=MAX_NODES)
        # NOTE: FOR THIS TO WORK, NEED TO ADJUST sb3 policies.py
        #           def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        #       to create a linear layer that maps to 1-dim instead of self.action_dim
        #       reason: our model is invariant to the action space (number of nodes in the graph) 
    )

    model = MaskablePPO(s2v_ActorCriticPolicy, env, \
        #learning_rate=1e-4,\
        seed=0,\
        #clip_range=0.1,\    
        #max_grad_norm=0.1,\
        policy_kwargs = policy_kwargs, verbose=1, tensorboard_log="results/gnn-ppo/sb3/test/tensorboard/")

    print_parameters(model.policy)
    model.learn(total_timesteps = 300000, callback=TestCallBack())
    model.save("results/gnn-ppo/sb3/test/ppo_trained_on_all_3x3")
    policy = model.policy
    policy.save("results/gnn-ppo/sb3/test/ppo_policy_trained_on_all_3x3")    

    res = evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=-15, warn=False, return_episode_rewards=False)
    print('Test result: avg rew:', res[0], 'std:', res[1])

def check_custom_position_probs(env,ppo_policy,hashint=None,entry=None,targetnodes=[1],epath=[4],upaths=[[6]]):    
    if hashint is not None:
        obs=env.reset(hashint, entry)
    else:
        obs=env.reset(entry=entry)
    nfm = env.get_custom_nfm(epath,targetnodes,upaths)
    p = MAX_NODES - env.sp.V
    nfm = nn.functional.pad(nfm,(0,0,0,p))
    print(nfm)
    W = nn.functional.pad(env.sp.W,(0,p,0,p))
    obs = torch.cat((nfm, W, torch.index_select(W, 1, torch.tensor(epath[-1]))),1)#.to(device)
    reachable=np.zeros(env.sp.V).astype(np.bool)
    idx=np.array(env.neighbors[epath[-1]]).astype(int)
    reachable[idx]=True
    action_masks=list(reachable)+[False]*p
    action_masks=torch.tensor(action_masks,dtype=torch.bool)#.to(device)

    action, _state = ppo_policy.predict(obs, deterministic=True, action_masks=action_masks)
    probs1 = F.softmax(ppo_policy.get_distribution(obs[None].to(device)).log_prob(torch.tensor(env.neighbors[epath[-1]]).to(device)))
    np.set_printoptions(formatter={'float':"{0:0.2f}".format})
    print('actions',[a for a in env.neighbors[epath[-1]]],'; probs:',probs1.detach().cpu().numpy(), 'chosen:', action)

    newsp=copy.deepcopy(env.sp)
    newsp.target_nodes=targetnodes
    fname='results/gnn-ppo/sb3/test/'+'hashint'+str(hashint)+'_target'+str(targetnodes)+'_epath'+str(epath)+'_upaths'+str(upaths)
    PlotEUPathsOnGraph_(newsp,epath,upaths,filename=fname,fig_show=False,fig_save=True,goal_reached=False,last_step_only=False)

from modules.rl.rl_plotting import PlotEUPathsOnGraph_
import copy
if eval:
    # nfm_funcs = {
    #     'NFM_ev_ec_t'       : NFM_ev_ec_t(),
    #     'NFM_ec_t'          : NFM_ec_t(),
    #     'NFM_ev_t'          : NFM_ev_t(),
    #     'NFM_ev_ec_t_u'     : NFM_ev_ec_t_u(),
    #     'NFM_ev_ec_t_um_us' : NFM_ev_ec_t_um_us(),
    # }
    # nfm_func=nfm_funcs[args.nfm_func]
    # edge_blocking = args.edge_blocking
    # solve_select = 'solvable' # only solvable worlds (so best achievable performance is 100%)

    # world_name='MetroU3_e17tborder_FixedEscapeInit'
    # scenario_name='TrainMetro'
    # state_repr='etUte0U0'
    # state_enc='nfm'
    MAX_NODES=33
    EMB_DIM = 64
    EMB_ITER_T = 5
    #env=CreateEnv(world_name='Manhattan3x3_WalkAround', max_nodes=MAX_NODES)
    #env=get_super_env(Utrain=[1], Etrain=[4],max_nodes=MAX_NODES)
    env=CreateEnv(world_name='MetroU3_e17tborder_FixedEscapeInit', max_nodes=MAX_NODES)
    #env=CreateEnv(world_name='Manhattan3x3_WalkAround', max_nodes=MAX_NODES)
    #SimulateInteractiveMode(env,filesave_with_time_suffix=False)
    
    
    NODE_DIM = env.F

    #model=PPO.load('ppo_trained_on_all_3x3')
    saved_policy = s2v_ActorCriticPolicy.load('results/gnn-ppo/sb3/test/ppo_policy_trained_on_all_3x3')
    #check_custom_position_probs(env,saved_policy,hashint=4041,entry=5,targetnodes=[1],  epath=[4],upaths=[[6]])
    #check_custom_position_probs(env,saved_policy,hashint=4041,entry=5,targetnodes=[3],  epath=[4],upaths=[[6]])
    #check_custom_position_probs(env,saved_policy,hashint=4041,entry=5,targetnodes=[6],  epath=[4],upaths=[[1]])
    #check_custom_position_probs(env,saved_policy,hashint=4041,entry=5,targetnodes=[3],  epath=[4],upaths=[[1]])
    
    check_custom_position_probs(env,saved_policy,targetnodes=[8],epath=[1],upaths=[[6,3],[7,4]])

    check_custom_position_probs(env,saved_policy,hashint=4041,entry=5,targetnodes=[2,7],epath=[4],upaths=[[5,2]])
    check_custom_position_probs(env,saved_policy,hashint=4041,entry=5,targetnodes=[2,7],epath=[3,4],upaths=[[5,2]])
    check_custom_position_probs(env,saved_policy,hashint=4041,entry=5,targetnodes=[2,7],epath=[4],upaths=[[2]])
    check_custom_position_probs(env,saved_policy,hashint=4041,entry=5,targetnodes=[2,7],epath=[4],upaths=[[5]])


    #res = evaluate_policy(saved_policy, env, n_eval_episodes=20, reward_threshold=-100, warn=False, return_episode_rewards=True)
    #print('Test result: avg rew:', res[0], 'std:', res[1])

    res_det = evaluate_policy(saved_policy, env, n_eval_episodes=20, reward_threshold=-100, warn=False, return_episode_rewards=False, deterministic=True)
    print('Test result (deterministic): avg rew:', res_det[0], 'std:', res_det[1])
    res_nondet = evaluate_policy(saved_policy, env, n_eval_episodes=20, reward_threshold=-100, warn=False, return_episode_rewards=False, deterministic=False)
    print('Test result (non-deterministic): avg rew:', res_nondet[0], 'std:', res_nondet[1])

    np.set_printoptions(formatter={'float':"{0:0.2f}".format})
    fpath='results/gnn-ppo/sb3/'
    for i in range(10):
        obs = env.reset()
        print('\nHashint: ',env.sp.hashint, ', entry:', env.current_entry)
        print('Initial state:',env.state)
        env.env.render(fname=fpath+'test'+str(i))
        done=False
        while not done:
            action_masks = get_action_masks(env)
            action, _state = saved_policy.predict(obs, deterministic=True, action_masks=action_masks)
            probs1 = F.softmax(saved_policy.get_distribution(obs[None].to(device)).log_prob(torch.tensor(env.neighbors[env.state[0]]).to(device)))
            print('actions',[a for a in env.neighbors[env.state[0]]],'; probs:',probs1.detach().cpu().numpy(), 'chosen:', action)
            obs, reward, done, info = env.step(action)
            env.env.render(fname=fpath+'test'+str(i))
            if done:
                env.env.render_eupaths(fname=fpath+'test'+str(i)+'_final')

                #probs1 = F.softmax(saved_policy.get_distribution(obs[None].to(device)).log_prob(torch.tensor(env.neighbors[env.state[0]]).to(device)))
                # check (all nodes):
                # probs2 = F.softmax(saved_policy.get_distribution(obs[None].to(device),action_masks=action_masks).log_prob(torch.tensor([i for i in range(33)]).to(device)))
                obs = env.reset()