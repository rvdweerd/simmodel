from spinup import ppo_pytorch as ppo
import gym


import numpy as np
import torch
import torch.nn as nn 
from modules.gnn.nfm_gen import NFM_ec_t, NFM_ev_t, NFM_ev_ec_t, NFM_ev_ec_t_um_us, NFM_ev_ec_t_u
from modules.ppo.ppo_wrappers import PPO_ActWrapper, PPO_ObsWrapper
#import modules.sim.simdata_utils as su
from modules.rl.rl_custom_worlds import GetCustomWorld

nfm_funcs = {
    'NFM_ev_ec_t'       : NFM_ev_ec_t(),
    'NFM_ec_t'          : NFM_ec_t(),
    'NFM_ev_t'          : NFM_ev_t(),
    'NFM_ev_ec_t_u'     : NFM_ev_ec_t_u(),
    'NFM_ev_ec_t_um_us' : NFM_ev_ec_t_um_us(),
    }


def CreateEnv():
    world_name='Manhattan3x3_WalkAround'
    state_repr='etUte0U0'
    state_enc='nfm'
    nfm_func_name = 'NFM_ev_ec_t'
    edge_blocking = True
    remove_world_pool = True
    env = GetCustomWorld(world_name, make_reflexive=True, state_repr=state_repr, state_enc=state_enc)
    env.redefine_nfm(nfm_funcs[nfm_func_name])
    env.capture_on_edges = edge_blocking
    if remove_world_pool:
        env._remove_world_pool()
    MAX_NODES=9
    env = PPO_ObsWrapper(env, max_possible_num_nodes = MAX_NODES)        
    env = PPO_ActWrapper(env)        

    return env
#env_fn = lambda : gym.make('GraphWorld-v0',**kwargs)
#env_fn = lambda : gym.make('LunarLander-v2')


env=CreateEnv()

#from modules.rl.environments import GraphWorld
#ac_kwargs = dict(hidden_sizes=[64,64], activation=torch.nn.ReLU)
ac_kwargs = dict(
    emb_dim       = 32,
    emb_iter_T    = 3,
    #max_num_nodes = 12,
)
logger_kwargs = dict(output_dir='results/gnn-ppo/spinningup/', exp_name='experiment_name')
from modules.ppo.models_spinup import Struc2VecActorCritic

ppo(env_fn=CreateEnv, actor_critic=Struc2VecActorCritic, ac_kwargs=ac_kwargs, 
        steps_per_epoch=100, 
        epochs=250, 
        train_pi_iters=4,
        train_v_iters=4,
        clip_ratio=0.1, 
        pi_lr=1e-4,
        vf_lr=1e-4,
        logger_kwargs=logger_kwargs)
#ppo(env_fn=CreateEnv, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=250, logger_kwargs=logger_kwargs)


k=0