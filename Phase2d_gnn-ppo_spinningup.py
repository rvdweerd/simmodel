from spinup import ppo_pytorch as ppo
import gym
from gym import ObservationWrapper, ActionWrapper
from gym import spaces
import numpy as np
import torch
import torch.nn as nn 
from modules.gnn.nfm_gen import NFM_ec_t, NFM_ev_t, NFM_ev_ec_t, NFM_ev_ec_t_um_us, NFM_ev_ec_t_u
#import modules.sim.simdata_utils as su
from modules.rl.rl_custom_worlds import GetCustomWorld

nfm_funcs = {
    'NFM_ev_ec_t'       : NFM_ev_ec_t(),
    'NFM_ec_t'          : NFM_ec_t(),
    'NFM_ev_t'          : NFM_ev_t(),
    'NFM_ev_ec_t_u'     : NFM_ev_ec_t_u(),
    'NFM_ev_ec_t_um_us' : NFM_ev_ec_t_um_us(),
    }

class PPO_ObsWrapper(ObservationWrapper):
    """Wrapper for stacking nfm|W|reachable nodes."""

    def __init__(self, env, max_possible_num_nodes = 12):
        super().__init__(env)
        assert max_possible_num_nodes >= self.sp.V
        self.max_nodes = max_possible_num_nodes
        self.observation_space= spaces.Box(0., self.sp.U, shape=(self.max_nodes, (self.F+self.max_nodes+1)), dtype=np.float32)
        self.action_space     = spaces.Discrete(self.max_nodes) # all possible nodes 
        #self.observation_space
        #self.action_space
    
    def observation(self, observation):
        """convert observation."""
        p = self.max_nodes - env.sp.V
        nfm = nn.functional.pad(self.nfm,(0,0,0,p))
        W = nn.functional.pad(self.sp.W,(0,p,0,p))
        obs = torch.cat((nfm, W, torch.index_select(W, 1, torch.tensor(self.state[0]))),1)
        return obs

class PPO_ActWrapper(ActionWrapper):
    """Wrapper for processing actions defined as next node label."""

    def __init__(self, env):
        super().__init__(env)
        
    def action(self, action):
        """convert action."""
        assert action in self.neighbors[self.state[0]]
        a= self.neighbors[self.state[0]].index(action)
        #print('Node_select action:',action,'Neighbor_index action:',a)
        return a

def CreateEnv():
    world_name='Manhattan3x3_WalkAround'
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
    MAX_NODES=9
    env = PPO_ObsWrapper(env, max_possible_num_nodes = MAX_NODES)        
    env = PPO_ActWrapper(env)        

    return env
#env_fn = lambda : gym.make('GraphWorld-v0',**kwargs)
#env_fn = lambda : gym.make('LunarLander-v2')


env=CreateEnv()

from modules.rl.environments import GraphWorld
#ac_kwargs = dict(hidden_sizes=[64,64], activation=torch.nn.ReLU)
ac_kwargs = dict(
    emb_dim       = 32,
    emb_iter_T    = 3,
    #max_num_nodes = 12,
)
logger_kwargs = dict(output_dir='results/gnn-ppo/spinningup/', exp_name='experiment_name')
from modules.ppo.models_spinup import Struc2VecActorCritic

ppo(env_fn=CreateEnv, actor_critic=Struc2VecActorCritic, ac_kwargs=ac_kwargs, 
        steps_per_epoch=2, 
        epochs=250, 
        clip_ratio=0.1, 
        pi_lr=5e-5,
        vf_lr=5e-5,
        logger_kwargs=logger_kwargs)
#ppo(env_fn=CreateEnv, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=250, logger_kwargs=logger_kwargs)


k=0