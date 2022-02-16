from gym import ObservationWrapper, ActionWrapper, Wrapper
from gym import spaces
import torch
import torch.nn as nn 
import numpy as np
import random

class VarTargetWrapper(Wrapper):
    def __init__(self, env, var_targets):
        # var_targets: list [t1,t2] defines minimum of t1 and maximum of t2 nodes to be selected as target nodes
        super().__init__(env)
        print('Wrapping the env with an variable target wrapper')
        self.min_num_targets = var_targets[0]
        self.max_num_targets = var_targets[1]

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        num_targets = random.randint(self.min_num_targets,self.max_num_targets)
        pool = set(range(self.env.sp.V))
        pool.remove(self.env.state[0])
        assert num_targets <= len(pool)
        new_targets = np.random.choice(list(pool),num_targets,replace=False)
        self.env.redefine_goal_nodes(new_targets)
        return self.env.obs
    
    def render(self, mode=None, fname=None, t_suffix=True, size=None):
        return self.env.render(mode,fname,t_suffix,size)


class PPO_ObsWrapper(ObservationWrapper):
    """Wrapper for stacking nfm|W|reachable nodes."""

    def __init__(self, env, max_possible_num_nodes = 12):
        super().__init__(env)
        assert max_possible_num_nodes >= self.sp.V
        self.V=env.sp.V
        self.max_nodes = max_possible_num_nodes
        self.observation_space= spaces.Box(0., self.sp.U, shape=(self.max_nodes, (self.F+self.max_nodes+1)), dtype=np.float32)
        self.action_space     = spaces.Discrete(self.max_nodes) # all possible nodes 
        print('Wrapping the env with a customized observation definition for GNN integration: nfm-W-reachable_nodes')
        #self.observation_space
        #self.action_space
    
    def getUpositions(self,t=0):
        return self.env.getUpositions(t)
        # upos = []
        # for i,P_path in enumerate(self.u_paths):
        #     p = P_path[-1] if t >= len(P_path) else P_path[t]
        #     upos.append(p)
        # return upos
    
    def action_masks(self):
        m = self.env.action_masks() + [False] * (self.max_nodes - self.V)
        return m

    def render(self, mode=None, fname=None, t_suffix=True, size=None):
        return self.env.render(mode,fname,t_suffix,size)

    def observation(self, observation):
        """convert observation."""
        p = self.max_nodes - self.V
        nfm = nn.functional.pad(self.nfm,(0,0,0,p))
        W = nn.functional.pad(self.sp.W,(0,p,0,p))
        obs = torch.cat((nfm, W, torch.index_select(W, 1, torch.tensor(self.state[0]))),1)
        return obs

class PPO_ActWrapper(ActionWrapper):
    """Wrapper for processing actions defined as next node label."""

    def __init__(self, env):
        super().__init__(env)
        print('Wrapping the env with an action wrapper to redefine action inputs as node labels')

    def render(self, mode=None, fname=None, t_suffix=True, size=None):
        return self.env.render(mode,fname,t_suffix,size)
        
    def action(self, action):
        """convert action."""
        assert action in self.neighbors[self.state[0]]
        a= self.neighbors[self.state[0]].index(action)
        #print('Node_select action:',action,'Neighbor_index action:',a)
        return a
