from gym import ObservationWrapper, ActionWrapper, Wrapper
from gym import spaces
import torch
import torch.nn as nn 
import numpy as np
import random
MAX_NODES=2700
MAX_EDGES=4000

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

class PPO_ObsFlatWrapper(ObservationWrapper):
    """Wrapper for constructing a flattened nfm|edge_list|reachable nodes observation tensor.
    Flattening: nfm (NxF) | edge_list (Ex2) | reachable (N,) | num_nodes (1,) | max_num_nodes (1,) | num_edges (1,) | max_num_edges (1,) | node_dim (1,)
    """
    
    def __init__(self, env, max_possible_num_nodes = 3000, max_possible_num_edges = 4000, obs_mask='None', obs_rate=1, seed=0):
        super().__init__(env)
        assert max_possible_num_nodes >= self.sp.V
        assert obs_mask in ['None','freq','prob','prob_per_u','prob_per_u_test']
        assert (obs_rate >=0 and obs_rate <=1) if obs_mask in ['prob','prob_per_u','None','prob_per_u_test'] else (obs_rate >1e-2 and obs_rate <=1)
        self.obs_mask = obs_mask  # Type of observation masking of pursuit units
        self.obs_rate = int(1/obs_rate) if obs_mask=='freq' else obs_rate  # If observations are masked, either frequency (mask every n) or probability (mask with probability p)
        self.max_possible_num_nodes = max_possible_num_nodes
        self.max_possible_num_edges = max_possible_num_edges
        self.nflat = self.max_possible_num_nodes * (1+self.F) + self.max_possible_num_edges * 2 + 5
        self.observation_space= spaces.Box(0., 10., shape=(self.nflat,), dtype=np.float32)
        self.action_space     = spaces.Discrete(self.max_possible_num_nodes) # all possible nodes 
        print('Wrapping the env with a customized observation definition for GNN integration: flattened nfm-W-reachable_nodes-N-E')
        if obs_mask == 'prob_per_u_test':
            rng = np.random.default_rng(seed)
            num_worlds = len(self.all_worlds)
            self.pre_calculated_masks = rng.integers(2, size=(num_worlds, self.sp.T, self.sp.U), dtype=np.bool)

            
    def getUpositions(self,t=0):
        return self.env.getUpositions(t)
    
    def action_masks(self):
        m = self.env.action_masks() + [False] * (self.max_possible_num_nodes - self.V)
        return m

    def render(self, mode=None, fname=None, t_suffix=True, size=None):
        return self.env.render(mode,fname,t_suffix,size)

    def observation(self, observation):
        """convert observation."""
        # Apply observation masking
        if self.obs_mask != 'None':
            #all_observable=True
            if self.obs_mask == 'freq':
                self.env.u_observable = [True]*self.env.sp.U
                if ((self.global_t) % self.obs_rate != 0):
                    #print('freq criterion: obs set to False')
                    #all_observable=False
                    self.env.u_observable = [False]*self.env.sp.U # for plotting purposes
                    self.nfm[:, self.nfm_calculator.uindx] = 0 # erase all U observations
            elif self.obs_mask == 'prob': # probability of observing all Us at timestep t
                self.env.u_observable = [True]*self.env.sp.U                
                if self.env.global_t > 0:  # first frame always visible
                    p = np.random.rand() 
                    #print('p=',p)
                    if p > self.obs_rate:
                        #print('prob criterion: obs set to False')
                        #all_observable=False
                        self.env.u_observable = [False]*self.env.sp.U # for plotting purposes
                        self.nfm[:, self.nfm_calculator.uindx] = 0 # erase all U observations
            elif self.obs_mask == 'prob_per_u': # probability of observing an individual U at timestep t
                if self.env.global_t > 0:  # first frame always visible
                    mask_u = np.random.rand(self.sp.U) > self.obs_rate
                    self.env.u_observable = list(~mask_u) # for plotting purposes
                    self.mask_units(mask_u) # sets appropriate self.nfm values to 0
            elif self.obs_mask == 'prob_per_u_test':
                if self.env.global_t > 0:  # first frame always visible
                    mask_u = self.pre_calculated_masks[self.current_entry][self.global_t] > self.obs_rate
                    self.env.u_observable = list(~mask_u) # for plotting purposes
                    self.mask_units(mask_u) # sets appropriate self.nfm
            else: assert False    
            #print('\n>> state',self.state,'units observable:',self.u_observable,'units positions:',self.getUpositions(self.local_t))
            assert self.u_observable==self.env.u_observable
            assert self.global_t == self.local_t

        # Convert observation to flat tensor according to init definitions
        pv = self.max_possible_num_nodes - self.sp.V  # padding
        nfm = nn.functional.pad(self.nfm.clone(),(0,0,0,pv)) # pad downward to (max_N, F)
        num_edges = self.sp.EI.shape[1]
        pe = self.max_possible_num_edges - num_edges
        pygei = nn.functional.pad(self.sp.EI.clone(),(0,pe)) # pad rightward to (2,MAX_EDGES)
        reachable = torch.index_select(self.sp.W, 1, torch.tensor(self.state[0])).clone()
        reachable = nn.functional.pad(reachable,(0,0,0,pv)) # pad downward to (max_N, 1)
        self.obs = torch.cat((torch.flatten(nfm), 
                         torch.flatten(pygei), 
                         torch.flatten(reachable),
                         torch.tensor([self.sp.V, self.max_possible_num_nodes, num_edges, self.max_possible_num_edges, self.F]))
                        ,dim=0)

        # # TEST
        # # deserialize single obs (:,)
        # num_nodes, max_nodes, num_edge, max_edges, F = obs[-5:].to(torch.int64).tolist()
        # nf,py,re,_ = torch.split(obs,(F*max_nodes, 2*max_edges, max_nodes, 5),dim=0)
        # nf=nf.reshape(max_nodes,-1)[:num_nodes]
        # py=py.reshape(2,-1)[:,:num_edges].to(torch.int64)
        # re=re.reshape(-1,1)[:num_nodes].to(torch.int64)
        # assert nf.shape[1]==F
        # assert torch.allclose(self.nfm,nf)
        # assert torch.allclose(self.sp.EI,py)
        return self.obs

class PPO_ObsDictWrapper(ObservationWrapper):
    """Wrapper for dict of nfm|W|reachable|pyg_data"""

    def __init__(self, env, max_possible_num_nodes = 12, max_possible_num_edges = 300):
        super().__init__(env)
        assert max_possible_num_nodes >= self.sp.V
        #self.V=env.sp.V
        self.max_nodes = max_possible_num_nodes
        self.max_edges = max_possible_num_edges
        dspaces  = {
            'nfm':      spaces.Box(0., self.sp.V, shape=(self.max_nodes, self.F), dtype=np.float32),
            'W':        spaces.Box(0., 1., shape=(self.max_nodes, self.max_nodes), dtype=np.float32),
            'reachable_nodes':spaces.Box(0., 1., shape=(self.max_nodes, 1), dtype=np.float32),
            'pygx':     spaces.Box(0., self.sp.V, shape=(self.max_nodes, self.F), dtype=np.float32),
            'pygei':    spaces.Box(0., self.max_nodes, shape=(2,self.max_edges), dtype=np.float32),
            'num_nodes':spaces.Box(0., self.max_nodes, shape=(1,), dtype=np.float32),
            'num_edges':spaces.Box(0., self.max_edges, shape=(1,), dtype=np.float32),
        }

        self.observation_space= spaces.Dict(dspaces)
        self.action_space     = spaces.Discrete(self.max_nodes) # all possible nodes 
        print('Wrapping the env with a observation dict space')
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
        m = self.env.action_masks() + [False] * (self.max_nodes - self.sp.V)
        return m

    def render(self, mode=None, fname=None, t_suffix=True, size=None):
        return self.env.render(mode,fname,t_suffix,size)

    def observation(self, observation):
        """convert observation."""
        p = self.max_nodes - self.sp.V
        nfm = nn.functional.pad(self.nfm.clone(),(0,0,0,p))
        W = nn.functional.pad(self.sp.W.clone(),(0,p,0,p))
        reachable = torch.index_select(W, 1, torch.tensor(self.state[0])).clone()
        pygx = self.nfm.clone()
        pygx = nn.functional.pad(pygx,(0,0,0,p)) # pad downward to (MAX_NODES,F)
        pygei = self.sp.EI.clone()        
        pygei = nn.functional.pad(pygei,(0, self.max_edges - self.sp.EI.shape[1])) # pad rightward to (2,MAX_EDGES)
        #obs = torch.cat((nfm, W, torch.index_select(W, 1, torch.tensor(self.state[0]))),1)
        obs = {
            'nfm':      nfm,
            'W':        W,
            'reachable_nodes':reachable,
            'pygx':     pygx,
            'pygei':    pygei,
            'num_nodes':self.sp.V,
            'num_edges':self.sp.EI.shape[1],
        }
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

# class MaskVelocityWrapper(gym.ObservationWrapper):
#     """
#     Gym environment observation wrapper used to mask velocity terms in
#     observations. The intention is the make the MDP partially observatiable.
#     """
#     def __init__(self, env):
#         super(MaskVelocityWrapper, self).__init__(env)
#         if ENV == "CartPole-v1":
#             self.mask = np.array([1., 0., 1., 0.])
#         elif ENV == "Pendulum-v0":
#             self.mask = np.array([1., 1., 0.])
#         elif ENV == "LunarLander-v2":
#             self.mask = np.array([1., 1., 0., 0., 1., 0., 1., 1,])
#         elif ENV == "LunarLanderContinuous-v2":
#             self.mask = np.array([1., 1., 0., 0., 1., 0., 1., 1,])
#         else:
#             raise NotImplementedError

#     def observation(self, observation):
#         return  observation * self.mask
