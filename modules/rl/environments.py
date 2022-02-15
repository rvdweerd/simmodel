#from os import stat_result
from typing import List, Optional

from rdflib import Graph
import modules.sim.simdata_utils as su
import random 
#import time
from modules.rl.rl_plotting import PlotAgentsOnGraph, PlotAgentsOnGraph_, PlotEPathOnGraph_, PlotEUPathsOnGraph_
import numpy as np
#import matplotlib.pyplot as plt
import torch
import gym
from gym import spaces
from gym import register
import copy
import networkx as nx
from modules.gnn.nfm_gen import NFM_ec_t, NFM_ev_t, NFM_ev_ec_t

def is_edge_crossing(old_Upositions, new_Upositions, prev_node, next_node):
    if prev_node in new_Upositions and next_node in old_Upositions:
        for i,j in zip(old_Upositions,new_Upositions):
            if prev_node == j and next_node == i:
                return True
    return False

class GraphWorld(gym.Env):
    """"""
    def __init__(self, config, optimization_method='static', fixed_initial_positions=None, state_representation='etUt', state_encoding='nodes'):
        super(GraphWorld,self).__init__()
        self.type                   ='GraphWorld'
        self.sp                     = su.DefineSimParameters(config)
        self.capture_on_edges       = False
        self.optimization           = optimization_method
        self.fixed_initial_positions= fixed_initial_positions
        self.state_representation   = state_representation
        self.state_encoding         = state_encoding
        self.state_encoding_dim, self.state_chunks, self.state_len = su.GetStateEncodingDimension(state_representation, self.sp.V, self.sp.U)
        self.render_fileprefix      = 'test_scenario_t='
        # Load relevant pre-saved optimization runs for the U trajectories
        dirname                     = su.make_result_directory(self.sp, optimization_method)
        register_coords, databank_coords, iratios = su.LoadDatafile(dirname)
        self.register, self.databank, self.iratios = self._ConvertDataFile(register_coords, databank_coords, iratios) 
        
        # Create a world_pool list of indices of pre-saved initial conditions and their rollouts of U positions
        self.all_worlds             = [ind for k,ind in self.register['labels'].items()]
        self.world_pool             = su.GetWorldPool(self.all_worlds, fixed_initial_positions, self.register)
        self.world_pool_solvable    = {} # filled in later uses when available
        self._encode                = self._encode_nodes if state_encoding == 'nodes' else self._encode_tensor if state_encoding == 'tensors' else self._encode_nfm
        if state_encoding not in ['nodes', 'tensors', 'nfm']: assert False

        # Dynamics parameters
        self.current_entry          = 0    # which entry in the world pool is active
        self.u_paths                = []
        self.e_path                 = []   # nodes visited by e during rollout
        self.u_paths_taken          = [ [] for i in range(self.sp.U)]
        self.iratio                 = 0
        self.state0                 = ()
        self.state                  = ()   # current internal state in node labels: (e,U1,U2,...)
        self.obs                    = None # observation, encoded state based on state representation (et,etUt,etU0 or etUte0U0)
        self.global_t               = 0
        self.local_t                = 0
        self.max_timesteps          = self.sp.T
        self.neighbors, self.in_degree, self.max_indegree, self.out_degree, self.max_outdegree = su.GetGraphData(self.sp)

        # Define NFM: Node Feature Matrix, (VxF) with V number of nodes, F number of features
        self.nfm_calculator=NFM_ev_ec_t()
        if self.state_encoding=='nfm':
            self.nfm_calculator.init(self)

        # Gym objects
        if state_encoding == 'nfm':
            self.observation_space = spaces.Box(0., self.sp.U, shape=(self.sp.V, (self.F+self.sp.V+1)), dtype=np.float32)
            self.action_space = spaces.Discrete(self.sp.V) # all possible nodes 
        else:
            self.observation_space = spaces.Box(0., self.sp.U, shape=(self.state_encoding_dim,), dtype=np.float32)
            self.action_space = spaces.Discrete(self.max_outdegree)# + int(config['make_reflextive']))
        self.metadata = {'render.modes':['human']}
        self.max_episode_length = self.max_timesteps

        self.reset()
    
    def redefine_nfm(self, nfm_function=None):
        if nfm_function == None:
            return
        #assert  self.state_encoding=='nfm'
        self.nfm_calculator = nfm_function
        self.nfm_calculator.init(self)
        self.reset()

    def get_custom_nfm(self, epath, targetnodes, upaths):
        return self.nfm_calculator.get_custom_nfm(self, epath, targetnodes, upaths)

    def redefine_goal_nodes(self, goal_nodes):
        self.sp.target_nodes=goal_nodes
        self.sp.CalculateShortestPath()
        self.nfm_calculator.init(self)

    def redefine_graph_structure(self, W, in_nodeid2coord, new_nodeids=False):
        # W:          adjacency matrix (numpy array), based on some node ordering
        # node2coord: mapping from node number to 2d coords
        #assert W.shape[0]==self.sp.V
        if new_nodeids:
            self.sp.labels={}
            self.sp.pos={}
            for k,v in in_nodeid2coord.items():
                self.sp.labels[v]=k
                self.sp.pos[v]=v
        target_coords=[self.sp.labels2coord[n] for n in self.sp.target_nodes]
        self.sp.nodeids2labels={}
        self.sp.coord2nodeid={}
        self.sp.coord2labels={}
        self.sp.labels2coord={}
        self.sp.labels2nodeids={}
        
        self.sp.nodeid2coord=in_nodeid2coord
        for id,coord in in_nodeid2coord.items():
            label=self.sp.labels[coord]
            self.sp.nodeids2labels[id]   =label
            self.sp.coord2nodeid[coord]  =id
            self.sp.coord2labels[coord]  =label
            self.sp.labels2nodeids[label]=id
            self.sp.labels2coord[label]  =coord

        if W.shape[0] != self.sp.V:
            self.sp.V = W.shape[0]
            self.sp.N = None
        self.register = {'coords': {}, 'labels': {}}
        self.databank = {'coords': {}, 'labels': {}}
        self.iratios = []
        self.all_worlds = []
        self.world_pool = []

        self.sp.G = nx.from_numpy_matrix(W, create_using=nx.DiGraph())
        self.sp.G = nx.relabel_nodes(self.sp.G, in_nodeid2coord)
     
        self.sp.target_nodes=[]
        for tc in target_coords:
            if tc in self.sp.coord2labels.keys():
                self.sp.target_nodes.append(self.sp.coord2labels[tc])
        
        self.sp.start_escape_route_node = self.sp.coord2labels[self.sp.start_escape_route]
        self.state_encoding_dim, self.state_chunks, self.state_len = su.GetStateEncodingDimension(self.state_representation, self.sp.V, self.sp.U)      
        self.sp.W = torch.tensor(W, dtype=torch.float32)
        self.neighbors, self.in_degree, self.max_indegree, self.out_degree, self.max_outdegree = su.GetGraphData(self.sp)
        self.current_entry          = 0    # which entry in the world pool is active
        self.u_paths                = []
        self.u_paths_taken          = [ [] for i in range(self.sp.U)]
        self.iratio                 = 0
        self.state0                 = ()
        self.state                  = ()   # current internal state in node labels: (e,U1,U2,...)
        self.global_t               = 0
        self.local_t                = 0     
        self.observation_space = spaces.Box(0., self.sp.U, shape=(self.state_encoding_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.max_outdegree)
        self.nfm_calculator.init(self)
        self.sp.CalculateShortestPath()
        self.reset()
        self.state=(self.sp.coord2labels[self.sp.start_escape_route],)

    def reload_unit_paths(self, register_coords, databank_coords, iratios):
        self.register, self.databank, self.iratios = self._ConvertDataFile(register_coords, databank_coords, iratios) 
        
        # Create a world_pool list of indices of pre-saved initial conditions and their rollouts of U positions
        self.all_worlds             = [ind for k,ind in self.register['labels'].items()]
        self.world_pool             = su.GetWorldPool(self.all_worlds, self.fixed_initial_positions, self.register)
        self.reset()

    def _encode_nodes(self, s):
        # used to return a tuple of node labels of E and U positions as observation
        return s

    def _encode_tensor(self, s):
        # used to return a one-hot-coded tensor of E and Upositions as observation
        return self._state2vec_packed(s)

    def _encode_nfm(self, s):
        # used to return the node feature matrix as observation
        return self.nfm

    def _state2np_mat(self,s):
        if self.state_representation == 'etUt':
            out = np.zeros((self.sp.U+1,self.sp.V))
            out[0,s[0]] = 1
            upos = list(s[1:])
            #upos.sort()
            for i,u in enumerate(upos):
                out[i+1,u] = 1
        else:
            return NotImplementedError
        return out.flatten()

    def _state2vec(self, state, sort_units=False):
        out=np.zeros(self.sp.V * len(state)) # num_nodes x number of objects to be represented
        if sort_units:
            return NotImplementedError
        else:
            for i, pos in enumerate(state):
                out[i*self.sp.V + pos] = 1
        return out

    def _state2vec_packed(self, state, sort_units=False):
        out=np.zeros(self.sp.V * len(self.state_chunks)) # 
        if sort_units:
            return NotImplementedError
        else:
            for i, chunk in enumerate(self.state_chunks):
                for pos in chunk:
                    if pos < len(state): #CHECK!! DONE TO DEAL WITH EMPTY U positions
                        out[i*self.sp.V + state[pos]] += 1
        return out.astype(np.int64)

    def _ConvertDataFile(self, register_coords, databank_coords, iratios):
        register_labels={}
        databank_labels=[]
        # Convert register
        for key, idx in register_coords.items():
            newkey = self._to_state_from_coords(key[0], list(key[1:]))
            if newkey in register_labels:
                assert False
            register_labels[newkey] = idx
        # Convert databank
        for d in databank_coords:
            start_escape_route_coords = d['start_escape_route']
            start_escape_route_nodeid = self.sp.coord2labels[start_escape_route_coords]

            start_units_coords = d['start_units']
            start_units_labels = []
            for u in start_units_coords:
                start_units_labels.append(self.sp.coord2labels[u])

            paths_coords = d['paths']
            paths_labels = []
            for path_coords in paths_coords:
                path_labels=[]
                for step in path_coords:
                    path_labels.append(self.sp.coord2labels[step])
                paths_labels.append(path_labels)
            databank_labels.append({
                    'start_escape_route': start_escape_route_nodeid,\
                    'start_units': start_units_labels,\
                    'paths': paths_labels,
                })
        register_returned = {'coords': register_coords, 'labels': register_labels}
        databank_returned = {'coords': databank_coords, 'labels': databank_labels}
        return register_returned, databank_returned, iratios

    def _getUpositions(self,t=0):
        upos = []
        for i,P_path in enumerate(self.u_paths):
            p = P_path[-1] if t >= len(P_path) else P_path[t]
            upos.append(p)
        return upos

    def getUpositions(self, t=0):
        return self._getUpositions(t=0)        

    def _availableActionsInCurrentState(self):
        return self.neighbors[self.state[0]]

    def _to_coords_from_state(self):
        e_init_coord = self.sp.labels2coord[self.state[0]]
        u_init_coords = []
        for u in self.state[1:]:
            u_init_coords.append(self.sp.labels2coord[u])
        u_init_coords.sort()
        return tuple([e_init_coord] + u_init_coords)

    def _to_state_from_coords(self, e_init_coord, u_init_coords):
        e_init_label = self.sp.coord2labels[e_init_coord]
        u_init_labels = []
        for u in u_init_coords:
            u_init_labels.append(self.sp.coord2labels[u])
        u_init_labels.sort()
        return tuple([e_init_label] + u_init_labels)

    def _to_state(self, e_init_label, u_init_labels):
        u_init_labels.sort()
        return tuple([e_init_label] + u_init_labels)

    def _remove_world_pool(self):
        assert len(self.world_pool)>0
        assert self.world_pool == self.all_worlds
        self.sp.U_=self.sp.U
        self.sp.U=0
        self.u_paths_taken =[]
        self.world_pool=[]
        self.state_encoding_dim, self.state_chunks, self.state_len = su.GetStateEncodingDimension(self.state_representation, self.sp.V, self.sp.U)      
        #self.current_entry=-1
        self.iratio=-1
        self.u_paths=[]
        self.reset(self.current_entry)

    def _restore_world_pool(self):
        assert self.world_pool==[]
        self.sp.U=self.sp.U_
        self.u_paths_taken = [ [] for i in range(self.sp.U)]
        self.state_encoding_dim, self.state_chunks, self.state_len = su.GetStateEncodingDimension(self.state_representation, self.sp.V, self.sp.U)      
        self.world_pool = self.all_worlds
        self.reset(self.current_entry)

    def reset(self, entry=None):
        # Reset time
        self.global_t = 0
        self.local_t = 0 # used if optimization is dynamic; lookup time for new paths is set to 0 after each step
        if len(self.world_pool)==0:
            #self.current_entry=-1
            self.iratio=-1
            self.state=self._to_state((self.sp.coord2labels[self.sp.start_escape_route]),[])
            self.state0 =self.state
            pass
            #return
        else:
            if entry is not None:
                assert self.databank['labels'][entry]['start_escape_route'] not in self.sp.target_nodes
            if entry==None:
                valid=False
                count=0
                while not valid:    
                    entry = random.choice(self.world_pool)
                    count+=1
                    assert count < 1000
                    if self.databank['labels'][entry]['start_escape_route'] not in self.sp.target_nodes:
                        valid=True
            
            self.current_entry=entry
            data_sample = self.databank['labels'][entry]
            self.iratio = self.iratios[entry]
            
            # Reassign initial positions
            if data_sample['start_escape_route'] != self.sp.start_escape_route_node:
                self.sp.start_escape_route = self.databank['coords'][entry]['start_escape_route']
                self.sp.start_escape_route_node = data_sample['start_escape_route']
                self.sp.CalculateShortestPath()

            # Assign initial state
            e_init_labels = data_sample['start_escape_route'] # (e0)
            u_init_labels = data_sample['start_units'] # [(u1),(u2), ...]
            self.u_paths  = data_sample['paths']
            self.state    = self._to_state(e_init_labels,u_init_labels)
            self.state0   = self.state
        
        self.e_path = [ self.state[0] ]
        new_Upositions = self._getUpositions(self.local_t)
        for i,u in enumerate(new_Upositions):
            self.u_paths_taken[i] = [u]
    
        # Initialize graph feature matrix
        if self.state_encoding=='nfm':
            self.nfm_calculator.reset(self)

        # Return initial state in appropriate form
        if self.state_representation == 'etUt':
            self.obs = self._encode(self.state)
        elif self.state_representation == 'et':
            self.obs = self._encode((self.state[0],))
        elif self.state_representation == 'etUte0U0':
            self.obs = self._encode(self.state+self.state0)
        elif self.state_representation == 'ete0U0':
            self.obs = self._encode(tuple([self.state[0]])+self.state0)
        else:
            assert False
        return self.obs

    def action_masks(self) -> List[bool]:
        reachable=np.zeros(self.sp.V).astype(np.bool)
        idx=np.array(self.neighbors[self.state[0]]).astype(int)
        reachable[idx]=True
        return list(reachable)

    def step(self, action_idx):
        # Take a step
        assert self.state[0] not in self.sp.target_nodes # can only happen if start position is on a target node

        info = {'Solved':False, 'Captured':False, 'u_positions':self.state[1:], 'Misc':None}
        prev_node=self.state[0]
        if action_idx >= len(self.neighbors[self.state[0]]): # account for invalid action choices
            next_node = self.state[0]
            reward=-2.
            info['Misc']='action_out_of_bounds'
            done=False
        else:          
            next_node = self.neighbors[self.state[0]][action_idx]
            reward = -1.
            if next_node == self.state[0]:
               reward = -1.5
        self.global_t += 1
        self.local_t  += 1
        
        # Update unit positions
        old_Upositions = self._getUpositions(self.local_t-1)
        new_Upositions = self._getUpositions(self.local_t) # uses local time: u_paths may have been updated from last state if sim is dynamic
        new_Upositions_sorted = list(np.sort(new_Upositions))

        self.state = tuple([next_node] + new_Upositions_sorted)
        self.e_path.append(self.state[0])
        for i,u in enumerate(new_Upositions):
            self.u_paths_taken[i].append(u)

        # Check termination conditions
        done = False
        if self.global_t >= self.max_timesteps: 
            # max timesteps reached
            done=True
        if next_node in new_Upositions: 
            # captured: escaper shares node with pursuit unit
            done=True
            reward += -10
            info['Captured']=True
        elif self.capture_on_edges and is_edge_crossing(old_Upositions, new_Upositions, prev_node, next_node):
            # captured: escaper and pursuit unit have crossed on an edge
            done=True
            reward += -10
            info['Captured']=True
        elif next_node in self.sp.target_nodes: 
            # goal reached
            done = True
            reward += +10
            info['Solved']=True
        if self.optimization == 'dynamic' and not done: 
            # update optimization paths for units
            self.u_paths = self.databank['labels'][self.register['labels'][self.state]]['paths']
            if len(self.u_paths) < 2:
                assert False
            self.local_t = 0

        # Update feature matrix
        if self.state_encoding=='nfm':
            self.nfm_calculator.update(self)

        # Return s',r',done,info (new state in appropriate form)
        if self.state_representation == 'etUt':
            self.obs = self._encode(self.state)#, reward, done, info
        elif self.state_representation == 'et':
            self.obs = self._encode((self.state[0],))#, reward, done, info
        elif self.state_representation == 'etUte0U0':
            self.obs = self._encode(self.state+self.state0)#, reward, done, info
        elif self.state_representation == 'ete0U0':
            self.obs = self._encode(tuple([self.state[0]])+self.state0)#, reward, done, info
        else:
            assert False
        return self.obs, reward, done, info

    def render(self, mode=None, fname=None, t_suffix=True, size=None):#file_name=None):
        e = self.state[0]
        p = self.state[1:]
        if fname == None:
            #file_name=self.render_fileprefix+'_t='+str(self.global_t)
            file_name=None
        elif t_suffix:
            file_name = fname+'_t='+str(self.global_t)
        else: file_name = fname
        if size == None:
            if self.sp.V < 1:
                size='large'
            else:
                size='small'
        plot = PlotAgentsOnGraph_(self.sp, e, p, self.global_t, fig_show=False, fig_save=True, filename=file_name, goal_reached=(self.state[0] in self.sp.target_nodes), size=size)
        return plot
    
    def render_epath(self, fname=None, t_suffix=True, size=None):
        p = self.state[1:]
        if fname == None:
            #file_name=self.render_fileprefix+'_t='+str(self.global_t)
            file_name=None
        elif t_suffix:
            file_name = fname+'_t='+str(self.global_t)
        else: file_name = fname
        if size == None:
            if self.sp.V < 1:
                size='large'
            else:
                size='small'
        plot = PlotEPathOnGraph_(self.sp, self.e_path, p, fig_show=False, fig_save=True, filename=file_name, goal_reached=(self.state[0] in self.sp.target_nodes), size=size)
        return plot

    def render_eupaths(self, mode=None, fname=None, t_suffix=True, size=None, last_step_only=False):
        if fname == None:
            #file_name=self.render_fileprefix+'_t='+str(self.global_t)
            file_name=None
        elif t_suffix:
            file_name = fname+'_t='+str(self.global_t)
        else: file_name = fname
        if size == None:
            if self.sp.V < 1:
                size='large'
            else:
                size='small'
        plot = PlotEUPathsOnGraph_(self.sp, self.e_path, self.u_paths_taken, fig_show=False, fig_save=True, filename=file_name, goal_reached=(self.state[0] in self.sp.target_nodes), size=size, last_step_only=last_step_only)
        return plot



register(
    id='GraphWorld-v0',
    entry_point='modules.rl.environments:GraphWorld'
)

class VariableTargetGraphWorld(GraphWorld):
    def __init__(self, config, optimization_method='static', fixed_initial_positions=None, state_representation='etUt', state_encoding='nodes', target_range=[1,1]):
        self.min_num_targets=target_range[0]
        self.max_num_targets=target_range[1]
        super(VariableTargetGraphWorld,self).__init__(config, optimization_method='static', fixed_initial_positions=None, state_representation='etUt', state_encoding='nodes')
    def reset(self, entry=None):
        super(VariableTargetGraphWorld, self).reset(entry)
        num_targets = random.randint(self.min_num_targets,self.max_num_targets)
        pool = set(range(self.sp.V))
        pool.remove(self.state[0])
        assert num_targets <= len(pool)
        new_targets = np.random.choice(list(pool),num_targets,replace=False)
        self.redefine_goal_nodes(new_targets)

class SuperEnv(gym.Env):
    def __init__(self, all_env, hashint2env, max_possible_num_nodes = 9, probs=None):
        super(SuperEnv,self).__init__()
        self.hashint2env = hashint2env
        self.max_num_nodes = max_possible_num_nodes
        self.all_env = all_env
        self.num_env = len(all_env)
        if probs == None:
            self.probs = np.ones(self.num_env,dtype=np.float)/self.num_env
        else:
            assert len(probs) == self.num_env
            self.probs = np.array(probs,dtype=np.float)
            assert (self.probs<0).sum() == 0
            if abs(self.probs.sum() - 1) > 1e-5:
                print('probs dont sum up to 1, rescaling')
                self.probs = self.probs / (self.probs).sum()
        self.reset()
    
    def reset(self, hashint=None, entry=None):
        if hashint == None:
            #self.current_env_nr = random.randint(0, self.num_env-1)
            self.current_env_nr = int(np.random.choice(self.num_env, p=self.probs))
        else:
            self.current_env_nr = self.hashint2env[hashint]
        self.env = self.all_env[self.current_env_nr]
        obs = self.env.reset(entry=entry)

        self._updateGraphLevelData()
        self._updateInstanceLevelData()
        self._updateStepLevelData()
        return obs

    def step(self, action):
        s,r,d,i = self.env.step(action)
        self._updateStepLevelData()
        return s,r,d,i
    
    def get_custom_nfm(self, epath, targetnodes, upaths):
        return self.env.get_custom_nfm(epath, targetnodes, upaths)

    def render(self, mode=None, fname=None, t_suffix=True, size=None):
        return self.env.render(mode,fname,t_suffix,size)

    def render_epath(self, fname=None, t_suffix=True, size=None):
        return self.env.render_epath(fname, t_suffix, size)

    def render_eupaths(self, mode=None, fname=None, t_suffix=True, size=None, last_step_only=False):
        self.env.render_eupaths(mode, fname, t_suffix, size, last_step_only)

    def action_masks(self):
        m = self.env.action_masks()# + [False] * (self.max_num_nodes - self.sp.V)
        return m

    def _getUpositions(self, t=0):
        return self.env._getUpositions(t)

    def _updateGraphLevelData(self):
        self.sp = self.env.sp
        self.F = self.env.F
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.neighbors = self.env.neighbors
        self.state_encoding = self.env.state_encoding

    def _updateInstanceLevelData(self):
        self.current_entry = self.env.current_entry
        self.u_paths = self.env.u_paths

    
    def _updateStepLevelData(self):
        self.global_t = self.env.global_t
        self.local_t = self.env.local_t        
        self.state = self.env.state
        self.obs = self.env.obs
        self.nfm = self.env.nfm

