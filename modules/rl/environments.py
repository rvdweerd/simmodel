#from os import stat_result
import modules.sim.simdata_utils as su
import random 
#import time
from modules.rl.rl_plotting import PlotAgentsOnGraph, PlotAgentsOnGraph_
import numpy as np
#import matplotlib.pyplot as plt
import gym
from gym import spaces
from gym import register
import copy
import networkx as nx
from modules.gnn.nfm_gen import BasicNFM

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
        self.state_encoding_dim, self.state_chunks, self.state_len = su.GetStateEncodingDimension(state_representation, self.sp.V, self.sp.U)
        self.render_fileprefix      = 'test_scenario_t='
        # Load relevant pre-saved optimization runs for the U trajectories
        dirname                     = su.make_result_directory(self.sp, optimization_method)
        register_coords, databank_coords, iratios = su.LoadDatafile(dirname)
        self.register, self.databank, self.iratios = self._ConvertDataFile(register_coords, databank_coords, iratios) 
        
        # Create a world_pool list of indices of pre-saved initial conditions and their rollouts of U positions
        self.all_worlds             = [ind for k,ind in self.register['labels'].items()]
        self.world_pool             = su.GetWorldPool(self.all_worlds, fixed_initial_positions, self.register)
        self._encode                = self._encode_nodes if state_encoding == 'nodes' else self._encode_tensor
        if state_encoding not in ['nodes', 'tensors']: assert False

        # Dynamics parameters
        self.current_entry          = 0    # which entry in the world pool is active
        self.u_paths                = []
        self.iratio                 = 0
        self.state0                 = ()
        self.state                  = ()   # current internal state in node labels: (e,U1,U2,...)
        self.obs                    = None # observation, encoded state based on state representation (et,etUt,etU0 or etUte0U0)
        self.global_t               = 0
        self.local_t                = 0
        self.max_timesteps          = self.sp.T
        self.neighbors, self.in_degree, self.max_indegree, self.out_degree, self.max_outdegree = su.GetGraphData(self.sp)

        # Gym objects
        #self.observation_space = spaces.Discrete(self.sp.V) if state_encoding == 'nodes' else spaces.MultiBinary(self.state_encoding_dim)
        self.observation_space = spaces.Box(0., self.sp.U, shape=(self.state_encoding_dim,), dtype=np.float32)
        #self.observation_space = spaces.Tuple([spaces.Discrete(self.sp.V) for i in range(self.state_len)])             
        # @property
        # def action_space(self):
        #     return spaces.Discrete(self.out_degree[self.state[0]])
        self.action_space = spaces.Discrete(self.max_outdegree)
        self.metadata = {'render.modes':['human']}
        self.max_episode_length = self.max_timesteps

        # Graph feature matrix, (FxV) with F number of features, V number of nodes
        # 0  [.] node number
        # 1  [.] 1 if target node, 0 otherwise 
        # 2  [.] # of units present at node at current time
        # 3  [.] ## off ## 1 if node previously visited by unit
        # 4  [.] ## off ## 1 if node previously visited by escaper
        # 5  [.] ## off ## distance from nearest target node
        # 6  [.] ...
        self.nfm_calculator=BasicNFM()
        self.nfm_calculator.init(self)
        # self.F = 3
        # self.gfm0 = np.zeros((self.sp.V,self.F))
        # self.gfm0[:,0] = np.array([i for i in range(self.sp.V)])
        # self.gfm0[np.array(list(self.sp.target_nodes)),1]=1 # set target nodes, fixed for the given graph
        # self.gfm  = copy.deepcopy(self.gfm0)
        # self.reset()

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
        self.u_paths=[]

        self.sp.G = nx.from_numpy_matrix(W, create_using=nx.DiGraph())
        self.sp.G = nx.relabel_nodes(self.sp.G, in_nodeid2coord)
        #self.sp.nodeid2coord = dict( (i, n) for i,n in enumerate(self.sp.G.nodes()) )
        # CHECK IF self.sp.target_nodes are still consistent
        #self.sp.coord2nodeid = dict( (n, i) for i,n in enumerate(self.sp.G.nodes()) )
        #self.sp.coord2labels = self.sp.labels
        #self.sp.labels2coord = dict( (v,k) for k,v in self.sp.coord2labels.items())
        #for coord, nodeid in self.sp.coord2nodeid.items():
        #    label = self.sp.coord2labels[coord]
        #    self.sp.labels2nodeids[label]=nodeid
        #    self.sp.nodeids2labels[nodeid]=label
        
        self.sp.target_nodes=[]
        for tc in target_coords:
            if tc in self.sp.coord2labels.keys():
                self.sp.target_nodes.append(self.sp.coord2labels[tc])
        
        self.state_encoding_dim, self.state_chunks, self.state_len = su.GetStateEncodingDimension(self.state_representation, self.sp.V, self.sp.U)      
        self.sp.W = W
        self.neighbors, self.in_degree, self.max_indegree, self.out_degree, self.max_outdegree = su.GetGraphData(self.sp)
        self.current_entry          = 0    # which entry in the world pool is active
        self.u_paths                = []
        self.iratio                 = 0
        self.state0                 = ()
        self.state                  = ()   # current internal state in node labels: (e,U1,U2,...)
        self.global_t               = 0
        self.local_t                = 0     
        self.observation_space = spaces.Box(0., self.sp.U, shape=(self.state_encoding_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.max_outdegree)
        self.nfm_calculator.init(self)
        self.reset()
        self.state=(self.sp.coord2labels[self.sp.start_escape_route],)

    def reload_unit_paths(self, register_coords, databank_coords, iratios):
        self.register, self.databank, self.iratios = self._ConvertDataFile(register_coords, databank_coords, iratios) 
        
        # Create a world_pool list of indices of pre-saved initial conditions and their rollouts of U positions
        self.all_worlds             = [ind for k,ind in self.register['labels'].items()]
        self.world_pool             = su.GetWorldPool(self.all_worlds, self.fixed_initial_positions, self.register)
        self.reset()

    def _encode_nodes(self, s):
        return s

    def _encode_tensor(self, s):
        return self._state2vec_packed(s)

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

    def _availableActionsInCurrentState(self):
        return self.neighbors[self.state[0]]

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

    def reset(self, entry=None):
        # Reset time
        if len(self.world_pool)==0:
            return
        self.global_t = 0
        self.local_t = 0 # used if optimization is dynamic; lookup time for new paths is set to 0 after each step
        if entry==None:
            entry = random.choice(self.world_pool) 
        self.current_entry=entry
        data_sample = self.databank['labels'][entry]
        self.iratio = self.iratios[entry]
        
        # Assign initial state
        e_init_labels = data_sample['start_escape_route'] # (e0)
        u_init_labels = data_sample['start_units'] # [(u1),(u2), ...]
        self.u_paths  = data_sample['paths']
        self.state    = self._to_state(e_init_labels,u_init_labels)
        self.state0   = self.state
        
        # Initialize graph feature matrix
        self.nfm_calculator.reset(self)
        # self.gfm = copy.deepcopy(self.gfm0)
        # for u_index, u in enumerate(self.state[1:]): 
        #     self.gfm[u,2]+=1         # f2: current presence of units
        #     #self.gfm[u,3]=1         # f3: node previously visited by any unit
        # for p in self.u_paths:
        #     if self.local_t >= len(p)-1:
        #         self.gfm[p[-1],2] += 10
        #self.gfm[self.state[0],4]=1 # f4: node previously visited by escaper

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

    def step(self, action_idx):
        # Take a step
        info = {'Captured':False, 'u_positions':self.state[1:], 'Misc':None}
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
        
        # Check termination conditions
        done = False
        if self.global_t >= self.max_timesteps: # max timesteps reached
            done=True
        if next_node in new_Upositions: # captured
            done=True
            reward += -10
            info['Captured']=True
        elif self.capture_on_edges and is_edge_crossing(old_Upositions, new_Upositions, prev_node, next_node):
            done=True
            reward += -10
            info['Captured']=True
        elif next_node in self.sp.target_nodes: # goal reached
            done = True
            reward += +10 
        if self.optimization == 'dynamic' and not done: # update optimization paths for units
            self.u_paths = self.databank['labels'][self.register['labels'][self.state]]['paths']
            if len(self.u_paths) < 2:
                assert False
            self.local_t = 0

        # Update feature matrix
        self.nfm_calculator.update(self)
        # self.gfm[:,2]=0
        # for u_index, u in enumerate(self.state[1:]): 
        #     self.gfm[u,2]+=1         # f2: current presence of units
        #     #self.gfm[u,2]=1         # f3: node previously visited by any unit
        # for p in self.u_paths:
        #     if self.local_t >= len(p)-1:
        #         self.gfm[p[-1],2] += 10
        #self.gfm[self.state[0],4]=1 # f4: node previously visited by escaper

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

    def render(self, mode=None, fname=None):#file_name=None):
        e = self.state[0]
        p=self.state[1:]
        if fname == None:
            #file_name=self.render_fileprefix+'_t='+str(self.global_t)
            file_name=None
        else:
            file_name = fname+'_t='+str(self.global_t)
        plot=PlotAgentsOnGraph_(self.sp, e, p, self.global_t, fig_show=False, fig_save=True, filename=file_name)
        return plot

register(
    id='GraphWorld-v0',
    entry_point='modules.rl.environments:GraphWorld'
)

# class GraphWorldFromDatabank(GraphWorld):
#     def __init__(self, config, env_data, optimization_method='static', fixed_initial_positions=None, state_representation='etUt', state_encoding='nodes'):
#         #super().__init__('EpsGreedy')
#         W_           =env_data['W']
#         #hashint      =env_data['hashint']
#         #databank_full=env_data['databank_full']
        
#         self.type                   ='GraphWorld'
#         self.sp                     = su.DefineSimParameters(config)
#         self.optimization           = optimization_method
#         self.fixed_initial_positions= fixed_initial_positions
#         self.state_representation   = state_representation
#         self.state_encoding_dim, self.state_chunks, self.state_len = su.GetStateEncodingDimension(state_representation, self.sp.V, self.sp.U)
#         self.render_fileprefix      = 'test_scenario_t='
        
#         # Load relevant optimization runs for the U trajectories from databank
#         register_coords = env_data['register']
#         databank_coords = env_data['databank']
#         iratios  = env_data['iratios']
#         self.register, self.databank, self.iratios = self._ConvertDataFile(register_coords, databank_coords, iratios) 

#         # Create a world_pool list of indices of pre-saved initial conditions and their rollouts of U positions
#         self.all_worlds             = [ind for k,ind in self.register['labels'].items()]
#         self.world_pool             = su.GetWorldPool(self.all_worlds, fixed_initial_positions, self.register)
#         self._encode                = self._encode_nodes if state_encoding == 'nodes' else self._encode_tensor
#         if state_encoding not in ['nodes', 'tensors']: assert False

#         # Dynamics parameters
#         self.current_entry          = 0    # which entry in the world pool is active
#         self.u_paths                = []
#         self.iratio                 = 0
#         self.state0                 = ()
#         self.state                  = ()   # current internal state in node labels: (e,U1,U2,...)
#         self.global_t               = 0
#         self.local_t                = 0
#         self.max_timesteps          = self.sp.T
#         self.neighbors, self.in_degree, self.max_indegree, self.out_degree, self.max_outdegree = su.GetGraphData(self.sp)

#         # Gym objects
#         #self.observation_space = spaces.Discrete(self.sp.V) if state_encoding == 'nodes' else spaces.MultiBinary(self.state_encoding_dim)
#         self.observation_space = spaces.Box(0., self.sp.U, shape=(self.state_encoding_dim,), dtype=np.float32)
#         #self.observation_space = spaces.Tuple([spaces.Discrete(self.sp.V) for i in range(self.state_len)])             
#         # @property
#         # def action_space(self):
#         #     return spaces.Discrete(self.out_degree[self.state[0]])
#         self.action_space = spaces.Discrete(self.max_outdegree)
#         self.metadata = {'render.modes':['human']}
#         self.max_episode_length = self.max_timesteps

#         # Graph feature matrix, (FxV) with F number of features, V number of nodes
#         # 0  [.] node number
#         # 1  [.] 1 if target node, 0 otherwise 
#         # 2  [.] # of units present at node at current time
#         # 3  [.] ## off ## 1 if node previously visited by unit
#         # 4  [.] ## off ## 1 if node previously visited by escaper
#         # 5  [.] ## off ## distance from nearest target node
#         # 6  [.] ...
#         self.F = 3
#         self.gfm0 = np.zeros((self.sp.V,self.F))
#         self.gfm0[:,0] = np.array([i for i in range(self.sp.V)])
#         self.gfm0[np.array(list(self.sp.target_nodes)),1]=1 # set target nodes, fixed for the given graph
#         self.gfm  = copy.deepcopy(self.gfm0)
#         self.redefine_graph_structure(W_,self.sp.nodeid2coord)
#         #self.reset()

register(
    id='GraphWorldFromDB-v0',
    entry_point='modules.rl.environments:GraphWorldFromDatabank'
)

