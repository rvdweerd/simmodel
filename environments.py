#from os import stat_result
import simdata_utils as su
import random 
#import time
from rl_plotting import PlotAgentsOnGraph, PlotAgentsOnGraph_
import numpy as np

class GraphWorld(object):
    """"""
    def __init__(self, config, optimization_method='static', fixed_initial_positions=None, state_representation='etUt', state_encoding='nodes'):
        self.type                   ='GraphWorld'
        self.sp                     = su.DefineSimParameters(config)
        self.optimization           = optimization_method
        self.fixed_initial_positions= fixed_initial_positions
        self.state_representation   = state_representation
        self.state_encoding_dim, self.state_chunks = su.GetStateEncodingDimension(state_representation, self.sp.V, self.sp.U)
        
        # Load relevant pre-saved optimization runs for the U trajectories
        dirname                     = su.make_result_directory(self.sp, optimization_method)
        self.register, self.databank, self.iratios = self._LoadAndConvertDataFile(dirname) #su.LoadDatafile(dirname)
        
        # Create a world_pool list of indices of pre-saved initial conditions and their rollouts of U positions
        self.all_worlds             = [ind for k,ind in self.register['labels'].items()]
        self.world_pool             = su.GetWorldPool(self.all_worlds, fixed_initial_positions, self.register)
        self._encode                = self._encode_nodes if state_encoding == 'nodes' else self._encode_tensor

        # Dynamics parameters
        self.current_entry          = 0    # which entry in the world pool is active
        self.u_paths                = []
        self.iratio                 = 0
        self.state0                 = ()
        self.state                  = ()   # current internal state in node labels: (e,U1,U2,...)
        self.global_t               = 0
        self.local_t                = 0
        self.max_timesteps          = self.sp.T*2
        self.neighbors, self.in_degree, self.max_indegree, self.out_degree, self.max_outdegree = su.GetGraphData(self.sp)
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
        #chunks=[]
        #chunks.append((state[0],))
        #chunks.append(state[1:(1+self.sp.U)])
        #chunks.append(state[(1+self.sp.U):])
        #num_chunks= int(len(chunks[0])>0)+int(len(chunks[1])>0)+int(len(chunks[2])>0)
        out=np.zeros(self.sp.V * len(self.state_chunks)) # 
        if sort_units:
            return NotImplementedError
        else:
            for i, chunk in enumerate(self.state_chunks):
                for pos in chunk:
                    out[i*self.sp.V + state[pos]] += 1
        return out

    def _LoadAndConvertDataFile(self, dirname):
        register_coords, databank_coords, iratios = su.LoadDatafile(dirname)
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
        if self.state_representation == 'etUt':
            return self._encode(self.state)
        elif self.state_representation == 'et':
            return self._encode((self.state[0],))
        elif self.state_representation == 'etUte0U0':
            return self._encode(self.state+self.state0)
        elif self.state_representation == 'ete0U0':
            return self._encode(tuple([self.state[0]])+self.state0)
        else:
            assert False

    def step(self, action_idx):
        next_node = self.neighbors[self.state[0]][action_idx]
        info = {'Captured':False, 'u_positions':self.state[1:]}
        self.global_t += 1
        self.local_t  += 1
        assert next_node in self.neighbors[self.state[0]]
        
        new_Upositions = self._getUpositions(self.local_t) # uses local time: u_paths may have been updated from last state if sim is dynamic
        new_Upositions.sort()
        self.state = tuple([next_node] + new_Upositions)
        reward = -1.
        done = False
        if self.global_t >= self.max_timesteps:
            done=True
            #print('Time ran out')
        if next_node in new_Upositions: # captured
            done=True
            reward += -10
            info={'Captured':True}
            #print('Captured')
        #elif self.sp.labels2coord[next_node][1] == self.sp.most_northern_y: # northern boundary of manhattan graph reached
        elif next_node in self.sp.target_nodes: 
            done = True
            reward += +10
            #print('Goal reached')
        if self.optimization == 'dynamic' and not done:
            self.u_paths = self.databank['labels'][self.register['labels'][self.state]]['paths']
            if len(self.u_paths) < 2:
                assert False
            self.local_t = 0

        if self.state_representation == 'etUt':
            return self._encode(self.state), reward, done, info
        elif self.state_representation == 'et':
            return self._encode((self.state[0],)), reward, done, info
        elif self.state_representation == 'etUte0U0':
            return self._encode(self.state+self.state0), reward, done, info
        elif self.state_representation == 'ete0U0':
            return self._encode(tuple([self.state[0]])+self.state0), reward, done, info

    def render(self, file_name=None):
        e = self.state[0]
        p=self.state[1:]
        PlotAgentsOnGraph_(self.sp, e, p, self.global_t, fig_show=False, fig_save=True, filename=file_name)