import simdata_utils as su
import random 
import time

class GraphWorld(object):
    """"""
    def __init__(self, config, optimization='static'):
        self.type='GraphWorld'
        self.optimization=optimization
        self.sp = su.DefineSimParameters(config)
        dirname = su.make_result_directory(self.sp)
        if optimization == 'dynamic':
            dirname += '_allE'
        self.register, self.databank, self.iratios = self._LoadAndConvertDataFile(dirname) #su.LoadDatafile(dirname)
        self.current_entry=0
        self.u_paths=[]
        self.iratio=0
        self.state=()
        self.global_t=0
        self.local_t=0
        self.vec2dir={(0,1):'N',(1,0):'E',(0,-1):'S',(-1,0):'W'}
        self.dir2vec={d:c for c,d in self.vec2dir.items()}
        self.reachable_nodes = []
        self.neighbors, self.in_degree, self.out_degree = self._GetGraphData()
        self.reset()

    def _GetGraphData(self):
        G=self.sp.G.to_directed()
        
        neighbors_labels = {}
        for node_coord in G.nodes:
            local_neighbors_labels = []
            node_label = self.sp.coord2labels[node_coord]
            for n_coord in G.neighbors(node_coord):
                n_label = self.sp.coord2labels[n_coord]
                local_neighbors_labels.append(n_label)
            neighbors_labels[node_label] = local_neighbors_labels

        outdegrees_labels={}
        for coord,deg in G.out_degree:
            outdegrees_labels[self.sp.coord2labels[coord]] = deg
        
        indegrees_labels={}
        for coord,deg in G.in_degree:
            indegrees_labels[self.sp.coord2labels[coord]] = deg
        
        return neighbors_labels, indegrees_labels, outdegrees_labels

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
        upos=[]
        for i,P_path in enumerate(self.u_paths):
            p = P_path[-1] if t >= len(P_path) else P_path[t]
            upos.append(p)
        return upos

    def _availableActionsInCurrentState(self):
        # reachable_coords = list(self.sp.G.neighbors(self.state[0]))
        # reachable_nodes = [self.sp.coord2labels[c] for c in reachable_coords]
        # reachable_directions = None#[self.vec2dir[(n[0]-self.state[0][0],n[1]-self.state[0][1])] for n in reachable_coords]
        # return {'coords': reachable_coords, 'node_labels': reachable_nodes, 'directions': reachable_directions}
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


    def reset(self, initial_state=None):
        # Reset time
        self.global_t = 0
        self.local_t = 0 # used if optimization is dynamic; lookup time for new paths is set to 0 after each step
        
        # Load pre-saved dataset of pursuers movement
        if initial_state is not None:
            # if called with databank_entry (in coords), a specific saved initial position is loaded
            entry = self.register[initial_state]
        else:
            entry = random.randint(0,len(self.databank)-1)
        self.current_entry=entry
        data_sample = self.databank['labels'][entry]
        self.iratio = self.iratios[entry]
        
        # Assign initial state
        e_init_labels = data_sample['start_escape_route'] # (e0)
        u_init_labels = data_sample['start_units'] # [(u1),(u2), ...]
        self.u_paths  = data_sample['paths']
        self.state    = self._to_state(e_init_labels,u_init_labels)
        return self.state

    def step(self, next_node):
        info={'Captured':False}
        self.global_t += 1
        self.local_t += 1
        assert next_node in self.neighbors[self.state[0]]
        new_Upositions = self._getUpositions(self.local_t) # uses local time: u_paths may have been updated from last state if sim is dynamic
        testu0=new_Upositions[0]
        testu1=new_Upositions[1]
        new_Upositions.sort()
        if new_Upositions[0] is not testu0:
            k=0
        self.state = tuple([next_node] + new_Upositions)
        reward = +1
        done = False
        if next_node in new_Upositions: # captured
            done=True
            reward += -10
            info={'Captured':True}
        if self.sp.labels2coord[next_node][1] == self.sp.most_northern_y: # northern boundary of manhattan graph reached
           done = True
           reward += +10
        
        if self.optimization == 'dynamic' and not done:
            self.u_paths = self.databank[self.register[self.state]]['paths']
            self.local_t = 0
        return self.state, reward, done, info

    def render(self):
        #escape position
        e = self.state[0] # (.,.) coord
        #escape_path[-1] if t >= len(escape_path) else escape_path[t]

        #u positions
        p = []
        for P_path in self.u_paths:
            pos = P_path[-1] if self.local_t >= len(P_path) else P_path[self.local_t]
            p.append(pos)
        su.PlotAgentsOnGraph(self.sp, e, p, self.global_t, fig_show=False, fig_save=True)
        time.sleep(1)



