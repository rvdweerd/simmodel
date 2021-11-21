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
        self.register, self.databank, self.iratios = su.LoadDatafile(dirname)
        self.current_entry=0
        self.u_paths=[]
        self.iratio=0
        self.state=()
        self.global_t=0
        self.local_t=0
        self.reset()
        self.vec2dir={(0,1):'N',(1,0):'E',(0,-1):'S',(-1,0):'W'}
        self.dir2vec={d:c for c,d in self.vec2dir.items()}
        self.reachable_nodes = []

    def _getUpositions(self,t=0):
        upos=[]
        for i,P_path in enumerate(self.u_paths):
            p = P_path[-1] if t >= len(P_path) else P_path[t]
            upos.append(p)
        return upos

    def _availableActionsInCurrentState(self):
        reachable_coords = list(self.sp.G.neighbors(self.state[0]))
        reachable_nodes = [self.sp.coord2labels[c] for c in reachable_coords]
        reachable_directions = None#[self.vec2dir[(n[0]-self.state[0][0],n[1]-self.state[0][1])] for n in reachable_coords]
        return {'coords': reachable_coords, 'node_labels': reachable_nodes, 'directions': reachable_directions}
            
    def reset(self, initial_state=None):
        # if called with databank_entry, a specific saved initial position is loaded
        if initial_state is not None:
            entry = self.register[initial_state]
        else:
            entry = random.randint(0,len(self.databank)-1)
        self.current_entry=entry
        data_sample = self.databank[entry]
        self.iratio = self.iratios[entry]
        e_init=data_sample['start_escape_route'] # (x,y)
        u_init=data_sample['start_units'] # [(x0,y0)_1, (x0,y0)_2, ...]
        self.u_paths=data_sample['paths']
        self.global_t = 0
        self.local_t = 0
        self.state = tuple([e_init]+u_init)
        return self.state

    def step(self, next_node):
        info={'Captured':False}
        if type(next_node)==tuple: # input is a coordinate tuple 
            pass
        elif type(next_node)==str:
            next_node = self.sp.labels2coord[int(next_node)]
        else:
            assert False
        self.global_t += 1
        self.local_t += 1
        assert next_node in list(self.sp.G.neighbors(self.state[0]))
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
        # if next_node[1] == self.sp.most_northern_y: # northern boundary of manhattan graph reached
        #    done = True
        #    reward += +10
        
        if self.optimization == 'dynamic' and not done:
            self.u_paths = self.databank[self.register[self.state]]['paths']
            self.local_t = 0
        return self.state, reward, done, info

    def render(self):
        #escape position
        e = self.state[0]
        #escape_path[-1] if t >= len(escape_path) else escape_path[t]

        #u positions
        p = []
        for P_path in self.u_paths:
            pos = P_path[-1] if self.local_t >= len(P_path) else P_path[self.local_t]
            p.append(pos)
        su.PlotAgentsOnGraph(self.sp, e, p, self.global_t, fig_show=False, fig_save=True)
        time.sleep(1)



