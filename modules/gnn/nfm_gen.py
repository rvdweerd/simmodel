# Functions that construct a node feature matrix, used in the GraphWorld class
import numpy as np
import copy

class BasicNFM():
    def __init__(self):
        self.name='BasicNFM'
    def init(self, eo):
        eo.F = 3
        eo.gfm0 = np.zeros((eo.sp.V,eo.F))
        eo.gfm0[:,0] = np.array([i for i in range(eo.sp.V)])
        eo.gfm0[np.array(list(eo.sp.target_nodes)),1]=1 # set target nodes, fixed for the given graph
        eo.gfm  = copy.deepcopy(eo.gfm0)
        eo.reset()
    def reset(self, eo):
        eo.gfm = copy.deepcopy(eo.gfm0)
        for u_index, u in enumerate(eo.state[1:]): 
            eo.gfm[u,2]+=1         # f2: current presence of units
            #self.gfm[u,3]=1         # f3: node previously visited by any unit
        for p in eo.u_paths:
            if eo.local_t >= len(p)-1:
                eo.gfm[p[-1],2] += 10
    def update(self, eo):
        eo.gfm[:,2]=0
        for u_index, u in enumerate(eo.state[1:]): 
            eo.gfm[u,2]+=1         # f2: current presence of units
            #self.gfm[u,3]=1         # f3: node previously visited by any unit
        for p in eo.u_paths:
            if eo.local_t >= len(p)-1:
                eo.gfm[p[-1],2] += 10
