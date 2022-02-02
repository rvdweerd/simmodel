# Functions that construct a node feature matrix, used in the GraphWorld class
import numpy as np
import copy
import torch

class NFM_ev_t():
    def __init__(self):
        self.name='nfm-ev-t'
        # Features:
        # 0. visited by e
        # 1. target node
    def init(self, eo):
        eo.F = 2
        eo.nfm0 = np.zeros((eo.sp.V,eo.F))
        #eo.nfm0 = torch.concat((torch.zeros((eo.sp.V,eo.F),dtype=torch.float32),torch.tensor(eo.sp.W, dtype=torch.float32)),dim=1)
        
        #eo.nfm0[:,0] = np.array([i for i in range(eo.sp.V)])
        if len(eo.sp.target_nodes) > 0:
            eo.nfm0[np.array(list(eo.sp.target_nodes)),1]=1 # set target nodes, fixed for the given graph
        eo.nfm  = copy.deepcopy(eo.nfm0)
        #eo.reset()
    def reset(self, eo):
        eo.nfm = copy.deepcopy(eo.nfm0)
        eo.nfm[eo.sp.start_escape_route_node,0]=1
        #for u_index, u in enumerate(eo.state[1:]): 
        #    eo.nfm[u,2]+=1         # f2: current presence of units
            #self.nfm[u,3]=1         # f3: node previously visited by any unit
        #for p in eo.u_paths:
        #    if eo.local_t >= len(p)-1:
        #        eo.nfm[p[-1],2] += 10
    def update(self, eo):
        #eo.nfm[:,2]=0
        #for u_index, u in enumerate(eo.state[1:]): 
        #    eo.nfm[u,2]+=1         # f2: current presence of units
        #    #self.nfm[u,3]=1         # f3: node previously visited by any unit
        #for p in eo.u_paths:
        #    if eo.local_t >= len(p)-1:
        #        eo.nfm[p[-1],2] += 10
        eo.nfm[eo.state[0],0]=1

class NFM_ec_t():
    def __init__(self):
        self.name='nfm-ec-t'
        # Features:
        # 0. current position e
        # 1. target node
    def init(self, eo):
        eo.F = 2
        eo.nfm0 = np.zeros((eo.sp.V,eo.F))
        if len(eo.sp.target_nodes) > 0:
            eo.nfm0[np.array(list(eo.sp.target_nodes)),1]=1 # set target nodes, fixed for the given graph
        eo.nfm  = copy.deepcopy(eo.nfm0)

    def reset(self, eo):
        eo.nfm = copy.deepcopy(eo.nfm0)
        eo.nfm[eo.sp.start_escape_route_node,0]=1

    def update(self, eo):
        eo.nfm[:,0]=0
        eo.nfm[eo.state[0],0]=1

class NFM_ev_ec_t():
    def __init__(self):
        self.name='nfm-ev-ec-t'
        # Features:
        # 0. current position e
        # 1. visited e positions
        # 2. target node
    def init(self, eo):
        eo.F = 3
        eo.nfm0 = np.zeros((eo.sp.V,eo.F))
        if len(eo.sp.target_nodes) > 0:
            eo.nfm0[np.array(list(eo.sp.target_nodes)),2]=1 # set target nodes, fixed for the given graph
        eo.nfm  = copy.deepcopy(eo.nfm0)

    def reset(self, eo):
        eo.nfm = copy.deepcopy(eo.nfm0)
        eo.nfm[eo.sp.start_escape_route_node,0]=1
        eo.nfm[eo.sp.start_escape_route_node,1]=1

    def update(self, eo):
        eo.nfm[:,0]=0
        eo.nfm[eo.state[0],0]=1
        eo.nfm[eo.state[0],1]=1