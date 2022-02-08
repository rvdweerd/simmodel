# Functions that construct a node feature matrix, used in the GraphWorld class
from distutils.log import set_threshold
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
        # Set target nodes
        if len(eo.sp.target_nodes) > 0:
            eo.nfm0[np.array(list(eo.sp.target_nodes)),1]=1 
        eo.nfm  = copy.deepcopy(eo.nfm0)
        
    def reset(self, eo):
        eo.nfm = copy.deepcopy(eo.nfm0)
        eo.nfm[eo.sp.start_escape_route_node,0]=1

    def update(self, eo):
        eo.nfm[eo.state[0],0]=1

class NFM_ev_ec_t_um_us():
    def __init__(self):
        self.name='nfm-ev-ec-t-um-us'
        # Features:
        # 0. visited by e
        # 1. current e
        # 2. target node
        # 3. u positions (on the move)
        # 4. u positions (when settled)
    
    def init(self, eo):
        eo.F = 5
        #eo.nfm0 = np.zeros((eo.sp.V,eo.F))
        eo.nfm0 = torch.zeros((eo.sp.V,eo.F),dtype=torch.float32)
        # Set target nodes
        if len(eo.sp.target_nodes) > 0:
            #eo.nfm0[np.array(list(eo.sp.target_nodes)),2] = 1
            eo.nfm0[torch.tensor(list(eo.sp.target_nodes),dtype=torch.int64),2] = 1.
        eo.nfm  = copy.deepcopy(eo.nfm0)
        
    def reset(self, eo):
        eo.nfm = copy.deepcopy(eo.nfm0)
        # Set e position
        eo.nfm[eo.sp.start_escape_route_node, 0]=1
        eo.nfm[eo.sp.start_escape_route_node, 1]=1
        # Set u positions
        for path_index, path in enumerate(eo.u_paths): 
            if eo.local_t >= len(path)-1: #unit has settled
                assert path[-1] in eo.state
                eo.nfm[path[-1],4] += 1
            else:
                eo.nfm[path[eo.local_t],3] += 1

    def update(self, eo):
        eo.nfm[eo.state[0],0]=1
        eo.nfm[:,1]=0
        eo.nfm[eo.state[0],1]=1   
        # Set u positions
        eo.nfm[:,3] = 0     
        eo.nfm[:,4] = 0     
        for path_index, path in enumerate(eo.u_paths): 
            if eo.local_t >= len(path)-1: #unit has settled
                assert path[-1] in eo.state
                eo.nfm[path[-1],4] += 1
            else:
                eo.nfm[path[eo.local_t],3] += 1

class NFM_ev_ec_t_u():
    def __init__(self):
        self.name='nfm-ev-ec-t-u'
        # Features:
        # 0. visited by e
        # 1. current e
        # 2. target node
        # 3. u positions
    
    def init(self, eo):
        eo.F = 4
        eo.nfm0 = np.zeros((eo.sp.V,eo.F))
        # Set target nodes
        if len(eo.sp.target_nodes) > 0:
            eo.nfm0[np.array(list(eo.sp.target_nodes)),2] = 1
        eo.nfm  = copy.deepcopy(eo.nfm0)
        
    def reset(self, eo):
        eo.nfm = copy.deepcopy(eo.nfm0)
        # Set e position
        eo.nfm[eo.sp.start_escape_route_node, 0]=1
        eo.nfm[eo.sp.start_escape_route_node, 1]=1
        # Set u positions
        for u in eo.state[1:]:
            eo.nfm[u,3] += 1

    def update(self, eo):
        eo.nfm[eo.state[0],0]=1
        eo.nfm[:,1]=0
        eo.nfm[eo.state[0],1]=1   
        # Set u positions
        eo.nfm[:,3] = 0     
        for u in eo.state[1:]:
            eo.nfm[u,3] += 1

class NFM_ev_ec_t_um_us_xW(NFM_ev_ec_t_um_us):
    def __init__(self):
        self.name='nfm-ev-ec-t-um-us_xW'
        # Inherited, concats Adj matrix W to the output (used in bolts dqn)
        # Features:
        # 0. visited by e
        # 1. current e
        # 2. target node
        # 3. u positions (on the move)
        # 4. u positions (when settled)
    def init(self, eo):
        eo.F = 5
        eo.nfm0 = torch.concat((torch.zeros((eo.sp.V,eo.F),dtype=torch.float32),torch.tensor(eo.sp.W, dtype=torch.float32)),dim=1)
        # Set target nodes
        if len(eo.sp.target_nodes) > 0:
            eo.nfm0[np.array(list(eo.sp.target_nodes)),2] = 1
        eo.nfm  = copy.deepcopy(eo.nfm0)


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