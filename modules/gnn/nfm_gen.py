# Functions that construct a node feature matrix, used in the GraphWorld class
from distutils.log import set_threshold
import numpy as np
import copy
import torch
import networkx as nx

class NFM_ev_t():
    def __init__(self):
        self.name='nfm-ev-t'
        self.F=2
        # Features:
        # 0. visited by e
        # 1. target node
    
    def init(self, eo):
        eo.F = 2
        eo.nfm0 = torch.zeros((eo.sp.V,eo.F),dtype=torch.float32)
        # Set target nodes
        if len(eo.sp.target_nodes) > 0:
            eo.nfm0[torch.tensor(list(eo.sp.target_nodes),dtype=torch.int64),1]=1 
        #eo.nfm  = copy.deepcopy(eo.nfm0)
        self.reset(eo)
        
    def reset(self, eo):
        eo.nfm = eo.nfm0.clone()
        eo.nfm[eo.sp.start_escape_route_node,0]=1

    def update(self, eo):
        eo.nfm[eo.state[0],0]=1

class NFM_ev_ec_t_um_us():
    def __init__(self):
        self.name='nfm-ev-ec-t-um-us'
        self.F=5
        # Features:
        # 0. visited by e
        # 1. current e
        # 2. target node
        # 3. u positions (on the move)
        # 4. u positions (when settled)
    
    def init(self, eo):
        eo.F = 5
        eo.nfm0 = torch.zeros((eo.sp.V,eo.F),dtype=torch.float32)
        # Set target nodes
        if len(eo.sp.target_nodes) > 0:
            eo.nfm0[torch.tensor(list(eo.sp.target_nodes),dtype=torch.int64),2] = 1.
        self.reset(eo)
        
    def reset(self, eo):
        eo.nfm = eo.nfm0.clone()
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

    def get_custom_nfm(self, eo, epath, targetnodes, upaths):
        nfm = torch.zeros(eo.nfm0.shape,dtype=torch.float32)
        local_t = len(epath)-1
        if len(targetnodes) > 0:
            #eo.nfm0[np.array(list(eo.sp.target_nodes)),2] = 1
            nfm[torch.tensor(list(targetnodes),dtype=torch.int64),2] = 1.
        for e in epath:
            nfm[e,0] = 1 # visited nodes by e
        nfm[epath[-1],1] = 1 # current e position
        upos=[]
        for path_index, path in enumerate(upaths): 
            if local_t >= len(path)-1: #unit has settled
                nfm[path[-1],4] += 1
                upos.append(path[-1])
            else:
                nfm[path[eo.local_t],3] += 1
                upos.append(path[eo.local_t])
        upos.sort()
        state=tuple([epath[-1]]+upos)
        return nfm, state

class NFM_ev_ec_t_u():
    def __init__(self):
        self.name='nfm-ev-ec-t-u'
        self.F=4
        # Features:
        # 0. visited by e
        # 1. current e
        # 2. target node
        # 3. u positions
    
    def init(self, eo):
        eo.F = 4
        eo.nfm0 = torch.zeros((eo.sp.V,eo.F))
        # Set target nodes
        if len(eo.sp.target_nodes) > 0:
            eo.nfm0[torch.tensor(list(eo.sp.target_nodes),dtype=torch.int64),2] = 1
        self.reset(eo)
        
    def reset(self, eo):
        eo.nfm = eo.nfm0.clone()
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

class NFM_ec_dtscaled():
    def __init__(self):
        self.name='nfm-ec-dtscaled'
        self.F=2
        # Features:
        # 0. current position e
        # 1. measure of distance/options to target nodes
    def init(self, eo):
        eo.F = self.F
        eo.nfm0 = torch.zeros((eo.sp.V,eo.F),dtype=torch.float32)
        eo.nfm0[:,1] = eo.scaled_nodescores.clone()
        #eo.nfm  = copy.deepcopy(eo.nfm0)
        self.reset(eo)

    def reset(self, eo):
        eo.nfm = eo.nfm0.clone()
        eo.nfm[eo.sp.start_escape_route_node,0]=1

    def update(self, eo):
        eo.nfm[:,0]=0
        eo.nfm[eo.state[0],0]=1

class NFM_ec_dt_alt():
    def __init__(self):
        self.name='nfm-ec-dt'
        self.F=2
        # Features:
        # 0. current position e
        # 1. measure of distance/options to target nodes
    def init(self, eo):
        eo.F = 2
        eo.nfm0 = torch.zeros((eo.sp.V,eo.F),dtype=torch.float32)
        if len(eo.sp.target_nodes) > 0:
            #eo.nfm0[torch.tensor(list(eo.sp.target_nodes),dtype=torch.int64),1]=1 # set target nodes, fixed for the given graph
            # go over all node labels
            for sourcelabel, sourcecoord in eo.sp.labels2coord.items():
                if sourcelabel in eo.sp.target_nodes:
                    continue # assign fixed values to target nodes later
                distances, spaths = nx.single_source_dijkstra(eo.sp.G, sourcecoord) # dicts to all target coords
                # calc distance to all target nodes
                score = 0
                for targetlabel in eo.sp.target_nodes:
                    targetcoord = eo.sp.labels2coord[targetlabel]
                    if targetcoord in distances:
                        d = distances[targetcoord]
                        score += 1/d
                eo.nfm0[sourcelabel,1] = score
        max_score = eo.nfm0[:,1].max()
        eo.nfm0 /= max_score
        for n in eo.sp.target_nodes:
            eo.nfm0[n,1]=2
        
        eo.nfm  = copy.deepcopy(eo.nfm0)

    def reset(self, eo):
        eo.nfm = copy.deepcopy(eo.nfm0)
        eo.nfm[eo.sp.start_escape_route_node,0]=1

    def update(self, eo):
        eo.nfm[:,0]=0
        eo.nfm[eo.state[0],0]=1

class NFM_ec_dt():
    def __init__(self):
        self.name='nfm-ec-dt'
        self.F=2
        # Features:
        # 0. current position e
        # 1. measure of distance/options to target nodes
    def init(self, eo):
        eo.F = 2
        eo.nfm0 = torch.zeros((eo.sp.V,eo.F),dtype=torch.float32)
        eo.nfm0[:,1] = eo.nodescores.clone()
        self.reset(eo)

    def reset(self, eo):
        eo.nfm = eo.nfm0.clone()
        eo.nfm[eo.sp.start_escape_route_node,0]=1

    def update(self, eo):
        eo.nfm[:,0]=0
        eo.nfm[eo.state[0],0]=1

class NFM_ec_t_dt_at():
    def __init__(self):
        self.name='nfm-ec-t-dt'
        self.F=4
        # Features:
        # 0. current position e
        # 1. target node
        # 2. scaled measure of distance/options to target nodes
        # 3. absolute distance (hops) to nearest target node

    def init(self, eo):
        eo.F = self.F
        eo.nfm0 = torch.zeros((eo.sp.V,eo.F),dtype=torch.float32)
        eo.nfm0[:,2] = eo.nodescores.clone()
        eo.nfm0[:,3] = eo.min_target_distances.clone()
        if len(eo.sp.target_nodes) > 0:
            eo.nfm0[torch.tensor(list(eo.sp.target_nodes),dtype=torch.int64),1]=1 # set target nodes, fixed for the given graph
        self.reset(eo)

    def reset(self, eo):
        eo.nfm = eo.nfm0.clone()
        eo.nfm[eo.sp.start_escape_route_node,0]=1

    def update(self, eo):
        eo.nfm[:,0]=0
        eo.nfm[eo.state[0],0]=1

class NFM_ev_ec_t_dt_at_um_us():
    def __init__(self):
        self.name='nfm-ev-ec-t-dt-at-um-us'
        self.F=7
        self.uindx=torch.tensor([5,6],dtype=torch.int64) # feature indices of U position information
        # Features:
        # 0. visited by e
        # 1. current position e
        # 2. target node
        # 3. scaled measure of distance/options to target nodes
        # 4. absolute distance (hops) to nearest target node
        # 5. u positions (on the move)
        # 6. u positions (when settled)

    def init(self, eo):
        eo.F = self.F
        eo.nfm0 = torch.zeros((eo.sp.V,eo.F),dtype=torch.float32)
        eo.nfm0[:,3] = eo.nodescores.clone()
        eo.nfm0[:,4] = eo.min_target_distances.clone()
        if len(eo.sp.target_nodes) > 0:
            eo.nfm0[torch.tensor(list(eo.sp.target_nodes),dtype=torch.int64),2]=1 # set target nodes, fixed for the given graph
        self.reset(eo)

    def reset(self, eo):
        eo.nfm = eo.nfm0.clone()
        eo.nfm[eo.sp.start_escape_route_node,0]=1
        eo.nfm[eo.sp.start_escape_route_node,1]=1
        # Set u positions

        for path_index, path in enumerate(eo.u_paths): 
            if eo.local_t >= len(path)-1: #unit has settled
                assert path[-1] in eo.state
                eo.nfm[path[-1],6] += 1
            else:
                eo.nfm[path[eo.local_t],5] += 1

    def update(self, eo):
        eo.nfm[eo.state[0],0]=1
        eo.nfm[:,1]=0
        eo.nfm[eo.state[0],1]=1
        # Set u positions
        eo.nfm[:,5] = 0     
        eo.nfm[:,6] = 0     
        for unit_index, path in enumerate(eo.u_paths): 
            if eo.local_t >= len(path)-1: #unit has settled
                assert path[-1] in eo.state
                eo.nfm[path[-1],6] += 1
            else:
                eo.nfm[path[eo.local_t],5] += 1

    def mask_units(self, eo, mask_u):
        masked_nfm = eo.nfm.clone()
        for unit_index, path in enumerate(eo.u_paths): 
            if mask_u[unit_index]:
                if eo.local_t >= len(path)-1: #unit has settled
                    assert path[-1] in eo.state
                    masked_nfm[path[-1],6] -= 1
                    assert masked_nfm[path[-1],6] >= 0
                else:
                    masked_nfm[path[eo.local_t],5] -= 1
                    assert masked_nfm[path[eo.local_t],5] >= 0  
        return masked_nfm
        
class NFM_ev_ec_t_dt_at_ustack():
    def __init__(self, k=3):
        self.name='nfm-ev-ec-t-dt-at-ustack'
        self.F=5+k
        self.k=k
        self.uindx=torch.tensor([5+k-1],dtype=torch.int64) # which columns are erased if all units are unobservable
        # Features:
        # 0. visited by e
        # 1. current position e
        # 2. target node
        # 3. scaled measure of distance/options to target nodes
        # 4. absolute distance (hops) to nearest target node
        # 5.        u position at t-(k-1)
        # .         ...
        # 5+(k-1)   u position at t

    def reinit(self, k):
        assert k>=1
        self.F=5+k
        self.k=k
        self.uindx=torch.tensor([5+k-1],dtype=torch.int64)

    def init(self, eo):
        eo.F = self.F
        eo.nfm0 = torch.zeros((eo.sp.V,eo.F),dtype=torch.float32)
        eo.nfm0[:,3] = eo.nodescores.clone()
        eo.nfm0[:,4] = eo.min_target_distances.clone()
        if len(eo.sp.target_nodes) > 0:
            eo.nfm0[torch.tensor(list(eo.sp.target_nodes),dtype=torch.int64),2]=1 # set target nodes, fixed for the given graph
        self.reset(eo)

    def reset(self, eo):
        # set e positions
        eo.nfm = eo.nfm0.clone()
        eo.nfm[eo.sp.start_escape_route_node,0]=1
        eo.nfm[eo.sp.start_escape_route_node,1]=1
        
        # Set u positions
        for path_index, path in enumerate(eo.u_paths): 
            eo.nfm[path[eo.local_t],-1] += 1

    def update(self, eo):
        # set e positions
        eo.nfm[eo.state[0],0]=1
        eo.nfm[:,1]=0
        eo.nfm[eo.state[0],1]=1
        
        # Set u positions
        eo.nfm[:,5:(5+self.k-1)] = eo.nfm[:,6:] # shift the last k-1 columns to the left    
        eo.nfm[:,-1] = 0 # reset column for current positions
        for unit_index, path in enumerate(eo.u_paths):
            if eo.local_t >= len(path)-1: #unit has settled
                eo.nfm[path[-1],-1] += 1
            else:
                eo.nfm[path[eo.local_t],-1] += 1

    def mask_units(self, eo, mask_u):
        #masked_nfm = eo.nfm.clone()
        for unit_index, path in enumerate(eo.u_paths): 
            if mask_u[unit_index]:
                if eo.local_t >= len(path)-1: #unit has settled
                    eo.nfm[path[-1],-1] -= 1
                    assert eo.nfm[path[-1],-1] >= 0
                else:
                    eo.nfm[path[eo.local_t],-1] -= 1
                    assert eo.nfm[path[eo.local_t],-1] >= 0
        return eo.nfm.clone()

class NFM_ec_t():
    def __init__(self):
        self.name='nfm-ec-t'
        self.F=2
        # Features:
        # 0. current position e
        # 1. target node
    def init(self, eo):
        eo.F = 2
        eo.nfm0 = torch.zeros((eo.sp.V,eo.F),dtype=torch.float32)
        if len(eo.sp.target_nodes) > 0:
            eo.nfm0[torch.tensor(list(eo.sp.target_nodes),dtype=torch.int64),1]=1 # set target nodes, fixed for the given graph
        #eo.nfm  = copy.deepcopy(eo.nfm0)
        self.reset(eo)

    def reset(self, eo):
        eo.nfm = eo.nfm0.clone()
        eo.nfm[eo.sp.start_escape_route_node,0]=1

    def update(self, eo):
        eo.nfm[:,0]=0
        eo.nfm[eo.state[0],0]=1

class NFM_ev_ec_t():
    def __init__(self):
        self.name='nfm-ev-ec-t'
        self.F=3
        # Features:
        # 0. current position e
        # 1. visited e positions
        # 2. target node
    def init(self, eo):
        eo.F = 3
        eo.nfm0 = torch.zeros((eo.sp.V,eo.F),dtype=torch.float32)
        if len(eo.sp.target_nodes) > 0:
            eo.nfm0[torch.tensor(list(eo.sp.target_nodes),dtype=torch.int64),2]=1 # set target nodes, fixed for the given graph
        self.reset(eo)

    def reset(self, eo):
        eo.nfm = eo.nfm0.clone()
        eo.nfm[eo.sp.start_escape_route_node,0]=1
        eo.nfm[eo.sp.start_escape_route_node,1]=1

    def update(self, eo):
        eo.nfm[:,0]=0
        eo.nfm[eo.state[0],0]=1
        eo.nfm[eo.state[0],1]=1

nfm_funcs = {
    'NFM_ev_ec_t':NFM_ev_ec_t(),
    'NFM_ec_t':NFM_ec_t(),
    'NFM_ec_t_dt_at':NFM_ec_t_dt_at(),
    'NFM_ev_ec_t_dt_at_um_us':NFM_ev_ec_t_dt_at_um_us(),
    'NFM_ev_ec_t_dt_at_ustack':NFM_ev_ec_t_dt_at_ustack(),
    'NFM_ec_dt':NFM_ec_dt(),
    'NFM_ec_dtscaled':NFM_ec_dtscaled(),
    'NFM_ev_t':NFM_ev_t(),
    'NFM_ev_ec_t_um_us':NFM_ev_ec_t_um_us()
    }