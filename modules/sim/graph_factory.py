import numpy as np
from itertools import product

def all_Manhattan3x3_symmetric_adj_matrices(reflexive=True):
    n=3
    N=n**2
    K=(N-n)*2
    all_W = []
    W_per_num_edge_removals={i:[] for i in range(K+1)}
    hashes=set()
    for vals in product([0, 1], repeat=(K)):
        hash = int("".join(str(v) for v in vals), 2)
        if hash in hashes:
            assert False
        else:
            hashes.add(hash)
        arr = np.zeros((N,N))
        i_up=np.array([(0,1),(0,3),(1,2),(1,4),(2,5),(3,4),(3,6),(4,5),(4,7),(5,8),(6,7),(7,8)])
        i_y=np.array([i for (i,j) in i_up])
        i_x=np.array([j for (i,j) in i_up])        
        arr[i_y,i_x]=vals
        arr=arr+arr.T
        if reflexive:
            arr=arr+np.eye(N)
        # Check; no orphan nodes (nodes with no edges)
        if np.min(arr.sum(axis=1)) > 1:
            all_W.append(arr)
            W_per_num_edge_removals[K-np.sum(vals)].append(arr)
    return all_W, W_per_num_edge_removals