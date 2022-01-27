import numpy as np
from itertools import product
import random 
import networkx as nx
import pickle
from modules.rl.environments import GraphWorld#, GraphWorldFromDatabank

def rand_key(p):
    key1 = ""
    for i in range(p):
        temp = str(random.randint(0, 1))
        key1 += temp
    return(key1)

def rand_key_fixed_num_removed(K, num_removed):
    arr=np.array([0]*num_removed + [1]*(K-num_removed))
    np.random.shuffle(arr)
    return arr

def create_adj_matrix(N,coord_upper,vals,W_):
    arr = np.zeros((N,N))
    #i_up=np.array([(0,1),(0,3),(1,2),(1,4),(2,5),(3,4),(3,6),(4,5),(4,7),(5,8),(6,7),(7,8)])
    i_y=np.array([i for (i,j) in coord_upper]) # index the elements in the upper triangular to be set
    i_x=np.array([j for (i,j) in coord_upper])        
    arr[i_y,i_x]=vals
    arr += arr.T # make symmetric
    arr[np.diag_indices_from(arr)] = np.diag(W_) # copy the diagonal elements
    return arr

def target_reachable(W, start_node, target_nodes):
    # Checks if any (at least one) of the target nodes is reachable
    G = nx.from_numpy_matrix(W, create_using=nx.DiGraph())
    reachable = False
    for t in target_nodes:
        if nx.algorithms.shortest_paths.generic.has_path(G,start_node,t):
            reachable = True
            break
    return reachable

def get_all_edge_removals_symmetric(W_, start_node, target_nodes, removals=[1], instances_per_num_removed=1, cutoff = 1e4):
    ##
    # params:
    # W_            : adjacency matrix (numpy array)
    # start_node    : start position of escaper (in nodeid)
    # target_nodes  : should be reachable from start_node (in nodeid)
    # removals      : list with number of edges to be removed
    # instances_per_num_removed: how many graphs created for each case
    #
    # returns:
    # all_W                     : list containing all combinations of edges removed as tuple (W_reduced, num_edges_reduced, hash)
    # W_per_num_edge_removals   : dict from number of edges removed to list of tuples (W_reduced, hash)  
    assert np.array_equal(W_.T,W_) # symmetric
    assert W_.shape[0]<64 # int64 hashing, only works for graphs with < nodes (purpose is experimentation)

    N=W_.shape[0] # number of nodes
    #n=np.sqrt(N)
    idx_upper = np.triu_indices(N, k=1) # offset k=1 (excluding diagonal entries)
    coord_upper = [(i,j) for i,j in zip(idx_upper[0],idx_upper[1]) if W_[i,j]>0] # the edge pool up for removal
    K=len(coord_upper) # the number of edges in the edge pool up for removal (K=(N-n)*2 for Manhattan graph)

    # Containers
    all_W = []
    hashes_int=set()
    if K <= 12: # manageable number of permutation, 2^12=4096, return exhaustive list
        W_per_num_edge_removals={}#i:[] for i in range(K+1)}
        for vals in product([0, 1], repeat=(K)):
            hash_str = "".join(str(v) for v in vals)
            hash_int = int("".join(str(v) for v in vals), 2)
            if hash_int in hashes_int:
                assert False
            else:
                hashes_int.add(hash)
            arr = create_adj_matrix(N,coord_upper,vals,W_)
            # Check; no orphan nodes (nodes with no edges) and at least one target node is reachable
            #if np.min(arr.sum(axis=1)) > 1 and target_reachable(arr, start_node, target_nodes):
            if target_reachable(arr, start_node, target_nodes):
                num_edges_removed = K-np.sum(vals)
                all_W.append((arr, num_edges_removed, hash_int, hash_str))
                if (K-np.sum(vals)) not in W_per_num_edge_removals:
                    W_per_num_edge_removals[K-np.sum(vals)]=[]
                W_per_num_edge_removals[K-np.sum(vals)].append((arr, hash_int, hash_str))
    else:
        #assert instances_per_num_removed <= K
        W_per_num_edge_removals={i:[] for i in removals}
        for num_removed in removals:#range(1,K//2-1):
            attempts=0
            #theoretic_max = np.prod(range(K-num_removed+1,K+1))
            while len(W_per_num_edge_removals[num_removed]) < instances_per_num_removed:# and len(W_per_num_edge_removals[num_removed]) != :
                vals = rand_key_fixed_num_removed(K, num_removed)
                attempts+=1
                hash_str = "".join(str(v) for v in vals)
                hash_int = int(hash_str, 2)
                if hash_int in hashes_int:
                    assert False
                else:
                    hashes_int.add(hash)
                arr = create_adj_matrix(N,coord_upper,vals,W_)
                # Check; no orphan nodes (nodes with no edges) and at least one target node is reachable
                if np.min(arr.sum(axis=1)) > 1 and target_reachable(arr, start_node, target_nodes):
                    num_edges_removed = K-np.sum(vals)
                    all_W.append((arr, num_edges_removed, hash_int))
                    W_per_num_edge_removals[K-np.sum(vals)].append((arr, hash_int))
                assert attempts < cutoff, 'No feasible graphs found for '+str(num_removed)+' edges removed'

    return all_W, W_per_num_edge_removals

def LoadData():
    in_file=open("./datasets/__partial_graphs/Manhattan_N=3,L=4,R=100,Ndir=False/_databank_full","rb")
    databank_full=pickle.load(in_file)
    in_file.close()
    in_file=open("./datasets/__partial_graphs/Manhattan_N=3,L=4,R=100,Ndir=False/_partial_graph_register","rb")
    partial_graph_register=pickle.load(in_file)
    in_file.close()
    in_file=open("./datasets/__partial_graphs/Manhattan_N=3,L=4,R=100,Ndir=False/_reachable_by_pursuers","rb")
    reachable_by_pursuers=pickle.load(in_file)
    in_file.close()
    in_file=open("./datasets/__partial_graphs/Manhattan_N=3,L=4,R=100,Ndir=False/_solvable","rb")
    solvable=pickle.load(in_file)
    in_file.close()
    return databank_full, partial_graph_register, solvable, reachable_by_pursuers

# def GetPartialGraphEnvironments_Manh3x3(state_repr, state_enc, edge_removals, U, solvable=True, reachable_for_units=True):
#     config={
#         'graph_type': "Manhattan",
#         'make_reflexive': True,
#         'N': 3,    # number of nodes along one side
#         'U': 2,    # number of pursuer units
#         'L': 4,    # Time steps
#         'T': 7,
#         'R': 100,  # Number of escape routes sampled 
#         'direction_north': False,       # Directional preference of escaper
#         'loadAllStartingPositions': False
#     }
#     databank_full, register_full, solvable, reachable = LoadData()
#     all_envs=[]
#     for e in edge_removals:
#         for W_, hashint, hashstr in register_full[e]:
#         #W_, hashint, hashstr = random.choice(register_full[4])
#             env_data = databank_full['U='+str(U)][hashint] # dict contains  'register':{(e0,U0):index}, 'databank':[], 'iratios':[]
#             env_data['W'] = W_
#             env = GraphWorldFromDatabank(config,env_data,optimization_method='static',state_representation=state_repr,state_encoding=state_enc)
#             s = solvable['U=2'][hashint]
#             r = reachable['U=2'][hashint]
#             if solvable and reachable_for_units:
#                 valids = np.logical_and(s,r)
#             elif not solvable and reachable_for_units:
#                 valids = np.logical_and(np.logical_not(s),r)
#             if valids.sum() > 0:
#                 env.world_pool = list(np.array(env.all_worlds)[valids])
#                 env.reset()
#                 all_envs.append(env)
#     return all_envs
