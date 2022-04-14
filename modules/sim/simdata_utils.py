#from sim_visualization import plot_results
import copy
#from dataclasses import dataclass
import time
import networkx as nx
import numpy as np
#import matplotlib.image as mpimg
#import matplotlib.pyplot as plt
from datetime import datetime
#import plotly.graph_objects as go
import random
import torch
import torch.nn as nn 
import torch.nn.functional as F
from pathlib import Path
import os
import pickle
from modules.sim.sim_graphs import SparseManhattanGraph, graph, CircGraph, TKGraph, MetroGraph, MemGraph, MemGraphLong
from modules.rl.rl_plotting import PlotAgentsOnGraph
from modules.optim.escape_route_generator_MC import mutiple_escape_routes
from modules.optim.optimization_FIP_gurobipy import unit_ranges, optimization_alt, optimization
from modules.rl.rl_utils import EvalArgs2
from torch_geometric.utils.convert import from_networkx
from torch_geometric.utils import dense_to_sparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SimParameters(object):
    def __init__(self):
        self.graph_type = None
        self.G = None 
        self.W = None               # Adjacency matrix (numpy)  (V,V) V=num nodes
        self.EI= None               # Edge index (torch tensor) (2,E) E=num edges
        self.hashint = -1           # hash for M3x3 experiments, to identify edge removals
        self.labels = None
        self.pos = None
        self.N = None
        self.coord2nodeid = None
        self.coord2labels = None
        self.labels2coord = None
        self.labels2nodeids = {}
        self.nodeids2labels = {}
        self.nodeid2coord = None
        self.U = None
        self.L = None
        self.R = None
        self.V = None
        self.T = None
        self.direction_north = None
        self.start_escape_route = None
        self.start_escape_route_node = None
        self.most_northern_y = None
        self.most_eastern_x = None
        self.target_nodes = None
        self.loadAllStartingPositions = False
        self.spath_to_target = None
        self.spath_length = None
    def __str__(self):
        out = self.graph_type
        out += ', ('+str(self.N)+'x'+str(self.N)+') nodes, ...'
        return out
    
    def CalculateShortestPath(self):
        if self.target_nodes == []:
            self.spath_to_target=[]
            self.spath_length=0
        else:
            min_cost = 1e9
            for target_node_label in self.target_nodes:
                target_node_coord = self.labels2coord[target_node_label]
                try:
                    cost, path = nx.single_source_dijkstra(self.G, self.start_escape_route, target_node_coord, weight='weight')
                except:
                    cost, path = 1e8, [self.start_escape_route]
                if cost < min_cost:
                    best_path_coords = path
                    min_cost = cost
            self.spath_to_target = []
            for coord in best_path_coords:
                self.spath_to_target.append(self.coord2labels[coord])
            self.spath_length = int(min_cost)
        return self.spath_to_target, self.spath_length


def GetStateEncodingDimension(state_representation, V, U):
    if state_representation == 'et':
        state_dim=V
        chunks=[(0,)]
        state_len=1
    elif state_representation == 'etUt':
        state_dim= 2*V
        chunks=[(0,), tuple([1+i for i in range(U)]) ]
        state_len=1+U
    elif state_representation == 'ete0U0':
        state_dim= 2*V
        chunks=[(0,), tuple([1+i for i in range(U+1)])]
        state_len=2+U
    elif state_representation == 'etUte0U0':
        state_dim= 3*V
        chunks=[(0,), tuple([1+i for i in range(U)]), tuple([1+U+i for i in range(U+1)]) ]
        state_len=2*(1+U)
    else:
        assert False
    #print('State encoding vector dim:',state_dim, 'State node based dim:',state_len)
    return state_dim, chunks, state_len

def GetWorldPool(all_worlds, fixed_initial_positions, register):
    if fixed_initial_positions == None:
        return all_worlds
    elif type(fixed_initial_positions[0]) == tuple:
        return [register['coords'][fixed_initial_positions]]
    elif type(fixed_initial_positions[0]) == int:
        return [register['labels'][fixed_initial_positions]]
    else:
        assert False

def GetConfigs():
    configs = {
        "MemoryTaskU1": {
            # Note: ...
            'graph_type': "MemGraph",
            'make_reflexive': True,
            'N': 8,    # number of nodes along one side
            'U': 1,    # number of pursuer units
            'L': 2,    # Time steps
            'T': 5,
            'R': 10,  # Number of escape routes sampled 
            'direction_north': False,       # Directional preference of escaper
            'start_escape_route': '.', # Initial position of escaper
            'fixed_initial_positions': (3,0),
            'loadAllStartingPositions': False
        },       
        "MemoryTaskU1Long": {
            # Note: ...
            'graph_type': "MemGraphLong",
            'make_reflexive': True,
            'N': 11,    # number of nodes along one side
            'U': 1,    # number of pursuer units
            'L': 5,    # Time steps
            'T': 6,
            'R': 10,  # Number of escape routes sampled 
            'direction_north': False,       # Directional preference of escaper
            'start_escape_route': '.', # Initial position of escaper
            'fixed_initial_positions': (4,0),
            'loadAllStartingPositions': False
        },        
        "MemoryTaskU2": {
            # Note: ...
            'graph_type': "MemGraph",
            'make_reflexive': True,
            'N': 8,    # number of nodes along one side
            'U': 2,    # number of pursuer units
            'L': 2,    # Time steps
            'T': 5,
            'R': 10,  # Number of escape routes sampled 
            'direction_north': False,       # Directional preference of escaper
            'start_escape_route': '.', # Initial position of escaper
            'fixed_initial_positions': (3,0,6),
            'loadAllStartingPositions': False
        },        
        "MetroGraphU3": {
            # Note: E starting position is center node 17
            'graph_type': "MetroGraph",
            'make_reflexive': True,
            'N': 33,    # number of nodes along one side
            'U': 3,    # number of pursuer units
            'L': 6,    # Time steps
            'T': 20,
            'R': 500,  # Number of escape routes sampled 
            'direction_north': False,       # Directional preference of escaper
            'start_escape_route': 'bottom_center', # Initial position of escaper (always bottom center)
            'fixed_initial_positions': (17,5,7,28),
            'loadAllStartingPositions': False
        },
        "MetroGraphU4": {
            'graph_type': "MetroGraph",
            'make_reflexive': True,
            # Note: E starting position is center node 17
            'N': 33,    # number of nodes along one side
            'U': 4,    # number of pursuer units
            'L': 6,    # Time steps
            'T': 20,
            'R': 500,  # Number of escape routes sampled 
            'direction_north': False,       # Directional preference of escaper
            'start_escape_route': 'bottom_center', # Initial position of escaper (always bottom center)
            'fixed_initial_positions': (17,5,7,28),
            'loadAllStartingPositions': False
        },
        "MetroGraphU3L8_node1": {
            # Note: E starting position is bottom left node 1
            'graph_type': "MetroGraph",
            'make_reflexive': True,            
            'N': 33,    # number of nodes along one side
            'U': 3,    # number of pursuer units
            'L': 8,    # Time steps
            'T': 25,
            'R': 500,  # Number of escape routes sampled 
            'direction_north': False,       # Directional preference of escaper
            'start_escape_route': 'bottom_center', # Initial position of escaper (always bottom center)
            'fixed_initial_positions': (1,5,7,28),
            'loadAllStartingPositions': False
        },
        "MetroGraphU4L8_node1": {
            # Note: E starting position is bottom left node 1
            'graph_type': "MetroGraph",
            'make_reflexive': True,            
            'N': 33,    # number of nodes along one side
            'U': 4,    # number of pursuer units
            'L': 8,    # Time steps
            'T': 25,
            'R': 500,  # Number of escape routes sampled 
            'direction_north': False,       # Directional preference of escaper
            'start_escape_route': 'bottom_center', # Initial position of escaper (always bottom center)
            'fixed_initial_positions': (1,5,7,28),
            'loadAllStartingPositions': False
        },
        "Manhattan3": {
            'graph_type': "Manhattan",
            'make_reflexive': True,
            'N': 3,    # number of nodes along one side
            'U': 2,    # number of pursuer units
            'L': 4,    # Time steps
            'T': 7,
            'R': 100,  # Number of escape routes sampled 
            'direction_north': False,       # Directional preference of escaper
            'start_escape_route': 'bottom_center', # Initial position of escaper (always bottom center)
            'fixed_initial_positions': (1,6,8),
            'loadAllStartingPositions': False
        },
        "Manhattan5": {
            'graph_type': "Manhattan",
            'make_reflexive': True,            
            'N': 5,    # number of nodes along one side
            'U': 3,    # number of pursuer units
            'L': 6,    # Time steps
            'T': 11,
            'R': 200,  # Number of escape routes sampled 
            'direction_north': False,       # Directional preference of escaper
            'start_escape_route': 'bottom_center', # Initial position of escaper (always bottom center)
            'fixed_initial_positions': (2,15,19,22),
            'loadAllStartingPositions': False
        },
        "Manhattan11": {
            'graph_type': "Manhattan",
            'make_reflexive': True,            
            'N': 11,    # number of nodes along one side
            'U': 3,    # number of pursuer units
            'L': 16,    # Time steps
            'T': 23,
            'R': 1000,  # Number of escape routes sampled 
            'direction_north': False,       # Directional preference of escaper
            'start_escape_route': 'bottom_center', # Initial position of escaper (always bottom center)
            'fixed_initial_positions': (5,107,110,114),
            'loadAllStartingPositions': False
        },
        "CircGraph": {
            'graph_type': "CircGraph",
            'make_reflexive': True,            
            'N': 10,    # number of nodes along one side
            'U': 2,    # number of pursuer units
            'L': 6,    # Time steps
            'T': 7,
            'R': 100,  # Number of escape routes sampled 
            'direction_north': False,       # Directional preference of escaper
            'start_escape_route': 'bottom_center', # Initial position of escaper (always bottom center)
            'fixed_initial_positions': (9,1,2),
            'loadAllStartingPositions': False
        },
        "TKGraph": {
            'graph_type': "TKGraph",
            'make_reflexive': True,            
            'N': 6,    # number of nodes along one side
            'U': 1,    # number of pursuer units
            'L': 4,    # Time steps
            'T': 5,
            'R': 100,  # Number of escape routes sampled 
            'direction_north': False,       # Directional preference of escaper
            'start_escape_route': 'left', # Initial position of escaper (always bottom center)
            'loadAllStartingPositions': False
        },
        "SparseManhattan5x5" : {
            'graph_type': "SparseManhattan",
            'N': 5,    # number of nodes along one side
            'U': 3,    # number of pursuer units
            'L': 6,    # Time steps
            'T': 11,
            'R': 200,  # Number of escape routes sampled 
            'direction_north': False,       # Directional preference of escaper
            'start_escape_route': 'bottom_center' # Initial position of escaper (always bottom center)
        },
    }
    return configs



def DefineSimParameters(config):
    sp = SimParameters()
    sp.graph_type = config['graph_type']
    sp.U = config['U']              # number of pursuer units
    sp.L = config['L']              # Time steps
    sp.R = config['R']              # Number of escape routes sampled 
    sp.T = config['T']
    sp.loadAllStartingPositions = config['loadAllStartingPositions']
    
    if 'NWBGraph' in sp.graph_type:
        sp.G=config['obj']['G']
        sp.labels=config['obj']['labels']
        sp.pos = config['obj']['pos']
        sp.N=len(sp.G.nodes)
        sp.nodeid2coord = dict( (i, n) for i,n in enumerate(sp.G.nodes()) )
        sp.U = config['U']              # number of pursuer units
        sp.L = config['L']              # Time steps
        sp.R = config['R']              # Number of escape routes sampled 1000
        sp.V = sp.N                  # Total number of vertices
        sp.T = sp.L+1                   # Total steps in time taken (L + start node)
        sp.direction_north = config['direction_north']
        sp.start_escape_route = config['obj']['centernode_coord']
        sp.target_nodes=config['obj']['target_nodes']
    
    if sp.graph_type == 'SparseManhattan':
        sp.G, sp.labels, sp.pos = SparseManhattanGraph(config['N'])
        sp.N = config['N']
        sp.V = sp.N**2        # Total number of vertices
        #sp.T = (sp.N)*2+1     # Max timesteps for running experiments
        sp.direction_north = config['direction_north']
        sp.nodeid2coord = dict( (i, n) for i,n in enumerate(sp.G.nodes()) )
        sp.start_escape_route = (sp.N//2,0) # bottom center of grid
        sp.most_northern_y = max([c[1] for c in sp.G.nodes])
        #sp.target_nodes = set([sp.labels[sp.N//2,sp.N-1]])
        sp.target_nodes = set([sp.N*(sp.N-1)+i for i in range(sp.N)])
    if sp.graph_type == 'Manhattan':
        sp.G, sp.labels, sp.pos = graph(config['N'])
        sp.N = config['N']
        sp.V = sp.N**2        # Total number of vertices
        #sp.T = (sp.N)*2+1     # Max timesteps for running experiments
        sp.direction_north = config['direction_north']
        sp.nodeid2coord = dict( (i, n) for i,n in enumerate(sp.G.nodes()) )
        sp.start_escape_route = (sp.N//2,0) # bottom center of grid
        sp.most_northern_y = max([c[1] for c in sp.G.nodes])
        #sp.target_nodes = set([sp.labels[sp.N//2,sp.N-1]])
        sp.target_nodes = set([sp.N*(sp.N-1)+i for i in range(sp.N)])
    elif sp.graph_type == 'MetroGraph':
        sp.G, sp.labels, sp.pos = MetroGraph()#manhattan_graph(N)
        sp.N = config['N']
        sp.V = sp.N             # Total number of vertices (FIXED)
        #sp.T = sp.L+1         # Total steps in time taken (L + start node)
        sp.direction_north = False # (NOT VERY INTERESTING IF TRUE)
        sp.nodeid2coord = dict( (i, n) for i,n in enumerate(sp.G.nodes()) )
        sp.start_escape_route = sp.nodeid2coord[17]
        sp.most_northern_y = 18
        sp.most_eastern_x = 21
        if 'target_nodes' in config:
            sp.target_nodes = config['target_nodes']
        else:
            sp.target_nodes = set([0,1,2,3,4,9,10,15,20,26,30,31])
        #sp.target_nodes = set([ 0,4,9,20])
    elif sp.graph_type == 'CircGraph':
        sp.G, sp.labels, sp.pos = CircGraph()#manhattan_graph(N)
        sp.N = 10             # Number of nodes (FIXED)
        sp.V = 10             # Total number of vertices (FIXED)
        #sp.T = sp.L+1         # Total steps in time taken (L + start node)
        sp.direction_north = False # (NOT VERY INTERESTING IF TRUE)
        sp.nodeid2coord = dict( (i, n) for i,n in enumerate(sp.G.nodes()) )
        sp.start_escape_route = sp.nodeid2coord[9]
        sp.most_northern_y = max([c[1] for c in sp.G.nodes])
        sp.target_nodes = set([sp.labels[(3,6)]])
    elif sp.graph_type == 'MemGraph':
        sp.G, sp.labels, sp.pos = MemGraph()
        sp.N = 6             # Number of nodes (FIXED)
        sp.V = 8             # Total number of vertices (FIXED)
        #sp.T = sp.L+1         # Total steps in time taken (L + start node)
        sp.direction_north = False # (NOT VERY INTERESTING IF TRUE)
        sp.nodeid2coord = dict( (i, n) for i,n in enumerate(sp.G.nodes()) )
        sp.start_escape_route = sp.nodeid2coord[3]
        sp.most_northern_y = max([c[1] for c in sp.G.nodes])
        sp.target_nodes = set([2,5])
    elif sp.graph_type == 'MemGraphLong':
        sp.G, sp.labels, sp.pos = MemGraphLong()
        sp.N = 11             # Number of nodes (FIXED)
        sp.V = 11             # Total number of vertices (FIXED)
        #sp.T = sp.L+1         # Total steps in time taken (L + start node)
        sp.direction_north = False # (NOT VERY INTERESTING IF TRUE)
        sp.nodeid2coord = dict( (i, n) for i,n in enumerate(sp.G.nodes()) )
        sp.start_escape_route = sp.nodeid2coord[4]
        sp.most_northern_y = max([c[1] for c in sp.G.nodes])
        sp.target_nodes = set([3,7])

    elif sp.graph_type == 'TKGraph':
        sp.G, sp.labels, sp.pos = TKGraph()#manhattan_graph(N)
        sp.G = sp.G.to_undirected()
        sp.N = 7              # Number of nodes (FIXED)
        sp.V = 7              # Total number of vertices (FIXED)
        #sp.T = sp.L+1         # Total steps in time taken (L + start node)
        sp.direction_north = False # (NOT VERY INTERESTING IF TRUE)
        sp.nodeid2coord = dict( (i, n) for i,n in enumerate(sp.G.nodes()) )        
        sp.start_escape_route = sp.nodeid2coord[0]
        sp.most_northern_y = max([c[1] for c in sp.G.nodes])
        sp.target_nodes = set([4,6])
    # Define mappings between node naming conventions
    sp.coord2nodeid = dict( (n, i) for i,n in enumerate(sp.G.nodes()) )
    sp.coord2labels = sp.labels
    sp.labels2coord = dict( (v,k) for k,v in sp.coord2labels.items())
    for coord, nodeid in sp.coord2nodeid.items():
        label = sp.coord2labels[coord]
        sp.labels2nodeids[label]=nodeid
        sp.nodeids2labels[nodeid]=label
    if config['make_reflexive']:
        edgelist=[(e,e,{}) for e in sp.G.nodes()]
        sp.G.add_edges_from(edgelist)
    sp.start_escape_route_node = sp.coord2labels[sp.start_escape_route]
    sp.CalculateShortestPath()
    sp.W = torch.tensor(nx.to_numpy_matrix(sp.G),dtype=torch.float32)
    # make sure all nodes have the same attributes
    for key in sp.G.edges().keys():
        if len(sp.G.edges[key]) == 0:
            sp.G.edges[key]['N_pref']=-1
            sp.G.edges[key]['weight']=1
    sp.EI = dense_to_sparse(sp.W)[0] # [0] = edge_index, [1] = edge attributes
    # # test edge_index
    # pyg_graph = from_networkx(sp.G)
    # EI_test = pyg_graph.edge_index
    # S=set()
    # for i in range(EI_test.shape[1]):
    #     e=(EI_test[0,i].item(),EI_test[1,i].item())
    #     S.add(e)
    # for i in range(sp.EI.shape[1]):
    #     e = (sp.EI[0,i].item() ,sp.EI[1,i].item())
    #     assert (e) in S
    # assert sp.EI.shape == EI_test.shape
    # assert len(S) == sp.EI.shape[1]
    return sp

def GetUnitsInitialConditions(sp, register, specific_start_units, cutoff):
    if specific_start_units is not None:
        ulist = [sp.start_escape_route] + specific_start_units
        if tuple(ulist) in register:
            print('Initial condition already in register')
            assert False
    else:
        # Generate instance of starting conditions
        i=0
        #toprows = [(i,j) for i in range(0,sp.N) for j in range(sp.N-1,sp.N-3,-1)]
        coordlist = list(sp.G.nodes())
        while True:
            i+=1
            ulist=random.choices(coordlist,k=sp.U)
            #ulist=random.choices(toprows,k=sp.U)
            if sp.start_escape_route in ulist:
                continue
            ulist.sort()
            ulist=[sp.start_escape_route]+ulist
            if tuple(ulist) not in register:
                break
            if i>cutoff:
                #print('No new entries found')
                #print('full'+str(len(register)))
                ulist=-1
                return [-1]
    return ulist

import timeit
class FrameTimer():
    def __init__(self):
        self.s0= timeit.default_timer()
        self.s = self.s0
    def mark(self):
        e=timeit.default_timer()
        m=e-self.s
        self.s=e
        return m*1 # s
    def total(self):
        return (timeit.default_timer()-self.s0)*1 # s

def ObtainSimulationInstance(sp, register, specific_start_units=None, cutoff=1e5, print_InterceptRate=True, create_plot=False, method='ALT'):
    # Input:
    #   sp:         simulation parameters
    #   register:   dict ((start_escape_route),(start_unit1),(start_unit2),..)  -> index of databank
    #   specific_start_units: 
    #               list [(start_unit1),(start_unit2),..] size U, offers option to run specific initial conditions
    # 
    # Returns:
    #   new_registry_entry: new key for register entry
    #   new_databank_entry: list of unit paths, U entries, 

    # Get random initial conditions (positions of pursuit units), not yet in dataset
    ulist = GetUnitsInitialConditions(sp, register, specific_start_units, cutoff)
    if ulist[0] == -1:
        return ulist, None, None, None, None
    register_entry = (tuple(ulist),len(register))

    # Run optimization for instance
    start_escape_route = ulist[0]
    start_units = tuple(ulist[1:])
    routes_time_nodes, routes_time = mutiple_escape_routes(sp.G, sp.N, sp.L, sp.R, start_escape_route, sp.direction_north, start_units,sp.graph_type)

    units_range_time = unit_ranges(start_units, sp.U, sp.G, sp.L)

    if method == 'ALT':
        routes_intercepted, units_places = optimization_alt(sp.G, sp.U, routes_time_nodes, units_range_time, sp.R, sp.V, sp.L+1, sp.labels, start_units, sp.nodeid2coord)   
    elif method == 'IMA':
        routes_intercepted, units_places = optimization(sp.G, sp.U, routes_time_nodes, units_range_time, sp.R, sp.V, sp.L+1, sp.labels)   
    else:
        assert False
        
    interception_rate, num_intercepted = GetIR(sp.R, routes_intercepted)

    # Display results and prepare data to return
    if print_InterceptRate:
        print('Of the',sp.R,'sample escapte routes,',num_intercepted,'were intercepted. Interception rate = {:.2f}'.format(interception_rate))
    #if create_plot:
    #    plot_results(sp.G,  sp.R, sp.pos, routes_time, routes_time_nodes, start_escape_route, units_places, routes_intercepted)
    paths=[]
    for i,u in enumerate(units_places):
        paths.append(nx.dijkstra_path(sp.G, start_units[i], u)) 
    sim_instance = {
        'start_escape_route': start_escape_route,
        'start_units':        start_units,
        'paths':              paths
    }

    return register_entry, sim_instance, interception_rate, None, None#eval_time, np.array(time_marks)

def GetIR(R, routes_intercepted):
    num_intercepted=0
    for k,v in routes_intercepted.items():
        num_intercepted+=v
    return num_intercepted / R, num_intercepted



def GetGraphData(sp):
    G = sp.G.to_directed()
    neighbors_labels = {}
    for node_coord in G.nodes:
        local_neighbors_labels = []
        node_label = sp.coord2labels[node_coord]
        for n_coord in G.neighbors(node_coord):
            n_label = sp.coord2labels[n_coord]
            local_neighbors_labels.append(n_label)
        local_neighbors_labels.sort()
        neighbors_labels[node_label] = local_neighbors_labels

    max_outdegree=0
    outdegrees_labels={}
    for coord,deg in G.out_degree:
        outdegrees_labels[sp.coord2labels[coord]] = deg
        if deg>max_outdegree:
            max_outdegree=deg
    
    max_indegree=0
    indegrees_labels={}
    for coord,deg in G.in_degree:
        indegrees_labels[sp.coord2labels[coord]] = deg
        if deg>max_indegree:
            max_indegree=deg
    
    return neighbors_labels, indegrees_labels, max_indegree, outdegrees_labels, max_outdegree

def scale0_vec(a, newrange=[0.1,1.]):
    # scales all node distance to target scores to the range [0.1-1], irrespective of graph size
    aval=a[a!=0]
    lowmark=aval.min()
    b=aval-lowmark
    bmax=b.max()
    assert bmax > 0
    brange=newrange[1]-newrange[0]
    b=b/bmax * brange + newrange[0]
    #print(b)
    out=list(a)
    for i in range(len(out)):
        if out[i] != 0:
            #print(a[i])
            out[i] = (out[i]-lowmark)/bmax * brange + newrange[0]
    return torch.tensor(out,dtype=torch.float32)

from collections import deque
def GetNodeScores(sp, neighbors):
    if len(sp.target_nodes) > 0:
        #print('calculating node scores...')
        nodescores=torch.zeros(sp.V,dtype=torch.float32)
        scaled_nodescores=torch.zeros(sp.V,dtype=torch.float32)
        min_target_distances=torch.ones(sp.V)*torch.inf
        # go over all target nodes
        for t in sp.target_nodes:
            q = deque()
            visited = set([t])
            q.append((t,0))
            while len(q) > 0:
                c,d = q.popleft()
                d += 1
                for n in neighbors[c]:
                    if n not in sp.target_nodes and n not in visited: 
                        nodescores[n] += 1/d
                        min_target_distances[n] = min(d,min_target_distances[n])
                        visited.add(n)
                        q.append((n,d))
    
        max_score=nodescores.max()
        nodescores/=max_score
        scaled_nodescores = scale0_vec(nodescores, [0.1,1.])
    
        for n in sp.target_nodes:
            nodescores[n]=2
            scaled_nodescores[n]=2
            min_target_distances[n]=0

        infidx=min_target_distances==torch.inf
        min_target_distances[infidx]=min_target_distances[~infidx].max()+1
    return nodescores, scaled_nodescores, min_target_distances

def make_dirname(sp):
    #timestamp = datetime.now()
    dirname = "datasets/" \
        + str(sp.graph_type) + \
        "_N="+ str(sp.N) + \
        "_U="+ str(sp.U) + \
        "_L="+ str(sp.L) + \
        "_R="+ str(sp.R) + \
        "_Ndir="+ str(sp.direction_north)                        
    return dirname

def make_result_directory(sp, optimization_method='static'):
    ######## Create folder for results ########
    dirname = make_dirname(sp)
    # dirname = "results/" + str(config['name'])
    if optimization_method == 'dynamic' or sp.loadAllStartingPositions:
        dirname += '_allE'
    Path(dirname).mkdir(parents=True, exist_ok=True)
    return dirname

def LoadDatafile(dirname):
    all_lengths_fnames = [f for f in os.listdir(dirname) if f.endswith('.pkl')]
    if len(all_lengths_fnames)==0:
        print('No database found.')
        return {},[],[] # We start a new database
    else:
        biggest_dataset_fname = sorted(all_lengths_fnames, key=lambda s: float(s.split('.pkl')[0].split('_')[-1]))[-1]
        in_file = open(dirname + "/" + biggest_dataset_fname, "rb")
        results = pickle.load(in_file)
        in_file.close()
        print('Database found, contains',len(results['databank']),'entries.',in_file.name)
        return results['register'], results['databank'], results['interception_ratios']

def SaveResults(dirname,sp,register,databank,iratios):
    # Check if current databank is bigger than any previously saved databank for this config. If so, return without saving
    all_lengths_fnames = [f for f in os.listdir(dirname) if f.endswith('.pkl')]
    if len(all_lengths_fnames) != 0:
        biggest_dataset_fname = sorted(all_lengths_fnames, key=lambda s: float(s.split('.pkl')[0].split('_')[-1]))[-1]
        saved_dset_size = int(biggest_dataset_fname.split('.pkl')[0].split('_')[-1])
        if len(databank) <= saved_dset_size:
            print('No file saved, dataset size was not increased')
            return
    
    # Save results
    results= {
        'Simulation_parameters':sp,
        'register': register,
        'databank': databank,
        'interception_ratios': iratios
    }
    tstamp = str(datetime.now()).replace(" ", "_").replace(".", "_").replace(":", "-")
    filename = tstamp+"_Databank_"+str(len(databank))+".pkl"
    out_file = open(dirname + "/" + filename, "wb")
    pickle.dump(results, out_file)
    out_file.close()

def SimulatePursuersPathways(conf, optimization_method='dynamic', fixed_initial_positions=None):
    # Escaper position static, plot progression of pursuers motion
    sp = DefineSimParameters(conf)
    dirname = make_result_directory(sp, optimization_method)
    register, databank, iratios = LoadDatafile(dirname)
    if fixed_initial_positions == None:
        dataframe=random.randint(0,len(iratios)-1)
    else:
        dataframe=register[fixed_initial_positions]
    start_escape_node = databank[dataframe]['start_escape_route']
    unit_paths = databank[dataframe]['paths']
    print('unit_paths',unit_paths,'intercept_rate',iratios[dataframe])
    for t in range(sp.L):
        e = sp.coord2labels[start_escape_node]
        p = []
        for P_path in unit_paths:
            pos = P_path[-1] if t >= len(P_path) else P_path[t]
            p.append(sp.coord2labels[pos])
        PlotAgentsOnGraph(sp, e, p, t)

def SimulateInteractiveMode(env, filesave_with_time_suffix=True, entry=None):
    if entry is not None:
        env.reset(entry)
    else:
        env.reset()
    print('ENTRY:',env.current_entry)
    s=env.state
    done=False
    R=0
    env.render(mode=None, fname="results/test", t_suffix=filesave_with_time_suffix)
    while not done:
        print('e position:',env.state[0],env.sp.labels2coord[env.state[0]])
        print('u paths (node labels):',env.u_paths)
        print('u paths (node coords): ',end='')
        for p in env.u_paths:
            print('[',end='')
            for e in p:
                print(str(env.sp.labels2coord[e])+',',end='')
            print('],  ',end='')
        print('\nu positions per time-step:')
        for t in range(env.sp.T):
            print(env.getUpositions(t))
            if t < env.sp.T-1 and env.getUpositions(t) == env.getUpositions(t+1):
                print('[., ., .]')
                break
        print('------------')
        print('Current state:',s, 'spath to goal', env.sp.spath_to_target, '('+str(env.sp.spath_length)+' steps)')
        print('Current obs:','\n'+str(env.obs) if env.state_encoding=='nfm' else env.obs)
        n = env.neighbors[env.state[0]]
        print('Available actions: ',n)
        while True:
            a=input('Action nr '+str(env.global_t+1)+'/max '+str(env.sp.T)+' (new node)?  > ')
            if (a.isnumeric() and int(a) in n) or a.lower()=='q': break
        if a.lower() == 'q': break
        print()
        a=n.index(int(a))
        s,r,done,_=env.step(int(a))
        s=env.state
        env.render_eupaths(mode=None, fname="results/test", t_suffix=filesave_with_time_suffix, last_step_only=True)
        R+=r
    print('\n******************** done, reward='+str(R),'**********************')
    #input('> Press any key to continue')
    env.render_eupaths(mode=None, fname="results/final", t_suffix=filesave_with_time_suffix)
    input('> Press any key to continue')
    print('\n')
    return a

def SimulateAutomaticMode_DQN(env, dqn_policy, t_suffix=True, entries=None):
    if entries is not None:
        entry=random.choice(entries)
    else: entry = None
    obs=env.reset(entry=entry)
    print('Entry:',env.current_entry)
    print('spath length',env.sp.spath_length,'path',env.sp.spath_to_target)
    
    done=False
    endepi=False
            
    eval_arg_func = EvalArgs2
    while not done:
        action, _state = dqn_policy.sample_action(*eval_arg_func(env),printing=True)
        
        env.render_eupaths(fname='results/test',t_suffix=t_suffix,last_step_only=True)
        #ppo_value = ppo_policy.model.predict_values(obs[None,:].to(device))

        while True:
            a=input('\n             [q]-stop current, [enter]-take step, [n]-show nfm ...> ')
            if a.lower() == 'q':
                endepi=True
                break
            if a == 'n': print(env.obs)
            if a == 'c': env.render_eupaths(fname='results/final',t_suffix=t_suffix,last_step_only=False)
            if a == '': break
        if endepi:
            break        
        obs,r,done,i = env.step(action)
    env.render_eupaths(fname='results/test',t_suffix=t_suffix,last_step_only=True)
    env.render_eupaths(fname='results/final',t_suffix=t_suffix)
    if a.lower()!='q':
        input('')
    return a

def SimulateInteractiveMode_PPO(env, model=None, filesave_with_time_suffix=True, entry=None):
    if entry is not None:
        obs=env.reset(entry)
    else:
        obs=env.reset()
    print('\nENTRY:',env.current_entry)
    s=env.state
    done=False
    R=0
    env.render(mode=None, fname="results/test", t_suffix=filesave_with_time_suffix)
    #env.render(mode=None, fname="results/test", t_suffix=True)
    endepi=False
    while not done:
        print('e position:',env.state[0],env.sp.labels2coord[env.state[0]])
        print('u paths (node labels):',env.u_paths)
        print('u paths (node coords): ',end='')
        for p in env.u_paths:
            print('[',end='')
            for e in p:
                print(str(env.sp.labels2coord[e])+',',end='')
            print('],  ',end='')
        print('\nu positions per time-step:')
        for t in range(env.sp.T):
            print(env.getUpositions(t))
            if t < env.sp.T-1 and env.getUpositions(t) == env.getUpositions(t+1):
                print('[., ., .]')
                break
        print('------------')
        print('Current state:',s, 'spath to goal', env.sp.spath_to_target, '('+str(env.sp.spath_length)+' steps)')
        n = env.neighbors[env.state[0]]
        print('Available actions: ',n,end='')

        if model is not None:
            action_masks = env.action_masks()
            ppo_action, ppo_state = model.policy.predict(obs, deterministic=True, action_masks=action_masks)
            distro = model.policy.get_distribution(obs[None].to(device))
            actionlist = torch.tensor(env.neighbors[env.state[0]]).to(device)
            ppo_probs = F.softmax(distro.log_prob(actionlist),dim=0)
            ppo_value = model.policy.predict_values(obs[None].to(device))
            np.set_printoptions(formatter={'float':"{0:0.2f}".format})
            ppo_probs=ppo_probs.detach().cpu().numpy()
            print('; PPO action probs: ',ppo_probs,'Action:',n[np.argmax(ppo_probs)],'; Estimated value of current graph state:', ppo_value.detach().cpu().numpy())
        else:
            print()

        while True:
            a=input('           Action nr '+str(env.global_t+1)+'/max '+str(env.sp.T)+' (new node) [q=quit][n=print obs]?  > ')
            if a.isnumeric() and int(a) in n: break
            if a.lower() == 'q':
                endepi=True
                break
            if a == 'n':
                print('Current obs:','\n'+str(env.obs) if env.state_encoding=='nfm' else env.obs)
        if endepi:
            break
        print()
        #a=n.index(int(a))
        obs,r,done,_=env.step(int(a))
        s=env.state
        env.render_eupaths(mode=None, fname="results/test", t_suffix=filesave_with_time_suffix, last_step_only=True)
        R+=r
    print('\n******************** done, reward='+str(R),'**********************')
    #input('> Press any key to continue')
    env.render_eupaths(mode=None, fname="results/final", t_suffix=filesave_with_time_suffix)
    input('> Press any key to continue')
    print('\n')
    return a

def SimulateAutomaticMode_PPO(env, ppo_policy, t_suffix=True, entries=None):
    if entries is not None:
        entry=random.choice(entries)
    else: entry = None
    obs=env.reset(entry=entry)
    ppo_policy.reset_hidden_states()
    print('Entry:',env.current_entry)
    
    done=False
    endepi=False
    R=0        
    while not done:
        action, _state = ppo_policy.sample_greedy_action(obs, torch.tensor(env.neighbors[env.state[0]]), printing=True)
        
        env.render_eupaths(fname='results/test',t_suffix=t_suffix,last_step_only=True)
        #ppo_value = ppo_policy.model.predict_values(obs)#[None,:].to(device))

        while True:
            a=input('\n             [q]-stop current, [enter]-take step, [n]-show nfm ...> ')
            if a.lower() == 'q':
                endepi=True
                break
            if a == 'n': print(env.obs)
            if a == 'c': env.render_eupaths(fname='results/final',t_suffix=t_suffix,last_step_only=False)
            if a == '': break
        if endepi:
            break        
        obs,r,done,i = env.step(action)
        R+=r
    print('Done, R=',R,'\n\n')
    env.render_eupaths(fname='results/test',t_suffix=t_suffix,last_step_only=True)
    env.render_eupaths(fname='results/final',t_suffix=t_suffix)
    if a.lower()!='q':
        input('')
    return a


def SimulateAutomaticMode_PPO_LSTM(env, ppo_policy_lstm, ppo_policy_no_lstm, t_suffix=True, entries=None):
    if entries is not None:
        entry=random.choice(entries)
    else: entry = None
    obs=env.reset(entry=entry)
    env2 = copy.deepcopy(env)
    obs2=env2.reset(entry=env.current_entry)
    assert env2.state == env.state
    assert torch.allclose(obs,obs2)

    ppo_policy_lstm.reset_hidden_states()
    ppo_policy_no_lstm.reset_hidden_states()
    
    print('Entry:',env.current_entry)
    
    done=False
    done2=False
    endepi=False
    R=0     
    R2=0   
    while not (done and done2):
        if not done:
            action, _state = ppo_policy_lstm.sample_greedy_action(obs, torch.tensor(env.neighbors[env.state[0]]), printing=True)
            env.render_eupaths(fname='results/test_lstm',t_suffix=t_suffix,last_step_only=True)
        if not done2:
            print()
            action2, _state2 = ppo_policy_no_lstm.sample_greedy_action(obs2, torch.tensor(env2.neighbors[env2.state[0]]), printing=True)
            env2.render_eupaths(fname='results/test_no_lstm',t_suffix=t_suffix,last_step_only=True)

        while True:
            a=input('\n             [q]-stop current, [enter]-take step, [n1 or n2]-show nfm ...> ')
            if a.lower() == 'q':
                endepi=True
                break
            if a == 'n1': print(env.obs)
            if a == 'n2': print(env2.obs)
            if a == 'c': 
                env.render_eupaths(fname='results/final_lstm',t_suffix=t_suffix,last_step_only=False)
                env2.render_eupaths(fname='results/final_no_lstm',t_suffix=t_suffix,last_step_only=False)
            if a == '': break
        if endepi:
            break        
        if not done:
            obs,r,done,i = env.step(action)
        else:
            env.render_eupaths(fname='results/test_lstm',t_suffix=t_suffix,last_step_only=True)

        if not done2:
            obs2,r2,done2,i2 = env2.step(action2)
        else:
            env2.render_eupaths(fname='results/test_no_lstm',t_suffix=t_suffix,last_step_only=True)
        R+=r
        R2+=r2
    print('Done, R=',R,', R_no_lstm=',R2,'\n\n')
    env.render_eupaths(fname='results/final_lstm',t_suffix=t_suffix)
    env2.render_eupaths(fname='results/final_no_lstm',t_suffix=t_suffix)
    if a.lower()!='q':
        input('')
    return a