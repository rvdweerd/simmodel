#from sim_visualization import plot_results
#from dataclasses import dataclass
#import time
#import networkx as nx
#import matplotlib.image as mpimg
#import matplotlib.pyplot as plt
#from datetime import datetime
#import plotly.graph_objects as go
import random
from pathlib import Path
import os
import pickle
from sim_graphs import graph, CircGraph, TKGraph, MetroGraph
from rl_plotting import PlotAgentsOnGraph

class SimParameters(object):
    def __init__(self):
        self.graph_type = None
        self.G = None 
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
        self.most_northern_y = None
        self.most_eastern_x = None
        self.target_nodes = None
        self.loadAllStartingPositions = False
    def __str__(self):
        out = self.graph_type
        out += ', ('+str(self.N)+'x'+str(self.N)+') nodes, ...'
        return out

def GetStateEncodingDimension(state_representation, V, U):
    if state_representation == 'et':
        state_dim=V
        chunks=[(0,)]
    elif state_representation == 'etUt':
        #state_len=1+U
        state_dim= 2*V
        chunks=[(0,), tuple([1+i for i in range(U)]) ]
        #return (1+U)*V
    elif state_representation == 'ete0U0':
        #return (2+U)*V, None
        #state_len=1+1+U
        state_dim= 2*V
        chunks=[(0,), tuple([1+i for i in range(U+1)])]
        #return (2+U)*V
    elif state_representation == 'etUte0U0':
        #state_len=1+U+1+U
        state_dim= 3*V
        chunks=[(0,), tuple([1+i for i in range(U)]), tuple([1+U+i for i in range(U+1)]) ]
        #return 2*(1+U)*V
    else:
        assert False
    print('State encoding vector dim:',state_dim)
    return state_dim, chunks

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
        "MetroGraphU3": {
            # Note: E starting position is center node 17
            'graph_type': "MetroGraph",
            'make_reflexive': False,
            'N': 33,    # number of nodes along one side
            'U': 3,    # number of pursuer units
            'L': 6,    # Time steps
            'R': 500,  # Number of escape routes sampled 
            'direction_north': False,       # Directional preference of escaper
            'start_escape_route': 'bottom_center', # Initial position of escaper (always bottom center)
            'fixed_initial_positions': (17,5,7,28),
            'loadAllStartingPositions': False
        },
        "MetroGraphU4": {
            'graph_type': "MetroGraph",
            'make_reflexive': False,
            # Note: E starting position is center node 17
            'N': 33,    # number of nodes along one side
            'U': 4,    # number of pursuer units
            'L': 6,    # Time steps
            'R': 500,  # Number of escape routes sampled 
            'direction_north': False,       # Directional preference of escaper
            'start_escape_route': 'bottom_center', # Initial position of escaper (always bottom center)
            'fixed_initial_positions': (17,5,7,28),
            'loadAllStartingPositions': False
        },
        "MetroGraphU3L8_node1": {
            # Note: E starting position is bottom left node 1
            'graph_type': "MetroGraph",
            'make_reflexive': False,            
            'N': 33,    # number of nodes along one side
            'U': 3,    # number of pursuer units
            'L': 8,    # Time steps
            'R': 500,  # Number of escape routes sampled 
            'direction_north': False,       # Directional preference of escaper
            'start_escape_route': 'bottom_center', # Initial position of escaper (always bottom center)
            'fixed_initial_positions': (1,5,7,28),
            'loadAllStartingPositions': False
        },
        "Manhattan3": {
            'graph_type': "Manhattan",
            'make_reflexive': False,
            'make_reflexive': False,                        
            'N': 3,    # number of nodes along one side
            'U': 2,    # number of pursuer units
            'L': 4,    # Time steps
            'R': 100,  # Number of escape routes sampled 
            'direction_north': True,       # Directional preference of escaper
            'start_escape_route': 'bottom_center', # Initial position of escaper (always bottom center)
            'fixed_initial_positions': (1,6,8),
            'loadAllStartingPositions': False
        },
        "Manhattan5": {
            'graph_type': "Manhattan",
            'make_reflexive': False,            
            'N': 5,    # number of nodes along one side
            'U': 3,    # number of pursuer units
            'L': 6,    # Time steps
            'R': 200,  # Number of escape routes sampled 
            'direction_north': True,       # Directional preference of escaper
            'start_escape_route': 'bottom_center', # Initial position of escaper (always bottom center)
            'fixed_initial_positions': (2,15,19,22),
            'loadAllStartingPositions': False
        },
        "Manhattan11": {
            'graph_type': "Manhattan",
            'make_reflexive': False,            
            'N': 11,    # number of nodes along one side
            'U': 3,    # number of pursuer units
            'L': 16,    # Time steps
            'R': 1000,  # Number of escape routes sampled 
            'direction_north': True,       # Directional preference of escaper
            'start_escape_route': 'bottom_center', # Initial position of escaper (always bottom center)
            'fixed_initial_positions': (5,107,110,114),
            'loadAllStartingPositions': False
        },
        "CircGraph": {
            'graph_type': "CircGraph",
            'make_reflexive': False,            
            'N': 10,    # number of nodes along one side
            'U': 2,    # number of pursuer units
            'L': 6,    # Time steps
            'R': 100,  # Number of escape routes sampled 
            'direction_north': False,       # Directional preference of escaper
            'start_escape_route': 'bottom_center', # Initial position of escaper (always bottom center)
            'fixed_initial_positions': (9,1,2),
            'loadAllStartingPositions': False
        },
        "TKGraph": {
            'graph_type': "TKGraph",
            'make_reflexive': False,            
            'N': 6,    # number of nodes along one side
            'U': 1,    # number of pursuer units
            'L': 4,    # Time steps
            'R': 10000,  # Number of escape routes sampled 
            'direction_north': False,       # Directional preference of escaper
            'start_escape_route': 'left', # Initial position of escaper (always bottom center)
            'loadAllStartingPositions': False
        },
    }
    return configs

def DefineSimParameters(config):
    sp = SimParameters()
    sp.graph_type = config['graph_type']
    sp.U = config['U']              # number of pursuer units
    sp.L = config['L']              # Time steps
    sp.R = config['R']              # Number of escape routes sampled 
    sp.loadAllStartingPositions = config['loadAllStartingPositions']
    if sp.graph_type == 'Manhattan':
        sp.G, sp.labels, sp.pos = graph(config['N'])
        sp.N = config['N']
        sp.V = sp.N**2        # Total number of vertices
        sp.T = sp.L+1         # Total steps in time taken (L + start node)
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
        sp.T = sp.L+1         # Total steps in time taken (L + start node)
        sp.direction_north = False # (NOT VERY INTERESTING IF TRUE)
        sp.nodeid2coord = dict( (i, n) for i,n in enumerate(sp.G.nodes()) )
        sp.start_escape_route = sp.nodeid2coord[17]
        sp.most_northern_y = 18
        sp.most_eastern_x = 21
        sp.target_nodes = set([ 0,1,2,3,4,9,10,15,20,26,30,31])
        #sp.target_nodes = set([ 0,4,9,20])
    elif sp.graph_type == 'CircGraph':
        sp.G, sp.labels, sp.pos = CircGraph()#manhattan_graph(N)
        sp.N = 10             # Number of nodes (FIXED)
        sp.V = 10             # Total number of vertices (FIXED)
        sp.T = sp.L+1         # Total steps in time taken (L + start node)
        sp.direction_north = False # (NOT VERY INTERESTING IF TRUE)
        sp.nodeid2coord = dict( (i, n) for i,n in enumerate(sp.G.nodes()) )
        sp.start_escape_route = sp.nodeid2coord[9]
        sp.most_northern_y = max([c[1] for c in sp.G.nodes])
        sp.target_nodes = set([sp.labels[(3,6)]])
    elif sp.graph_type == 'TKGraph':
        sp.G, sp.labels, sp.pos = TKGraph()#manhattan_graph(N)
        sp.N = 7              # Number of nodes (FIXED)
        sp.V = 7              # Total number of vertices (FIXED)
        sp.T = sp.L+1         # Total steps in time taken (L + start node)
        sp.direction_north = False # (NOT VERY INTERESTING IF TRUE)
        sp.nodeid2coord = dict( (i, n) for i,n in enumerate(sp.G.nodes()) )        
        sp.start_escape_route = sp.nodeid2coord[0]
        sp.most_northern_y = max([c[1] for c in sp.G.nodes])
        sp.target_nodes = set([])
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
    return sp

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

def make_result_directory(sp, optimization_method):
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

def SimulateInteractiveMode(env):    
    s=env.reset()
    done=False
    R=0
    env.render()
    while not done:
        print('e position:',s[0],env.sp.labels2coord[s[0]])
        print('u paths (node labels):',env.u_paths)
        print('u paths (node coords): ',end='')
        for p in env.u_paths:
            print('[',end='')
            for e in p:
                print(str(env.sp.labels2coord[e])+',',end='')
            print('],  ',end='')
        print('\nu positions per time-step:')
        for t in range(env.sp.T):
            print(env._getUpositions(t))
        print('------------')
        print('Current state:')
        print(s)
        print('Available actions:\n> [ ',end='')
        n = env.neighbors[s[0]]
        print(n)
        a=input(']\nAction (new node)?\n> ')
        a=n.index(int(a))
        s,r,done,_=env.step(int(a))
        env.render()
        R+=r
    print('done, reward='+str(R),'\n---------------')