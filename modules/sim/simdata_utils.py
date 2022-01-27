#from sim_visualization import plot_results
#from dataclasses import dataclass
import time
import networkx as nx
import numpy as np
#import matplotlib.image as mpimg
#import matplotlib.pyplot as plt
from datetime import datetime
#import plotly.graph_objects as go
import random
from pathlib import Path
import os
import pickle
from modules.sim.sim_graphs import SparseManhattanGraph, graph, CircGraph, TKGraph, MetroGraph
from modules.rl.rl_plotting import PlotAgentsOnGraph
from modules.optim.escape_route_generator_MC import mutiple_escape_routes
from modules.optim.optimization_FIP_gurobipy import unit_ranges, optimization_alt

class SimParameters(object):
    def __init__(self):
        self.graph_type = None
        self.G = None 
        self.W = None               # Adjacency matrix (numpy)
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
    elif sp.graph_type == 'TKGraph':
        sp.G, sp.labels, sp.pos = TKGraph()#manhattan_graph(N)
        sp.N = 7              # Number of nodes (FIXED)
        sp.V = 7              # Total number of vertices (FIXED)
        #sp.T = sp.L+1         # Total steps in time taken (L + start node)
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
    sp.W = nx.to_numpy_matrix(sp.G)        
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

def ObtainSimulationInstance(sp, register, specific_start_units=None, cutoff=1e5, print_InterceptRate=True, create_plot=False):
    # Input:
    #   sp:         simulation parameters
    #   register:   dict ((start_escape_route),(start_unit1),(start_unit2),..)  -> index of databank
    #   specific_start_units: 
    #               list [(start_unit1),(start_unit2),..] size U, offers option to run specific initial conditions
    # 
    # Returns:
    #   new_registry_entry: new key for register entry
    #   new_databank_entry: list of unit paths, U entries, 

    #start_time = time.time()
    #ft = FrameTimer()
    #time_marks=[]
    # Get random initial conditions (positions of pursuit units), not yet in dataset
    ulist = GetUnitsInitialConditions(sp, register, specific_start_units, cutoff)
    if ulist[0] == -1:
        return ulist, None, None, None, None
    register_entry = (tuple(ulist),len(register))
    #time_marks.append(ft.mark())

    # Run optimization for instance
    start_escape_route = ulist[0]
    start_units = tuple(ulist[1:])
    routes_time_nodes, routes_time = mutiple_escape_routes(sp.G, sp.N, sp.L, sp.R, start_escape_route, sp.direction_north, start_units,sp.graph_type)
    #time_marks.append(ft.mark())

    units_range_time = unit_ranges(start_units, sp.U, sp.G, sp.L)
    #time_marks.append(ft.mark())

    routes_intercepted, units_places = optimization_alt(sp.G, sp.U, routes_time_nodes, units_range_time, sp.R, sp.V, sp.L+1, sp.labels, start_units, sp.nodeid2coord)   
    #routes_intercepted, units_places = optimization(sp.G, sp.U, routes_time_nodes, units_range_time, sp.R, sp.V, sp.T, sp.labels)   
    #time_marks.append(ft.mark())

    interception_rate, num_intercepted = GetIR(sp.R, routes_intercepted)
    #time_marks.append(ft.mark())

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
    #eval_time = time.time() - start_time # seconds
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

def SimulateInteractiveMode(env):    
    #s=env.reset()
    s=env.state
    done=False
    R=0
    env.render(mode=None,fname="testrun")
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
            print(env._getUpositions(t))
        print('------------')
        print('Current state:')
        print(s)
        print('Available actions:\n> [ ',end='')
        n = env.neighbors[env.state[0]]
        print(n)
        a=input(']\nAction (new node)?\n> ')
        a=n.index(int(a))
        s,r,done,_=env.step(int(a))
        env.render(mode=None,fname="testrun")
        R+=r
    print('done, reward='+str(R),'\n---------------')