from inspect import CORO_CREATED
from modules.sim.graph_factory import get_all_edge_removals_symmetric
from modules.rl.rl_custom_worlds import GetCustomWorld
from modules.sim.simdata_utils import SimulateInteractiveMode
from modules.rl.environments import GraphWorld
import networkx as nx
import random

config={
    'graph_type': "Manhattan",
    'make_reflexive': True,
    'N': 3,    # number of nodes along one side
    'U': 2,    # number of pursuer units
    'L': 4,    # Time steps
    'T': 7,
    'R': 100,  # Number of escape routes sampled 
    'direction_north': False,       # Directional preference of escaper
    'loadAllStartingPositions': False
}

state_repr='et'
state_enc='nodes'
env = GraphWorld(config, optimization_method='static', fixed_initial_positions=None, state_representation=state_repr, state_encoding=state_enc)

#world_name='MetroU3_e17tborder_FixedEscapeInit'
#world_name='Manhattan5x5_FixedEscapeInit'
#env = GetCustomWorld(world_name, make_reflexive=True, state_repr=state_repr, state_enc='nodes')
W_all, W_per_num_edge_removals = get_all_edge_removals_symmetric(
        W_          = nx.convert_matrix.to_numpy_matrix(env.sp.G),
        start_node  = env.sp.labels2nodeids[env.state[0]],
        target_nodes= [env.sp.labels2nodeids[i] for i in env.sp.target_nodes],
        removals    = [8,12,16],
        instances_per_num_removed = 2    
    )
for k,v in W_per_num_edge_removals.items():
    # v: list of tuples
    print(k,len(v))

# Plot example graphs
import copy
env0=copy.deepcopy(env)
for k,val in W_per_num_edge_removals.items():#len(W_per_num_edge_removals[2])):
    #if len(val)>0:
    if k!=9: continue
    for i,v in enumerate(val):
        #i=random.randint(0,len(v)-1)
        
        W=v[0]
        #W=v[0][0]
        H = nx.from_numpy_matrix(W, create_using=nx.DiGraph()).to_undirected()
        H = nx.relabel_nodes(H, env.sp.nodeid2coord)

        S = [H.subgraph(c).copy() for c in nx.connected_components(H)]
                   
        nodeid2coord_new={}
        valid=True
        for s in S:
            if (1,0) in s.nodes():
                W_new=nx.convert_matrix.to_numpy_matrix(s.to_directed())
                for j, coord in enumerate(s.nodes()):
                    nodeid2coord_new[j]=coord
            else:
                if len(s.nodes())>1:
                    valid=False
        if valid:
            env.redefine_graph_structure(W_new,nodeid2coord_new,new_nodeids=True)
            env.render(mode=None, fname='graph_e'+str(k)+'_i'+str(i))
            #SimulateInteractiveMode(env)
            env=copy.deepcopy(env0)#SimulateInteractiveMode(env)
            #break

