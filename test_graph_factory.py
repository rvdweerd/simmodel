from modules.sim.graph_factory import all_Manhattan3x3_symmetric_adj_matrices
from modules.rl.rl_custom_worlds import GetCustomWorld
from modules.rl.environments import GraphWorld
import networkx as nx

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

# https://networkx.org/documentation/networkx-1.7/reference/generated/networkx.convert.from_numpy_matrix.html
# G=nx.from_numpy_matrix(A)
# networkx.relabel_nodes
#world_name='Manhattan5x5_FixedEscapeInit'
#env = GetCustomWorld(world_name, make_reflexive=True, state_repr=state_repr, state_enc='nodes')
W_all, W_per_num_edge_removals=all_Manhattan3x3_symmetric_adj_matrices()

for k,v in W_per_num_edge_removals.items():
    print(k,len(v))

for i in range(4):#len(W_per_num_edge_removals[2])):
    W=W_per_num_edge_removals[3][i]
    env.redefine_graph_structure(W,env.sp.nodeid2coord)
    env.render(mode=None, fname='graph_'+str(i))
k=0