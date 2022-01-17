from modules.sim.graph_factory import all_Manhattan3x3_symmetric_adj_matrices, get_all_edge_removals_symmetric
from modules.rl.rl_custom_worlds import GetCustomWorld
from modules.sim.simdata_utils import SimulateInteractiveMode
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
#env = GraphWorld(config, optimization_method='static', fixed_initial_positions=None, state_representation=state_repr, state_encoding=state_enc)

#world_name='MetroU3_e17tborder_FixedEscapeInit'
world_name='Manhattan5x5_FixedEscapeInit'
env = GetCustomWorld(world_name, make_reflexive=True, state_repr=state_repr, state_enc='nodes')
W_all, W_per_num_edge_removals = get_all_edge_removals_symmetric(nx.convert_matrix.to_numpy_matrix(env.sp.G),removals=[4,8,12,16],instances_per_num_removed=10)

for k,v in W_per_num_edge_removals.items():
    # v: list of tuples
    print(k,len(v))

for k,v in W_per_num_edge_removals.items():#len(W_per_num_edge_removals[2])):
    W=v[0][0]
    env.redefine_graph_structure(W,env.sp.nodeid2coord)
    env.render(mode=None, fname='graph_'+str(k))
    SimulateInteractiveMode(env)
