import matplotlib.pyplot as plt
from modules.rl.rl_policy import EpsilonGreedyPolicySB3_PPO, Policy
from modules.rl.rl_custom_worlds import GetCustomWorld
from modules.rl.rl_utils import EvaluatePolicy, print_parameters, GetFullCoverageSample, NpWrapper
from modules.sim.simdata_utils import SimulateInteractiveMode
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from pathlib import Path
import random
import networkx as nx

def FindShortestPathToAnyTargetNode(G, labels2coord, source_node_label, target_nodes_labels):
    min_cost = 1e9
    for target_node_label in target_nodes_labels:
        target_node_coord = labels2coord[target_node_label]
        source_node_coord = labels2coord[source_node_label]
        cost, path = nx.single_source_dijkstra(G, source_node_coord, target_node_coord, weight='weight')
        if cost < min_cost:
            best_path_coords = path
            min_cost = cost
    return best_path_coords, min_cost

#world_name='Manhattan3x3_PauseFreezeWorld'
#world_name='Manhattan3x3_WalkAround'
#world_name='Manhattan5x5_FixedEscapeInit'
world_name='MetroU3_e17tborder_FixedEscapeInit'
#world_name='SparseManhattan5x5'

state_repr='etUte0U0'
state_enc='nodes'
env = GetCustomWorld(world_name, make_reflexive=True, state_repr=state_repr, state_enc=state_enc)

while True:
    nodelist=list(env.sp.labels2coord.keys())
    sourcenode=env.sp.coord2labels[env.sp.start_escape_route]
    nodelist.remove(sourcenode)
    #goal_nodes= [random.choice(nodelist)]
    goal_nodes= random.choices(nodelist,k=1)


    spath_coords, spath_cost = FindShortestPathToAnyTargetNode(env.sp.G, env.sp.labels2coord, env.sp.start_escape_route_node, goal_nodes)
    #spath_coords = nx.dijkstra_path(env.sp.G, env.sp.start_escape_route, env.sp.labels2coord[goal_nodes[0]])
    spath_nodes = [ env.sp.coord2labels[c] for c in spath_coords]
    print('Shortest path',spath_nodes,'length:',len(spath_nodes)-1,'hops')
    env.redefine_goal_nodes(goal_nodes)
    env._remove_world_pool()
    #env.reset()
    SimulateInteractiveMode(env)

    env._restore_world_pool()
    #env.reset()
    SimulateInteractiveMode(env)

#env.render(mode=None,fname='test.png')
# plt.imshow([[0.,1.],[0.,1.]],
#     cmap=plt.cm.Greens,
#     interpolation='bicubic',
#     vmin=0,vmax=255
# )
# plt.plot([0,1,2,3],[5,1,6,7])
# plt.savefig('test.png')