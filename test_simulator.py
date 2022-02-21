import matplotlib.pyplot as plt
from modules.rl.rl_policy import EpsilonGreedyPolicySB3_PPO, Policy
from modules.rl.rl_custom_worlds import GetCustomWorld
from modules.rl.rl_utils import EvaluatePolicy, print_parameters, GetFullCoverageSample, NpWrapper
from modules.sim.simdata_utils import SimulateInteractiveMode
from modules.gnn.nfm_gen import NFM_ec_t, NFM_ev_t, NFM_ev_ec_t, NFM_ev_ec_t_um_us
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from pathlib import Path
import random
import networkx as nx


#world_name='Manhattan3x3_PauseFreezeWorld'
#world_name='Manhattan3x3_PauseDynamicWorld'
#world_name='Manhattan3x3_WalkAround'
#world_name='Manhattan5x5_FixedEscapeInit'
#world_name='MetroU3_e17tborder_FixedEscapeInit'
#world_name='MetroU3_e17t0_FixedEscapeInit'
#world_name='MetroU3_e1t31_FixedEscapeInit'
#world_name='SparseManhattan5x5'
world_name = 'NWB_test'

state_repr='etUte0U0'
state_enc='nfm'
env = GetCustomWorld(world_name, make_reflexive=True, state_repr=state_repr, state_enc=state_enc)
k=0
# dmin = 1000
# ncenter = 0
# for n in env.sp.pos:
#     x, y = env.sp.pos[n]
#     d = (x - 5.) ** 2 + (y - 52.) ** 2
#     if d < dmin:
#         ncenter = n
#         dmin = d

# # color by path length from node near center
# p = dict(nx.single_source_shortest_path_length(env.sp.G, ncenter))

# plt.figure(figsize=(100, 100))
# nx.draw_networkx_edges(env.sp.G, env.sp.pos, alpha=0.4)
# nx.draw_networkx_nodes(
#     env.sp.G,
#     env.sp.pos,
#     nodelist=list(p.keys()),
#     node_size=10,
#     node_color='red',#list(p.values()),
#     #cmap=plt.cm.Reds_r,
# )

# plt.xlim(-0.05, 1.05)
# plt.ylim(-0.05, 1.05)
# plt.axis("off")
# plt.show()
# plt.savefig('test.png')



while True:
    # Select nfm type
    nfm_funcs = {'NFM_ev_ec_t':NFM_ev_ec_t(),'NFM_ec_t':NFM_ec_t(),'NFM_ev_t':NFM_ev_t(),'NFM_ev_ec_t_um_us':NFM_ev_ec_t_um_us()}
    nfm_func=nfm_funcs['NFM_ev_ec_t_um_us']
    env.redefine_nfm(nfm_func)

    # Selecting random goal nodes
    # nodelist=list(env.sp.labels2coord.keys())
    # sourcenode=env.sp.coord2labels[env.sp.start_escape_route]
    # nodelist.remove(sourcenode)
    # goal_nodes= random.choices(nodelist,k=1)
    # env.redefine_goal_nodes(goal_nodes)

    print('Shortest path', env.sp.spath_to_target,'length:', env.sp.spath_length,'hops')

    # Play without police units
    #env._remove_world_pool()
    #env.reset()
    #SimulateInteractiveMode(env)
    #env._restore_world_pool()
    #env.reset()
    
    SimulateInteractiveMode(env,filesave_with_time_suffix=True)

#env.render(mode=None,fname='test.png')
# plt.imshow([[0.,1.],[0.,1.]],
#     cmap=plt.cm.Greens,
#     interpolation='bicubic',
#     vmin=0,vmax=255
# )
# plt.plot([0,1,2,3],[5,1,6,7])
# plt.savefig('test.png')