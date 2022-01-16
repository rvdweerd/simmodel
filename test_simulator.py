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

#world_name='Manhattan3x3_PauseFreezeWorld'
world_name='Manhattan3x3_WalkAround'
#world_name='Manhattan5x5_FixedEscapeInit'
#world_name='MetroU3_e17tborder_FixedEscapeInit'
#world_name='SparseManhattan5x5'

state_repr='et'
env = GetCustomWorld(world_name, make_reflexive=True, state_repr=state_repr, state_enc='nodes')
SimulateInteractiveMode(env)
#env.render(mode=None,fname='test.png')
# plt.imshow([[0.,1.],[0.,1.]],
#     cmap=plt.cm.Greens,
#     interpolation='bicubic',
#     vmin=0,vmax=255
# )
# plt.plot([0,1,2,3],[5,1,6,7])
# plt.savefig('test.png')