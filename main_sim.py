import simdata_utils as su
from environments import GraphWorld
import random
import simdata_utils as su
from rl_utils import EvaluatePolicy
from rl_policy import LeftUpPolicy, RandomPolicy, MinIndegreePolicy

random.seed(911)

configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
#conf=configs['Manhattan3']
#conf=configs['Manhattan5']
conf=configs['Manhattan11'] # initial condition in register: ((5,0),(3,10),(6,10),(9,10))
#conf=configs['CircGraph']
#conf=configs['TKGraph']
conf['direction_north']=False

# DEMO: replay Gurobi optimizations
#su.SimulatePursuersPathways(conf, optimization_method='dynamic', fixed_initial_positions=((2,4),(0,0),(2,0),(4,0)))
#su.SimulatePursuersPathways(conf, optimization_method='static', fixed_initial_positions=((2,0),(0,0),(3,0),(4,0)))
#su.SimulatePursuersPathways(conf, optimization_method='static', fixed_initial_positions=None)
#su.SimulatePursuersPathways(conf, optimization_method='static', fixed_initial_positions=((0,1),(2,2)))

# DEMO: interactive mode
env=GraphWorld(conf, optimization_method='static', fixed_initial_positions=(5,114,115,116))#((0,1),(2,2)))#None)#((2,0),(3,4),(4,3),(4,4)))
su.SimulateInteractiveMode(env)

# DEMO: policy evaluation
#policy=LeftUpPolicy(env)
policy=RandomPolicy(env)
#policy=MinIndegreePolicy(env)
EvaluatePolicy(env, policy,number_of_runs=1, print_runs=False, save_plots=False)