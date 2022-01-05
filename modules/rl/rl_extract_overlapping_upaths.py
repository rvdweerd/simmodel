import modules.sim.simdata_utils as su
from modules.rl.environments import GraphWorld
import random
from rl_utils import EvaluatePolicy, SelectTrainset
from rl_policy import LeftUpPolicy, RandomPolicy


#random.seed(422)

configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
#conf=configs['Manhattan3']
conf=configs['Manhattan5']
#conf=configs['Manhattan11']
#conf=configs['CircGraph']
#conf=configs['TKGraph']
conf['direction_north']=False

env=GraphWorld(conf, optimization_method='static', fixed_initial_positions=None)
init_pos_trainset_indices0, init_pos_trainset_indices1 = SelectTrainset(env, min_y_coord=env.sp.N-1, min_num_same_positions=env.sp.U, min_num_worlds=4, print_selection=True)
env.world_pool = init_pos_trainset_indices1 # limit the training set to the selected entries
print('-------- sampling initial states')
su.SimulateInteractiveMode(env)
