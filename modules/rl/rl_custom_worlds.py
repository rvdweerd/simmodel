import modules.sim.simdata_utils as su
from modules.rl.rl_utils import CreateDuplicatesTrainsets
from modules.rl.environments import GraphWorld
from modules.sim.graph_factory import GetWorldSet
import modules.gnn.nfm_gen
import networkx as nx

    


def GetCustomWorld(world_name, make_reflexive=True, state_repr='et', state_enc='nodes'):
    if world_name == 'Manhattan3x3_PauseFreezeWorld':
        configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
        conf=configs['Manhattan3']
        conf['direction_north']=False
        conf['make_reflexive']=make_reflexive
        conf['U']=3
        env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None, state_representation=state_repr, state_encoding=state_enc)
        env.register['coords']={((1,0),(0,2),(1,2),(2,2)):0}
        env.register['labels']={(1,6,7,8):0}
        env.databank['coords']=[{'start_escape_route':(1,0), 'start_units':[(0,2),(1,2),(2,2)], 'paths':[[(0,2),(0,1)],[(1,2),(1,2),(1,2),(0,2)],[(2,2),(2,1)]]}]
        env.databank['labels']=[{'start_escape_route':1, 'start_units':[6,7,8], 'paths':[[6,3],[7,7,7,6],[8,5]]}]
        env.iratios=[1.]
        env.all_worlds=[0]
        env.world_pool=[0]
        env.reset()
        return env
    if world_name == 'Manhattan3x3_PauseDynamicWorld':
        configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
        conf=configs['Manhattan3']
        conf['direction_north']=False
        conf['make_reflexive']=make_reflexive
        conf['U']=3
        env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None, state_representation=state_repr, state_encoding=state_enc)
        env.register['coords']={((1,0),(0,2),(1,2),(2,2)):0}
        env.register['labels']={(1,6,7,8):0}
        env.databank['coords']=[{'start_escape_route':(1,0), 'start_units':[(0,2),(1,2),(2,2)], 'paths':[[(0,2),(0,1),(0,0)],[(1,2),(1,1),(0,1)],[(2,2),(2,1)]]}]
        env.databank['labels']=[{'start_escape_route':1, 'start_units':[6,7,8], 'paths':[[6,3,0],[7,4,3],[8,5]]}]
        env.iratios=[1.]
        env.all_worlds=[0]
        env.world_pool=[0]
        env.reset()
        return env
    if world_name == 'Manhattan3x3_WalkAround':
        configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
        conf=configs['Manhattan3']
        conf['direction_north']=False
        conf['make_reflexive']=make_reflexive
        conf['U']=2
        env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None, state_representation=state_repr, state_encoding=state_enc)
        env.register['coords']={((1,0),(0,2),(1,2)):0}
        env.register['labels']={(1,6,7):0}
        env.databank['coords']=[{'start_escape_route':(1,0), 'start_units':[(0,2),(1,2)], 'paths':[[(0,2),(0,1)],[(1,2),(1,1)]]}]
        env.databank['labels']=[{'start_escape_route':1, 'start_units':[6,7], 'paths':[[6,3],[7,4]]}]
        env.iratios=[1.]
        env.all_worlds=[0]
        env.world_pool=[0]
        env.redefine_goal_nodes([6])
        env.reset()
        return env
    if world_name == 'Manhattan5x5_DuplicateSetA':
        configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
        conf=configs['Manhattan5']
        conf['direction_north']=False
        conf['make_reflexive']=make_reflexive
        conf['U']=3
        env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None, state_representation=state_repr, state_encoding=state_enc)
        init_pos_trainset_indices0, init_pos_trainset_indices1 = CreateDuplicatesTrainsets(env, min_y_coord=env.sp.N-1, min_num_same_positions=env.sp.U, min_num_worlds=4)
        env.world_pool = init_pos_trainset_indices0 # limit the training set to the selected entries
        return env        
    if world_name == 'Manhattan5x5_DuplicateSetB':
        configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
        conf=configs['Manhattan5']
        conf['direction_north']=False
        conf['make_reflexive']=make_reflexive
        conf['U']=3
        env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None, state_representation=state_repr, state_encoding=state_enc)
        init_pos_trainset_indices0, init_pos_trainset_indices1 = CreateDuplicatesTrainsets(env, min_y_coord=env.sp.N-1, min_num_same_positions=env.sp.U, min_num_worlds=4)
        env.world_pool = init_pos_trainset_indices1 # limit the training set to the selected entries
        return env
    if world_name == 'Manhattan5x5_FixedEscapeInit':
        configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
        conf=configs['Manhattan5']
        conf['direction_north']=False
        conf['loadAllStartingPositions']=False
        conf['make_reflexive']=make_reflexive
        env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None,state_representation=state_repr, state_encoding=state_enc)
        return env
    if world_name == 'Manhattan5x5_VariableEscapeInit':
        configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
        conf=configs['Manhattan5']
        conf['direction_north']=False
        conf['loadAllStartingPositions']=True
        conf['make_reflexive']=make_reflexive
        env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None,state_representation=state_repr, state_encoding=state_enc)
        return env
    if world_name == 'MetroU3_e17tborder_FixedEscapeInit':
        configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
        conf=configs['MetroGraphU3']
        conf['direction_north']=False
        conf['loadAllStartingPositions']=False
        assert conf['T'] ==  20
        conf['make_reflexive']=make_reflexive
        env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None,state_representation=state_repr, state_encoding=state_enc)
        return env
    if world_name == 'MetroU3_e17tborder_VariableEscapeInit':
        configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
        conf=configs['MetroGraphU3']
        conf['direction_north']=False
        conf['loadAllStartingPositions']=True
        assert conf['T'] ==  20
        conf['make_reflexive']=make_reflexive
        env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None,state_representation=state_repr, state_encoding=state_enc)
        return env
    if world_name == 'MetroU3_e17t31_FixedEscapeInit':
        configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
        conf=configs['MetroGraphU3']
        conf['direction_north']=False
        conf['loadAllStartingPositions']=False
        conf['make_reflexive']=make_reflexive
        assert  conf['T'] == 20
        env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None,state_representation=state_repr, state_encoding=state_enc)
        env.sp.target_nodes=[31]
        return env
    if world_name == 'MetroU3_e17t0_FixedEscapeInit':
        configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
        conf=configs['MetroGraphU3']
        conf['direction_north']=False
        conf['loadAllStartingPositions']=False
        conf['make_reflexive']=make_reflexive
        assert  conf['T'] == 20
        env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None,state_representation=state_repr, state_encoding=state_enc)
        env.sp.target_nodes=[0]
        return env
    if world_name == 'MetroU3_e1t31_FixedEscapeInit':
        configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
        conf=configs['MetroGraphU3L8_node1']
        conf['direction_north']=False
        conf['loadAllStartingPositions']=False
        conf['make_reflexive']=make_reflexive
        assert conf['T'] == 25
        conf['target_nodes']=[31]
        env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None,state_representation=state_repr, state_encoding=state_enc)
        #env.sp.target_nodes=[31]
        return env
    if world_name == 'SparseManhattan5x5':
        configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
        conf=configs['SparseManhattan5x5']
        conf['direction_north']=False
        conf['loadAllStartingPositions']=False
        conf['make_reflexive']=make_reflexive
        env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None,state_representation=state_repr, state_encoding=state_enc)
        return env
    if world_name  in ['NWB_test_FixedEscapeInit', 'NWB_test_VariableEscapeInit']:
        conf={
            'graph_type': "NWBGraph",
            'make_reflexive': False,            
            'N': 975,    # number of nodes along one side
            'U': 10,    # number of pursuer units
            'L': 50,    # Time steps
            'T': 50,
            'R': 1000,  # Number of escape routes sampled 
            'direction_north': False,       # Directional preference of escaper
            #'start_escape_route': 'bottom_center', # Initial position of escaper (always bottom center)
            #'fixed_initial_positions': (1,5,7,28),
            'loadAllStartingPositions': False
        }
        if world_name == 'NWB_test_VariableEscapeInit':
            conf['loadAllStartingPositions'] = True
        conf['obj'] = nx.read_gpickle('datasets/G_nwb/4.GEPHI_to_SIM/G_test_DAM_1km_edited_V=975.bin')
        #conf['obj']['G']=conf['obj']['G'].to_undirected()
        assert not conf['obj']['G'].is_directed()
        env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None,state_representation=state_repr, state_encoding=state_enc)
        return env       
    if world_name  in ['NWB_ROT_FixedEscapeInit', 'NWB_ROT_VariableEscapeInit']:
        conf={
            'graph_type': "NWBGraphROT",
            'make_reflexive': False,            
            'N': 2602,    # number of nodes along one side
            'U': 10,    # number of pursuer units
            'L': 50,    # Time steps
            'T': 50,
            'R': 1000,  # Number of escape routes sampled 
            'direction_north': False,       # Directional preference of escaper
            #'start_escape_route': 'bottom_center', # Initial position of escaper (always bottom center)
            #'fixed_initial_positions': (1,5,7,28),
            'loadAllStartingPositions': False
        }
        if world_name == 'NWB_ROT_VariableEscapeInit':
            conf['loadAllStartingPositions'] = True
        conf['obj'] = nx.read_gpickle('datasets/G_nwb/4.GEPHI_to_SIM/G_test_ROT_2km_edited_V=2602.bin')
        assert not conf['obj']['G'].is_directed()
        env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None,state_representation=state_repr, state_encoding=state_enc)
        return env       
    if world_name  in ['NWB_UTR_FixedEscapeInit', 'NWB_UTR_VariableEscapeInit']:
        conf={
            'graph_type': "NWBGraphUTR",
            'make_reflexive': False,            
            'N': 1182,    # number of nodes along one side
            'U': 10,    # number of pursuer units
            'L': 50,    # Time steps
            'T': 50,
            'R': 1000,  # Number of escape routes sampled 
            'direction_north': False,       # Directional preference of escaper
            #'start_escape_route': 'bottom_center', # Initial position of escaper (always bottom center)
            #'fixed_initial_positions': (1,5,7,28),
            'loadAllStartingPositions': False
        }
        if world_name == 'NWB_UTR_VariableEscapeInit':
            conf['loadAllStartingPositions'] = True
        conf['obj'] = nx.read_gpickle('datasets/G_nwb/4.GEPHI_to_SIM/G_test_UTR_1km_edited_V=1182.bin')
        assert not conf['obj']['G'].is_directed()
        env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None,state_representation=state_repr, state_encoding=state_enc)
        return env       

def CreateWorlds(run_world_names, make_reflexive=True, state_repr='et', state_enc='nodes'):
    worlds=[]
    for world_name in run_world_names:
        worlds.append(GetCustomWorld(world_name, make_reflexive, state_repr, state_enc))
    if len(worlds) == 0: assert False
    return worlds
