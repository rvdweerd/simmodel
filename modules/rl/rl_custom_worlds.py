import modules.sim.simdata_utils as su
from modules.rl.rl_utils import CreateDuplicatesTrainsets
from modules.rl.environments import GraphWorld
from modules.sim.graph_factory import GetWorldSet, LoadData
import modules.gnn.nfm_gen
import networkx as nx
 
def GetCustomWorld(world_name, make_reflexive=True, state_repr='et', state_enc='nodes'):
    if world_name == 'Manhattan3x3_PredictionExample':
        nfm_func = modules.gnn.nfm_gen.NFM_ev_ec_t_um_us()
        edge_blocking = True
        solve_select = 'solvable'
        reject_u_duplicates = False
        Etrain=[3]
        Utrain=[1]
        databank_full, register_full, solvable = LoadData(edge_blocking = True)
        evalenv, hashint2env, env2hashint, env2hashstr = GetWorldSet(state_repr, state_enc, U=Utrain, E=Etrain, edge_blocking=edge_blocking, solve_select=solve_select, reject_duplicates=reject_u_duplicates, nfm_func=nfm_func)
        #hashint=4007
        #entry=0
        hashint=3759 # flipped vertically
        entry=1
        env_idx=hashint2env[hashint]
        env=evalenv[env_idx]
        env.redefine_goal_nodes([5])
        env.world_pool=[entry]
        return env
    if world_name == 'M3test1':
        configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
        conf=configs['M3test1']
        conf['direction_north']=False
        conf['make_reflexive']=make_reflexive
        conf['U']=1
        env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None, state_representation=state_repr, state_encoding=state_enc)
        env.register['coords']={((1,0),(1,2)):0}
        env.register['labels']={(7,1):0}
        env.databank['coords']=[{'start_escape_route':(1,0), 'start_units':[(1,2)], 'paths':[(1,2),(0,2)]}]
        env.databank['labels']=[{'start_escape_route':7, 'start_units':[1], 'paths':[[1,0]]}]
        env.iratios=[1.]
        env.redefine_goal_nodes([1])
        env.all_worlds=[0]
        env.world_pool=[0]
        env.reset()
        return env

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
        env.redefine_goal_nodes([7])
        env.reset()
        return env
    if world_name == 'MemoryTaskU1':
        configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
        conf=configs['MemoryTaskU1']
        conf['direction_north']=False
        conf['make_reflexive']=make_reflexive
        conf['U']=1
        env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None, state_representation=state_repr, state_encoding=state_enc)
        env.register['coords']={((0,1),(0,3)):0, ((0,1),(0,-1)):1}
        env.register['labels']={(3,0):0,(3,6):1}
        env.databank['coords']=[
            {'start_escape_route':(0,1), 'start_units':[(0,3)], 'paths':[[(0,3),(2,3),(4,2)]]},
            {'start_escape_route':(0,1), 'start_units':[(0,-1)], 'paths':[[(0,-1),(2,-1),(4,0)]]},
            ]
        env.databank['labels']=[
            {'start_escape_route':3, 'start_units':[0], 'paths':[[0,1,2]]},
            {'start_escape_route':3, 'start_units':[6], 'paths':[[6,7,5]]}
            ]
        env.iratios=[1.,1.]
        env.all_worlds=[0,1]
        env.world_pool=[0,1]
        env.redefine_goal_nodes([2,5])
        env.reset()
        return env
    if world_name == 'BifurGraphTask1':
        configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
        conf=configs['BifurGraphTask1']
        conf['direction_north']=False
        conf['make_reflexive']=make_reflexive
        conf['U']=2
        env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None, state_representation=state_repr, state_encoding=state_enc)
        #env.redefine_goal_nodes([8,17,26])
        env.world_pool=[269]#[196,188,269,19,309,256,183,178]#[20,4,5,17,25,7,6,19]
        env.reset()
        return env        
    if world_name == 'MemoryTaskU1Long':
        configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
        conf=configs['MemoryTaskU1Long']
        conf['direction_north']=False
        conf['make_reflexive']=make_reflexive
        conf['U']=1
        env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None, state_representation=state_repr, state_encoding=state_enc)
        env.register['coords']={((-2,1),(-2,3)):0, ((-2,1),(-2,-1)):1}
        env.register['labels']={(4,0):0,(4,8):1}
        env.databank['coords']=[
            {'start_escape_route':(-2,1), 'start_units':[(-2,3)], 'paths':[[(-2,3),(0,3),(2,3),(4,2)]]},
            {'start_escape_route':(-2,1), 'start_units':[(-2,-1)], 'paths':[[(-2,-1),(0,-1),(2,-1),(4,0)]]},
            ]
        env.databank['labels']=[
            {'start_escape_route':4, 'start_units':[0], 'paths':[[0,1,2,3]]},
            {'start_escape_route':4, 'start_units':[8], 'paths':[[8,9,10,7]]}
            ]
        env.iratios=[1.,1.]
        env.all_worlds=[0,1]
        env.world_pool=[0,1]
        env.redefine_goal_nodes([3,7])
        env.reset()
        return env
    if world_name == 'MemoryTaskU2T': # Top pursuer proceeds to nearest target node
        configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
        conf=configs['MemoryTaskU1']
        conf['direction_north']=False
        conf['make_reflexive']=make_reflexive
        conf['U']=2
        env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None, state_representation=state_repr, state_encoding=state_enc)
        env.register['coords']={((0,1),(0,3),(0,-1)):0}
        env.register['labels']={(3,0,6):0}
        env.databank['coords']=[
            {'start_escape_route':(0,1), 'start_units':[(0,3),(0,-1)], 'paths':[[(0,3),(2,3),(4,2)],[(0,-1),(2,-1)]]},
            ]
        env.databank['labels']=[
            {'start_escape_route':3, 'start_units':[0,6], 'paths':[[0,1,2],[6,7]]},
            ]
        env.iratios=[1.]
        env.all_worlds=[0]
        env.world_pool=[0]
        env.redefine_goal_nodes([2,5])
        env.reset()
        return env
    if world_name == 'MemoryTaskU2B': # Bottom pursuer proceeds to nearest target node
        configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
        conf=configs['MemoryTaskU1']
        conf['direction_north']=False
        conf['make_reflexive']=make_reflexive
        conf['U']=2
        env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None, state_representation=state_repr, state_encoding=state_enc)
        env.register['coords']={((0,1),(0,3),(0,-1)):0}
        env.register['labels']={(3,0,6):0}
        env.databank['coords']=[
            {'start_escape_route':(0,1), 'start_units':[(0,3),(0,-1)], 'paths':[[(0,3),(2,3)],[(0,-1),(2,-1),(4,0)]]},
            ]
        env.databank['labels']=[
            {'start_escape_route':3, 'start_units':[0,6], 'paths':[[0,1],[6,7,5]]},
            ]
        env.iratios=[1.]
        env.all_worlds=[0]
        env.world_pool=[0]
        env.redefine_goal_nodes([2,5])
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
    if world_name == 'Manhattan5x5_FixedEscapeInit2':
        configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
        conf=configs['Manhattan5']
        conf['graph_type']= "Manhattan2"
        conf['R']=5
        conf['direction_north']=False
        conf['loadAllStartingPositions']=False
        conf['make_reflexive']=make_reflexive
        env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None,state_representation=state_repr, state_encoding=state_enc)
        return env        
    if world_name == 'Manhattan11x11':
        configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
        conf=configs['Manhattan11']
        conf['direction_north']=False
        conf['loadAllStartingPositions']=False
        conf['make_reflexive']=make_reflexive
        env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None,state_representation=state_repr, state_encoding=state_enc)
        return env
    if world_name == 'CircGraph':
        configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
        conf=configs['CircGraph']
        conf['direction_north']=False
        conf['loadAllStartingPositions']=False
        conf['make_reflexive']=make_reflexive
        env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None,state_representation=state_repr, state_encoding=state_enc)
        return env
    if world_name == 'TKGraph':
        configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
        conf=configs['TKGraph']
        conf['direction_north']=False
        conf['loadAllStartingPositions']=False
        conf['make_reflexive']=True
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
    if world_name == 'MetroU2_e17tborder_VariableEscapeInit':
        configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
        conf=configs['MetroGraphU3']
        conf['U']=2
        conf['direction_north']=False
        conf['loadAllStartingPositions']=True
        assert conf['T'] ==  20
        conf['make_reflexive']=make_reflexive
        env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None,state_representation=state_repr, state_encoding=state_enc)
        return env 
    if world_name == 'MetroU1_e17tborder_VariableEscapeInit':
        configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
        conf=configs['MetroGraphU3']
        conf['U']=1
        conf['direction_north']=False
        conf['loadAllStartingPositions']=True
        assert conf['T'] ==  20
        conf['make_reflexive']=make_reflexive
        env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None,state_representation=state_repr, state_encoding=state_enc)
        return env    
    if world_name == 'MetroU2_e17tborder_FixedEscapeInit':
        configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
        conf=configs['MetroGraphU3']
        conf['U']=2
        conf['direction_north']=False
        conf['loadAllStartingPositions']=False
        assert conf['T'] ==  20
        conf['make_reflexive']=make_reflexive
        env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None,state_representation=state_repr, state_encoding=state_enc)
        return env 
    if world_name == 'MetroU1_e17tborder_FixedEscapeInit':
        configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
        conf=configs['MetroGraphU3']
        conf['U']=1
        conf['direction_north']=False
        conf['loadAllStartingPositions']=False
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
    if world_name  in ['NWB_test_FixedEscapeInit_U=15','NWB_test_FixedEscapeInit_U=20']:
        conf={
            'graph_type': "NWBGraph",
            'make_reflexive': False,            
            'N': 975,    # number of nodes along one side
            'U': int(world_name[-2:]),    # number of pursuer units
            'L': 50,    # Time steps
            'T': 50,
            'R': 1000,  # Number of escape routes sampled 
            'direction_north': False,       # Directional preference of escaper
            #'start_escape_route': 'bottom_center', # Initial position of escaper (always bottom center)
            #'fixed_initial_positions': (1,5,7,28),
            'loadAllStartingPositions': False
        }
        if world_name == 'NWB_test_VariableEscapeInit_U=15':
            conf['loadAllStartingPositions'] = True
        conf['obj'] = nx.read_gpickle('datasets/G_nwb/4.GEPHI_to_SIM/G_test_DAM_1km_edited_V=975.bin')
        #conf['obj']['G']=conf['obj']['G'].to_undirected()
        assert not conf['obj']['G'].is_directed()
        env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None,state_representation=state_repr, state_encoding=state_enc)
        return env   
    if world_name  in ['NWB_test_FixedEscapeInit2', 'NWB_test_VariableEscapeInit2']:
        conf={
            'graph_type': "NWBGraph2",
            'make_reflexive': False,            
            'N': 975,    # number of nodes along one side
            'U': 10,    # number of pursuer units
            'L': 50,    # Time steps
            'T': 50,
            'R': 113,  # Number of escape routes sampled 
            'direction_north': False,       # Directional preference of escaper
            #'start_escape_route': 'bottom_center', # Initial position of escaper (always bottom center)
            #'fixed_initial_positions': (1,5,7,28),
            'loadAllStartingPositions': False
        }
        if world_name == 'NWB_test_VariableEscapeInit2':
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
    if world_name  in ['NWB_ROT_FixedEscapeInit_U=15','NWB_ROT_FixedEscapeInit_U=20']:
        conf={
            'graph_type': "NWBGraphROT",
            'make_reflexive': False,            
            'N': 2602,    # number of nodes along one side
            'U': int(world_name[-2:]),    # number of pursuer units
            'L': 50,    # Time steps
            'T': 50,
            'R': 1000,  # Number of escape routes sampled 
            'direction_north': False,       # Directional preference of escaper
            #'start_escape_route': 'bottom_center', # Initial position of escaper (always bottom center)
            #'fixed_initial_positions': (1,5,7,28),
            'loadAllStartingPositions': False
        }
        if world_name == 'NWB_ROT_VariableEscapeInit_U=15':
            conf['loadAllStartingPositions'] = True
        conf['obj'] = nx.read_gpickle('datasets/G_nwb/4.GEPHI_to_SIM/G_test_ROT_2km_edited_V=2602.bin')
        #conf['obj']['G']=conf['obj']['G'].to_undirected()
        assert not conf['obj']['G'].is_directed()
        env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None,state_representation=state_repr, state_encoding=state_enc)
        return env   
    if world_name  in ['NWB_ROT_FixedEscapeInit2', 'NWB_ROT_VariableEscapeInit2']:
        conf={
            'graph_type': "NWBGraphROT2",
            'make_reflexive': False,            
            'N': 2602,    # number of nodes along one side
            'U': 10,    # number of pursuer units
            'L': 50,    # Time steps
            'T': 50,
            'R': 173,  # Number of escape routes sampled 
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
    if world_name  in ['NWB_UTR_FixedEscapeInit_U=15','NWB_UTR_FixedEscapeInit_U=20']:
        conf={
            'graph_type': "NWBGraphUTR",
            'make_reflexive': False,            
            'N': 1182,    # number of nodes along one side
            'U': int(world_name[-2:]),    # number of pursuer units
            'L': 50,    # Time steps
            'T': 50,
            'R': 1000,  # Number of escape routes sampled 
            'direction_north': False,       # Directional preference of escaper
            #'start_escape_route': 'bottom_center', # Initial position of escaper (always bottom center)
            #'fixed_initial_positions': (1,5,7,28),
            'loadAllStartingPositions': False
        }
        if world_name == 'NWB_UTR_VariableEscapeInit_U=15':
            conf['loadAllStartingPositions'] = True
        conf['obj'] = nx.read_gpickle('datasets/G_nwb/4.GEPHI_to_SIM/G_test_UTR_1km_edited_V=1182.bin')
        #conf['obj']['G']=conf['obj']['G'].to_undirected()
        assert not conf['obj']['G'].is_directed()
        env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None,state_representation=state_repr, state_encoding=state_enc)
        return env    
    if world_name  in ['NWB_UTR_FixedEscapeInit2', 'NWB_UTR_VariableEscapeInit2']:
        conf={
            'graph_type': "NWBGraphUTR2",
            'make_reflexive': False,            
            'N': 1182,    # number of nodes along one side
            'U': 10,    # number of pursuer units
            'L': 50,    # Time steps
            'T': 50,
            'R': 79,  # Number of escape routes sampled 
            'direction_north': False,       # Directional preference of escaper
            #'start_escape_route': 'bottom_center', # Initial position of escaper (always bottom center)
            #'fixed_initial_positions': (1,5,7,28),
            'loadAllStartingPositions': False
        }
        if world_name == 'NWB_UTR_VariableEscapeInit2':
            conf['loadAllStartingPositions'] = True
        conf['obj'] = nx.read_gpickle('datasets/G_nwb/4.GEPHI_to_SIM/G_test_UTR_1km_edited_V=1182.bin')
        assert not conf['obj']['G'].is_directed()
        env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=None,state_representation=state_repr, state_encoding=state_enc)
        return env       

    else: assert False, "Unknown world name: %s" % world_name

def CreateWorlds(run_world_names, make_reflexive=True, state_repr='et', state_enc='nodes'):
    worlds=[]
    for world_name in run_world_names:
        worlds.append(GetCustomWorld(world_name, make_reflexive, state_repr, state_enc))
    if len(worlds) == 0: assert False
    return worlds
