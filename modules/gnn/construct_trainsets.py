import torch
import torch.nn as nn 
import torch.nn.functional as F
from modules.ppo.helpfuncs import get_super_env, CreateEnv, eval_simple, evaluate_ppo#, get_logdirs
from modules.rl.environments import SuperEnv
#from modules.rl.rl_policy import ActionMaskedPolicySB3_PPO
#from modules.ppo.models_sb3 import s2v_ActorCriticPolicy, Struc2VecExtractor
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def ConstructTrainSet(config, apply_wrappers=True, remove_paths=False, tset='M3M5Mix'):
    # Requires:
    # config['max_nodes']
    # config['nfm_func']
    # config['edge_blocking']
    # config['solve_select']
    global_env=[]
    probs=[]
    env_all_list=[]
    config['solve_select'] = 'solvable'

    if tset == 'M3M5Mix':
        config['max_edges'] = 300
        config['max_nodes'] = 25
        env, _ = get_super_env(Uselected=[0,1,2,3,4], Eselected=[6,7,8,9], config=config, var_targets=None, apply_wrappers=apply_wrappers, remove_paths=remove_paths)
        env_all_list += env.all_env
        global_env.append(env)
        probs.append(1)
        
        env, _ = get_super_env(Uselected=[0,1,2,3,4], Eselected=[0,1,2,3,4,5], config=config, var_targets=None, apply_wrappers=apply_wrappers, remove_paths=remove_paths)
        env_all_list += env.all_env
        global_env.append(env)
        probs.append(2)

        env, _ = get_super_env(Uselected=[0,1,2,3,4], Eselected=[0,1,2,3,4,5], config=config, var_targets=None, apply_wrappers=apply_wrappers, remove_paths=remove_paths)
        env_all_list += env.all_env
        global_env.append(env)
        probs.append(2)

        world_name = 'Manhattan5x5_VariableEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(2)

        world_name = 'Manhattan5x5_VariableEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], nfm_func_name=config['nfm_func'], var_targets=[1,3], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(2)

        world_name = 'Manhattan5x5_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        world_name = 'Manhattan5x5_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], nfm_func_name=config['nfm_func'], var_targets=[1,3], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        world_name = 'Manhattan5x5_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], nfm_func_name=config['nfm_func'], var_targets=[1,1],remove_world_pool=True, apply_wrappers=apply_wrappers)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)    

        world_name = 'SparseManhattan5x5'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], nfm_func_name=config['nfm_func'], var_targets=[1,3], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        super_env=SuperEnv(
            global_env,
            hashint2env=None,
            max_possible_num_nodes=33,
            probs=probs)

    elif tset == 'NWB_AMS':
        world_name = 'NWB_test_VariableEscapeInit'
        env = CreateEnv(world_name, max_nodes=975, max_edges=3000, nfm_func_name = config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        world_name = 'NWB_test_VariableEscapeInit'
        env = CreateEnv(world_name, max_nodes=975, max_edges=3000, nfm_func_name = config['nfm_func'], var_targets=[10,20], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        world_name = 'NWB_test_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=975, max_edges=3000, nfm_func_name = config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(.1)

        world_name = 'NWB_test_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=975, max_edges=3000, nfm_func_name = config['nfm_func'], var_targets=[10,20], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        super_env=SuperEnv(
            global_env,
            hashint2env=None,
            max_possible_num_nodes=975,
            probs=probs)

    elif tset == 'NWB_AMS+Mix':
        config['max_edges'] = 3000
        config['max_nodes'] = 975

        world_name = 'NWB_test_VariableEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name = config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(3)

        world_name = 'NWB_test_VariableEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name = config['nfm_func'], var_targets=[10,20], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(3)

        world_name = 'NWB_test_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name = config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        world_name = 'NWB_test_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name = config['nfm_func'], var_targets=[10,20], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(3)

        world_name = 'MetroU3_e17tborder_VariableEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        world_name = 'MetroU3_e17tborder_VariableEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=[1,2], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(3)

        world_name = 'MetroU3_e17tborder_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(2)

        world_name = 'MetroU3_e17tborder_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=[1,2], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(3)

        env, _ = get_super_env(Uselected=[0,1,2,3,4], Eselected=[6,7,8,9], config=config, var_targets=None, apply_wrappers=apply_wrappers, remove_paths=remove_paths)
        env_all_list += env.all_env
        global_env.append(env)
        probs.append(5)
        
        env, _ = get_super_env(Uselected=[0,1,2,3,4], Eselected=[0,1,2,3,4,5], config=config, var_targets=None, apply_wrappers=apply_wrappers, remove_paths=remove_paths)
        env_all_list += env.all_env
        global_env.append(env)
        probs.append(5)

        env, _ = get_super_env(Uselected=[0,1,2,3,4], Eselected=[0,1,2,3,4,5], config=config, var_targets=None, apply_wrappers=apply_wrappers, remove_paths=remove_paths)
        env_all_list += env.all_env
        global_env.append(env)
        probs.append(5)

        world_name = 'Manhattan5x5_VariableEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(2)

        world_name = 'Manhattan5x5_VariableEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=[1,3], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(2)

        world_name = 'Manhattan5x5_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        world_name = 'Manhattan5x5_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=[1,3], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(2)

        world_name = 'Manhattan5x5_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=[1,1],remove_world_pool=True, apply_wrappers=apply_wrappers)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)    

        world_name = 'SparseManhattan5x5'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=[1,3], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        super_env=SuperEnv(
            global_env,
            hashint2env=None,
            max_possible_num_nodes=config['max_nodes'],
            probs=probs)

    elif tset == 'MetroConstructed':
        world_name = 'MetroU3_e17tborder_VariableEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=300, nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        world_name = 'MetroU3_e17tborder_VariableEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=300, nfm_func_name=config['nfm_func'], var_targets=[1,2], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        world_name = 'MetroU3_e17tborder_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=300, nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        world_name = 'MetroU3_e17tborder_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=300, nfm_func_name=config['nfm_func'], var_targets=[1,2], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        world_name = 'MetroU3_e1t31_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=300, nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(2)

        world_name = 'MetroU3_e1t31_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=300, nfm_func_name=config['nfm_func'], var_targets=[1,2], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        super_env=SuperEnv(
            global_env,
            hashint2env=None,
            max_possible_num_nodes=33,
            probs=probs)

    elif tset == 'MixAll33':
        config['max_edges'] = 300
        config['max_nodes'] = 33
        world_name = 'MetroU3_e17tborder_VariableEscapeInit'
        env = CreateEnv(world_name, max_nodes=33, max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        world_name = 'MetroU3_e17tborder_VariableEscapeInit'
        env = CreateEnv(world_name, max_nodes=33, max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=[1,2], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        world_name = 'MetroU3_e17tborder_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=33, max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        world_name = 'MetroU3_e17tborder_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=33, max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=[1,2], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        world_name = 'MetroU3_e1t31_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=33, max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(2)

        world_name = 'MetroU3_e1t31_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=33, max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=[1,2], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        env, _ = get_super_env(Uselected=[0,1,2,3,4], Eselected=[6,7,8,9], config=config, var_targets=None, apply_wrappers=apply_wrappers, remove_paths=remove_paths)
        env_all_list += env.all_env
        global_env.append(env)
        probs.append(1)
        
        env, _ = get_super_env(Uselected=[0,1,2,3,4], Eselected=[0,1,2,3,4,5], config=config, var_targets=None, apply_wrappers=apply_wrappers, remove_paths=remove_paths)
        env_all_list += env.all_env
        global_env.append(env)
        probs.append(2)

        env, _ = get_super_env(Uselected=[0,1,2,3,4], Eselected=[0,1,2,3,4,5], config=config, var_targets=None, apply_wrappers=apply_wrappers, remove_paths=remove_paths)
        env_all_list += env.all_env
        global_env.append(env)
        probs.append(2)

        world_name = 'Manhattan5x5_VariableEscapeInit'
        env = CreateEnv(world_name, max_nodes=33, max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(2)

        world_name = 'Manhattan5x5_VariableEscapeInit'
        env = CreateEnv(world_name, max_nodes=33, max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=[1,3], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(2)

        world_name = 'Manhattan5x5_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=33, max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        world_name = 'Manhattan5x5_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=33, max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=[1,3], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        world_name = 'Manhattan5x5_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=33, max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=[1,1],remove_world_pool=True, apply_wrappers=apply_wrappers)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)    

        world_name = 'SparseManhattan5x5'
        env = CreateEnv(world_name, max_nodes=33, max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=[1,3], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        # world_name = 'Manhattan3x3_WalkAround'
        # env = CreateEnv(world_name, max_nodes=33, max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=[1,3], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers)
        # env_all_list.append(env)
        # global_env.append(env)
        # probs.append(1)

        # world_name = 'Manhattan5x5_DuplicateSetA'
        # env = CreateEnv(world_name, max_nodes=33, max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=[1,3], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers)
        # env_all_list.append(env)
        # global_env.append(env)
        # probs.append(1)

        # world_name = 'Manhattan5x5_DuplicateSetB'
        # env = CreateEnv(world_name, max_nodes=33, max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=[1,3], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers)
        # env_all_list.append(env)
        # global_env.append(env)
        # probs.append(1)

        super_env=SuperEnv(
            global_env,
            hashint2env=None,
            max_possible_num_nodes=33,
            probs=probs)

    elif tset == 'M5x5Fixed':
        config['max_nodes']=25
        config['max_edges']=105

        world_name = 'Manhattan5x5_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        super_env=SuperEnv(
            global_env,
            hashint2env=None,
            max_possible_num_nodes=config['max_nodes'],
            probs=probs)

    elif tset == 'M5x5Fixed_mask_probperU_.5':
        config['max_nodes']=25
        config['max_edges']=105

        world_name = 'Manhattan5x5_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, obs_mask='prob_per_u', obs_rate=.5)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        super_env=SuperEnv(
            global_env,
            hashint2env=None,
            max_possible_num_nodes=config['max_nodes'],
            probs=probs)



    elif tset == 'TEST':
        config['max_edges']=300
        # env, _ = get_super_env(Uselected=[2], Eselected=[3,6,9], config=config, var_targets=None, apply_wrappers=apply_wrappers, remove_paths=remove_paths)
        # env_all_list += env.all_env
        # global_env.append(env)
        # probs.append(1)

        # world_name = 'Manhattan5x5_FixedEscapeInit'
        # env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers)
        # env_all_list.append(env)
        # global_env.append(env)
        # probs.append(1)

        world_name = 'Manhattan3x3_WalkAround'
        env = CreateEnv(world_name, max_nodes=9, max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        super_env=SuperEnv(
            global_env,
            hashint2env=None,
            max_possible_num_nodes=9,
            probs=probs)

    else:
        assert False

    return super_env, env_all_list

def get_train_configs(runname, load_trainset=True):
    assert False # depricated function
