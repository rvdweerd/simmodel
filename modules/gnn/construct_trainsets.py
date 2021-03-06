import torch
import torch.nn as nn 
import torch.nn.functional as F
from modules.ppo.helpfuncs import get_super_env, CreateEnv, eval_simple, evaluate_ppo#, get_logdirs
from modules.rl.environments import SuperEnv
#from modules.rl.rl_policy import ActionMaskedPolicySB3_PPO
#from modules.ppo.models_sb3 import s2v_ActorCriticPolicy, Struc2VecExtractor
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def ConstructTrainSet(config, apply_wrappers=True, type_obs_wrap='obs_flat', remove_paths=False, tset='M3M5Mix'):
    # Requires:
    # config['max_nodes']
    # config['nfm_func']
    # config['edge_blocking']
    # config['solve_select']
    global_env=[]
    probs=[]
    env_all_list=[]
    config['solve_select'] = 'solvable'

    if tset == 'MixM5Met_T':
        config['max_edges'] = 119
        config['max_nodes'] = 33
        
        world_name = 'Manhattan5x5_VariableEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=[1,2], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        world_name = 'Manhattan5x5_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=[1,2], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        world_name = 'MetroU1_e17tborder_VariableEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=[1,2], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        world_name = 'MetroU1_e17tborder_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=[1,2], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1) 

        world_name = 'MetroU2_e17tborder_VariableEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=[1,2], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        world_name = 'MetroU2_e17tborder_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=[1,2], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)        

        world_name = 'MetroU3_e17tborder_VariableEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=[1,2], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        world_name = 'MetroU3_e17tborder_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=[1,2], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        super_env=SuperEnv(
            global_env,
            hashint2env=None,
            max_possible_num_nodes=config['max_nodes'],
            probs=probs)

    elif tset == 'M3_012456':
        config['max_edges'] = 33
        config['max_nodes'] = 9
        env, _ = get_super_env(Uselected=[0,1,2], Eselected=[0,1,2,4,5,6,7], config=config, var_targets=[1,3], apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap, remove_paths=remove_paths)
        env_all_list += env.all_env
        global_env.append(env)
        probs.append(1)
     
        super_env=SuperEnv(
            global_env,
            hashint2env=None,
            max_possible_num_nodes=9,
            probs=probs)

    elif tset == 'M3M5Mix':
        config['max_edges'] = 300
        config['max_nodes'] = 25
        env, _ = get_super_env(Uselected=[0,1,2,3,4], Eselected=[6,7,8,9], config=config, var_targets=None, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap, remove_paths=remove_paths)
        env_all_list += env.all_env
        global_env.append(env)
        probs.append(1)
        
        env, _ = get_super_env(Uselected=[0,1,2,3,4], Eselected=[0,1,2,3,4,5], config=config, var_targets=None, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap, remove_paths=remove_paths)
        env_all_list += env.all_env
        global_env.append(env)
        probs.append(2)

        env, _ = get_super_env(Uselected=[0,1,2,3,4], Eselected=[0,1,2,3,4,5], config=config, var_targets=None, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap, remove_paths=remove_paths)
        env_all_list += env.all_env
        global_env.append(env)
        probs.append(2)

        world_name = 'Manhattan5x5_VariableEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(2)

        world_name = 'Manhattan5x5_VariableEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], nfm_func_name=config['nfm_func'], var_targets=[1,3], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(2)

        world_name = 'Manhattan5x5_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        world_name = 'Manhattan5x5_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], nfm_func_name=config['nfm_func'], var_targets=[1,3], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        world_name = 'Manhattan5x5_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], nfm_func_name=config['nfm_func'], var_targets=[1,1],remove_world_pool=True, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)    

        world_name = 'SparseManhattan5x5'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], nfm_func_name=config['nfm_func'], var_targets=[1,3], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        super_env=SuperEnv(
            global_env,
            hashint2env=None,
            max_possible_num_nodes=33,
            probs=probs)

    elif tset in ['NWB_AMS','NWB_AMS2']:
        if tset == 'NWB_AMS': # corresponds to U optim method1: random walk assumption for E
            world_nameV = 'NWB_test_VariableEscapeInit'
            world_nameF = 'NWB_test_FixedEscapeInit'
        else: # corresponds to U optim method2: shortest path walk to targt nodes assumption for E
            world_nameV = 'NWB_test_VariableEscapeInit2'
            world_nameF = 'NWB_test_FixedEscapeInit2'

        env = CreateEnv(world_nameV, max_nodes=975, max_edges=1425, nfm_func_name = config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        if tset == 'NWB_AMS': # can't apply variable targets to optim2; would require U optim at every step: too costly
            env = CreateEnv(world_nameV, max_nodes=975, max_edges=1425, nfm_func_name = config['nfm_func'], var_targets=[10,20], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap)
            env_all_list.append(env)
            global_env.append(env)
            probs.append(1)

        env = CreateEnv(world_nameF, max_nodes=975, max_edges=1425, nfm_func_name = config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(.1)

        if tset == 'NWB_AMS': # can't apply variable targets to optim2; would require U optim at every step: too costly
            env = CreateEnv(world_nameF, max_nodes=975, max_edges=1425, nfm_func_name = config['nfm_func'], var_targets=[10,20], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap)
            env_all_list.append(env)
            global_env.append(env)
            probs.append(1)

        super_env=SuperEnv(
            global_env,
            hashint2env=None,
            max_possible_num_nodes=975,
            probs=probs)

    elif tset == 'NWB_AMS+Mix':
        config['max_edges'] = 1425
        config['max_nodes'] = 975

        world_name = 'NWB_test_VariableEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name = config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap, obs_mask=config['obs_mask'], obs_rate=config['obs_rate'])
        env_all_list.append(env)
        global_env.append(env)
        probs.append(3)

        world_name = 'NWB_test_VariableEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name = config['nfm_func'], var_targets=[10,20], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap, obs_mask=config['obs_mask'], obs_rate=config['obs_rate'])
        env_all_list.append(env)
        global_env.append(env)
        probs.append(3)

        world_name = 'NWB_test_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name = config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap, obs_mask=config['obs_mask'], obs_rate=config['obs_rate'])
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        world_name = 'NWB_test_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name = config['nfm_func'], var_targets=[10,20], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap, obs_mask=config['obs_mask'], obs_rate=config['obs_rate'])
        env_all_list.append(env)
        global_env.append(env)
        probs.append(3)

        world_name = 'MetroU3_e17tborder_VariableEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap, obs_mask=config['obs_mask'], obs_rate=config['obs_rate'])
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        world_name = 'MetroU3_e17tborder_VariableEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=[1,2], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap, obs_mask=config['obs_mask'], obs_rate=config['obs_rate'])
        env_all_list.append(env)
        global_env.append(env)
        probs.append(3)

        world_name = 'MetroU3_e17tborder_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap, obs_mask=config['obs_mask'], obs_rate=config['obs_rate'])
        env_all_list.append(env)
        global_env.append(env)
        probs.append(2)

        world_name = 'MetroU3_e17tborder_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=[1,2], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap, obs_mask=config['obs_mask'], obs_rate=config['obs_rate'])
        env_all_list.append(env)
        global_env.append(env)
        probs.append(3)

        env, _ = get_super_env(Uselected=[0,1,2,3,4], Eselected=[6,7,8,9], config=config, var_targets=None, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap, remove_paths=remove_paths)
        env_all_list += env.all_env
        global_env.append(env)
        probs.append(5)
        
        env, _ = get_super_env(Uselected=[0,1,2,3,4], Eselected=[0,1,2,3,4,5], config=config, var_targets=None, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap, remove_paths=remove_paths)
        env_all_list += env.all_env
        global_env.append(env)
        probs.append(5)

        env, _ = get_super_env(Uselected=[0,1,2,3,4], Eselected=[0,1,2,3,4,5], config=config, var_targets=None, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap, remove_paths=remove_paths)
        env_all_list += env.all_env
        global_env.append(env)
        probs.append(5)

        world_name = 'Manhattan5x5_VariableEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap, obs_mask=config['obs_mask'], obs_rate=config['obs_rate'])
        env_all_list.append(env)
        global_env.append(env)
        probs.append(2)

        world_name = 'Manhattan5x5_VariableEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=[1,3], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap, obs_mask=config['obs_mask'], obs_rate=config['obs_rate'])
        env_all_list.append(env)
        global_env.append(env)
        probs.append(2)

        world_name = 'Manhattan5x5_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap, obs_mask=config['obs_mask'], obs_rate=config['obs_rate'])
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        world_name = 'Manhattan5x5_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=[1,3], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap, obs_mask=config['obs_mask'], obs_rate=config['obs_rate'])
        env_all_list.append(env)
        global_env.append(env)
        probs.append(2)

        world_name = 'Manhattan5x5_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=[1,1],remove_world_pool=True, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap, obs_mask=config['obs_mask'], obs_rate=config['obs_rate'])
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)    

        world_name = 'SparseManhattan5x5'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=[1,3], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap, obs_mask=config['obs_mask'], obs_rate=config['obs_rate'])
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        super_env=SuperEnv(
            global_env,
            hashint2env=None,
            max_possible_num_nodes=config['max_nodes'],
            probs=probs)

    elif tset in ['NWB_AMS_mixed_obs','NWB_AMS2_mixed_obs']:
        config['max_nodes']=975
        config['max_edges']=1425
        assert config['obs_mask']=='mix'

        if tset == 'NWB_AMS_mixed_obs': # corresponds to U optim method1: random walk assumption for E
            world_set = ['NWB_test_VariableEscapeInit','NWB_test_FixedEscapeInit']
        else: # corresponds to U optim method2: shortest path walk to targt nodes assumption for E
            world_set = ['NWB_test_VariableEscapeInit2','NWB_test_FixedEscapeInit2']

        for world_name in world_set:
            env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap, obs_mask='None', obs_rate=1.0)
            env_all_list.append(env)
            global_env.append(env)
            probs.append(1)

            for rate in [0.9, 0.8, 0.7,0.6,0.5]:
                env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap, obs_mask='prob_per_u', obs_rate=rate)
                env_all_list.append(env)
                global_env.append(env)
            probs+=[1,3,5,3,1]

            if tset == 'NWB_AMS_mixed_obs': # # can't apply variable targets to optim2; would require U optim at every step: too costly 
                env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=[10,20], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap, obs_mask='None', obs_rate=1.0)
                env_all_list.append(env)
                global_env.append(env)
                probs.append(1)

                for rate in [0.9, 0.8, 0.7,0.6,0.5]:
                    env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=[10,20], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap, obs_mask='prob_per_u', obs_rate=rate)
                    env_all_list.append(env)
                    global_env.append(env)
                probs+=[1,3,5,3,1]

        super_env=SuperEnv(
            global_env,
            hashint2env=None,
            max_possible_num_nodes=config['max_nodes'],
            probs=probs)       

    elif tset == 'NWB_AMS_mixed_obs2':
        config['max_nodes']=975
        config['max_edges']=1425
        assert config['obs_mask']=='mix'
        for world_name in ['NWB_test_FixedEscapeInit', 'NWB_test_VariableEscapeInit']:
            env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap, obs_mask='None', obs_rate=1.0)
            env_all_list.append(env)
            global_env.append(env)
            probs.append(10)

            for rate in [0.9, 0.8, 0.7,0.6,0.5,.4,.3,.2,.1,.0]:
                env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap, obs_mask='prob_per_u', obs_rate=rate)
                env_all_list.append(env)
                global_env.append(env)
            probs+=[1,1,1,1,1,1,1,1,1,1]

            env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=[10,20], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap, obs_mask='None', obs_rate=1.0)
            env_all_list.append(env)
            global_env.append(env)
            probs.append(10)

            for rate in [0.9, 0.8, 0.7,0.6,0.5,.4,.3,.2,.1,.0]:
                env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=[10,20], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap, obs_mask='prob_per_u', obs_rate=rate)
                env_all_list.append(env)
                global_env.append(env)
            probs+=[1,1,1,1,1,1,1,1,1,1]

        super_env=SuperEnv(
            global_env,
            hashint2env=None,
            max_possible_num_nodes=config['max_nodes'],
            probs=probs)       

    elif tset == 'MetroConstructed':
        world_name = 'MetroU3_e17tborder_VariableEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=300, nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        world_name = 'MetroU3_e17tborder_VariableEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=300, nfm_func_name=config['nfm_func'], var_targets=[1,2], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        world_name = 'MetroU3_e17tborder_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=300, nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        world_name = 'MetroU3_e17tborder_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=300, nfm_func_name=config['nfm_func'], var_targets=[1,2], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        world_name = 'MetroU3_e1t31_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=300, nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(2)

        world_name = 'MetroU3_e1t31_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=300, nfm_func_name=config['nfm_func'], var_targets=[1,2], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        super_env=SuperEnv(
            global_env,
            hashint2env=None,
            max_possible_num_nodes=33,
            probs=probs)

    elif tset == 'Metro_mixed_obs':
        config['max_nodes']=33
        config['max_edges']=119
        assert config['obs_mask']=='mix'
        world_name = 'MetroU3_e17tborder_FixedEscapeInit'

        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap, obs_mask='None', obs_rate=1.0)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        for rate in [0.9, 0.8, 0.7,0.6,0.5]:
            env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap, obs_mask='prob_per_u', obs_rate=rate)
            env_all_list.append(env)
            global_env.append(env)
            probs.append(1)

        super_env=SuperEnv(
            global_env,
            hashint2env=None,
            max_possible_num_nodes=config['max_nodes'],
            probs=probs)

    elif tset == 'MixAll33':
        config['max_edges'] = 300
        config['max_nodes'] = 33
        world_name = 'MetroU3_e17tborder_VariableEscapeInit'
        env = CreateEnv(world_name, max_nodes=33, max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        world_name = 'MetroU3_e17tborder_VariableEscapeInit'
        env = CreateEnv(world_name, max_nodes=33, max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=[1,2], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        world_name = 'MetroU3_e17tborder_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=33, max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        world_name = 'MetroU3_e17tborder_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=33, max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=[1,2], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        world_name = 'MetroU3_e1t31_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=33, max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(2)

        world_name = 'MetroU3_e1t31_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=33, max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=[1,2], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        env, _ = get_super_env(Uselected=[0,1,2,3,4], Eselected=[6,7,8,9], config=config, var_targets=None, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap, remove_paths=remove_paths)
        env_all_list += env.all_env
        global_env.append(env)
        probs.append(1)
        
        env, _ = get_super_env(Uselected=[0,1,2,3,4], Eselected=[0,1,2,3,4,5], config=config, var_targets=None, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap, remove_paths=remove_paths)
        env_all_list += env.all_env
        global_env.append(env)
        probs.append(2)

        env, _ = get_super_env(Uselected=[0,1,2,3,4], Eselected=[0,1,2,3,4,5], config=config, var_targets=None, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap, remove_paths=remove_paths)
        env_all_list += env.all_env
        global_env.append(env)
        probs.append(2)

        world_name = 'Manhattan5x5_VariableEscapeInit'
        env = CreateEnv(world_name, max_nodes=33, max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(2)

        world_name = 'Manhattan5x5_VariableEscapeInit'
        env = CreateEnv(world_name, max_nodes=33, max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=[1,3], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(2)

        world_name = 'Manhattan5x5_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=33, max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        world_name = 'Manhattan5x5_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=33, max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=[1,3], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        world_name = 'Manhattan5x5_FixedEscapeInit'
        env = CreateEnv(world_name, max_nodes=33, max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=[1,1],remove_world_pool=True, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)    

        world_name = 'SparseManhattan5x5'
        env = CreateEnv(world_name, max_nodes=33, max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=[1,3], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        # world_name = 'Manhattan3x3_WalkAround'
        # env = CreateEnv(world_name, max_nodes=33, max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=[1,3], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap)
        # env_all_list.append(env)
        # global_env.append(env)
        # probs.append(1)

        # world_name = 'Manhattan5x5_DuplicateSetA'
        # env = CreateEnv(world_name, max_nodes=33, max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=[1,3], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap)
        # env_all_list.append(env)
        # global_env.append(env)
        # probs.append(1)

        # world_name = 'Manhattan5x5_DuplicateSetB'
        # env = CreateEnv(world_name, max_nodes=33, max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=[1,3], remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap)
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
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap, obs_mask=config['obs_mask'], obs_rate=config['obs_rate'])
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        super_env=SuperEnv(
            global_env,
            hashint2env=None,
            max_possible_num_nodes=config['max_nodes'],
            probs=probs)

    elif tset == 'MemTask-U1+2':
        config['max_nodes']=8
        config['max_edges']=14

        world_name = 'MemoryTaskU1'
        env = CreateEnv(world_name, max_nodes=8, max_edges=14, nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap, obs_mask=config['obs_mask'], obs_rate=config['obs_rate'])
        env_all_list.append(env)
        global_env.append(env)
        probs.append(2)

        world_name = 'MemoryTaskU2T'
        env = CreateEnv(world_name, max_nodes=8, max_edges=14, nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap, obs_mask=config['obs_mask'], obs_rate=config['obs_rate'])
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        world_name = 'MemoryTaskU2B'
        env = CreateEnv(world_name, max_nodes=8, max_edges=14, nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap, obs_mask=config['obs_mask'], obs_rate=config['obs_rate'])
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        super_env=SuperEnv(
            global_env,
            hashint2env=None,
            max_possible_num_nodes=config['max_nodes'],
            probs=probs)

    elif tset == 'MemTask-U1':
        config['max_nodes']=10#8
        config['max_edges']=22

        world_name = 'MemoryTaskU1'
        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap, obs_mask=config['obs_mask'], obs_rate=config['obs_rate'])
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        super_env=SuperEnv(
            global_env,
            hashint2env=None,
            max_possible_num_nodes=config['max_nodes'],
            probs=probs)

    elif tset == 'MemTaskLong-U1':
        config['max_nodes']=11
        config['max_edges']=20

        world_name = 'MemoryTaskU1Long'
        env = CreateEnv(world_name, max_nodes=11, max_edges=20, nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap, obs_mask=config['obs_mask'], obs_rate=config['obs_rate'])
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        super_env=SuperEnv(
            global_env,
            hashint2env=None,
            max_possible_num_nodes=config['max_nodes'],
            probs=probs)

    elif tset == 'M5x5F_mixed_obs':
        config['max_nodes']=25
        config['max_edges']=105
        assert config['obs_mask']=='mix'
        world_name = 'Manhattan5x5_FixedEscapeInit'

        env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap, obs_mask='None', obs_rate=1.0)
        env_all_list.append(env)
        global_env.append(env)
        probs.append(1)

        for rate in [0.9, 0.8, 0.7,0.6,0.5]:
            env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap, obs_mask='prob_per_u', obs_rate=rate)
            env_all_list.append(env)
            global_env.append(env)
            probs.append(1)

        super_env=SuperEnv(
            global_env,
            hashint2env=None,
            max_possible_num_nodes=config['max_nodes'],
            probs=probs)

    elif tset == 'TEST':
        config['max_nodes']=9
        config['max_edges']=21

        env, _ = get_super_env(Uselected=[2], Eselected=[3,9], config=config, var_targets=None, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap, remove_paths=remove_paths)
        env_all_list += env.all_env
        global_env.append(env)
        probs.append(1)
        super_env=env
        
        # world_name = 'Manhattan5x5_FixedEscapeInit'
        # env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap)
        # env_all_list.append(env)
        # global_env.append(env)
        # probs.append(1)

        # world_name = 'Manhattan3x3_WalkAround'
        # env = CreateEnv(world_name, max_nodes=config['max_nodes'], max_edges=config['max_edges'], nfm_func_name=config['nfm_func'], var_targets=None, remove_world_pool=remove_paths, apply_wrappers=apply_wrappers, type_obs_wrap=type_obs_wrap, obs_mask=config['obs_mask'], obs_rate=config['obs_rate'])
        # env_all_list.append(env)
        # global_env.append(env)
        # probs.append(1)

        # super_env=SuperEnv(
        #     global_env,
        #     hashint2env=None,
        #     max_possible_num_nodes=config['max_nodes'],
        #     probs=probs)

    else:
        assert False

    return super_env, env_all_list

def get_train_configs(runname, load_trainset=True):
    assert False # depricated function
