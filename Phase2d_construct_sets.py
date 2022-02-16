import torch
import torch.nn as nn 
import torch.nn.functional as F
from modules.ppo.helpfuncs import get_super_env, CreateEnv, eval_simple, evaluate_ppo, get_logdirs
from modules.rl.environments import SuperEnv
from modules.rl.rl_policy import ActionMaskedPolicySB3_PPO
from modules.ppo.models_sb3 import s2v_ActorCriticPolicy, Struc2Vec
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def ConstructTrainSet(config):
    # Requires:
    # config['max_nodes']
    # config['nfm_func_name']
    # config['edge_blocking']
    # config['solve_select']
    global_env=[]
    probs=[]
    #env, _ = get_super_env(Uselected=[0,1,2,3], Eselected=[0,1,2,3,4,5,6,7,8,9], config=config)
    
    solve_select_orig = config['solve_select']
    config['solve_select'] = 'solvable'
    env, _ = get_super_env(Uselected=[0,1,2,3,4], Eselected=[6,7,8,9], config=config, var_targets=None)
    global_env.append(env)
    probs.append(1)

    config['solve_select'] = solve_select_orig
    env, _ = get_super_env(Uselected=[0,1,2,3,4], Eselected=[0,1,2,3,4,5], config=config, var_targets=None)
    global_env.append(env)
    probs.append(2)
    
    env, _ = get_super_env(Uselected=[0,1,2,3,4], Eselected=[0,1,2,3,4,5], config=config, var_targets=None)
    global_env.append(env)
    probs.append(2)

    world_name = 'Manhattan5x5_VariableEscapeInit'
    env = CreateEnv(world_name, max_nodes=config['max_nodes'], var_targets=None)
    global_env.append(env)
    probs.append(2)

    world_name = 'Manhattan5x5_VariableEscapeInit'
    env = CreateEnv(world_name, max_nodes=config['max_nodes'], var_targets=[1,3])
    global_env.append(env)
    probs.append(2)

    world_name = 'Manhattan5x5_FixedEscapeInit'
    env = CreateEnv(world_name, max_nodes=config['max_nodes'], var_targets=None)
    global_env.append(env)
    probs.append(1)

    world_name = 'Manhattan5x5_FixedEscapeInit'
    env = CreateEnv(world_name, max_nodes=config['max_nodes'], var_targets=[1,3])
    global_env.append(env)
    probs.append(1)

    world_name = 'Manhattan5x5_FixedEscapeInit'
    env = CreateEnv(world_name, max_nodes=config['max_nodes'], var_targets=[1,1],remove_world_pool=True)
    global_env.append(env)
    probs.append(1)    

    world_name = 'SparseManhattan5x5'
    env = CreateEnv(world_name, max_nodes=config['max_nodes'], var_targets=[1,3])
    global_env.append(env)
    probs.append(1)

    super_env=SuperEnv(
        global_env,
        hashint2env=None,
        max_possible_num_nodes=33,
        probs=probs)
    
    return super_env

def get_train_configs():
    config={}
    config['RunA']={}
    config['RunA']['train_on']      = 'ContructedSuperSet'
    config['RunA']['solve_select']  = 'both'
    config['RunA']['edge_blocking'] = True
    config['RunA']['scenario_name'] = ''
    config['RunA']['nfm_func_name'] = 'NFM_ev_ec_t_um_us'
    config['RunA']['s2v_layers']    = 2
    config['RunA']['emb_dim']       = 64
    config['RunA']['emb_iter_T']    = 5
    config['RunA']['num_step']      = 500000
    config['RunA']['seed0']         = 0
    config['RunA']['numseeds']      = 1
    config['RunA']['max_nodes']     = 33
    rootdir, logdir = get_logdirs(config['RunA'])
    config['RunA']['rootdir'] = rootdir
    config['RunA']['logdir'] = logdir

    config['RunB']={}
    config['RunB']['train_on']      = 'ContructedSuperSet'
    config['RunB']['solve_select']  = 'both'
    config['RunB']['edge_blocking'] = True
    config['RunB']['scenario_name'] = ''
    config['RunB']['nfm_func_name'] = 'NFM_ev_ec_t_um_us'
    config['RunB']['s2v_layers']    = 2
    config['RunB']['emb_dim']       = 64
    config['RunB']['emb_iter_T']    = 5
    config['RunB']['num_step']      = 300000
    config['RunB']['seed0']         = 0
    config['RunB']['numseeds']      = 1
    config['RunB']['max_nodes']     = 33
    rootdir, logdir = get_logdirs(config['RunB'])
    config['RunB']['rootdir'] = rootdir
    config['RunB']['logdir'] = logdir

    config['RunC']={}
    config['RunC']['train_on']      = 'ContructedSuperSet'
    config['RunC']['solve_select']  = 'solvable'
    config['RunC']['edge_blocking'] = True
    config['RunC']['scenario_name'] = ''
    config['RunC']['nfm_func_name'] = 'NFM_ev_ec_t_um_us'
    config['RunC']['s2v_layers']    = 2
    config['RunC']['emb_dim']       = 64
    config['RunC']['emb_iter_T']    = 5
    config['RunC']['num_step']      = 300000
    config['RunC']['seed0']         = 0
    config['RunC']['numseeds']      = 1
    config['RunC']['max_nodes']     = 33
    rootdir, logdir = get_logdirs(config['RunC'])
    config['RunC']['rootdir'] = rootdir
    config['RunC']['logdir'] = logdir

    key='runD'
    config[key]={}
    config[key]['train_on']      = 'ContructedSuperSet'
    config[key]['solve_select']  = 'solvable'
    config[key]['edge_blocking'] = True
    config[key]['scenario_name'] = ''
    config[key]['nfm_func_name'] = 'NFM_ev_ec_t_um_us'
    config[key]['s2v_layers']    = 2
    config[key]['emb_dim']       = 128
    config[key]['emb_iter_T']    = 5
    config[key]['num_step']      = 500000
    config[key]['seed0']         = 0
    config[key]['numseeds']      = 1
    config[key]['max_nodes']     = 33
    rootdir, logdir = get_logdirs(config[key])
    config[key]['rootdir'] = rootdir
    config[key]['logdir'] = logdir

    key='runE'
    config[key]={}
    config[key]['train_on']      = 'ContructedSuperSet'
    config[key]['solve_select']  = 'solvable'
    config[key]['edge_blocking'] = True
    config[key]['scenario_name'] = ''
    config[key]['nfm_func_name'] = 'NFM_ev_ec_t_um_us'
    config[key]['s2v_layers']    = 2
    config[key]['emb_dim']       = 128
    config[key]['emb_iter_T']    = 7
    config[key]['num_step']      = 500000
    config[key]['seed0']         = 0
    config[key]['numseeds']      = 1
    config[key]['max_nodes']     = 33
    rootdir, logdir = get_logdirs(config[key])
    config[key]['rootdir'] = rootdir
    config[key]['logdir'] = logdir

    return config

