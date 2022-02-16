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

def get_train_configs(runname, load_trainset=True):
    config={}

    if runname == 'RunB':
        config[runname]={}
        config[runname]['train_on']      = 'ContructedSuperSet'
        config[runname]['solve_select']  = 'both'
        config[runname]['edge_blocking'] = True
        config[runname]['scenario_name'] = ''
        config[runname]['nfm_func_name'] = 'NFM_ev_ec_t_um_us'
        config[runname]['node_dim'] = 5
        config[runname]['s2v_layers']    = 2
        config[runname]['emb_dim']       = 64
        config[runname]['emb_iter_T']    = 5
        config[runname]['num_step']      = 300000
        config[runname]['seed0']         = 1
        config[runname]['numseeds']      = 5
        config[runname]['max_nodes']     = 33
        rootdir, logdir = get_logdirs(config[runname])
        config[runname]['rootdir'] = rootdir
        config[runname]['logdir'] = logdir
        env = ConstructTrainSet(config[runname])
        config[runname]['env_train'] = env
    elif runname == 'RunC':
        config[runname]={}
        config[runname]['train_on']      = 'ContructedSuperSet'
        config[runname]['solve_select']  = 'solvable'
        config[runname]['edge_blocking'] = True
        config[runname]['scenario_name'] = ''
        config[runname]['nfm_func_name'] = 'NFM_ev_ec_t_um_us'
        config[runname]['node_dim'] = 5
        config[runname]['s2v_layers']    = 2
        config[runname]['emb_dim']       = 64
        config[runname]['emb_iter_T']    = 5
        config[runname]['num_step']      = 300000
        config[runname]['seed0']         = 1
        config[runname]['numseeds']      = 5
        config[runname]['max_nodes']     = 33
        rootdir, logdir = get_logdirs(config[runname])
        config[runname]['rootdir'] = rootdir
        config[runname]['logdir'] = logdir
        if load_trainset:
            env = ConstructTrainSet(config['RunC'])
            config['RunC']['env_train'] = env
        else:
            config['RunC']['env_train'] = None    
    elif runname == 'RunD':
        key='RunD'
        config[key]={}
        config[key]['train_on']      = 'ContructedSuperSet'
        config[key]['solve_select']  = 'solvable'
        config[key]['edge_blocking'] = True
        config[key]['scenario_name'] = ''
        config[key]['nfm_func_name'] = 'NFM_ev_ec_t_um_us'
        config[key]['node_dim'] = 5
        config[key]['s2v_layers']    = 2
        config[key]['emb_dim']       = 128
        config[key]['emb_iter_T']    = 5
        config[key]['num_step']      = 500000
        config[key]['seed0']         = 1
        config[key]['numseeds']      = 5
        config[key]['max_nodes']     = 33
        rootdir, logdir = get_logdirs(config[key])
        config[key]['rootdir'] = rootdir
        config[key]['logdir'] = logdir
        env = ConstructTrainSet(config[key])
        config[key]['env_train'] = env    
    
    elif runname == 'RunE':
        key='RunE'
        config[key]={}
        config[key]['train_on']      = 'ContructedSuperSet'
        config[key]['solve_select']  = 'solvable'
        config[key]['edge_blocking'] = True
        config[key]['scenario_name'] = ''
        config[key]['nfm_func_name'] = 'NFM_ev_ec_t_um_us'
        config[key]['node_dim'] = 5    
        config[key]['s2v_layers']    = 2
        config[key]['emb_dim']       = 128
        config[key]['emb_iter_T']    = 7
        config[key]['num_step']      = 500000
        config[key]['seed0']         = 1
        config[key]['numseeds']      = 5
        config[key]['max_nodes']     = 33
        rootdir, logdir = get_logdirs(config[key])
        config[key]['rootdir'] = rootdir
        config[key]['logdir'] = logdir
        env = ConstructTrainSet(config[key])
        config[key]['env_train'] = env    

    elif runname == 'train_on_metro':
        key='train_on_metro'
        config[key]={}
        config[key]['train_on']      = 'MetroWorlds'
        config[key]['solve_select']  = 'both'
        config[key]['edge_blocking'] = True
        config[key]['scenario_name'] = ''
        config[key]['nfm_func_name'] = 'NFM_ev_ec_t_um_us'
        config[key]['node_dim'] = 5    
        config[key]['s2v_layers']    = 2
        config[key]['emb_dim']       = 64
        config[key]['emb_iter_T']    = 5
        config[key]['num_step']      = 300000
        config[key]['seed0']         = 1
        config[key]['numseeds']      = 5
        config[key]['max_nodes']     = 33
        rootdir, logdir = get_logdirs(config[key])
        config[key]['rootdir'] = rootdir
        config[key]['logdir'] = logdir
        global_env=[]
        env = CreateEnv('MetroU3_e17tborder_FixedEscapeInit',max_nodes=33,var_targets=None, remove_world_pool=False)
        global_env.append(env)
        env = CreateEnv('MetroU3_e17tborder_FixedEscapeInit',max_nodes=33,var_targets=[1,5], remove_world_pool=True)
        global_env.append(env)
        env=SuperEnv(global_env, None, max_possible_num_nodes = 33)
        config[key]['env_train'] = env    

    return config

