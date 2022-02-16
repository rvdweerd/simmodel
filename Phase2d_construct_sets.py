import torch
import torch.nn as nn 
import torch.nn.functional as F
from modules.ppo.helpfuncs import get_super_env, CreateEnv, eval_simple, evaluate_ppo
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
    #env, _ = get_super_env(Uselected=[0,1,2,3], Eselected=[0,1,2,3,4,5,6,7,8,9], config=config)
    
    solve_select_orig = config['solve_select']
    config['solve_select'] = 'solvable'
    env, _ = get_super_env(Uselected=[0,1,2,3,4], Eselected=[6,7,8,9], config=config, variable_targets=None)
    global_env.append(env)
    config['solve_select'] = solve_select_orig
    env, _ = get_super_env(Uselected=[0,1,2,3,4], Eselected=[0,1,2,3,4,5], config=config, variable_targets=None)
    global_env.append(env)
    env, _ = get_super_env(Uselected=[0,1,2,3,4], Eselected=[0,1,2,3,4,5], config=config, variable_targets=None)
    global_env.append(env)

    world_names=[
        #'Manhattan5x5_FixedEscapeInit',
        'Manhattan5x5_VariableEscapeInit_VarTargets',
        'Manhattan5x5_VariableEscapeInit',
        #'MetroU3_e17tborder_FixedEscapeInit',
        'SparseManhattan5x5',
        'SparseManhattan5x5_VarTargets',
    ]
    for w in world_names:
        env = CreateEnv(w,max_nodes=config['max_nodes'])
        global_env.append(env)

    super_env=SuperEnv(
        global_env,
        hashint2env=None,
        max_possible_num_nodes=33,
        probs=[10,10,10,10,5,1,1])
    
    return super_env