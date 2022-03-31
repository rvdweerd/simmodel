#from torch import nn
#from torch import distributions
#import copy
#from itertools import count
#from IPython.display import HTML
#from Phase1_hyperparameters import GetHyperParams_PPO_RNN
import torch
import gym
from torch.utils.tensorboard import SummaryWriter
from torch import optim
import numpy as np
import torch.nn.functional as F
import re
import time
import math
import pathlib
import time
import pickle
import os
from dataclasses import dataclass
#import gc
from dotmap import DotMap
#from base64 import b64encode
from modules.rl.rl_custom_worlds import GetCustomWorld
from modules.ppo.helpfuncs import get_super_env
from modules.sim.graph_factory import GetWorldSet
import argparse
from modules.rl.environments import SuperEnv
from modules.dqn.dqn_utils import seed_everything
from modules.sim.simdata_utils import SimulateAutomaticMode_PPO
from modules.rl.rl_utils import GetFullCoverageSample
from modules.ppo.ppo_custom import gather_trajectories,split_trajectories_episodes,  pad_and_compute_returns, start_or_resume_from_checkpoint, train_model,  save_checkpoint , save_parameters, StopConditions, TrajectoryDataset, HyperParameters
from modules.gnn.construct_trainsets import ConstructTrainSet
import modules.gnn.nfm_gen
from modules.ppo.models_ngo import MaskablePPOPolicy, MaskablePPOPolicy_shared_lstm, MaskablePPOPolicy_shared_lstm_concat
from collections.abc import Iterable
from gym.vector.async_vector_env import AsyncVectorEnv
from gym.vector.sync_vector_env import SyncVectorEnv
from gym.vector.vector_env import VectorEnv, VectorEnvWrapper
import modules.gnn.nfm_gen
from modules.ppo.ppo_wrappers import PPO_ActWrapper, PPO_ObsWrapper, PPO_ObsDictWrapper, VarTargetWrapper, PPO_ObsFlatWrapper
#__all__ = ["AsyncVectorEnv", "SyncVectorEnv", "VectorEnv", "VectorEnvWrapper", "make"]

# Heavily adapted to work with customized environment and GNNs (trainable with varying graph sizes in the trainset) from: 
# https://gitlab.com/ngoodger/ppo_lstm/-/blob/master/recurrent_ppo.ipynb

# Set maximum threads for torch to avoid inefficient use of multiple cpu cores.
torch.set_num_threads(6)

def make_custom(config, num_envs=1, asynchronous=True, wrappers=None, **kwargs):
    """Create a vectorized environment from multiple copies of an environment,
    from its id.
    Parameters
    """
    from gym.envs import make as make_

    def _make_env():
        #env = make_(id, **kwargs)
        #env=GetCustomWorld(env_id, make_reflexive=MAKE_REFLEXIVE, state_repr=STATE_REPR, state_enc='tensors')
        #max_nodes = 9
        state_repr='etUte0U0'
        state_enc='nfm'
        edge_blocking = True
        remove_world_pool = False
        nfm_func_name='NFM_ev_ec_t_dt_at_um_us'
        var_targets = None#[1,1]
        apply_wrappers = True        

        # METHOD 0
        env,_ = ConstructTrainSet(config, apply_wrappers=True, remove_paths=False, tset=config['train_on'])

        # METHOD 1
        # env = GetCustomWorld(world_name, make_reflexive=True, state_repr=state_repr, state_enc=state_enc)
        # env.redefine_nfm(modules.gnn.nfm_gen.nfm_funcs[nfm_func_name])
        # env.capture_on_edges = edge_blocking
        # if remove_world_pool:
        #     env._remove_world_pool()
        # if var_targets is not None:
        #     env = VarTargetWrapper(env, var_targets)
        # if apply_wrappers:
        #     env = PPO_ObsFlatWrapper(env, max_possible_num_nodes = hp.max_possible_nodes, max_possible_num_edges=hp.max_possible_edges)
        #     env = PPO_ActWrapper(env) 

        # METHOD 2
        #env_all_train, hashint2env, env2hashint, env2hashstr, eprobs = GetWorldSet(state_repr, state_enc, U=[2], E=[0,6], edge_blocking=edge_blocking, solve_select='solvable', reject_duplicates=False, nfm_func=modules.gnn.nfm_gen.nfm_funcs[nfm_func_name], var_targets=None, remove_paths=False, return_probs=True)
        # if apply_wrappers:
        #     for i in range(len(env_all_train)):
        #         env_all_train[i]=PPO_ObsFlatWrapper(env_all_train[i], max_possible_num_nodes = hp.max_possible_nodes, max_possible_num_edges=hp.max_possible_edges)        
        #         env_all_train[i]=PPO_ActWrapper(env_all_train[i])        
        # env = SuperEnv(env_all_train, hashint2env, max_possible_num_nodes = hp.max_possible_nodes, probs=eprobs)

        # METHOD 3
        # config={}
        # config['max_edges']=300
        # config['max_nodes']=33
        # config['nfm_func']='NFM_ev_ec_t_dt_at_um_us'
        # config['remove_paths']=False
        # config['edge_blocking']=True
        # config['train_on']='MixAll33'
        # env, env_all_train_list = ConstructTrainSet(config, apply_wrappers=True, remove_paths=config['remove_paths'], tset=config['train_on'])

        # METHOD 4
        # with open(WORKSPACE_PATH + "mixall_vecenv.bin", "wb") as f:
        #     pickle.dump(env, f)
        # with open("./results/mixall_vecenv.bin", "rb") as f:
        #     env = pickle.load(f)       

        if wrappers is not None:
            if callable(wrappers):
                env = wrappers(env)
            elif isinstance(wrappers, Iterable) and all(
                [callable(w) for w in wrappers]
            ):
                for wrapper in wrappers:
                    env = wrapper(env)
            else:
                raise NotImplementedError
        
        return env

    env_fns = [_make_env for _ in range(num_envs)]
    return AsyncVectorEnv(env_fns) if asynchronous else SyncVectorEnv(env_fns)



from modules.rl.rl_policy import LSTM_GNN_PPO_Policy
from modules.rl.rl_utils import EvaluatePolicy
import pickle
def EvaluateSavedModel():
    #env=GetCustomWorld(WORLD_NAME, make_reflexive=MAKE_REFLEXIVE, state_repr=STATE_REPR, state_enc='tensors')
    world_name=WORLD_NAME
    env_ = make_custom(world_name, hp.parallel_rollouts, asynchronous=ASYNCHRONOUS_ENVIRONMENT)   
    lstm_ppo_model, ppo_optimizer, max_checkpoint_iteration, stop_conditions = start_or_resume_from_checkpoint(env_, hp, tp)
    
    env=env_.envs[0]
    policy = LSTM_GNN_PPO_Policy(env, lstm_ppo_model, deterministic=tp['eval_deterministic'])
    while True:
        entries=None#[5012,218,3903]
        #demo_env = random.choice(evalenv)
        a = SimulateAutomaticMode_PPO(env, policy, t_suffix=False, entries=entries)
        if a == 'Q': break
    
    lengths, returns, captures, solves = EvaluatePolicy(env, policy  , env.world_pool*SAMPLE_MULTIPLIER, print_runs=False, save_plots=False, logdir=exp_rootdir)    
    plotlist = GetFullCoverageSample(returns, env.world_pool*SAMPLE_MULTIPLIER, bins=10, n=10)
    EvaluatePolicy(env, policy, plotlist, print_runs=True, save_plots=True, logdir=exp_rootdir)

def GetConfigs(args):
    config = {}
    #for i in range(args.num_configs):
    #    config.add(args.config_class(**args.config_args))
    #return config
    #config['lstm_dropout']    = args.lstm_dropout
    config['train_on']        = args.train_on
    config['batch_size']      = args.batch_size
    config['mask_type']       = args.mask_type
    config['mask_rate']       = args.mask_rate
    config['emb_dim']         = args.emb_dim
    config['lstm_type']       = args.lstm_type
    config['lstm_hdim']       = args.lstm_hdim
    config['lstm_layers']     = args.lstm_layers
    config['emb_iterT']       = args.emb_iterT
    config['nfm_func']        = args.nfm_func
    config['qnet']            = args.qnet
    config['train']           = args.train
    config['eval']            = args.eval
    config['test']            = args.test
    config['num_seeds']       = args.num_seeds
    config['seed0']           = args.seed0
    config['seedrange']=range(config['seed0'], config['seed0']+config['num_seeds'])
    config['demoruns']        = args.demoruns
    lstm_filestring = config['lstm_type']
    if config['lstm_type'] != 'none':
        lstm_filestring  += '_' + str(config['lstm_hdim']) + '_' + str(config['lstm_layers'])
    config['rootdir'] = './results/results_Phase3/ppo/'+ config['train_on']+'/'+ config['qnet'] + '/lstm_' + lstm_filestring + '_bsize' + str(config['batch_size'])
    config['logdir']  = config['rootdir'] + '/' + config['nfm_func']+'/'+ 'emb'+str(config['emb_dim']) + '_itT'+str(config['emb_iterT'])

    hp = HyperParameters(
                        emb_dim          = config['emb_dim'],
                        node_dim         = modules.gnn.nfm_gen.nfm_funcs[config['nfm_func']].F,
                        hidden_size      = config['lstm_hdim'],
                        recurrent_layers = config['lstm_layers'],
                        batch_size       = config['batch_size'], 
                        learning_rate    = 3e-4,
                        recurrent_seq_len= 2,
                        parallel_rollouts= 4, 
                        rollout_steps    = 200, 
                        patience         = 500
                        )

    FORCE_CPU_GATHER=True
    tp= {"world_name":                 args.train_on,
        "force_cpu_gather":            FORCE_CPU_GATHER, # Force using CPU for gathering trajectories.
        "gather_device":               "cuda" if torch.cuda.is_available() and not FORCE_CPU_GATHER else "cpu", 
        "train_device":                "cuda" if torch.cuda.is_available() else "cpu", 
        "asynchronous_environment":    False,  # Step env asynchronously using multiprocess or synchronously.
        "base_checkpoint_path":        f"{config['logdir']}/checkpoints/",
        "workspace_path":              config['logdir'],
        "invalid_tag_characters":      re.compile(r"[^-/\w\.]"), 
        'save_metrics_tensorboard':    True,
        'save_parameters_tensorboard': False,
        'checkpoint_frequency':        100,
        'eval_deterministic':          True}

    batch_count = hp.parallel_rollouts * hp.rollout_steps / hp.recurrent_seq_len / hp.batch_size
    print(f"batch_count: {batch_count}")
    assert batch_count >= 1., "Less than 1 batch per trajectory.  Are you sure that's what you want?"  
    return config, hp, tp

def main(args):
    config, hp, tp = GetConfigs(args)
    if config['train']:
        for seed in config['seedrange']:
            seed_everything(seed)
            logdir_=config['logdir']+'/SEED'+str(seed)
            tp['writer'] = SummaryWriter(log_dir=f"{logdir_}/logs")

            env = make_custom(config, hp.parallel_rollouts, asynchronous=tp['asynchronous_environment'])
            hp.max_possible_nodes = env.envs[0].env.max_nodes
            hp.max_possible_edges = env.envs[0].env.max_edges

            ppo_model, ppo_optimizer, iteration, stop_conditions = start_or_resume_from_checkpoint(env, hp, tp)
            score = train_model(env, ppo_model, ppo_optimizer, iteration, stop_conditions, hp, tp)

    if config['eval']:
        EvaluateSavedModel()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('--train_on', default='None', type=str)
    parser.add_argument('--batch_size', default=48, type=int)
    parser.add_argument('--mask_type', default='None', type=str, help='U obervation masking type', choices=['None','freq','prob'])
    parser.add_argument('--mask_rate', default=1.0, type=float)
    parser.add_argument('--emb_dim', default=64, type=int)
    parser.add_argument('--lstm_type', default='shared-noncat', type=str, choices=['None','shared-concat','shared-noncat','separate-concat','separate-noncat'])
    parser.add_argument('--lstm_hdim', default=64, type=int)
    parser.add_argument('--lstm_layers', default=1, type=int)
    #parser.add_argument('--lstm_dropout', default=0.0, type=float)
    parser.add_argument('--emb_iterT', default=2, type=int)
    parser.add_argument('--nfm_func', default='NFM_ev_ec_t', type=str)
    parser.add_argument('--qnet', default='gat2', type=str)
    parser.add_argument('--train', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
    parser.add_argument('--eval', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])   
    parser.add_argument('--test', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])       
    parser.add_argument('--num_seeds', default=5, type=int)
    parser.add_argument('--seed0', default=10, type=int)
    parser.add_argument('--demoruns', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
    args=parser.parse_args()
    main(args)