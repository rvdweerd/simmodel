from logging import critical
import torch
import gym
#from torch.utils.tensorboard import SummaryWriter
import copy
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
from collections.abc import Iterable
from gym.vector.async_vector_env import AsyncVectorEnv
from gym.vector.sync_vector_env import SyncVectorEnv
#from gym.vector.vector_env import VectorEnv, VectorEnvWrapper
from dataclasses import dataclass, fields
#import gc
from dotmap import DotMap
#from base64 import b64encode
# from modules.rl.rl_custom_worlds import GetCustomWorld
# from modules.ppo.helpfuncs import get_super_env
# from modules.sim.graph_factory import GetWorldSet
# import argparse
import modules.gnn.nfm_gen
# from modules.rl.environments import SuperEnv
# from modules.dqn.dqn_utils import seed_everything
# from modules.sim.simdata_utils import SimulateAutomaticMode_PPO
# from modules.rl.rl_utils import GetFullCoverageSample
from modules.gnn.construct_trainsets import ConstructTrainSet
from modules.ppo.models_ngo import MaskablePPOPolicy, MaskablePPOPolicy_CONCAT, MaskablePPOPolicy_EMB_LSTM, MaskablePPOPolicy_FE_LSTM
# This code is based on 
# https://gitlab.com/ngoodger/ppo_lstm/-/blob/master/recurrent_ppo.ipynb
# Heavily adapted to work with customized environment and GNNs (trainable with varying graph sizes in the trainset)

@dataclass
class HyperParameters():
    max_possible_nodes:   int   = -1
    max_possible_edges:   int   = -1
    emb_dim:              int   = -1
    critic:               str   = 'q'              
    node_dim:             int   = -1
    lstm_on:              bool  = True
    hidden_size:          float = -1.
    recurrent_layers:     int   = -1
    batch_size:           int   = -1
    min_reward:           float = -1e6
    discount:             float = .99
    gae_lambda:           float = .95
    ppo_clip:             float = .2
    ppo_epochs:           int   = 10
    scale_reward:         float = 1.
    max_grad_norm:        float = .5
    entropy_factor:       float = 0.
    learning_rate:        float = -1.
    recurrent_seq_len:    int   = -1
    parallel_rollouts:    int   = -1
    rollout_steps:        int   = -1
    patience:             int   = -1
    trainable_std_dev:    bool  = False
    init_log_std_dev:     float = 0.
    env_mask_velocity:    bool  = False

@dataclass
class StopConditions():
    """
    Store parameters and variables used to stop training. 
    """
    best_reward: float = -1e6
    iteration: int = 0
    fail_to_improve_count: int = 0
    max_iterations: int = 1000000

@dataclass
class TrajectorBatch():
    """
    Dataclass for storing data batch.
    """
    states: torch.tensor
    selector: torch.tensor
    actions: torch.tensor
    action_probabilities: torch.tensor
    advantages: torch.tensor
    discounted_returns: torch.tensor
    batch_size: torch.tensor
    actor_hidden_states: torch.tensor
    actor_cell_states: torch.tensor
    critic_hidden_states: torch.tensor
    critic_cell_states: torch.tensor

class TrajectoryDataset():
    """
    Fast dataset for producing training batches from trajectories.
    """
    def __init__(self, trajectories, device, hp):
        
        # Combine multiple trajectories into
        self.trajectories = {key: value.to(device) for key, value in trajectories.items()}
        self.batch_len = hp.recurrent_seq_len
        truncated_seq_len = torch.clamp(trajectories["seq_len"] - self.batch_len + 1, 0, hp.rollout_steps)
        self.cumsum_seq_len =  np.cumsum(np.concatenate( (np.array([0]), truncated_seq_len.numpy())))
        self.batch_size = hp.batch_size
        
    def __iter__(self):
        self.valid_idx = np.arange(self.cumsum_seq_len[-1])
        self.batch_count = 0
        return self
        
    def __next__(self):
        if self.batch_count * self.batch_size >= math.ceil(self.cumsum_seq_len[-1] / self.batch_len):
            raise StopIteration
        else:
            actual_batch_size = min(len(self.valid_idx), self.batch_size) 
            start_idx = np.random.choice(self.valid_idx, size=actual_batch_size, replace=False )
            self.valid_idx = np.setdiff1d(self.valid_idx, start_idx)
            eps_idx = np.digitize(start_idx, bins = self.cumsum_seq_len, right=False) - 1
            seq_idx = start_idx - self.cumsum_seq_len[eps_idx]
            series_idx = np.linspace(seq_idx, seq_idx + self.batch_len - 1, num=self.batch_len, dtype=np.int64)
            self.batch_count += 1

            return TrajectorBatch(**{key: value[eps_idx, series_idx]for key, value
                                     in self.trajectories.items() if key in TrajectorBatch.__dataclass_fields__.keys()},
                                  batch_size=actual_batch_size)

@dataclass
class TrajectorBatch2():
    """
    Dataclass for storing data batch.
    """
    states: torch.tensor
    selector: torch.tensor
    actions: torch.tensor
    action_probabilities: torch.tensor
    advantages: torch.tensor
    discounted_returns: torch.tensor
    batch_size: torch.tensor
    FE_hidden_states: torch.tensor
    FE_cell_states: torch.tensor

class TrajectoryDataset2():
    """
    Fast dataset for producing training batches from trajectories.
    """
    def __init__(self, trajectories, device, hp):
        
        # Combine multiple trajectories into
        self.trajectories = {key: value.to(device) for key, value in trajectories.items()}
        self.batch_len = hp.recurrent_seq_len
        truncated_seq_len = torch.clamp(trajectories["seq_len"] - self.batch_len + 1, 0, hp.rollout_steps)
        self.cumsum_seq_len =  np.cumsum(np.concatenate( (np.array([0]), truncated_seq_len.numpy())))
        self.batch_size = hp.batch_size
        
    def __iter__(self):
        self.valid_idx = np.arange(self.cumsum_seq_len[-1])
        self.batch_count = 0
        return self
        
    def __next__(self):
        if self.batch_count * self.batch_size >= math.ceil(self.cumsum_seq_len[-1] / self.batch_len):
            raise StopIteration
        else:
            actual_batch_size = min(len(self.valid_idx), self.batch_size) 
            start_idx = np.random.choice(self.valid_idx, size=actual_batch_size, replace=False )
            self.valid_idx = np.setdiff1d(self.valid_idx, start_idx)
            eps_idx = np.digitize(start_idx, bins = self.cumsum_seq_len, right=False) - 1
            seq_idx = start_idx - self.cumsum_seq_len[eps_idx]
            series_idx = np.linspace(seq_idx, seq_idx + self.batch_len - 1, num=self.batch_len, dtype=np.int64)
            self.batch_count += 1

            return TrajectorBatch2(**{key: value[eps_idx, series_idx]for key, value
                                     in self.trajectories.items() if key in TrajectorBatch2.__dataclass_fields__.keys()},
                                  batch_size=actual_batch_size)

@dataclass
class TrajectorBatch_EMB():
    """
    Dataclass for storing data batch.
    """
    states: torch.tensor
    selector: torch.tensor
    actions: torch.tensor
    action_probabilities: torch.tensor
    advantages: torch.tensor
    discounted_returns: torch.tensor
    batch_size: torch.tensor
    EMB_hidden_states: torch.tensor
    EMB_cell_states: torch.tensor

class TrajectoryDataset_EMB():
    """
    Fast dataset for producing training batches from trajectories.
    """
    def __init__(self, trajectories, device, hp):
        
        # Combine multiple trajectories into
        self.trajectories = {key: value.to(device) for key, value in trajectories.items()}
        self.batch_len = hp.recurrent_seq_len
        truncated_seq_len = torch.clamp(trajectories["seq_len"] - self.batch_len + 1, 0, hp.rollout_steps)
        self.cumsum_seq_len =  np.cumsum(np.concatenate( (np.array([0]), truncated_seq_len.numpy())))
        self.batch_size = hp.batch_size
        
    def __iter__(self):
        self.valid_idx = np.arange(self.cumsum_seq_len[-1])
        self.batch_count = 0
        return self
        
    def __next__(self):
        if self.batch_count * self.batch_size >= math.ceil(self.cumsum_seq_len[-1] / self.batch_len):
            raise StopIteration
        else:
            actual_batch_size = min(len(self.valid_idx), self.batch_size) 
            start_idx = np.random.choice(self.valid_idx, size=actual_batch_size, replace=False )
            self.valid_idx = np.setdiff1d(self.valid_idx, start_idx)
            eps_idx = np.digitize(start_idx, bins = self.cumsum_seq_len, right=False) - 1
            seq_idx = start_idx - self.cumsum_seq_len[eps_idx]
            series_idx = np.linspace(seq_idx, seq_idx + self.batch_len - 1, num=self.batch_len, dtype=np.int64)
            self.batch_count += 1

            return TrajectorBatch_EMB(**{key: value[eps_idx, series_idx]for key, value
                                     in self.trajectories.items() if key in TrajectorBatch_EMB.__dataclass_fields__.keys()},
                                  batch_size=actual_batch_size)



def GetConfigs(args):
    config = {}
    #for i in range(args.num_configs):
    #    config.add(args.config_class(**args.config_args))
    #return config
    #config['lstm_dropout']    = args.lstm_dropout
    config['train_on']        = args.train_on
    config['batch_size']      = args.batch_size
    config['obs_mask']        = str(args.obs_mask)
    config['obs_rate']        = args.obs_rate
    config['emb_dim']         = args.emb_dim
    config['lstm_type']       = args.lstm_type
    config['lstm_hdim']       = args.lstm_hdim
    config['lstm_layers']     = args.lstm_layers
    config['recurrent_seq_len'] = args.recurrent_seq_len
    config['emb_iterT']       = args.emb_iterT
    config['nfm_func']        = args.nfm_func
    config['edge_blocking']   = True
    config['solve_select']    = 'solvable'
    config['qnet']            = args.qnet
    config['critic']          = args.critic
    config['train']           = args.train
    config['eval']            = args.eval
    config['test']            = args.test
    config['num_seeds']       = args.num_seeds
    config['seed0']           = args.seed0
    config['seedrange']=range(config['seed0'], config['seed0']+config['num_seeds'])
    config['demoruns']        = args.demoruns
    lstm_filestring = config['lstm_type']
    if config['lstm_type'] != 'None':
        lstm_filestring  += '_' + str(config['lstm_hdim']) + '_' + str(config['lstm_layers'])
    if config['recurrent_seq_len'] != 2:
        lstm_filestring += '_seqlen' + str(config['recurrent_seq_len'])
    mask_filestring = config['obs_mask']
    if config['obs_mask'] != 'None':
        mask_filestring += str(config['obs_rate'])
    config['rootdir'] = './results/results_Phase3/ppo/'+ config['train_on']+'/'+ \
                        config['qnet'] + '-' + config['critic'] + '/'+ \
                        'emb'+str(config['emb_dim']) + '_itT'+str(config['emb_iterT']) + '/'+ \
                        'lstm_' + lstm_filestring 
                        
    config['logdir']  = config['rootdir'] + '/' + config['nfm_func']+'/' \
                        'omask_' + mask_filestring + '/' +\
                        'bsize' + str(config['batch_size']) #+'_lr{:.1e}'.format(args.lr)
                        

    hp = HyperParameters(
                        emb_dim          = config['emb_dim'],
                        critic           = config['critic'],
                        node_dim         = modules.gnn.nfm_gen.nfm_funcs[config['nfm_func']].F,
                        lstm_on          = config['lstm_type'] != 'None',
                        hidden_size      = config['lstm_hdim'],
                        recurrent_layers = config['lstm_layers'],
                        batch_size       = config['batch_size'], 
                        learning_rate    = args.lr,#3e-4,
                        recurrent_seq_len= args.recurrent_seq_len,#2,
                        parallel_rollouts= args.parallel_rollouts,#4, 
                        rollout_steps    = args.rollout_steps, #200 
                        patience         = args.patience, #500
                        )

    FORCE_CPU_GATHER=True
    tp= {"world_name":                 args.train_on,
        "force_cpu_gather":            FORCE_CPU_GATHER, # Force using CPU for gathering trajectories.
        "gather_device":               "cuda" if torch.cuda.is_available() and not FORCE_CPU_GATHER else "cpu", 
        "train_device":                "cuda" if torch.cuda.is_available() else "cpu", 
        "asynchronous_environment":    False,  # Step env asynchronously using multiprocess or synchronously.
        "invalid_tag_characters":      re.compile(r"[^-/\w\.]"), 
        'save_metrics_tensorboard':    True,
        'save_parameters_tensorboard': False,
        'checkpoint_frequency':        100,
        'eval_deterministic':          args.eval_deter}

    batch_count = hp.parallel_rollouts * hp.rollout_steps / hp.recurrent_seq_len / hp.batch_size
    print(f"batch_count: {batch_count}")
    assert batch_count >= 1., "Less than 1 batch per trajectory.  Are you sure that's what you want?"  
    return config, hp, tp

def WriteTrainParamsToFile(config,hp,tp):
    if not os.path.exists(config['logdir']):
        os.makedirs(config['logdir'])    
    OF = open(config['logdir']+'/train-parameters.txt', 'w')
    def printing(text):
        print(text)
        OF.write(text + "\n")
    np.set_printoptions(formatter={'float':"{0:0.3f}".format})
    
    printing('config args:\n-----------------')
    for k,v in config.items():
        printing(k + ': ' + str(v))
    printing('\nhyperparameters:\n-----------------')
    for field in fields(hp):
        printing(field.name + ': ' + str(getattr(hp, field.name)))
    printing('\ntrain parameters:\n-----------------')
    for k,v in tp.items():
        printing(k + ': ' + str(v))

def WriteModelParamsToFile(config,model):
    if not os.path.exists(config['logdir']):
        os.makedirs(config['logdir'])    
    OF = open(config['logdir']+'/model-parameters.txt', 'w')
    def printing(text):
        print(text)
        OF.write(text + "\n")
    np.set_printoptions(formatter={'float':"{0:0.3f}".format})
    printing('\nModel layout:\n-----------------')
    printing(str(model))
    printing('\nNumber of trainable parameters:\n-----------------')
    num, parameter_string = model.numTrainableParameters()
    printing(parameter_string)

def save_parameters(writer, tag, model, batch_idx, tp):
    """
    Save model parameters for tensorboard.
    """
    for k, v in model.state_dict().items():
        shape = v.shape
        # Fix shape definition for tensorboard.
        shape_formatted = tp['invalid_tag_characters'].sub("_", str(shape))
        # Don't do this for single weights or biases
        if np.any(np.array(shape) > 1):
            mean = torch.mean(v)
            std_dev = torch.std(v)
            maximum = torch.max(v)
            minimum = torch.min(v)
            writer.add_scalars(
                "{}_weights/{}{}".format(tag, k, shape_formatted),
                {"mean": mean, "std_dev": std_dev, "max": maximum, "min": minimum},
                batch_idx,
            )
        else:
            writer.add_scalar("{}_{}{}".format(tag, k, shape_formatted), v.data, batch_idx)
            
def get_env_space(env):
    """
    Return obsvervation dimensions, action dimensions and whether or not action space is continuous.
    """
    #env = gym.make(ENV)
    #env=GetCustomWorld(WORLD_NAME, make_reflexive=MAKE_REFLEXIVE, state_repr=STATE_REPR, state_enc='tensors')
    aspace = env.action_space[0]
    continuous_action_space = type(aspace) is gym.spaces.box.Box
    if continuous_action_space:
        action_dim =  aspace.shape[0]
    else:
        action_dim = aspace.n 
    obsv_dim= env.observation_space.shape[1] 
    return obsv_dim, action_dim, continuous_action_space

def get_last_checkpoint_iteration(tp):
    """
    Determine latest checkpoint iteration.
    """
    if os.path.isdir(tp['base_checkpoint_path']):
        max_checkpoint_iteration = max([int(dirname) for dirname in os.listdir(tp['base_checkpoint_path'])])
    else:
        max_checkpoint_iteration = 0
    return max_checkpoint_iteration

def save_checkpoint(ppo_model, ppo_optimizer, iteration, stop_conditions, hp, tp):
    """
    Save training checkpoint.
    """
    checkpoint = DotMap()
    checkpoint.world_name = tp['world_name']
    checkpoint.env_mask_velocity = hp.env_mask_velocity
    checkpoint.iteration = iteration
    checkpoint.stop_conditions = stop_conditions
    checkpoint.hp = hp
    CHECKPOINT_PATH = tp['base_checkpoint_path'] + f"{iteration}/"
    pathlib.Path(CHECKPOINT_PATH).mkdir(parents=True, exist_ok=True) 
    with open(CHECKPOINT_PATH + "parameters.pt", "wb") as f:
        pickle.dump(checkpoint, f)
    with open(CHECKPOINT_PATH + "ppo_model_class.pt", "wb") as f:
        pickle.dump(MaskablePPOPolicy, f)
    #with open(CHECKPOINT_PATH + "critic_class.pt", "wb") as f:
    #    pickle.dump(Critic, f)
    torch.save(ppo_model.state_dict(), CHECKPOINT_PATH + "ppo_model.pt")
    #torch.save(critic.state_dict(), CHECKPOINT_PATH + "critic.pt")
    torch.save(ppo_optimizer.state_dict(), CHECKPOINT_PATH + "ppo_optimizer.pt")
    #torch.save(critic_optimizer.state_dict(), CHECKPOINT_PATH + "critic_optimizer.pt")
    OF_ = open(tp['seed_path']+'/model_best_save_history.txt', 'a')
    OF_.write('iteration:'+str(stop_conditions.iteration)+', avg det res:'+str(stop_conditions.best_reward)+'\n')
    OF_.close()

def start_or_resume_from_checkpoint(env, config, hp, tp):
    """
    Create actor, critic, actor optimizer and critic optimizer from scratch
    or load from latest checkpoint if it exists. 
    """
    max_checkpoint_iteration = get_last_checkpoint_iteration(tp)
    
    obsv_dim, action_dim, continuous_action_space = get_env_space(env)
#['None','shared-concat','shared-noncat','separate-concat','separate-noncat']
    if config['qnet']=='s2v': assert False, "s2v not implemented"
    if config['lstm_type'] in ['None','Dual']:	
        ppo_algo = MaskablePPOPolicy
    elif config['lstm_type'] == 'DualCC':
        ppo_algo = MaskablePPOPolicy_CONCAT
    elif config['lstm_type'] == 'EMB':
        ppo_algo = MaskablePPOPolicy_EMB_LSTM
    elif config['lstm_type'] == 'FE':
        ppo_algo = MaskablePPOPolicy_FE_LSTM
    else:
        assert False

    ppo_model = ppo_algo(obsv_dim,
                  action_dim,
                  continuous_action_space=continuous_action_space,
                  trainable_std_dev=hp.trainable_std_dev,
                  init_log_std_dev=hp.init_log_std_dev,
                  hp=hp)
        
    ppo_optimizer = optim.AdamW(ppo_model.parameters(), lr=hp.learning_rate)

    stop_conditions = StopConditions()
        
    # If max checkpoint iteration is greater than zero initialise training with the checkpoint.
    if max_checkpoint_iteration > 0:
        ppo_model_state_dict, ppo_optimizer_state_dict, stop_conditions = load_checkpoint(max_checkpoint_iteration, hp, tp)
        
        ppo_model.load_state_dict(ppo_model_state_dict, strict=True)
        ppo_optimizer.load_state_dict(ppo_optimizer_state_dict)#, strict=True)

        # We have to move manually move optimizer states to TRAIN_DEVICE manually since optimizer doesn't yet have a "to" method.
        for state in ppo_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                   state[k] = v.to(tp['train_device'])

    return ppo_model, ppo_optimizer, max_checkpoint_iteration, stop_conditions
    
def load_checkpoint(iteration, hp, tp):
    """
    Load from training checkpoint.
    """
    CHECKPOINT_PATH = tp['base_checkpoint_path'] + f"{iteration}/"
    with open(CHECKPOINT_PATH + "parameters.pt", "rb") as f:
        checkpoint = pickle.load(f)
        
    assert tp['world_name'] == checkpoint.world_name, "To resume training environment must match current settings."
    assert hp.env_mask_velocity== checkpoint.env_mask_velocity
    #REVERT assert hp == checkpoint.hp, "To resume training hyperparameters must match current settings."
    print(hp)
    print(checkpoint.hp)
    ppo_model_state_dict = torch.load(CHECKPOINT_PATH + "ppo_model.pt", map_location=torch.device(tp['train_device']))
    ppo_optimizer_state_dict = torch.load(CHECKPOINT_PATH + "ppo_optimizer.pt", map_location=torch.device(tp['train_device']))
    
    return (ppo_model_state_dict, ppo_optimizer_state_dict, checkpoint.stop_conditions)
     
def calc_discounted_return(rewards, discount, final_value):
    """
    Calculate discounted returns based on rewards and discount factor.
    """
    seq_len = len(rewards)
    discounted_returns = torch.zeros(seq_len)
    discounted_returns[-1] = rewards[-1] + discount * final_value
    for i in range(seq_len - 2, -1 , -1):
        discounted_returns[i] = rewards[i] + discount * discounted_returns[i + 1]
    return discounted_returns

def compute_advantages(rewards, values, discount, gae_lambda):
    """
    Compute General Advantage.
    """
    deltas = rewards + discount * values[1:] - values[:-1]
    seq_len = len(rewards)
    advs = torch.zeros(seq_len + 1)
    multiplier = discount * gae_lambda
    for i in range(seq_len - 1, -1 , -1):
        advs[i] = advs[i + 1] * multiplier  + deltas[i]
    return advs[:-1]

def gather_trajectories(input_data,hp):
    """
    Gather policy trajectories from gym environment.
    """
    gather_device = input_data['gather_device']
    _min_reward_values = torch.full([hp.parallel_rollouts], hp.min_reward)
    # Unpack inputs.
    env = input_data["env"]
    ppo_model = input_data["ppo_model"]
    #critic = input_data["critic"]
    
    # Initialise variables.
    obsv = env.reset()
    trajectory_data = {"states": [],
                 "selector": [],
                 "actions": [],
                 "action_probabilities": [],
                 "rewards": [],
                 "true_rewards": [],
                 "values": [],
                 "terminals": [],
                 "actor_hidden_states": [],
                 "actor_cell_states": [],
                 "critic_hidden_states": [],
                 "critic_cell_states": []}
    terminal = torch.ones(hp.parallel_rollouts) 

    with torch.no_grad():
        # Reset actor and critic state.
        first_pass = True
        bsize=obsv.shape[0]
        # Take 1 additional step in order to collect the state and value for the final state.
        for i in range(hp.rollout_steps):
            state = torch.tensor(obsv, dtype=torch.float32)
            features, nodes_in_batch, valid_entries_idx, num_nodes = ppo_model.FE(state.unsqueeze(0).to(gather_device))
            selector=[]
            for j in range(bsize):
                selector+= [True]*int(num_nodes[j])+[False]*int(hp.max_possible_nodes-num_nodes[j])
            selector=torch.tensor(selector,dtype=torch.float32).reshape(bsize,-1)
            
            trajectory_data["states"].append(state.clone())
            trajectory_data["selector"].append(selector.clone())
            batch_size=state.shape[0]

            if first_pass: # initialize hidden states
                ppo_model.PI.reset_init_state(batch_size*hp.max_possible_nodes, gather_device)
                ppo_model.V.reset_init_state(batch_size*hp.max_possible_nodes, gather_device)
                first_pass = False
            else: # reset hidden states of terminated states to zero
                terminal_dense = torch.repeat_interleave(terminal.to(torch.bool), torch.ones(batch_size,dtype=torch.int64)*hp.max_possible_nodes, dim=0)
                ppo_model.PI.hidden_cell[0][:,terminal_dense,:] = 0.
                ppo_model.PI.hidden_cell[1][:,terminal_dense,:] = 0.
                ppo_model.V.hidden_cell[0][:,terminal_dense,:] = 0.
                ppo_model.V.hidden_cell[1][:,terminal_dense,:] = 0.

            trajectory_data["actor_hidden_states"].append( ppo_model.PI.hidden_cell[0].clone().squeeze(0).reshape(batch_size,-1).cpu() ) # (6,:)
            trajectory_data["actor_cell_states"].append(   ppo_model.PI.hidden_cell[1].clone().squeeze(0).reshape(batch_size,-1).cpu() )
            trajectory_data["critic_hidden_states"].append(ppo_model.V.hidden_cell[0].clone().squeeze(0).reshape(batch_size,-1).cpu() )
            trajectory_data["critic_cell_states"].append(  ppo_model.V.hidden_cell[1].clone().squeeze(0).reshape(batch_size,-1).cpu() )
            
            # Choose next action 
            action_dist, value = ppo_model(features, terminal.to(gather_device), selector=selector.flatten().to(torch.bool))
            action = action_dist.sample().reshape(hp.parallel_rollouts, -1)
            if not ppo_model.continuous_action_space:
                action = action.squeeze(1)
            trajectory_data["actions"].append(action.clone().cpu())
            trajectory_data["action_probabilities"].append(action_dist.log_prob(action).squeeze(0).clone().cpu())
            #value = ppo_model.V(features, terminal.to(gather_device), selector=selector.flatten().to(torch.bool))
            trajectory_data["values"].append( value.clone().reshape(-1).cpu())

            # Step environment 
            action_np = action.cpu().numpy()
            obsv, reward, done, _ = env.step(action_np)
            reward=copy.deepcopy(reward)
            terminal = torch.tensor(done).float()
            transformed_reward = hp.scale_reward * torch.max(_min_reward_values, torch.tensor(reward).float())
                                                             
            trajectory_data["rewards"].append(transformed_reward.clone().cpu())
            trajectory_data["true_rewards"].append(torch.tensor(reward).float())
            trajectory_data["terminals"].append(terminal.clone().cpu())
    
        # Compute final value to allow for incomplete episodes.
        state = torch.tensor(obsv, dtype=torch.float32)
        features, nodes_in_batch, valid_entries_idx, num_nodes = ppo_model.FE(state.unsqueeze(0).to(gather_device))        
        terminal_dense = torch.repeat_interleave(terminal.to(torch.bool), torch.ones(batch_size,dtype=torch.int64)*hp.max_possible_nodes, dim=0)
        ppo_model.PI.hidden_cell[0][:,terminal_dense,:] = 0.
        ppo_model.PI.hidden_cell[1][:,terminal_dense,:] = 0.
        ppo_model.V.hidden_cell[0][:,terminal_dense,:] = 0.
        ppo_model.V.hidden_cell[1][:,terminal_dense,:] = 0.

        value = ppo_model.get_values(features.to(gather_device), terminal.to(gather_device))
        # Future value for terminal episodes is 0.
        #trajectory_data["values"].append(value.squeeze().cpu() * (1 - terminal))
        trajectory_data["values"].append((value.reshape(-1).cpu() * (1 - terminal)).clone())

    # Combine step lists into tensors.
    trajectory_tensors = {key: torch.stack(value) for key, value in trajectory_data.items()}
    return trajectory_tensors

def gather_trajectories_FE(input_data,hp):
    """
    Gather policy trajectories from gym environment.
    """
    gather_device = input_data['gather_device']
    _min_reward_values = torch.full([hp.parallel_rollouts], hp.min_reward)
    # Unpack inputs.
    env = input_data["env"]
    ppo_model = input_data["ppo_model"]
    #critic = input_data["critic"]
    
    # Initialise variables.
    obsv = env.reset()
    trajectory_data = {"states": [],
                 "selector": [],
                 "actions": [],
                 "action_probabilities": [],
                 "rewards": [],
                 "true_rewards": [],
                 "values": [],
                 "terminals": [],
                 "FE_hidden_states": [],
                 "FE_cell_states": [],
                 }
    terminal = torch.ones(hp.parallel_rollouts) 

    with torch.no_grad():
        # Reset actor and critic state.
        first_pass = True
        bsize=obsv.shape[0]
        # Take 1 additional step in order to collect the state and value for the final state.
        for i in range(hp.rollout_steps):
            state = torch.tensor(obsv, dtype=torch.float32)
            trajectory_data["states"].append(state.clone())
            
            if first_pass: # initialize hidden states
                ppo_model.FE.reset_init_state(bsize*hp.max_possible_nodes, gather_device)
                first_pass = False
            else: # reset hidden states of terminated states to zero
                terminal_dense = torch.repeat_interleave(terminal.to(torch.bool), torch.ones(bsize,dtype=torch.int64)*hp.max_possible_nodes, dim=0)
                ppo_model.FE.hidden_cell[0][:,terminal_dense,:] = 0.
                ppo_model.FE.hidden_cell[1][:,terminal_dense,:] = 0.

            # Hidden cells appended before we run the lstm                
            trajectory_data["FE_hidden_states"].append( ppo_model.FE.hidden_cell[0].clone().squeeze(0).reshape(bsize,-1).cpu() ) # (6,:)
            trajectory_data["FE_cell_states"].append(   ppo_model.FE.hidden_cell[1].clone().squeeze(0).reshape(bsize,-1).cpu() )

            features, nodes_in_batch, valid_entries_idx, num_nodes = ppo_model.FE(state.unsqueeze(0).to(gather_device))
            
            selector=[]
            for j in range(bsize):
                selector+= [True]*int(num_nodes[j])+[False]*int(hp.max_possible_nodes-num_nodes[j])
            selector=torch.tensor(selector,dtype=torch.float32).reshape(bsize,-1)
            trajectory_data["selector"].append(selector.clone())
            batch_size=state.shape[0]

            
            # Choose next action 
            action_dist, value = ppo_model(features, terminal.to(gather_device), selector=selector.flatten().to(torch.bool))
            action = action_dist.sample().reshape(hp.parallel_rollouts, -1)
            if not ppo_model.continuous_action_space:
                action = action.squeeze(1)
            trajectory_data["actions"].append(action.clone().cpu())
            trajectory_data["action_probabilities"].append(action_dist.log_prob(action).squeeze(0).clone().cpu())
            #value = ppo_model.V(features, terminal.to(gather_device), selector=selector.flatten().to(torch.bool))
            #trajectory_data["values"].append( value.squeeze().cpu())
            trajectory_data["values"].append( value.clone().reshape(-1).cpu())

            # Step environment 
            action_np = action.cpu().numpy()
            obsv, reward, done, _ = env.step(action_np)
            reward=copy.deepcopy(reward)
            terminal = torch.tensor(done).float()
            transformed_reward = hp.scale_reward * torch.max(_min_reward_values, torch.tensor(reward).float())
                                                             
            trajectory_data["rewards"].append(transformed_reward.clone().cpu())
            trajectory_data["true_rewards"].append(torch.tensor(reward).float())
            trajectory_data["terminals"].append(terminal.clone().cpu())
    
        # Compute final value to allow for incomplete episodes.
        state = torch.tensor(obsv, dtype=torch.float32)
        terminal_dense = torch.repeat_interleave(terminal.to(torch.bool), torch.ones(bsize,dtype=torch.int64)*hp.max_possible_nodes, dim=0)
        ppo_model.FE.hidden_cell[0][:,terminal_dense,:] = 0.
        ppo_model.FE.hidden_cell[1][:,terminal_dense,:] = 0.
        features, nodes_in_batch, valid_entries_idx, num_nodes = ppo_model.FE(state.unsqueeze(0).to(gather_device))        
        
        value = ppo_model.get_values(features.to(gather_device), terminal.to(gather_device))
        # Future value for terminal episodes is 0.
        #trajectory_data["values"].append(value.squeeze().cpu() * (1 - terminal))
        trajectory_data["values"].append((value.reshape(-1).cpu() * (1 - terminal)).clone())

    # Combine step lists into tensors.
    trajectory_tensors = {key: torch.stack(value) for key, value in trajectory_data.items()}
    return trajectory_tensors

def gather_trajectories_EMB(input_data,hp):
    """
    Gather policy trajectories from gym environment.
    """
    gather_device = input_data['gather_device']
    _min_reward_values = torch.full([hp.parallel_rollouts], hp.min_reward)
    # Unpack inputs.
    env = input_data["env"]
    ppo_model = input_data["ppo_model"]
    #critic = input_data["critic"]
    
    # Initialise variables.
    obsv = env.reset()
    trajectory_data = {"states": [],
                 "selector": [],
                 "actions": [],
                 "action_probabilities": [],
                 "rewards": [],
                 "true_rewards": [],
                 "values": [],
                 "terminals": [],
                 "EMB_hidden_states":[],
                 "EMB_cell_states":[],
    }
    terminal = torch.ones(hp.parallel_rollouts) 

    with torch.no_grad():
        # Reset actor and critic state.
        first_pass = True
        bsize=obsv.shape[0]
        # Take 1 additional step in order to collect the state and value for the final state.
        for i in range(hp.rollout_steps):
            state = torch.tensor(obsv, dtype=torch.float32)
            features, nodes_in_batch, valid_entries_idx, num_nodes = ppo_model.FE(state.unsqueeze(0).to(gather_device))
            selector=[]
            for j in range(bsize):
                selector+= [True]*int(num_nodes[j])+[False]*int(hp.max_possible_nodes-num_nodes[j])
            selector=torch.tensor(selector,dtype=torch.float32).reshape(bsize,-1)
            
            trajectory_data["states"].append(state.clone())
            trajectory_data["selector"].append(selector.clone())
            batch_size=state.shape[0]

            if first_pass: # initialize hidden states
                ppo_model.LSTM.reset_init_state(batch_size*hp.max_possible_nodes, gather_device)
                #ppo_model.PI.reset_init_state(batch_size*hp.max_possible_nodes, gather_device)
                #ppo_model.V.reset_init_state(batch_size*hp.max_possible_nodes, gather_device)
                first_pass = False
            else: # reset hidden states of terminated states to zero
                terminal_dense = torch.repeat_interleave(terminal.to(torch.bool), torch.ones(batch_size,dtype=torch.int64)*hp.max_possible_nodes, dim=0)
                ppo_model.LSTM.hidden_cell[0][:,terminal_dense,:] = 0.
                ppo_model.LSTM.hidden_cell[1][:,terminal_dense,:] = 0.

            trajectory_data["EMB_hidden_states"].append( ppo_model.LSTM.hidden_cell[0].clone().squeeze(0).reshape(batch_size,-1).cpu() ) # (6,:)
            trajectory_data["EMB_cell_states"].append(   ppo_model.LSTM.hidden_cell[1].clone().squeeze(0).reshape(batch_size,-1).cpu() )
            
            # Choose next action 
            features_mem = ppo_model.LSTM(features,terminal.to(gather_device), selector=selector.flatten().to(torch.bool))

            action_dist, value = ppo_model(features_mem, terminal.to(gather_device), selector=selector.flatten().to(torch.bool))
            action = action_dist.sample().reshape(hp.parallel_rollouts, -1)
            if not ppo_model.continuous_action_space:
                action = action.squeeze(1)
            trajectory_data["actions"].append(action.clone().cpu())
            trajectory_data["action_probabilities"].append(action_dist.log_prob(action).squeeze(0).clone().cpu())
            #value = ppo_model.V(features, terminal.to(gather_device), selector=selector.flatten().to(torch.bool))
            trajectory_data["values"].append( value.clone().reshape(-1).cpu())

            # Step environment 
            action_np = action.cpu().numpy()
            obsv, reward, done, _ = env.step(action_np)
            reward=copy.deepcopy(reward)
            terminal = torch.tensor(done).float()
            transformed_reward = hp.scale_reward * torch.max(_min_reward_values, torch.tensor(reward).float())
                                                             
            trajectory_data["rewards"].append(transformed_reward.clone().cpu())
            trajectory_data["true_rewards"].append(torch.tensor(reward).float())
            trajectory_data["terminals"].append(terminal.clone().cpu())
    
        # Compute final value to allow for incomplete episodes.
        state = torch.tensor(obsv, dtype=torch.float32)
        features, nodes_in_batch, valid_entries_idx, num_nodes = ppo_model.FE(state.unsqueeze(0).to(gather_device))       
        selector=[]
        for j in range(bsize):
            selector+= [True]*int(num_nodes[j])+[False]*int(hp.max_possible_nodes-num_nodes[j])
        selector=torch.tensor(selector,dtype=torch.float32).reshape(bsize,-1)
        terminal_dense = torch.repeat_interleave(terminal.to(torch.bool), torch.ones(batch_size,dtype=torch.int64)*hp.max_possible_nodes, dim=0)
        ppo_model.LSTM.hidden_cell[0][:,terminal_dense,:] = 0.
        ppo_model.LSTM.hidden_cell[1][:,terminal_dense,:] = 0.        

        features_mem = ppo_model.LSTM(features,terminal.to(gather_device), selector=selector.flatten().to(torch.bool))
        value = ppo_model.get_values(features_mem.to(gather_device), terminal.to(gather_device))
        # Future value for terminal episodes is 0.
        #trajectory_data["values"].append(value.squeeze().cpu() * (1 - terminal))
        trajectory_data["values"].append((value.reshape(-1).cpu() * (1 - terminal)).clone())

    # Combine step lists into tensors.
    trajectory_tensors = {key: torch.stack(value) for key, value in trajectory_data.items()}
    return trajectory_tensors

def split_trajectories_episodes(trajectory_tensors, hp):
    """
    Split trajectories by episode.
    """
    len_episodes = []
    trajectory_episodes = {key: [] for key in trajectory_tensors.keys()}
    for i in range(hp.parallel_rollouts):
        terminals_tmp = trajectory_tensors["terminals"].clone()
        terminals_tmp=torch.cat((torch.ones(hp.parallel_rollouts)[None,:],terminals_tmp),dim=0)
        terminals_tmp[-1, i] = 1
        split_points = (terminals_tmp[:, i] == 1).nonzero() + 1
        split_lens = split_points[1:] - split_points[:-1]
        
        len_episode = [split_len.item() for split_len in split_lens]
        len_episodes += len_episode
        for key, value in trajectory_tensors.items():
            # Value includes additional step.
            if key == "values":
                value_split = list(torch.split(value[:, i], len_episode[:-1] + [len_episode[-1] + 1]))
                # Append extra 0 to values to represent no future reward.
                for j in range(len(value_split) - 1):
                    value_split[j] = torch.cat((value_split[j], torch.zeros(1)))
                trajectory_episodes[key] += value_split
            else:
                trajectory_episodes[key] += torch.split(value[:, i], len_episode)
    return trajectory_episodes, len_episodes

def pad_and_compute_returns(trajectory_episodes, len_episodes, hp):

    """
    Pad the trajectories up to hp.rollout_steps so they can be combined in a
    single tensor.
    Add advantages and discounted_returns to trajectories.
    """

    episode_count = len(len_episodes)
    #advantages_episodes, discounted_returns_episodes = [], []
    padded_trajectories = {key: [] for key in trajectory_episodes.keys()}
    padded_trajectories["advantages"] = []
    padded_trajectories["discounted_returns"] = []

    for i in range(episode_count):
        single_padding = torch.zeros(hp.rollout_steps - len_episodes[i])
        for key, value in trajectory_episodes.items():
            if value[i].ndim > 1:
                padding = torch.zeros(hp.rollout_steps - len_episodes[i], value[0].shape[1], dtype=value[i].dtype)
            else:
                padding = torch.zeros(hp.rollout_steps - len_episodes[i], dtype=value[i].dtype)
            padded_trajectories[key].append(torch.cat((value[i], padding)))
        padded_trajectories["advantages"].append(torch.cat((compute_advantages(rewards=trajectory_episodes["rewards"][i],
                                                        values = trajectory_episodes["values"][i],
                                                        discount = hp.discount,
                                                        gae_lambda = hp.gae_lambda), single_padding)))
        padded_trajectories["discounted_returns"].append(torch.cat((calc_discounted_return(rewards=trajectory_episodes["rewards"][i],
                                                                    discount = hp.discount,
                                                                    final_value = trajectory_episodes["values"][i][-1]), single_padding)))
    return_val = {k: torch.stack(v) for k, v in padded_trajectories.items()} 
    return_val["seq_len"] = torch.tensor(len_episodes)
    
    return return_val

def train_model(env, ppo_model, ppo_optimizer, iteration, stop_conditions, hp, tp):   
    # Vector environment manages multiple instances of the environment.
    # A key difference between this and the standard gym environment is it automatically resets.
    # Therefore when the done flag is active in the done vector the corresponding state is the first new state.
    # env = gym.vector.make(ENV, hp.parallel_rollouts, asynchronous=ASYNCHRONOUS_ENVIRONMENT)
    # GLOBAL_COUNT=0   
    #ppo_optimizer.param_groups[0]['lr'] *= 0.25
    while iteration < stop_conditions.max_iterations:      
        ppo_model = ppo_model.to(tp['gather_device'])
        start_gather_time = time.time()
        stop_conditions.iteration = iteration

        # Gather trajectories.
        input_data = {"env": env, "ppo_model": ppo_model, "gather_device":tp['gather_device']}
        trajectory_tensors = gather_trajectories(input_data, hp)
        trajectory_episodes, len_episodes = split_trajectories_episodes(trajectory_tensors, hp)
        trajectories = pad_and_compute_returns(trajectory_episodes, len_episodes,  hp)

        # Calculate mean reward.
        complete_episode_count = trajectories["terminals"].sum().item()
        terminal_episodes_rewards = (trajectories["terminals"].sum(axis=1) * trajectories["true_rewards"].sum(axis=1)).sum().item()
        mean_reward =  terminal_episodes_rewards / (complete_episode_count)

        # Check stop conditions.
        if mean_reward > stop_conditions.best_reward:
            stop_conditions.best_reward = mean_reward
            stop_conditions.fail_to_improve_count = 0
            print("NEW BEST MEAN REWARD----------------------<<<<<<<<<<< ")
            print('rec seq len',hp.recurrent_seq_len)
            print('actor lr',ppo_optimizer.param_groups[0]['lr'])

            if iteration>=1:
                print('#################### SAVE CHECKPOINT #######################')
                save_checkpoint(ppo_model, ppo_optimizer, 99999, stop_conditions, hp, tp)
                # best model saved as iteration0
        else:
            stop_conditions.fail_to_improve_count += 1
        if stop_conditions.fail_to_improve_count > hp.patience:
            print(f"Policy has not yielded higher reward for {hp.patience} iterations...  Stopping now.")
            break

        trajectory_dataset = TrajectoryDataset(trajectories, device=tp['train_device'], hp=hp)
        end_gather_time = time.time()
        start_train_time = time.time()
        
        ppo_model = ppo_model.to(tp['train_device'])

        # Train actor and critic. 
        for epoch_idx in range(hp.ppo_epochs): 
            for batch in trajectory_dataset:

                # Prime the LSTMs
                ppo_model.PI.hidden_cell = ( batch.actor_hidden_states[0].reshape(hp.recurrent_layers,-1,hp.hidden_size), 
                                             batch.actor_cell_states[0].reshape(hp.recurrent_layers,-1,hp.hidden_size) 
                                            )
                ppo_model.V.hidden_cell = ( batch.critic_hidden_states[0].reshape(hp.recurrent_layers,-1,hp.hidden_size), 
                                             batch.critic_cell_states[0].reshape(hp.recurrent_layers,-1,hp.hidden_size) 
                                            )
                
                # GLOBAL_COUNT+=1
                # if GLOBAL_COUNT%WARMUP==0:
                #     actor_optimizer.param_groups[0]['lr'] *= 0.5
                #     hp.recurrent_seq_len=3
                #     print('############################################# LR adjusted to',actor_optimizer.param_groups[0]['lr'])
                # Actor loss
                ppo_optimizer.zero_grad()
                features, nodes_in_batch, valid_entries_idx, num_nodes = ppo_model.FE(batch.states)

                selector_bool = batch.selector[0].clone().flatten().to(torch.bool)
                action_dist, values = ppo_model(features, selector=selector_bool)
                # Action dist runs on cpu as a workaround to CUDA illegal memory access.
                action_probabilities = action_dist.log_prob(batch.actions.to("cpu")).to(tp['train_device'])
                
                NORMALIZE_ADVANTAE=True
                if NORMALIZE_ADVANTAE:
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

                # Compute probability ratio from probabilities in logspace.
                probabilities_ratio = torch.exp(action_probabilities - batch.action_probabilities)
                surrogate_loss_0 = probabilities_ratio * batch.advantages
                surrogate_loss_1 =  torch.clamp(probabilities_ratio, 1. - hp.ppo_clip, 1. + hp.ppo_clip) * batch.advantages#[-1, :]
                surrogate_loss_2 = action_dist.entropy().to(tp['train_device'])
                actor_loss = -torch.mean(torch.min(surrogate_loss_0, surrogate_loss_1)) - torch.mean(hp.entropy_factor * surrogate_loss_2)
                
                # Critic loss
                #values = ppo_model.V(features, selector=selector_bool)
                critic_loss = F.mse_loss(batch.discounted_returns, values.squeeze(-1))
                
                vf_coef = 0.5 # basis: paper & sb3 implementation
                loss = actor_loss + vf_coef * critic_loss
                loss.backward()
                torch.nn.utils.clip_grad.clip_grad_norm_(ppo_model.parameters(), hp.max_grad_norm)
                ppo_optimizer.step()

                del loss
                del action_probabilities
                del features
                del action_dist
                del values

        del trajectory_dataset
        torch.cuda.empty_cache()

                
        end_train_time = time.time()
        print(f"Iteration: {iteration},  Mean reward: {mean_reward}, Mean Entropy: {torch.mean(surrogate_loss_2)}, " +
              f"complete_episode_count: {complete_episode_count}, Gather time: {end_gather_time - start_gather_time:.2f}s, " +
              f"Train time: {end_train_time - start_train_time:.2f}s")

        if tp['save_metrics_tensorboard']:
            tp['writer'].add_scalar("complete_episode_count", complete_episode_count, iteration)
            tp['writer'].add_scalar("total_reward", mean_reward , iteration)
            tp['writer'].add_scalar("actor_loss", actor_loss.item(), iteration)
            tp['writer'].add_scalar("critic_loss", critic_loss.item(), iteration)
            tp['writer'].add_scalar("policy_entropy", torch.mean(surrogate_loss_2).item(), iteration)
        if tp['save_parameters_tensorboard']:
            save_parameters(tp['writer'], "ppo_model", ppo_model, iteration, tp)
            #save_parameters(writer, "value", critic, iteration)
        if iteration % tp['checkpoint_frequency'] == 0: 
            print('rec seq len',hp.recurrent_seq_len)
            print('actor lr',ppo_optimizer.param_groups[0]['lr'])
            save_checkpoint(ppo_model, ppo_optimizer, iteration, stop_conditions, hp, tp)
        iteration += 1
        
    return stop_conditions.best_reward 

def train_model_FE(env, ppo_model, ppo_optimizer, iteration, stop_conditions, hp, tp):   
    # Vector environment manages multiple instances of the environment.
    # A key difference between this and the standard gym environment is it automatically resets.
    # Therefore when the done flag is active in the done vector the corresponding state is the first new state.
    # env = gym.vector.make(ENV, hp.parallel_rollouts, asynchronous=ASYNCHRONOUS_ENVIRONMENT)
    # GLOBAL_COUNT=0   
    #ppo_optimizer.param_groups[0]['lr'] *= 0.25
    while iteration < stop_conditions.max_iterations:      
        ppo_model = ppo_model.to(tp['gather_device'])
        start_gather_time = time.time()
        stop_conditions.iteration = iteration

        # Gather trajectories.
        input_data = {"env": env, "ppo_model": ppo_model, "gather_device":tp['gather_device']}
        trajectory_tensors = gather_trajectories_FE(input_data, hp)
        trajectory_episodes, len_episodes = split_trajectories_episodes(trajectory_tensors, hp)
        trajectories = pad_and_compute_returns(trajectory_episodes, len_episodes,  hp)

        # Calculate mean reward.
        complete_episode_count = trajectories["terminals"].sum().item()
        terminal_episodes_rewards = (trajectories["terminals"].sum(axis=1) * trajectories["true_rewards"].sum(axis=1)).sum().item()
        mean_reward =  terminal_episodes_rewards / (complete_episode_count)

        # Check stop conditions.
        if mean_reward > stop_conditions.best_reward:
            stop_conditions.best_reward = mean_reward
            stop_conditions.fail_to_improve_count = 0
            print("NEW BEST MEAN REWARD----------------------<<<<<<<<<<< ")
            print('rec seq len',hp.recurrent_seq_len)
            print('actor lr',ppo_optimizer.param_groups[0]['lr'])

            if iteration>=1:
                print('#################### SAVE CHECKPOINT #######################')
                save_checkpoint(ppo_model, ppo_optimizer, 99999, stop_conditions, hp, tp)
                # best model saved as iteration0
        else:
            stop_conditions.fail_to_improve_count += 1
        if stop_conditions.fail_to_improve_count > hp.patience:
            print(f"Policy has not yielded higher reward for {hp.patience} iterations...  Stopping now.")
            break

        trajectory_dataset = TrajectoryDataset2(trajectories, device=tp['train_device'], hp=hp)
        end_gather_time = time.time()
        start_train_time = time.time()
        
        ppo_model = ppo_model.to(tp['train_device'])

        # Train actor and critic. 
        for epoch_idx in range(hp.ppo_epochs): 
            for batch in trajectory_dataset:

                # Prime the LSTMs
                ppo_model.FE.hidden_cell = ( batch.FE_hidden_states[0].reshape(hp.recurrent_layers,-1,hp.hidden_size), 
                                             batch.FE_cell_states[0].reshape(hp.recurrent_layers,-1,hp.hidden_size) 
                                            )
                
                # GLOBAL_COUNT+=1
                # if GLOBAL_COUNT%WARMUP==0:
                #     actor_optimizer.param_groups[0]['lr'] *= 0.5
                #     hp.recurrent_seq_len=3
                #     print('############################################# LR adjusted to',actor_optimizer.param_groups[0]['lr'])
                # Actor loss
                ppo_optimizer.zero_grad()
                features, nodes_in_batch, valid_entries_idx, num_nodes = ppo_model.FE(batch.states)

                selector_bool = batch.selector[0].clone().flatten().to(torch.bool)
                action_dist, values = ppo_model(features, selector=selector_bool)
                # Action dist runs on cpu as a workaround to CUDA illegal memory access.
                action_probabilities = action_dist.log_prob(batch.actions.to("cpu")).to(tp['train_device'])
                
                NORMALIZE_ADVANTAE=True
                if NORMALIZE_ADVANTAE:
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

                # Compute probability ratio from probabilities in logspace.
                probabilities_ratio = torch.exp(action_probabilities - batch.action_probabilities)
                surrogate_loss_0 = probabilities_ratio * batch.advantages
                surrogate_loss_1 =  torch.clamp(probabilities_ratio, 1. - hp.ppo_clip, 1. + hp.ppo_clip) * batch.advantages#[-1, :]
                surrogate_loss_2 = action_dist.entropy().to(tp['train_device'])
                actor_loss = -torch.mean(torch.min(surrogate_loss_0, surrogate_loss_1)) - torch.mean(hp.entropy_factor * surrogate_loss_2)
                
                # Critic loss
                #values = ppo_model.V(features, selector=selector_bool)
                critic_loss = F.mse_loss(batch.discounted_returns, values.squeeze(-1))
                
                vf_coef = 0.5 # basis: paper & sb3 implementation
                loss = actor_loss + vf_coef * critic_loss
                loss.backward()
                torch.nn.utils.clip_grad.clip_grad_norm_(ppo_model.parameters(), hp.max_grad_norm)
                ppo_optimizer.step()

                del loss
                del action_probabilities
                del features
                del action_dist
                del values

        del trajectory_dataset
        torch.cuda.empty_cache()

                
        end_train_time = time.time()
        print(f"Iteration: {iteration},  Mean reward: {mean_reward}, Mean Entropy: {torch.mean(surrogate_loss_2)}, " +
              f"complete_episode_count: {complete_episode_count}, Gather time: {end_gather_time - start_gather_time:.2f}s, " +
              f"Train time: {end_train_time - start_train_time:.2f}s")

        if tp['save_metrics_tensorboard']:
            tp['writer'].add_scalar("complete_episode_count", complete_episode_count, iteration)
            tp['writer'].add_scalar("total_reward", mean_reward , iteration)
            tp['writer'].add_scalar("actor_loss", actor_loss.item(), iteration)
            tp['writer'].add_scalar("critic_loss", critic_loss.item(), iteration)
            tp['writer'].add_scalar("policy_entropy", torch.mean(surrogate_loss_2).item(), iteration)
        if tp['save_parameters_tensorboard']:
            save_parameters(tp['writer'], "ppo_model", ppo_model, iteration, tp)
            #save_parameters(writer, "value", critic, iteration)
        if iteration % tp['checkpoint_frequency'] == 0: 
            print('rec seq len',hp.recurrent_seq_len)
            print('actor lr',ppo_optimizer.param_groups[0]['lr'])
            save_checkpoint(ppo_model, ppo_optimizer, iteration, stop_conditions, hp, tp)
        iteration += 1
        
    return stop_conditions.best_reward 

def train_model_EMB(env, ppo_model, ppo_optimizer, iteration, stop_conditions, hp, tp):   
    # Vector environment manages multiple instances of the environment.
    # A key difference between this and the standard gym environment is it automatically resets.
    # Therefore when the done flag is active in the done vector the corresponding state is the first new state.
    # env = gym.vector.make(ENV, hp.parallel_rollouts, asynchronous=ASYNCHRONOUS_ENVIRONMENT)
    # GLOBAL_COUNT=0   
    #ppo_optimizer.param_groups[0]['lr'] *= 0.25
    while iteration < stop_conditions.max_iterations:      
        ppo_model = ppo_model.to(tp['gather_device'])
        start_gather_time = time.time()
        stop_conditions.iteration = iteration

        # Gather trajectories.
        input_data = {"env": env, "ppo_model": ppo_model, "gather_device":tp['gather_device']}
        trajectory_tensors = gather_trajectories_EMB(input_data, hp)
        trajectory_episodes, len_episodes = split_trajectories_episodes(trajectory_tensors, hp)
        trajectories = pad_and_compute_returns(trajectory_episodes, len_episodes,  hp)

        # Calculate mean reward.
        complete_episode_count = trajectories["terminals"].sum().item()
        terminal_episodes_rewards = (trajectories["terminals"].sum(axis=1) * trajectories["true_rewards"].sum(axis=1)).sum().item()
        mean_reward =  terminal_episodes_rewards / (complete_episode_count)

        # Check stop conditions.
        if mean_reward > stop_conditions.best_reward:
            stop_conditions.best_reward = mean_reward
            stop_conditions.fail_to_improve_count = 0
            print("NEW BEST MEAN REWARD----------------------<<<<<<<<<<< ")
            print('rec seq len',hp.recurrent_seq_len)
            print('actor lr',ppo_optimizer.param_groups[0]['lr'])

            if iteration>=1:
                print('#################### SAVE CHECKPOINT #######################')
                save_checkpoint(ppo_model, ppo_optimizer, 99999, stop_conditions, hp, tp)
                # best model saved as iteration0
        else:
            stop_conditions.fail_to_improve_count += 1
        if stop_conditions.fail_to_improve_count > hp.patience:
            print(f"Policy has not yielded higher reward for {hp.patience} iterations...  Stopping now.")
            break

        trajectory_dataset = TrajectoryDataset_EMB(trajectories, device=tp['train_device'], hp=hp)
        end_gather_time = time.time()
        start_train_time = time.time()
        
        ppo_model = ppo_model.to(tp['train_device'])

        # Train actor and critic. 
        for epoch_idx in range(hp.ppo_epochs): 
            for batch in trajectory_dataset:

                # Prime the LSTMs
                ppo_model.LSTM.hidden_cell = ( batch.EMB_hidden_states[0].reshape(hp.recurrent_layers,-1,hp.hidden_size), 
                                             batch.EMB_cell_states[0].reshape(hp.recurrent_layers,-1,hp.hidden_size) 
                                            )
                
                # GLOBAL_COUNT+=1
                # if GLOBAL_COUNT%WARMUP==0:
                #     actor_optimizer.param_groups[0]['lr'] *= 0.5
                #     hp.recurrent_seq_len=3
                #     print('############################################# LR adjusted to',actor_optimizer.param_groups[0]['lr'])
                # Actor loss
                ppo_optimizer.zero_grad()
                features, nodes_in_batch, valid_entries_idx, num_nodes = ppo_model.FE(batch.states)

                selector_bool = batch.selector[0].clone().flatten().to(torch.bool)
                features=ppo_model.LSTM(features, selector=selector_bool)
                action_dist, values = ppo_model(features, selector=selector_bool)
                # Action dist runs on cpu as a workaround to CUDA illegal memory access.
                action_probabilities = action_dist.log_prob(batch.actions.to("cpu")).to(tp['train_device'])
                
                NORMALIZE_ADVANTAE=True
                if NORMALIZE_ADVANTAE:
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

                # Compute probability ratio from probabilities in logspace.
                probabilities_ratio = torch.exp(action_probabilities - batch.action_probabilities)
                surrogate_loss_0 = probabilities_ratio * batch.advantages
                surrogate_loss_1 =  torch.clamp(probabilities_ratio, 1. - hp.ppo_clip, 1. + hp.ppo_clip) * batch.advantages#[-1, :]
                surrogate_loss_2 = action_dist.entropy().to(tp['train_device'])
                actor_loss = -torch.mean(torch.min(surrogate_loss_0, surrogate_loss_1)) - torch.mean(hp.entropy_factor * surrogate_loss_2)
                
                # Critic loss
                #values = ppo_model.V(features, selector=selector_bool)
                critic_loss = F.mse_loss(batch.discounted_returns, values.squeeze(-1))
                
                vf_coef = 0.5 # basis: paper & sb3 implementation
                loss = actor_loss + vf_coef * critic_loss
                loss.backward()
                torch.nn.utils.clip_grad.clip_grad_norm_(ppo_model.parameters(), hp.max_grad_norm)
                ppo_optimizer.step()

                del loss
                del action_probabilities
                del features
                del action_dist
                del values

        del trajectory_dataset
        torch.cuda.empty_cache()

                
        end_train_time = time.time()
        print(f"Iteration: {iteration},  Mean reward: {mean_reward}, Mean Entropy: {torch.mean(surrogate_loss_2)}, " +
              f"complete_episode_count: {complete_episode_count}, Gather time: {end_gather_time - start_gather_time:.2f}s, " +
              f"Train time: {end_train_time - start_train_time:.2f}s")

        if tp['save_metrics_tensorboard']:
            tp['writer'].add_scalar("complete_episode_count", complete_episode_count, iteration)
            tp['writer'].add_scalar("total_reward", mean_reward , iteration)
            tp['writer'].add_scalar("actor_loss", actor_loss.item(), iteration)
            tp['writer'].add_scalar("critic_loss", critic_loss.item(), iteration)
            tp['writer'].add_scalar("policy_entropy", torch.mean(surrogate_loss_2).item(), iteration)
        if tp['save_parameters_tensorboard']:
            save_parameters(tp['writer'], "ppo_model", ppo_model, iteration, tp)
            #save_parameters(writer, "value", critic, iteration)
        if iteration % tp['checkpoint_frequency'] == 0: 
            print('rec seq len',hp.recurrent_seq_len)
            print('actor lr',ppo_optimizer.param_groups[0]['lr'])
            save_checkpoint(ppo_model, ppo_optimizer, iteration, stop_conditions, hp, tp)
        iteration += 1
        
    return stop_conditions.best_reward

def make_custom(config, num_envs=1, asynchronous=True, wrappers=None, **kwargs):
    """Create a vectorized environment from multiple copies of an environment,
    from its id.
    Parameters
    """
    from gym.envs import make as make_
    # dirname = "./results/results_Phase3"
    # filename = dirname+"/"+config['train_on']+"_"+config['nfm_func']+"_obs"+config['obs_mask']
    # if config['obs_mask'] != "None": filename += str(config['obs_rate'])
    # filename+=".bin"
    # try:
    #     OF=open(filename, "rb")
    #     OF.close()
    # except:
        # env,_ = ConstructTrainSet(config, apply_wrappers=True, remove_paths=False, tset=config['train_on'])
        # pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
        # with open(filename, "wb") as f:
        #     env = pickle.dump(env, f)
    
    _, env_all_list = ConstructTrainSet(config, apply_wrappers=True, remove_paths=False, tset=config['train_on'])
    def _make_env():
        #print('env func loading started...',end='')
        #with open(filename, "rb") as f:
        #    env = pickle.load(f)  
        env, _ = ConstructTrainSet(config, apply_wrappers=True, remove_paths=False, tset=config['train_on'])

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
        print('env func executed...')
        return env

    env_fns = [_make_env for _ in range(num_envs)]
    if asynchronous:
        return AsyncVectorEnv(env_fns), env_all_list
    else:
        return SyncVectorEnv(env_fns), env_all_list
