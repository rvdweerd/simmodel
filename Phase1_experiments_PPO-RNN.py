import torch
from torch import nn
from torch import distributions
import gym
import copy
from torch.utils.tensorboard import SummaryWriter
from torch import optim
import numpy as np
from itertools import count
import torch.nn.functional as F
import re
import copy
import time
import math
import pathlib
import time
import pickle
import os
from dataclasses import dataclass
import gc
from dotmap import DotMap
from base64 import b64encode
from IPython.display import HTML
from rl_custom_worlds import GetCustomWorld
import argparse
from Phase1_hyperparameters import GetHyperParams_PPO_RNN
from dqn_utils import seed_everything
from rl_utils import GetFullCoverageSample

# Source of this code: 
# https://gitlab.com/ngoodger/ppo_lstm/-/blob/master/recurrent_ppo.ipynb

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)    

# Model hyperparameters
parser.add_argument('--world_name', default='Manhattan3x3_PauseDynamicWorld', type=str, 
                    help='Environment to run',
                    choices=[
                        'Manhattan3x3_PauseFreezeWorld',
                        'Manhattan3x3_PauseDynamicWorld',
                        'Manhattan5x5_FixedEscapeInit',
                        'Manhattan5x5_VariableEscapeInit',
                        'Manhattan5x5_DuplicateSetA',
                        'Manhattan5x5_DuplicateSetB',
                        'MetroU3_e17tborder_FixedEscapeInit',
                        'MetroU3_e17tborder_VariableEscapeInit',
                        'MetroU3_e17t31_FixedEscapeInit',
                        'MetroU3_e17t0_FixedEscapeInit' ])
parser.add_argument('--state_repr', default='et', type=str, 
                    help='Which part of the state is observable',
                    choices=[
                        'et',
                        'etUt',
                        'ete0U0',
                        'etUte0U0' ])
parser.add_argument('--train', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
parser.add_argument('--eval', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
parser.add_argument('--num_seeds', default=1, type=int, 
                    help='Number of seeds to run')

args=parser.parse_args()
WORLD_NAME=args.world_name
STATE_REPR=args.state_repr
EVALUATE=args.eval
TRAIN=args.train
print(TRAIN)
NUM_SEEDS=args.num_seeds
hp_lookup=GetHyperParams_PPO_RNN(WORLD_NAME)
# Save metrics for viewing with tensorboard.
SAVE_METRICS_TENSORBOARD = True
# Save actor & critic parameters for viewing in tensorboard.
SAVE_PARAMETERS_TENSORBOARD = False
# Save training state frequency in PPO iterations.
CHECKPOINT_FREQUENCY = 100
# Step env asynchronously using multiprocess or synchronously.
ASYNCHRONOUS_ENVIRONMENT = False
# Force using CPU for gathering trajectories.
FORCE_CPU_GATHER = True

# Set maximum threads for torch to avoid inefficient use of multiple cpu cores.
torch.set_num_threads(1)
TRAIN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GATHER_DEVICE = "cuda" if torch.cuda.is_available() and not FORCE_CPU_GATHER else "cpu"

# Environment parameters
MAKE_REFLEXIVE=True
ENV=WORLD_NAME
ENV_MASK_VELOCITY = False 
exp_rootdir='./results/PPO-RNN/'+WORLD_NAME+'/'+STATE_REPR+'/'
WORKSPACE_PATH = exp_rootdir

# Default Hyperparameters                                           For PauseFreeWold   For MetroU3_e17_FixedEscapeInit
SCALE_REWARD:         float = 1. 
MIN_REWARD:           float = -15.                                  
HIDDEN_SIZE:          float = hp_lookup['HIDDEN_SIZE'][STATE_REPR]  # 128?
LIN_SIZE:             float = hp_lookup['LIN_SIZE'][STATE_REPR]     # 128               128 3 layers
BATCH_SIZE:           int   = hp_lookup['BATCH_SIZE'][STATE_REPR]   # 80                64
DISCOUNT:             float = 0.99
GAE_LAMBDA:           float = 0.95
PPO_CLIP:             float = 0.2                                   
PPO_EPOCHS:           int   = 10
MAX_GRAD_NORM:        float = 1.                                    
ENTROPY_FACTOR:       float = 0.
ACTOR_LEARNING_RATE:  float = hp_lookup['ACTOR_LEARNING_RATE'][STATE_REPR]                                
CRITIC_LEARNING_RATE: float = hp_lookup['CRITIC_LEARNING_RATE'][STATE_REPR]                                
RECURRENT_SEQ_LEN:    int = hp_lookup['RECURRENT_SEQ_LEN'][STATE_REPR]# 2                 2
RECURRENT_LAYERS:     int = 1                                       # 1     
ROLLOUT_STEPS:        int = hp_lookup['ROLLOUT_STEPS'][STATE_REPR]  # 40               400
PARALLEL_ROLLOUTS:    int = hp_lookup['PARALLEL_ROLLOUTS'][STATE_REPR]# 4                 6
PATIENCE:             int = hp_lookup['PATIENCE'][STATE_REPR]        # 200
TRAINABLE_STD_DEV:    bool = False                                  # False
INIT_LOG_STD_DEV:     float = 0.                                    # 0.
EVAL_DETERMINISTIC:   bool = hp_lookup['EVAL_DETERMINISTIC'][STATE_REPR]
SAMPLE_MULTIPLIER:    int  = hp_lookup['SAMPLE_MULTIPLIER'][STATE_REPR]
# WARMUP=5000
#GLOBAL_COUNT=0

@dataclass
class HyperParameters():
    scale_reward:         float = SCALE_REWARD
    min_reward:           float = MIN_REWARD
    hidden_size:          float = HIDDEN_SIZE
    batch_size:           int   = BATCH_SIZE
    discount:             float = DISCOUNT
    gae_lambda:           float = GAE_LAMBDA
    ppo_clip:             float = PPO_CLIP
    ppo_epochs:           int   = PPO_EPOCHS
    max_grad_norm:        float = MAX_GRAD_NORM
    entropy_factor:       float = ENTROPY_FACTOR
    actor_learning_rate:  float = ACTOR_LEARNING_RATE
    critic_learning_rate: float = CRITIC_LEARNING_RATE
    recurrent_seq_len:    int = RECURRENT_SEQ_LEN
    recurrent_layers:     int = RECURRENT_LAYERS 
    rollout_steps:        int = ROLLOUT_STEPS
    parallel_rollouts:    int = PARALLEL_ROLLOUTS
    patience:             int = PATIENCE
    # Apply to continous action spaces only 
    trainable_std_dev:    bool = TRAINABLE_STD_DEV
    init_log_std_dev:     float = INIT_LOG_STD_DEV

#hp = HyperParameters(parallel_rollouts=32, rollout_steps=1024, batch_size=1024, recurrent_seq_len=8, trainable_std_dev=True,  patience=1000)
hp = HyperParameters(parallel_rollouts=PARALLEL_ROLLOUTS, rollout_steps=ROLLOUT_STEPS, batch_size=BATCH_SIZE, recurrent_seq_len=RECURRENT_SEQ_LEN, trainable_std_dev=TRAINABLE_STD_DEV,  patience=PATIENCE)
batch_count = hp.parallel_rollouts * hp.rollout_steps / hp.recurrent_seq_len / hp.batch_size
print(f"batch_count: {batch_count}")
assert batch_count >= 1., "Less than 1 batch per trajectory.  Are you sure that's what you want?"

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

_INVALID_TAG_CHARACTERS = re.compile(r"[^-/\w\.]")
BASE_CHECKPOINT_PATH = f"{WORKSPACE_PATH}/checkpoints/"

def save_parameters(writer, tag, model, batch_idx):
    """
    Save model parameters for tensorboard.
    """
    for k, v in model.state_dict().items():
        shape = v.shape
        # Fix shape definition for tensorboard.
        shape_formatted = _INVALID_TAG_CHARACTERS.sub("_", str(shape))
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
            
def get_env_space():
    """
    Return obsvervation dimensions, action dimensions and whether or not action space is continuous.
    """
    #env = gym.make(ENV)
    env=GetCustomWorld(WORLD_NAME, make_reflexive=MAKE_REFLEXIVE, state_repr=STATE_REPR, state_enc='tensors')

    continuous_action_space = type(env.action_space) is gym.spaces.box.Box
    if continuous_action_space:
        action_dim =  env.action_space.shape[0]
    else:
        action_dim = env.action_space.n 
    obsv_dim= env.observation_space.shape[0] 
    return obsv_dim, action_dim, continuous_action_space

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, continuous_action_space, trainable_std_dev, init_log_std_dev=None):
        super().__init__()
        self.counter=0
        self.lstm = nn.LSTM(state_dim, hp.hidden_size, num_layers=hp.recurrent_layers)
        
        if type(LIN_SIZE) is not list:
            assert False
        layers=[]
        layer_sizes=[hp.hidden_size]+LIN_SIZE
        for layer_idx in range(1,len(layer_sizes)):
            layers+= [nn.Linear(layer_sizes[layer_idx-1], layer_sizes[layer_idx]), nn.ELU() ]
        layers+= [nn.Linear(layer_sizes[-1], action_dim)]
        self.lin_layers = nn.Sequential(*layers)

        # self.layer_hidden1 = nn.Linear(hp.hidden_size, LIN_SIZE)
        # self.layer_hidden2 = nn.Linear(LIN_SIZE, LIN_SIZE)
        # self.layer_hidden3 = nn.Linear(LIN_SIZE, LIN_SIZE//2)
        # self.layer_policy_logits = nn.Linear(LIN_SIZE//2, action_dim)

        self.action_dim = action_dim
        self.continuous_action_space = continuous_action_space 
        self.log_std_dev = nn.Parameter(init_log_std_dev * torch.ones((action_dim), dtype=torch.float), requires_grad=trainable_std_dev)
        self.covariance_eye = torch.eye(self.action_dim).unsqueeze(0)
        self.hidden_cell = None
        print('Actor network:')
        self.numTrainableParameters()
        print(self)
        
    def get_init_state(self, batch_size, device):
        self.hidden_cell = (torch.zeros(hp.recurrent_layers, batch_size, hp.hidden_size).to(device),
                            torch.zeros(hp.recurrent_layers, batch_size,hp.hidden_size).to(device))
        
    def forward(self, state, terminal=None):
        batch_size = state.shape[1]
        device = state.device
        if self.hidden_cell is None or batch_size != self.hidden_cell[0].shape[1]:
            self.get_init_state(batch_size, device)
        if terminal is not None:
            self.hidden_cell = [value * (1. - terminal).reshape(1, batch_size, 1) for value in self.hidden_cell]
        self.counter+=1

        _, self.hidden_cell = self.lstm(state, self.hidden_cell)
        policy_logits_out = self.lin_layers(self.hidden_cell[0][-1])
        
        if self.continuous_action_space:
            cov_matrix = self.covariance_eye.to(device).expand(batch_size, self.action_dim, self.action_dim) * torch.exp(self.log_std_dev.to(device))
            # We define the distribution on the CPU since otherwise operations fail with CUDA illegal memory access error.
            policy_dist = torch.distributions.multivariate_normal.MultivariateNormal(policy_logits_out.to("cpu"), cov_matrix.to("cpu"))
        else:
            policy_dist = distributions.Categorical(F.softmax(policy_logits_out, dim=1).to("cpu"))
        return policy_dist
    def numTrainableParameters(self):
        print('Qnet size:')
        print('------------------------------------------')
        total = 0
        for name, p in self.named_parameters():
            if p.requires_grad:
                total += np.prod(p.shape)
            print("{:24s} {:12s} requires_grad={}".format(name, str(list(p.shape)), p.requires_grad))
        print("Total number of trainable parameters: {}".format(total))
        print('------------------------------------------')
        assert total == sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total

class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.layer_lstm = nn.LSTM(state_dim, hp.hidden_size, num_layers=hp.recurrent_layers)
        self.counter=0
        if type(LIN_SIZE) is not list:
            assert False
        layers=[]
        layer_sizes=[hp.hidden_size]+LIN_SIZE
        for layer_idx in range(1,len(layer_sizes)):
            layers+= [nn.Linear(layer_sizes[layer_idx-1], layer_sizes[layer_idx]), nn.ELU() ]
        layers+= [nn.Linear(layer_sizes[-1], 1)]
        self.lin_layers = nn.Sequential(*layers)
        self.hidden_cell = None
        
    def get_init_state(self, batch_size, device):
        self.hidden_cell = (torch.zeros(hp.recurrent_layers, batch_size, hp.hidden_size).to(device),
                            torch.zeros(hp.recurrent_layers, batch_size, hp.hidden_size).to(device))
    
    def forward(self, state, terminal=None):
        batch_size = state.shape[1]
        device = state.device
        if self.hidden_cell is None or batch_size != self.hidden_cell[0].shape[1]:
            self.get_init_state(batch_size, device)
        if terminal is not None:
            self.hidden_cell = [value * (1. - terminal).reshape(1, batch_size, 1) for value in self.hidden_cell]
        
        _, self.hidden_cell = self.layer_lstm(state, self.hidden_cell)
        value_out = self.lin_layers(self.hidden_cell[0][-1])
        return value_out

def get_last_checkpoint_iteration():
    """
    Determine latest checkpoint iteration.
    """
    if os.path.isdir(BASE_CHECKPOINT_PATH):
        max_checkpoint_iteration = max([int(dirname) for dirname in os.listdir(BASE_CHECKPOINT_PATH)])
    else:
        max_checkpoint_iteration = 0
    return max_checkpoint_iteration

def save_checkpoint(actor, critic, actor_optimizer, critic_optimizer, iteration, stop_conditions):
    """
    Save training checkpoint.
    """
    checkpoint = DotMap()
    checkpoint.env = ENV
    checkpoint.env_mask_velocity = ENV_MASK_VELOCITY 
    checkpoint.iteration = iteration
    checkpoint.stop_conditions = stop_conditions
    checkpoint.hp = hp
    CHECKPOINT_PATH = BASE_CHECKPOINT_PATH + f"{iteration}/"
    pathlib.Path(CHECKPOINT_PATH).mkdir(parents=True, exist_ok=True) 
    with open(CHECKPOINT_PATH + "parameters.pt", "wb") as f:
        pickle.dump(checkpoint, f)
    with open(CHECKPOINT_PATH + "actor_class.pt", "wb") as f:
        pickle.dump(Actor, f)
    with open(CHECKPOINT_PATH + "critic_class.pt", "wb") as f:
        pickle.dump(Critic, f)
    torch.save(actor.state_dict(), CHECKPOINT_PATH + "actor.pt")
    torch.save(critic.state_dict(), CHECKPOINT_PATH + "critic.pt")
    torch.save(actor_optimizer.state_dict(), CHECKPOINT_PATH + "actor_optimizer.pt")
    torch.save(critic_optimizer.state_dict(), CHECKPOINT_PATH + "critic_optimizer.pt")

def start_or_resume_from_checkpoint():
    """
    Create actor, critic, actor optimizer and critic optimizer from scratch
    or load from latest checkpoint if it exists. 
    """
    max_checkpoint_iteration = get_last_checkpoint_iteration()
    
    obsv_dim, action_dim, continuous_action_space = get_env_space()
    actor = Actor(obsv_dim,
                  action_dim,
                  continuous_action_space=continuous_action_space,
                  trainable_std_dev=hp.trainable_std_dev,
                  init_log_std_dev=hp.init_log_std_dev)
    critic = Critic(obsv_dim)
        
    actor_optimizer = optim.AdamW(actor.parameters(), lr=hp.actor_learning_rate)
    critic_optimizer = optim.AdamW(critic.parameters(), lr=hp.critic_learning_rate)
    
    stop_conditions = StopConditions()
        
    # If max checkpoint iteration is greater than zero initialise training with the checkpoint.
    if max_checkpoint_iteration > 0:
        actor_state_dict, critic_state_dict, actor_optimizer_state_dict, critic_optimizer_state_dict, stop_conditions = load_checkpoint(max_checkpoint_iteration)
        
        actor.load_state_dict(actor_state_dict, strict=True) 
        critic.load_state_dict(critic_state_dict, strict=True)
        actor_optimizer.load_state_dict(actor_optimizer_state_dict)
        critic_optimizer.load_state_dict(critic_optimizer_state_dict)

        # We have to move manually move optimizer states to TRAIN_DEVICE manually since optimizer doesn't yet have a "to" method.
        for state in actor_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                   state[k] = v.to(TRAIN_DEVICE)

        for state in critic_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(TRAIN_DEVICE)

    return actor, critic, actor_optimizer, critic_optimizer, max_checkpoint_iteration, stop_conditions
    
def load_checkpoint(iteration):
    """
    Load from training checkpoint.
    """
    CHECKPOINT_PATH = BASE_CHECKPOINT_PATH + f"{iteration}/"
    with open(CHECKPOINT_PATH + "parameters.pt", "rb") as f:
        checkpoint = pickle.load(f)
        
    assert ENV == checkpoint.env, "To resume training environment must match current settings."
    assert ENV_MASK_VELOCITY == checkpoint.env_mask_velocity, "To resume training model architecture must match current settings."
    assert hp == checkpoint.hp, "To resume training hyperparameters must match current settings."

    actor_state_dict = torch.load(CHECKPOINT_PATH + "actor.pt", map_location=torch.device(TRAIN_DEVICE))
    critic_state_dict = torch.load(CHECKPOINT_PATH + "critic.pt", map_location=torch.device(TRAIN_DEVICE))
    actor_optimizer_state_dict = torch.load(CHECKPOINT_PATH + "actor_optimizer.pt", map_location=torch.device(TRAIN_DEVICE))
    critic_optimizer_state_dict = torch.load(CHECKPOINT_PATH + "critic_optimizer.pt", map_location=torch.device(TRAIN_DEVICE))
    
    return (actor_state_dict, critic_state_dict,
           actor_optimizer_state_dict, critic_optimizer_state_dict,
           checkpoint.stop_conditions)
        
@dataclass
class StopConditions():
    """
    Store parameters and variables used to stop training. 
    """
    best_reward: float = -1e6
    fail_to_improve_count: int = 0
    max_iterations: int = 1000000
        
class MaskVelocityWrapper(gym.ObservationWrapper):
    """
    Gym environment observation wrapper used to mask velocity terms in
    observations. The intention is the make the MDP partially observatiable.
    """
    def __init__(self, env):
        super(MaskVelocityWrapper, self).__init__(env)
        if ENV == "CartPole-v1":
            self.mask = np.array([1., 0., 1., 0.])
        elif ENV == "Pendulum-v0":
            self.mask = np.array([1., 1., 0.])
        elif ENV == "LunarLander-v2":
            self.mask = np.array([1., 1., 0., 0., 1., 0., 1., 1,])
        elif ENV == "LunarLanderContinuous-v2":
            self.mask = np.array([1., 1., 0., 0., 1., 0., 1., 1,])
        else:
            raise NotImplementedError

    def observation(self, observation):
        return  observation * self.mask

_MIN_REWARD_VALUES = torch.full([hp.parallel_rollouts], hp.min_reward)

def gather_trajectories(input_data):
    """
    Gather policy trajectories from gym environment.
    """
    
    # Unpack inputs.
    env = input_data["env"]
    actor = input_data["actor"]
    critic = input_data["critic"]
    
    # Initialise variables.
    obsv = env.reset()
    trajectory_data = {"states": [],
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
        actor.get_init_state(hp.parallel_rollouts, GATHER_DEVICE)
        critic.get_init_state(hp.parallel_rollouts, GATHER_DEVICE)
        # Take 1 additional step in order to collect the state and value for the final state.
        for i in range(hp.rollout_steps):
            
            trajectory_data["actor_hidden_states"].append(actor.hidden_cell[0].squeeze(0).cpu())
            trajectory_data["actor_cell_states"].append(actor.hidden_cell[1].squeeze(0).cpu())
            trajectory_data["critic_hidden_states"].append(critic.hidden_cell[0].squeeze(0).cpu())
            trajectory_data["critic_cell_states"].append(critic.hidden_cell[1].squeeze(0).cpu())
            
            # Choose next action 
            state = torch.tensor(obsv, dtype=torch.float32)
            trajectory_data["states"].append(state)
            value = critic(state.unsqueeze(0).to(GATHER_DEVICE), terminal.to(GATHER_DEVICE))
            trajectory_data["values"].append( value.squeeze(1).cpu())
            action_dist = actor(state.unsqueeze(0).to(GATHER_DEVICE), terminal.to(GATHER_DEVICE))
            action = action_dist.sample().reshape(hp.parallel_rollouts, -1)
            if not actor.continuous_action_space:
                action = action.squeeze(1)
            trajectory_data["actions"].append(action.cpu())
            trajectory_data["action_probabilities"].append(action_dist.log_prob(action).cpu())

            # Step environment 
            action_np = action.cpu().numpy()
            obsv, reward, done, _ = env.step(action_np)
            terminal = torch.tensor(done).float()
            transformed_reward = hp.scale_reward * torch.max(_MIN_REWARD_VALUES, torch.tensor(reward).float())
                                                             
            trajectory_data["rewards"].append(transformed_reward)
            trajectory_data["true_rewards"].append(torch.tensor(reward).float())
            trajectory_data["terminals"].append(terminal)
    
        # Compute final value to allow for incomplete episodes.
        state = torch.tensor(obsv, dtype=torch.float32)
        value = critic(state.unsqueeze(0).to(GATHER_DEVICE), terminal.to(GATHER_DEVICE))
        # Future value for terminal episodes is 0.
        trajectory_data["values"].append(value.squeeze(1).cpu() * (1 - terminal))

    # Combine step lists into tensors.
    trajectory_tensors = {key: torch.stack(value) for key, value in trajectory_data.items()}
    return trajectory_tensors

def split_trajectories_episodes(trajectory_tensors):
    """
    Split trajectories by episode.
    """

    states_episodes, actions_episodes, action_probabilities_episodes = [], [], []
    rewards_episodes, terminal_rewards_episodes, terminals_episodes, values_episodes = [], [], [], []
    policy_hidden_episodes, policy_cell_episodes, critic_hidden_episodes, critic_cell_episodes = [], [], [], []
    len_episodes = []
    trajectory_episodes = {key: [] for key in trajectory_tensors.keys()}
    for i in range(hp.parallel_rollouts):
        terminals_tmp = trajectory_tensors["terminals"].clone()
        terminals_tmp[0, i] = 1
        terminals_tmp[-1, i] = 1
        split_points = (terminals_tmp[:, i] == 1).nonzero() + 1

        split_lens = split_points[1:] - split_points[:-1]
        split_lens[0] += 1
        
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

def pad_and_compute_returns(trajectory_episodes, len_episodes):

    """
    Pad the trajectories up to hp.rollout_steps so they can be combined in a
    single tensor.
    Add advantages and discounted_returns to trajectories.
    """

    episode_count = len(len_episodes)
    advantages_episodes, discounted_returns_episodes = [], []
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
                                                        values=trajectory_episodes["values"][i],
                                                        discount=DISCOUNT,
                                                        gae_lambda=GAE_LAMBDA), single_padding)))
        padded_trajectories["discounted_returns"].append(torch.cat((calc_discounted_return(rewards=trajectory_episodes["rewards"][i],
                                                                    discount=DISCOUNT,
                                                                    final_value=trajectory_episodes["values"][i][-1]), single_padding)))
    return_val = {k: torch.stack(v) for k, v in padded_trajectories.items()} 
    return_val["seq_len"] = torch.tensor(len_episodes)
    
    return return_val

@dataclass
class TrajectorBatch():
    """
    Dataclass for storing data batch.
    """
    states: torch.tensor
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
    def __init__(self, trajectories, batch_size, device, batch_len):
        
        # Combine multiple trajectories into
        self.trajectories = {key: value.to(device) for key, value in trajectories.items()}
        self.batch_len = batch_len 
        truncated_seq_len = torch.clamp(trajectories["seq_len"] - batch_len + 1, 0, hp.rollout_steps)
        self.cumsum_seq_len =  np.cumsum(np.concatenate( (np.array([0]), truncated_seq_len.numpy())))
        self.batch_size = batch_size
        
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

try:
    from collections.abc import Iterable
except ImportError:
    Iterable = (tuple, list)
from gym.vector.async_vector_env import AsyncVectorEnv
from gym.vector.sync_vector_env import SyncVectorEnv
from gym.vector.vector_env import VectorEnv, VectorEnvWrapper
__all__ = ["AsyncVectorEnv", "SyncVectorEnv", "VectorEnv", "VectorEnvWrapper", "make"]
def make_custom(env_id, num_envs=1, asynchronous=True, wrappers=None, **kwargs):
    """Create a vectorized environment from multiple copies of an environment,
    from its id.
    Parameters
    """
    from gym.envs import make as make_

    def _make_env():
        #env = make_(id, **kwargs)
        env=GetCustomWorld(env_id, make_reflexive=MAKE_REFLEXIVE, state_repr=STATE_REPR, state_enc='tensors')
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


def train_model(actor, critic, actor_optimizer, critic_optimizer, iteration, stop_conditions):   
    # Vector environment manages multiple instances of the environment.
    # A key difference between this and the standard gym environment is it automatically resets.
    # Therefore when the done flag is active in the done vector the corresponding state is the first new state.
    #env = gym.vector.make(ENV, hp.parallel_rollouts, asynchronous=ASYNCHRONOUS_ENVIRONMENT)
    # GLOBAL_COUNT=0
    env = make_custom(WORLD_NAME, hp.parallel_rollouts, asynchronous=ASYNCHRONOUS_ENVIRONMENT)
    if ENV_MASK_VELOCITY:
        env = MaskVelocityWrapper(env)

    while iteration < stop_conditions.max_iterations:      

        actor = actor.to(GATHER_DEVICE)
        critic = critic.to(GATHER_DEVICE)
        start_gather_time = time.time()

        # Gather trajectories.
        input_data = {"env": env, "actor": actor, "critic": critic, "discount": hp.discount,
                      "gae_lambda": hp.gae_lambda}
        trajectory_tensors = gather_trajectories(input_data)
        trajectory_episodes, len_episodes = split_trajectories_episodes(trajectory_tensors)
        trajectories = pad_and_compute_returns(trajectory_episodes, len_episodes)

        # Calculate mean reward.
        complete_episode_count = trajectories["terminals"].sum().item()
        terminal_episodes_rewards = (trajectories["terminals"].sum(axis=1) * trajectories["true_rewards"].sum(axis=1)).sum()
        mean_reward =  terminal_episodes_rewards / (complete_episode_count)

        # Check stop conditions.
        if mean_reward > stop_conditions.best_reward:
            stop_conditions.best_reward = mean_reward
            stop_conditions.fail_to_improve_count = 0
            print("NEW BEST MEAN REWARD----------------------<<<<<<<<<<< ")#global count: ",GLOBAL_COUNT)
            print('rec seq len',hp.recurrent_seq_len)
            print('actor lr',actor_optimizer.param_groups[0]['lr'])

            if iteration>=50:
                print('#################### SAVE CHECKPOINT #######################')
                save_checkpoint(actor,critic, actor_optimizer, critic_optimizer, 99999, stop_conditions)
                # best model saved as iteration0
        else:
            stop_conditions.fail_to_improve_count += 1
        if stop_conditions.fail_to_improve_count > hp.patience:
            print(f"Policy has not yielded higher reward for {hp.patience} iterations...  Stopping now.")
            break

        trajectory_dataset = TrajectoryDataset(trajectories, batch_size=hp.batch_size,
                                        device=TRAIN_DEVICE, batch_len=hp.recurrent_seq_len)
        end_gather_time = time.time()
        start_train_time = time.time()
        
        actor = actor.to(TRAIN_DEVICE)
        critic = critic.to(TRAIN_DEVICE)

        # Train actor and critic. 
        for epoch_idx in range(hp.ppo_epochs): 
            for batch in trajectory_dataset:

                # Get batch 
                actor.hidden_cell = (batch.actor_hidden_states[:1], batch.actor_cell_states[:1])
                
                # GLOBAL_COUNT+=1
                # if GLOBAL_COUNT%WARMUP==0:
                #     actor_optimizer.param_groups[0]['lr'] *= 0.5
                #     hp.recurrent_seq_len=3
                #     print('############################################# LR adjusted to',actor_optimizer.param_groups[0]['lr'])
                # Update actor
                actor_optimizer.zero_grad()
                action_dist = actor(batch.states)
                # Action dist runs on cpu as a workaround to CUDA illegal memory access.
                action_probabilities = action_dist.log_prob(batch.actions[-1, :].to("cpu")).to(TRAIN_DEVICE)
                # Compute probability ratio from probabilities in logspace.
                probabilities_ratio = torch.exp(action_probabilities - batch.action_probabilities[-1, :])
                surrogate_loss_0 = probabilities_ratio * batch.advantages[-1, :]
                surrogate_loss_1 =  torch.clamp(probabilities_ratio, 1. - hp.ppo_clip, 1. + hp.ppo_clip) * batch.advantages[-1, :]
                surrogate_loss_2 = action_dist.entropy().to(TRAIN_DEVICE)
                actor_loss = -torch.mean(torch.min(surrogate_loss_0, surrogate_loss_1)) - torch.mean(hp.entropy_factor * surrogate_loss_2)
                actor_loss.backward() 
                torch.nn.utils.clip_grad.clip_grad_norm_(actor.parameters(), hp.max_grad_norm)
                actor_optimizer.step()

                # Update critic
                critic_optimizer.zero_grad()
                critic.hidden_cell = (batch.critic_hidden_states[:1], batch.critic_cell_states[:1])
                values = critic(batch.states)
                critic_loss = F.mse_loss(batch.discounted_returns[-1, :], values.squeeze(1))
                torch.nn.utils.clip_grad.clip_grad_norm_(critic.parameters(), hp.max_grad_norm)
                critic_loss.backward() 
                critic_optimizer.step()
                
        end_train_time = time.time()
        print(f"Iteration: {iteration},  Mean reward: {mean_reward}, Mean Entropy: {torch.mean(surrogate_loss_2)}, " +
              f"complete_episode_count: {complete_episode_count}, Gather time: {end_gather_time - start_gather_time:.2f}s, " +
              f"Train time: {end_train_time - start_train_time:.2f}s")

        if SAVE_METRICS_TENSORBOARD:
            writer.add_scalar("complete_episode_count", complete_episode_count, iteration)
            writer.add_scalar("total_reward", mean_reward , iteration)
            writer.add_scalar("actor_loss", actor_loss, iteration)
            writer.add_scalar("critic_loss", critic_loss, iteration)
            writer.add_scalar("policy_entropy", torch.mean(surrogate_loss_2), iteration)
        if SAVE_PARAMETERS_TENSORBOARD:
            save_parameters(writer, "actor", actor, iteration)
            save_parameters(writer, "value", critic, iteration)
        if iteration % CHECKPOINT_FREQUENCY == 0: 
            print('rec seq len',hp.recurrent_seq_len)
            print('actor lr',actor_optimizer.param_groups[0]['lr'])
            save_checkpoint(actor,critic, actor_optimizer, critic_optimizer, iteration, stop_conditions)
        iteration += 1
        
    return stop_conditions.best_reward 

writer = SummaryWriter(log_dir=f"{WORKSPACE_PATH}/logs")
def TrainAndSaveModel():
    actor, critic, actor_optimizer, critic_optimizer, iteration, stop_conditions = start_or_resume_from_checkpoint()
    score = train_model(actor, critic, actor_optimizer, critic_optimizer, iteration, stop_conditions)

from rl_policy import EpsilonGreedyPolicyLSTM_PPO2
from rl_utils import EvaluatePolicy
def EvaluateSavedModel():
    actor, critic, actor_optimizer, critic_optimizer, iteration, stop_conditions = start_or_resume_from_checkpoint()
    @dataclass
    class LSTP_PPO_MODEL():
        lstm_hidden_dim:    int 
        pi:                 torch.nn.modules.module.Module = None
        v:                  torch.nn.modules.module.Module = None
    lstm_ppo_model = LSTP_PPO_MODEL(lstm_hidden_dim=HIDDEN_SIZE ,pi=actor.to(TRAIN_DEVICE), v=critic.to(TRAIN_DEVICE))

    env=GetCustomWorld(WORLD_NAME, make_reflexive=MAKE_REFLEXIVE, state_repr=STATE_REPR, state_enc='tensors')
    policy = EpsilonGreedyPolicyLSTM_PPO2(env,lstm_ppo_model, deterministic=EVAL_DETERMINISTIC)
    lengths, returns, captures = EvaluatePolicy(env, policy  , env.world_pool*SAMPLE_MULTIPLIER, print_runs=False, save_plots=False, logdir=exp_rootdir)    
    plotlist = GetFullCoverageSample(returns, env.world_pool*SAMPLE_MULTIPLIER, bins=10, n=10)
    EvaluatePolicy(env, policy, plotlist, print_runs=True, save_plots=True, logdir=exp_rootdir)

if TRAIN:
    for RANDOM_SEED in range(NUM_SEEDS):
        # Set random seed for consistant runs.
        #torch.random.manual_seed(RANDOM_SEED)
        #np.random.seed(RANDOM_SEED)
        seed_everything(RANDOM_SEED)
        TrainAndSaveModel()
if EVALUATE:
    EvaluateSavedModel()