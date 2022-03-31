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
from modules.gnn.construct_trainsets import ConstructTrainSet
from modules.ppo.models_ngo import MaskablePPOPolicy, MaskablePPOPolicy_shared_lstm, MaskablePPOPolicy_shared_lstm_concat

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
            value = ppo_model.V(features, terminal.to(gather_device), selector=selector.flatten().to(torch.bool))
            trajectory_data["values"].append( value.squeeze().cpu())
            action_dist = ppo_model.PI(features, terminal.to(gather_device), selector=selector.flatten().to(torch.bool))
            action = action_dist.sample().reshape(hp.parallel_rollouts, -1)
            if not ppo_model.continuous_action_space:
                action = action.squeeze(1)
            trajectory_data["actions"].append(action.cpu())
            trajectory_data["action_probabilities"].append(action_dist.log_prob(action).squeeze(0).cpu())

            # Step environment 
            action_np = action.cpu().numpy()
            obsv, reward, done, _ = env.step(action_np)
            terminal = torch.tensor(done).float()
            transformed_reward = hp.scale_reward * torch.max(_min_reward_values, torch.tensor(reward).float())
                                                             
            trajectory_data["rewards"].append(transformed_reward)
            trajectory_data["true_rewards"].append(torch.tensor(reward).float())
            trajectory_data["terminals"].append(terminal)
    
        # Compute final value to allow for incomplete episodes.
        state = torch.tensor(obsv, dtype=torch.float32)
        features, nodes_in_batch, valid_entries_idx, num_nodes = ppo_model.FE(state.unsqueeze(0).to(gather_device))        
        terminal_dense = torch.repeat_interleave(terminal.to(torch.bool), torch.ones(batch_size,dtype=torch.int64)*hp.max_possible_nodes, dim=0)
        ppo_model.PI.hidden_cell[0][:,terminal_dense,:] = 0.
        ppo_model.PI.hidden_cell[1][:,terminal_dense,:] = 0.
        ppo_model.V.hidden_cell[0][:,terminal_dense,:] = 0.
        ppo_model.V.hidden_cell[1][:,terminal_dense,:] = 0.

        value = ppo_model.V(features.to(gather_device), terminal.to(gather_device))
        # Future value for terminal episodes is 0.
        trajectory_data["values"].append(value.squeeze().cpu() * (1 - terminal))

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
