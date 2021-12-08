import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, graph_env, epsilon0, initial_Q_values=0.):
        # const
        self.nS = graph_env.sp.V
        self.initial_Q_values = initial_Q_values
        self.Q = {}
        self.epsilon0 = epsilon0
        self.epsilon   = epsilon0
        # graph attributes
        self.actions_from_node = graph_env.neighbors
        self.in_degrees = graph_env.in_degree
        self.out_degrees = graph_env.out_degree
        max_out_degree=0
        for n,deg in self.out_degrees.items():
            if deg > max_out_degree:
                max_out_degree = deg
        self.max_out_degree = max_out_degree
        # vars
        self.epsilon = self.epsilon0
        self.state_count = {} 
        self.sa_count = {} #np.zeros((self.nS,max_outdegree))
    
    def Reset(self):
        self.epsilon = self.epsilon0
        self.state_count = {}
        self.sa_count = {}
        self.Q = {}
        #self.eps_slope  = (self.epsilon_0-self.eps_min)/self.eps_cutoff
        # for i in range(self.nS):
        #     self.sa_count[i] = 0
        #     self.Q[(i,self.e_node0)] = np.ones(self.num_actions[i]).astype(np.float32) * self.initial_Q_values
    
    def sample_action(self, obs, available_actions):
        """
        """
        # If state has never been visited, create Q table entry and initiate counts
        if obs not in self.Q:
            self.Q[obs] = np.ones(len(self.actions_from_node[obs[0]])).astype(np.float32) * self.initial_Q_values
            self.state_count[obs] = 0.
        if obs not in self.sa_count:
            self.sa_count[obs] = np.zeros(len(self.actions_from_node[obs[0]]))

        # Determine action
        epsilon = self.epsilon
        greedy = np.random.choice([False, True], p=[epsilon, 1-epsilon])
        if greedy:
            max_actions = np.max(self.Q[obs])
            max_action_indices = np.where(self.Q[obs]==max_actions)
            action_idx = np.random.choice(max_action_indices[0])
            action = self.actions_from_node[obs[0]][action_idx]
        else:
            action_idx = np.random.choice(np.arange(len(self.actions_from_node[obs[0]])))
            action = self.actions_from_node[obs[0]][action_idx]
        self.sa_count[obs][action_idx] += 1
        return action_idx, action

    def sample_greedy_action(self, obs, available_actions=None):
        """
        """
        # If state has never been visited, create Q table entry and initiate counts
        if obs not in self.Q:
            self.Q[obs] = np.ones(len(self.actions_from_node[obs[0]])).astype(np.float32) * self.initial_Q_values
            self.state_count[obs] = 0.
        if obs not in self.sa_count:
            self.sa_count[obs] = np.zeros(len(self.actions_from_node[obs[0]]))

        # Determine action (greedy)
        max_actions = np.max(self.Q[obs])
        max_action_indices = np.where(self.Q[obs]==max_actions)
        action_idx = np.random.choice(max_action_indices[0])
        action = self.actions_from_node[obs[0]][action_idx]

        self.sa_count[obs][action_idx] += 1
        return action_idx, action

class LeftUpPolicy(object):
    def __init__(self, env):
        self.env=env
    def sample_greedy_action(self, s, available_actions=None):
        possible_actions = self.env.neighbors[s[0]]
        if s[0]-1 in possible_actions: # can go left
            action = possible_actions.index(int(s[0]-1))
        elif s[0]+self.env.sp.N in possible_actions: # can't go left, go up
            action = possible_actions.index(int(s[0]+self.env.sp.N))
        elif s[0]+1 in possible_actions: # can't go up, go right
            action = possible_actions.index(int(s[0]+1))
        else: # go down
            action = possible_actions.index(int(s[0]-self.env.sp.N))
        return action, None

class RandomPolicy(object):
    def __init__(self, env):
        #self.env=env
        self.out_degree=env.out_degree
    def sample_action(self, s, available_actions=None):
        num_actions = self.out_degree[s[0]]

        #possible_actions = self.env.neighbors[s[0]]
        action = random.randint(0,num_actions-1)
        #action = random.choice(possible_actions)
        return action, None

import networkx as nx
class MinIndegreePolicy(object):
    def __init__(self, env):
        self.env=env
        self.G = self.env.sp.G.to_directed()
        self._assign_weights()

    def _assign_weights(self):    
        for e in self.G.edges():
            v_label = self.env.sp.coord2labels[e[1]]
            self.G[e[0]][e[1]]['weight'] = self.env.in_degree[v_label]**6 # exponential to sufficiently punish larger indegrees

    def sample_action(self, s, available_actions=None):
        return self.sample_greedy_action(s, available_actions)

    def sample_greedy_action(self, s, available_actions=None):
        target_node_label = self.env.sp.coord2labels[self.env.sp.target_node]
        target_node_coord = self.env.sp.target_node
        if type(s) == np.ndarray:
            source_node_label = int(np.where(s[:self.env.sp.V]>0)[0])
        elif type(s) == tuple:
            source_node_label = s[0]
        source_node_coord = self.env.sp.labels2coord[source_node_label]
        best_path = nx.dijkstra_path(self.G, source_node_coord, target_node_coord, weight='weight')
        next_node = self.env.sp.coord2labels[best_path[1]]
        action = self.env.neighbors[source_node_label].index(int(next_node))
        return action, None

class EpsilonGreedyPolicyDQN(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, env, eps_0 = 1., eps_min=0.1, eps_cutoff=100):
        self.Q = Q
        self.rng = np.random.RandomState(1)
        # Epsilon scheduling
        self.epsilon = eps_0
        self.epsilon0   = eps_0
        self.eps_min    = eps_min
        self.eps_cutoff = eps_cutoff
        self.eps_slope  = (eps_0-eps_min)/eps_cutoff
        # Graph attributes
        self.actions_from_node = env.neighbors
        self.out_degree=env.out_degree
        self.V = env.sp.V
        self.max_outdegree=env.max_outdegree
    
    def sample_action(self, obs, available_actions):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        draw = self.rng.uniform(0,1,1)
        if draw <= self.epsilon:
            num_actions = len(available_actions)
            action_idx = random.randint(0,num_actions-1)
            return action_idx, available_actions[action_idx]
        else:
            with torch.no_grad():
                y, modelstate = self.Q(torch.tensor(obs,dtype=torch.float32).to(device))
                num_actions=len(available_actions)
                action_idx = torch.argmax(y[:num_actions]).item()
            return action_idx, None

    def set_epsilon(self, episodes_run):
        if self.eps_cutoff > 0:
            if episodes_run > self.eps_cutoff:
                self.epsilon = self.eps_min
            else:
                self.epsilon = self.epsilon0 - self.eps_slope * episodes_run
        #else:
        #   self.epsilon = self.epsilon0



class EpsilonGreedyPolicyRDQN(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, env, eps_0 = 1., eps_min=0.1, eps_cutoff=100):
        self.Q = Q
        # Attributes used to manage hidden state of lstm
        self.lstm_input_size=Q.lstm.input_size
        self.lstm_hidden_size=Q.lstm.hidden_size
        #self.ht=torch.zeros(self.lstm_hidden_size).to(device)
        #self.ct=torch.zeros(self.lstm_hidden_size).to(device)
        self.reset_init_states()
        # Epsilon scheduling
        self.epsilon = eps_0
        self.epsilon0   = eps_0
        self.eps_min    = eps_min
        self.eps_cutoff = eps_cutoff
        self.eps_slope  = (eps_0-eps_min)/eps_cutoff
        # Graph attributes
        self.actions_from_node = env.neighbors
        self.out_degree=env.out_degree
        self.V = env.sp.V
        self.max_outdegree=env.max_outdegree
        # Sampling generator
        self.rng = np.random.RandomState(1)
    
    def reset_init_states(self):
        self.ht=torch.zeros(self.lstm_hidden_size)[None,None,:].to(device)
        self.ct=torch.zeros(self.lstm_hidden_size)[None,None,:].to(device)

    def sample_action(self, obs, available_actions):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        draw = self.rng.uniform(0,1,1)
        if draw <= self.epsilon:
            num_actions = len(available_actions)
            action_idx = random.randint(0,num_actions-1)
            return action_idx, available_actions[action_idx]
        else:
            with torch.no_grad():
                y, (ht_,ct_) = self.Q(torch.tensor(obs,dtype=torch.float32)[None,None,:].to(device), (self.ht,self.ct))
                self.ht=ht_
                self.ct=ct_
                num_actions=len(available_actions)
                action_idx = torch.argmax(y.squeeze()[:num_actions]).item()
                # if num_actions<4:
                #     if action_idx > num_actions-1:
                #         assert False
                # if len(y.shape) != 2:
                #     k=0
            return action_idx, None

    def set_epsilon(self, episodes_run):
        if self.eps_cutoff > 0:
            if episodes_run > self.eps_cutoff:
                self.epsilon = self.eps_min
            else:
                self.epsilon = self.epsilon0 - self.eps_slope * episodes_run
        #else:
        #   self.epsilon = self.epsilon0
