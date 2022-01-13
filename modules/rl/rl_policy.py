import numpy as np
import random
import torch
#from torch import nn
#import torch.nn.functional as F
from torch.distributions import Categorical
#from torch import optim
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Policy():
    def __init__(self, type_name='none'):
        self.type = type_name
        self.model = None
        self.__name__ = 'none'
        self.deterministic=True
    def sample_action(self, obs, available_actions):
        return self.sample_greedy_action(obs, available_actions)
    def sample_greedy_action(self, obs, available_actions):
        pass
    def reset_hidden_states(self):
        pass
    def reset_epsilon(self):
        pass
    def get_action_probs(self):
        return None

class EpsilonGreedyPolicy(Policy):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, graph_env, epsilon0, epsilon_min=0., initial_Q_values=0.):
        # const
        self.nS = graph_env.sp.V
        self.initial_Q_values = initial_Q_values
        self.Q = {}
        self.epsilon0 = epsilon0
        self.epsilon_min = epsilon_min
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
        self.__name__ = 'EpsGreedy, tabular Q'

    def reset_epsilon(self):
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

class LeftUpPolicy(Policy):
    def __init__(self, env):
        super().__init__('Heuristic policy')
        self.env=env
        self.__name__ = 'LeftUp'
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

class RandomPolicy(Policy):
    def __init__(self, env):
        #self.env=env
        super().__init__('Heuristic policy')
        self.out_degree=env.out_degree
        self.__name__ = 'RandomWalker'
    def sample_action(self, s, available_actions=None):
        num_actions = self.out_degree[s[0]]

        #possible_actions = self.env.neighbors[s[0]]
        action = random.randint(0,num_actions-1)
        #action = random.choice(possible_actions)
        return action, None

import networkx as nx

class ShortestPathPolicy(Policy):
    def __init__(self, env, weights='equal'):
        super().__init__('Heuristic policy')
        self.env=env
        self.weights=weights
        self.G = self.env.sp.G.to_directed()
        self._assign_weights()
        self.cache = {}
        if weights=='equal':
            self.__name__ = 'ShortestPath'
        elif weights=='min_indegree':
            self.__name__ = 'MinInDegPath'
    def _assign_weights(self):    
        for e in self.G.edges():
            v_label = self.env.sp.coord2labels[e[1]]
            if self.weights == 'equal':
                self.G[e[0]][e[1]]['weight'] = 1 # 
            elif self.weights == 'min_indegree':
                self.G[e[0]][e[1]]['weight'] = self.env.in_degree[v_label]**2 # exponential to sufficiently punish larger indegrees
            else:
                assert False

    def sample_action(self, s, available_actions=None):
        return self.sample_greedy_action(s, available_actions)

    def sample_greedy_action(self, s, available_actions=None):
        if type(s) == np.ndarray:
            source_node_label = int(np.where(s[:self.env.sp.V]>0)[0])
        elif type(s) == tuple:
            source_node_label = s[0]
        if source_node_label not in self.cache:
            min_cost=1e12
            for target_node_label in self.env.sp.target_nodes:
                target_node_coord = self.env.sp.labels2coord[target_node_label]
                source_node_coord = self.env.sp.labels2coord[source_node_label]
                cost, path = nx.single_source_dijkstra(self.G, source_node_coord, target_node_coord, weight='weight')
                if cost < min_cost:
                    best_path=path
                    min_cost = cost
                self.cache[source_node_label] = best_path
        else:
            best_path = self.cache[source_node_label]
        next_node = self.env.sp.coord2labels[best_path[1]]
        action = self.env.neighbors[source_node_label].index(int(next_node))
        #print('----------')
        return action, None

class EpsilonGreedyPolicyDQN(Policy):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, env, eps_0 = 1., eps_min=0.1, eps_cutoff=100):
        super().__init__('DQN based policy')
        self.model = Q
        self.rng = np.random.RandomState(1)
        # Epsilon scheduling
        self.epsilon = eps_0
        self.epsilon0   = eps_0
        self.eps_min    = eps_min
        self.eps_cutoff = eps_cutoff
        self.eps_slope  = (eps_0-eps_min)/eps_cutoff if eps_cutoff > 0 else 0.  
        # Graph attributes
        self.actions_from_node = env.neighbors
        self.out_degree=env.out_degree
        self.V = env.sp.V
        self.max_outdegree=env.max_outdegree
        self.__name__ = 'EpsGreedy, DQN'

    
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
                y, modelstate = self.model(torch.tensor(obs,dtype=torch.float32).to(device))
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

class EpsilonGreedyPolicyDRQN(Policy):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, env, eps_0 = 1., eps_min=0.1, eps_cutoff=100):
        super().__init__('Recurrent DQN based policy')
        self.model = Q
        # Attributes used to manage hidden state of lstm
        self.lstm_input_size=Q.lstm.input_size
        self.lstm_hidden_size=Q.lstm.hidden_size
        #self.ht=torch.zeros(self.lstm_hidden_size).to(device)
        #self.ct=torch.zeros(self.lstm_hidden_size).to(device)
        self.reset_hidden_states()
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
        self.__name__ = 'EpsGreedy, DRQN'
    
    def reset_hidden_states(self):
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
        #draw = self.rng.uniform(0,1,1)
        draw=random.uniform(0,1)
        if draw <= self.epsilon:
            # CHECK! if this is necessary
            with torch.no_grad():
                y, (ht_,ct_) = self.model(torch.tensor(obs,dtype=torch.float32)[None,None,:].to(device), (self.ht,self.ct))
                self.ht=ht_
                self.ct=ct_
            #####
            num_actions = len(available_actions)
            action_idx = random.randint(0,num_actions-1)
            return action_idx, available_actions[action_idx]
        else:
            with torch.no_grad():
                y, (ht_,ct_) = self.model(torch.tensor(obs,dtype=torch.float32)[None,None,:].to(device), (self.ht,self.ct))
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

class EpsilonGreedyPolicyLSTM_PPO(Policy):
    def __init__(self, env, lstm_ppo_model, deterministic=False):
        super().__init__('RecurrentNetwork')
        #self.env=env
        self.lstm_ppo_model = lstm_ppo_model
        self.deterministic = deterministic
        self.lstm_hidden_dim = lstm_ppo_model.lstm_hidden_dim
        self.reset_hidden_states()
        self.__name__ = 'EpsGreedy, LSTM_PPO'

    def reset_hidden_states(self):
        self.lstm_hidden = (torch.zeros([1, 1, self.lstm_hidden_dim], dtype=torch.float), torch.zeros([1, 1, self.lstm_hidden_dim], dtype=torch.float))

    def sample_greedy_action(self, obs, available_actions):
        with torch.no_grad():
            prob, self.lstm_hidden = self.lstm_ppo_model.pi(torch.from_numpy(obs).float(), self.lstm_hidden)
            prob = prob.view(-1)
            if self.deterministic:
                a=prob.argmax().item()
            else:
                m = Categorical(prob)
                a = m.sample().item()               
            # if num_actions<4:
            #     if action_idx > num_actions-1:
            #         assert False
            # if len(y.shape) != 2:
            #     k=0
        return a, None

class EpsilonGreedyPolicyLSTM_PPO2(Policy):
    def __init__(self, env, lstm_ppo_model, deterministic=False):
        super().__init__('RecurrentNetwork')
        #self.env=env
        self.model = lstm_ppo_model.pi
        self.deterministic = deterministic
        #self.lstm_hidden_dim = lstm_ppo_model.lstm_hidden_dim
        self.reset_hidden_states()
        self.__name__ = 'EpsGreedy, LSTM_PPO2'
        self.probs = None
    
    def get_action_probs(self):
        return self.probs

    def reset_hidden_states(self):
        #self.lstm_hidden = (torch.zeros([1, 1, self.lstm_hidden_dim], dtype=torch.float), torch.zeros([1, 1, self.lstm_hidden_dim], dtype=torch.float))
        self.model.hidden_cell=None

    def sample_greedy_action(self, obs, available_actions):
        with torch.no_grad():
            prob = self.model(torch.from_numpy(obs).reshape(1,1,-1).to(dtype=torch.float32).to(device))
            self.probs=prob.probs.flatten().cpu().numpy()
            #prob = prob.view(-1)
            if self.deterministic:
                a=prob.logits.argmax().item()
            else:
            #    m = Categorical(prob)
            #    a = m.sample().item()               
                a=prob.sample().item()
            # if num_actions<4:
            #     if action_idx > num_actions-1:
            #         assert False
            # if len(y.shape) != 2:
            #     k=0
        return a, None

class EpsilonGreedyPolicySB3_PPO(Policy):
    def __init__(self, env, model, deterministic=False):
        super().__init__('Non-RecurrentNetwork')
        #self.env=env
        self.model = model
        self.deterministic = deterministic
        self.reset_hidden_states()
        self.__name__ = 'EpsGreedy, SB3_PPO'
        self.probs = None
        self.all_actions=[i for i in range(env.max_outdegree)]
    def get_action_probs(self):
        return self.probs.astype(np.float64)

    def sample_greedy_action(self, obs, available_actions):
        with torch.no_grad():
            action, _states = self.model.predict(obs, deterministic=self.deterministic)
            obs = torch.tensor(obs)[None,:].to(device)
            all_actions = torch.tensor(self.all_actions).to(device)
            self.probs = torch.exp(self.model.policy.get_distribution(obs).log_prob(all_actions)).detach().cpu().numpy()
        return action, None