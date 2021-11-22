import numpy as np

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, graph_env, eps_0 = 1., eps_min=0.05, eps_cutoff=1000, initial_Q_values=0.):
        # const
        self.nS = graph_env.sp.V
        self.initial_Q_values = initial_Q_values
        self.Q = {}
        self.epsilon0   = eps_0
        self.eps_min    = eps_min
        self.eps_cutoff = eps_cutoff
        self.eps_slope  = (eps_0-eps_min)/eps_cutoff
        # var
        G_directed = graph_env.sp.G.to_directed()
        max_outdegree=0
        for n in G_directed.nodes():
            outdegree=G_directed.out_degree(n)
            if outdegree>max_outdegree:
                max_outdegree=outdegree
        self.epsilon = self.epsilon0
        self.state_count = np.zeros(self.nS).astype(np.float32)
        self.sa_count = np.zeros((self.nS,max_outdegree))
        self.num_actions = []
        # for i in range(self.nS):
        #     self.sa_count[i] = 0
        #     self.Q[(i,self.e_node0)] = np.ones(graph_env.nA[i]).astype(np.float32) * initial_Q_values
        #     self.num_actions.append(graph_env.nA[i])
    
    def Reset(self):
        self.epsilon = self.epsilon0
        self.state_count = np.zeros(self.nS).astype(np.float32)
        self.Q = {}
        # for i in range(self.nS):
        #     self.sa_count[i] = 0
        #     self.Q[(i,self.e_node0)] = np.ones(self.num_actions[i]).astype(np.float32) * self.initial_Q_values
    
    def get_epsilon(self, obs):
        if self.eps_cutoff > 0:
            if self.state_count[obs[0]] > self.eps_cutoff:
                epsilon = self.eps_min
            else:
                epsilon = self.epsilon0 - self.eps_slope * self.state_count[obs[0]]
            self.state_count[obs[0]]+=1
        else:
            epsilon = self.epsilon0
        return epsilon

    def sample_action(self, obs):
        """
        """
        epsilon = self.get_epsilon(obs)
        greedy = np.random.choice([False, True], p=[epsilon, 1-epsilon])
        if greedy:
            if obs not in Q:
                possible_actions = self.
                _availableActionsInCurrentState
                Q[obs]=
            max_actions = np.max(self.Q[obs])
            max_action_idc = np.where(self.Q[obs]==max_actions)
            action = np.random.choice(max_action_idc[0])
        else:
            action = np.random.choice(np.arange(self.num_actions[obs[0]]))
        self.sa_count[obs[0],action]+=1
        return action

class EpsilonGreedyPolicy_graph_Pursuit(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, graph_env, eps_0 = 1., eps_min=0.05, eps_cutoff=1000, initial_Q_values=0.):
        # const
        self.nS = graph_env.number_of_nodes
        self.p_node0 = graph_env.p_state
        self.initial_Q_values = initial_Q_values
        self.Q = {}
        self.epsilon0   = eps_0
        self.eps_min    = eps_min
        self.eps_cutoff = eps_cutoff
        self.eps_slope  = (eps_0-eps_min)/eps_cutoff
        # var
        self.epsilon = self.epsilon0
        self.state_count = np.zeros(self.nS).astype(np.float32)
        self.sa_count = np.zeros((self.nS,graph_env.max_degree))
        self.num_actions = []
        for i in range(self.nS):
            self.sa_count[i] = 0
            self.Q[(i,self.p_node0)] = np.ones(graph_env.nA[i]).astype(np.float32) * initial_Q_values
            self.num_actions.append(graph_env.nA[i])
    
    def Reset(self):
        self.epsilon = self.epsilon0
        self.state_count = np.zeros(self.nS).astype(np.float32)
        self.Q = {}
        for i in range(self.nS):
            self.sa_count[i] = 0
            self.Q[(i,self.p_node0)] = np.ones(self.num_actions[i]).astype(np.float32) * self.initial_Q_values
    
    def get_epsilon(self, obs):
        if self.eps_cutoff > 0:
            if self.state_count[obs[0]] > self.eps_cutoff:
                epsilon = self.eps_min
            else:
                epsilon = self.epsilon0 - self.eps_slope * self.state_count[obs[0]]
            self.state_count[obs[0]]+=1
        else:
            epsilon = self.epsilon0
        return epsilon

    def sample_action(self, obs):
        """
        """
        epsilon = self.get_epsilon(obs)
        greedy = np.random.choice([False, True], p=[epsilon, 1-epsilon])
        if greedy:
            max_actions = np.max(self.Q[obs])
            max_action_idc = np.where(self.Q[obs]==max_actions)
            action = np.random.choice(max_action_idc[0])
        else:
            action = np.random.choice(np.arange(self.num_actions[obs[0]]))
        self.sa_count[obs[0],action]+=1
        return action