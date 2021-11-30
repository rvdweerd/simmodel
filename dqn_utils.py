import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class QNetwork(nn.Module):
    
    def __init__(self, num_in, num_out, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(num_in, num_hidden)
        self.l2 = nn.Linear(num_hidden, num_out)
        self.numTrainableParameters()

    def forward(self, x):
        # YOUR CODE HERE
        return self.l2(nn.ReLU()(self.l1(x)))
    
    def numTrainableParameters(self):
        total = 0
        for name, p in self.named_parameters():
            total += np.prod(p.shape)
            print("{:24s} {:12s} requires_grad={}".format(name, str(list(p.shape)), p.requires_grad))
        print("\nTotal number of parameters: {}\n".format(total))
        assert total == sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total

def TorchTest():
    # Let's instantiate and test if it works
    print('\n########## TorchTest #################')
    num_hidden = 128
    torch.manual_seed(1)
    Q_net = QNetwork(4,2,num_hidden)

    torch.manual_seed(1)
    test_model = nn.Sequential(
        nn.Linear(4, num_hidden), 
        nn.ReLU(), 
        nn.Linear(num_hidden, 2)
    )

    x = torch.rand(10, 4)

    # If you do not need backpropagation, wrap the computation in the torch.no_grad() context
    # This saves time and memory, and PyTorch complaints when converting to numpy
    with torch.no_grad():
        assert np.allclose(Q_net(x).numpy(), test_model(x).numpy())
    print('Torch test... passed')

class ReplayMemory:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        #s  = self.state2np(transition[0])
        #s_ = self.state2np(transition[3])
        if len(self.memory) >= self.capacity:
            del self.memory[0]
        self.memory.append(transition)

    def sample(self, batch_size):
        # YOUR CODE HERE
        assert batch_size <= len(self.memory)
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def MemTest(env,print_output=True):
    if print_output:
        print('\n########## MemoryTest #################')
    capacity = 10
    memory = ReplayMemory(capacity)

    # Sample a transition
    s = env.reset()
    #a = env.action_space.sample()
    for i in range(5):
        #a = random.choice(env._availableActionsInCurrentState())
        a = random.randint(0,env.out_degree[env.state[0]]-1)
        s_next, r, done, _ = env.step(a)

        # Push a transition
        memory.push((s, a, r, s_next, done))
        s=s_next

    # Sample a batch size of 1
    out=memory.sample(3)
    #print(out)
    s=out[0][0]
    if print_output:
        if type(s) == tuple:
            print('State:',s)
        elif type(s) == np.ndarray:
            print('State shape:',s.shape)
            print('Non-zero entries:',np.where(s!=0))
        print('Memory sampling... passed')
    return out

def TensorTest(env):
    print('\n########## TensorTest #################')
    transitions=MemTest(env,print_output=False)
    state, action, reward, next_state, done = zip(*transitions)
    state = torch.tensor(state, dtype=torch.float).to(device)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    print('Tensor conversion: passed')
    print('State shape:',state.shape)
    print('Torch test... passed')

def PolicyTest(env):
    print('\n########## PolicyTest #################')
    dim_in=(1+env.sp.U)*env.sp.V
    dim_out=4 #(max out-degree)
    dim_hidden=128
    qnet=QNetwork(dim_in,dim_out,dim_hidden)
    s = env.reset()
    #print (torch.tensor(s,dtype=torch.float32).to(device))
    epg = EpsilonGreedyPolicyDQN(qnet, 0.05,env)
    a = epg.sample_action(s,env._availableActionsInCurrentState())
    assert not torch.is_tensor(a)
    print('Policy test... passed')

def get_epsilon(it):
    if it > 1000:
        return 0.05
    else:
        return 1 - .00095 * it

class EpsilonGreedyPolicyDQN(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon, env):
        self.Q = Q
        self.epsilon = epsilon
        self.rng = np.random.RandomState(1)
        self.actions_from_node = env.neighbors
        self.V = env.sp.V
        self.max_outdegree=4
    
    def sample_action(self, obs, available_actions):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        #epos = np.where(obs[:self.V]==1)
        draw = self.rng.uniform(0,1,1)
        #print(draw)
        if draw <= self.epsilon:
            return self.rng.choice(available_actions)
        else:
            with torch.no_grad():
                y = self.Q(torch.tensor(obs,dtype=torch.float32))
                num_actions=len(available_actions)
                action = torch.argmax(y[:num_actions]).item()
            return action
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

if __name__ == '__main__':
    TorchTest()
    