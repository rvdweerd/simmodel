import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
from torch import optim
from dqn_utils import QNetwork, EpsilonGreedyPolicyDQN, ReplayMemory, FastReplayMemory, train
#import sys
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

def MemTest(env,capacity=10, num_episodes=3, print_output=True):
    if print_output:
        print('\n########## MemoryTest #################')
    #capacity = 10
    memory = ReplayMemory(capacity)

    for episode in range(num_episodes):
        # Sample a transition
        s = env.reset()
        #a = env.action_space.sample()
        while True:
            #a = random.choice(env._availableActionsInCurrentState())
            a = random.randint(0,env.out_degree[env.state[0]]-1)
            s_next, r, done, _ = env.step(a)

            # Push a transition
            memory.push((s, a, r, s_next, done))
            s=s_next
            if done:
                break

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
    return memory

def FastMemTest(env,capacity=10, num_episodes=3, print_output=True):
    if print_output:
        print('\n########## FastMemoryTest #################')
    #capacity = 10
    memory = FastReplayMemory(capacity=10000,tensor_length=100)

    for episode in range(num_episodes):
        # Sample a transition
        s = env.reset()
        #a = env.action_space.sample()
        while True:
            #a = random.choice(env._availableActionsInCurrentState())
            a = random.randint(0,env.out_degree[env.state[0]]-1)
            s_next, r, done, _ = env.step(a)

            # Push a transition
            memory.push((s, a, r, s_next, done))
            s=s_next
            if done:
                break

    # Sample a batch size of 1
    out=memory.sample(3)
    #print(out)
    s=out[0][0]
    if print_output:
        print('State shape:',s.shape)
        print('Non-zero entries:',torch.where(s!=0)[0])
        print('Memory sampling... passed')
    return memory


def TensorTest(env):
    print('\n########## TensorTest #################')
    replay_buffer=MemTest(env,print_output=False)
    transitions=replay_buffer.sample(3)
    state, action, reward, next_state, done = zip(*transitions)
    state = torch.tensor(state, dtype=torch.float).to(device)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    print('Tensor conversion: passed')
    print('State shape:',state.shape)
    print('Torch test... passed')

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

def TrainTest(env):
    replay_buffer = MemTest(env,capacity=100, num_episodes=50, print_output=False)

    # You may want to test your functions individually, but after you do so lets see if the method train works.
    batch_size = 64
    discount_factor = 0.8
    learn_rate = 1e-3
    dim_in=(1+env.sp.U)*env.sp.V
    dim_out=4 #(max out-degree)
    dim_hidden=128
    qnet=QNetwork(dim_in,dim_out,dim_hidden)

    # Simple gradient descent may take long, so we will use Adam
    optimizer = optim.Adam(qnet.parameters(), learn_rate)
    loss = train(qnet, replay_buffer, optimizer, batch_size, discount_factor)
    print (loss)