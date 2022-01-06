import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
from torch import optim
from modules.rl.rl_models import QNetwork, RecurrentQNetwork
from modules.rl.rl_policy import EpsilonGreedyPolicyDQN, EpsilonGreedyPolicyDRQN
from modules.dqn.dqn_utils import ReplayMemory, FastReplayMemory, SeqReplayMemory, train
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from modules.rl.environments import GraphWorld
import simdata_utils as su

from torch.nn import LSTM
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def PolicyTest(env):
    print('\n########## PolicyTest #################')
    dim_in=(1+env.sp.U)*env.sp.V
    dim_out=4 #(max out-degree)
    dim_hidden=[128]
    qnet=QNetwork(dim_in,dim_out,dim_hidden)
    s = env.reset()
    #print (torch.tensor(s,dtype=torch.float32).to(device))
    epg = EpsilonGreedyPolicyDQN(qnet, env, eps_0=1.0, eps_min=0., eps_cutoff=100)
    a = epg.sample_action(s,env._availableActionsInCurrentState())
    assert not torch.is_tensor(a)
    print('Policy test... [passed]')

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
        print('Memory sampling... [passed]')
    return memory

def FastMemTest(env,capacity=10, num_episodes=3, print_output=True):
    if print_output:
        print('\n########## FastMemoryTest #################')
    #capacity = 10
    s=env.reset()
    assert  env.state_encoding_dim == s.shape[0]
    memory = FastReplayMemory(capacity=10000,tensor_length=env.state_encoding_dim)

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
        print('Memory sampling... [passed]')
    return memory

def SeqMemTest(env, capacity=1000, num_episodes=1100, print_output=True):
    if print_output:
        print('\n########## SeqMemoryTest #################')
    #capacity = 10
    memory = SeqReplayMemory(capacity=capacity)

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
    packed_sequences, actions, rewards, next_states, dones = memory.sample(8)
    #state, action, reward, next_state, done = memory.sample(batch_size)

    lstm=LSTM(input_size=s.shape[0],hidden_size=13,batch_first=True).to(device)
    packed_output, (ht, ct) = lstm(packed_sequences)
    output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
    
    if print_output:
        mem_len       = memory.__len__()
        mem_num_trans = memory.__num_transitions__()
        print('# episodes in memory    :', mem_len)
        print('# transitions in memory :',  mem_num_trans)
        print('Avg #trans. per episode :', mem_num_trans/mem_len)
        print('Memory usage (kb)       : {:.1f}'.format(memory.__memsize__()))
        #print('Sequences tensor shape  :',seq_tensor.shape)
        #print('Sequence tensor lengts  :',seq_lengths)
        print('Packed seq.tensor shape :',packed_sequences.data.shape)
        print('Packed seq. batch sizes :',packed_sequences.batch_sizes)
        print('lstm packed output shape:', packed_output.data.shape)
        print('lstm output tensor shape:',output.shape)
        print('lstm input sizes (check):',input_sizes)
        print('Last h.states, each seq :',ht[-1].shape)
        #print('Non-zero entries:',torch.where(s!=0)[0])
        print('SeqMemory sampling... [passed]')
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
    print('Tensor conversion: [passed]')
    print('State shape:',state.shape)
    print('Torch test... [passed]')

def TorchTest():
    # Let's instantiate and test if it works
    print('\n########## TorchTest #################')
    num_hidden = [128]
    torch.manual_seed(1)
    Q_net = QNetwork(4,2,num_hidden)

    torch.manual_seed(1)
    test_model = nn.Sequential(
        nn.Linear(4, num_hidden[0]), 
        nn.ReLU(), 
        nn.Linear(num_hidden[0], 2)
    )

    x = torch.rand(10, 4)

    # If you do not need backpropagation, wrap the computation in the torch.no_grad() context
    # This saves time and memory, and PyTorch complaints when converting to numpy
    with torch.no_grad():
        assert np.allclose(Q_net(x)[0].numpy(), test_model(x).numpy())
    print('Torch test... [passed]')

def TrainTest(env):
    print('\n########## TrainTest #################')
    replay_buffer = FastMemTest(env,capacity=100, num_episodes=50, print_output=False)
    s=env.reset()
    assert s.shape[0] == env.state_encoding_dim
    # You may want to test your functions individually, but after you do so lets see if the method train works.
    batch_size = 64
    discount_factor = 0.8
    learn_rate = 1e-3
    dim_in=env.state_encoding_dim #(1+env.sp.U)*env.sp.V
    dim_out=4 #(max out-degree)
    dim_hidden=[128]
    qnet=QNetwork(dim_in,dim_out,dim_hidden).to(device)
    qnet_t=QNetwork(dim_in,dim_out,dim_hidden).to(device)
    # Simple gradient descent may take long, so we will use Adam
    optimizer = optim.Adam(qnet.parameters(), learn_rate)
    loss = train(qnet, qnet_t, replay_buffer, optimizer, batch_size, discount_factor)
    print ('Loss: {:.2f}'.format(loss))
    print('Train test... [passed]')

def TestAllDQN():
    configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
    conf=configs['Manhattan5']
    conf['direction_north']=False
    env = GraphWorld(conf, optimization_method='dynamic', fixed_initial_positions=(2,15,19,22),state_representation='etUt', state_encoding='tensors')
    MemTest(env)
    FastMemTest(env)
    SeqMemTest(env)
    TorchTest()
    TensorTest(env)
    PolicyTest(env)
    #TrainTest(env)
    print('\n#######################################\n')

#if __name__ == '__main__':