import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import random
import os
from tqdm import tqdm as _tqdm
import time
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # if you are using multi-GPU.
    np.random.seed(seed)
    # Numpy module.
    random.seed(seed)
    # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

class QNetwork(nn.Module):
    
    def __init__(self, num_in, num_out, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(num_in, num_hidden)
        self.l2 = nn.Linear(num_hidden, num_out)
        self.numTrainableParameters()

    def forward(self, x):
        return self.l2(nn.ReLU()(self.l1(x)))
    
    def numTrainableParameters(self):
        total = 0
        for name, p in self.named_parameters():
            total += np.prod(p.shape)
            print("{:24s} {:12s} requires_grad={}".format(name, str(list(p.shape)), p.requires_grad))
        print("Total number of parameters: {}".format(total))
        assert total == sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total

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
        assert batch_size <= len(self.memory)
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class FastReplayMemory:
    def __init__(self, capacity, tensor_length):
        # invenstory management
        self.capacity = capacity
        self.tensor_length=tensor_length
        self.num_filled=0
        self.insert_index=0
        self.memsize=0
        # memory buffers
        self.state=torch.zeros(capacity,tensor_length,dtype=torch.float).to(device)
        self.memsize += self.state.element_size()*self.state.nelement()
        self.actions=torch.zeros(capacity,dtype=torch.int64).to(device)
        self.memsize += self.actions.element_size()*self.actions.nelement()
        self.rewards=torch.zeros(capacity,dtype=torch.float).to(device)
        self.memsize += self.rewards.element_size()*self.rewards.nelement()
        self.next_state=torch.zeros(capacity,tensor_length,dtype=torch.float).to(device)
        self.memsize += self.next_state.element_size()*self.next_state.nelement()
        self.dones=torch.zeros(capacity,dtype=torch.bool).to(device)
        self.memsize += self.dones.element_size()*self.dones.nelement()
        print('Memory buffer size (kb)',self.memsize/1000)

    def push(self, transition):
        # fill the buffers
        self.state[self.insert_index,:]=torch.tensor(transition[0])
        self.actions[self.insert_index]=transition[1]
        self.rewards[self.insert_index]=transition[2]
        self.next_state[self.insert_index]=torch.tensor(transition[3])
        self.dones[self.insert_index]=transition[4]
        # inventory admin
        self.insert_index = (self.insert_index+1)%self.capacity        
        self.num_filled+=1

    def sample(self, batch_size):
        assert batch_size <= self.num_filled
        # NOTE: with replacement (check impact!)
        indices=torch.randint(self.num_filled,(batch_size,))
        ## Without replacement (two options):
        #indices=torch.multinomial(torch.ones(self.num_filled),batch_size,replacement=False)
        #indices = torch.from_numpy(np.random.choice(self.num_filled,batch_size,replace=False))#.unsqueeze(dim=1)

        return  self.state[indices], \
                self.actions[indices][:,None],\
                self.rewards[indices][:,None],\
                self.next_state[indices], \
                self.dones[indices][:,None],#.unsqueeze(dim=1)
        #return random.sample(self.memory, batch_size)

    def __len__(self):
        return self.num_filled


class EpsilonGreedyPolicyDQN(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, env, eps_0 = 1., eps_min=0.05, eps_cutoff=1000):
        self.Q = Q
        self.epsilon = eps_0
        self.epsilon0   = eps_0
        self.eps_min    = eps_min
        self.eps_cutoff = eps_cutoff
        self.eps_slope  = (eps_0-eps_min)/eps_cutoff
        self.rng = np.random.RandomState(1)
        self.actions_from_node = env.neighbors
        self.out_degree=env.out_degree
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
            num_actions = len(available_actions)
            action_idx = random.randint(0,num_actions-1)
            return action_idx, available_actions[action_idx]
        else:
            with torch.no_grad():
                y = self.Q(torch.tensor(obs,dtype=torch.float32).to(device))
                num_actions=len(available_actions)
                action_idx = torch.argmax(y[:num_actions]).item()
            return action_idx, None
        
    def set_epsilon(self, global_steps):
        if self.eps_cutoff > 0:
            if global_steps > self.eps_cutoff:
                self.epsilon = self.eps_min
            else:
                self.epsilon = self.epsilon0 - self.eps_slope * global_steps
        #else:
        #   self.epsilon = self.epsilon0

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """
    # YOUR CODE HERE
    #states = torch.tensor(states,dtype=torch.float32)
    #actions = torch.tensor(actions,dtype=torch.int64)
    Qvals = Q(states)
    return torch.gather(Qvals,1,actions)
    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """
    # YOUR CODE HERE
    next_qvals = Q(next_states).max(dim=1)[0].unsqueeze(1)
    done_bool = dones.squeeze()#.bool()
    # If we are done (terminal state is reached) next q value is always 0
    next_qvals[done_bool] = 0.
    # Perform the update
    res = rewards + discount_factor * next_qvals
    return res

def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if memory.__len__() < batch_size:
        return None

    state, action, reward, next_state, done = memory.sample(batch_size)
    # # random transition batch is taken from experience replay memory
    # transitions = memory.sample(batch_size)
    # # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    # state, action, reward, next_state, done = zip(*transitions)
    # state=np.array(state)
    # action=np.array(action)
    # reward=np.array(reward)
    # next_state=np.array(next_state)
    # done=np.array(done)
    # # convert to PyTorch and define types
    # state = torch.tensor(state, dtype=torch.float).to(device)
    # action = torch.tensor(action, dtype=torch.int64)[:, None].to(device)  # Need 64 bit to use them as index
    # next_state = torch.tensor(next_state, dtype=torch.float).to(device)
    # reward = torch.tensor(reward, dtype=torch.float)[:, None].to(device)
    # done = torch.tensor(done, dtype=torch.uint8)[:, None].to(device)  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)
    #loss = F.mse_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())


def run_episodes(train, Q, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate, noise=False):
    print_every=10
    optimizer = optim.Adam(Q.parameters(), learn_rate)
    
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  
    episode_returns = []
    losses = []
    start_time=time.time()
    for i in range(num_episodes):
        state = env.reset() 
        if noise:
            state += np.random.rand(100)/200.
        
        steps = 0
        R=0
        while True:
            # Run one episode
            policy.set_epsilon(global_steps)
            action_idx, next_node = policy.sample_action(state,env._availableActionsInCurrentState())
            s_next, r, done, _ = env.step(action_idx)
            if noise:
                s_next += np.random.rand(100)/200.
            memory.push((state,action_idx,r,s_next,done))
            state = s_next
            steps += 1
            global_steps += 1
            R+=r # need to apply discounting!!!
            # Take a train step
            loss = train(Q, memory, optimizer, batch_size, discount_factor)
            if done:
                if (i) % print_every == 0:
                    duration=time.time()-start_time
                    start_time=time.time()
                    print("{2} Episode {0} finished after {1} steps. "
                          .format(i, steps, '\033[92m' if steps >= 195 else '\033[99m'), end='')
                    print("Last episode return:",R, "epsilon {:.2f}".format(policy.epsilon), "time per episode(ms) {:.2f}".format(duration/print_every*1000))
                episode_durations.append(steps)
                episode_returns.append(R)
                losses.append(loss)#.detach().item())
                #plot_durations()
                break
        
    return episode_durations, episode_returns, losses

if __name__ == '__main__':
    pass
    