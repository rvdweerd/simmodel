import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import random
import os
from tqdm import tqdm as _tqdm
import time
import copy
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
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

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        if len(self.memory) >= self.capacity:
            del self.memory[0]
        self.memory.append(transition)

    def sample(self, batch_size):
        assert batch_size <= len(self.memory)
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class SeqReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity # number of sequences allowed
        self.memory = [[] for i in range(self.capacity)]
        self.new_entry=[]
        self.new_s_tensor       =None
        self.new_a_tensor       =None
        self.new_r_tensor       =None
        self.new_sprime_tensor  =None
        self.new_d_tensor       =None
        self.insert_idx = 0
        self.num_filled = 0

    def push(self, transition):
        # transition_sequenceone entry contains: (s,a,r,s',d)
        # 
        self.new_entry.append(transition)
        if type(self.new_s_tensor).__name__ == 'NoneType':
            self.new_s_tensor       =torch.Tensor(transition[0][None,:])#.to(device)
            self.new_a_tensor       =torch.tensor([transition[1]],dtype=torch.int64)#.to(device)
            self.new_r_tensor       =torch.Tensor([transition[2]])#.to(device)
            self.new_sprime_tensor  =torch.Tensor(transition[3][None,:])#.to(device)
            self.new_d_tensor       =torch.Tensor([transition[4]])#.to(device)
        else:
            self.new_s_tensor       =torch.cat((self.new_s_tensor,torch.Tensor(transition[0][None,:])))
            self.new_a_tensor       =torch.cat((self.new_a_tensor,torch.tensor([transition[1]],dtype=torch.int64)))
            self.new_r_tensor       =torch.cat((self.new_r_tensor,torch.Tensor([transition[2]])))
            self.new_sprime_tensor  =torch.cat((self.new_sprime_tensor,torch.Tensor(transition[3][None,:])))
            self.new_d_tensor       =torch.cat((self.new_d_tensor,torch.Tensor([transition[4]])))

        if transition[-1] == True: # Sequence done?
            self.memory[self.insert_idx] = (
                self.new_s_tensor,
                self.new_a_tensor,
                self.new_r_tensor,
                self.new_sprime_tensor,
                self.new_d_tensor,
                [i for i in range(0,len(self.new_a_tensor),1)]
                )
            self.new_s_tensor       = None
            self.new_a_tensor       = None
            self.new_r_tensor       = None
            self.new_sprime_tensor  = None
            self.new_d_tensor       = None

            if self.num_filled < self.capacity:
                self.num_filled+=1
            self.insert_idx= (self.insert_idx + 1) % self.capacity

    def sample(self, batch_size):
        # Insights used from https://github.com/HarshTrivedi/packing-unpacking-pytorch-minimal-tutorial
        seq_tensors=[]
        seq_lengths=[]
        actions=[]
        rewards=[]
        next_states=None
        dones=[]
        assert batch_size <= self.num_filled
        episode_samples = random.sample(self.memory[:self.num_filled], batch_size)
        for ep_sample in episode_samples:
            seq_end_idx = random.choice(ep_sample[5]) # choose sequence length for this saved episode
            seq_lengths.append(seq_end_idx+1)
            seq_tensors.append(ep_sample[0][:seq_end_idx+1,:])
            if type(next_states).__name__ == 'NoneType':
                next_states       = ep_sample[3][seq_end_idx][None,:]#.to(device)
            else:
                next_states       = torch.cat((next_states,ep_sample[3][seq_end_idx][None,:]))
            actions.append(ep_sample[1][seq_end_idx])
            rewards.append(ep_sample[2][seq_end_idx])
            dones.append(ep_sample[4][seq_end_idx])
        #actions=torch.tensor(actions,dtype=torch.int64).to(device)
        actions=torch.tensor(actions)[:,None].to(device)
        rewards=torch.tensor(rewards)[:,None].to(device)
        dones=torch.tensor(dones, dtype=torch.bool).to(device)
        next_states=next_states[:,None,:].to(device) # convert to (bsize, seq_length=1, emb_dim)
        # Put state sequences in a packed datastructure ready to feed an RNN
        seq_lengths = torch.tensor(seq_lengths, dtype = torch.int64,device=device)
        seq_tensors = pad_sequence(seq_tensors, batch_first = True).to(device)
        seq_lengths, indices = torch.sort(seq_lengths, descending=True)
        seq_tensors=seq_tensors[indices]
        packed_input = pack_padded_sequence(seq_tensors, seq_lengths.cpu().numpy(), batch_first=True)
        return packed_input, actions, rewards, next_states, dones

    def __len__(self):
        return self.num_filled

    def __num_transitions__(self):
        count=0
        for e in range(self.num_filled):
            entry = self.memory[e]
            count += len(entry[0])
        return count

    def __memsize__(self):
        # returns memory used in kb
        memsize=0
        for e in range(self.num_filled):
            entry = self.memory[e]
            for i in range(4):
                memsize += entry[i].element_size()*entry[i].nelement()
        return memsize/1000

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
        if self.num_filled<self.capacity:
            self.num_filled+=1

    def sample(self, batch_size):
        assert batch_size <= self.num_filled
        # NOTE: with replacement (CHECK impact!)
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

    def __num_transitions__(self):
        return self.num_filled



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
    Qvals, modelstate = Q(states)
    return torch.gather(Qvals,1,actions), modelstate
    
def compute_targets(Q, rewards, next_states, dones, discount_factor, modelstate=None):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of actions. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
        modelstate=(ht,ct) if recurrent network
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """
    # YOUR CODE HERE
    predictions, modelstate = Q(next_states, modelstate)
    next_qvals = predictions.max(dim=1)[0].unsqueeze(1)
    done_bool = dones.squeeze()#.bool()
    # If we are done (terminal state is reached) next q value is always 0
    next_qvals[done_bool] = 0.
    # Perform the update
    res = rewards + discount_factor * next_qvals
    return res

def train(Q, Q_target, memory, optimizer, batch_size, discount_factor):
    # don't learn without enough experience to replay
    if memory.__len__() < batch_size:
        return 0.

    state, action, reward, next_state, done = memory.sample(batch_size)
    q_val, modelstate = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q_target, reward, next_state, done, discount_factor, modelstate)
    loss = F.smooth_l1_loss(q_val, target)
    #loss = F.mse_loss(q_val, target)

    # backpropagation of loss to Neural Network
    optimizer.zero_grad()
    loss.backward()
    #print('mean grad lstm_hh_10 {:02f}'.format(torch.mean(Q.lstm.weight_hh_l0.grad).cpu().item()),end='')
    #print('mean grad lstm_ih_10 {:02f}'.format(torch.mean(Q.lstm.weight_ih_l0.grad).cpu().item()),end='')
    #print('mean grad lin_layer1 {:02f}'.format(torch.mean(Q.layers[0].weight.grad).cpu().item()),end='')
    #print('mean grad lin_layer2 {:02f}'.format(torch.mean(Q.layers[2].weight.grad).cpu().item()))
    optimizer.step()
    
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

from torch.utils.tensorboard import SummaryWriter
def run_episodes(train, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate, print_every=100,  noise=False, logdir='./temp'):
    writer=writer = SummaryWriter(log_dir=logdir)
    optimizer = optim.Adam(policy.model.parameters(), learn_rate)
    Q_target=copy.deepcopy(policy.model)
    best_model=copy.deepcopy(Q_target).to('cpu')
    #max_return = 0.
    max_return_abs = -1e6
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_lengths = []  
    episode_returns = []
    episode_losses = []
    #best_model_path = None
    start_time=time.time()
    for epi in range(num_episodes):
        if (epi+1)%4000 == 0:
            optimizer.param_groups[0]['lr'] *= 0.9
        state = env.reset() 
        if noise:
            state += np.random.rand(env.state_encoding_dim)/env.state_encoding_dim
        policy.set_epsilon(epi)
        if type(policy).__name__ == 'EpsilonGreedyPolicyDRQN':
            policy.reset_hidden_states()
        steps = 0
        R=0
        while True:
            # Run one episode
            action_idx, next_node = policy.sample_action(state,env._availableActionsInCurrentState())
            s_next, r, done, _ = env.step(action_idx)
            if noise:
                s_next += np.random.rand(env.state_encoding_dim)/env.state_encoding_dim
            memory.push((state,action_idx,r,s_next,done))
            state = s_next
            steps += 1
            global_steps += 1
            R+=r # need to apply discounting!!! CHECK
            # Take a train step
            loss = train(policy.model, Q_target, memory, optimizer, batch_size, discount_factor)
            if done:
                if (epi) % print_every == 0:
                    Q_target.load_state_dict(policy.model.state_dict())
                    duration=time.time()-start_time
                    avg_steps = np.mean(episode_lengths[-print_every:])
                    avg_returns = np.mean(episode_returns[-print_every:])
                    avg_loss = np.mean(episode_losses[-print_every:])
                    start_time=time.time()
                    print("{2} Epi {0}. Avg epi len {1:0.1f}; "
                          .format(epi, avg_steps, '\033[92m' if avg_returns >= 0 else '\033[97m'), end='')
                    print("Avg epi ret:",avg_returns, \
                        "eps: {:.1f}".format(policy.epsilon),\
                        "lr: {:.2E}".format(optimizer.param_groups[0]['lr']),\
                        "Avg loss: {:.2f}".format(avg_loss),\
                        "time/epi (ms) {:.2f}".format(duration/print_every*1000),\
                        "#entr in mem {:.0f}".format(memory.num_filled),\
                        "#trans in mem {:.0f}".format(memory.__num_transitions__())
                        )
                    writer.add_scalar("1. epsilon", policy.epsilon,epi)
                    writer.add_scalar("2. learning_rate", optimizer.param_groups[0]['lr'],epi)
                    writer.add_scalar("3. loss", avg_loss,epi)
                    writer.add_scalar("4. steps_per_epi", avg_steps, epi)
                    writer.add_scalar("5. return_per_epi", avg_returns,epi)

                    # if avg_returns > max_return:
                    #     max_return=avg_returns
                    #     best_model_path='models/dqn_best_model_{:.2f}'.format(max_return)+'.pt'
                    #     torch.save(Q.state_dict(), best_model_path)
                    if avg_returns > max_return_abs:
                        max_return_abs = avg_returns
                        best_model.load_state_dict(policy.model.state_dict())
                episode_lengths.append(steps)
                episode_returns.append(R)
                episode_losses.append(loss)#.detach().item())
                #plot_durations()
                break
        
    print('\033[97m')
    return episode_lengths, episode_returns, episode_losses, best_model

if __name__ == '__main__':
    pass
    