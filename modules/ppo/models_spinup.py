#import numpy as np
from gym.spaces import Box, Discrete
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import numpy as np

class Struc2Vec(nn.Module):
    def __init__(self, emb_dim, emb_iter_T, node_dim):
        super().__init__()
        self.emb_dim        = emb_dim
        self.T              = emb_iter_T
        self.node_dim       = node_dim

        self.theta1a    = nn.Linear(self.node_dim, self.emb_dim, True)#, dtype=torch.float32)
        self.theta1b    = nn.Linear(self.emb_dim, self.emb_dim, True)#, dtype=torch.float32)
        self.theta2     = nn.Linear(self.emb_dim, self.emb_dim, True)#, dtype=torch.float32)
        self.theta3     = nn.Linear(self.emb_dim, self.emb_dim, True)#, dtype=torch.float32)
        self.theta4     = nn.Linear(1, self.emb_dim, True)#, dtype=torch.float32)
        self.theta6     = nn.Linear(self.emb_dim, self.emb_dim, True)#, dtype=torch.float32)
        self.theta7     = nn.Linear(self.emb_dim, self.emb_dim, True)#, dtype=torch.float32)
        print("PARAMETERS Struc2Vec core")
        self.numTrainableParameters()

    def forward(self, xv, Ws):
        # xv: The node features (batch_size, num_nodes, node_dim)
        # Ws: The graphs (batch_size, num_nodes, num_nodes)
        num_nodes = xv.shape[1]
        batch_size = xv.shape[0]
        
        # pre-compute 1-0 connection matrices masks (batch_size, num_nodes, num_nodes)
        #conn_matrices = torch.where(Ws > 0, torch.ones_like(Ws), torch.zeros_like(Ws)).to(device)
        conn_matrices = Ws # we have only edge weights of 1

        # Graph embedding
        # Note: we first compute s1 and s3 once, as they are not dependent on mu
        mu = torch.zeros(batch_size, num_nodes, self.emb_dim, dtype=torch.float32)#,device=device)
        #s1 = self.theta1a(xv)  # (batch_size, num_nodes, emb_dim)
        s1 = self.theta1b(F.relu(self.theta1a(xv)))  # (batch_size, num_nodes, emb_dim)
        #for layer in self.theta1_extras:
        #    s1 = layer(F.relu(s1))  # we apply the extra layer
        
        s3_1 = F.relu(self.theta4(Ws.unsqueeze(3)))  # (batch_size, nr_nodes, nr_nodes, emb_dim) - each "weigth" is a p-dim vector        
        s3_2 = torch.sum(s3_1, dim=1)  # (batch_size, nr_nodes, emb_dim) - the embedding for each node
        s3 = self.theta3(s3_2)  # (batch_size, nr_nodes, emb_dim)
        
        for t in range(self.T):
            s2 = self.theta2(conn_matrices.matmul(mu))    
            mu = F.relu(s1 + s2 + s3) # (batch_size, nr_nodes, emb_dim)

        # we repeat the global state (summed over nodes) for each node, 
        # in order to concatenate it to local states later
        global_state = self.theta6(torch.sum(mu, dim=1, keepdim=True).repeat(1, num_nodes, 1))
        local_action = self.theta7(mu)  # (batch_dim, nr_nodes, emb_dim)
        out = F.relu(torch.cat([global_state, local_action], dim=2)) # concat creates (batch_dim, nr_nodes, 2*emb_dim)
        return out
    def numTrainableParameters(self):
        print('Struc2Vec core size:')
        print('------------------------------------------')
        total = 0
        for name, p in self.named_parameters():
            total += np.prod(p.shape)
            print("{:24s} {:12s} requires_grad={}".format(name, str(list(p.shape)), p.requires_grad))
        print("Total number of parameters: {}".format(total))
        print('------------------------------------------')
        assert total == sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total

class Stuc2Vec_policynet(nn.Module):
    def __init__(self, struc2vec, emb_dim):
        super().__init__()
        self.struc2vec = struc2vec
        self.emb_dim = emb_dim
        self.theta5_pi  = nn.Linear(2*self.emb_dim, 1, True)#, dtype=torch.float32)
        #self.theta5_pi2 = nn.Linear(self.emb_dim, 1, True, dtype=torch.float32) # Maybe too complex, perhaps share weights with th_5_pi CHECK / TODO 

        print("PARAMETERS pi Policynet")
        self.numTrainableParameters()

    def forward(self, X, actions=None):
        if len(X.shape)==2:
            X=X.unsqueeze(dim=0)
        # X (bsize,num_nodes,F+V+1)
        
        num_nodes = X.shape[1]
        node_dim = X.shape[2]-1-num_nodes
        nfm, W, reachable_nodes = torch.split(X,[node_dim, num_nodes, 1],2)
        rep = self.struc2vec(nfm, W)
        prob_logits = self.theta5_pi(rep).squeeze(dim=2) # (batch_dim, nr_nodes)
        # mask invalid actions
        reachable_nodes=reachable_nodes.squeeze(dim=2).type(torch.BoolTensor)
        prob_logits[~reachable_nodes] = -1e20#float("Inf")#-1.e12#torch.inf
        D=Categorical(logits=prob_logits)

        if type(actions) == type(None):
            action_probs = None
        else:
            logprobs = D.logits
            action_probs = torch.gather(logprobs,1,actions.unsqueeze(dim=1).to(dtype=torch.int64))

        return D, action_probs
    def numTrainableParameters(self):
        print('pi size:')
        print('------------------------------------------')
        total = 0
        for name, p in self.named_parameters():
            total += np.prod(p.shape)
            print("{:24s} {:12s} requires_grad={}".format(name, str(list(p.shape)), p.requires_grad))
        print("Total number of parameters: {}".format(total))
        print('------------------------------------------')
        assert total == sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total

class Stuc2Vec_valuenet(nn.Module):
    def __init__(self, struc2vec, emb_dim):
        super().__init__()
        self.struc2vec = struc2vec
        self.emb_dim = emb_dim
        self.theta5_v1  = nn.Linear(2*self.emb_dim, 1, True)#, dtype=torch.float32)
        #self.theta5_v2  = nn.Linear(self.emb_dim, 1, True, dtype=torch.float32) # Maybe too complex, perhaps share weights with th_5_pi CHECK / TODO

        print("PARAMETERS v Valuenet")
        self.numTrainableParameters()

    def forward(self, X):
        if len(X.shape)==2:
            X=X.unsqueeze(dim=0)
        # X (bsize,num_nodes,F+V+1)
        
        num_nodes = X.shape[1]
        node_dim = X.shape[2]-1-num_nodes
        nfm, W, reachable_nodes = torch.split(X,[node_dim, num_nodes, 1],2)
        rep = self.struc2vec(nfm, W)
        rep = self.theta5_v1(rep).squeeze(dim=2) # (batch_dim, nr_nodes)
        v = rep.mean(dim=1)      
        return v # flat (batch_dim)
    def numTrainableParameters(self):
        print('v size:')
        print('------------------------------------------')
        total = 0
        for name, p in self.named_parameters():
            total += np.prod(p.shape)
            print("{:24s} {:12s} requires_grad={}".format(name, str(list(p.shape)), p.requires_grad))
        print("Total number of parameters: {}".format(total))
        print('------------------------------------------')
        assert total == sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total

class Struc2VecActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, emb_dim=8, emb_iter_T=3):
        super().__init__()
        self.emb_dim        = emb_dim
        self.emb_iter_T   = emb_iter_T
        obs_dim = observation_space.shape
        self.node_dim       = obs_dim[1]-obs_dim[0]-1 # dimension of original node features
        self.struc2vec1 = Struc2Vec(self.emb_dim, self.emb_iter_T, self.node_dim)
        self.struc2vec2 = Struc2Vec(self.emb_dim, self.emb_iter_T, self.node_dim)
        
        self.pi = Stuc2Vec_policynet(self.struc2vec1, emb_dim=self.emb_dim)
        self.v  = Stuc2Vec_valuenet(self.struc2vec2,  emb_dim=self.emb_dim)

        print("PARAMETERS Struc2VecActorCritic")
        self.numTrainableParameters()

    def step(self,obs):
        if len(obs.shape)==2:
            obs=obs.unsqueeze(dim=0)
        # X: obs will be (bsize,num_nodes,F+V+1)
        #nfm, W, reachable_nodes = torch.split(obs,[self.node_dim, self.num_nodes, 1],2)
        with torch.no_grad():
            distro, _ = self.pi(obs)
            a = distro.sample()
            logp_a = torch.gather(distro.logits, 1, a.unsqueeze(dim=1))
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        with torch.no_grad():
            distro, _ = self.pi(obs)
            a = distro.sample()
        return a

    def numTrainableParameters(self):
        print('Struc2Vec size:')
        print('------------------------------------------')
        total = 0
        for name, p in self.named_parameters():
            total += np.prod(p.shape)
            print("{:24s} {:12s} requires_grad={}".format(name, str(list(p.shape)), p.requires_grad))
        print("Total number of parameters: {}".format(total))
        print('------------------------------------------')
        assert total == sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total


