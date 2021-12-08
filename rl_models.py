import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class QNetwork(nn.Module):   
    def __init__(self, num_in, num_out, num_hidden=[128]):
        nn.Module.__init__(self)
        layers      = []
        layer_sizes = [num_in]+num_hidden
        for layer_idx in range(1,len(layer_sizes)):
            layers += [ nn.Linear(layer_sizes[layer_idx-1], layer_sizes[layer_idx]), nn.ReLU() ]
        layers     += [ nn.Linear(layer_sizes[-1], num_out) ]
        self.layers = nn.Sequential(*layers)
        self.out_dim= num_out
        self.numTrainableParameters()

    def forward(self, x, modelstate=None):
        return self.layers(x), None
    
    def numTrainableParameters(self):
        print('Qnet size:')
        print('------------------------------------------')
        total = 0
        for name, p in self.named_parameters():
            total += np.prod(p.shape)
            print("{:24s} {:12s} requires_grad={}".format(name, str(list(p.shape)), p.requires_grad))
        print("Total number of parameters: {}".format(total))
        print('------------------------------------------')
        assert total == sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total

class RecurrentQNetwork(nn.Module):   
    def __init__(self, num_in, lstm_hidden, num_out, num_hidden=[128]):
        nn.Module.__init__(self)
        self.lstm   = nn.LSTM(input_size=num_in, hidden_size=lstm_hidden, batch_first=True).to(device)
        layers      = []
        layer_sizes = [lstm_hidden]+num_hidden
        for layer_idx in range(1,len(layer_sizes)):
            layers += [ nn.Linear(layer_sizes[layer_idx-1], layer_sizes[layer_idx]), nn.ReLU() ]
        layers     += [ nn.Linear(layer_sizes[-1], num_out) ]
        self.layers = nn.Sequential(*layers)
        self.out_dim= num_out
        self.lstm.flatten_parameters()
        self.numTrainableParameters()

    def forward(self, x, modelstate=None):
        # x= packed sequence of states
        self.lstm.flatten_parameters()
        if type(x).__name__ == 'PackedSequence': # running lstm for sequences of different lengths
            packed_output, (ht, ct) = self.lstm(x)
            return self.layers(ht[-1]), (ht,ct)
        elif type(x).__name__ == 'Tensor': # running lstm for single sequence
            output, (hn,cn) = self.lstm(x, modelstate)
            return self.layers(hn)[-1], (hn,cn)
        else:
            assert False
    
    def numTrainableParameters(self):
        print('Qnet size:')
        print('------------------------------------------')
        total = 0
        for name, p in self.named_parameters():
            total += np.prod(p.shape)
            print("{:24s} {:12s} requires_grad={}".format(name, str(list(p.shape)), p.requires_grad))
        print("Total number of parameters: {}".format(total))
        print('------------------------------------------')
        assert total == sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total