#import torch_geometric as pyg
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import to_networkx, from_networkx
import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from modules.rl.rl_custom_worlds import GetCustomWorld
from modules.gnn.nfm_gen import NFM_ec_t, NFM_ev_t, NFM_ev_ec_t, NFM_ev_ec_t_um_us

world_name='SparseManhattan5x5'
state_repr='etUt'
state_enc='nfm'
nfm_func = NFM_ev_ec_t_um_us()

env = GetCustomWorld(world_name, make_reflexive=True, state_repr=state_repr, state_enc=state_enc)
nfm_funcs = {'NFM_ev_ec_t':NFM_ev_ec_t(),'NFM_ec_t':NFM_ec_t(),'NFM_ev_t':NFM_ev_t(),'NFM_ev_ec_t_um_us':NFM_ev_ec_t_um_us()}
nfm_func = nfm_func
for key in env.sp.G.edges().keys():
    if len(env.sp.G.edges[key]) == 0:
        env.sp.G.edges[key]['N_pref']=-1
        env.sp.G.edges[key]['weight']=-1

env.redefine_nfm(nfm_func)
pyg_graph = from_networkx(env.sp.G)
data=Data(x=env.nfm, edge_index=pyg_graph.edge_index)

class MyDataset(Dataset):
    def __init__(self, root="", transform=None, pre_transform=None, pre_filter=None, datalist=[]):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.datalist=datalist
    
    def get(self,idx):
        return self.datalist[idx]

    def len(self):
        return len(self.datalist)


data_list = [data for i in range(100)]
dataset = MyDataset(datalist=data_list)
loader2 = DataLoader(dataset, batch_size=32,shuffle=True)

loader = DataLoader(data_list, batch_size=32, shuffle=True)
for batch in loader2:
    k=0

class SAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(SAGEConv, self).__init__(aggr='max')
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.act = torch.nn.ReLU()

        self.update_lin = torch.nn.Linear(in_channels + out_channels, in_channels, bias=False)
        self.update_act = torch.nn.ReLU()
        
    def message(self, x_j):
        # x_j has shape [E, in_channels]
        x_j = self.lin(x_j)
        x_j = self.act(x_j)
        return x_j
    
    def update(self, aggr_out, x):
        # aggr_out has shape [N, out_channels]
        new_embedding = torch.cat([aggr_out, x], dim=1)
        new_embedding = self.update_lin(new_embedding)
        new_embedding = self.update_act(new_embedding)
        return new_embedding

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)