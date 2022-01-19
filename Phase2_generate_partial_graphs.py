import argparse
from modules.optim.simdata_create import GetDatabankForPartialGraph
from modules.rl.environments import GraphWorld, GraphWorldFromDatabank
from modules.rl.rl_utils import EvaluatePolicy
from modules.rl.rl_policy import EpsilonGreedyPolicy
from modules.rl.rl_plotting import PlotPerformanceCharts
from modules.rl.rl_algorithms import q_learning_exhaustive
from modules.sim.graph_factory import get_all_edge_removals_symmetric
from modules.sim.simdata_utils import SimulateInteractiveMode
import numpy as np
import networkx as nx
import random
import pickle
import tqdm

def RunInstance(args):
    # Genertes all partial graphs with number of edges removed as defined in the argument args.num_edges.
    # For each partial graph, calculate pusuer paths for all initial positions and save as a datafile.
    # Only implemented for Manhattan3x3 experiment
    config={
        'graph_type': "Manhattan",
        'make_reflexive': True,
        'N': 3,    # number of nodes along one side
        'U': 2,    # number of pursuer units
        'L': 4,    # Time steps
        'T': 7,
        'R': 100,  # Number of escape routes sampled 
        'direction_north': False,       # Directional preference of escaper
        'loadAllStartingPositions': False
    }
    state_repr='et'
    state_enc='nodes'
    env = GraphWorld(config, optimization_method='static', fixed_initial_positions=None, state_representation=state_repr, state_encoding=state_enc)
    #world_name='MetroU3_e17tborder_FixedEscapeInit'
    #world_name='Manhattan5x5_FixedEscapeInit'
    #env = GetCustomWorld(world_name, make_reflexive=True, state_repr=state_repr, state_enc='nodes')

    W_all, W_per_num_edge_removals = get_all_edge_removals_symmetric(
            W_          = nx.convert_matrix.to_numpy_matrix(env.sp.G),
            start_node  = env.sp.labels2nodeids[env.state[0]],
            target_nodes= [env.sp.labels2nodeids[i] for i in env.sp.target_nodes],
            removals    = [8,12,16],
            instances_per_num_removed = 2    
        )
    for k,v in W_per_num_edge_removals.items():
        # v: list of tuples
        print(k,len(v))
    out_file = open("./datasets/_partial_graphs/Manhattan_N=3,L=4,R=100,Ndir=False/partial_graph_register","wb")
    pickle.dump(W_per_num_edge_removals, out_file)
    out_file.close()


    # Initialize the dataset generator
    REQUIRED_DATASET_SIZE=100
    UPDATE_EVERY = 100

    all_databanks={}
    #env.render(mode=None, fname='graph')
    Us=[2,1]
    for U in Us:
        env.sp.U=U
        #for edges_removed, partial_graphs in W_per_num_edge_removals.items():
        edges_removed=args.num_edges
        partial_graphs=W_per_num_edge_removals[edges_removed]

        databank_per_hash = {}
        num_entries=[]
        for entry in tqdm.tqdm(partial_graphs):
            W_partial = entry[0]
            hashint   = entry[1]
            hashstr   = entry[2]
            env.redefine_graph_structure(W_partial, env.sp.nodeid2coord)
            register, databank, iratios = GetDatabankForPartialGraph(env.sp, REQUIRED_DATASET_SIZE, UPDATE_EVERY)
            num_entries.append(len(iratios))
            databank_per_hash[hashint] = {'register':register, 'databank':databank, 'iratios':iratios}
        print('Finished for '+str(edges_removed)+' edges removed; ',
                'number of databanks:', len(databank_per_hash),
                '; avg size of databank per hash:',sum(num_entries)/len(num_entries)
                )
        all_databanks[U] = databank_per_hash
        print('Finished for U=',U,'size of databank per hash:', len(all_databanks[U]))

    out_file = open("./datasets/_partial_graphs/Manhattan_N=3,L=4,R=100,Ndir=False/_databanks_num_edg_rem="+str(edges_removed),"wb")
    pickle.dump(all_databanks, out_file)
    out_file.close()

def SaveSolvabilityData(args):
    in_file=open("./datasets/_partial_graphs/Manhattan_N=3,L=4,R=100,Ndir=False/_databank_full","rb")
    databank_full=pickle.load(in_file)
    in_file.close()
    in_file=open("./datasets/_partial_graphs/Manhattan_N=3,L=4,R=100,Ndir=False/_partial_graph_register","rb")
    partial_graph_register=pickle.load(in_file)
    in_file.close()

    solvable_dict={}
    reachable_by_pursuers_dict={}
    
    num_edges=args.num_edges
    U=args.U
    solvable_dict['U='+str(U)]={}
    config={
        'graph_type': "Manhattan",
        'make_reflexive': True,
        'N': 3,    # number of nodes along one side
        'U': U,    # number of pursuer units
        'L': 4,    # Time steps
        'T': 7,
        'R': 100,  # Number of escape routes sampled 
        'direction_north': False,       # Directional preference of escaper
        'loadAllStartingPositions': False
    }
    state_repr='etUt'
    state_enc='nodes'
    #W_, hashint, hashstr = random.choice(partial_graph_register[3])
    
    for W_, hashint, hashstr in tqdm.tqdm(partial_graph_register[num_edges]):
        env_data={'W':W_, 'hashint':hashint, 'databank_full':databank_full}
        env = GraphWorldFromDatabank(config, env_data, optimization_method='static', fixed_initial_positions=None, state_representation=state_repr, state_encoding=state_enc)
        solvable=IsSolvable(env)
        solvable_dict['U='+str(U)][hashint]=solvable
    out_file = open("./datasets/_partial_graphs/Manhattan_N=3,L=4,R=100,Ndir=False/solvable_U="+str(U)+"_e="+str(num_edges),"wb")
    pickle.dump(solvable_dict, out_file)
    out_file.close()
    #SimulateInteractiveMode(env)

def MergeDataFiles():
    merged_databank={'U=1':{},'U=2':{}}
    for i in range(7):
        in_file=open("./datasets/_partial_graphs/Manhattan_N=3,L=4,R=100,Ndir=False/databanks_num_edg_rem="+str(i),"rb")
        databanks_i=pickle.load(in_file)
        in_file.close()
        for k,v in databanks_i.items():
            for k2,v2 in v.items():
                #print(k2,v2)
                merged_databank['U='+str(k)][k2]=v2
    out_file = open("./datasets/_partial_graphs/Manhattan_N=3,L=4,R=100,Ndir=False/_databank_full","wb")
    pickle.dump(merged_databank, out_file)
    out_file.close()

def IsSolvable(env):
    # Runs single tabQL algo to see if target area is reachable by escape agent without getting caught
    logdir='./datasets/_partial_graphs/Manhattan_N=3,L=4,R=100,Ndir=False'
    num_seeds   = 1
    eps_0       = 1.
    eps_min     = 0.1
    num_iter    = 1000
    gamma       = .9
    alpha_0     = .2
    alpha_decay = 0.
    initial_Q_values = 10.
    
    policy = EpsilonGreedyPolicy(env, eps_0, eps_min, initial_Q_values)
    
    # Learn the policy
    metrics_episode_returns = {}
    metrics_episode_lengths = {}
    metrics_avgperstep = {}
    Q_tables = {}

    algos  = [q_learning_exhaustive]
    for algo in algos:
        metrics_all = np.zeros((num_seeds,2,num_iter*len(env.world_pool)))
        for s in range(num_seeds):
            policy.reset_epsilon()
            Q_table, metrics_singleseed, policy, _ = algo(env, policy, num_iter, discount_factor=gamma, alpha_0=alpha_0, alpha_decay=alpha_decay,print_episodes=False)
            metrics_all[s] = metrics_singleseed
            print('entries in Q table:',len(Q_table))
        
        Q_tables[algo.__name__] = Q_table
        metrics_episode_returns[algo.__name__] = metrics_all[:, 0, :]
        metrics_episode_lengths[algo.__name__] = metrics_all[:, 1, :]
        metrics_avgperstep[algo.__name__] = np.sum(
            metrics_episode_returns[algo.__name__], axis=0)/np.sum(metrics_episode_lengths[algo.__name__], axis=0)
    performance_metrics = { 'e_returns': metrics_episode_returns, 'e_lengths':metrics_episode_lengths, 'rps':metrics_avgperstep}

    # Evaluate the learned policy
    policy.epsilon=0.
    _, returns, _ = EvaluatePolicy(env,policy,env.world_pool,print_runs=False, save_plots=False, logdir=logdir, has_Q_table=True)
    return np.array(returns)>0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)    

    # Model hyperparameters
    parser.add_argument('--num_edges', default=0, type=int, 
                        help='Number of edges to be removed from default graph')
    parser.add_argument('--U', default=0, type=int, 
                    help='Number of units')


    args=parser.parse_args()
    #RunInstance(args)
    #MergeDataFiles()
    SaveSolvabilityData(args)
