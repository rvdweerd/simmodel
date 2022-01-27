import copy
import argparse
from modules.optim.optimization_FIP_gurobipy import optimization_alt
from modules.optim.simdata_create import GetDatabankForPartialGraph
from modules.rl.environments import GraphWorld#, GraphWorldFromDatabank
from modules.rl.rl_utils import EvaluatePolicy
from modules.rl.rl_policy import EpsilonGreedyPolicy
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
    out_file = open("./datasets/__partial_graphs/Manhattan_N=3,L=4,R=100,Ndir=False/partial_graph_register","wb")
    pickle.dump(W_per_num_edge_removals, out_file)
    out_file.close()


    # Initialize the dataset generator
    REQUIRED_DATASET_SIZE=100
    UPDATE_EVERY = 100

    all_databanks={}
    env0=copy.deepcopy(env)
    #env.render(mode=None, fname='graph')
    Us=[1,2,3]
    for U in Us:
        env.sp.U=U
        env0.sp.U=U
        #for edges_removed, partial_graphs in W_per_num_edge_removals.items():
        edges_removed=args.num_edges
        partial_graphs=W_per_num_edge_removals[edges_removed]

        databank_per_hash = {}
        num_entries=[]
        for entry in tqdm.tqdm(partial_graphs):
            W_partial = entry[0]
            hashint   = entry[1]
            hashstr   = entry[2]

            # remove components that are not connected to the initial escape position
            H = nx.from_numpy_matrix(W_partial, create_using=nx.DiGraph()).to_undirected()
            H = nx.relabel_nodes(H, env.sp.nodeid2coord)
            S = [H.subgraph(c).copy() for c in nx.connected_components(H)]                  
            nodeid2coord_new={}
            valid=True
            for s in S:
                if (1,0) in s.nodes():
                    W_new=nx.convert_matrix.to_numpy_matrix(s.to_directed())
                    for j, coord in enumerate(s.nodes()):
                        nodeid2coord_new[j]=coord
                else:
                    if len(s.nodes())>1:
                        valid=False
            if valid:
                env.redefine_graph_structure(W_new, nodeid2coord_new, new_nodeids=True)
                register, databank, iratios = GetDatabankForPartialGraph(env.sp, REQUIRED_DATASET_SIZE, UPDATE_EVERY)
                num_entries.append(len(iratios))
                databank_per_hash[hashint] = {'register':register, 'databank':databank, 'iratios':iratios, 'W':W_new, 'nodeid2coord':nodeid2coord_new}
                env=copy.deepcopy(env0)

        print('Finished for '+str(edges_removed)+' edges removed; ',
                'number of databanks:', len(databank_per_hash),
                '; avg size of databank per hash:',sum(num_entries)/len(num_entries)
                )
        all_databanks[U] = databank_per_hash
        print('Finished for U=',U,'size of databank per hash:', len(all_databanks[U]))

    out_file = open("./datasets/__partial_graphs/Manhattan_N=3,L=4,R=100,Ndir=False/_databanks_num_edg_rem="+str(edges_removed),"wb")
    pickle.dump(all_databanks, out_file)
    out_file.close()

def SaveSolvabilityData(args):
    in_file=open("./datasets/__partial_graphs/Manhattan_N=3,L=4,R=100,Ndir=False/_databank_full","rb")
    databank_full=pickle.load(in_file)
    in_file.close()
    in_file=open("./datasets/__partial_graphs/Manhattan_N=3,L=4,R=100,Ndir=False/_partial_graph_register","rb")
    partial_graph_register=pickle.load(in_file)
    in_file.close()

    solvable_dict={}
     
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
    env0 = GraphWorld(config, optimization_method='static', fixed_initial_positions=None, state_representation=state_repr, state_encoding=state_enc)
    for W_, hashint, hashstr in tqdm.tqdm(partial_graph_register[num_edges]):
        #env_data={'W':W_, 'hashint':hashint, 'databank_full':databank_full}
        if hashint not in databank_full['U='+str(U)]: continue
        env=copy.deepcopy(env0)
        env_data=databank_full['U='+str(U)][hashint]
        env.redefine_graph_structure(env_data['W'],env_data['nodeid2coord'],new_nodeids=True)
        env.reload_unit_paths(env_data['register'],env_data['databank'],env_data['iratios'])
        solvable=IsSolvable(env)
        solvable_dict['U='+str(U)][hashint]=solvable
    out_file = open("./datasets/__partial_graphs/Manhattan_N=3,L=4,R=100,Ndir=False/solvable_U="+str(U)+"_e="+str(num_edges),"wb")
    pickle.dump(solvable_dict, out_file)
    out_file.close()
    #SimulateInteractiveMode(env)

def SaveReachabilityData():
    in_file=open("./datasets/__partial_graphs/Manhattan_N=3,L=4,R=100,Ndir=False/_databank_full","rb")
    databank_full=pickle.load(in_file)
    in_file.close()
    in_file=open("./datasets/__partial_graphs/Manhattan_N=3,L=4,R=100,Ndir=False/_partial_graph_register","rb")
    partial_graph_register=pickle.load(in_file)
    in_file.close()

    reachable_by_pursuers_dict={}
    
    for U in [1,2,3]:
        reachable_by_pursuers_dict['U='+str(U)]={}
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
        
        for num_edges in range(7):
            for W_, hashint, hashstr in partial_graph_register[num_edges]:
                env_data=databank_full['U='+str(U)][hashint]
                env_data['W']=W_
                #env_data={'W':W_, 'hashint':hashint, 'databank_full':databank_full}
                env = GraphWorldFromDatabank(config, env_data, optimization_method='static', fixed_initial_positions=None, state_representation=state_repr, state_encoding=state_enc)
                reachable = IsReachable(env)
                reachable_by_pursuers_dict['U='+str(U)][hashint]=reachable
    out_file = open("./datasets/__partial_graphs/Manhattan_N=3,L=4,R=100,Ndir=False/_reachable_by_pursuers","wb")
    pickle.dump(reachable_by_pursuers_dict, out_file)
    out_file.close()
    #SimulateInteractiveMode(env)

def MergeDataFiles():
    merged_databank={'U=1':{},'U=2':{}, 'U=3':{}}
    #in_file=open('./datasets/__partial_graphs/Manhattan_N=3,L=4,R=100,Ndir=False/_databank_full','rb')
    #merged_databank=pickle.load(in_file)
    #in_file.close()
    in_file=open("./datasets/__partial_graphs/Manhattan_N=3,L=4,R=100,Ndir=False/_partial_graph_register","rb")
    partial_graph_register_old=pickle.load(in_file)
    partial_graph_register_new={i:[] for i in range(11)}
    in_file.close()

    for i in range(11):
        in_file=open("./datasets/__partial_graphs/Manhattan_N=3,L=4,R=100,Ndir=False/_databanks_num_edg_rem="+str(i),"rb")
        databanks_i=pickle.load(in_file)
        in_file.close()
        for num_u, hash2data in databanks_i.items():
            for hashint, data in hash2data.items():
                if len(data['databank'][0]['paths']) != num_u:
                    assert False
                merged_databank['U='+str(num_u)][hashint]=data
    out_file = open("./datasets/__partial_graphs/Manhattan_N=3,L=4,R=100,Ndir=False/_databank_full","wb")
    pickle.dump(merged_databank, out_file)
    out_file.close()
    for u in [1,2,3]:
        for e in range(11):
            for W_, hashint, hashstr in partial_graph_register_old[e]:
                if hashint in merged_databank['U='+str(u)]:
                    partial_graph_register_new[e].append((W_,hashint,hashstr))
    out_file=open("./datasets/__partial_graphs/Manhattan_N=3,L=4,R=100,Ndir=False/_partial_graph_register_trimmed","wb")
    pickle.dump(partial_graph_register_new, out_file)
    out_file.close()



def MergeDataFilesSolvability():
    solvable_global = {'U=1':{},'U=2':{}, 'U=3':{}}
    #in_file=open('./datasets/__partial_graphs/Manhattan_N=3,L=4,R=100,Ndir=False/_solvable','rb')
    #solvable_global=pickle.load(in_file)
    #in_file.close()

    for e in range(11):
        for U in [1,2,3]:
            in_file=open("./datasets/__partial_graphs/Manhattan_N=3,L=4,R=100,Ndir=False/solvable_U="+str(U)+"_e="+str(e),"rb")
            solvable_local = pickle.load(in_file)
            in_file.close()
            for k,v in solvable_local.items():
                for k2,v2 in v.items():
                    #print(k2,v2)
                    solvable_global['U='+str(U)][k2]=v2
    out_file = open("./datasets/__partial_graphs/Manhattan_N=3,L=4,R=100,Ndir=False/_solvable","wb")
    pickle.dump(solvable_global, out_file)
    out_file.close()

def IsSolvable(env):
    # Runs single tabQL algo to see if target area is reachable by escape agent without getting caught
    logdir='./datasets/__partial_graphs/Manhattan_N=3,L=4,R=100,Ndir=False'
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

def IsReachable(env):
    # Checks if E is reachable by Pursuers on the given graph instance
    reachable=[]
    for i, entry in enumerate(env.world_pool):
        s = env.reset(entry = entry)
        epos=s[0]
        can_reach=False
        for target in s[1:]:
            if nx.algorithms.shortest_paths.generic.has_path(env.sp.G, env.sp.labels2coord[epos], env.sp.labels2coord[target]):
                can_reach=True
                break
        reachable.append(can_reach)
        if not can_reach:
            k=0
    return np.array(reachable)

def LoadData():
    in_file=open("./datasets/__partial_graphs/Manhattan_N=3,L=4,R=100,Ndir=False/_databank_full","rb")
    databank_full=pickle.load(in_file)
    in_file.close()
    in_file=open("./datasets/__partial_graphs/Manhattan_N=3,L=4,R=100,Ndir=False/_partial_graph_register_trimmed","rb")
    partial_graph_register=pickle.load(in_file)
    in_file.close()
    #in_file=open("./datasets/__partial_graphs/Manhattan_N=3,L=4,R=100,Ndir=False/_reachable_by_pursuers","rb")
    #reachable_by_pursuers=pickle.load(in_file)
    #in_file.close()
    in_file=open("./datasets/__partial_graphs/Manhattan_N=3,L=4,R=100,Ndir=False/_solvable_edge_crossing","rb")
    solvable=pickle.load(in_file)
    in_file.close()
    return databank_full, partial_graph_register, solvable#, reachable_by_pursuers

def GetConfig(u=2):
    config={
        'graph_type': "Manhattan",
        'make_reflexive': True,
        'N': 3,    # number of nodes along one side
        'U': u,    # number of pursuer units
        'L': 4,    # Time steps
        'T': 7,
        'R': 100,  # Number of escape routes sampled 
        'direction_north': False,       # Directional preference of escaper
        'loadAllStartingPositions': False
    }
    return config

def TestSim(e=6,u=2):
    in_file=open("./datasets/__partial_graphs/Manhattan_N=3,L=4,R=100,Ndir=False/_databanks_num_edg_rem="+str(e),"rb")
    databank_full=pickle.load(in_file)
    in_file.close()
    in_file=open("./datasets/__partial_graphs/Manhattan_N=3,L=4,R=100,Ndir=False/_partial_graph_register","rb")
    partial_graph_register=pickle.load(in_file)
    in_file.close()
    config=GetConfig(u=2)
    state_repr='etUt'
    state_enc='nodes'
    env = GraphWorld(config, optimization_method='static', fixed_initial_positions=None, state_representation=state_repr, state_encoding=state_enc)

    while True:
        W_, hashint, hashstr = random.choice(partial_graph_register[e])
        if hashint in databank_full[u]: break
    env_data=databank_full[u][3177]
    env.redefine_graph_structure(env_data['W'],env_data['nodeid2coord'],new_nodeids=True)
    env.reload_unit_paths(env_data['register'],env_data['databank'],env_data['iratios'])
    
    env.reset(12)
    SimulateInteractiveMode(env)

def TestInteractiveSimulation(U=[2],E=[8]):
    config=GetConfig(u=2)#only needed to find datafile
    state_repr = 'etUt'
    state_enc  = 'tensors'
    databank_full, register_full, solvable = LoadData()
    env0 = GraphWorld(config, optimization_method='static', fixed_initial_positions=None, state_representation=state_repr, state_encoding=state_enc)

    all_envs=[]
    hashint2env={}
    for u in U:
        env0.sp.U = u
        for e in E:#range(11):
            for W_, hashint, hashstr in register_full[e]:
            #W_, hashint, hashstr = random.choice(register_full[4])
            #for hashint, env_data in databank_full['U='+str(u)].items():
                if hashint not in databank_full['U='+str(u)]:
                    assert False
                env_data = databank_full['U='+str(u)][hashint] # dict contains  'register':{(e0,U0):index}, 'databank':[], 'iratios':[]
                for entry in env_data['databank']:
                    if len(entry['paths']) < u:
                        assert False
                env=copy.deepcopy(env0)
                env.redefine_graph_structure(env_data['W'],env_data['nodeid2coord'],new_nodeids=True)
                env.reload_unit_paths(env_data['register'],env_data['databank'],env_data['iratios'])
                s = solvable['U='+str(u)][hashint]
                valids = s#np.logical_and(np.logical_not(s),r)
                if valids.sum() > 0:

                    env.world_pool = list(np.array(env.all_worlds)[valids])
                    all_envs.append(env)
                    hashint2env[hashint]=len(all_envs)-1
    while True:
        env=random.choice(all_envs) 
        s=env.reset()
        if has_duplicates(env.state[1:]):
            continue
        SimulateInteractiveMode(env)

def has_duplicates(arr):
    dups=False
    mem=set()
    for i in arr:
        if i in mem:
            dups=True
            break
        mem.add(i)
    return dups

def CalculateStatistics():
    config=GetConfig()
    state_repr = 'etUt'
    state_enc  = 'nodes'
    databank_full, register_full, solvable, reachable = LoadData()
    
    for e in [0,1,2,3,4,5,6]:
        for U in [1,2,3]:
            S=[]
            R=[]
            graphcount=0
            env_all=[]
            for W_, hashint, hashstr in register_full[e]:
                #all_envs=[]
                s = solvable['U='+str(U)][hashint]
                r = reachable['U='+str(U)][hashint]
                S+=list(s)
                R+=list(r)
                graphcount+=1
                env_data = databank_full['U='+str(U)][hashint] # dict contains  'register':{(e0,U0):index}, 'databank':[], 'iratios':[]
                env_data['W'] = W_
                env = GraphWorldFromDatabank(config,env_data,optimization_method='static',state_representation=state_repr,state_encoding=state_enc)
                valids = np.logical_and(s,r)
                if valids.sum() > 0:
                    env.world_pool = list(np.array(env.all_worlds)[valids])
                    env_all.append(env)
                Total = len(S)
                SandR = np.logical_and(np.array(S),np.array(R)).sum()
            print('---------------')
            print('e=',e)
            print('U=',U)
            print('total # graphs        :',graphcount)
            print('total # instances     :',Total)
            print('solvable and reachable:',SandR)
            print('                     %: {:.1f}'.format(SandR/Total*100))
            env_select=random.choice(env_all)
            env_select.reset()
            env_select.render(fname='example_3x3instance_e='+str(e)+'_u='+str(U))

            k=0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)    

    # Model hyperparameters
    parser.add_argument('--num_edges', default=6, type=int, 
                        help='Number of edges to be removed from default graph')
    parser.add_argument('--U', default=0, type=int, 
                    help='Number of units')


    args=parser.parse_args()
    
    ### Pipeline to create datasets of edge-removed graphs and unit paths
    #RunInstance(args)
    #TestSim()
    #MergeDataFiles()
    #SaveSolvabilityData(args)
    #MergeDataFilesSolvability()
    ##SaveReachabilityData()

    ### Testing the data
    TestInteractiveSimulation(U=[1,2,3],E=[0])#i for i in range(11)])
    #CalculateStatistics()