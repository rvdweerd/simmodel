import copy
import argparse
from sys import breakpointhook
from modules.optim.optimization_FIP_gurobipy import optimization_alt
from modules.optim.simdata_create import GetDatabankForPartialGraph
from modules.rl.environments import GraphWorld#, GraphWorldFromDatabank
from modules.rl.rl_utils import EvaluatePolicy
from modules.rl.rl_policy import EpsilonGreedyPolicy
from modules.rl.rl_algorithms import q_learning_exhaustive
from modules.sim.graph_factory import get_all_edge_removals_symmetric, LoadData
from modules.sim.simdata_utils import SimulateInteractiveMode, ObtainSimulationInstance
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
                if (1,0) in s.nodes(): # the escape start position node is in this component
                    W_new=nx.convert_matrix.to_numpy_matrix(s.to_directed())
                    for j, coord in enumerate(s.nodes()):
                        nodeid2coord_new[j]=coord
                else:
                    if len(s.nodes())>1: # avoid duplicate subgraphs being registered; all selected edges must be on the main component
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

def SaveSolvabilityData(args, edge_blocking=False):
    in_file=open("./datasets/__partial_graphs/Manhattan_N=3,L=4,R=100,Ndir=False/_databank_full","rb")
    databank_full=pickle.load(in_file)
    in_file.close()
    in_file=open("./datasets/__partial_graphs/Manhattan_N=3,L=4,R=100,Ndir=False/_partial_graph_register_trimmed","rb")
    partial_graph_register=pickle.load(in_file)
    in_file.close()

    num_edges=args.num_edges
    U=args.U
    solvable_dict={}  
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
    env0 = GraphWorld(config, optimization_method='static', fixed_initial_positions=None, state_representation=state_repr, state_encoding=state_enc)
    env0.capture_on_edges = edge_blocking

    for W_, hashint, hashstr in tqdm.tqdm(partial_graph_register[num_edges]):
        #env_data={'W':W_, 'hashint':hashint, 'databank_full':databank_full}
        if hashint not in databank_full['U='+str(U)]: 
            print('Warning: hashint not in databank')
            continue
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

    visited=set()
    for u in [1,2,3]: #HERE US A PROBLEM!!
        for e in range(11):
            for W_, hashint, hashstr in partial_graph_register_old[e]:
                if hashint in merged_databank['U='+str(u)] and hashint not in visited:
                    partial_graph_register_new[e].append((W_,hashint,hashstr))
                    visited.add(hashint)
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

def  print_world_properties(env, env_idx, entry, hashint, hashstr, edge_blocking, solve_select, reject_u_duplicates, solvable_):
    print('\nenv index',env_idx,', current entry',env.current_entry,'| edge_blocking:',edge_blocking, '| solvable:', solve_select,'| reject duplicates:',reject_u_duplicates)
    print('> graph hash:', hashint,' /', hashstr, '| state_repr:',env.state_representation, '| state_encoding:',env.state_encoding,)
    print('> state:', env.state)
    print('> obs:\n',env.obs)
    print('> example is registered as: '+('Solvable' if solvable_[entry] else 'Unsolvable'))
    print('-----------------------------')


def TestInteractiveSimulation(U=[2],E=[8], edge_blocking=False, solve_select='both', reject_u_duplicates=False):
    state_repr = 'et'
    state_enc  = 'tensors'
    databank_full, register_full, solvable = LoadData(edge_blocking = edge_blocking)
    all_envs, hashint2env, env2hashint, env2hashstr = GetWorldSet(state_repr, state_enc, U=U, E=E, edge_blocking=edge_blocking, solve_select=solve_select, reject_duplicates=reject_u_duplicates)
    
    while True:
        env_idx = random.randint(0,len(all_envs)-1)
        env = all_envs[env_idx]
        env.reset()
        u=env.sp.U
        e0U0lookup = env._to_coords_from_state()
        hashint = env2hashint[env_idx]
        hashstr = env2hashstr[env_idx]
        s = solvable['U='+str(u)][hashint]
        entry = databank_full['U='+str(u)][hashint]['register'][e0U0lookup]
        assert entry == env.current_entry
        if reject_u_duplicates and has_duplicates(env.state[1:]):
            continue

        print_world_properties(env, env_idx, entry, hashint, hashstr, edge_blocking, solve_select, reject_u_duplicates, solvable_=s)
        
        env._remove_world_pool()
        print_world_properties(env, env_idx, entry, hashint, hashstr, edge_blocking, solve_select, reject_u_duplicates, solvable_=s)
        SimulateInteractiveMode(env, filesave_with_time_suffix=False)

        env._restore_world_pool()
        print_world_properties(env, env_idx, entry, hashint, hashstr, edge_blocking, solve_select, reject_u_duplicates, solvable_=s)
        SimulateInteractiveMode(env,filesave_with_time_suffix=False)
        
def RunSpecficInstance(U0=[(2,2)], hashint=1775, edge_blocking=False):
    config=GetConfig(u=len(U0))
    state_repr = 'etUt'
    state_enc  = 'nfm'
    databank_full, register_full, solvable = LoadData(edge_blocking = edge_blocking)
    e0U0lookup = tuple([(1,0)]+U0)

    s = solvable['U='+str(len(U0))][hashint]
    idx = databank_full['U='+str(len(U0))][hashint]['register'][e0U0lookup]
  
    env0 = GraphWorld(config, optimization_method='static', fixed_initial_positions=None, state_representation=state_repr, state_encoding=state_enc)
    env0.capture_on_edges = edge_blocking
    all_envs=[]
    hashint2env={}
    
    u=len(U0)
    env0.sp.U = u
    idx = databank_full['U='+str(u)][hashint]['register'][e0U0lookup]
    env_data = databank_full['U='+str(u)][hashint]
    env=copy.deepcopy(env0)
    env.redefine_graph_structure(env_data['W'],env_data['nodeid2coord'],new_nodeids=True)
    env.reload_unit_paths(env_data['register'],env_data['databank'],env_data['iratios'])
    env.reset(idx)

    print_world_properties(env, 'n/a', idx, hashint, bin(hashint), edge_blocking, 'n/a', 'n/a', solvable_=s)
    SimulateInteractiveMode(env, filesave_with_time_suffix=False)

def has_duplicates(arr):
    dups=False
    mem=set()
    for i in arr:
        if i in mem:
            dups=True
            break
        mem.add(i)
    return dups

def CalculateStatistics(E=[i for i in range(11)], U=[1,2,3], edge_blocking=False, plotting=False):
    config=GetConfig()
    state_repr = 'etUt'
    state_enc  = 'nodes'
    databank_full, register_full, solvable = LoadData(edge_blocking=edge_blocking)
    env0 = GraphWorld(config, optimization_method='static', fixed_initial_positions=None, state_representation=state_repr, state_encoding=state_enc)
    env0.capture_on_edges = False

    for e in E:
        for u in U:
            S=[]
            R=[]
            graphcount=0
            env_all=[]
            for W_, hashint, hashstr in register_full[e]:
                s = solvable['U='+str(u)][hashint]
                S+=list(s)
                graphcount+=1
                env_data = databank_full['U='+str(u)][hashint] # dict contains  'register':{(e0,U0):index}, 'databank':[], 'iratios':[]
                env=copy.deepcopy(env0)
                env.sp.U=u
                env.redefine_graph_structure(env_data['W'],env_data['nodeid2coord'],new_nodeids=True)
                env.reload_unit_paths(env_data['register'],env_data['databank'],env_data['iratios'])
                valids = s#np.logical_and(s,r)
                if True:#valids.sum() > 0:
                    env.world_pool = list(np.array(env.all_worlds)[valids])
                    env_all.append(env)
            Total = len(S)
            Solvable = np.array(S).sum()
            print('---------------')
            print('e=',e)
            print('U=',u)
            print('total # graphs    :',graphcount)
            print('total # instances :',Total)
            print('solvable          :',Solvable)
            print('                     %: {:.1f}'.format(Solvable/Total*100))
            if plotting:
                while True:
                    env_select=random.choice(env_all)
                    env_select.reset()
                    if len(set(env_select.state[1:])) == u:
                        env_select.render(fname='example_3x3instance_e='+str(e)+'_u='+str(u))
                        break

def GetWorldSet(state_repr = 'et', state_enc  = 'tensors', U=[1,2,3], E=[i for i in range(11)], edge_blocking=False, solve_select='solvable', reject_duplicates=True):
    config=GetConfig(u=2)#only needed to find datafile
    databank_full, register_full, solvable = LoadData(edge_blocking = edge_blocking)
    
    env0 = GraphWorld(config, optimization_method='static', fixed_initial_positions=None, state_representation=state_repr, state_encoding=state_enc)
    env0.capture_on_edges = edge_blocking
    all_envs=[]
    hashint2env={}
    env2hashint={}
    env2hashstr={}
    for u in U:
        env0.sp.U = u
        for e in E:
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
                valids = s #np.logical_and(np.logical_not(s),r)
                # Filter out solvable intial conditions
                if solve_select == 'both':
                    env.world_pool = env.all_worlds
                elif solve_select == 'solvable':
                    env.world_pool = list(np.array(env.all_worlds)[valids]) # only use solvable puzzles
                elif solve_select == 'non_solvable':
                    env.world_pool = list(np.array(env.all_worlds)[np.logical_not(valids)]) # only use solvable puzzles    
                else:
                    assert False
                if len(env.world_pool) > 0:
                    env.all_worlds = env.world_pool
                    all_envs.append(env)
                    hashint2env[hashint]=len(all_envs)-1
                    env2hashint[len(all_envs)-1]=hashint
                    env2hashstr[len(all_envs)-1]=hashstr
    return all_envs, hashint2env, env2hashint, env2hashstr

def TestOptimOutcome(hashint, env_idx, entry, U=[1,2,3], E=[i for i in range(11)], edge_blocking=False, solve_select='solvable', reject_duplicates=True):
    state_repr = 'et'
    state_enc  = 'tensors'
    all_envs, hashint2env, env2hashint, env2hashstr = GetWorldSet(state_repr, state_enc, U=U, E=E, edge_blocking=edge_blocking, solve_select=solve_select, reject_duplicates=reject_duplicates)
    env=all_envs[env_idx]
    env.reset(entry)
    s_coords = list(env._to_coords_from_state())
    #register, databank, iratios = GetDatabankForPartialGraph(env.sp, 1,1)
    reg_entry, sim_instance, iratio, eval_time, marktimes = ObtainSimulationInstance(env.sp, {}, specific_start_units=s_coords[1:], cutoff=1e5, print_InterceptRate=True, create_plot=False)

    k=0
    #register, databank, iratios = GetDatabankForPartialGraph(env.sp, REQUIRED_DATASET_SIZE, UPDATE_EVERY)


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
    #SaveSolvabilityData(args, edge_blocking=True)
    #MergeDataFilesSolvability()
    ##SaveReachabilityData()

    ### Testing the data: NOTE CHECK INSTANCE WITH IRENE
    #TestOptimOutcome(hashint=4056, env_idx=592, entry=7, U=[3], E=[i for i in range(11)], edge_blocking=True, solve_select='solvable', reject_duplicates=True)
    #RunSpecficInstance(U0=[(0,0),(0,2),(1,2)], hashint=4056, edge_blocking=True)

    TestInteractiveSimulation(U=[1,2,3], E=[i for i in range(11)], edge_blocking=False, solve_select='solvable', reject_u_duplicates=False)
    #TestInteractiveSimulation(U=[1],E=[0],edge_blocking=False)#i for i in range(11)])
    #RunSpecficInstance(U0=[(1,1),(2,2)], hashint=1396, edge_blocking=False)
    #RunSpecficInstance(U0=[(0,0),(2,0),(2,1)], hashint=1059, edge_blocking=False)
    #CalculateStatistics(E=[i for i in range(11)], U=[1,2,3], edge_blocking=False, plotting=False)
    #CalculateStatistics(E=[10], U=[2],plotting=False)
