from itertools import combinations
import numpy as np

def EvaluatePolicy(env, policy, number_of_runs=1, print_runs=True, save_plots=False):
    # Escaper chooses random neighboring nodes until temination
    # Inputs:
    #   optimization_method: if dynamic, new optimal unit position targets are used at every time-step
    #                        if static, only optimal unit position targets calculated at start are used and fixed
    captured=[]
    iratios_sampled=[]
    rewards=[]
    for i in range(number_of_runs):
        s=env.reset()
        iratios_sampled.append(env.iratio)
        done=False
        R=0
        if print_runs:
            print('Run',i+1,": [",end='')
        count=0
        #e_history=[]
        while not done:
            #e_history.append(s[0])
            if save_plots:
                env.render()
            if print_runs:
                print(str(s[0])+'->',end='')
            
            action,_ = policy.sample_greedy_action(s)

            s,r,done,info = env.step(action)
            count+=1
            R+=r
            # if count >= env.sp.L:
            #     break
        if print_runs:
            print(str(s[0])+']. Done after',count,'steps, Captured:',info['Captured'],'Reward:',str(R))
        if save_plots:
            env.render()
        captured.append(int(info['Captured']))
        rewards.append(R)
    print('------------------')
    print('Observed capture ratio: {:.3f}'.format(sum(captured)/len(captured)),', Average reward: {:.2f}'.format(sum(rewards)/len(rewards)))
    print('Capture ratio at data generation: last {:.3f}'.format(env.iratio),', avg at generation {:.3f}'.format(sum(env.iratios)/len(env.iratios)),\
        ', avg sampled {:.3f}'.format(sum(iratios_sampled)/len(iratios_sampled)),'\n')

# find u0's in higher rows
def GetInitialStatesList(env, min_y_coord):
    idx=[]
    initpos=[]
    for k,v in env.register['coords'].items():
        if k[0][1]==0:
            valid = True
            for u in k[1:]:
                if u[1] < min_y_coord:
                    valid = False
                    break
            #if k[1][1]>=min_y_coord and k[2][1]>=min_y_coord and k[3][1]>=min_y_coord:
            if valid:
                #print(k,v)
                idx.append(v)
                #initpos.append(env._to_state_from_coords(k[0],k[1:]))
                ipos=[env.sp.coord2labels[k[0]]]
                for u in k[1:]:
                    ipos.append(env.sp.coord2labels[u])
                ipos.sort()
                initpos.append(tuple(ipos))

    return idx, initpos

def GetSetOfPermutations(path):
    out=set()
    for n in range(len(path) + 1):
        for i in combinations(path,n):
            if len(i)==0: continue
            j=list(i)
            j.sort()
            out.add(tuple(j))
        #yield from combinations(iterable, n)
    return out

def GetDuplicatePathsIndices(paths, min_num_same_positions, min_num_worlds):
    duplicates_idx=set()
    for row in range(paths.shape[1]):
        map={}
        map2envs={}
        map2time={}
        for n in range(paths.shape[0]):
            perms=GetSetOfPermutations(paths[n][row])
            for p in perms:
                if p not in map2envs:
                    map2envs[p]=[n]
                else:
                    map2envs[p].append(n)
                if p not in map2time:
                    map2time[p]=[row]
                else:
                    map2time[p].append(row)                
                if p not in map:
                    map[p]=1
                else:
                    map[p]+=1
        for k,v in map.items():
            if len(k) >= min_num_same_positions and v >= min_num_worlds: # at least 3 units on same position, at least in 2 worlds
                print('identical U positions',k,'appear in worlds',map2envs[k],'in time-step',row)
                for i in map2envs[k]:
                    duplicates_idx.add(i)
    return list(duplicates_idx)

def GetPathsTensor(env,db_indices):
    paths = []
    for i in db_indices:
        e = env.databank['labels'][i]
        #print(  'start_escape_route',e['start_escape_route'],\
        #        'start_units',e['start_units'],\
        #        'paths',e['paths'])
        unit_paths=e['paths']
        ipaths = []
        for t in range(env.sp.L):
            p = []
            for P_path in unit_paths:
                pos = P_path[-1] if t >= len(P_path) else P_path[t]
                p.append(pos)
            ipaths.append(tuple(p))
        paths.append(ipaths)
    paths_np=np.array(paths)
    print('Paths found:',paths_np.shape)
    return paths_np, paths

def SelectTrainset(env, min_y_coord, min_num_same_positions, min_num_worlds):
    db_indices, init_pos_list = GetInitialStatesList(env, min_y_coord)
    paths_np, paths = GetPathsTensor(env, db_indices)

    duplicates_indices0=set()
    for world_source in range(len(paths)):
        for i in range(len(paths[world_source])):
            key1=list(paths[world_source][i])
            key1.sort()
            for world_target in range(len(paths)):
                if world_target==world_source: continue
                for t in range(env.sp.L-1):
                    key2_t=list(paths[world_target][t])
                    key2_t.sort()
                    key2_tt=list(paths[world_target][t+1])
                    key2_tt.sort()
                    #if key2_t[:2] == key1[:2] and key2_tt[:2] != key1[:2]:
                    if key2_t == key1 and key2_tt != key1:
                        print('t=',t,'Pair found:',paths[world_source],paths[world_target])
                        duplicates_indices0.add(world_target)
                        duplicates_indices0.add(world_source)
    
    duplicates_indices0 = list(duplicates_indices0)
    duplicates_indices1 = GetDuplicatePathsIndices(paths_np, min_num_same_positions, min_num_worlds)    
    
    init_pos_trainset0=[init_pos_list[i] for i in duplicates_indices0]
    init_pos_trainset_indices0=[db_indices[i] for i in duplicates_indices0]
    init_pos_trainset1=[init_pos_list[i] for i in duplicates_indices1]
    init_pos_trainset_indices1=[db_indices[i] for i in duplicates_indices1]

    # print('Initial positions in trainset:')
    # for entry in init_pos_trainset1:
    #     print(entry)
    return init_pos_trainset_indices0, init_pos_trainset_indices0