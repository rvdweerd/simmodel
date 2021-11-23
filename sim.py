import simdata_utils as su
from environments import GraphWorld
import random

def SimulatePursuersPathways(conf, dataframe=0):
    # Escaper position static, plot progression of pursuers motion
    sp = su.DefineSimParameters(conf)
    dirname = su.make_result_directory(sp)
    register, databank, iratios = su.LoadDatafile(dirname)
    start_escape_node = databank[dataframe]['start_escape_route']
    unit_paths = databank[dataframe]['paths']
    print('unit_paths',unit_paths,'intercept_rate',iratios[dataframe])
    for t in range(sp.L):
        e = sp.coord2labels[start_escape_node]
        p = []
        for P_path in unit_paths:
            pos = P_path[-1] if t >= len(P_path) else P_path[t]
            p.append(sp.coord2labels[pos])
        su.PlotAgentsOnGraph(sp, e, p, t)

def SimulateInteractiveMode(conf, optimization_method = 'static'):
    env=GraphWorld(conf, optimization_method)
    s=env.reset()
    #s=env.reset(((2,0),(0,3),(3,4),(4,1))) # Used for testing
    # while True:
    #     s=env.reset()#((1,0),(0,2),(2,2)))
    #     print(s[0],' ',end='')
    #     if s[0]==(2,0):
    #         break
    done=False
    R=0
    env.render()
    while not done:
        print('e position:',s[0],env.sp.labels2coord[s[0]])
        print('u paths (node labels):',env.u_paths)
        print('u paths (node coords): ',end='')
        for p in env.u_paths:
            print('[',end='')
            for e in p:
                print(str(env.sp.labels2coord[e])+',',end='')
            print('],  ',end='')
        print('\nu positions per time-step:')
        for t in range(env.sp.T):
            print(env._getUpositions(t))
        print('------------')
        print('Current state:')
        print(s)
        print('Available actions:\n> [ ',end='')
        for n in env.neighbors[s[0]]:
            print(str(n)+', ',end='')
        a=input(']\nAction (new node)?\n> ')
        s,r,done,_=env.step(int(a))
        env.render()
        R+=r
    print('done, reward='+str(R),'\n---------------')

def SimulateWalker(conf, policy='Random', number_of_runs=1, optimization_method='static', print_runs=True, save_plots=False):
    # Escaper chooses random neighboring nodes until temination
    # Inputs:
    #   optimization_method: if dynamic, new optimal unit position targets are used at every time-step
    #                        if static, only optimal unit position targets calculated at start are used and fixed
    env=GraphWorld(conf, optimization=optimization_method)
    captured=[]
    iratios_sampled=[]
    rewards=[]
    for i in range(number_of_runs):
        s=env.reset()
        #s=env.reset(((2,0),(0,3),(3,4),(4,1))) # Used for testing
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
            
            if policy == 'Random':
                possible_actions = env._availableActionsInCurrentState()
                action = random.choice(possible_actions)
            elif policy == 'LeftUp':
            # Always all the way left, then all the way up
                possible_actions = env.neighbors[s[0]]
                if s[0]-1 in possible_actions: # can go left
                    action = s[0]-1
                elif s[0]+env.sp.N in possible_actions: # can't go left, go up
                    action = s[0]+env.sp.N
                elif s[0]+1 in possible_actions: # can't go up, go right
                    action = s[0]+1
                else: # go down
                    action = s[0]-env.sp.N

            s,r,done,info = env.step(action)
            count+=1
            R+=r
            if count >= env.sp.L:
                break
        if print_runs:
            print(str(s[0])+']. Done after',count,'steps, Captured:',info['Captured'],'Reward:',str(R))
        if save_plots:
            env.render()
        captured.append(int(info['Captured']))
        rewards.append(R)
    print('------------------')
    print('Observed capture ratio: ',sum(captured)/len(captured),', Average reward:',sum(rewards)/len(rewards))
    print('Capture ratio at data generation: last',env.iratio,' avg at generation',sum(env.iratios)/len(env.iratios),\
        'avg sampled',sum(iratios_sampled)/len(iratios_sampled))

configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
#conf=configs['Manhattan3']
conf=configs['Manhattan5']
#conf=configs['Manhattan11']
#conf=configs['CircGraph']
#conf=configs['TKGraph']
conf['direction_north']=False

random.seed(9)
#SimulatePursuersPathways(conf)
SimulateInteractiveMode(conf, optimization_method='static')
#SimulateWalker(conf, policy='LeftUp',number_of_runs=50000, optimization_method='dynamic', print_runs=False, save_plots=False)