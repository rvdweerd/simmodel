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
        e = start_escape_node
        p = []
        for P_path in unit_paths:
            pos = P_path[-1] if t >= len(P_path) else P_path[t]
            p.append(pos)
        su.PlotAgentsOnGraph(sp, e, p, t)

def SimulateInteractiveMode(conf):
    env=GraphWorld(conf,optimization='static')
    #s=env.reset(((1,0),(0,2),(2,2)))
    s=env.reset()
    done=False
    R=0
    env.render()
    while not done:
        print('u paths:',env.u_paths,'\nu positions:')
        for t in range(5):
            print(env._getUpositions(t))
        print('------------')
        print('Current state:')
        print(s)
        print('Available actions:')
        for k,v in env._availableActionsInCurrentState().items():
            print('>',k,v)
        dir=input('Action (new node)?\n> ')
        s,r,done,_=env.step(dir)
        env.render()
        R+=r
    print('done, reward='+str(R),'\n---------------')

def SimulateRandomWalker(conf, number_of_runs=10, save_plots=False):
    # Escaper chooses random neighboring nodes until temination
    env=GraphWorld(conf,optimization='static')
    #s=env.reset(((1,0),(0,2),(2,2)))
    captured=[]
    rewards=[]
    for i in range(number_of_runs):
        s=env.reset()
        done=False
        R=0
        print('Run',i+1,": [",end='')
        count=0
        #e_history=[]
        while not done:
            #e_history.append(s[0])
            if save_plots:
                env.render()
            print(str(s[0])+'->',end='')
            possible_actions = env._availableActionsInCurrentState()['coords']
            action = random.choice(possible_actions)
            s,r,done,info = env.step(action)
            count+=1
            R+=r
            if count >= env.sp.T:
                break
        print(']. Done after',count,'steps, Captured:',info['Captured'],'Reward:',str(R))
        captured.append(int(info['Captured']))
        rewards.append(R)
    print('------------------')
    print('Capture ratio: ',sum(captured)/len(captured),', Average reward:',sum(rewards)/len(rewards))

configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
conf=configs['Manhattan5']
#conf=configs['Manhattan11']
#conf=configs['CircGraph']
#conf=configs['TKGraph']
conf['direction_north']=False

#SimulatePursuersPathways(conf)
#SimulateInteractiveMode(conf)
SimulateRandomWalker(conf, number_of_runs=1000, save_plots=False)