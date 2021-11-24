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
