import traceback
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker

# Itility functions to extract and plot data from tensorboard log files

def LSTM_info_from_path(path):
    i1=path.find('lstm')
    i2=path.find('NFM')
    string = path[i1:(i2-1)]
    infolist = string.split('_')
    if infolist[1]=='None':
        return 'None'
    lstm_type=infolist[1]
    hdim=infolist[2]
    nlay=infolist[3]
    return lstm_type+'-hdim='+hdim+'-num_layers='+nlay

# Extraction function
def tflog2pandas(path):
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data
    #df=df[(df.metric != 'params/lr')&(df.metric != 'params/mm')&(df.metric != 'train/loss')] #delete the mentioned rows
    #df.to_csv("output.csv")

def tdlog2np(path, metric):
    try:
        event_acc = EventAccumulator(path)
        event_acc.Reload()
        #tags = event_acc.Tags()["scalars"]
        event_list = event_acc.Scalars(metric)
        values = list(map(lambda x: x.value, event_list))
        step = list(map(lambda x: x.step, event_list))        
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return np.array(step), np.array(values)

def realign(stepsource, valuesource, steptarget):
    newlist=[]
    for s in steptarget:
        if s < stepsource[0]:
            newlist.append(valuesource[0])
        else:
            idx = np.searchsorted(stepsource, s, side='left', sorter=None)
            #newlist.append(valuesource[idx])
            interpolated = valuesource[idx-1] + (valuesource[idx]-valuesource[idx-1])*(s-stepsource[idx-1])/(stepsource[idx]-stepsource[idx-1])
            newlist.append(interpolated)
    return np.array(newlist)

def CreateLearningCurvePlotsPhase1(path, n_smoothing=20, nseeds=5, numiter=1200, yrange=[-10.,2.]):
    rawlist=[]
    maxlen=0
    for seed in range(nseeds):
        path_ = path+'/DQN'+str(seed+1)
        step,rew = tdlog2np(path_,'5. return_per_epi')
        rawlist.append(rew)
        maxlen = max(maxlen, len(rew))

    smtlist=[]
    smoothing_vector=[1/n_smoothing]*n_smoothing
    best_line_idx=-1
    globalbest=-1e6
    for i in range(len(rawlist)):
        p=maxlen-len(rawlist[i])
        rawlist[i]=np.pad(rawlist[i],(0,p),'constant',constant_values=np.mean(rawlist[i][-n_smoothing*30:]))#'edge')
        avg=np.convolve(smoothing_vector, rawlist[i],mode='valid')
        localbest=avg.max()
        if localbest > globalbest:
            globalbest = localbest
            best_line_idx=i
        smtlist.append(avg)
    step=step[2:-2]
    rawlist=np.array(rawlist)[:,2:-2] # (seeds, max_iterations)
    smtlist=np.array(smtlist) # (seeds, max_iterations)
    smt_avgline=np.mean(smtlist,axis=0) # (max_iterations, )
    smt_stdline=np.std(smtlist,axis=0)
    smt_minline=np.min(smtlist,axis=0)
    smt_maxline=np.max(smtlist,axis=0)
    raw_minline=np.min(rawlist,axis=0)
    raw_maxline=np.max(rawlist,axis=0)
    smt_bestline=smtlist[best_line_idx,:]
    # # PLOT SMOOTHED CURVE FOR EACH SEED
    plt.plot(np.transpose(smtlist))
    plt.savefig(path+'/Train_reward_smoothed_per_seed.png')
    plt.clf()

    # # PLOT UNSMOOTHED CURVE FOR EACH SEED
    plt.plot(np.transpose(rawlist))
    plt.savefig(path+'/Train_reward_raw_per_seed.png')
    plt.clf()
    
    # PLOT SMOOTHED MAX,AVG,MINMAX RANGE, STD
    fig,ax=plt.subplots(figsize=(12,5))
    #ax.plot(smt_maxline,color="green", label='best seed')
    
    #ax.plot(step,smtlist.transpose(),color="orange",alpha=.2)
    ax.plot(step,smt_avgline,color="orange",alpha=1., label='avg seed')
    ax.fill_between(step,raw_minline,raw_maxline,facecolor='orange',alpha=0.35, label='minmax')
    ax.fill_between(step,smt_avgline-smt_stdline,np.minimum((smt_avgline+smt_stdline),smt_maxline),facecolor='gray',alpha=0.3,label='std')
    #ax.plot(smt_bestline,color="green", label='best seed')
    #ax.legend()
    #ax=plt.gca()
    if numiter==None:
        #numiter=len(smt_maxline)
        numiter=max(step+100)
    ax.set_xlim([0,numiter])
    ax.set_ylim(yrange)
    #plt.ylabel('average reward')
    #ax.axes.get_xaxis().set_visible(False)
    #ax.axes.get_xaxis().set_ticklabels([])
    #ax.axes.get_xaxis().axhline(y=0,color='black')
    #ax.yaxis.set_ticklabels([])
    ax.axhline(y=0,color='black',linewidth=.5)
    #lstm_dentifier = "title"
    #plt.title(lstm_dentifier)
    plt.savefig(path+'/Train_reward_Learningcurves.png')
    plt.clf()

def CreateLearningCurvePlots(path, n_smoothing=21, seed0=0, nseeds=5, numiter=1200, yrange=[-10.,2.]):
    assert n_smoothing%2==1
    rawlist=[]
    steplist=[]
    maxlen=0
    for seed in range(seed0, seed0+nseeds):
        path_ = path+'/SEED'+str(seed)+'/logs'
        try:
            step,rew = tdlog2np(path_,'return_per_epi')
        except:
            continue
        rawlist.append(rew)
        steplist.append(step)
        maxlen = max(maxlen, len(rew))

    # USE if step arrays are not aligned (seed 2200 issue)
    rawlist[0]=realign(steplist[0],rawlist[0],steplist[1])
    steparray=steplist[-1]

    smtlist=[]
    smoothing_vector=[1/n_smoothing]*n_smoothing
    best_line_idx=-1
    globalbest=-1e6
    for i in range(len(rawlist)):
        p=maxlen-len(rawlist[i])
        rawlist[i]=np.pad(rawlist[i],(0,p),'constant',constant_values=np.mean(rawlist[i][-n_smoothing*30:]))#'edge')
        avg=np.convolve(smoothing_vector, rawlist[i],mode='valid')
        localbest=avg.max()
        if localbest > globalbest:
            globalbest = localbest
            best_line_idx=i
        smtlist.append(avg)
    rawlist=np.array(rawlist) # (seeds, max_iterations)
    smtlist=np.array(smtlist) # (seeds, max_iterations)
    smt_avgline=np.mean(smtlist,axis=0) # (max_iterations, )
    smt_stdline=np.std(smtlist,axis=0)
    smt_minline=np.min(smtlist,axis=0)
    smt_maxline=np.max(smtlist,axis=0)
    raw_minline=np.min(rawlist,axis=0)
    raw_maxline=np.max(rawlist,axis=0)
    smt_bestline=smtlist[best_line_idx,:]
    # # PLOT SMOOTHED CURVE FOR EACH SEED
    plt.plot(np.transpose(smtlist))
    plt.savefig(path+'/Train_reward_smoothed_per_seed.png')
    plt.clf()

    # # PLOT UNSMOOTHED CURVE FOR EACH SEED
    plt.plot(np.transpose(rawlist))
    plt.savefig(path+'/Train_reward_raw_per_seed.png')
    plt.clf()
    
    # PLOT SMOOTHED MAX,AVG,MINMAX RANGE, STD
    steparray=steparray[(n_smoothing//2):-(n_smoothing//2)]
    fig,ax=plt.subplots(figsize=(5,5))
    #ax.plot(smt_maxline,color="green", label='best seed')
    #ax.plot(steparray,smtlist.transpose(),color="gray",alpha=.2, linewidth=0.4)
    #ax.plot(steparray,smt_bestline,color="green", label='best seed', linewidth=0.4)
    ax.plot(steparray,smt_avgline,color="orange",alpha=1., label='avg seed')
    ax.fill_between(steparray,smt_avgline-smt_stdline,np.minimum((smt_avgline+smt_stdline),smt_maxline),facecolor='orange',alpha=0.4,label='std')
    ax.fill_between(steparray,smt_minline,smt_maxline,facecolor='gray',alpha=0.15, label='minmax')
    #ax.legend()
    #ax=plt.gca()
    ax.set_xlim([0,numiter])
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x/1000) + 'k'))
    ax.set_ylim(yrange)
    #plt.ylabel('average reward')
    #ax.axes.get_xaxis().set_visible(False)
    #ax.axes.get_xaxis().set_ticklabels([])
    #ax.axes.get_xaxis().axhline(y=0,color='black')
    #ax.yaxis.set_ticklabels([])
    ax.axhline(y=0,color='black',linewidth=.5)
    #lstm_dentifier = LSTM_info_from_path(path)
    #plt.title(lstm_dentifier)
    plt.savefig(path+'/Train_reward_Learningcurves_'+str(n_smoothing)+'.png')
    plt.clf()

def CreateLossCurvePlots(path, n_smoothing=21, seed0=0, nseeds=1, numiter=25000, yrange=[-0.5,.2]):
    assert n_smoothing%2==1
    fig,ax=plt.subplots(figsize=(5,5))
    colorlist=['orange','royalblue','green']
    #for curve_num,curve_name in enumerate(['loss1_ratio','loss2_value','loss3_entropy']):#,'loss_total']:
    for curve_num,curve_name in enumerate(['return_per_epi']): #['ep_rew_mean']:#['4. Reward per epi']
        rawlist=[]
        steplist=[]
        maxlen=0
        for seed in range(seed0, seed0+nseeds):
            path_ = path+'/SEED'+str(seed)+'/logs'
            try:
                step,l1 = tdlog2np(path_, curve_name)
            except:
                continue
            rawlist.append(l1)
            steplist.append(step)
            maxlen = max(maxlen, len(l1))

        # USE if step arrays are not aligned (seed 2200 issue)
        #rawlist[0]=realign(steplist[0],rawlist[0],steplist[1])
        steparray=steplist[-1]

        smtlist=[]
        smoothing_vector=[1/n_smoothing]*n_smoothing
        best_line_idx=-1
        globalbest=-1e6
        for i in range(len(rawlist)):
            p=maxlen-len(rawlist[i])
            rawlist[i]=np.pad(rawlist[i],(0,p),'constant',constant_values=np.mean(rawlist[i][-n_smoothing*30:]))#'edge')
            avg=np.convolve(smoothing_vector, rawlist[i],mode='valid')
            localbest=avg.max()
            if localbest > globalbest:
                globalbest = localbest
                best_line_idx=i
            smtlist.append(avg)
        rawlist=np.array(rawlist) # (seeds, max_iterations)
        smtlist=np.array(smtlist) # (seeds, max_iterations)
        smt_avgline=np.mean(smtlist,axis=0) # (max_iterations, )
        smt_stdline=np.std(smtlist,axis=0)
        smt_minline=np.min(smtlist,axis=0)
        smt_maxline=np.max(smtlist,axis=0)
        raw_minline=np.min(rawlist[:,(n_smoothing//2):-(n_smoothing//2)],axis=0)
        raw_maxline=np.max(rawlist[:,(n_smoothing//2):-(n_smoothing//2)],axis=0)
        smt_bestline=smtlist[best_line_idx,:]
        
        # PLOT SMOOTHED MAX,AVG,MINMAX RANGE, STD
        steparray=steparray[(n_smoothing//2):-(n_smoothing//2)]
        #ax.plot(steparray,smt_bestline,alpha=1., label='best', color="green",linewidth=1.5)
        ax.plot(steparray,smt_avgline,alpha=1., label=curve_name, color=colorlist[curve_num],linewidth=1.5)
        ax.fill_between(steparray,smt_avgline-smt_stdline,np.minimum((smt_avgline+smt_stdline),smt_maxline),facecolor=colorlist[curve_num],alpha=0.4,label='std')
        ax.fill_between(steparray,smt_minline,smt_maxline,facecolor='gray',alpha=0.15, label='minmax')
    ax.set_xlim([0,numiter])
    #ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x/1000) + 'k'))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))
    ax.set_ylim(yrange)
    ax.axhline(y=0,color='black',linewidth=.5)
    #plt.savefig(path+'/losscurves_'+str(n_smoothing)+'.png')
    plt.savefig(path+'/returncurve_'+str(n_smoothing)+'_'+str(numiter)+'.png')
    plt.clf()



#root="./results/results_Phase3simp/test_lstm_simp"
#root="./results/results_Phase3simp/ppo/MemTask-U1"
#root="./results/results_Phase3/ppo/M5x5Fixed"
root="./results/results_Phase3simp/ppo/MemTask-U1/gat2-v/emb48_itT5/lstm_Dual_48_1/"
#root="./results/results_Phase3simp/ppo/MemTask-U1/gat2-v/emb48_itT5/lstm_None/NFM_ev_ec_t_dt_at_um_us-BasicDict"
#root="./results/results_Phase3/ppo/M5x5Fixed/gat2-v/emb24_itT5/lstm_None/NFM_ev_ec_t_dt_at_um_us"
#root="./results/results_Phase3simp/ppo/NWB_AMS_mixed_obs/gat2-q/emb64_itT5/lstm_EMB_64_1"
#root="./results/results_Phase3simp/ppo/NWB_AMS/gat2-q/emb64_itT5/lstm_None"

# USE FOR LSTM
for path in [x[0] for x in os.walk(root)]:
    if os.path.isfile(path+'/train-parameters.txt'):
        #path="./results/results_Phase3/ppo/MemTask-U1/gat2-q/emb24_itT5/lstm_Dual_24_1/NFM_ev_ec_t_dt_at_um_us/omask_freq0.2/bsize48" #folderpath
        #CreateLearningCurvePlots(path=path, seed0=2200, nseeds=5, n_smoothing=201, numiter=25000, yrange=[-9.,9.])
        CreateLossCurvePlots(path=path, seed0=0, nseeds=5, n_smoothing=15, numiter=300, yrange=[-6,11])

#path="./results/results_Phase1/DQN/Manhattan5x5_VariableEscapeInit/etUt/tensorboard"
#CreateLearningCurvePlotsPhase1(path=path, n_smoothing=5, nseeds=5, numiter=None, yrange=[-10.,5.])