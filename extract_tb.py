import traceback
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import matplotlib.pyplot as plt
import os

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

def CreateLearningCurvePlots(path, n_smoothing=20, nseeds=5, numiter=1200, yrange=[-10.,2.]):
    rawlist=[]
    maxlen=0
    for seed in range(nseeds):
        path_ = path+'/SEED'+str(seed)+'/logs'
        step,rew = tdlog2np(path_,'return_per_epi')
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
    fig,ax=plt.subplots(figsize=(5,5))
    #ax.plot(smt_maxline,color="green", label='best seed')
    #ax.plot(smtlist.transpose(),color="gray",alpha=0.2)
    #ax.plot(smt_bestline,color="green", label='best seed')
    ax.plot(smt_avgline,color="orange",alpha=1., label='avg seed')
    ax.fill_between([i for i in range(len(smt_maxline))],smt_avgline-smt_stdline,np.minimum((smt_avgline+smt_stdline),smt_maxline),facecolor='orange',alpha=0.4,label='std')
    ax.fill_between([i for i in range(len(smt_maxline))],smt_minline,smt_maxline,facecolor='gray',alpha=0.15, label='minmax')
    #ax.legend()
    #ax=plt.gca()
    ax.set_xlim([0,numiter])
    ax.set_ylim(yrange)
    #plt.ylabel('average reward')
    #ax.axes.get_xaxis().set_visible(False)
    #ax.axes.get_xaxis().set_ticklabels([])
    #ax.axes.get_xaxis().axhline(y=0,color='black')
    #ax.yaxis.set_ticklabels([])
    ax.axhline(y=0,color='black',linewidth=.5)
    #lstm_dentifier = LSTM_info_from_path(path)
    #plt.title(lstm_dentifier)
    plt.savefig(path+'/Train_reward_Learningcurves.png')
    plt.clf()

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
    #ax.plot(smt_bestline,color="green", label='best seed')
    ax.plot(step,smt_avgline,color="orange",alpha=1., label='avg seed')
    ax.fill_between(step,raw_minline,raw_maxline,facecolor='orange',alpha=0.35, label='minmax')
    ax.fill_between(step,smt_avgline-smt_stdline,np.minimum((smt_avgline+smt_stdline),smt_maxline),facecolor='gray',alpha=0.3,label='std')
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

root="./results/results_Phase3simp/test_lstm_simp"
#root="./results/results_Phase3simp/ppo/MemTask-U1"
#root="./results/results_Phase3/ppo/M5x5Fixed"
#root="./results/results_Phase3/ppo/M5x5Fixed/gat2-q/emb24_itT5/lstm_Dual_24_1/NFM_ev_ec_t_dt_at_um_us"
#root="./results/results_Phase3/ppo/M5x5Fixed/gat2-v/emb24_itT5/lstm_None/NFM_ev_ec_t_dt_at_um_us"

# USE FOR LSTM
for path in [x[0] for x in os.walk(root)]:
    if os.path.isfile(path+'/train-parameters.txt'):
        #path="./results/results_Phase3/ppo/MemTask-U1/gat2-q/emb24_itT5/lstm_Dual_24_1/NFM_ev_ec_t_dt_at_um_us/omask_freq0.2/bsize48" #folderpath
        CreateLearningCurvePlots(path=path, n_smoothing=15, nseeds=10, numiter=500, yrange=[-9.,9.])

#path="./results/results_Phase1/DQN/Manhattan5x5_VariableEscapeInit/etUt/tensorboard"
#CreateLearningCurvePlotsPhase1(path=path, n_smoothing=5, nseeds=5, numiter=None, yrange=[-10.,5.])