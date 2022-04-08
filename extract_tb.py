import traceback
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import matplotlib.pyplot as plt
import os

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

def CreateLearningCurvePlots(path, n_smoothing=20):
    rawlist=[]
    maxlen=0
    for seed in range(5):
        path_ = path+'/SEED'+str(seed)+'/logs'
        step,rew = tdlog2np(path_,'total_reward')
        rawlist.append(rew)
        maxlen = max(maxlen, len(rew))

    smtlist=[]
    smoothing_vector=[1/n_smoothing]*n_smoothing
    for i in range(len(rawlist)):
        p=maxlen-len(rawlist[i])
        rawlist[i]=np.pad(rawlist[i],(0,p),'constant',constant_values=np.mean(rawlist[i][-n_smoothing:]))#'edge')
        avg=np.convolve(smoothing_vector, rawlist[i],mode='valid')
        smtlist.append(avg)
    rawlist=np.array(rawlist) # (seeds, max_iterations)
    smtlist=np.array(smtlist) # (seeds, max_iterations)
    smt_avgline=np.mean(smtlist,axis=0) # (max_iterations, )
    smt_stdline=np.std(smtlist,axis=0)
    smt_minline=np.min(smtlist,axis=0)
    smt_maxline=np.max(smtlist,axis=0)
    raw_minline=np.min(rawlist,axis=0)
    raw_maxline=np.max(rawlist,axis=0)

    # # PLOT SMOOTHED CURVE FOR EACH SEED
    plt.plot(np.transpose(smtlist))
    plt.savefig(path+'/Train_reward_smoothed_per_seed.png')
    plt.clf()

    # # PLOT UNSMOOTHED CURVE FOR EACH SEED
    plt.plot(np.transpose(rawlist))
    plt.savefig(path+'/Train_reward_raw_per_seed.png')
    plt.clf()
    
    # PLOT SMOOTHED MAX,AVG,MINMAX RANGE, STD
    plt.plot(smt_maxline,color="green", label='best seed')
    plt.plot(smt_avgline,color="gray",alpha=0.6, label='avg seed')
    plt.fill_between([i for i in range(len(smt_maxline))],smt_avgline-smt_stdline,np.minimum((smt_avgline+smt_stdline),smt_maxline),facecolor='gray',alpha=0.3,label='std')
    plt.fill_between([i for i in range(len(smt_maxline))],smt_minline,smt_maxline,facecolor='gray',alpha=0.15, label='minmax')
    plt.legend()
    ax=plt.gca()
    #ax.axes.get_xaxis().set_visible(False)
    ax.set_xlim([0,600])
    ax.set_ylim([-5.,9.])
    plt.ylabel('avg reward')
    plt.savefig(path+'/Train_reward_Learningcurves.png')
    plt.clf()

root="./results/results_Phase3/ppo/MemTask-U1"
for path in [x[0] for x in os.walk(root)]:
    if os.path.isfile(path+'/train-parameters.txt'):
        #path="./results/results_Phase3/ppo/MemTask-U1/gat2-q/emb24_itT5/lstm_Dual_24_1/NFM_ev_ec_t_dt_at_um_us/omask_freq0.2/bsize48" #folderpath
        CreateLearningCurvePlots(path=path)