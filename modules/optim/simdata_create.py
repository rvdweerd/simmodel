# -*- coding: utf-8 -*-
"""
@author: rogier
"""
#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
#import networkx as nx
import random
#import time
import simdata_utils as su
from datetime import datetime
import pickle
import numpy as np
import cpuinfo
import gurobipy

def GenerateForSingleEscapeStartPosition(sp, required_dataset_size, save_every, update_every):
    cpuinf=cpuinfo.get_cpu_info()
    durations=[]
    markTimes=[]
    dirname = su.make_result_directory(sp)
    register, databank, iratios = su.LoadDatafile(dirname)
    # Generate
    while len(databank) < required_dataset_size:
        #ssu=[(21,0),(21,0),(21,0)]#None #specific start unit positions
        reg_entry, sim_instance, iratio, eval_time, marktimes = su.ObtainSimulationInstance(sp,register,specific_start_units=None,cutoff=1e5,print_InterceptRate=False,create_plot=False)
        # for k,v in sim_instance.items():
        #     print(k,v)
        markTimes.append(marktimes)
        if reg_entry[0] == -1:
            break
        durations.append(eval_time)
        register[reg_entry[0]] = reg_entry[1]
        databank.append(sim_instance)
        iratios.append(iratio)

        # process intermediate saves
        if len(databank) % save_every == 0:
            su.SaveResults(dirname,sp,register,databank,iratios)
            print('Intermediate save, dataset size:',len(databank))

        # generate progress updates
        if len(durations) % update_every == 0:
            print('\nOptimization runtime analysis')
            print(cpuinf['brand_raw'],cpuinf['arch'])
            print('Python version:',cpuinf['python_version'], 'Gurobi version:',gurobipy.gurobi.version())
            print('Model parameters: #nodes V:',sp.V,', #units U:',sp.U,', #timesteps L:',sp.L,', #routes R:',sp.R)
            print('\nSaved dataset size:',len(databank),'recent average time per run [s]: {:.3f}'.format(sum(durations)/len(durations)))
            #print(np.array(markTimes))
            print('Average mark times [s]:')
            mt_avg=np.mean(np.array(markTimes),axis=0)
            print('Get initial conditions.......: {:>8.3f}'.format(mt_avg[0]))
            print('Create multiple escape routes: {:>8.3f}'.format(mt_avg[1]))
            print('Get unit ranges..............: {:>8.3f}'.format(mt_avg[2]))
            print('Run optimizer loop...........: {:>8.3f}'.format(mt_avg[3]))
            print('Get intercepted routes.......: {:>8.3f}'.format(mt_avg[4]))          
            print('-------------------------------------------')          
            print('Total........................: {:>8.3f}'.format(np.sum(mt_avg)))      
            durations=[]

    su.SaveResults(dirname,sp,register,databank,iratios)

def GenerateForAllEscapeStartPosition(sp,required_dataset_size,save_every,update_every):
    durations=[]
    markTimes=[]
    dirname = su.make_result_directory(sp,allE=True)
    register, databank, iratios = su.LoadDatafile(dirname)
    # Generate for all E start positions
    #for i in range(sp.V):
    
    while len(databank) < required_dataset_size:
        i=random.randint(0,sp.V-1)
        sp.start_escape_route=sp.nodeid2coord[i]
        reg_entry, sim_instance, iratio, eval_time, marktimes = su.ObtainSimulationInstance(sp,register,cutoff=1e7,print_InterceptRate=False,create_plot=False)
        markTimes.append(marktimes)
        if reg_entry[0] == -1:
            break
        durations.append(eval_time)
        register[reg_entry[0]] = reg_entry[1]
        databank.append(sim_instance)
        iratios.append(iratio)

        # process intermediate saves
        if len(databank) % save_every == 0:
            su.SaveResults(dirname,sp,register,databank,iratios)
            print('Intermediate save, dataset size:',len(databank))

        # generate progress updates
        if len(durations) % update_every == 0:
            print('\nDataset size:',len(databank),'recent average time per run [s]:',sum(durations)/len(durations))
            print('Average mark times:')
            mt_avg=np.mean(np.array(markTimes),axis=1)
            print('MT1: {:.1f}'.format(mt_avg[0]))
            print('MT2: {:.1f}'.format(mt_avg[1]))
            print('MT3: {:.1f}'.format(mt_avg[2]))
            print('MT4: {:.1f}'.format(mt_avg[3]))
            print('MT5: {:.1f}'.format(mt_avg[4]))
            durations=[]

    su.SaveResults(dirname,sp,register,databank,iratios)


def GetDatabankForPartialGraph(sp, required_dataset_size, update_every):
    #cpuinf=cpuinfo.get_cpu_info()
    #durations=[]
    #markTimes=[]
    #dirname = su.make_result_directory(sp)
    #register, databank, iratios = su.LoadDatafile(dirname)
    register={}
    databank=[]
    iratios=[]

    # Generate
    while len(databank) < required_dataset_size:
        reg_entry, sim_instance, iratio, eval_time, marktimes = su.ObtainSimulationInstance(sp, register, specific_start_units=None, cutoff=1e5, print_InterceptRate=False, create_plot=False)
        #markTimes.append(marktimes)
        if reg_entry[0] == -1:
            break

        #durations.append(eval_time)
        register[reg_entry[0]] = reg_entry[1]
        databank.append(sim_instance)
        iratios.append(iratio)

        # generate progress updates
        #if len(durations) % update_every == 0:
            #pass
            # print('\nOptimization runtime analysis')
            # print(cpuinf['brand_raw'],cpuinf['arch'])
            # print('Python version:',cpuinf['python_version'], 'Gurobi version:',gurobipy.gurobi.version())
            # print('Model parameters: #nodes V:',sp.V,', #units U:',sp.U,', #timesteps L:',sp.L,', #routes R:',sp.R)
            # print('\nSaved dataset size:',len(databank),'recent average time per run [s]: {:.3f}'.format(sum(durations)/len(durations)))
            # #print(np.array(markTimes))
            # print('Average mark times [s]:')
            # mt_avg=np.mean(np.array(markTimes),axis=0)
            # print('Get initial conditions.......: {:>8.3f}'.format(mt_avg[0]))
            # print('Create multiple escape routes: {:>8.3f}'.format(mt_avg[1]))
            # print('Get unit ranges..............: {:>8.3f}'.format(mt_avg[2]))
            # print('Run optimizer loop...........: {:>8.3f}'.format(mt_avg[3]))
            # print('Get intercepted routes.......: {:>8.3f}'.format(mt_avg[4]))          
            # print('-------------------------------------------')          
            # print('Total........................: {:>8.3f}'.format(np.sum(mt_avg)))      
            # durations=[]
    return register, databank, iratios

