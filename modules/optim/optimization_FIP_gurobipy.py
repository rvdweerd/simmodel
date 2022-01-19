# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 14:18:52 2021

@author: Irene
"""
import copy
import pandas as pd
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import random
import gurobipy as gp
from gurobipy import GRB, Model, quicksum

def unit_ranges(start_units, U, G, L):
    """
    # empty matrixes    
    routes_time = pd.DataFrame([])
    
    routes_nodes_index = pd.MultiIndex.from_product([range(R),G.nodes()],names=('route','node'))
    routes_nodes = pd.DataFrame(index=routes_nodes_index,columns=['presence'])
    
    ### Generate multiple walks
    for i in range(R):
        walk = escape_route(G, N, L, start_escape_route, direction_north) 
        for t in range(len(walk)):
            routes_nodes.loc[(i, walk[t]), 'presence'] = 1
        routes_time = routes_time.append([walk], ignore_index=True)

    routes_nodes = routes_nodes.fillna(0)
    """
    
    units_range_index = pd.MultiIndex.from_product([range(U), range(1+L),G.nodes()],names=('unit', 'time', 'node'))
    units_range_time = pd.DataFrame(index=units_range_index,columns=['inrange'])
    
    for u in range(U):
        for t in range(1+L):
            neighbors = list(nx.single_source_shortest_path_length(G, source=start_units[u], cutoff=t).keys())       
            for neighbor in neighbors:
                units_range_time.loc[(u,t,neighbor)]['inrange'] = 1
    
    units_range_time = units_range_time.fillna(0)
    
    return units_range_time

def optimization_alt(G, U, routes_time_nodes, units_range_time, R, V, T, labels, start_units, nodes_dict=None):
    # Disable verbose output of gurobi
    env=gp.Env(empty=True)
    env.setParam('Outputflag', 0)
    env.start()
    m = Model('FIP_unittravel', env=env) 

    # Create variables (have to be vectors because gurobi supports at msot conditions vec=vec and not matrix=matrix)
    pi_uv = m.addMVar((U * V), vtype=GRB.BINARY, name="pi_uv")
    z_r = m.addMVar(R, vtype=GRB.BINARY, name="z_r")
    
    # Create matrix for 5.28 constraint (it is U * V * V but can be made sparse if necessary)
    #pi_uv_const = np.repeat(np.identity(U), repeats=V, axis=1)
    data = np.ones(shape=(U, 1, V))
    indptr = np.arange(U + 1, dtype=int)
    indices = np.arange(U, dtype=int)
    pi_uv_const = sp.bsr_matrix((data, indices, indptr), blocksize=(1, V), shape=(U, U * V))

    # Objective
    m.setObjective(quicksum(z_r), GRB.MAXIMIZE)

    # Interception constraint
    alpha_rtv = routes_time_nodes.to_numpy().reshape((R, T, V))
    tau_utv = units_range_time.to_numpy().reshape((U, T, V))
    
    # Can get rid of T index by summing over it, used in 5.29 constraint
    z_r_const = np.einsum('ijk,ljk->ilk', alpha_rtv, tau_utv).reshape(R, U * V)
    
    # 5.28 constraint
    m.addConstr(pi_uv_const @ pi_uv <= 1)

    # 5.29 constraint
    m.addConstr(z_r <= z_r_const @ pi_uv)

    m.optimize()

    z_r_np = z_r.x
    routes_intercepted = {i: z_r_np[i] for i in range(z_r_np.shape[0])}
    pi_uv_np = pi_uv.x.reshape(U, V) 

    # create dictionary to translate nodes to node numbers
    if nodes_dict==None:
        i = 0
        nodes_dict = {}
        for node in G.nodes():
            nodes_dict[i] = node
            i += 1
    else:
        i = 0
        for node in G.nodes():
            assert nodes_dict[i] == node
            i += 1

    # list of unit nodes from results
    list_unit_nodes = list(start_units)
    for agent in range(U):
        for pos in range(V):
            if pi_uv_np[agent, pos] == 1:
                list_unit_nodes[agent] = nodes_dict[pos]

    return routes_intercepted, list_unit_nodes
