# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 11:54:40 2021

@author: Irene
"""
import pandas as pd
import random

def start_position_escaperoute(G):
    """
    start node offender (random or chosen)
    
    Parameters
    ----------
    G : networkx graph
        graph of size (N,N).

    Returns
    -------
    start_escape_route : tuple
        start node.

    """
    # random
    # start_escape_route = sample(list(G.nodes()),1)[0]
    
    # chosen
    start_escape_route = (5,5)
    
    return start_escape_route


def escape_route_anygraph(G, N, L, start_escape_route, direction_north):
    if direction_north == True:
        
        walk = []
        node = start_escape_route
        walk.append(node)
                
        # weighted direction 
        left=20
        right=20
        down=0      # The offender cann't go back to precious node, no u-turn
        up=60
        
             
        for i in range(L):      
            list_neighbor = list(G.neighbors(node))
            number_of_neighbors = len(list_neighbor)            
            probs = tuple([G.edges[node,neighbor]['N_pref'] for neighbor in list_neighbor]) 
            
            nextnode = random.choices(list_neighbor, weights=probs)[0]
            #nextnode = random.choice(list_neighbor) 
            
            walk.append(nextnode)    
            node = nextnode
        
    else: #direction_north = False
            
        walk = [] 
        ### Generate random escape route 
        node = start_escape_route
        walk.append(node)
        
        #previous_node = node
        
        for i in range(L):      
            list_neighbor = list(G.neighbors(node))
            
            if i == 0:
                previous_node = node
                nextnode = random.choice(list_neighbor)

            else:
                #exclude previous node for 'normal' walk
                if previous_node in (list_neighbor) and len(list_neighbor)>1:
                    list_neighbor.remove(previous_node)
                
                #save previous node
                previous_node = node
                nextnode = random.choice(list_neighbor)
            
            walk.append(nextnode)
            node=nextnode
        
    return walk


def escape_route(G, N, L, start_escape_route, direction_north):
    """
    Generate an escape route from a starting position

    Parameters
    ----------
    G : networkx graph
        graph of size (N,N).
    N : int
        width of network.
    L : int
        length of escape route.
    start_escape_route : tuple
        starting node of offender.
    direction_north : bool
        whether or not the offender has a preference for going straight.

    Returns
    -------
    walk : list
        escape route (random walk) given starting position.

    """
    
    if direction_north == True:
        
        walk = []
        node = start_escape_route
        walk.append(node)
                
        # weighted direction 
        left=20
        right=20
        down=0      # The offender cann't go back to precious node, no u-turn
        up=60
        
             
        for i in range(L):      
            list_neighbor = list(G.neighbors(node))
            number_of_neighbors = len(list_neighbor)            
            
            if number_of_neighbors == 4:
                nextnode = random.choices(list_neighbor, weights=(left, right, down, up))[0]
                
            
            elif number_of_neighbors == 3:
                if node[0] == 0: # most left node
                    nextnode = random.choices(list_neighbor, weights=(right, down, up))[0]
                elif node[1] == 0: # bottem node
                    nextnode = random.choices(list_neighbor, weights=(left, right, up))[0]
                elif node[0] == N-1: # most right node
                    nextnode = random.choices(list_neighbor, weights=(left, down, up))[0]
                elif node[1] == N-1: # top node
                    # not leaving the grid
                    #node = random.choices(list_neighbor, weights=(left, right, down))[0]
                    # staying on top node
                    nextnode = node 
                      
            # 'else' does not happen for a manhatten graph, but may happen for other cases
            else:
                nextnode = random.choice(list_neighbor) 
            
            walk.append(nextnode)    
            node = nextnode
        
    else: #direction_north = False
            
        walk = [] 
        ### Generate random escape route 
        node = start_escape_route
        walk.append(node)
        
        #previous_node = node
        
        for i in range(L):      
            list_neighbor = list(G.neighbors(node))
            
            if i == 0:
                previous_node = node
                nextnode = random.choice(list_neighbor)

            else:
                #exclude previous node for 'normal' walk
                if previous_node in (list_neighbor):
                    list_neighbor.remove(previous_node)
                
                #save previous node
                previous_node = node
                nextnode = random.choice(list_neighbor)
            
            walk.append(nextnode)
            node=nextnode
        
    return walk

import multiprocessing as mp
def mutiple_escape_routes(G, N, L, R, start_escape_route, direction_north, start_units, graph_type='Manhattan'): 
    
    """
    Generate a dataframe of multiple random walks (escape routes)
    
    Parameters
    ----------
    G : networkx graph
        graph of size (N,N).
    N : int
        width of network.
    L : int
        length of escape route.
    R : int
        number of escape routes.
    start_escape_route : tuple
        starting node of offender.
    direction_north : bool
        whether or not the offender has a preference for going straight.
    start_units : list of tuples
        starting nodes of the police units.
        

    Returns
    -------
    escape_routes_nodes : DataFrame (R, N*N)
        for each escape route, whether or not a node is visited. #where is this used?
    occupation_node : Series (N*N,)
        for each node, the number of visits. #where is this used?
    escape_routes_time : DataFrame (R,T)
        for each route, the nodes visited per time step.
    routes_time_nodes : DataFrame (R*T*V, 1); (V=N*N)
        rho_rtv: for each combination of r,t,v; the presence at a node.

    """
    # empty matrixes    
    routes_time = pd.DataFrame([])
    
    routes_time_nodes_index = pd.MultiIndex.from_product([range(R),range(1+L),G.nodes()],names=('route','time','node'))
    routes_time_nodes = pd.DataFrame(index=routes_time_nodes_index,columns=['presence'])
    

    ### Generate multiple walks
    ## Multiprocessing test
    # pool=mp.Pool(mp.cpu_count())
    # args=[(G, N, L, start_escape_route, direction_north) for i in range(R)]
    # results=pool.starmap(escape_route,args)
    # for i,walk in enumerate(results):
    #     for t in range(len(walk)):
    #         routes_time_nodes.loc[(i,t,walk[t])]['presence'] = 1
    #         #routes_time_nodes.loc[(i, t, walk[t]), 'presence'] = 1
    #     routes_time = routes_time.append([walk], ignore_index=True)        
    for i in range(R):
        if graph_type == 'Manhattan':
            walk = escape_route(G, N, L, start_escape_route, direction_north) 
        else:
            walk = escape_route_anygraph(G, N, L, start_escape_route, direction_north) 
        for t in range(len(walk)):
            routes_time_nodes.loc[(i,t,walk[t])]['presence'] = 1
            #routes_time_nodes.loc[(i, t, walk[t]), 'presence'] = 1
        routes_time = routes_time.append([walk], ignore_index=True)
    
    routes_time_nodes = routes_time_nodes.fillna(0)
    
    
    return routes_time_nodes, routes_time