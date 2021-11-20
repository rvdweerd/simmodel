# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 14:17:39 2021

@author: Irene
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import random
from gurobipy import *

def color_nodes(G, routes_time_nodes, start_escape_route, list_unit_nodes):
    """
    Generates a list of the color codes for each node for plotting

    Parameters
    ----------
    routes_time_nodes : DataFrame of size (R*T*V, 1)
        For each combination of r,t,v whether or not a route is present.
    start_escape_route : tuple
        node of start escape route.
    results_Piutv : DataFrame of size (U*T, 3)
        For each unit, at each time, the node at which the unit is present.
    t : int
        time step being plotted.

    Returns
    -------
    color_nodes_map : list of length N (number of nodes)
        color for each node.
    thickness_nodes_map : list of length N (number of nodes)
        thickness for each node.

    """
    """
    #create dictionary to translate nodes to node numbers
    i = 0
    nodes_dict = {}
    for node in G.nodes():
        nodes_dict[i] = node
        i += 1
        
    #list of unit nodes from results
    list_unit_nodes = []
    for unit, presence in units_places.items():
        if presence == 1:
            unitnr, unitnode = unit
            list_unit_nodes.append(unitnode)
    
    """
    
    ### Color nodes
    color_nodes_map = []
    thickness_nodes_map = []
    
    """
    # Set thickness of the nodes, based on occupation of the route
    max_thickness = 150.0
    min_thickness = 10.0
    max_occurance = R
    thickness_nodes = min_thickness + (occupation_node/max_occurance) * (max_thickness - min_thickness)
    """

    
    for node in G.nodes():
        #thickness_nodes_map.append(100)
        if node in list_unit_nodes:
            thickness_nodes_map.append(100)
            color_nodes_map.append('#0000ff') #blue 
        elif node == start_escape_route:     
            thickness_nodes_map.append(50)
            color_nodes_map.append('#7f0000') #dark red starting point offender
        # elif node in intercept_route:
        #     color_nodes_map.append('#255339') #green
        else:
            thickness_nodes_map.append(10)
            color_nodes_map.append('#bbc9db') #light grey 
    
    return color_nodes_map, thickness_nodes_map


def color_edges(G, routes_time, routes_intercepted, R):
    """
    Generates a list of the color codes for each edge for plotting

    Parameters
    ----------
    G : networkx graph
        graph of size (N,N).
    routes_time : DataFrame of size (R,T)
        the vertex of the escape route for each route and time.
    routes_intercepted : dict of length R
        For each route, whether or not it was intercepted.
    R : int
        number of routes.
    t : int
        time step being plotted.

    Returns
    -------
    color_edges_map : list of length E (number of edges)
        color for each edge.
    weight_edges : list of length E (number of edges)
        thickness for each edge.

    """

    
    edges_on_route = [] 
    edges_on_intercepted_route = []
    for i in range(R):
        edges_on_route1 = [(routes_time.iloc[i][j], routes_time.iloc[i][j+1]) for j in range(len(routes_time.iloc[i])-1)]
        edges_on_route2 = [(routes_time.iloc[i][j+1], routes_time.iloc[i][j]) for j in range(len(routes_time.iloc[i])-1)]
        if routes_intercepted.get(i) == 1:
            edges_on_intercepted_route.extend(edges_on_route1)
            edges_on_intercepted_route.extend(edges_on_route2)
        else:
            edges_on_route.extend(edges_on_route1)
            edges_on_route.extend(edges_on_route2)
    
    #remove duplicates
    edges_on_intercepted_route = list(dict.fromkeys(edges_on_intercepted_route))
    edges_on_route = list(dict.fromkeys(edges_on_route))
    
    edges_notintercepted_route = [x for x in edges_on_route if x not in edges_on_intercepted_route]
    
    ### Thickness edges
    # Normalise the thickness of the edges and color
    routes_time_flat = [item for sublist in routes_time.values.tolist() for item in sublist]
    max_occurance = routes_time_flat.count(max(routes_time_flat))
    max_thickness = 10
    min_thickness = 1
        
    ### Color and thickness edges
    # Set the thickness of the edges
    weight_edges = []               
    # Color the edges                  
    color_edges_map = []
    for e in G.edges():
        #weight_edges.append(1)
        #if all((e not in edges_on_route, e not in edges_on_intercepted_route)):
        #    color_edges_map.append('#323232')   #grey
        #    weight_edges.append(1)
            #weight_edges.append(min_thickness/2)
        if e in edges_on_intercepted_route:
            color_edges_map.append('#008000')   #green
            weight_edges.append(2)
            #weight_edges.append(min_thickness 
            #                    + ((edges_on_route.count(e) + edges_on_intercepted_route.count(e))/max_occurance) 
            #                    * (max_thickness - min_thickness))
        
        elif e in edges_notintercepted_route:  
            color_edges_map.append('#E57300')   #orange
            weight_edges.append(2)
            #weight_edges.append(min_thickness 
            #                    + ((edges_on_route.count(e) + edges_on_intercepted_route.count(e))/max_occurance) 
            #                    * (max_thickness - min_thickness))
            


        else: 
            color_edges_map.append('#323232')   #grey
            weight_edges.append(1)
        
    return color_edges_map, weight_edges


def plot_results(G, R, pos, routes_time, routes_time_nodes, start_escape_route, units_places, routes_intercepted):
    """
    generate figure for each t displaying: 
     - nodes and edges not visited in grey
     - units on nodes in blue
     - routes on nodes in orange
     - intercepted routes on edges in green
     - not intercepted routes on edges in orange
     
     For now, edges do not 'change color' after interception

    Parameters
    ----------
    G : networkx graph
        graph of size (N,N).
    routes_time_nodes : DataFrame of size (R*T*V, 1)
        For each combination of r,t,v whether or not a route is present.
    start_escape_route : tuple
        node of start escape route.
    results_Piutv : DataFrame of size (U*T, 3)
        For each unit, at each time, the node at which the unit is present.
    routes_intercepted : dict of length R
        For each route, whether or not it was intercepted.

    Returns
    -------
    None.

    """
    
      
    # Color nodes
    color_nodes_map, thickness_nodes = color_nodes(G, routes_time_nodes, start_escape_route, units_places)
    # Color edges
    color_edges_map, weight_edges = color_edges(G, routes_time, routes_intercepted, R)
    
    
    # Plot graph 
    nx.draw_networkx(G, pos=pos
                       , node_color = color_nodes_map
                       , edge_color = color_edges_map
                       , width = weight_edges
                       , node_size = thickness_nodes
                       , with_labels = False
                       #, labels=labels
                       )
    
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    
    plt.axis('off')
    #fig = plt.gcf()
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.savefig('figures/FIP_unittravel.png')
    plt.show()