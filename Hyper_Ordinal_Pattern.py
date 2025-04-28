# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 23:04:09 2024

@author: makai
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 14:52:50 2024

@author: 
"""

import numpy as np

import itertools




def Hyperedge_neighbors_node(incidence_matrix,v):
    #return the hyperedge neightors of node v
    HyperedgeSet=np.where(incidence_matrix[v,:]==1)[0]
    return HyperedgeSet

def Hyperedge_neighbors_Nodes(incidence_matrix,V):
    #return the hyperedge neightors of nodes in V
    HedgeSets=[]
    for i in range(len(V)):
        HyperedgeSet=Hyperedge_neighbors_node(incidence_matrix,V[i])
        HedgeSets.append(HyperedgeSet)
    
    return HedgeSets


def Node_neighbors_Hyperedge(incidence_matrix,e):
    #return the node neightors of hyperedge e
    NodesSet=np.where(incidence_matrix[:,e]==1)
    return NodesSet

def Node_neighbors_Hyperedges(incidence_matrix,E):
    #return the hyperedge neightors of nodes in E
    NodeSets=[]
    for i in range(len(E)):
        Nodeset=Node_neighbors_Hyperedge(incidence_matrix,E[i])
        NodeSets.append(Nodeset)
    
    return NodeSets

#find the no common elemnts in V1 and V2
def NoCommonElements(V1,V2):
    #V1 and V2 are two numpy arrays
   V1diff=np.setdiff1d(V1, V2)
   V2diff=np.setdiff1d(V2, V1)
   
   V12= np.hstack((V1diff,V2diff))
   
   return V12

def delete_redundant_hyperE(E_V):
    result = list(itertools.chain.from_iterable(E_V))
    # result=np.concatenate(result,axis=0)
    result=np.unique(result)
    return result

def delete_redundant_node(Nodesets):
    result = list(itertools.chain.from_iterable(Nodesets))
    result=np.concatenate(result,axis=0)
    result=np.unique(result)
    return result


def Choose(vistnode,incidence_matrix,Hyperedge_weights,W_V,e_Max):
    W_V_sorted = np.sort(W_V)[::-1]
    i=0
    while(i<len(W_V_sorted)):
        if(W_V_sorted[i]<Hyperedge_weights[e_Max]):
            e_choose=np.where(Hyperedge_weights==W_V_sorted[i])[0][0]
            break
        else:
            i=i+1
    if(i!=len(W_V_sorted)):
        NodesSet=Node_neighbors_Hyperedge(incidence_matrix,e_choose)[0]
        v=np.setdiff1d(NodesSet, np.array(vistnode))
        v=v[0]
    else:
        v=-100
    
    return v



            


def Hyper_Ordinal_Patterns_Of_Node(Brain_hypernetwork,v):
    incidence_matrix=Brain_hypernetwork.InciMatrix #incidence matrix
    Hg=Brain_hypernetwork.Hg
    Hyperedge_weights=Brain_hypernetwork.weights
    vistnode=[]
    visitedge=[]
    vistnode.append(v)
    j=0
    w_Max=10000
    E=Hyperedge_neighbors_node(incidence_matrix,v)
    k=0
    while (j<len(E)):
        
        W_E=Hyperedge_weights[E]
        W_E_sorted= np.sort(W_E)[::-1]
        W_e_Max=W_E_sorted[j]
        e_Max=np.where(Hyperedge_weights==W_e_Max)[0][0]
        if (W_e_Max<w_Max):
            k=k+1
            w_Max=W_e_Max
            visitedge.append(e_Max)
            V=Node_neighbors_Hyperedge(incidence_matrix,e_Max)[0]
            V=np.setdiff1d(V, np.array(vistnode))
            if(len(V)>0):
                
                E_V=Hyperedge_neighbors_Nodes(incidence_matrix,V)
                E_V=delete_redundant_hyperE(E_V)
                E_V=np.setdiff1d(E_V, visitedge)
                W_V=Hyperedge_weights[E_V]
                v=Choose(vistnode,incidence_matrix,Hyperedge_weights,W_V,e_Max)
                if(v!=-100):
                    vistnode.append(v)
                    E=Hyperedge_neighbors_node(incidence_matrix,v)
                else:
                    break
            else:
                break
               
        else:
           j=j+1
    print("vistnode=",vistnode)
    print("visitedge=",visitedge)

        
    return vistnode,visitedge,incidence_matrix
        
    
    
    
    
    
    
    
    
    
