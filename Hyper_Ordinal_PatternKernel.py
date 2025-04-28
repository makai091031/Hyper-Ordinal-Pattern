# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 22:16:17 2024

@author: 
"""

import numpy as np

from Hyper_Ordinal_Pattern import Node_neighbors_Hyperedges,Hyper_Ordinal_Patterns_Of_Node2,delete_redundant_node




def Node_HyperOP_kernel(Brain_hypernetwork1,Brain_hypernetwork2,v):
#Brain_hypernetwork1,Brain_hypernetwork2 are two brain hyper-networks
    print("Calculate NHOP kernel...")
    Net1vistnode,Net1visitedge,Net1incidence_matrix=Hyper_Ordinal_Patterns_Of_Node2(Brain_hypernetwork1,v)
    Net2vistnode,Net2visitedge,Net2incidence_matrix=Hyper_Ordinal_Patterns_Of_Node2(Brain_hypernetwork2,v)
    NodeSets1=Node_neighbors_Hyperedges(Net1incidence_matrix,Net1visitedge)
    NodeSets2=Node_neighbors_Hyperedges(Net2incidence_matrix,Net2visitedge)
    NodeSets1=delete_redundant_node(NodeSets1)
    NodeSets2=delete_redundant_node(NodeSets2)
    common_elements = list(set(NodeSets1) & set(NodeSets2))
    
    Hg1=Brain_hypernetwork1.Hg
    Hg2=Brain_hypernetwork1.Hg

    Lap_net1=Hg1.laplacian().toarray()
    Lap_net2=Hg2.laplacian().toarray()
    
    E_vals1,E_vecs1 = np.linalg.eig(Lap_net1)
    E_vals2,E_vecs2 = np.linalg.eig(Lap_net2)
    Lap_Vec1=abs(E_vecs1)
    Lap_Vec2=abs(E_vecs2)
    
    if common_elements is not None:
        print("There are common elements between NodeSets1 and NodeSets2")
        Lap_net1_Fea=Lap_Vec1[common_elements,:]
        Lap_net2_Fea=Lap_Vec2[common_elements,:]
    else:
        print("No common elements between NodeSets1 and NodeSets2")
        Lap_net1_Fea=np.zeros(Brain_hypernetwork1.shape[0])
        Lap_net2_Fea=np.zeros(Brain_hypernetwork2.shape[0])


    K_NHOP=np.linalg.norm(np.mean(Lap_net1_Fea,axis=1)-np.mean(Lap_net2_Fea,axis=1))**2


    # K_NHOP=len(common_elements)
    
    return K_NHOP

#Calculate ordinal pattern based hyper-network (OPHN) kernel
def Ordinalpattern_for_HyperNet(Brain_hypernetwork1,Brain_hypernetwork2):
    print("Calculate OPHN kernel...")
    K_OPHN=0
    NodeNum=Brain_hypernetwork1.shape[0]
    for i in range(NodeNum):
        K_NHOP=Node_HyperOP_kernel(Brain_hypernetwork1,Brain_hypernetwork2,i)
        K_OPHN=K_OPHN+K_NHOP
    K_OPHN=K_OPHN/NodeNum
        
    return K_OPHN


def OPHN_Kernel_For_BrainNet(Brain_hypernetwork):
    SubNum=Brain_hypernetwork.shape[2]
    OPHNKernel=np.zeros((SubNum,SubNum))
    for i in range(SubNum):
        for j in range(i,SubNum):
            K_OPHN=Ordinalpattern_for_HyperNet(Brain_hypernetwork[:,:,i],Brain_hypernetwork[:,:,j])
            OPHNKernel[i,j]=K_OPHN
    OPHNKernel=OPHNKernel+OPHNKernel.T
    print("Calculate OPHN kernel matrix...")
    
    return OPHNKernel


def NHOP_Kernel_For_BrainNet(Brain_hypernetwork,v):
    SubNum=Brain_hypernetwork.shape[2]
    NHOPKernel=np.zeros((SubNum,SubNum))
    for i in range(SubNum):
        for j in range(i,SubNum):
            K_NHOP=Node_HyperOP_kernel(Brain_hypernetwork[:,:,i],Brain_hypernetwork[:,:,j],v)
            NHOPKernel[i,j]=K_NHOP
    NHOPKernel=NHOPKernel+NHOPKernel.T
    print("Calculate NHOP kernel matrix...")
    
    return NHOPKernel
    

