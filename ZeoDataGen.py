#Import Packages
import ase
import os
from ase.io import read
import numpy as np
import csv
from ase.io.jsonio import read_json
import json
from scipy.stats import rankdata
from ase.visualize import view
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Dataset, Data, InMemoryDataset
from torch_geometric.utils import dense_to_sparse, degree, add_self_loops
import torch_geometric.transforms as T
from torch_geometric.utils import degree

import glob, os

import networkx as nx

import trimesh
import pickle

def threshold_sort(matrix, threshold, neighbors, reverse=False, adj=False):
    '''
    reverse = True --> reverses ranking
    adj = True --> does sorting
    ''' 

    mask = matrix > threshold
    distance_matrix_trimmed = np.ma.array(matrix, mask=mask) # Removes Values above set threshold

    # Generates a rank matrix ---> assigns values a integer corresponding to size eg. [5,1,3] --> [3,1,2]
    if reverse == False:
        distance_matrix_trimmed = rankdata(
            distance_matrix_trimmed, method="ordinal", axis=1
        )
    elif reverse == True:
        distance_matrix_trimmed = rankdata(
            distance_matrix_trimmed * -1, method="ordinal", axis=1
        )
    distance_matrix_trimmed = np.nan_to_num(
        np.where(mask, np.nan, distance_matrix_trimmed)
    )
    
    distance_matrix_trimmed[distance_matrix_trimmed > neighbors + 1] = 0

    if adj == False:
        # Returns OG matrix with values above threshold set to 0

        distance_matrix_trimmed = np.where(
            distance_matrix_trimmed == 0, distance_matrix_trimmed, matrix
        )
        return distance_matrix_trimmed
    elif adj == True:
        adj_list = np.zeros((matrix.shape[0], neighbors + 1))
        adj_attr = np.zeros((matrix.shape[0], neighbors + 1))
        for i in range(0, matrix.shape[0]):
            temp = np.where(distance_matrix_trimmed[i] != 0)[0]
            adj_list[i, :] = np.pad(
                temp,
                pad_width=(0, neighbors + 1 - len(temp)),
                mode="constant",
                constant_values=0,
            )
            adj_attr[i, :] = matrix[i, adj_list[i, :].astype(int)]
        distance_matrix_trimmed = np.where(
            distance_matrix_trimmed == 0, distance_matrix_trimmed, matrix
        )
        return distance_matrix_trimmed, adj_list, adj_attr

# Loads Zeolite Structure

max_nodes = 9999

# os.chdir("./MOF_data")
zeo_uc = []
i = 1
for file in glob.glob("pcod2_new\\raw\\*.cif"): # List of Cifs of files --> glob.glob looks for .json or *<>    
    structure = ase.io.read(file) ### Relaxed Structure [No need for ]
    del structure[[atom.index for atom in structure if atom.symbol=='O']] # Removes Oxygens
    distance_matrix = structure.get_all_distances(mic=True) # Generates a pairwise distance matrix for all nodes
    num_of_nodes = distance_matrix.shape[0] # Returns number of nodes (~number of silicon atoms)
    if num_of_nodes <= max_nodes: # Only Files with less than max_nodes nodes
        zeo_uc.append(structure) # Append Unit Cells to array
    
    if i % 1000 == 0:
        print("Processed", i, "files")
        print("================================================")
    
    i = i + 1

unit_cell = []
zeo_graph = []
for i in range(len(zeo_uc)): # Iterate over unit cells
    print("Processing ",i,"-th graph", sep="")
    s1 = zeo_uc[i] # Select a Unit Cell
    distance_matrix = s1.get_all_distances(mic=True) # Compute Pairwise Distances
    # Thresholds distance matrix, all pairwise distances above threshold are set to 0, includes a max of 4 neighbours
    distance_matrix_trimmed = threshold_sort(distance_matrix,4,4,adj=False) # matrix, threshold, neighbors, reverse=False, adj=False
    distance_matrix_trimmed = torch.Tensor(distance_matrix_trimmed)

    # NOTE: Here a 0 represents no interaction (interaction weight = 0); 1 will represent an interaction weight of 1
    # Does it make sense to set interaction weights to 0 or 1???
    # If this line is commented out, graph interaction weights will be distance, which may also not be the best representatino
    # Perhaps we can use the Lennard-Jones Potential?
    distance_matrix_trimmed[distance_matrix_trimmed != 0] = 1 # If matrix value is not zero set to 1
    # print(distance_matrix_trimmed)

    # graph_tmp = nx.convert_matrix.from_numpy_matrix(distance_matrix_trimmed.numpy())
    zeo_graph.extend([nx.from_numpy_array(distance_matrix_trimmed.numpy())])
    unit_cell.extend([i])

max_uc_nodes = 0
for i in zeo_graph:
    if max_uc_nodes < (i.number_of_nodes()):
        max_uc_nodes = (i.number_of_nodes())
        print(max_uc_nodes)


unit_cell = []
zeo_graph = []
for i in range(len(zeo_uc)):
    print("Processing ",i,"-th graph", sep="")
    s1 = zeo_uc[i]
    for j in range(3):
        for k in range(3):
            for l in range(3):
                distance_matrix = s1.repeat((j + 1, k + 1, l + 1)).get_all_distances(mic=True)
                distance_matrix_trimmed = threshold_sort(distance_matrix,4,4,adj=False)
                distance_matrix_trimmed = torch.Tensor(distance_matrix_trimmed)
                distance_matrix_trimmed[distance_matrix_trimmed != 0] = 1
                # graph_tmp = nx.convert_matrix.from_numpy_matrix(distance_matrix_trimmed.numpy())
                # zeo_graph.extend([nx.from_numpy_array(distance_matrix_trimmed.numpy())] * 10)
                # unit_cell.extend([i] * 10)
                zeo_graph.extend([nx.from_numpy_array(distance_matrix_trimmed.numpy())])
                unit_cell.extend([i])


if not os.path.isdir("Repeated_Cells_1x"):
    os.mkdir("Repeated_Cells_1x")

if not os.path.isdir("Repeated_Cells_1x/raw"):
    os.mkdir("Repeated_Cells_1x/raw")

if not os.path.isdir("Repeated_Cells_1x/processed"):
    os.mkdir("Repeated_Cells_1x/processed")

with open('Repeated_Cells_1x/raw/ZeoGraphs.p', 'wb') as f:
    pickle.dump(zeo_graph, f) 

with open('Repeated_Cells_1x/raw/ZeoUnitCells.p', 'wb') as f:
    pickle.dump(unit_cell, f) 
