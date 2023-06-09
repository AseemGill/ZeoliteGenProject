from __future__ import (division, print_function)
# import time

import networkx as nx
import pickle

import pandas as pd
import random
from torch_geometric.loader import DataLoader

import torch
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F

from arg_helper import *
from model_torch import *
# from args import *
from data import *
from data_parallel import *
from evaluation import *
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--graphfile', type=str, default="Already Added")
parser.add_argument('--ucfile', type=str, default="Already Added")
parser.add_argument('--folder', type=str, default="zeo")
parser.add_argument('--train', type=str, default="True")
parser.add_argument('--eval', type=str, default="False")
# parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--config', type=str, default="one-gpu.yaml")


args = parser.parse_args()
print(f'Agrs: {args}')
file_name = args.folder + "_train.pt"

max_num_nodes = 432
batch_size = 20
batch_share = 5 # What is batch sharing?
max_num_nodes_l = 12
max_num_nodes_g = 36
num_per_unit_cell = 60 # How many instances of the same unit cell graph will be included in each training and testing sample 
print("Zeo data is being used")
datafolder = "data/" + args.folder
graphfile = args.graphfile
ucfile = args.ucfile
epochs = args.epoch
config = get_config(args.config)

if args.folder != "Already Added":
    
    if not os.path.isdir(datafolder + "/processed"):
        os.mkdir(datafolder + "/processed")

    if not os.path.isdir(datafolder + "/raw"):
        os.mkdir(datafolder + "/raw")

# Make train and test data
graph_dataset = ZeoliteDataset(datafolder)
graph_train = DataLoader(graph_dataset,batch_size=batch_size,shuffle=True)
# graph_loader = Make_batch_data(num_pad = max_num_nodes, batch_size = batch_size, batch_share = batch_share)
# graph_train, graph_test = graph_loader.makeTrain(dataset = graphs_whole[:270*10], unit_cell = unit_cell_whole[:270*10], num_per_unit_cell = num_per_unit_cell)

# print(graph_train[0][0].shape)

# if args.train == "True":
#     seed = 666
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
        
#     model = GRANMixtureBernoulli(config = config, max_num_nodes = max_num_nodes, max_num_nodes_l = max_num_nodes_l, max_num_nodes_g = max_num_nodes_g, num_cluster = 4, num_layer = 3, batch_size = batch_size, dim_l = 512, dim_g = 512)
#     model = DataParallel(model, device_ids=config.gpus).to(config.device)
    
#     ################################ Training process #############################
#     # Set up optimizer
#     params = filter(lambda p: p.requires_grad, model.parameters())
#     optimizer = optim.Adam(params, lr=args.lr, weight_decay=0)
    
#     # Adjust learning rate
#     lr_scheduler = optim.lr_scheduler.MultiStepLR(
#             optimizer,
#             milestones=[10, 30, 50, 70, 100, 130, 160, 200, 250, 350, 500],
#             gamma=0.1)
    
#     # Save loss values
#     ## Total loss
#     total_loss_record = []
#     ## Reconstruction loss
#     adj_loss_record = []
#     ## KL loss
#     kl_loss_record = []
#     ## Contrastive loss
#     reg_loss_record = []
    
#     # Training iteration
#     for epoch in range(epochs):
#         model.train()
#         lr_scheduler.step()
        
#         # z_l_mu_record = []
#         # z_g_mu_record = []
#         # A_pred_record = []
#         for i in range(len(graph_train)):
#             optimizer.zero_grad()
#             total_loss, adj_loss, kl_loss, reg, A_tmp, zl, zg = model(*[(graph_train[i],)])
            
#             total_loss_record.append(total_loss.detach().cpu().numpy())
#             adj_loss_record.append(adj_loss.detach().cpu().numpy())
#             kl_loss_record.append(kl_loss.detach().cpu().numpy())
#             reg_loss_record.append(reg.detach().cpu().numpy())
    
#             print("epoch: ", epoch, "iter: ", i, "total loss: ", total_loss, "adj loss: ", adj_loss, "kl loss: ", kl_loss, "reg loss: ", reg)
    
#             total_loss.backward()
#             optimizer.step()


################
print(get_config)

print(config.gpus)
print(config.device)
# Initialize model
if args.train == "True":
    seed = 666
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    model = GRANMixtureBernoulli(config = config, max_num_nodes = max_num_nodes, max_num_nodes_l = max_num_nodes_l, max_num_nodes_g = max_num_nodes_g, num_cluster = 4, num_layer = 3, batch_size = batch_size, dim_l = 512, dim_g = 512)
    model = DataParallel(model, device_ids=config.gpus).to(config.device)

 
    
    ################################ Training process #############################
    # Set up optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=0)
    
    # Adjust learning rate
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[10, 30, 50, 70, 100, 130, 160, 200, 250, 350, 500],
            gamma=0.1)
    
    # Save loss values
    ## Total loss
    total_loss_record = []
    ## Reconstruction loss
    adj_loss_record = []
    ## KL loss
    kl_loss_record = []
    ## Contrastive loss
    reg_loss_record = []
    
    # Training iteration
    for epoch in tqdm(range(epochs)):
        model.train()
        lr_scheduler.step()
        
        # z_l_mu_record = []
        # z_g_mu_record = []
        # A_pred_record = []
        it = 0

        for batch in graph_train:
            optimizer.zero_grad()
            total_loss, adj_loss, kl_loss, reg, A_tmp, zl, zg = model(*[(batch,)])
            
            total_loss_record.append(total_loss.detach().cpu().numpy())
            adj_loss_record.append(adj_loss.detach().cpu().numpy())
            kl_loss_record.append(kl_loss.detach().cpu().numpy())
            reg_loss_record.append(reg.detach().cpu().numpy())
    

            it += 1
            
            total_loss.backward()
            optimizer.step()
    

        if (epoch + 1) % 50 == 0:
            torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, file_name)
            
        print("epoch: ", epoch, "iter: ", it, "total loss: ", total_loss, "adj loss: ", adj_loss, "kl loss: ", kl_loss, "reg loss: ", reg)


if args.eval == "True":
    print(file_name)
    model = GRANMixtureBernoulli(config = config, max_num_nodes = max_num_nodes, max_num_nodes_l = max_num_nodes_l, max_num_nodes_g = max_num_nodes_g, num_cluster = 4, num_layer = 3, batch_size = batch_size, dim_l = 512, dim_g = 512)
    model.load_state_dict(torch.load(file_name)['model_state_dict'])
    model.eval()
    
    graph_gen_test = []
    
    with torch.no_grad():
        for i in range(len(graph_test)):
            total_loss, adj_loss, kl_loss, reg, A_tmp, zl, zg = model(*[(graph_test[i],)])
            graph_gen_test.append(A_tmp)
            
    graph_test_true = []
    graph_test_pred = []
    print("Evaluating generated graphs:")
    for i in range(2):
        print("Processing ", i, "-th graph")
        for j in range(len(graph_gen_test[0])):
            for k in range(graph_gen_test[0][0].shape[0]):
                graph_test_true.append(nx.convert_matrix.from_numpy_matrix(graph_test[i][j][k, :, :].detach().cpu().numpy()))
                graph_test_pred.append(nx.convert_matrix.from_numpy_matrix(graph_gen_test[i][j][k, :, :]))
    print("Degree: ")
    print(compute_kld(metric_name = 'degree', generate_graph_list = graph_test_pred, real_graph_list = graph_test_true))
    
    print("Cluster: ")
    print(compute_kld(metric_name = 'cluster', generate_graph_list = graph_test_pred, real_graph_list = graph_test_true))

    print("Average clustering distance: ")
    print(compute_KLD_from_graph(metric = 'avg_clustering_dist', generated_graph_list = graph_test_pred, real_graph_list = graph_test_true))

    print("Density: ")
    print(compute_KLD_from_graph(metric = 'density', generated_graph_list = graph_test_pred, real_graph_list = graph_test_true))
    print("Evaluation completed!")