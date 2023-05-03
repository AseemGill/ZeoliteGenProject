from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import pytorch_lightning as pl

import torch
import torch.utils.data
import torch.nn.functional as F

from arg_helper import *
from data import *
from data_parallel import *
from evaluation import *
from torch_geometric.loader import DataLoader

from torch import distributed as dist
# torch.distributed.init_process_group(
#     backend='gloo',
#     init_method='env://'
# )

class GINLayer(pl.LightningModule):
    def __init__(self, num_feature, batch_size, eps):
        super().__init__()
        self.num_feature = num_feature
        self.eps = eps
        
        self.MLP_GIN = nn.Sequential(
            nn.Linear(self.num_feature, self.num_feature),
            nn.ReLU()
            )
        
    def forward(self, A, X):
        X_tmp = (1+self.eps)*X + torch.matmul(A, X)
        X_new = self.MLP_GIN(X_tmp)
        return X_new

class GRU_plain(pl.LightningModule):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, has_input=True, has_output=False, output_size=None):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.has_input = has_input
        self.has_output = has_output

        if has_input:
            self.input = nn.Linear(input_size, embedding_size)
            self.rnn = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        if has_output:
            self.output = nn.Sequential(
                nn.Linear(hidden_size, embedding_size),
                nn.ReLU(),
                nn.Linear(embedding_size, output_size)
            )

        self.relu = nn.ReLU()
        # initialize

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.25)
            elif 'weight' in name:
                nn.init.xavier_uniform(param,gain=nn.init.calculate_gain('sigmoid'))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, input_raw, hidden, pack=False, input_len=None):
        if self.has_input:
            input = self.input(input_raw)
            input = self.relu(input)
        else:
            input = input_raw
        if pack:
            input = pack_padded_sequence(input, input_len, batch_first=True)
        output_raw, hidden = self.rnn(input, hidden)
        if pack:
            output_raw = pad_packed_sequence(output_raw, batch_first=True)[0]
        if self.has_output:
            output_raw = self.output(output_raw)
        # return hidden state at each time step
        return output_raw



class GRANMixtureBernoulli(pl.LightningModule):
    def __init__(self, config, max_num_nodes, max_num_nodes_l, max_num_nodes_g, num_cluster, num_layer, batch_size, dim_l, dim_g):
        super().__init__()
        self.max_num_nodes_w = max_num_nodes
        self.num_layer_w = num_layer
        self.batch = batch_size
        self.num_cluster = num_cluster
        self.hidden_dim = self.max_num_nodes_w + self.max_num_nodes_w * self.num_cluster

        # Dimension of z_l and z_g
        self.dim_zl = dim_l
        self.dim_zg = dim_g

        # Encoder     
        ## GIN for local and global filters
        # self.eps_l = nn.Parameter(torch.zeros(self.num_layer_w))
        self.register_buffer("eps_l", nn.Parameter(torch.zeros(self.num_layer_w)))

        self.gin_l = torch.nn.ModuleList()
        for i in range(self.num_layer_w):
            self.gin_l.append(GINLayer(self.max_num_nodes_w, self.batch, self.eps_l[i]))
        # self.eps_g = nn.Parameter(torch.zeros(self.num_layer_w))
        self.register_buffer("eps_g", nn.Parameter(torch.zeros(self.num_layer_w)))

        self.gin_g = torch.nn.ModuleList()
        for i in range(self.num_layer_w):
            self.gin_g.append(GINLayer(self.max_num_nodes_w, self.batch, self.eps_g[i]))
        
        ## Compute mu and sigma for VAE
        self.mu_l = nn.Sequential(
            nn.Linear(self.max_num_nodes_w * self.num_layer_w * self.num_cluster, self.dim_zl),
            nn.ReLU(),
            nn.BatchNorm1d(self.dim_zl),
            nn.Linear(self.dim_zl, self.dim_zl))
        self.sigma_l = nn.Sequential(
            nn.Linear(self.max_num_nodes_w * self.num_layer_w * self.num_cluster, self.dim_zl),
            nn.ReLU(),
            nn.BatchNorm1d(self.dim_zl),
            nn.Linear(self.dim_zl, self.dim_zl),
            nn.ReLU())
        self.mu_g = nn.Sequential(
            nn.Linear(self.max_num_nodes_w * self.num_layer_w, self.dim_zg),
            nn.ReLU(),
            nn.BatchNorm1d(self.dim_zg),
            nn.Linear(self.dim_zg, self.dim_zg))
        self.sigma_g = nn.Sequential(
            nn.Linear(self.max_num_nodes_w * self.num_layer_w, self.dim_zg),
            nn.ReLU(),
            nn.BatchNorm1d(self.dim_zg),
            nn.Linear(self.dim_zg, self.dim_zg),
            nn.ReLU())
        
        # MLP for node clustering
        self.MLP_NodeClustering = nn.Sequential(
            nn.Linear(self.max_num_nodes_w * self.num_layer_w, self.num_cluster),
            nn.ReLU(),
            nn.BatchNorm1d(self.max_num_nodes_w),
            nn.Softmax(dim = 1))
        
        # Decoder
        # Import parameters
        self.max_num_nodes_l = max_num_nodes_l
        self.max_num_nodes_g = max_num_nodes_g
        # Local
        self.LocalPred = nn.Sequential(
            nn.Linear(self.dim_zl, self.dim_zl),
            nn.ReLU(),
            nn.Linear(self.dim_zl, self.max_num_nodes_l**2)
            )
        
        # Global
        self.GlobalPred = nn.Sequential(
            nn.Linear(self.dim_zg, self.dim_zg),
            nn.ReLU(),
            nn.Linear(self.dim_zg, self.max_num_nodes_g**2)
            )
        
        ## Link prediction between two unit cells
        self.AsPred = nn.Sequential(
            nn.Linear(self.dim_zl, self.dim_zl),
            nn.ReLU(),
            nn.Linear(self.dim_zl, self.max_num_nodes_l**2)
            )
        
    def ClusterAssign(self, X):
        # print("Cluster Assign X.Shape: {}".format(X.shape))
        nodeCluster = self.MLP_NodeClustering(X)
        nodeClusterIndex = torch.argmax(nodeCluster, dim = 2)
        nodeRowIndex = torch.arange(0, nodeCluster.shape[1])
        nodeClusterAssign = torch.zeros(X.shape[0], nodeCluster.shape[1], nodeCluster.shape[2])
        nodeClusterAssign = nodeClusterAssign.type_as(X)
        # print("cluster assign: ", nodeClusterAssign.shape)
        for i in range(X.shape[0]):
            nodeClusterAssign[i, nodeRowIndex, nodeClusterIndex[i, :]] = 1
            
        cluster_tmp = torch.zeros(X.shape[0], self.num_cluster, self.num_cluster)
        nodeClusterAssign = nodeClusterAssign.type_as(X)
        clusterDegree = torch.sum(nodeClusterAssign, dim = 1)
        clusterDegreeInv = 1 / clusterDegree
        clusterDegreeInv[clusterDegreeInv == float("inf")] = 0
        for i in range(X.shape[0]):
            cluster_tmp[i, :, :] = torch.diag(clusterDegreeInv[i, :])
            cluster_tmp = cluster_tmp.type_as(X)

        
        nodeClusterNorm = torch.matmul(cluster_tmp, torch.transpose(nodeClusterAssign, dim0 = 1, dim1 = 2))
        
        return nodeClusterNorm
    
    def encoder(self, A, X):
        # print("ENCODER")
        # print(A.shape[0])
        
        z_l = torch.zeros(A.shape[0], self.max_num_nodes_w, self.max_num_nodes_w)
        z_l = z_l.type_as(A)
        i = 0
        for layer in self.gin_l:
            X = layer(A, X)
            z_l = torch.cat((z_l, X), dim = 2)
            i = i + 1
        
        z_l = z_l[:, :, self.max_num_nodes_w:]
        z_l = torch.matmul(self.ClusterAssign(z_l), z_l)
        z_l = z_l.view(X.shape[0], -1)
        z_l_mu = self.mu_l(z_l)
        z_l_sigma = self.sigma_l(z_l)
        
        z_g = torch.zeros(A.shape[0], self.max_num_nodes_w, self.max_num_nodes_w)
        z_g = z_g.type_as(A)
        i = 0
        for layer in self.gin_g:
            X = layer(A, X)
            z_g = torch.cat((z_g, X), dim = 2)
            i = i + 1
        
        z_g = z_g[:, :, self.max_num_nodes_w:]
        z_g = torch.sum(z_g, dim = 1)
        z_g_mu = self.mu_g(z_g)
        z_g_sigma = self.sigma_g(z_g)
        
        z_l_graph = z_l_mu + torch.randn(z_l_sigma.size()).type_as(z_l_mu) * torch.exp(z_l_sigma).type_as(z_l_mu)
        z_g_graph = z_g_mu + torch.randn(z_g_sigma.size()).type_as(z_g_mu) * torch.exp(z_g_sigma).type_as(z_g_mu)
        
        z_sigma_graph = torch.cat((z_l_sigma, z_g_sigma), dim = 1)
        z_mu_graph = torch.cat((z_l_mu, z_g_mu), dim = 1)

        print("ENCODER END")

        return z_l_graph.type_as(A), z_g_graph.type_as(A), z_l_mu.type_as(A), z_g_mu.type_as(A), z_mu_graph.type_as(A), z_sigma_graph.type_as(A)
    
    # Decoder process
    def decoder(self, z_l, z_g):


        Al = self.LocalPred(z_l).view(z_l.shape[0], self.max_num_nodes_l, -1)
        Al = torch.sigmoid(Al)

        Ag = self.GlobalPred(z_g).view(z_l.shape[0], self.max_num_nodes_g, -1)
        Ag = torch.sigmoid(Ag)
        n_g = Ag.shape[1]
        Ag = Ag * (1 - torch.eye(n_g).reshape(1, n_g, -1).repeat(Ag.shape[0], 1, 1).type_as(Ag))

        As = self.AsPred(z_l.view(z_l.shape[0], 1, -1)).view(z_l.shape[0], self.max_num_nodes_l, -1)
        As = torch.sigmoid(As)

        n_l = Al.shape[1]
        
        Al_tmp = torch.tile(Al, (n_g, n_g)).type_as(Ag)
        Al_mask = torch.eye(n_g).reshape(1, n_g, -1).repeat(Al.shape[0], 1, 1)
        Al_mask = Al_mask.type_as(Ag)
        Al_mask = torch.repeat_interleave(Al_mask, n_l, dim = 1)
        Al_mask = torch.repeat_interleave(Al_mask, n_l, dim = 2)
        A_tmp = Al_tmp * Al_mask

        As_tmp = torch.tile(As, (n_g, n_g))
        As_tmp = As_tmp.type_as(Ag)
        Ag_tmp = torch.repeat_interleave(Ag, n_l, dim = 1)
        Ag_tmp = torch.tril(torch.repeat_interleave(Ag_tmp, n_l, dim = 2),-1)
        
        A_pred = Ag_tmp * As_tmp + torch.transpose(Ag_tmp * As_tmp, dim0=1, dim1=2).type_as(Ag) + A_tmp

        print("DECODER END")

        return Al, Ag, As, A_pred

    # Combine encoder and decoder process
    def vae(self, A_pad, X):
        # encoder
        z_l, z_g, z_l_mu, z_g_mu, z_mu_graph, z_sigma_graph = self.encoder(A_pad, X)
        
        # decoder
        Al_pred, Ag_pred, As_pred, A_pred = self.decoder(z_l, z_g)
        return z_l_mu, z_g_mu, z_mu_graph, z_sigma_graph, Al_pred, Ag_pred, As_pred, A_pred


    def forward(self, inputgraph):
        # print("FORWARD")

        # print(inputgraph)
        # print(len(inputgraph))
        graph = (inputgraph[0].float(),)
        # print(graph)
        # Input data
        z_l_mu = ()
        z_g_mu = ()
        kl_loss = 0
        adj_loss = 0
        A_gen = []

        # print(graph.shape)
        for i in range(len(graph)):
            # A is the Adjaency matrix of the graph
            # X is an identity matrix with teh same dimensions
            A = graph[i]
            # print(A.shape[0])
            X = torch.eye(A.shape[1]).view(1, A.shape[1], -1).repeat(A.shape[0], 1, 1)
            X = X.type_as(A)
            # print(X.shape)
            # raise Exception("STOP")
            # print("BEFORE VAE\nX Shape: {}\n A Shape: {}\n".format(A.shape,X.shape))

            z_l_mu_tmp, z_g_mu_tmp, z_mu_graph, z_sigma_graph, Al_pred, Ag_pred, As_pred, A_pred = self.vae(A, X)
            
            z_l_mu = z_l_mu + (z_l_mu_tmp, )
            z_g_mu = z_g_mu + (z_g_mu_tmp, )
            kl_loss = kl_loss + torch.mean(-(0.5) * (1 + z_sigma_graph - z_mu_graph**2 - torch.exp(z_sigma_graph) ** 2))
            adj_loss = adj_loss + F.binary_cross_entropy(A_pred, A)
            print("KL and ADJ Loss")
            A_gen.append(A_pred.detach().cpu().numpy())
            
        # z_l_mu = torch.cat(z_l_mu, dim = 0).cpu()
        z_g_mu = torch.cat(z_g_mu, dim = 0)
        z_l_mu = torch.cat(z_l_mu, dim = 0)
        T= 0.2
        sim_matrix = torch.einsum('ik,jk->ij', z_l_mu, z_l_mu) / torch.sqrt(torch.einsum("i,j->ij", torch.einsum('ij,ij->i', z_l_mu, z_l_mu), torch.einsum('ij,ij->i', z_l_mu, z_l_mu)))
        sim_matrix = torch.exp(sim_matrix / T)
        sim_node = torch.sum(sim_matrix, dim = 0)
        if len(graph) > 1:
            sim_node_tmp = sim_matrix[:X.shape[0], :X.shape[0]]
            for j in range(len(graph) - 1):
                sim_node_tmp = torch.cat((sim_node_tmp, sim_matrix[X.shape[0]*(j+1):X.shape[0]*(j+2), X.shape[0]*(j+1):X.shape[0]*(j+2)]), dim = 1)
            sim_node = sim_node - torch.sum(sim_node_tmp, dim = 0)
 
        sim_node = sim_matrix / sim_node
        regularization = - torch.log(sim_node).mean()    
        
        # Total loss
        total_loss = 100000000*adj_loss + 1000000*kl_loss + 10000000*regularization
        # Output
        return total_loss, adj_loss, kl_loss, regularization, A_gen, z_l_mu, z_g_mu
    
    def training_step(self, batch, batch_idx):
        total_loss, adj_loss, kl_loss, reg, A_tmp, zl, zg = self(*[(batch,)])
        return total_loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    

max_num_nodes = 432
batch_size = 20
batch_share = 5 # What is batch sharing?
max_num_nodes_l = 12
max_num_nodes_g = 36
num_per_unit_cell = 60 # How many instances of the same unit cell graph will be included in each training and testing sample 


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default="test")
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--config', type=str, default="one-gpu.yaml")
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--epochs', type=int, default=100)

    args = parser.parse_args()
    print(f'Agrs: {args}')
    config = get_config(args.config)

    datafolder = "data/" + args.folder

    graph_dataset = ZeoliteDataset(datafolder)
    graph_train = DataLoader(graph_dataset,batch_size=batch_size,shuffle=True)

    for i in graph_train:
        print(i.get_device())

    model = GRANMixtureBernoulli(config = config, max_num_nodes = max_num_nodes, max_num_nodes_l = max_num_nodes_l, max_num_nodes_g = max_num_nodes_g, num_cluster = 4, num_layer = 3, batch_size = batch_size, dim_l = 512, dim_g = 512)
    # trainer = pl.Trainer(max_epochs=args.epochs, accelerator="gpu")
    trainer = pl.Trainer(fast_dev_run=args.debug,devices=args.gpus, accelerator="gpu", strategy="ddp",max_epochs=args.epochs)
    print(type(model))
    print(trainer.accelerator)
    trainer.fit(model, train_dataloaders=graph_train)
    # model = model.cuda()
    # x = torch.randn((2,432,432)).cuda()
    # print(x)
    # out = model(x)
    # print(out)



# class Trainer:
#     def __init__(self, rank, world_size):
#         self.rank = rank
#         self.world_size = world_size
#         self.log('Initializing distributed')
#         os.environ['MASTER_ADDR'] = self.args.distributed_addr
#         os.environ['MASTER_PORT'] = self.args.distributed_port
#         dist.init_process_group("gloo", rank=rank, world_size=world_size)

# if __name__ == '__main__':
#     world_size = torch.cuda.device_count()
#     mp.spawn(
#         Trainer,
#         nprocs=world_size,
#         args=(world_size,),
#         join=True)
