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

class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(432 * 432, 128), nn.ReLU(), nn.Linear(128, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 432 * 432))

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x = batch
        x = x.view(x.size(0), -1)
        x = x.float()
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

if __name__ == "__main__":

    import argparse

    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default="test")
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--config', type=str, default="one-gpu.yaml")
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--cpus', type=int, default=8)

    experiment_no = len(os.listdir("checkpoints")) + 1
    default_save = "checkpoints/exp{}".format(experiment_no)
    parser.add_argument('--save_path', type=str, default=default_save)
    parser.add_argument('--filename', type=str, default="test")

    args = parser.parse_args()
    print(f'Agrs: {args}')
    config = get_config(args.config)

    datafolder = "data/" + args.folder

    batch_size = args.batch_size
    graph_dataset = ZeoliteDataset(datafolder)
    graph_train = DataLoader(graph_dataset,batch_size=batch_size,shuffle=True,num_workers=args.cpus)

    # for i in graph_train:
    #     print(i.get_device())

    model = LitAutoEncoder()
    from pytorch_lightning.callbacks import ModelCheckpoint
    # from lightning.pytorch.loggers import CSVLogger

    # DEFAULTS used by the Trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath =args.save_path,
        filename=args.filename,
        every_n_epochs =2,
        verbose=True,
        monitor='train_loss',
        save_top_k = args.epochs//10,
        mode="min"
    )

    print(args.save_path)

    # logger = CSVLogger(save_dir=args.save_path,flush_logs_every_n_steps=1)
    # from lightning.pytorch.loggers import TensorBoardLogger

    # default logger used by trainer (if tensorboard is installed)
    # logger = TensorBoardLogger(save_dir=os.getcwd(), version=1, name="lightning_logs")
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator="gpu",default_root_dir=args.save_path, callbacks=checkpoint_callback)
    # trainer = pl.Trainer(fast_dev_run=args.debug,devices=args.gpus, accelerator="gpu", strategy='ddp_find_unused_parameters_true',max_epochs=args.epochs,default_root_dir=args.save_path,callbacks=checkpoint_callback)
    # model.hparams.batch_size = args.batch_size
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
