from data import *
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default="Already Added")
parser.add_argument('--size', type=int, default=8)


args = parser.parse_args()
print(f'Agrs: {args}')

datafolder = "data/" + args.data

if not os.path.exists(datafolder + "/processed"):
    os.mkdir(datafolder + "/processed")

if args.size == 8:  
    graph_dataset = ZeoliteDataset(datafolder)
else:
    graph_dataset = Zeolite_32_Dataset(datafolder)

os.remove(datafolder + "/processed/pre_filter.pt")
os.remove(datafolder + "/processed/pre_transform.pt")