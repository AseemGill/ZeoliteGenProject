from data import *
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default="Already Added")

args = parser.parse_args()
print(f'Agrs: {args}')

datafolder = "data/" + args.data

if not os.path.exists(datafolder + "/processed"):
    os.mkdir(datafolder + "/processed")

graph_dataset = ZeoliteDataset(datafolder)

os.remove(datafolder + "/processed/pre_filter.pt")
os.remove(datafolder + "/processed/pre_transform.pt")