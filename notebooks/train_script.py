# %load_ext autoreload
# %autoreload 2
import numpy as np
import pandas as pd
import sys
sys.path.append('../scripts/')
from data.dataset_binary import GetData
from train.train_binary import train_binary_fn
from model.c3d import C3D
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
from functools import partial

obj = GetData(label_path='./train.csv')
obj.split_datasets() 
obj.perp_dataset()
obj.create_dataset()
obj.create_loaders(batch_size=64)

train_loader, val_loader, test_loader = obj.return_loaders()

wandb_dict = {
    "epochs": 30,
    "batch_size": 64,
    "learning_rate": 0.01,
    "model": "C3D",
    "dataset": "Binary",
    "optimizer": "SGD",
    "load_model" : True,
    "load_weights_path" : "../../weights/C3D/model_29.pth",
    "save_weights_path" : "../../weights/C3D_v1/"
}

model = C3D()

partial_binary_fn = partial(train_binary_fn, 
                            model = model, 
                            train_loader = train_loader, 
                            val_loader = val_loader, 
                            wandb_dict = wandb_dict)

partial_binary_fn(epochs = 30, 
                  save_weights_path='../../weights/C3D_v1/', 
                  load_model = '../../weights/C3D/model_29.pth')

