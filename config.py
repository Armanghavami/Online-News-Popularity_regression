import pandas as pd 
import wandb
import shap 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , MinMaxScaler

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset,DataLoader

from sklearn.metrics import mean_absolute_error,r2_score,root_mean_squared_error , mean_squared_error
import numpy as np 

__all__ = ["MinMaxScaler","pd","train_test_split","StandardScaler","torch","nn","TensorDataset","DataLoader","F","mean_absolute_error","r2_score","root_mean_squared_error","np","wandb","mean_squared_error","shap","plt"]




config = {

    "data_path" :"file.pth",
    "batch_size" :128,
    "data_normalization_method" :"z_score",
    "train_size" :0.7,
    "epochs":1,
    "learning_rate":0.05,
    "samples_count":3950,

    "weight_decay":0.0002930081658981912,
    "lr_factor":0.5,
    "lr_patioence":3,

    "early_stopping_delta":0,
    "early_stopping_patience":20,

    "l1_lambda":0.0006115926667062532,
    "saved_model":True,
    "continue_training":True,


    "layer_config":[
  {"size":128, "batch_norm":True, "dropout":0.6, "activation_function":"leaky_relu"},
  {"size":64, "batch_norm":True, "dropout":0.6, "activation_function":"leaky_relu"},
  {"size":64, "batch_norm":True, "dropout":0, "activation_function":"leaky_relu"},
  {"size":64, "batch_norm":True, "dropout":0, "activation_function":"leaky_relu"},
  {"size":1, "batch_norm":False, "dropout":0, "activation_function":"None"}
]
,
    

}
#"""1e-3,"""
#"""5e-5,"""

"""

[

        {"size":32,"batch_norm":True,"dropout":0.4,"activation_function":"leaky_relu"},
        {"size":64,"batch_norm":True,"dropout":0.4,"activation_function":"leaky_relu"},
        {"size":32,"batch_norm":True,"dropout":0,"activation_function":"leaky_relu"},
        {"size":16,"batch_norm":True,"dropout":0,"activation_function":"leaky_relu"},
        {"size":1,"batch_norm":False,"dropout":0,"activation_function":"None"},

                    
                    ]"""