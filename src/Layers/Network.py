import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import inspect

import src.Layers.Autoencoder as Autoencoder
import src.Layers.transformer as Transformer

class Network(nn.Module):
    def __init__(self, exp_args : dict):
        super(Network, self).__init__()
        
        self.args        = exp_args
        self.select_models()
                
    def select_models(self):
        
        autoencoder_models = {name: member for name, member in inspect.getmembers(Autoencoder) if inspect.isclass(member)}
        seq_models         = {name: member for name, member in inspect.getmembers(Transformer) if inspect.isclass(member)}

        self.autoencoder = autoencoder_models[self.args["autoencoder_model"]](self.args) 
        self.transformer  = seq_models[self.args["seq_model"]](self.args)

    def _num_parameters(self):
        count = 0
        for name, param in self.named_parameters():
            print(name, param.numel())
            count += param.numel()
        return count



        

        


