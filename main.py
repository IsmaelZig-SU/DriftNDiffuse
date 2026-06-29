import torch, pickle, os
import pandas as pd
from src.Experiment import Experiment


torch.manual_seed(99)
import argparse

if __name__ == "__main__":
    #Parsing arguments
    parser = argparse.ArgumentParser(description='LatentSDE')

    #Training Params
    parser.add_argument('--load_epoch',             type = int, default = 0 ,        help = "loads model at a particular epoch for training")
    parser.add_argument('--pred_horizon',           type = int, default = 40,        help = "Number of steps to predict over while calculating loss")

    #Models
    parser.add_argument('--seq_model',  type = str, default = "TransformerModel",  help = "Sequence model to be used for the training")
    parser.add_argument('--AE_Model',   type = str, default = "AE", help = "Autoencoder model to be used for the training")

    #training Params ARGS
    parser.add_argument('--lr',      type = float, default=5e-5)
    parser.add_argument('--nepochs', type = int,   default=3000, help = "Number of epochs for training")
    parser.add_argument('--lambda_stateloss', type = float, default = 1.0, help = "Direct supervision weight")
    parser.add_argument('--beta', type = float, default = 1e-3, help = "regularization beta term for KLD")

    #AUTOENCODER Params ARGS
    parser.add_argument('--num_obs',            type = int,   default=64,   help = "Latent Size of the Autoencoder")

    #Transfo Params ARGS
    parser.add_argument('--seq_len',          type = int,   default=20,     help = "length of the sequence for memory term")
    parser.add_argument('--nattblocks',       type = int,   default=4,     help = "Number of attention blocks in the transformer")
    parser.add_argument('--nheads',       type = int,   default=8,     help = "Number of attention heads")
    parser.add_argument('--hidden_dim',       type = int,   default=64,     help = "Number of attention heads")
    parser.add_argument('--bound_sup', type = int,   default=2,     help = "Maximum log var diffusion")
    parser.add_argument('--bound_inf', type = int,   default=-6,     help = "Minimum log var diffusion")


    #Data Params ARGS
    parser.add_argument('--ntransients', type = int,   default = 0, help = "number of trainsients to discard in the intial part of the dataset")
    parser.add_argument('--nenddata',    type = int,   default = None,  help = "if we want to skip last parts of the dataset")
    parser.add_argument('--bs',          type = int,   default = 64,   help = "BatchSize")
    parser.add_argument('--train_size',  type = float, default = 0.8,   help = "Train Data Proportion")

    #Directory Params ARGS
    parser.add_argument('--exp_dir',         type = str, default = "D:/model_SDE/Trained_Models/",   help = "Directory for the Experiment") #D:/model_SDE/
    parser.add_argument('--load_exp_name',   type = str, default = "",   help = "Name of the experiment to be loaded")
    parser.add_argument('--data_dir',        type = str, default = "D:/data/Kolmogorov_SDE/KS_10_ens_nu0.8_512.npy", help = "Directory for the Data") #kolmo_120s_10traj_90, KS_10_ens_nu0.8_512
    parser.add_argument('--nsave',           type = int,   default = 500, help = "save every nsave number of epochs")
    parser.add_argument('--no_save_model',   action = 'store_false',     help = "doesn't save model")
    parser.add_argument('--info',            type = str, default = "fs",  help = "extra infomration to be added to the experiment name")

    args = parser.parse_args()
    #############################################################################
    
    if args.load_epoch == 0:

        lat_sde = Experiment(args)
        lat_sde.main_train(load_model = False)

    if args.load_epoch != 0:

        lat_sde = Experiment(args)
        lat_sde.main_train(load_model = True)


