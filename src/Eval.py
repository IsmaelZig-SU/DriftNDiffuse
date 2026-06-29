import torch
import torch.nn as nn
import pickle
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import statsmodels.api as sm
from scipy.stats import gaussian_kde
from src.Experiment import Experiment
from torch.utils.data import DataLoader

torch.manual_seed(99)

class Eval(Experiment):

    def __init__(self, exp_dir, exp_name):

        args = pickle.load(open(exp_dir + "/" + exp_name + "/args","rb"))
        #safety measure for new parameters added in model
            
        super().__init__(args)
        self.exp_dir = exp_dir
        self.exp_name = exp_name
            
##################################################################################################################
    def load_weights(self, epoch_num = 500, min_test_loss = False, min_train_loss = False):

        if min_test_loss:
            PATH = self.exp_dir+'/'+ self.exp_name+"/model_weights/min_test_loss".format(epoch=epoch_num)

        elif min_train_loss:
            PATH = self.exp_dir+'/'+ self.exp_name+"/model_weights/min_train_loss".format(epoch=epoch_num)

        else:
            PATH = self.exp_dir+'/'+ self.exp_name+"/model_weights/at_epoch{epoch}".format(epoch=epoch_num)
        
    
        checkpoint = torch.load(PATH)
        self.model.load_state_dict(checkpoint['model_state_dict'])

##################################################################################################################
    @staticmethod
    def state_mse(Phi,Phi_hat):
        '''
        Input
        -----
        Phi (torch tensor): [num_tajs timesteps statedim]
        Phi_hat (torch tensor): [num_tajs timesteps statedim]

        Returns
        -------
        StateMSE [timesteps]
        '''
        Phi_sm = Phi.to("cpu")
        Phi_hat_sm = Phi_hat.to("cpu")
        mseLoss     = nn.MSELoss(reduction = 'none')
        StateMSE    = mseLoss(Phi_sm, Phi_hat_sm) #[num_trajs timesteps statedim]
        # print(StateMSE.shape)
        StateMSE    = torch.mean(StateMSE, dim = (0,*tuple(range(2, StateMSE.ndim)))) #[timesteps]

        return StateMSE

##################################################################################################################
    @staticmethod 
    def calc_pdf(ke):
        '''
        Input
        -----
        ke (numpy array): [num_trajs timesteps 1]

        Returns
        -------
        StateMSE [timesteps]
        '''

        kde = gaussian_kde(ke)
        k = np.linspace(min(ke), max(ke), 10000)
        pdf = kde.evaluate(k)
        return k, pdf

##################################################################################################################
    @staticmethod
    def ccf_values(data1, data2):

        '''
        Calculates Cross Correlation Function

        Input
        -----
        data1 (ndarray): [num_trajs timesteps statedim]   
        data2 (ndarray): [num_trajs timesteps statedim]

        Returns
        -------
        CCF (ndarray): [num_trajs timesteps statedim] 
        '''

        p = data1
        q = data2
        p = (p - np.mean(p)) / (np.std(p) * len(p))
        q = (q - np.mean(q)) / (np.std(q))  
        c = np.correlate(p, q, 'full')
        return c


##################################################################################################################

    def predict_multistep(self, initial_conditions, timesteps):

            '''
            Input
            -----
            initial_conditions (torch tensor): [num_trajs, statedim]
            timesteps (int): Number timesteps for prediction

            Returns
            x (torch tensor): [num_trajs timesteps obsdim] observable vetcor
            Phi (torch tensor): [num_trajs timesteps statedim] state vector
            '''

            self.model.eval()
            initialisation = False
            Phi_n  = initial_conditions  
            # x_n, Phi_n, mu_n, log_var = self.model.autoencoder(Phi_n)    #[num_trajs obsdim]
            x_n, Phi_n  = self.model.autoencoder(Phi_n)
            x   = x_n[None,...].to("cpu")                    #[timesteps num_trajs obsdim]
            Phi = Phi_n[None, ...].to("cpu")                    #[timesteps num_trajs statedim]

            for n in range(timesteps):

                non_time_dims = (1,)*(x.ndim-1)   #dims apart from timestep in tuple form (1,1,...)
                if n >= self.seq_len:
                    i_start = n - self.seq_len + 1
                    x_seq_n = x[i_start:(n+1), ...].to(self.device)
                elif n==0:
                    # padding = torch.zeros(x[0].repeat(self.seq_len - 1, *non_time_dims).shape).to(self.device)
                    padding = x[0].repeat(self.seq_len - 1, *non_time_dims).to(self.device)
                    x_seq_n = x[0:(n+1), ...].to(self.device)
                    x_seq_n = torch.cat((padding, x_seq_n), 0)
                else:
                    # padding = torch.zeros(x[0].repeat(self.seq_len - n, *non_time_dims).shape).to(self.device)
                    padding = x[0].repeat(self.seq_len - n, *non_time_dims).to(self.device)
                    x_seq_n = x[1:(n+1), ...].to(self.device)
                    x_seq_n = torch.cat((padding, x_seq_n), 0)

                x_seq_n = torch.movedim(x_seq_n, 1, 0) #[num_trajs seq_len obsdim]
                x_seq_n = x_seq_n[:,:-1,:]
                x_nn  = self.model.transformer(x_seq_n)
                Phi_nn = self.model.autoencoder.recover(x_nn)

                x   = torch.cat((x,x_nn[None,...].detach().cpu()), 0)
                Phi = torch.cat((Phi,Phi_nn[None,...].detach().cpu()), 0)

            x      = torch.movedim(x, 1, 0)   #[num_trajs timesteps obsdim]
            Phi    = torch.movedim(Phi, 1, 0) #[num_trajs timesteps statedim]

            return x, Phi


    def get_latent_dynamics(self, phi_test) : 

        self.model.eval()
       
        x_n, mu, log_var = self.model.autoencoder.encode(phi_test)

        return x_n, mu, log_var


    def variational_UQ_scale(self, phi_test, ens_var) : 

        Phi_n_ens = []

        for i in range(ens_var) : 

            x_n, Phi_n, mu, log_var = self.model.autoencoder(phi_test)
            Phi_n_ens.append(Phi_n)

        Phi_n_ens = torch.stack(Phi_n_ens, dim = 0)

        return Phi_n_ens

    def forecast(self, initial_conditions, timesteps):

        '''
        Input
        -----
        initial_conditions (torch tensor): [t_in, statedim]
        timesteps (int): Number of timesteps for prediction

        Returns
        x (torch tensor): [num_trajs timesteps obsdim] observable vetcor
        Phi (torch tensor): [num_trajs timesteps statedim] state vector
        '''

        t_in = initial_conditions.shape[0]
        if t_in < self.seq_len:
            # repeat first row
            first_val = initial_conditions[0].unsqueeze(0)        # [1, dim]
            padding = first_val.expand(self.seq_len - t_in, *first_val.shape[1:])
            Phi_in = torch.cat((padding, initial_conditions), dim=0)

        elif t_in == self.seq_len:
            Phi_in = initial_conditions

        else:  # t_in > seq_len
            Phi_in = initial_conditions[-self.seq_len:, ...] 


        self.model.eval()
        m = Phi_in.shape[0]
        x_n, mu_n, log_var = self.model.autoencoder.encode(Phi_in)    #x : [t_in obsdim]

        for n in range(timesteps - 1):

            x_in = x_n[:-1, ...].unsqueeze(0)
            # print(x_in.shape, context.shape)
            x_nn   = self.model.transformer(x_in) #[1 obsdim]
            x_n  = torch.cat((x_n,x_nn), 0)
            mu_n = x_n[-self.seq_len:, ...]
        
        Phi    = self.model.autoencoder.recover(x_n)

        if t_in < self.seq_len : 
            return x_n[self.seq_len - t_in:, ...].unsqueeze(0), Phi[self.seq_len - t_in:, ...].unsqueeze(0)
        else : 
            return x_n.unsqueeze(0),Phi.unsqueeze(0) 


    def forecast_CNN(self, initial_conditions, timesteps):

        '''
        Input
        -----
        initial_conditions (torch tensor): [t_in, statedim]
        timesteps (int): Number of timesteps for prediction

        Returns
        x (torch tensor): [num_trajs timesteps obsdim] observable vetcor
        Phi (torch tensor): [num_trajs timesteps statedim] state vector
        '''

        t_in = initial_conditions.shape[0]
        if t_in < self.seq_len:
            # repeat first row
            first_val = initial_conditions[0].unsqueeze(0)        # [1, dim]
            padding = first_val.expand(self.seq_len - t_in, *first_val.shape[1:])
            Phi_in = torch.cat((padding, initial_conditions), dim=0)

        elif t_in == self.seq_len:
            Phi_in = initial_conditions

        else:  # t_in > seq_len
            Phi_in = initial_conditions[-self.seq_len:, ...] 


        self.model.eval()
        m = Phi_in.shape[0]
        x_n = self.model.autoencoder.encode(Phi_in)    #x : [t_in obsdim]
        x_in = x_n[:-1, ...].unsqueeze(0)
        for n in range(timesteps - 1):

            print(x_in.shape)
            x_nn   = self.model.transformer(x_in) #[1 obsdim]
            x_n  = torch.cat((x_n,x_nn), 0)
            x_in = x_n[:-1, :].unsqueeze(0)
            x_in = x_in[:, -self.seq_len:, ...]
        
        Phi  = self.model.autoencoder.recover(x_n)

        if t_in < self.seq_len : 
            return x_n[self.seq_len - t_in:, ...].unsqueeze(0), Phi[self.seq_len - t_in:, ...].unsqueeze(0)
        else : 
            return x_n.unsqueeze(0),Phi.unsqueeze(0) 

###########################################################################################################
    def plot_learning_curves(self):

        df = pd.read_csv(self.exp_dir+'/'+self.exp_name+"/out_log/log")

        min_trainloss = df.loc[df['Train_Loss'].idxmin(), 'epoch']
        print("Epoch with Minimum train_error: ", min_trainloss)

        min_testloss = df.loc[df['Test_Loss'].idxmin(), 'epoch']
        print("Epoch with Minimum test_error: ", min_testloss)

        #Total Loss
        plt.figure()
        plt.plot(df['epoch'],df['Train_Loss'], label="Train Loss")
        plt.plot(df['epoch'], df['Test_Loss'], label="Test Loss")
        plt.legend()
        plt.xlabel("Epochs")
        plt.savefig(self.exp_dir+'/'+self.exp_name+"/out_log/TotalLoss.png", dpi = 256, facecolor = 'w', bbox_inches='tight')

        #KoopEvo Loss
        plt.figure()
        plt.plot(df['epoch'],df['Train_TransEvo_Loss'], label="Train TransEvo Loss")
        plt.plot(df['epoch'], df['Test_TransEvo_Loss'], label="Test TransEvo Loss")
        plt.legend()
        plt.xlabel("Epochs")
        plt.savefig(self.exp_dir+'/'+self.exp_name+"/out_log/AutoencoderLoss.png", dpi = 256, facecolor = 'w', bbox_inches='tight')

        #Autoencoder Loss
        plt.figure()
        plt.semilogy(df['epoch'],df['Train_Autoencoder_Loss'], label="Train Autoencoder Loss")
        plt.semilogy(df['epoch'], df['Test_Autoencoder_Loss'], label="Test Autoencoder Loss")
        plt.legend()
        plt.xlabel("Epochs")
        plt.savefig(self.exp_dir+'/'+self.exp_name+"/out_log/AutoencoderLoss.png", dpi = 256, facecolor = 'w', bbox_inches='tight')

        #State Loss
        plt.figure()
        plt.semilogy(df['epoch'],df['Train_StateEvo_Loss'], label="Train State Evolution Loss")
        plt.semilogy(df['epoch'], df['Test_StateEvo_Loss'], label="Test State Evolution Loss")
        plt.legend()
        plt.xlabel("Epochs")
        plt.savefig(self.exp_dir+'/'+self.exp_name+"/out_log/StateLoss.png", dpi = 256, facecolor = 'w', bbox_inches='tight')

    ###########################################################################
