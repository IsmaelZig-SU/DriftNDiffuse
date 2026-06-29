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
from src.MZA_Experiment import MZA_Experiment
from torch.utils.data import DataLoader

torch.manual_seed(99)

class Eval_MZA(MZA_Experiment):

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
    def predict_onestep(self, dataset, num_trajs, batch_size = 32):

        '''
        Input
        ----- 
        dataset, num_trajs

        Returns
        -------
        x_nn_hat (torch tensor)  : [num_trajs timesteps obsdim]
        Phi_nn_hat (torch tensor): [num_trajs timesteps statedim]
        Phi_nn (torch tensor): [num_trajs timesteps statedim]
        StateEvoLoss (torch tensor): [timesteps]
        '''

        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False)

        self.model.eval()

        for count, (Phi_seq, Phi_nn) in enumerate(dataloader):
            
            # Phi_n   = torch.squeeze(Phi_seq[:,-1,...])  #[bs statedim]
            Phi_seq = Phi_seq.to(self.device)
            Phi_nn = Phi_nn.to(self.device)

            #flattening batchsize seqlen
            Phi_seq = torch.flatten(Phi_seq, start_dim = 0, end_dim = 1)   #[bs*seqlen, statedim]
            Phi_nn = torch.squeeze(Phi_nn, dim = 1)

            #obtain observables
            print(count, " ", Phi_nn.shape) 
            x_seq, Phi_seq_hat = self.model.autoencoder(Phi_seq)
            x_nn , _   = self.model.autoencoder(Phi_nn)
            del _

            #reshaping tensors in desired form
            adaptive_bs = int(x_seq.shape[0]/self.seq_len)   #adaptive batchsize due to change in size for the last batch
            x_seq = x_seq.reshape(adaptive_bs, self.seq_len, self.num_obs) #[bs seqlen obsdim]
            x_n   = x_seq[:,-1,:]  #[bs obsdim]
            
            sd = (self.statedim,) if str(type(self.statedim)) == "<class 'int'>" else self.statedim
            Phi_seq_hat = Phi_seq_hat.reshape(adaptive_bs, self.seq_len, *sd) #[bs seqlen statedim]
            # Phi_n_hat   = torch.squeeze(Phi_seq_hat[:, -1, :]) 
            
            #Evolving in Time
            x_nn_hat     = self.model.transformer(x_seq)
            Phi_nn_hat   = self.model.autoencoder.recover(x_nn_hat)
            
            if count == 0:

                x_nn_hat_all     = x_nn_hat.detach().to("cpu")
                Phi_nn_hat_all   = Phi_nn_hat.detach().to("cpu")
                Phi_nn_all       = Phi_nn.detach().to("cpu")
                x_nn_all         = x_nn.detach().to("cpu")

            else:

                x_nn_hat_all     = torch.cat((x_nn_hat_all, x_nn_hat.detach().to("cpu")), 0)
                Phi_nn_hat_all   = torch.cat((Phi_nn_hat_all, Phi_nn_hat.detach().to("cpu")), 0)
                Phi_nn_all       = torch.cat((Phi_nn_all, Phi_nn.detach().to("cpu")), 0)
                x_nn_all         = torch.cat((x_nn_all, x_nn.detach().to("cpu")), 0)
            
        x_nn_hat     = x_nn_hat_all
        Phi_nn_hat   = Phi_nn_hat_all
        Phi_nn       = Phi_nn_all
        x_nn         = x_nn_all      

        
        re_x_nn_hat  = x_nn_hat.reshape(int(x_nn_hat.shape[0]/num_trajs), num_trajs, *x_nn_hat.shape[1:])
        x_nn_hat     = torch.movedim(re_x_nn_hat, 1, 0) #[num_trajs timesteps obsdim]

        re_Phi_nn_hat  = Phi_nn_hat.reshape(int(Phi_nn_hat.shape[0]/num_trajs), num_trajs, *Phi_nn_hat.shape[1:])
        Phi_nn_hat     = torch.movedim(re_Phi_nn_hat, 1, 0) #[num_trajs timesteps statedim]

        re_Phi_nn    = Phi_nn.reshape(int(Phi_nn.shape[0]/num_trajs), num_trajs, *Phi_nn.shape[1:])
        Phi_nn       = torch.movedim(re_Phi_nn, 1, 0) #[num_trajs timesteps statedim]

        re_x_nn    = x_nn.reshape(int(x_nn.shape[0]/num_trajs), num_trajs, *x_nn.shape[1:])
        x_nn       = torch.movedim(re_x_nn, 1, 0) #[num_trajs timesteps statedim]

        StateEvo_Loss = Eval_MZA.state_mse(Phi_nn, Phi_nn_hat)

        return x_nn_hat, Phi_nn_hat, x_nn, Phi_nn, StateEvo_Loss

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

            Phi_n  = initial_conditions  
            _, _, x_n, log_var = self.model.autoencoder(Phi_n)    #[num_trajs obsdim]
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

                x_nn     = self.model.transformer(x_seq_n)
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

    # def variational_UQ(self, phi_test, context, ens_var) : 

    #     Phi_n_ens = []

    #     for i in range(ens_var) : 

    #         x_n, Phi_n, mu, log_var = self.model.autoencoder(phi_test, context)
    #         Phi_n_ens.append(Phi_n)

    #     Phi_n_ens = torch.stack(Phi_n_ens, dim = 0)
    #     var = torch.var(Phi_n_ens, dim = 0)

    #     return Phi_n_ens, var

    def variational_UQ_scale(self, phi_test, context, ens_var) : 

        Phi_n_ens = []

        for i in range(ens_var) : 

            x_n, Phi_n, mu, log_var = self.model.autoencoder(phi_test, context)
            Phi_n_ens.append(Phi_n)

        Phi_n_ens = torch.stack(Phi_n_ens, dim = 0)

        return Phi_n_ens


    # def predict_multistep_variational(self, initial_conditions, timesteps, context, ens_size):

    #         '''
    #         Input
    #         -----
    #         initial_conditions (torch tensor): [num_trajs, statedim]
    #         timesteps (int): Number timesteps for prediction

    #         Returns
    #         x (torch tensor): [num_trajs timesteps obsdim] observable vetcor
    #         Phi (torch tensor): [num_trajs timesteps statedim] state vector
    #         '''

    #         self.model.eval()

    #         Phi_n  = initial_conditions  
    #         context_transfo = context.repeat(1, 1, 4)[:, -1, :].unsqueeze(1)
    #         context_vae = context_transfo[:, :, 0]

    #         x_n_ens, mu, log_var = self.model.encode_variational(Phi_n, context, ens_size)

    #         for i in range(ens_size) :

    #             x_n = x_n_ens[i, ...]
    #             Phi_ens = []
    #             x_ens = []

    #             x   = x_n[None,...].to("cpu")                  
    #             Phi = Phi_n[None, ...].to("cpu")                

    #             for n in range(timesteps):

    #                 non_time_dims = (1,)*(x.ndim-1)   #dims apart from timestep in tuple form (1,1,...)
    #                 if n >= self.seq_len:
    #                     i_start = n - self.seq_len + 1
    #                     x_seq_n = x[i_start:(n+1), ...].to(self.device)
    #                 elif n==0:
    #                     # padding = torch.zeros(x[0].repeat(self.seq_len - 1, *non_time_dims).shape).to(self.device)
    #                     padding = x[0].repeat(self.seq_len - 1, *non_time_dims).to(self.device)
    #                     x_seq_n = x[0:(n+1), ...].to(self.device)
    #                     x_seq_n = torch.cat((padding, x_seq_n), 0)
    #                 else:
    #                     # padding = torch.zeros(x[0].repeat(self.seq_len - n, *non_time_dims).shape).to(self.device)
    #                     padding = x[0].repeat(self.seq_len - n, *non_time_dims).to(self.device)
    #                     x_seq_n = x[1:(n+1), ...].to(self.device)
    #                     x_seq_n = torch.cat((padding, x_seq_n), 0)
                    
    #                 x_seq_n = torch.movedim(x_seq_n, 1, 0) #[num_trajs seq_len obsdim]
    #                 x_seq_n = x_seq_n[:,:-1,:]

    #                 x_nn     = self.model.transformer(x_seq_n, context_transfo)
    #                 Phi_nn = self.model.autoencoder.recover(x_nn, context_vae)

    #                 x   = torch.cat((x,x_nn[None,...].detach().cpu()), 0)
    #                 Phi = torch.cat((Phi,Phi_nn[None,...].detach().cpu()), 0)

    #             x      = torch.movedim(x, 1, 0) 
    #             Phi    = torch.movedim(Phi, 1, 0) 

    #             x_ens.append(x)
    #             Phi_ens.append(Phi)

    #         Phi_ens = torch.stack(Phi_ens, dim = 0)
    #         x_ens = torch.stack(x_ens, dim = 0)

    #         return x_ens, Phi_ens


########################################################################################
    @staticmethod
    def prediction_limit(Phi_ms_hat, Phi, x):
        '''
        Computes Prediction limit at threshold of 0.5
        Input
        -----
        Phi_ms_hat (ndarray): [timesteps statedim]
        Phi        (ndarray): [timesteps statedim]
        x          (ndarray): [timesteps]

        Returns
        -------
        Prediction limit (float) 
        '''
        pl  = []
        for i in range(Phi_ms_hat.shape[0]):
            pli = np.sqrt(np.mean((Phi_ms_hat[i] - Phi[i])**2)/np.mean(Phi[:i+1]**2))
            pl.append(pli)
            if (pli > 0.5):
                return x[i]
    
    def sampled_prediction_limit(self, true_data, timesteps, lt, num_samples):
        '''
        Computes Prediction limit at threshold of 0.5 over multiple initial conditions
        Input
        -----
        true_data  (ndarray): [num_trajs timesteps statedim]
        timesteps  (int)  :  Number of timesteps for prediction
        lt         (float):  lyapunov timeunits 
        num_samples(int)  :  Number of samples 

        Returns
        -------
        Coff_x (ndarray): [Number of samples] 
        '''

        Phi = torch.from_numpy(true_data).to(torch.float)
        coff_x = np.array([])
        x = np.arange(timesteps)/lt

        #looping over different number of initial conditions 
        for i in tqdm(range(num_samples), desc="Processing", unit="iteration"):
            
            initial_step = random.randint(100,6000)
            initial_conditions = Phi[:,initial_step,:].to(self.device)

            _,Phi_ms_hat = self.predict_multistep(initial_conditions, timesteps)
            Phi_ms_hat_ = Phi_ms_hat[0,0:timesteps].detach().cpu().numpy()
            Phi_        = Phi[0,initial_step:timesteps+initial_step].numpy()

            coxi = self.prediction_limit(Phi_ms_hat_, Phi_, x)
            
            # if coxi > 0.6:
            coff_x = np.append(coff_x, coxi)

        coff_x = np.array(coff_x)

        return coff_x

###########################################################################################################
    def plot_learning_curves(self):

        df = pd.read_csv(self.exp_dir+'/'+self.exp_name+"/out_log/log")

        min_trainloss = df.loc[df['Train_Loss'].idxmin(), 'epoch']
        print("Epoch with Minimum train_error: ", min_trainloss)

        min_testloss = df.loc[df['Test_Loss'].idxmin(), 'epoch']
        print("Epoch with Minimum test_error: ", min_testloss)

        #Total Loss
        plt.figure()
        plt.semilogy(df['epoch'],df['Train_Loss'], label="Train Loss")
        plt.semilogy(df['epoch'], df['Test_Loss'], label="Test Loss")
        plt.legend()
        plt.xlabel("Epochs")
        plt.savefig(self.exp_dir+'/'+self.exp_name+"/out_log/TotalLoss.png", dpi = 256, facecolor = 'w', bbox_inches='tight')

        #KoopEvo Loss
        plt.figure()
        plt.semilogy(df['epoch'],df['Train_TransEvo_Loss'], label="Train TransEvo Loss")
        plt.semilogy(df['epoch'], df['Test_TransEvo_Loss'], label="Test TransEvo Loss")
        plt.legend()
        plt.xlabel("Epochs")
        # plt.savefig(self.exp_dir+'/'+self.exp_name+"/out_log/AutoencoderLoss.png", dpi = 256, facecolor = 'w', bbox_inches='tight')

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
