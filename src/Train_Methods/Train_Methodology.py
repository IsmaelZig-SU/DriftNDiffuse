import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from time import time
import math
import numpy as np 
# import random

class Train_Methodology():

    def time_evolution(self, initial_x_n, initial_x_seq, ph_size):

        """
        Calculates multistep prediction from koopman and seqmodel while training
        Inputs
        ------
        initial_x_n (torch tensor): [bs obsdim]
        initial_x_seq (torch tensor): [bs seq_len obsdim]
        ph_size (int) : variable pred_horizon acccording to future data available

        Returns
        -------
        x_nn_hat_ph (torch_tensor): [bs pred_horizon obsdim]
        Phi_nn_hat (torch_tensor): [bs pred_horizon statedim]
        """

        bs = initial_x_n.shape[0]
        x_nn_hat_ph = torch.empty((bs, ph_size, self.num_obs), device=self.device)
        mu_nn_hat_ph = torch.empty((bs, ph_size, self.num_obs), device=self.device)
        log_var_nn_hat_ph = torch.empty((bs, ph_size, self.num_obs), device=self.device)

        x_seq = initial_x_seq.clone()

        for t in range(ph_size):
            mu_nn_hat, log_var_nn_hat, x_nn_hat = self.model.transformer(x_seq)
            x_last = x_seq[:, -1, :]
            x_p_1 = x_nn_hat + x_last
            x_p_1_mu = mu_nn_hat + x_last

            x_nn_hat_ph[:, t] = x_p_1
            mu_nn_hat_ph[:, t] = x_p_1_mu
            log_var_nn_hat_ph[:, t] = log_var_nn_hat

            x_seq = torch.cat((x_seq[:, 1:, :], x_p_1[:, None, :]), dim=1)

        Phi_nn_hat_ph = self.model.autoencoder.recover(mu_nn_hat_ph.reshape((-1, self.num_obs)))
        Phi_nn_hat_ph = Phi_nn_hat_ph.reshape((bs, ph_size, *self.statedim))

        return x_nn_hat_ph, mu_nn_hat_ph, log_var_nn_hat_ph, Phi_nn_hat_ph #sequence of states and diffusion term [bs, pred_horizon, dim] 

################################################################################################################################################

    def nll(self, x_next, mu, log_var):
        """
        x_next  : [bs, T, d]   true next state
        mu      : [bs, T, d]   drift prediction (mean)
        log_var : [bs, T, d]   diffusion prediction (log variance)
        """
        var = torch.exp(log_var)

        nll = 0.5 * (
            math.log(2 * math.pi)
            + log_var
            + (x_next - mu)**2 / var
        )          # [bs, T, D]
        loss = nll.sum(dim=-1).mean() / self.num_obs

        return loss 



    def train_test_loss(self, mode = "Train", dataloader = None, epoch = 0):
        '''
        One Step Prediction method
        Requires: dataloader, model, optimizer
        '''
        # self.args = args

        if mode == "Train":
            dataloader = self.train_dataloader 
            self.model.train() 
            # When training, we want gradients enabled
            context = torch.enable_grad()
        elif mode == "Test":
            dataloader = self.test_dataloader if dataloader is None else dataloader
            self.model.eval()
            # When testing, we disable gradient tracking completely
            context = torch.inference_mode() 
        else:
            print("mode can be Train or Test")
            return None

        num_batches = len(dataloader)
        total_loss, total_Autoencoder_Loss, total_TransEvo_Loss, total_StateEvo_Loss = 0,0,0,0
        mseLoss = nn.MSELoss() 
        with context:

            for Phi_seq, Phi_nn_ph in dataloader:

                Phi_seq = Phi_seq.to(self.device)          #[bs, seq_len, **statedim]
                Phi_nn_ph = Phi_nn_ph.to(self.device)      #[bs, pred_horizon, **statedim]

                ph_size = self.pred_horizon
                
                ####### flattening batchsize seqlen / batchsize pred_horizon ######
                Phi_seq   = torch.flatten(Phi_seq, start_dim = 0, end_dim = 1) #[bs*seqlen, statedim]
                Phi_nn_ph = torch.flatten(Phi_nn_ph, start_dim = 0, end_dim = 1) #[bs*ph_size, statedim]
                ###### obtain observables ######

                if self.autoencoder_model == 'CNN' or self.autoencoder_model == 'AE': 

                    x_seq, Phi_seq_hat = self.model.autoencoder(Phi_seq)
                    x_nn_ph , Phi_nn_hat_ph_nolatentevol = self.model.autoencoder(Phi_nn_ph)

                else : 

                    x_seq, Phi_seq_hat, mu, log_var = self.model.autoencoder(Phi_seq)
                    x_nn_ph , Phi_nn_hat_ph_nolatentevol, mu_nn_ph, log_var_nn_ph = self.model.autoencoder(Phi_nn_ph)

                    mu = mu.reshape(-1, self.seq_len, self.num_obs)
                    log_var = log_var.reshape(-1, self.seq_len, self.num_obs)
                    mu_nn_ph = mu_nn_ph.reshape(-1, ph_size, self.num_obs)
                    log_var_nn_ph = log_var_nn_ph.reshape(-1, ph_size, self.num_obs)

                ###### reshaping tensors in desired form ######
                sd = self.statedim
                    
                x_nn_ph  = x_nn_ph.reshape(-1, ph_size, self.num_obs) #[bs ph_size obsdim]
                x_seq = x_seq.reshape(-1, self.seq_len, self.num_obs) #[bs seqlen obsdim]

                x_n   = torch.squeeze(x_seq[:,-1,:])

                Phi_seq = Phi_seq.reshape(-1, self.seq_len, *sd) #[bs seq_len statedim]
                Phi_seq_hat = Phi_seq_hat.reshape(-1, self.seq_len, *sd) #[bs seq_len statedim]
                Phi_nn_ph = Phi_nn_ph.reshape(-1, ph_size, *sd) #[bs pred_horizon statedim]
                Phi_nn_hat_ph_nolatentevol = Phi_nn_hat_ph_nolatentevol.reshape(-1, ph_size, *sd)
                
                x_nn_hat_ph, mu_nn_hat_ph, log_var_nn_hat_ph, Phi_nn_hat_ph = self.time_evolution(x_n, x_seq, ph_size)

                if len(self.statedim) > 1 : 

                    U_nn_hat = Phi_nn_hat_ph[:, :, 0, :, :]
                    V_nn_hat = Phi_nn_hat_ph[:, :, 1, :, :]

                    U_nn = Phi_nn_ph[:, :, 0, :, :]
                    V_nn = Phi_nn_ph[:, :, 1, :, :]

                    energy = 0.5*(U_nn**2+V_nn**2)
                    energy_hat = 0.5*(U_nn_hat**2+V_nn_hat**2)
                    energy_t = torch.mean(energy, dim = (-2,-1))
                    energy_t_hat = torch.mean(energy_hat, dim = (-2,-1))
                    energy_loss = mseLoss(energy_t, energy_t_hat)

                    U_loss = mseLoss(U_nn, U_nn_hat)
                    V_loss = mseLoss(V_nn, V_nn_hat)

                    StateEvo_Loss  = U_loss + V_loss + energy_loss

                else : 

                    energy_t = torch.mean(Phi_nn_ph**2, dim = -1)
                    energy_t_hat = torch.mean(Phi_nn_hat_ph**2, dim = -1)
                    energy_loss = mseLoss(energy_t, energy_t_hat)

                    StateEvo_Loss = mseLoss(Phi_nn_hat_ph, Phi_nn_ph) + energy_loss
                
                reconstruction_Loss = 0.5 * (mseLoss(Phi_nn_hat_ph_nolatentevol, Phi_nn_ph) + mseLoss(Phi_seq_hat, Phi_seq))

                if self.autoencoder_model == 'CNN' or self.autoencoder_model == 'AE' : 

                    Autoencoder_Loss = reconstruction_Loss

                else : 

                    KLD_next = -0.5 * torch.mean(1 + log_var_nn_ph - mu_nn_ph.pow(2) - log_var_nn_ph.exp())
                    KLD_past = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

                    KLD = 0.5*(KLD_next + KLD_past)

                    Autoencoder_Loss = reconstruction_Loss + self.beta * KLD 

                latent_loss = self.nll(x_nn_ph, mu_nn_hat_ph, log_var_nn_hat_ph)

                loss = latent_loss + 1e2*Autoencoder_Loss + self.lambda_stateloss*StateEvo_Loss

                if mode == "Train":
                    # self.optimizer.zero_grad()
                    for p in self.model.parameters():
                        p.grad = None

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                total_loss += loss.item()
                total_TransEvo_Loss +=  latent_loss.item()
                total_Autoencoder_Loss += Autoencoder_Loss.item()
                total_StateEvo_Loss    += StateEvo_Loss.item()

        if mode == "Train" : 
            self.scheduler.step()

        avg_loss             = total_loss / num_batches
        avg_TransEvo_Loss     = total_TransEvo_Loss / num_batches
        avg_Autoencoder_Loss = total_Autoencoder_Loss / num_batches
        avg_StateEvo_Loss    = total_StateEvo_Loss / num_batches

        current_lr = self.optimizer.param_groups[0]['lr']

        Ldict = {'avg_loss': avg_loss, 'avg_TransEvo_Loss': avg_TransEvo_Loss,'avg_Autoencoder_Loss': avg_Autoencoder_Loss, 'avg_StateEvo_Loss': avg_StateEvo_Loss,  'learning_rate' : current_lr} 

        return Ldict
    
################################################################################################################################################
    
    def training_loop(self):
        '''
        Requires:
        model, optimizer, train_dataloader, val_dataloader, device
        '''
        print("Device: ", self.device)
        print("Untrained Test\n--------")

        test_Ldict = self.train_test_loss(mode = "Test", dataloader = self.test_dataloader)

        print(f"Test Loss: {test_Ldict['avg_loss']:<{6}}, Transfo Loss : {test_Ldict['avg_TransEvo_Loss']:<{6}}, Auto : {test_Ldict['avg_Autoencoder_Loss']:<{6}}, StateEvo : {test_Ldict['avg_StateEvo_Loss']:<{6}}")

        # min train loss
        self.min_train_loss = 1000 
        self.min_test_loss  = 1000
        
        print(f"################## Starting Training ###############")
         
        for ix_epoch in range(self.load_epoch, self.load_epoch + self.nepochs):

            #start time
            start_time = time()
            
            #CALCULATING LOSS
            
            train_Ldict = self.train_test_loss(mode = "Train", dataloader = None, epoch = ix_epoch)
            test_Ldict  = self.train_test_loss(mode = "Test", dataloader = self.test_dataloader, epoch = ix_epoch)
            
            #PRINTING AND SAVING DATA
            print(f"Epoch {ix_epoch} ")
            print(f"Train Loss: {train_Ldict['avg_loss']:<{6}}, Latent loss : {train_Ldict['avg_TransEvo_Loss']:<{6}}, Compression loss : {train_Ldict['avg_Autoencoder_Loss']:<{6}}, Determinsitic loss : {train_Ldict['avg_StateEvo_Loss']:<{6}},  Learning rate: {train_Ldict['learning_rate']}")
            print(f"Test Loss: {test_Ldict['avg_loss']:<{6}}, Latent loss : {test_Ldict['avg_TransEvo_Loss']:<{6}}, Compression loss : {test_Ldict['avg_Autoencoder_Loss']:<{6}}, Determinsitic loss : {test_Ldict['avg_StateEvo_Loss']:<{6}}")

            indentation = 0
            writeable_loss = {"epoch":str(ix_epoch).rjust(indentation),"Train_Loss":str(train_Ldict['avg_loss']).rjust(indentation), "Train_TransEvo_Loss":str(train_Ldict['avg_TransEvo_Loss']).rjust(indentation),\
                              "Train_Autoencoder_Loss":str(train_Ldict["avg_Autoencoder_Loss"]).rjust(indentation),\
                              "Train_StateEvo_Loss":str(train_Ldict["avg_StateEvo_Loss"]).rjust(indentation),\
                              "Test_Loss":str(test_Ldict['avg_loss']).rjust(indentation), "Test_TransEvo_Loss":str(test_Ldict['avg_TransEvo_Loss']).rjust(indentation),\
                              "Test_Autoencoder_Loss":str(test_Ldict["avg_Autoencoder_Loss"]).rjust(indentation), "Test_StateEvo_Loss":str(test_Ldict["avg_StateEvo_Loss"]).rjust(indentation)}
            
            self.log.writerow(writeable_loss)
            self.logf.flush()
            
            #saving Min Loss weights and optimizer state
            if self.min_test_loss > test_Ldict["avg_loss"]:
                self.min_test_loss = test_Ldict["avg_loss"]
                torch.save({
                    'epoch':ix_epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict':self.optimizer.state_dict()
                    }, self.exp_dir+'/'+ self.exp_name+"/model_weights/min_test_loss")
            
            if self.min_train_loss > train_Ldict["avg_loss"]:
                self.min_train_loss = train_Ldict["avg_loss"]
                torch.save({
                    'epoch':ix_epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict':self.optimizer.state_dict()
                    }, self.exp_dir+'/'+ self.exp_name+"/model_weights/min_train_loss")

            if (ix_epoch%self.nsave == 0):
                #saving weights and plotting loss

                try:
                    self.plot_learning_curves()
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")

                torch.save({
                    'epoch':ix_epoch,
                    'model_state_dict': self.model.state_dict(),
                    }, self.exp_dir+'/'+ self.exp_name+"/model_weights/at_epoch{epoch}".format(epoch=ix_epoch))
                

            #ending time
            end_time = time()
            print("Epoch Time Taken: ", end_time - start_time)
        

        #saving final weights
        torch.save({
                    'epoch':ix_epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict':self.optimizer.state_dict()
                    }, self.exp_dir+'/'+ self.exp_name+"/model_weights/at_epoch{epoch}".format(epoch=ix_epoch))
        
        self.logf.close()
