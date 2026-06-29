import numpy as np
import csv, h5py, json, pickle
import torch
from torch.utils.data import DataLoader
from src.PreProc_Data.DataProc import Dataset


class DynSystem_Data:

    def load_and_preproc_data(self):
        '''
        loads and preprocesses data
        Requires
        --------
        data_dir, norm_input
        Generates
        ---------
        lp_data (numpy tensor): [num_traj, timesteps, statedim] Loaded Data
        data_args (dict)      :  Attributes of the loaded data
        '''
        
        raw_data = np.load(self.data_dir)
        fields = torch.tensor(raw_data, dtype=torch.float32)
       
        self.lp_data = fields[:, self.ntransients:self.nenddata, ...]

        try : 
            self.N, self.T, self.F, self.nx, self.ny = self.lp_data.shape
        except : 
            self.N, self.T, self.nx = self.lp_data.shape

        self.statedim = self.lp_data.shape[2:]

        print("State Dims: ", self.statedim)
    
    def create_dataset(self, mode = "Both"):

        '''
        Creates non sequence dataset for state variables and divides into test, train and val dataset
        Requires
        --------
        lp_data: [num_traj, timesteps, statedim] state variables
        mode   : "Train" for only train dataset, "Test" for only test dataset, "Both" for both datset

        Returns
        -------
        Dataset : [num_traj, timesteps, statedim] Input , Output (both test and train)
        Dataloader: [num_traj*timesteps, statedim] 
        '''

        if mode == "Both" or mode == "Train":

            print('Total trajectories :', self.N)
            print('Total snapshots :', self.T)
            print('-------- Total samples :', self.T * self.N)

            train_size = int(self.train_size*self.T)

            self.train_data = self.lp_data[:,:train_size, ...]
            print("Train_Shape: ", self.train_data.shape)

            self.train_dataset    = Dataset(self.train_data, self.seq_len, self.pred_horizon)
            self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle = True, num_workers = 0)
    
        if mode == "Both" or mode == "Test":
            
            self.test_data = self.lp_data[:,train_size:, ...]
            
            print("Test_Shape: " , self.test_data.shape)
            self.test_dataset     = Dataset(self.test_data, self.seq_len, self.pred_horizon)
            self.test_dataloader  = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle = False, num_workers = 0)
