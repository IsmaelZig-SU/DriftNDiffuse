import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class Dataset(Dataset):
    def __init__(self, fields, seq_len=5, pred_horizon=1, padding_mode='replicate'):
        '''
        Input
        -----
        'fields': numpy array or tensor of shape [num_traj, timesteps, statedim...]

        padding_mode : str
            'replicate' for repetition padding (repeats initial condition), 
            'constant' for zero padding.
        '''

        self.fields = fields
        
        self.seq_len = seq_len
        self.pred_horizon = pred_horizon
        self.padding_mode = padding_mode
        
        self.num_traj = self.fields.shape[0]
        self.total_timesteps = self.fields.shape[1]
        
        self.valid_timesteps_per_traj = self.total_timesteps - self.pred_horizon
        self.total_samples = self.num_traj * self.valid_timesteps_per_traj

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Map flat 1D index to a unique (trajectory_idx, time_idx) pair
        traj_idx = idx // self.valid_timesteps_per_traj
        time_idx = idx % self.valid_timesteps_per_traj
        
        # 1. Target data: Always ahead of the current time_idx, no padding needed
        target_start = time_idx + 1
        target_end = target_start + self.pred_horizon
        target_data = self.fields[traj_idx, target_start:target_end]
        
        # 2. Context data: Might overshoot t=0 into negative indices
        context_start = time_idx - self.seq_len + 1
        context_end = time_idx + 1
        
        if context_start < 0:
            # Grab what's available from t=0 up to context_end
            available_context_data = self.fields[traj_idx, 0:context_end]
            padding_needed = self.seq_len - available_context_data.shape[0]
            
            if self.padding_mode == 'constant':
                
                pad_config_data = [0, 0] * (available_context_data.ndim - 1) + [padding_needed, 0]
                context_data = F.pad(available_context_data, pad_config_data, mode='constant', value=0.0)
                
            elif self.padding_mode == 'replicate':
    
                # 1. Extract the initial condition at t=0: shape [statedim...]
                init_condition_data = self.fields[traj_idx, 0]
                
                # 2. Add a pseudo-time dimension to it: shape [1, statedim...]
                init_condition_data = init_condition_data.unsqueeze(0)
                
                # 3. Replicate it to match the padding amount needed
                # Create a repeat tuple: pad 'padding_needed' times on dim 0, 1 time on everything else
                repeat_sizes_data = [padding_needed] + [1] * (init_condition_data.ndim - 1)

                padding_data = init_condition_data.repeat(*repeat_sizes_data)

                # 4. Concatenate the history prefix with what's available
                context_data = torch.cat((padding_data, available_context_data), dim=0)

        else:
            # Normal slice without padding
            context_data = self.fields[traj_idx, context_start:context_end]
        
        return context_data, target_data