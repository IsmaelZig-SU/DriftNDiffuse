import torch, pickle, os
from torch.utils.data import Dataset, DataLoader
from src.PreProc_Data.DataProc import StackedSequenceDataset
import numpy as np
from src.Eval_MZA import Eval_MZA
import matplotlib.pyplot as plt
import pandas as pd
import csv
import scienceplots
from tqdm import tqdm
import gc

# --- Paper Styling ---
plt.style.use(['science', 'grid', 'no-latex'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache()

exp_dir = "D:/model_SDE/Trained_models/Trained_Models"

exp_names = [
    "sl20_ph40_obs64_attblks4_beta_0.001_2_-6",
    "sl20_ph40_obs64_attblks4_beta_0.001_2_-7",
    "sl20_ph40_obs64_attblks4_beta_0.001_2_-8",  
    "sl20_ph40_obs64_attblks4_beta_0.001_2_-9", 
    "sl20_ph40_obs64_attblks4_beta_0.001_2_-10",
    "sl20_ph40_obs64_attblks4_beta_0.001_2_-11",
    "sl20_ph40_obs64_attblks4_beta_0.001_2_-12"
]

kolmo_flow = "D:/data/Kolmogorov_SDE/kolmo_filtered_120s_10ens_re90.npy"
Phi_test = np.load(kolmo_flow) 
Phi_test = torch.tensor(Phi_test, dtype=torch.float32).to(torch.device("cuda"))
N, T, D, Nx, Ny = Phi_test.shape

def evaluate_ensemble_extended(ensemble, ground_truth):
    """
    Evaluates ensemble predictions.
    Returns: MSE of the mean flow, PICP at 1 and 3 sigmas, and MPIW.
    """
    N_ens, T_len, C, H, W = ensemble.shape
    
    # --- 1. MSE of the Mean Flow ---
    mean_flow = np.mean(ensemble, axis=0) # Shape: [T, C, H, W]
    
    rrmse_mean_flow = np.sqrt(np.sum((mean_flow - ground_truth)**2))/np.sqrt(np.sum(ground_truth**2))
    mse_mean_flow = np.mean((mean_flow - ground_truth)**2)
    
    # --- 2. PICP and MPIW Logic ---
    def get_coverage(low_p, high_p):
        lower = np.percentile(ensemble, low_p, axis=0)
        upper = np.percentile(ensemble, high_p, axis=0)
        
        coverage_mask = (ground_truth >= lower) & (ground_truth <= upper)
        picp = np.mean(coverage_mask)
        mpiw = np.mean(upper - lower)
        return picp, mpiw

    picp_1s, mpiw_1s = get_coverage(15.85, 84.15)
    picp_3s, mpiw_3s = get_coverage(0.135, 99.865)

    return {
        "rrmse_mean_flow": float(rrmse_mean_flow),
        "mse_mean_flow": float(mse_mean_flow),
        "picp_1sigma": float(picp_1s),
        "picp_3sigma": float(picp_3s),
        "mpiw_1sigma": float(mpiw_1s),
        "mpiw_3sigma": float(mpiw_3s)
    }

def moving_average_ensemble(data, window_size=20):
    kernel = np.ones(window_size) / window_size
    pad_width = window_size // 2
    padded_data = np.pad(data, ((0,0), (pad_width, pad_width -1), (0,0), (0,0), (0,0)), mode='edge')
    
    def smooth_1d(line):
        return np.convolve(line, kernel, mode='valid')

    return np.apply_along_axis(smooth_1d, axis=1, arr=padded_data)

seq_len = 20
ens = 60
T_test_idx = Phi_test.shape[1] 
dt = 0.12
timesteps = 1200 
index_x, index_y = 32, 32
phi = 0

# --- Loop Over Experiments ---
for exp_name in exp_names: 
    print(f"\n=========================================\nStarting Evaluation for: {exp_name}\n=========================================")
    
    # 1. Setup save directory dynamically under the root exp_dir
    save_dir = os.path.join(exp_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    
    model = Eval_MZA(exp_dir, exp_name)
    model.load_weights(min_train_loss=True)
    model.model.eval()

    # Dictionary to aggregate results across trajectories
    all_results = {}

    fig, axes = plt.subplots(
        N // 2, 2, 
        figsize=(8, 12), 
        sharex=True, 
        sharey=True,
        dpi=150
    )
    axes = axes.flatten()

    for traj_idx in range(N):
        ax = axes[traj_idx]
        ens_traj = []
        print(f'Processing trajectory {traj_idx+1}/{N}...')

        for j in tqdm(range(ens)):
            initial_condition = Phi_test[traj_idx, :1].to(device)
            padding = torch.repeat_interleave(initial_condition, seq_len - 1, dim=0)
            Phi_tp = torch.cat((padding, initial_condition), dim=0)
            x_lat, mu_lat, log_var = model.model.autoencoder.encode(Phi_tp)
            x = x_lat.unsqueeze(0)
            traj = []

            for t in range(timesteps):
                res_mu_tp1, diffusion, res_x_tp1 = model.model.transformer(x)
                x_tp1 = res_x_tp1 + x[:, -1, :]
                Phi_tp1 = model.model.autoencoder.recover(x_tp1)
                traj.append(Phi_tp1.detach().cpu())
                x = torch.cat([x, x_tp1.unsqueeze(1)], dim=1)[:, 1:, :]

            ens_traj.append(torch.stack(traj))
            torch.cuda.empty_cache()

        # --- Stack ensemble ---
        ens_traj = torch.stack(ens_traj)  # [ens, T, 1, Phi, Nx, Ny]
        ens_traj_np = ens_traj[:, :, 0].detach().cpu().numpy()
        
        # Calculate Metrics
        dict_eval_train = evaluate_ensemble_extended(ens_traj_np[:, :800], Phi_test[traj_idx, :800].detach().cpu().numpy())
        dict_eval_test = evaluate_ensemble_extended(ens_traj_np[:, 800:T_test_idx], Phi_test[traj_idx, 800:T_test_idx].detach().cpu().numpy())
        
        # Save metrics dictionary for this trajectory
        all_results[f"trajectory_{traj_idx+1}"] = {
            "train_metrics": dict_eval_train,
            "test_metrics": dict_eval_test
        }

        # Ploting setup
        ens_traj_np = moving_average_ensemble(ens_traj_np, window_size=10)
        std = np.std(ens_traj_np, axis=0)
        mean = np.mean(ens_traj_np, axis=0)
        time = np.arange(timesteps) * dt
        
        ax.axvspan(800*dt, 1000*dt, color='grey', alpha=0.2, label='_nolegend_')
        ax.axvspan(1000*dt, timesteps*dt, color='grey', alpha=0.4, label='_nolegend_')
        ax.axvline(800*dt, color='black', alpha=1, linestyle='--', linewidth=0.3, label='_nolegend_')
        ax.axvline(1000*dt, color='black', alpha=1, linestyle='--', linewidth=0.3, label='_nolegend_')
        
        # 3 Sigma
        ax.fill_between(
            time, 
            mean[:, phi, index_x, index_y] - 3*std[:, phi, index_x, index_y],
            mean[:, phi, index_x, index_y] + 3*std[:, phi, index_x, index_y],
            color="#1f77b4", alpha=0.25, label=r"$\pm 3\sigma$"
        )
        # 1 Sigma
        ax.fill_between(
            time, 
            mean[:, phi, index_x, index_y] - std[:, phi, index_x, index_y],
            mean[:, phi, index_x, index_y] + std[:, phi, index_x, index_y],
            color="#ff7f0e", alpha=0.25, label=r"$\pm 1\sigma$"
        )

        # Ground Truth
        true_traj = Phi_test[traj_idx, :, phi, index_x, index_y].detach().cpu().numpy()
        ax.plot(
            np.arange(len(true_traj)) * dt, 
            true_traj,
            color="black", linestyle="--", linewidth=1.2, label="Ground Truth"
        )
        
        # Mean Flow
        ax.plot(
            time, 
            mean[:, phi, index_x, index_y],
            color="red", label="Mean Forecast", linewidth=1
        )

        tick_pos = [400*dt, 900*dt, 1100*dt]
        tick_labels = [r'$T_{train}$', r'$T_{test}$', 'Forecast']
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels)
        
        ax.set_title(f"Trajectory {traj_idx+1}", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_ylim(-5.5, 5.5)
        break

    # --- Global Adjustments ---
    fig.text(0.5, 0.01, "Temporal Domain", ha="center")
    fig.text(0.01, 0.5, "Field Amplitude ($\Phi$)", va="center", rotation="vertical")

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=5, frameon=False)

    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    
    # 2. Save the figures and text files inside the targeted directory
    fig_save_path = os.path.join(save_dir, "evaluation_plot.png")
    plt.savefig(fig_save_path, bbox_inches='tight', dpi=150)
    print(f"Saved figure to: {fig_save_path}")
    
    dict_save_path = os.path.join(save_dir, "results_dict.pkl")
    with open(dict_save_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"Saved dictionary data to: {dict_save_path}")
    
    # Clean closing of figures to clear up RAM/VRAM during iterative plots
    plt.close(fig)
    del model, ens_traj, ens_traj_np, mean, std, all_results, fig, axes
    
    gc.collect()
    
    # 4. Clear PyTorch's internal VRAM cache allocator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()