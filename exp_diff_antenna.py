import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import matplotlib.pyplot as plt
import time
import pickle
import joblib
from WSR_algorithm import WMMSE_MIMO, MM_plus_MIMO
from unfolding_algorithm import MM_unfolding_single


# Define the parallel processing function
def parallel_processing(H, V_0, models, w, P_max, sigma, max_iter, tor, tor_mu):
    # WMMSE and MM algorithms
    _, obj_list, T_list = WMMSE_MIMO(w, H, V_0, P_max, sigma, max_iter, tor, tor_mu)
    _, obj_plus_list, T_plus_list = MM_plus_MIMO(w, H, V_0, P_max, sigma, max_iter, tor)
    _, obj_step_diag_list, T_step_diag_list = MM_unfolding_single(models['Diag']['model'], 'Diag', w, H, V_0, P_max, sigma, max_iter, tor)

    # print(f'\rRound [{index_all}/{total_iter}] Montecarlo Processing: [{n + 1:0=4d}/{N_mc}]', end='')

    return obj_list, T_list, obj_plus_list, T_plus_list, obj_step_diag_list, T_step_diag_list

# Signal model parameters
print(f'CPU number: {joblib.cpu_count()}')
num_monte_carlo = 1000
N_r = 4                 # number of receive antennas
N_s = 4                 # length of the transmit signal
K_list = np.int64(np.power(2, [i for i in range(2, 3)]))        # number of users
N_t_list = np.int64(np.power(2, [i for i in range(2, 8)]))      # number of transmit antennas
total_iter = len(K_list) * len(N_t_list)
index_all = 0

results = {
    'config': {
        'num_monte_carlo': num_monte_carlo,
        'N_r': N_r,
        'N_s': N_s,
        'K_list': K_list,
        'N_t_list': N_t_list,
    },
    'WMMSE': {
        'WSR': np.zeros((len(K_list), len(N_t_list))),
        'CPU_time': np.zeros((len(K_list), len(N_t_list))),
    },
    'MM': {
        'WSR': np.zeros((len(K_list), len(N_t_list))),
        'CPU_time': np.zeros((len(K_list), len(N_t_list))),
    },
    'MM-Net': {
        'WSR': np.zeros((len(K_list), len(N_t_list))),
        'CPU_time': np.zeros((len(K_list), len(N_t_list))),
    },
}

index_i = 0
for K in K_list:
    index_j = 0
    for N_t in N_t_list:
        index_all += 1
        w = np.ones(K)          # weight for each user
        SNR = 0                # signal to noise ratio
        P_max = K * N_t * N_s                           # maximum power budget
        sigma = np.sqrt(P_max * 10 ** (-SNR / 10))      # noise power

        data_type = torch.float64                       # model datatype (numpy default float64)
        max_iter = np.int64(2e1)                        # WSR algorithm maximum iteration

        tor = 0                 # WSR algorithm stop criterion
        tor_mu = 1e-4           # Lagrangian multiplier accuracy
        eta = 1e2
        alpha = 0.5
        beta = 0.5
        step_size = 1e1

        num_hidden_FCN = 2
        input_size_FCN = 2 * N_t ** 2
        hidden_sizes_FCN = [N_t for _ in range(num_hidden_FCN)]
        output_size_FCN = 1
        
        num_hidden_diag = 2
        input_size_diag_NN = N_t
        hidden_sizes_diag_NN = [N_t for _ in range(num_hidden_diag)]
        output_size_diag_NN = 1

        class Diag_Stepsize_NN(nn.Module):
            def __init__(self, input_size, hidden_sizes, output_size, slope=0.01):
                super(Diag_Stepsize_NN, self).__init__()
                self.hidden_layers = nn.ModuleList()
                self.leaky_relu = nn.LeakyReLU(negative_slope=slope)

                prev_size = input_size
                for hidden_size in hidden_sizes:
                    self.hidden_layers.append(nn.Linear(prev_size, hidden_size, dtype=data_type))
                    prev_size = hidden_size
                
                self.output_layer = nn.Linear(prev_size, output_size, dtype=data_type)

            def forward(self, x):
                for layer in self.hidden_layers:
                    x = self.leaky_relu(layer(x))
                x = F.softplus(self.output_layer(x))

                return x

        models = {
            'Diag': {
                'model': Diag_Stepsize_NN(input_size_diag_NN, hidden_sizes_diag_NN, output_size_diag_NN),
                },
            }

        for model_name, config in models.items():
            config['model'].load_state_dict(torch.load(f'Store_models/model_{model_name}_{N_t}N_t_{K}K_{max_iter}T_{SNR}dB.pth', weights_only=True))
            config['model'].eval()

        obj_list = np.zeros((num_monte_carlo, max_iter + 1))
        obj_plus_list = np.zeros((num_monte_carlo, max_iter + 1))

        obj_step_list = np.zeros((num_monte_carlo, max_iter + 1))
        obj_step_diag_list = np.zeros((num_monte_carlo, max_iter + 1))

        T_list = np.zeros((num_monte_carlo, max_iter))
        T_plus_list = np.zeros((num_monte_carlo, max_iter))
        T_step_diag_list = np.zeros((num_monte_carlo, max_iter))

        # Generate Channel
        H_real = np.random.randn(num_monte_carlo, K, N_r, N_t) / np.sqrt(2)
        H_imag = np.random.randn(num_monte_carlo, K, N_r, N_t) / np.sqrt(2)
        H_all = H_real + 1j * H_imag
        P_0_all = np.random.rand(num_monte_carlo, 1, 1, 1) * P_max
        V_0_all = np.random.randn(num_monte_carlo, K, N_t, N_s) + 1j * np.random.randn(num_monte_carlo, K, N_t, N_s)
        V_0_all = np.sqrt(P_0_all) * V_0_all / np.sqrt(np.sum(np.abs(V_0_all) ** 2, axis=(1, 2, 3), keepdims=True))

        # # Debug
        # for n in range(num_monte_carlo):
        #     print(f'\rRound [{index_all}/{total_iter}] Montecarlo Processing: [{n + 1:0=4d}/{num_monte_carlo}]', end='')
        #     parallel_processing(H_all[n, :, :, :], V_0_all[n, :, :, :], \
        #      models, w, P_max, sigma, max_iter, tor, tor_mu)
            
        # Parallel processing
        time_start = time.time()
        results_list_monte_carlo = joblib.Parallel(n_jobs=-1, verbose=2)(
            joblib.delayed(parallel_processing)
            (H_all[n, :, :, :], V_0_all[n, :, :, :], \
             models, w, P_max, sigma, max_iter, tor, tor_mu)
            for n in range(num_monte_carlo))
        time_end = time.time()
        print(f'\rRound [{index_all}/{total_iter}] Monte Carlo CPU time: {time_end - time_start:.1f}s\n')

        # Store the results list
        for n, result in enumerate(results_list_monte_carlo):
            obj_list[n, :] = result[0]
            T_list[n, :] = result[1]
            obj_plus_list[n, :] = result[2]
            T_plus_list[n, :] = result[3]
            obj_step_diag_list[n, :] = result[4]
            T_step_diag_list[n, :] = result[5]

        obj = np.mean(obj_list, axis=0) / np.log(2)
        obj_plus = np.mean(obj_plus_list, axis=0) / np.log(2)
        obj_step_diag = np.mean(obj_step_diag_list, axis=0) / np.log(2)

        T = np.mean(T_list, axis=0)
        T_plus = np.mean(T_plus_list, axis=0)
        T_step_diag = np.mean(T_step_diag_list, axis=0)
        
        # Store the results
        results['WMMSE']['WSR'][index_i, index_j] = obj[-1]
        results['WMMSE']['CPU_time'][index_i, index_j] = T[-1]
        results['MM']['WSR'][index_i, index_j] = obj_plus[-1]
        results['MM']['CPU_time'][index_i, index_j] = T_plus[-1]
        results['MM-Net']['WSR'][index_i, index_j] = obj_step_diag[-1]
        results['MM-Net']['CPU_time'][index_i, index_j] = T_step_diag[-1]
        index_j += 1
    index_i += 1

# Store the results
with open('Store_results/diff_antennas.pkl', 'wb') as f:
    pickle.dump(results, f)

#%% Show the results
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

# Load the results
with open('Store_results/diff_antennas.pkl', 'rb') as f:
    results = pickle.load(f)

K_list = results['config']['K_list']            # number of users
N_t_list = results['config']['N_t_list']        # number of transmit antennas

# Plot the results
size = 22
legend_size = 18
width = 2
colors = ['#4C72B0', '#DD8452', '#55A868', '#B84A39', '#8C8C8C', '#CCB974', '#DA8BC3', '#64B5CD', '#937860', '#8172B3', '#C44E52']
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Times New Roman'
marker_list = ['o', '^', '*', 's', 'v', 'd']

plt.figure(1)
plt.semilogx(N_t_list, results['WMMSE']['WSR'][0, :], label='WMMSE', color=colors[2], marker='v', linewidth=width, base=2)
plt.semilogx(N_t_list, results['MM-Net']['WSR'][0, :], label='MM-Net', color=colors[0], marker='s', linewidth=width, base=2)
plt.semilogx(N_t_list, results['MM']['WSR'][0, :], label='MM', color=colors[1], marker='^', linewidth=width, base=2)

plt.xlabel('Number of Transmit Antennas', fontsize=size)
plt.ylabel('WSR (bits/s/Hz)', fontsize=size)
plt.xticks(N_t_list, N_t_list, fontsize=size)
plt.yticks(fontsize=size)
plt.legend(loc='upper left', fontsize=legend_size)
# plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, fontsize=size)
plt.grid(True)
plt.tight_layout()

plt.savefig(f'figures/pdf/diff_antennas_WSR.pdf')
plt.savefig(f'figures/png/diff_antennas_WSR.png', dpi=300)

plt.show()

fig, ax = plt.subplots()
plt.loglog(N_t_list, results['WMMSE']['CPU_time'][0, :], label='WMMSE', color=colors[2], marker='v', linewidth=width)
plt.loglog(N_t_list, results['MM-Net']['CPU_time'][0, :], label='MM-Net', color=colors[0], marker='s', linewidth=width)
plt.loglog(N_t_list, results['MM']['CPU_time'][0, :], label='MM', color=colors[1], marker='^', linewidth=width)

ax.set_xscale('log', base=2)
ax.set_yscale('log', base=10)
plt.xlabel('Number of Transmit Antennas', fontsize=size)
plt.ylabel('CPU Time (s)', fontsize=size)
plt.xticks(N_t_list, N_t_list, fontsize=size)
plt.yticks(fontsize=size)
plt.legend(loc='upper left', fontsize=legend_size)
# plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, fontsize=size)
plt.grid(True)
plt.tight_layout()

plt.savefig(f'figures/pdf/diff_antennas_CPU_time.pdf')
plt.savefig(f'figures/png/diff_antennas_CPU_time.png', dpi=300)

plt.show()

# %%
