
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


# Multiprocess count
print(f'\nCPU number: {joblib.cpu_count()}')

# Signal model parameters
num_monte_carlo = 1000
K = 4                   # number of users
N_r = 4                 # number of receive antennas
N_t = 128                # number of transmit antennas
N_s = 4                 # length of the transmit signal

w = np.ones(K)          # weight for each user

SNR = 0                 # signal to noise ratio
P_max = K * N_t * N_s                           # maximum power budget
sigma = np.sqrt(P_max * 10 ** (-SNR / 10))      # noise power

data_type = torch.float64                       # model datatype (numpy default float64)
max_iter = np.int64(2e1)                        # WSR algorithm maximum iteration

tor = 0                 # WSR algorithm stop criterion
tor_mu = 1e-4           # Lagrangian multiplier accuracy

num_hidden_diag = 2
input_size_diag_NN = N_t
hidden_sizes_diag_NN = [N_t for _ in range(num_hidden_diag)]
output_size_diag_NN = 1

results = {
    'config':{
        'K': K,
        'N_r': N_r,
        'N_t': N_t,
        'N_s': N_s,
        'SNR': SNR,
        'max_iter': max_iter
    },
    'WMMSE': {
        'WSR': np.zeros((max_iter, )),
        'CPU_time': np.zeros((max_iter, )),
    },
    'MM': {
        'WSR': np.zeros((max_iter, )),
        'CPU_time': np.zeros((max_iter, )),
    },
    'MM-Net': {
        'WSR': np.zeros((max_iter, )),
        'CPU_time': np.zeros((max_iter, )),
    },
}

def process_monte_carlo(n, H, V_0, models, w, P_max, sigma, max_iter, tor, tor_mu):
    # WMMSE and MM algorithms
    _, obj, T = WMMSE_MIMO(w, H, V_0, P_max, sigma, max_iter, tor, tor_mu)
    _, obj_plus, T_plus = MM_plus_MIMO(w, H, V_0, P_max, sigma, max_iter, tor)
    _, obj_step_diag, T_step_diag = MM_unfolding_single(models['Diag']['model'], 'Diag', w, H, V_0, P_max, sigma, max_iter, tor)

    # print(f'\rMontecarlo Processing: [{n + 1:0=4d}/{num_monte_carlo}]', end='')

    return obj, T, obj_plus, T_plus, obj_step_diag, T_step_diag,
    
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
obj_plus_SQUAREM_list = np.zeros((num_monte_carlo, max_iter + 1))

obj_step_list = np.zeros((num_monte_carlo, max_iter + 1))
obj_step_diag_list = np.zeros((num_monte_carlo, max_iter + 1))
obj_step_CNN_list = np.zeros((num_monte_carlo, max_iter + 1))

obj_step_multi_list = np.zeros((num_monte_carlo, max_iter + 1))
obj_step_diag_multi_list = np.zeros((num_monte_carlo, max_iter + 1))
obj_step_CNN_multi_list = np.zeros((num_monte_carlo, max_iter + 1))

obj_step_sum_list = np.zeros((num_monte_carlo, max_iter + 1))
obj_step_diag_sum_list = np.zeros((num_monte_carlo, max_iter + 1))
obj_step_CNN_sum_list = np.zeros((num_monte_carlo, max_iter + 1))

obj_IAIDNN = np.zeros((num_monte_carlo, 1))
obj_WMMSE_Net = np.zeros((num_monte_carlo, max_iter + 1))

obj_PGD_list = np.zeros((num_monte_carlo, max_iter + 1))
obj_PGD_backtracking_list = np.zeros((num_monte_carlo, max_iter + 1))

T_list = np.zeros((num_monte_carlo, max_iter))
T_plus_list = np.zeros((num_monte_carlo, max_iter))
T_plus_SQUAREM_list = np.zeros((num_monte_carlo, max_iter))

T_step_list = np.zeros((num_monte_carlo, max_iter))
T_step_diag_list = np.zeros((num_monte_carlo, max_iter))
T_step_CNN_list = np.zeros((num_monte_carlo, max_iter))

T_step_multi_list = np.zeros((num_monte_carlo, max_iter))
T_step_diag_multi_list = np.zeros((num_monte_carlo, max_iter))
T_step_CNN_multi_list = np.zeros((num_monte_carlo, max_iter))

T_step_sum_list = np.zeros((num_monte_carlo, max_iter))
T_step_diag_sum_list = np.zeros((num_monte_carlo, max_iter))
T_step_CNN_sum_list = np.zeros((num_monte_carlo, max_iter))

T_PGD_list = np.zeros((num_monte_carlo, max_iter))
T_PGD_backtracking_list = np.zeros((num_monte_carlo, max_iter))

T_IAIDNN = np.zeros((num_monte_carlo, 1))
T_WMMSE_Net = np.zeros((num_monte_carlo, max_iter))

# Generate Channel
H_real = np.random.randn(num_monte_carlo, K, N_r, N_t) / np.sqrt(2)
H_imag = np.random.randn(num_monte_carlo, K, N_r, N_t) / np.sqrt(2)
H_all = H_real + 1j * H_imag
H_all_tran = np.conj(np.transpose(H_all, axes=(0, 1, 3, 2)))

# Random beamforming
P_0 = np.random.rand(num_monte_carlo).reshape(-1, 1, 1, 1) * P_max
V_0_all = np.random.randn(num_monte_carlo, K, N_t, N_s) + 1j * np.random.randn(num_monte_carlo, K, N_t, N_s)
V_0_all = np.sqrt(P_0) * V_0_all / np.sqrt(np.sum(np.abs(V_0_all) ** 2, axis=(1, 2, 3), keepdims=True))

t_start = time.time()
results_list = joblib.Parallel(n_jobs=-1, verbose=2)\
    (joblib.delayed(process_monte_carlo)\
    (n, H_all[n, :, :, :], V_0_all[n, :, :, :], models, w, P_max, sigma, max_iter, tor, tor_mu) \
        for n in range(num_monte_carlo))
t_end = time.time()
print(f'\n{num_monte_carlo} Monte Carlo CPU time: {t_end - t_start:.1f}s')

for n, result in enumerate(results_list):
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

results['WMMSE']['WSR'] = obj
results['WMMSE']['CPU_time'] = T
results['MM']['WSR'] = obj_plus
results['MM']['CPU_time'] = T_plus
results['MM-Net']['WSR'] = obj_step_diag
results['MM-Net']['CPU_time'] = T_step_diag

# Store the results
with open(f'Store_results/monte_carlo_{N_t}N_t_{K}K_{max_iter}T_{SNR}dB.pkl', 'wb') as f:
    pickle.dump(results, f)

#%% Plot
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt


# N_t = 8
# K = 4
# max_iter = 20
# SNR = 0

# Load the results
with open(f'Store_results/monte_carlo_{N_t}N_t_{K}K_{max_iter}T_{SNR}dB.pkl', 'rb') as f:
    results = pickle.load(f)

K = results['config']['K']                  # number of users
N_r = results['config']['N_r']              # number of receive antennas
N_t = results['config']['N_t']              # number of transmit antennas
N_s = results['config']['N_s']              # length of the transmit signal
SNR = results['config']['SNR']              # signal to noise ratio
max_iter = results['config']['max_iter']    # WSR algorithm maximum iteration

obj = results['WMMSE']['WSR']
T = results['WMMSE']['CPU_time']
obj_plus = results['MM']['WSR']
T_plus = results['MM']['CPU_time']
obj_step_diag = results['MM-Net']['WSR']
T_step_diag = results['MM-Net']['CPU_time']

size = 22
legend_size = 18
width = 2
colors = ['#4C72B0', '#DD8452', '#55A868', '#B84A39', '#8C8C8C', '#CCB974', '#DA8BC3', '#64B5CD', '#937860', '#8172B3', '#C44E52']
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Times New Roman'

# Convergence Behavior
plt.figure(1)
plt.plot(range(len(obj)), obj, label='WMMSE', color=colors[2], marker='v', linewidth=width)
plt.plot(range(len(obj_step_diag)), obj_step_diag, label='MM-Net', color=colors[0], marker='s', linewidth=width)
plt.plot(range(len(obj_plus)), obj_plus, label='MM', color=colors[1], marker='^', linewidth=width)

plt.xlabel('Iteration', fontsize=size)
plt.ylabel('WSR (bits/s/Hz)', fontsize=size)
plt.xticks(range(0, len(obj), 5), fontsize=size)
plt.yticks(fontsize=size)
plt.legend(loc='lower right', fontsize=legend_size)
# plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, fontsize=size)
plt.grid(True)

plt.tight_layout()
plt.savefig(f'figures/pdf/MM_Net_convergence_{N_t}N_t_{K}_K_{max_iter}_T_{SNR}dB.pdf')
plt.savefig(f'figures/png/MM_Net_convergence_{N_t}N_t_{K}_K_{max_iter}_T_{SNR}dB.png', dpi=300)
plt.show()

# CPU Time
plt.figure(2)
plt.semilogx(T, obj[1:], label='WMMSE', color=colors[2], marker='v', linewidth=width)
plt.semilogx(T_step_diag, obj_step_diag[1:], label='MM-Net', color=colors[0], marker='s', linewidth=width)
plt.semilogx(T_plus, obj_plus[1:], label='MM', color=colors[1], marker='^', linewidth=width)

plt.xlabel('CPU Time (s)', fontsize=size)
plt.ylabel('WSR (bits/s/Hz)', fontsize=size)
plt.xticks(fontsize=size)
plt.yticks(fontsize=size)
plt.legend(loc='lower right', fontsize=legend_size)
# plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, fontsize=size)
plt.grid(True)

plt.tight_layout()
plt.savefig(f'figures/pdf/MM_Net_CPU_time_{N_t}N_t_{K}_K_{max_iter}_T_{SNR}dB.pdf')
plt.savefig(f'figures/png/MM_Net_CPU_time_{N_t}N_t_{K}_K_{max_iter}_T_{SNR}dB.png', dpi=300)
plt.show()

# %%
