import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from WSR_algorithm import Compute_WSR, Compute_AB
from copy import deepcopy


def preprocess_PSD(PSD_matrix, mode):
    size = PSD_matrix.shape[0]

    if mode == 'FCN':
        np_input = np.concat((PSD_matrix.real.flatten(), PSD_matrix.imag.flatten()))
        network_input = torch.tensor(np_input).reshape(1, -1)
    elif mode == 'Diag':
        np_input = np.real(np.diag(PSD_matrix))
        network_input = torch.tensor(np_input)
    elif mode == 'CNN':
        np_input = np.stack((np.real(PSD_matrix), np.imag(PSD_matrix)), axis=0)
        network_input = torch.tensor(np_input).unsqueeze(0)
    else:
        raise ValueError("Invalid mode. Available modes are 'FCN', 'Diag', and 'CNN'.")
    
    return network_input

def MM_unfolding_single(model_single, mode, w, H, V_0, P_max, sigma, max_iter=100, tor=0):
    K, N_r, N_t = H.shape
    _, _, N_s = V_0.shape
    sigma2 = sigma ** 2
    V = deepcopy(V_0)
    obj = [Compute_WSR(w, H, V, sigma)]
    T = []
    t_start = time.perf_counter()
    model_single.eval()
    
    for n in range(max_iter):
        A, B = Compute_AB(H, V, sigma)
        A_sum = np.sum(w[:, np.newaxis, np.newaxis] * A, axis=0)
        wB = w[:, np.newaxis, np.newaxis] * B

        # Compute eta
        input = preprocess_PSD(A_sum, mode)
        
        with torch.no_grad():
            eta = model_single(input).numpy()

        V += eta * (wB - A_sum @ V)
        V = V * min(np.sqrt(P_max / np.sum(np.abs(V) ** 2)), 1)

        obj.append(Compute_WSR(w, H, V, sigma))
        T.append(time.perf_counter() - t_start)
        if np.abs(obj[n + 1] - obj[n]) < tor:
            return V, obj, T
    
    return V, obj, T
