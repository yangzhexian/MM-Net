
import numpy as np
import matplotlib.pyplot as plt
import time
from one_dimensional_search import bisection_method
from copy import deepcopy


def Compute_WSR(w, H, V, sigma):
    dim = len(H.shape)
    sigma2 = sigma ** 2
    K, N_r, N_t = H.shape
    _, _, N_s = V.shape
    HV = np.matmul(H[:, np.newaxis, :, :], V[np.newaxis, :, :, :])
    HV_diag = HV[np.arange(K), np.arange(K), :, :]
    HV2 = np.matmul(HV, np.conj(np.transpose(HV, axes=(0, 1, 3, 2))))
    F = np.sum(HV2, axis=1) - HV2[np.arange(K), np.arange(K), :, :] + sigma2 * np.eye(N_r)
    SINR = np.conj(np.transpose(HV_diag, axes=(0, 2, 1))) @ np.linalg.solve(F, HV_diag)
    sign, rate = np.linalg.slogdet(np.eye(N_s) + SINR)
    WSR = np.sum(w * np.real(sign * rate))
    return WSR

def Compute_AB(H, V, sigma):
    dim = len(H.shape)
    sigma2 = sigma ** 2
    if len(H.shape) == 2:
        N_t, K = H.shape
        HV = H.T.conj() @ V
        HV2 = np.abs(HV) ** 2
        HV2_diag = np.diag(HV2)
        HV2_sum = np.sum(HV2, axis=1) + sigma2
        A = HV2_diag / (HV2_sum * (HV2_sum - HV2_diag))
        B = np.conj(np.diag(HV)) / (HV2_sum - HV2_diag)
    elif len(H.shape) == 3:
        K, N_r, N_t = H.shape
        _, _, N_s = V.shape
        H_tran = np.conj(np.transpose(H, axes=(0, 2, 1)))

        # HV = np.einsum('imn,jnl->ijml', H, V)   # Einstein summation

        # HV = np.tensordot(H, V, axes=([2], [1]))
        # HV = np.transpose(HV, axes=(0, 2, 1, 3))

        HV = np.matmul(H[:, np.newaxis, :, :], V[np.newaxis, :, :, :])
        HV_diag = HV[np.arange(K), np.arange(K), :, :]
        HV2 = np.matmul(HV, np.conj(np.transpose(HV, axes=(0, 1, 3, 2))))
        HV2_sum = np.sum(HV2, axis=1) + sigma2 * np.eye(N_r)
        HV2_diag = HV2[np.arange(K), np.arange(K), :, :]
        B = H_tran @ np.linalg.solve(HV2_sum - HV2_diag, HV_diag)
        A = H_tran @ np.linalg.solve(HV2_sum, HV_diag @ np.conj(np.transpose(B, axes=(0, 2, 1))))
        # # Symmetric form
        # A = H.T.conj() @ (np.linalg.inv(HV2_sum - HV2_diag) - np.linalg.inv(HV2_sum)) @ H
    else:
        raise ValueError('The dimension of H is not 2 or 3.')
    return A, B

def WMMSE_MIMO(w, H, V_0, P_max, sigma, max_iter=100, tor=0, tor_mu=1e-4):
    eps = 1e-4
    K, N_r, N_t = H.shape
    _, _, N_s = V_0.shape
    sigma2 = sigma ** 2
    V = deepcopy(V_0)
    obj = [Compute_WSR(w, H, V, sigma)]
    T = []
    t_start = time.perf_counter()
    for n in range(max_iter):
        U = np.complex128(np.zeros((K, N_r, N_s)))
        W = np.complex128(np.zeros((K, N_s, N_s)))
        for i in range(K):
            HV2_sum = sigma2 * np.complex128(np.eye(N_r))
            for j in range(K):
                H_iV_j = H[i, :, :] @ V[j, :, :]
                HV2_sum += H_iV_j @ np.conj(np.transpose(H_iV_j, (1, 0)))
            H_iV_i = H[i, :, :] @ V[i, :, :]
            U[i, :, :] = np.linalg.inv(HV2_sum) @ H_iV_i
            W[i, :, :] = np.linalg.inv(np.eye(N_s) - np.conj(np.transpose(U[i, :, :])) @ H_iV_i)

        quadratic_term = np.complex128(np.zeros((N_t, N_t)))
        for j in range(K):
            U_jH_j = np.conj(np.transpose(U[j, :, :])) @ H[j, :, :]
            quadratic_term += w[i] * np.conj(np.transpose(U_jH_j)) @ W[j, :, :] @ U_jH_j

        linear_term = np.complex128(np.zeros((K, N_t, N_s)))
        for i in range(K):
            linear_term[i, :, :] = np.conj(np.transpose(H[i, :, :])) @ U[i, :, :] @ W[i, :, :]
            V[i, :, :] = w[i] * np.linalg.lstsq(quadratic_term, linear_term[i, :, :])[0]

        if np.sum(np.abs(V) ** 2) > P_max:
            Lambda, U_eig = np.linalg.eig(quadratic_term)
            Psi = np.complex128(np.zeros((N_t, N_t)))
            for i in range(K):
                Psi += linear_term[i, :, :] @ np.conj(np.transpose(linear_term[i, :, :]))
            Psi = U_eig.T.conj() @ Psi @ U_eig
            Lambda = np.maximum(np.real(Lambda), 0)
            Psi_diag = np.real(np.diag(Psi))
            lb, ub = eps, np.sqrt(np.sum(Psi_diag) / P_max)
            f = lambda mu:  np.sum(Psi_diag / np.reshape(Lambda + mu, (1, -1)) ** 2) - P_max
            while f(lb) * f(ub) > 0:
                lb /= 2
                if lb < 1e-12:
                    break
            
            if lb < 1e-12:
                V = V * min(np.sqrt(P_max / np.sum(np.abs(V) ** 2)), 1)
            else:
                # Use bisection method
                mu_opt = bisection_method(f, lb, ub, tor_mu)
                for i in range(K):
                    V[i, :, :] = w[i] * np.linalg.solve(quadratic_term + mu_opt * np.eye(N_t), linear_term[i, :, :])
        
        obj.append(Compute_WSR(w, H, V, sigma))
        T.append(time.perf_counter() - t_start)
        if np.abs(obj[n + 1] - obj[n]) < tor:
            return V, obj, T
    return V, obj, T

def MM_MIMO(w, H, V_0, P_max, sigma, max_iter=100, tor=0, tor_mu=1e-4):
    K, N_r, N_t = H.shape
    _, _, N_s = V_0.shape
    sigma2 = sigma ** 2
    V = deepcopy(V_0)
    obj = [Compute_WSR(w, H, V, sigma)]
    T = []
    t_start = time.perf_counter()
    for n in range(max_iter):
        A, B = Compute_AB(H, V, sigma)
        A_sum = np.sum(w[:, np.newaxis, np.newaxis] * A, axis=0)
        # A_sum = np.tile(A_sum, (K, 1, 1))
        wB = w[:, np.newaxis, np.newaxis] * B

        try:
            V = np.linalg.solve(A_sum, wB)
        except np.linalg.LinAlgError:
            for i in range(np.shape(wB)[0]):
                V[i, :, :], _, _, _ = np.linalg.lstsq(A_sum, wB[i, :, :])

        if np.sum(np.abs(V) ** 2) > P_max:
            Lambda, U = np.linalg.eig(A_sum)
            UH = U.T.conj() @ wB
            Lambda = np.real(Lambda)
            UH_diag = np.real(np.sum(UH * UH.conj(), 2))
            lb, ub = 0, np.sqrt(np.sum(UH_diag) / P_max)
            f = lambda mu:  np.sum(UH_diag / np.reshape(Lambda + mu, (1, -1)) ** 2) - P_max
            # Use bisection method
            mu_opt = bisection_method(f, lb, ub, tor_mu)
            V = U @ (np.reshape(1 / (Lambda + mu_opt), (-1, 1)) * U.T.conj()) @ wB
        
        obj.append(Compute_WSR(w, H, V, sigma))
        T.append(time.perf_counter() - t_start)
        if np.abs(obj[n + 1] - obj[n]) < tor:
            return V, obj, T
    return V, obj, T

def MM_plus_MIMO(w, H, V_0, P_max, sigma, max_iter=100, tor=0):
    K, N_r, N_t = H.shape
    _, _, N_s = V_0.shape
    sigma2 = sigma ** 2
    V = deepcopy(V_0)
    obj = [Compute_WSR(w, H, V, sigma)]
    T = []
    t_start = time.perf_counter()
    for n in range(max_iter):
        A, B = Compute_AB(H, V, sigma)
        A_sum = np.sum(w[:, np.newaxis, np.newaxis] * A, axis=0)
        wB = w[:, np.newaxis, np.newaxis] * B
        eta = np.linalg.norm(A_sum, 'fro')
        # eta = np.linalg.norm(A_sum, 2)
        step_size = 1 / eta
        V += step_size * (wB - A_sum @ V)
        V = V * min(np.sqrt(P_max / np.sum(np.abs(V) ** 2)), 1)

        obj.append(Compute_WSR(w, H, V, sigma))
        T.append(time.perf_counter() - t_start)
        if np.abs(obj[n + 1] - obj[n]) < tor:
            return V, obj, T
    return V, obj, T

# Algorithm testing
if __name__ == '__main__':
    # np.random.seed(1)
    K, N_r, N_t, N_s = 4, 4, 16, 4
    SNR = 10
    P_max = 1
    sigma = np.sqrt(P_max * 10 ** (-SNR / 10))
    w = np.ones(K)
    max_iter = np.int64(1e2)
    tor = 0
    tor_mu = 1e-12
    
    # MIMO Case
    H_MIMO = (np.random.randn(K, N_r, N_t) + 1j * np.random.randn(K, N_r, N_t)) / np.sqrt(2)

    H_MIMO_tran = np.conj(np.transpose(H_MIMO, axes=(0, 2, 1)))
    V_MIMO_0 = np.linalg.solve(np.eye(N_t)[np.newaxis, ...] + (P_max / (K * sigma ** 2)) * 
                                np.sum(H_MIMO_tran @ H_MIMO, axis=0, keepdims=True), H_MIMO_tran)       # RZF
    V_MIMO_0 = np.sqrt(P_max / np.sum(np.abs(V_MIMO_0) ** 2)) * V_MIMO_0
    
    V_MIMO, obj_MIMO, T_MIMO = MM_MIMO(w, H_MIMO, V_MIMO_0, P_max, sigma, max_iter, tor, tor_mu)
    V_plus_MIMO, obj_plus_MIMO, T_plus_MIMO = MM_plus_MIMO(w, H_MIMO, V_MIMO_0, P_max, sigma, max_iter, tor)
    V_WMMSE, obj_WMMSE_MIMO, T_WMMSE_MIMO = WMMSE_MIMO(w, H_MIMO, V_MIMO_0, P_max, sigma, max_iter, tor, tor_mu)

    # fontsize, linewidth, color
    size = 16
    width = 2
    # colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860', '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD']
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    # Plot the convergence curve
    plt.figure(1, figsize=(8, 6))
    x_range_MIMO = np.arange(len(obj_MIMO)) + 1
    x_range_plus_MIMO = np.arange(len(obj_plus_MIMO)) + 1
    x_range_WMMSE_MIMO = np.arange(len(obj_WMMSE_MIMO)) + 1

    plt.plot(x_range_MIMO, obj_MIMO, label='MM', linewidth=width, color=colors[0])
    plt.plot(x_range_plus_MIMO, obj_plus_MIMO, label='MM+', linewidth=width, color=colors[1])
    plt.plot(x_range_WMMSE_MIMO, obj_WMMSE_MIMO, label='WMMSE', linewidth=width, color=colors[2])
    plt.xlabel('Iteration', fontsize=size)
    plt.ylabel('Weighted Sum Rate (nat/s/Hz)', fontsize=size)
    plt.xticks(fontsize=size)
    plt.yticks(fontsize=size)
    plt.xlim(left=1)
    plt.legend(loc='lower right', fontsize=size)
    plt.grid(True)
    plt.title('Convergence', fontsize=size)

    # Show
    plt.tight_layout()
    plt.savefig(f'figures/pdf/WSR_convergence_{K}K{N_t}N_t{N_r}N_r{N_s}N_s_{SNR}dB.pdf')
    plt.savefig(f'figures/png/WSR_convergence_{K}K{N_t}N_t{N_r}N_r{N_s}N_s_{SNR}dB.png', dpi=300)
    plt.show()

    # Plot the algorithm runtime curve with log time
    plt.figure(2, figsize=(8, 6))
    plt.semilogx(T_MIMO, obj_MIMO[1:], label='MM', linewidth=width, color=colors[0])
    plt.semilogx(T_plus_MIMO, obj_plus_MIMO[1:], label='MM+', linewidth=width, color=colors[1])
    plt.semilogx(T_WMMSE_MIMO, obj_WMMSE_MIMO[1:], label='WMMSE', linewidth=width, color=colors[2])
    plt.xlabel('Time (s)', fontsize=size)
    plt.ylabel('Weighted Sum Rate (nat/s/Hz)', fontsize=size)
    plt.xticks(fontsize=size)
    plt.yticks(fontsize=size)
    plt.legend(loc='lower right', fontsize=size)
    plt.grid(True)
    plt.title('Runtime', fontsize=size)

    # Show
    plt.tight_layout()
    plt.savefig(f'figures/pdf/WSR_runtime_{K}K{N_t}N_t{N_r}N_r{N_s}N_s_{SNR}dB.pdf')
    plt.savefig(f'figures/png/WSR_runtime_{K}K{N_t}N_t{N_r}N_r{N_s}N_s_{SNR}dB.png', dpi=300)
    plt.show()