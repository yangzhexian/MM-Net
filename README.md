# MM-Net: Recurrent Unfolding with Adaptive Majorization for Weighted Sum-Rate Beamforming

This repository contains the Python implementation of the algorithms presented in the paper:

> **MM-Net: Recurrent Unfolding with Adaptive Majorization for Weighted Sum-Rate Beamforming**  
> Zhexian Yang, Zepeng Zhang, Ziping Zhao

![MM-Net Architecture](images/MM_Net_architecture.svg)

The project provides optimization-based and deep learning-based methods for weighted sum-rate (WSR) maximization in MIMO broadcast channels. It includes:
- Classical WMMSE and MM algorithms
- A deep unfolded network (MM-Net) that learns the step size parameters to speed up convergence

## Repository Structure

```
.
├── README.md
├── WSR_algorithm.py          # Core optimization algorithms
├── unfolding_algorithm.py    # MM-Net implementation
├── one_dimensional_search.py # Auxiliary line-search methods
├── exp_diff_antenna.py       # Experiment: performance vs. number of transmit antennas
├── exp_diff_channels.py      # Experiment: convergence behavior over random channels
├── Store_models/             # Folder containing pre-trained MM-Net models
├── Store_results/            # Output folder for Monte Carlo results
└── figures/                  # Output folder for generated plots
└── images/                   # Images in README.md
```

## Requirements

- Python 3.12
- NumPy
- PyTorch 2.7.0
- Matplotlib
- Joblib

Install dependencies with:

```bash
pip install torch>=2.7.0 numpy matplotlib joblib
```

## Pre-trained Models

The MM-Net requires pre-trained neural networks for step-size prediction.  

For each scenario (`N_t`, `K`, `max_iter`, `SNR`), a model file named  
`model_Diag_{N_t}Nt_{K}K_{max_iter}T_{SNR}dB.pth` should be placed inside the `Store_models/` directory.

## Usage

### Quick Test

To verify the algorithms on a single random channel, run:

```bash
python WSR_algorithm.py
```

This will generate convergence and runtime plots for WMMSE and MM.

### Reproducing Paper Results

#### Experiment 1: Varying Number of Transmit Antennas

```bash
python exp_diff_antenna.py
```

This script performs 1000 Monte Carlo runs for each antenna configuration (from 4 to 128) and saves the results in `Store_results/diff_antennas.pkl`.  

After execution, it plots WSR and CPU time versus the number of transmit antennas.

> **Note:** This experiment may take several minutes to complete. You can reduce the number of Monte Carlo runs by modifying `num_monte_carlo` in the script.

#### Experiment 2: Convergence Behavior Over Channels

```bash
python exp_diff_channels.py
```

This script runs 1000 Monte Carlo simulations for a fixed antenna setup (e.g., 128 transmit antennas, 4 users) and saves the averaged convergence curves.  

It produces two figures:
- WSR vs. iteration
- WSR vs. CPU time

The results are stored as `Store_results/monte_carlo_{N_t}Nt_{K}K_{max_iter}T_{SNR}dB.pkl`.

### Customizing Parameters

You can adjust system parameters (number of users, antennas, SNR, etc.) directly inside the experiment scripts. The main parameters are:

- `K`: number of users
- `N_t`: number of transmit antennas
- `N_r`: number of receive antennas
- `N_s`: number of data streams
- `SNR`: signal-to-noise ratio (dB)
- `max_iter`: maximum number of iterations
- `num_monte_carlo`: number of Monte Carlo trials

## Contact

For questions or issues, please open an issue on GitHub or contact zhexianyang@shanghaitech.edu.cn
