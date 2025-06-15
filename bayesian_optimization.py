import numpy as np
import pandas as pd
from scipy import stats


dt = 1


def particle_filtering():
    pass


def sample_parameters():
    pass


# Gruszka and Szwabiński, 2022
def estimate_heston(S: pd.Series, dt, ns, N):
    n = len(S) - 1
    mu_prior_eta = 1.00125
    sigma_prior_eta = 0.001
    precision_prior_vol = np.array([[10, 0], [0, 5]])
    mu_prior_vol = np.array([35e-6, 0.988])
    a_prior_sigma = 149
    b_prior_sigma = 0.025
    mu_prior_psi = -0.45
    sigma_prior_psi = 0.3
    a_prior_omega = 1.03
    b_prior_omgea = 0.05
    # lambda_prior_jump = 0.15
    # mu_prior_jump = -0.96
    # sigma_prior_jump = 0.3

    parameters_sample = {
        "mu": np.zeros(ns),
        "kappa": np.zeros(ns),
        "theta": np.zeros(ns),
        "sigma": np.zeros(ns),
        "rho": np.zeros(ns),
    }
    R = S[1:] / S[:-1]

    for i in ns:  # MCMC
        for k in range(1, n - 1):  # Particle Filtering
            for j in range(1, N):
                particle_filtering()
        sample_parameters()

    mc_estimates = {
        parameter: np.mean(sample) for parameter, sample in parameters_sample.items()
    }
    return mc_estimates


def eta_to_mu(eta):
    return (eta - 1) / dt


def beta2_to_kappa(beta2):
    return (1 - beta2) / dt


def beta1_to_theta(beta1, kappa):
    return beta1 / kappa / dt


def psi_omega_to_rho(psi, omega):
    return psi / np.sqrt(psi**2 + omega)
