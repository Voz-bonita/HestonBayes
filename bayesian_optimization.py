import numpy as np
import pandas as pd
from scipy import stats


dt = 1


def particle_filtering():
    pass


def eta_to_mu(eta):
    return (eta - 1) / dt


def sample_mu(rt: np.ndarray, vt: np.ndarray, dt: float, mu_eta_prior, tau_eta_prior):
    xt = 1 / np.sqrt(dt) / np.sqrt(vt)
    yt = xt * rt

    inner_x = xt.T @ xt  # scalar
    tau_eta = inner_x + tau_eta_prior
    ols_eta = 1 / inner_x * (xt.T @ yt)
    mu_eta = 1 / tau_eta * (tau_eta_prior * mu_eta_prior + inner_x * ols_eta)

    eta = stats.norm.rvs(loc=mu_eta, scale=np.sqrt(tau_eta))
    return eta_to_mu(eta)


def beta2_to_kappa(beta2):
    return (1 - beta2) / dt


def beta1_to_theta(beta1, kappa):
    return beta1 / kappa / dt


def sample_kappa_theta_sigma(
    vt: np.ndarray,
    dt: float,
    n: int,
    mu_beta_prior: np.ndarray,
    precision_beta_prior: np.ndarray,
    sigma2_past: float,
    a_sigma2_prior: float,
    b_sigma2_prior: float,
):
    xt_2 = 1 / np.sqrt(dt) * np.sqrt(vt[:-1])
    xt_1 = 1 / np.sqrt(dt) / np.sqrt(vt[:-1])
    yt = vt[1:] * xt_1

    xt = np.array([xt_1, xt_2]).reshape((-1, 2))

    gram_matrix = xt.T @ xt
    inv_gram_matrix = np.linalg.inv(gram_matrix)

    precision_beta = precision_beta_prior + gram_matrix
    inv_precision_beta = np.linalg.inv(precision_beta)

    ols_beta = inv_gram_matrix @ xt.T @ yt
    mu_beta = inv_precision_beta @ (
        precision_beta_prior @ mu_beta_prior + gram_matrix @ ols_beta
    )

    betas = stats.norm.rvs(loc=mu_beta, scale=sigma2_past * inv_precision_beta, size=2)
    kappa = beta2_to_kappa(betas[1])
    theta = beta1_to_theta(betas[0], kappa)

    a_sigma2 = a_sigma2_prior + n / 2
    b_sigma2 = b_sigma2_prior + 1 / 2 * (
        yt.T @ yt
        + mu_beta_prior.T @ precision_beta_prior @ mu_beta_prior
        - mu_beta.T @ precision_beta @ mu_beta
    )
    sigma2 = stats.invgamma.rvs(loc=a_sigma2, scale=b_sigma2, size=1)

    return kappa, theta, sigma2


def psi_omega_to_rho(psi, omega):
    return psi / np.sqrt(psi**2 + omega)


def sample_rho(
    rt: np.ndarray,
    vt: np.ndarray,
    dt: float,
    mu: float,
    kappa: float,
    theta: float,
    mu_prior_psi,
    tau_prior_psi,
    a_prior_omega,
    b_prior_omega,
):
    price_residuals = (rt - mu * dt - 1) / np.sqrt(dt * vt[:-1])
    volatility_residuals = (
        vt[1:] - vt[:-1] - kappa * (theta - vt[:-1]) * dt
    ) / np.sqrt(dt * vt[:-1])
    residuals = np.array([price_residuals, volatility_residuals]).reshape((-1, 2))

    A = residuals.T @ residuals
    tau_psi = A[0, 0] + tau_prior_psi
    mu_psi = (A[0, 1] + mu_prior_psi * tau_prior_psi) / tau_psi
    a_omega = a_prior_omega + rt.shape[0] / 2
    b_omega = b_prior_omega + 1 / 2 * (A[1, 1] - A[0, 1] ** 2 / A[0, 0])

    omega = stats.invgamma.rvs(loc=a_omega, scale=b_omega, size=1)
    psi = stats.norm.rvs(loc=mu_psi, scale=np.sqrt(omega / tau_psi), size=1)

    rho = psi_omega_to_rho(psi, omega)
    return rho


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
