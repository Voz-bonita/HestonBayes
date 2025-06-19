import numpy as np
import pandas as pd
from scipy import stats


def particle_filtering(
    N: int, Rtk: float, Rtk_1: float, dt, Vt_past, mu, kappa, theta, sigma, rho
):
    epsilon = stats.norm.rvs(size=N)

    zt = (Rtk - mu * dt - 1) / np.sqrt(dt * Vt_past)
    wt = zt * rho + epsilon * np.sqrt(1 - rho**2)

    Vt_candidates = (
        Vt_past + kappa * (theta - Vt_past) * dt + sigma * np.sqrt(dt * Vt_past) * wt
    )
    Wt_prob = (
        1
        / np.sqrt(2 * np.pi * Vt_candidates * dt)
        * np.exp(-1 / 2 * (Rtk_1 - mu * dt - 1) ** 2 / Vt_candidates / dt)
    )
    Wt_prob /= np.sum(Wt_prob)

    Ut = np.array([Vt_candidates, Wt_prob]).T
    Ut_sorted = Ut[Ut[:, 0].argsort()]
    Wt_sorted_cumsum = np.cumsum(Ut_sorted[:, 1])

    sampled_probabilities = stats.uniform.rvs(size=N)
    Vt_refined = np.zeros(N)
    for i, prob in enumerate(sampled_probabilities):
        j = np.argmin(prob > Wt_sorted_cumsum)
        if j != 0:
            j -= 1
        W_j = Ut_sorted[j, 1]
        W_j1 = Ut_sorted[j + 1, 1]
        V_j = Ut_sorted[j, 0]
        V_j1 = Ut_sorted[j + 1, 0]
        V_amplitude = V_j1 - V_j

        if j == 0:
            prev_prob = 0
            prob_avg = W_j + W_j1 / 2
        elif j == N - 2:
            prev_prob = Wt_sorted_cumsum[j - 1] + W_j / 2
            prob_avg = W_j / 2 + W_j1
        else:
            prev_prob = Wt_sorted_cumsum[j - 1] + W_j / 2
            prob_avg = W_j / 2 + W_j1 / 2

        v = (prob - prev_prob) / prob_avg * V_amplitude + V_j
        Vt_refined[i] = v
    return Vt_refined


def eta_to_mu(eta, dt):
    return (eta - 1) / dt


def sample_mu(rt: np.ndarray, vt: np.ndarray, dt: float, mu_eta_prior, tau_eta_prior):
    xt = 1 / np.sqrt(dt) / np.sqrt(vt[:-1])
    yt = xt * rt

    inner_x = xt.T @ xt  # scalar
    tau_eta = inner_x + tau_eta_prior
    ols_eta = 1 / inner_x * (xt.T @ yt)
    mu_eta = 1 / tau_eta * (tau_eta_prior * mu_eta_prior + inner_x * ols_eta)

    eta = stats.norm.rvs(loc=mu_eta, scale=np.sqrt(1 / np.sqrt(tau_eta)))
    return eta_to_mu(eta, dt)


def beta2_to_kappa(beta2, dt):
    return (1 - beta2) / dt


def beta1_to_theta(beta1, kappa, dt):
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
    xt_2 = 1 / np.sqrt(dt) * np.sqrt(vt[1:-1])
    xt_1 = 1 / np.sqrt(dt) / np.sqrt(vt[1:-1])
    yt = vt[2:] * xt_1

    xt = np.array([xt_1, xt_2]).T

    gram_matrix = xt.T @ xt
    inv_gram_matrix = np.linalg.inv(gram_matrix)

    precision_beta = precision_beta_prior + gram_matrix
    inv_precision_beta = np.linalg.inv(precision_beta)

    ols_beta = inv_gram_matrix @ xt.T @ yt
    mu_beta = inv_precision_beta @ (
        precision_beta_prior @ mu_beta_prior + gram_matrix @ ols_beta
    )

    betas = stats.multivariate_normal.rvs(
        mean=mu_beta, cov=sigma2_past * inv_precision_beta, size=1
    )
    kappa = beta2_to_kappa(betas[1], dt)
    theta = beta1_to_theta(betas[0], kappa, dt)

    a_sigma2 = a_sigma2_prior + n / 2
    b_sigma2 = b_sigma2_prior + 1 / 2 * (
        yt.T @ yt
        + mu_beta_prior.T @ precision_beta_prior @ mu_beta_prior
        - mu_beta.T @ precision_beta @ mu_beta
    )
    sigma2 = stats.invgamma.rvs(a_sigma2, b_sigma2, size=1)

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
    residuals = np.array([price_residuals, volatility_residuals]).T

    A = residuals.T @ residuals
    tau_psi = A[0, 0] + tau_prior_psi
    mu_psi = (A[0, 1] + mu_prior_psi * tau_prior_psi) / tau_psi
    a_omega = a_prior_omega + rt.shape[0] / 2
    b_omega = b_prior_omega + 1 / 2 * (A[1, 1] - A[0, 1] ** 2 / A[0, 0])

    omega = stats.invgamma.rvs(a_omega, b_omega, size=1)
    psi = stats.norm.rvs(loc=mu_psi, scale=np.sqrt(omega / tau_psi), size=1)

    rho = psi_omega_to_rho(psi, omega)
    return rho


def sample_parameters():
    pass


# Gruszka and Szwabiński, 2022
def estimate_heston(S: pd.Series, dt, ns, N):
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
        "mu": np.zeros(ns + 1),
        "kappa": np.zeros(ns + 1),
        "theta": np.zeros(ns + 1),
        "sigma": np.zeros(ns + 1),
        "rho": np.zeros(ns + 1),
    }
    parameters_sample["mu"][0] = 0.1
    parameters_sample["theta"][0] = 0.05
    parameters_sample["kappa"][0] = 1
    parameters_sample["sigma"][0] = 0.02
    parameters_sample["rho"][0] = -0.1

    R = S[1:] / S[:-1]
    n = len(R)

    for i in range(ns):  # MCMC
        vt = np.zeros(n + 1)
        Vt = np.repeat([parameters_sample["theta"][i]], N)
        vt[0] = np.mean(Vt)
        for k in range(0, n - 1):  # Particle Filtering
            Vt = particle_filtering(
                N,
                R[k],
                R[k + 1],
                dt,
                Vt,
                parameters_sample["mu"][i],
                parameters_sample["kappa"][i],
                parameters_sample["theta"][i],
                parameters_sample["sigma"][i],
                parameters_sample["rho"][i],
            )
            vt[k + 1] = np.mean(Vt)
        vt[n] = vt[n - 1]
        sample_parameters()

    mc_estimates = {
        parameter: np.mean(sample) for parameter, sample in parameters_sample.items()
    }
    return mc_estimates
