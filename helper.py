import numpy as np
import pandas as pd
from scipy import optimize
from arch import arch_model


class HestonStock:
    def __init__(self):
        self.vt_parameters = None
        self.st_parameters = 0.98

    def calc_vol(self, vt_last, dt, t, z):
        kappa, theta, sigma, lambda_ = self.vt_parameters
        kappa_star = kappa + lambda_
        theta_star = kappa * theta / (kappa_star)

        drift = kappa_star * (theta_star - vt_last) * dt
        diffusion = sigma * np.sqrt(vt_last * t) * z
        return max(drift + diffusion, 0)

    def calc_price(self, vt_last, st_last, dt, t, z):
        mu = self.st_parameters

        drift = mu * st_last * dt
        diffusion = np.sqrt(vt_last * t) * st_last * z
        return drift + diffusion

    def simulate(self, v0, s0, num_days, num_paths):
        dt = day / num_days

        vol_simulations = np.zeros((num_paths, num_days))
        price_simulations = np.zeros((num_paths, num_days))
        vol_simulations[:, 0] = v0
        price_simulations[:, 0] = s0

        for path in range(num_paths):
            for day in range(num_days - 1):
                vol_z, price_z = multivariate_normal.rvs(size=2)

                vt_last = vol_simulations[path, day]
                vt_new = self.calc_vol(vt_last, dt, dt * day, vol_z)
                vol_simulations[path, day + 1] = vt_new

                st_last = price_simulations[path, day]
                st_new = self.calc_price(vt_last, st_last, dt, dt * day, price_z)
                price_simulations[path, day + 1] = st_new

        return vol_simulations, price_simulations

    def _optimize_par_vol_path(self, vt, par0) -> None:
        z = np.random.normal(size=len(vt))
        t = np.linspace(0, 1, len(vt))

        def heston_vol_rmse(par, vt, t, z):
            kappa, theta, sigma, lambda_ = par
            kappa_star = kappa + lambda_
            theta_star = kappa * theta / (kappa_star)

            dvt_hat = kappa_star * (theta_star - vt) + sigma * np.sqrt(vt * t) * z
            return np.sqrt((dvt_hat[:-1].to_numpy() - vt.diff()[1:]) ** 2).sum()

        vt_pameters = optimize.minimize(
            fun=lambda x: heston_vol_rmse(x, vt, t, z), x0=par0
        )
        self.vt_parameters = vt_pameters["x"]

    def _fit_vol_timeseries_model(self, close_prices):
        returns = 100 * close_prices.pct_change().dropna()
        model = arch_model(returns, vol="Garch", p=1, q=1)
        fit = model.fit(disp="off")
        volatility = fit.conditional_volatility
        return volatility

    def fit_vol_parameters(self, close_prices, par0):
        volatility = self._fit_vol_timeseries_model(close_prices)
        vt = (volatility / 100) ** 2

        self._optimize_par_vol_path(vt, par0)
