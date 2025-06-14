import numpy as np
import pandas as pd
from scipy import optimize
from arch import arch_model


class HestonStock:
    def __init__(self):
        self.vt_parameters = None

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
