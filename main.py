from bayesian_optimization import estimate_heston
from scipy import stats
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os


def heston_next(last_st, last_vt, par, dt, t):
    mu = par["mu"]
    kappa = par["kappa"]
    theta = par["theta"]
    sigma2 = par["sigma"]
    rho = par["rho"]

    z1, z_add = stats.norm.rvs(size=2)
    z2 = rho * z1 + np.sqrt(1 - rho**2) * z_add
    vt_new = (
        last_vt
        + kappa * (theta - last_vt) * dt
        + np.sqrt(sigma2 * last_vt * t) * last_st * z2
    )
    st_new = last_st + mu * last_st * dt + np.sqrt(last_vt * t) * last_st * z1
    return st_new


def main(ticker):
    start_date = "2025-01-01"
    end_date = datetime.today().strftime("%Y-%m-%d")

    data = yf.download(ticker, start=start_date, end=end_date)
    data = data["Close"].dropna()

    dt = 1
    sample_size = 100
    full_data = data.to_numpy().flatten()
    window_size = 22 * 3
    remaining_size = full_data.shape[0] - window_size

    prediction = np.zeros(remaining_size)
    lower_bounds = np.zeros(remaining_size)
    upper_bounds = np.zeros(remaining_size)

    for idx in range(remaining_size):
        train_data = full_data[idx : window_size + idx]
        results = estimate_heston(train_data, dt, 100, 100)
        heston_par = results["par"]
        vt = results["vt"]

        pred_s = np.zeros(sample_size)

        while np.mean(pred_s) <= 0:
            for i in range(sample_size):
                pred_s[i] = heston_next(train_data[-1], vt[-1], heston_par, dt, 1)

        prediction[idx] = np.mean(pred_s)
        lower_bounds[idx] = np.quantile(pred_s, 0.05)
        upper_bounds[idx] = np.quantile(pred_s, 0.95)

    dates = data.index[window_size:]
    val_data = full_data[window_size:]
    pd.DataFrame(
        {
            "Date": dates,
            "Close": val_data,
            "Prediction": prediction,
            "LB": lower_bounds,
            "UB": upper_bounds,
        }
    ).to_parquet(f"data/{ticker}.parquet")

    # Convert string dates to datetime objects

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(
        dates,
        data.to_numpy().flatten()[window_size:],
        label="Real Observations",
        marker="o",
    )
    plt.plot(dates, prediction, label="Model Predictions", marker="s", linestyle="--")
    # plt.fill_between(dates, lower_bounds, upper_bounds, color='gray', alpha=0.3, label='Confidence Interval')

    # Formatting the date axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.gcf().autofmt_xdate()  # Auto-rotate date labels

    # Add labels and title
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title(f"{ticker} vs Heston Model")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"assets/{ticker}.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    files_found = [file.replace(".parquet", "") for file in os.listdir("data")]
    tickers = [
        "PETR4.SA",
        "BBAS3.SA",
        "ITUB4.SA",
        "CSMG3.SA",
        "SAPR11.SA",
        "CMIG4.SA",
        "ISAE4.SA",
        "BTC-USD",
    ]
    remaining_tickers = set(tickers) - set(files_found)
    for ticker in remaining_tickers:
        print(ticker)
        main(ticker)
