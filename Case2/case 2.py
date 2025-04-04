import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize
import os

# Load the data
data = pd.read_csv('Case2.csv', index_col=0)

# Split data into training and test sets
TRAIN, TEST = train_test_split(data, test_size=0.2, shuffle=False)

class Allocator():
    def __init__(self, train_data):
        self.running_price_paths = train_data.copy()
        self.train_data = train_data.copy()
        self.n_assets = train_data.shape[1]
        self.daily_closes = train_data.iloc[29::30].reset_index(drop=True)
        self.daily_vols = self._calculate_realized_vol(train_data)
        self.last_weights = np.ones(self.n_assets) / self.n_assets
        self.tick_count = 0
        self.intraday_prices = []
        self.risk_free_rate = 0.0  # Set to 0.03 or other if desired

    def _calculate_realized_vol(self, data):
        vol_list = []
        for day in range(len(data) // 30):
            prices = data.iloc[day * 30:(day + 1) * 30]
            log_ret = np.diff(np.log(prices), axis=0)
            vol_list.append(np.sqrt((log_ret ** 2).sum(axis=0)))
        return pd.DataFrame(vol_list, columns=data.columns)

    def allocate_portfolio(self, asset_prices):
        self.intraday_prices.append(asset_prices)
        self.tick_count += 1

        if self.tick_count < 30:
            return self.last_weights

        prices_matrix = np.vstack(self.intraday_prices)
        log_ret_matrix = np.diff(np.log(prices_matrix), axis=0)
        realized_vol_today = np.sqrt(np.sum(log_ret_matrix ** 2, axis=0))

        self.daily_closes = pd.concat(
            [self.daily_closes, pd.DataFrame([asset_prices], columns=self.train_data.columns)],
            ignore_index=True)
        self.daily_vols = pd.concat(
            [self.daily_vols, pd.DataFrame([realized_vol_today], columns=self.train_data.columns)],
            ignore_index=True)

        returns = self.daily_closes.pct_change().dropna()
        if len(returns) < 30:
            self.tick_count = 0
            self.intraday_prices = []
            return self.last_weights

        mu = returns.mean().values
        lw = LedoitWolf()
        Sigma = lw.fit(returns).covariance_

        def objective(w):
            port_return = np.dot(w, mu)
            port_vol = np.sqrt(np.dot(w, Sigma @ w))
            return -(port_return - self.risk_free_rate) / port_vol if port_vol > 1e-8 else 0

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda w: 0.15 / np.sqrt(252) - np.sqrt(w @ Sigma @ w.T)}
        ]

        result = minimize(objective,
                          x0=self.last_weights,
                          bounds=[(-1, 1)] * self.n_assets,
                          constraints=constraints,
                          method='SLSQP',
                          options={'maxiter': 300})

        new_weights = result.x if result.success else self.last_weights
        delta = new_weights - self.last_weights
        max_turnover = 0.05
        delta_norm = np.linalg.norm(delta, 1)
        if delta_norm > max_turnover:
            new_weights = self.last_weights + delta * (max_turnover / delta_norm)

        self.last_weights = np.clip(new_weights, -1, 1)
        self.tick_count = 0
        self.intraday_prices = []
        return self.last_weights

def grading(train_data, test_data):
    weights = np.full(shape=(len(test_data.index), 6), fill_value=0.0)
    alloc = Allocator(train_data)
    for i in range(0, len(test_data)):
        weights[i, :] = alloc.allocate_portfolio(test_data.iloc[i, :])
        if np.any(weights[i, :] < -1) or np.any(weights[i, :] > 1):
            raise Exception("Weights Outside of Bounds")

    capital = [1]
    for i in range(len(test_data) - 1):
        shares = capital[-1] * weights[i] / np.array(test_data.iloc[i, :])
        balance = capital[-1] - np.dot(shares, np.array(test_data.iloc[i, :]))
        net_change = np.dot(shares, np.array(test_data.iloc[i + 1, :]))
        capital.append(balance + net_change)

    capital = np.array(capital)
    returns = (capital[1:] - capital[:-1]) / capital[:-1]

    sharpe = np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0
    return sharpe, capital, weights

# Run grading
sharpe, capital, weights = grading(TRAIN, TEST)
print(f"Sharpe Ratio: {sharpe:.4f}")

# Plotting
plt.figure(figsize=(10, 6), dpi=80)
plt.title("Capital Over Time")
plt.plot(np.arange(len(TEST)), capital)
plt.xlabel("Time")
plt.ylabel("Capital")
plt.grid(True)
plt.show()


# credits:
# https://github.com/sabvan/UChicagoTradingComp/blob/main/case2
# https://github.com/PaiViji/PythonFinance-PortfolioOptimization/blob/master/Lesson6_SharpeRatioOptimization/Lesson6_MainContent.ipynb
