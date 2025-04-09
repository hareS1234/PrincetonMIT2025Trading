import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize

# Load the data
data = pd.read_csv('Case2.csv', index_col=0)
TRAIN, TEST = train_test_split(data, test_size=0.2, shuffle=False)

class Allocator():
    def __init__(self, train_data, alpha):
        self.train_data = train_data.copy()
        self.n_assets = train_data.shape[1]
        self.last_weights = np.ones(self.n_assets) / self.n_assets
        self.tick_count = 0
        self.intraday_prices = []
        self.ema_returns = None
        self.ema_var = None
        self.alpha = alpha
        self.risk_free_rate = 0.0

    def _detect_outliers_and_replace(self, prices):
        if len(self.intraday_prices) < 10:
            return prices
        rolling_mean = pd.DataFrame(self.intraday_prices).rolling(window=10, min_periods=1).mean().iloc[-1].values
        rolling_std = pd.DataFrame(self.intraday_prices).rolling(window=10, min_periods=1).std().iloc[-1].values + 1e-8
        z_scores = (prices - rolling_mean) / rolling_std
        prices[np.abs(z_scores) > 3] = rolling_mean[np.abs(z_scores) > 3]
        return prices

    def allocate_portfolio(self, asset_prices):
        asset_prices = self._detect_outliers_and_replace(np.array(asset_prices))
        self.intraday_prices.append(asset_prices)
        self.tick_count += 1

        if self.tick_count < 30:
            return self.last_weights

        prices_matrix = np.vstack(self.intraday_prices)
        log_ret_matrix = np.diff(np.log(prices_matrix), axis=0)
        realized_returns = np.mean(log_ret_matrix, axis=0)
        realized_var = np.var(log_ret_matrix, axis=0)

        if self.ema_returns is None:
            self.ema_returns = realized_returns
            self.ema_var = realized_var
        else:
            self.ema_returns = self.alpha * self.ema_returns + (1 - self.alpha) * realized_returns
            self.ema_var = self.alpha * self.ema_var + (1 - self.alpha) * realized_var

        mu = self.ema_returns
        Sigma = np.diag(self.ema_var)

        def objective(w):
            port_return = np.dot(w, mu)
            port_vol = np.sqrt(np.dot(w, Sigma @ w))
            return -(port_return - self.risk_free_rate) / port_vol if port_vol > 1e-8 else 0

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: np.sqrt(w @ Sigma @ w.T) - np.sqrt(np.mean(self.ema_var))}
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

def grading(train_data, test_data, alpha):
    weights = np.full(shape=(len(test_data.index), train_data.shape[1]), fill_value=0.0)
    alloc = Allocator(train_data, alpha)
    for i in range(len(test_data)):
        weights[i, :] = alloc.allocate_portfolio(test_data.iloc[i, :])

    capital = [1]
    for i in range(len(test_data) - 1):
        shares = capital[-1] * weights[i] / np.array(test_data.iloc[i, :])
        balance = capital[-1] - np.dot(shares, np.array(test_data.iloc[i, :]))
        net_change = np.dot(shares, np.array(test_data.iloc[i + 1, :]))
        capital.append(balance + net_change)

    capital = np.array(capital)
    returns = (capital[1:] - capital[:-1]) / capital[:-1]
    sharpe = np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0
    return sharpe, capital

# Run simulation
alpha_range = np.arange(0.1, 1.0, 0.01)
sharpe_values = []

for alpha in alpha_range:
    sharpe, _ = grading(TRAIN, TEST, alpha)
    sharpe_values.append(sharpe)

# Plotting alpha vs Sharpe
plt.figure(figsize=(10, 6))
plt.plot(alpha_range, sharpe_values, marker='o')
plt.title("Sharpe Ratio vs. Alpha")
plt.xlabel("Alpha (EMA Smoothing Factor)")
plt.ylabel("Sharpe Ratio")
plt.grid(True)
plt.show()
