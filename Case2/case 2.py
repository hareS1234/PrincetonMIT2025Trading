import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize

# Load the data
data = pd.read_csv('Case2.csv', index_col=0)
TRAIN, TEST = train_test_split(data, test_size=0.2, shuffle=False)

# Global best alpha (will be set after tuning)
BEST_ALPHA = None
BEST_LAMBDA = None

class Allocator():


    class Allocator():
    def __init__(self, train_data, alpha=None, lambda_=None):
        global BEST_ALPHA, BEST_LAMBDA
        self.train_data = train_data.copy()
        self.n_assets = train_data.shape[1]
        self.last_weights = np.ones(self.n_assets) / self.n_assets
        self.tick_count = 0
        self.intraday_prices = []
        self.ema_returns = None
        self.ema_cov = None
        self.alpha = alpha if alpha is not None else BEST_ALPHA
        self.lambda_ = lambda_ if lambda_ is not None else BEST_LAMBDA
        self.risk_free_rate = 0.0

    def _detect_outliers_and_replace(self, prices):
        if len(self.intraday_prices) < 10:
            return prices
        rolling = pd.DataFrame(self.intraday_prices).rolling(window=10, min_periods=1)
        rolling_mean = rolling.mean().iloc[-1].values
        rolling_std = rolling.std().iloc[-1].values + 1e-8
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

        if self.ema_returns is None:
            self.ema_returns = realized_returns
            self.ema_cov = np.cov(log_ret_matrix.T)
        else:
            self.ema_returns = self.alpha * self.ema_returns + (1 - self.alpha) * realized_returns
            centered = log_ret_matrix - realized_returns
            sample_cov = centered.T @ centered / (log_ret_matrix.shape[0] - 1)
            self.ema_cov = self.alpha * self.ema_cov + (1 - self.alpha) * sample_cov

        mu = self.ema_returns
        Sigma = self.ema_cov

        def markowitz_objective(w, lambda_):
            port_var = w.T @ Sigma @ w
            port_return = w.T @ mu
            return port_var - lambda_ * (port_return - self.risk_free_rate)

        best_portfolio = self.last_weights
        best_objective = np.inf

        for lam in range(1, 21):
            result = minimize(
                fun=lambda w: markowitz_objective(w, lam),
                x0=self.last_weights,
                bounds=[(-1, 1)] * self.n_assets,
                constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}],
                method='SLSQP',
                options={'maxiter': 300}
            )
            if result.success and markowitz_objective(result.x, lam) < best_objective:
                best_objective = markowitz_objective(result.x, lam)
                best_portfolio = result.x

        new_weights = best_portfolio
        delta = new_weights - self.last_weights
        max_turnover = 0.05
        delta_norm = np.linalg.norm(delta, 1)
        if delta_norm > max_turnover:
            new_weights = self.last_weights + delta * (max_turnover / delta_norm)

        self.last_weights = np.clip(new_weights, -1, 1)
        self.tick_count = 0
        self.intraday_prices = []
        return self.last_weights

# Sharpe-tuned lambda evaluator
def evaluate_lambda(train_data, test_data, alpha, lambda_):
    weights = np.full(shape=(len(test_data.index), train_data.shape[1]), fill_value=0.0)
    alloc = Allocator(train_data, alpha=alpha, lambda_=lambda_)
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
    return sharpe

# Tune lambda using best alpha
lambda_range = range(1, 21)
lambda_sharpes = []
for lam in lambda_range:
    sharpe = evaluate_lambda(TRAIN, TEST, alpha=BEST_ALPHA, lambda_=lam)
    lambda_sharpes.append(sharpe)

BEST_LAMBDA = lambda_range[np.argmax(lambda_sharpes)]
print(f"Best Lambda: {BEST_LAMBDA} with Sharpe Ratio: {max(lambda_sharpes):.4f}")

# Plot Sharpe vs Lambda
plt.figure(figsize=(10, 6))
plt.plot(lambda_range, lambda_sharpes, marker='o')
plt.title("Sharpe Ratio vs. Lambda (Markowitz)")
plt.xlabel("Lambda")
plt.ylabel("Sharpe Ratio")
plt.axvline(x=BEST_LAMBDA, color='red', linestyle='--', label=f'Best λ = {BEST_LAMBDA}')
plt.grid(True)
plt.legend()
plt.show()

 
        


def evaluate_alpha(train_data, test_data, alpha):
    weights = np.full(shape=(len(test_data.index), train_data.shape[1]), fill_value=0.0)
    alloc = Allocator(train_data, alpha=alpha)
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
    return sharpe

# Find best alpha
alpha_range = np.arange(0.1, 1.0, 0.1)
sharpe_values = []

for alpha in alpha_range:
    sharpe = evaluate_alpha(TRAIN, TEST, alpha)
    sharpe_values.append(sharpe)

BEST_ALPHA = alpha_range[np.argmax(sharpe_values)]
print(f"Best Alpha: {BEST_ALPHA:.2f} with Sharpe Ratio: {max(sharpe_values):.4f}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(alpha_range, sharpe_values, marker='o')
plt.axvline(x=BEST_ALPHA, color='red', linestyle='--', label=f'Best α = {BEST_ALPHA:.2f}')
plt.title("Sharpe Ratio vs. Alpha")
plt.xlabel("Alpha (EMA Smoothing Factor)")
plt.ylabel("Sharpe Ratio")
plt.grid(True)
plt.legend()
plt.show()

# === DO NOT MODIFY ===
def grading(train_data, test_data): 
    '''
    Grading Script
    '''
    weights = np.full(shape=(len(test_data.index),6), fill_value=0.0)
    alloc = Allocator(train_data)
    for i in range(0,len(test_data)):
        weights[i,:] = alloc.allocate_portfolio(test_data.iloc[i,:])
        if np.sum(weights < -1) or np.sum(weights > 1):
            raise Exception("Weights Outside of Bounds")
    
    capital = [1]
    for i in range(len(test_data) - 1):
        shares = capital[-1] * weights[i] / np.array(test_data.iloc[i,:])
        balance = capital[-1] - np.dot(shares, np.array(test_data.iloc[i,:]))
        net_change = np.dot(shares, np.array(test_data.iloc[i+1,:]))
        capital.append(balance + net_change)
    capital = np.array(capital)
    returns = (capital[1:] - capital[:-1]) / capital[:-1]
    
    if np.std(returns) != 0:
        sharpe = np.mean(returns) / np.std(returns)
    else:
        sharpe = 0
        
    return sharpe, capital, weights

# Final grading run with best alpha
sharpe, capital, weights = grading(TRAIN, TEST)
print(sharpe)

plt.figure(figsize=(10, 6), dpi=80)
plt.title("Capital")
plt.plot(np.arange(len(TEST)), capital)
plt.show()

plt.figure(figsize=(10, 6), dpi=80)
plt.title("Weights")
plt.plot(np.arange(len(TEST)), weights)
plt.legend(TEST.columns)
plt.show()
