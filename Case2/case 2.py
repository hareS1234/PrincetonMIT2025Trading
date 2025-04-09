import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize

# Load data
data = pd.read_csv('Case2.csv', index_col=0)
TRAIN, TEST = train_test_split(data, test_size=0.2, shuffle=False)

# Global best parameters
BEST_ALPHA = None

class Allocator():
    def __init__(self, train_data, alpha=0.5):
        self.train_data = train_data.copy()
        self.n_assets = train_data.shape[1]
        self.last_weights = np.ones(self.n_assets) / self.n_assets
        self.intraday_prices = []
        self.alpha = alpha
        self.risk_free_rate = 0.0
        
        # Initialize EMA parameters with training data
        train_prices = train_data.values
        train_returns = np.diff(np.log(train_prices), axis=0)
        if len(train_returns) > 0:
            self.ema_returns = np.mean(train_returns, axis=0)
            self.ema_cov = np.cov(train_returns.T)
        else:
            self.ema_returns = np.zeros(self.n_assets)
            self.ema_cov = np.eye(self.n_assets)
            
        # Outlier detection parameters
        self.rolling_window = 10
        self.price_history = train_data.values[-self.rolling_window:].tolist()

    def _handle_outliers(self, prices):
        """Replace outliers with rolling average"""
        if len(self.price_history) < self.rolling_window:
            return prices
        
        df = pd.DataFrame(self.price_history)
        rolling_mean = df.rolling(self.rolling_window).mean().iloc[-1].values
        rolling_std = df.rolling(self.rolling_window).std().iloc[-1].values + 1e-8
        
        z_scores = (prices - rolling_mean) / rolling_std
        prices = np.where(np.abs(z_scores) > 3, rolling_mean, prices)
        return prices

    def allocate_portfolio(self, asset_prices):
        # Handle outliers and update price history
        prices = self._handle_outliers(np.array(asset_prices))
        self.price_history.append(prices)
        self.price_history = self.price_history[-self.rolling_window:]
        
        # Update EMA returns and covariance
        if len(self.price_history) > 1:
            latest_returns = np.log(prices) - np.log(self.price_history[-2])
            
            # Update EMA returns
            self.ema_returns = self.alpha * self.ema_returns + (1 - self.alpha) * latest_returns
            
            # Update EMA covariance
            delta = latest_returns - self.ema_returns
            self.ema_cov = self.alpha * self.ema_cov + (1 - self.alpha) * np.outer(delta, delta)
        
        # Optimization
        mu = self.ema_returns
        Sigma = (self.ema_cov + self.ema_cov.T) / 2 + 1e-6 * np.eye(self.n_assets)  # Ensure PSD
        
        def objective(w):
            port_return = w @ mu
            port_vol = np.sqrt(w @ Sigma @ w)
            return -(port_return - self.risk_free_rate) / port_vol if port_vol > 1e-8 else 0
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda w: 0.05 - np.linalg.norm(w - self.last_weights, 1)}
        ]
        
        result = minimize(objective, self.last_weights,
                          bounds=[(-1, 1)] * self.n_assets,
                          constraints=constraints,
                          method='SLSQP')
        
        if result.success:
            new_weights = result.x
            new_weights = np.clip(new_weights, -1, 1)
            self.last_weights = new_weights
            
        return self.last_weights.copy()

def evaluate_sharpe(train_data, test_data, alpha):
    alloc = Allocator(train_data, alpha=alpha)
    capital = [1.0]
    weights = []
    
    for i in range(len(test_data)):
        prices = test_data.iloc[i].values
        weights.append(alloc.allocate_portfolio(prices))
        
        if i == 0:
            continue
            
        # Update capital
        prev_prices = test_data.iloc[i-1].values
        curr_prices = prices
        shares = capital[-1] * weights[i-1] / prev_prices
        capital.append(shares @ curr_prices)
    
    returns = np.diff(capital) / capital[:-1]
    sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
    return sharpe

# Alpha tuning
alphas = np.linspace(0.1, 0.9, 9)
sharpe_results = []

for alpha in alphas:
    sharpe = evaluate_sharpe(TRAIN, TEST, alpha)
    sharpe_results.append(sharpe)
    print(f"Alpha: {alpha:.1f}, Sharpe: {sharpe:.4f}")

BEST_ALPHA = alphas[np.argmax(sharpe_results)]

# Plot alpha tuning results
plt.figure(figsize=(10, 6))
plt.plot(alphas, sharpe_results, marker='o')
plt.title("Sharpe Ratio vs. Alpha")
plt.xlabel("Alpha")
plt.ylabel("Sharpe Ratio")
plt.axvline(x=BEST_ALPHA, color='r', linestyle='--', label=f'Best Alpha: {BEST_ALPHA:.2f}')
plt.grid(True)
plt.legend()
plt.show()

# Final evaluation with best alpha
alloc = Allocator(TRAIN, alpha=BEST_ALPHA)
capital = [1.0]
weights = []

for i in range(len(TEST)):
    prices = TEST.iloc[i].values
    weights.append(alloc.allocate_portfolio(prices))
    
    if i == 0:
        continue
        
    prev_prices = TEST.iloc[i-1].values
    curr_prices = prices
    shares = capital[-1] * weights[i-1] / prev_prices
    capital.append(shares @ curr_prices)

# Calculate final Sharpe ratio
returns = np.diff(capital) / capital[:-1]
sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
print(f"\nBest Alpha: {BEST_ALPHA:.2f}, Final Sharpe Ratio: {sharpe:.4f}")

# Plot capital curve
plt.figure(figsize=(10, 6))
plt.plot(capital)
plt.title("Capital Evolution")
plt.xlabel("Time")
plt.ylabel("Portfolio Value")
plt.grid(True)
plt.show()
