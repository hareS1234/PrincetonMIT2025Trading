import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf, OAS
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, leaves_list

# Enhanced parameter tuning with validation split
def tune_parameters(train):
    best_sharpe = -np.inf
    best_params = {'alpha': 0.96, 'lambda_': 3}  # Defaults if grid search fails
    
    # Create validation split
    train_sub, val_sub = train_test_split(train, test_size=0.3, shuffle=False)
    
    # Expanded parameter grid
    grid = {
        'alpha': [0.92, 0.95, 0.98],     # More reactive EMA
        'lambda_': [1, 2, 3, 5],          # Lower risk aversion
        'vol_target': [0.18, 0.20],       # Higher risk tolerance
        'max_weight': [0.7, 0.8]          # Allow concentration
    }
    
    # Iterate through all combinations
    from itertools import product
    for params in product(*grid.values()):
        alpha, lambda_, vol_target, max_weight = params
        alloc = Allocator(train_sub, alpha=alpha, lambda_=lambda_,
                         vol_target=vol_target, max_weight=max_weight)
        capital = [1.0]
        
        for i in range(len(val_sub)):
            prices = val_sub.iloc[i].values
            w = alloc.allocate_portfolio(prices)
            
            if i > 0:
                prev_prices = val_sub.iloc[i-1].values
                shares = capital[-1] * w / prev_prices
                capital.append(shares @ prices)
        
        returns = np.diff(capital)/capital[:-1]
        if len(returns) < 2: continue
        
        sharpe = np.mean(returns)/np.std(returns)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = dict(zip(grid.keys(), params))
    
    return best_params

class Allocator():
    def __init__(self, train_data, alpha=0.96, lambda_=3, 
                vol_target=0.20, max_weight=0.7):
        self.train_data = train_data.copy()
        self.n_assets = train_data.shape[1]
        self.last_weights = np.ones(self.n_assets) / self.n_assets
        self.alpha = alpha              # More reactive EMA
        self.lambda_ = lambda_          # Balanced risk-return
        self.vol_target = vol_target    # Higher risk tolerance
        self.max_weight = max_weight    # Allow concentration
        self.transaction_cost = 0.0005  # Lower transaction friction
        
        # Initialize with expanded lookback
        self.price_history = train_data.values[-100:].tolist()  # 100-period lookback
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        prices = np.array(self.price_history)
        returns = np.diff(np.log(prices), axis=0)
        
        # Hybrid covariance estimation
        self.ema_returns = np.mean(returns, axis=0)
        self.ema_cov = LedoitWolf().fit(returns).covariance_
        
        # Momentum signals (3M and 6M)
        if len(prices) > 60:
            mom3 = np.log(prices[-60]) - np.log(prices[-90])
            mom6 = np.log(prices[-120]) - np.log(prices[-180]) if len(prices) > 180 else mom3
            self.momentum = 0.7*mom3 + 0.3*mom6
        else:
            self.momentum = np.zeros(self.n_assets)

    def _regularize_covariance(self, returns):
        """Improved covariance regularization with dynamic mixing"""
        # Dynamic correlation shrinkage
        D = np.diag(np.sqrt(np.diag(self.ema_cov)))
        corr = np.corrcoef(returns.T)
        corr = 0.5*corr + 0.5*np.eye(self.n_assets)  # Stronger shrinkage
        
        # Volatility scaling
        vols = np.sqrt(np.diag(self.ema_cov))
        vol_adj = vols * np.sqrt(252) / self.vol_target
        return (D @ corr @ D) * np.outer(vol_adj, vol_adj)

    def allocate_portfolio(self, asset_prices):
        # Update price history and calculate returns
        self.price_history.append(asset_prices)
        if len(self.price_history) > 200:  # Rolling window
            self.price_history.pop(0)
        
        # Update parameters with momentum
        returns = np.diff(np.log(np.array(self.price_history)), axis=0)
        self.ema_returns = self.alpha * self.ema_returns + (1-self.alpha)*returns[-1]
        self.ema_cov = self._regularize_covariance(returns)
        
        # Combine EMA returns with momentum
        combined_returns = 0.7*self.ema_returns + 0.3*self.momentum
        mu = combined_returns * 252  # Annualized
        
        # Volatility scaling
        Sigma = self.ema_cov
        
        # Optimization setup
        def objective(w):
            risk = w @ Sigma @ w
            return_penalty = -mu @ w
            cost = self.transaction_cost * np.sum(np.abs(w - self.last_weights))
            return risk + self.lambda_*return_penalty + cost
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda w: self.max_weight - np.max(np.abs(w))},
            {'type': 'ineq', 'fun': lambda w: 2.0 - np.sum(np.abs(w))}  # Allow moderate leverage
        ]
        
        # Solve with improved initial weights (momentum-weighted)
        init_weights = np.exp(mu) / np.sum(np.exp(mu))
        result = minimize(objective, init_weights,
                          method='SLSQP',
                          bounds=[(-1.0, 1.0)]*self.n_assets,
                          constraints=constraints,
                          options={'maxiter': 2000})
        
        if result.success:
            new_weights = result.x
            new_weights /= np.sum(np.abs(new_weights))  # Leverage control
            new_weights = np.clip(new_weights, -self.max_weight, self.max_weight)
            self.last_weights = 0.8*new_weights + 0.2*self.last_weights  # Smoother transition
        
        return self.last_weights.copy()

# Backtest with proper walk-forward validation
TRAIN, TEST = train_test_split(data, test_size=0.2, shuffle=False)
best_params = tune_parameters(TRAIN)

# Final allocation with best parameters
alloc = Allocator(TRAIN, **best_params)
capital = [1.0]
weights = []

for i in range(len(TEST)):
    prices = TEST.iloc[i].values
    weights.append(alloc.allocate_portfolio(prices))
    
    if i == 0:
        continue
        
    prev_prices = TEST.iloc[i-1].values
    shares = capital[-1] * weights[i-1] / prev_prices
    capital.append(shares @ prices)

# Calculate performance metrics
returns = np.diff(capital)/capital[:-1]
sharpe_ratio = np.sqrt(252) * np.mean(returns)/np.std(returns)
sortino_ratio = np.sqrt(252) * np.mean(returns)/np.std(returns[returns < 0])
max_drawdown = (np.maximum.accumulate(capital) - capital).max()

print(f"Annualized Sharpe: {sharpe_ratio:.2f}")
print(f"Sortino Ratio: {sortino_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown*100:.1f}%")
