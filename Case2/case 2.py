import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf, OAS
from scipy.spatial.distance import squareform  

# Load data
data = pd.read_csv('Case2.csv', index_col=0)
TRAIN, TEST = train_test_split(data, test_size=0.2, shuffle=False)

# Global best parameters
BEST_ALPHA = 0.85  # Will be tuned
BEST_LAMBDA = 12   # Risk aversion parameter

class Allocator():
    def __init__(self, train_data, alpha=BEST_ALPHA, lambda_=BEST_LAMBDA):
        self.train_data = train_data.copy()
        self.n_assets = train_data.shape[1]
        self.last_weights = np.ones(self.n_assets) / self.n_assets
        self.alpha = alpha
        self.lambda_ = lambda_
        self.risk_free_rate = 0.0
        
        # Initialize EMA parameters with robust covariance estimation
        train_prices = train_data.values
        self.ema_returns, self.ema_cov = self._initialize_parameters(train_prices)
        
        # Outlier detection parameters
        self.price_history = train_data.values[-50:].tolist()  # 50-period lookback
        self.mad = np.median(np.abs(train_data - np.median(train_data, axis=0)), axis=0)
        
        # Regularization parameters
        self.cov_mix = 0.7  # Blend EMA with Ledoit-Wolf
        self.vol_target = 0.15  # Annualized volatility target
        
    def _initialize_parameters(self, prices):
        """Initialize with shrunk covariance matrix"""
        returns = np.diff(np.log(prices), axis=0)
        if len(returns) < 2:
            return np.zeros(self.n_assets), np.eye(self.n_assets)
        
        # Initial estimates using multiple covariance estimators
        lw = LedoitWolf().fit(returns)
        oas = OAS().fit(returns)
        ema_returns = np.mean(returns, axis=0)
        ema_cov = 0.5*lw.covariance_ + 0.5*oas.covariance_
        
        return ema_returns, ema_cov
    
    def _handle_outliers(self, prices):
        """Robust outlier detection using MAD"""
        if len(self.price_history) < 10:
            return prices
        
        median_prices = np.median(self.price_history, axis=0)
        mad = np.median(np.abs(self.price_history - median_prices), axis=0)
        scaled_diff = np.abs((prices - median_prices) / (mad + 1e-8))
        return np.where(scaled_diff > 3, median_prices, prices)
    
    def _regularize_covariance(self, returns):
        """Advanced covariance regularization"""
        # 1. Ledoit-Wolf shrinkage
        lw = LedoitWolf().fit(returns)
        
        # 2. Correlation matrix regularization
        D = np.diag(np.sqrt(np.diag(self.ema_cov)))
        corr = np.linalg.inv(D) @ self.ema_cov @ np.linalg.inv(D)
        np.fill_diagonal(corr, 1.0)
        corr = 0.9*corr + 0.1*np.eye(self.n_assets)  # Shrink correlations
        
        # 3. Blend covariance matrices
        regularized_cov = self.cov_mix*(D @ corr @ D) + (1-self.cov_mix)*lw.covariance_
        return regularized_cov
    
    def allocate_portfolio(self, asset_prices):
        # Handle outliers and update price history
        prices = self._handle_outliers(np.array(asset_prices))
        self.price_history.append(prices)
        
        # Update EMA parameters
        if len(self.price_history) > 1:
            latest_returns = np.log(prices) - np.log(self.price_history[-2])
            
            # EMA update for returns
            self.ema_returns = self.alpha * self.ema_returns + (1 - self.alpha) * latest_returns
            
            # Update covariance matrix with regularization
            returns_matrix = np.array([np.log(p) - np.log(prev_p) 
                                      for p, prev_p in zip(self.price_history[1:], self.price_history[:-1])])
            self.ema_cov = self._regularize_covariance(returns_matrix)
        
        # Volatility targeting
        annualized_vol = np.sqrt(np.diag(self.ema_cov) * np.sqrt(252))
        vol_scaling = self.vol_target / np.mean(annualized_vol)
        Sigma = self.ema_cov * vol_scaling**2
        
        # Hierarchical Risk Parity weights as starting point
        hrp_weights = self._hrp_portfolio(Sigma)
        
        # Mean-variance optimization with transaction cost penalty
        mu = self.ema_returns * 252  # Annualized returns
        
        def objective(w):
            risk = w @ Sigma @ w
            return_penalty = -mu @ w
            transaction_cost = 0.002 * np.sum(np.abs(w - self.last_weights))
            return risk + self.lambda_*return_penalty + transaction_cost
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda w: 0.5 - np.max(np.abs(w))},  # Concentration limit
            {'type': 'ineq', 'fun': lambda w: 1.0 - np.sum(np.abs(w))}   # Leverage limit
        ]
        
        result = minimize(objective, hrp_weights,
                          method='SLSQP',
                          bounds=[(-0.5, 0.5)]*self.n_assets,  # Shorting limit
                          constraints=constraints,
                          options={'maxiter': 1000})
        
        if result.success:
            new_weights = result.x
            new_weights /= np.sum(np.abs(new_weights))  # Leverage normalization
            new_weights = np.clip(new_weights, -0.5, 0.5)
            self.last_weights = 0.7*new_weights + 0.3*self.last_weights  # Smoothing
                
        return self.last_weights.copy()
        from scipy.spatial.distance import squareform


    def _hrp_portfolio(self, cov):
        """Hierarchical Risk Parity portfolio"""
        corr = cov.copy()
        np.fill_diagonal(corr, 1)
        dist = np.sqrt(0.5 * (1 - corr))
        np.fill_diagonal(dist, 0)

        # Convert to condensed distance form
        dist_condensed = squareform(dist, checks=False)

        # Hierarchical clustering
        from scipy.cluster.hierarchy import linkage, leaves_list
        Z = linkage(dist_condensed, 'single')
        ordering = leaves_list(Z)

        # Recursive bisection
        weights = np.ones_like(ordering, dtype=float)
        clusters = [ordering]

        while len(clusters) > 0:
            cluster = clusters.pop(0)
            if len(cluster) == 1:
                continue

            left, right = self._bisect(cluster, dist)
            left_var = self._cluster_var(left, cov)
            right_var = self._cluster_var(right, cov)
            alpha = 1 - left_var / (left_var + right_var)

            weights[left] *= alpha
            weights[right] *= (1 - alpha)
            clusters += [left, right]

        return weights / np.sum(weights)

    def _bisect(self, cluster, dist):
        # Helper for HRP
        matrix = dist[np.ix_(cluster, cluster)]
        return cluster[:len(cluster)//2], cluster[len(cluster)//2:]

    def _cluster_var(self, cluster, cov):
        # Helper for HRP
        w = np.ones(len(cluster))/len(cluster)
        return w @ cov[np.ix_(cluster, cluster)] @ w

def tune_parameters(train, test):
    # Parameter tuning using grid search
    best_sharpe = -np.inf
    best_params = {}
    
    for alpha in [0, 1, 0.02]:
        for lambda_ in [10, 12, 15]:
            alloc = Allocator(train, alpha=alpha, lambda_=lambda_)
            capital = [1.0]
            
            for i in range(len(test)):
                prices = test.iloc[i].values
                w = alloc.allocate_portfolio(prices)
                
                if i > 0:
                    prev_prices = test.iloc[i-1].values
                    shares = capital[-1] * w / prev_prices
                    capital.append(shares @ prices)
            
            returns = np.diff(capital)/capital[:-1]
            sharpe = np.mean(returns)/np.std(returns) if np.std(returns) > 0 else 0
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = {'alpha': alpha, 'lambda_': lambda_}
    
    return best_params



# Tune parameters
best_params = tune_parameters(TRAIN, TEST)
BEST_ALPHA = best_params['alpha']
BEST_LAMBDA = best_params['lambda_']

# Final allocation
alloc = Allocator(TRAIN, alpha=BEST_ALPHA, lambda_=BEST_LAMBDA)
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

# Calculate Sharpe ratio
returns = np.diff(capital)/capital[:-1]
annualized_sharpe = np.sqrt(252) * np.mean(returns)/np.std(returns) if np.std(returns) > 0 else 0
print(f"Annualized Sharpe Ratio: {annualized_sharpe:.2f}")

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(capital, label='Portfolio Value')
plt.title(f"Portfolio Performance (Sharpe: {annualized_sharpe:.2f})")
plt.xlabel("Days")
plt.ylabel("Capital")
plt.grid(True)
plt.legend()
plt.show()
