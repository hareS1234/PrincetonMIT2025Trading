import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import OAS
from scipy.spatial.distance import squareform

class SharpeMaximizer():
    def __init__(self, train_data, momentum_period=20, vol_target=0.25):
        self.price_history = train_data.values[-252:].tolist()  # 1-year lookback
        self.n_assets = train_data.shape[1]
        self.vol_target = vol_target
        self.momentum_period = momentum_period
        self.returns = None
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        prices = np.array(self.price_history)
        self.returns = np.diff(np.log(prices), axis=0)
        
        # Initialize covariance with OAS shrinkage
        self.cov = OAS().fit(self.returns).covariance_
        self.chol = np.linalg.cholesky(self.cov)
        
    def _update_parameters(self, new_price):
        # Update price history and returns
        self.price_history.append(new_price)
        if len(self.price_history) > 252:
            self.price_history.pop(0)
            
        # Calculate new returns
        prev_price = self.price_history[-2]
        new_return = np.log(new_price) - np.log(prev_price)
        self.returns = np.vstack([self.returns, new_return])
        
        # Update covariance with exponential weighting
        decay = 0.94
        weights = decay**np.arange(len(self.returns))[::-1]
        weighted_returns = self.returns * weights[:, None]
        self.cov = np.cov(weighted_returns, rowvar=False, bias=True)
        self.chol = np.linalg.cholesky(self.cov)
        
    def _momentum_scores(self):
        # Calculate momentum as exponential weighted returns
        prices = np.array(self.price_history)
        returns = np.diff(np.log(prices), axis=0)
        
        # Short-term momentum (1-2 weeks)
        mom_short = np.mean(returns[-5:], axis=0)
        
        # Medium-term momentum (1 month)
        mom_medium = np.mean(returns[-21:], axis=0)
        
        # Long-term momentum (3 months)
        mom_long = np.mean(returns[-63:], axis=0)
        
        # Combined momentum score
        return 0.5*mom_short + 0.3*mom_medium + 0.2*mom_long
        
    def allocate_portfolio(self, asset_prices):
        self._update_parameters(asset_prices)
        
        # Get momentum scores and normalize
        mu = self._momentum_scores() * 252  # Annualized
        mu = (mu - np.mean(mu)) / np.std(mu)  # Z-score normalization
        
        # De-noise covariance matrix
        cov = self._denoise_covariance()
        
        # Convex optimization problem
        def objective(w):
            port_var = w @ cov @ w
            return -(w @ mu) / np.sqrt(port_var)  # Direct Sharpe maximization
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda w: 1 - np.abs(w).max()}  # Respect weight bounds
        ]
        
        # Smart initialization using momentum-weighted portfolio
        init_weights = np.exp(mu) / np.sum(np.exp(mu))
        init_weights = init_weights / init_weights.sum()
        
        # Solve with improved numerical stability
        result = minimize(objective, init_weights,
                          method='SLSQP',
                          bounds=[(-1.0, 1.0)]*self.n_assets,
                          constraints=constraints,
                          options={'maxiter': 2500, 'ftol': 1e-8})
        
        if result.success:
            weights = result.x
            # Volatility targeting
            current_vol = np.sqrt(weights @ cov @ weights) * np.sqrt(252)
            weights *= (self.vol_target / current_vol)
            return weights / np.sum(weights)  # Ensure sum to 1
        else:
            # Fallback to momentum-weighted portfolio
            return init_weights
            
    def _denoise_covariance(self):
        """Marcenko-Pastur covariance denoising"""
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(self.cov)
        
        # Filter eigenvalues
        n, p = self.returns.shape
        q = p/n
        lambda_max = (1 + np.sqrt(q))**2 * np.median(eigenvalues)
        filtered = np.clip(eigenvalues, 0, lambda_max)
        
        return eigenvectors @ np.diag(filtered) @ eigenvectors.T

# Backtest implementation
data = pd.read_csv('Case2.csv', index_col=0)
TRAIN, TEST = train_test_split(data, test_size=0.2, shuffle=False)

alloc = SharpeMaximizer(TRAIN, vol_target=0.30)  # Higher volatility target
capital = [1.0]
weights = []

for i in range(len(TEST)):
    prices = TEST.iloc[i].values
    if i == 0:
        weights.append(np.ones(len(prices))/len(prices))
        continue
        
    w = alloc.allocate_portfolio(prices)
    weights.append(w)
    
    # Calculate portfolio value
    prev_prices = TEST.iloc[i-1].values
    shares = capital[-1] * weights[i-1] / prev_prices
    capital.append(shares @ prices)

# Calculate performance metrics
returns = np.diff(capital)/capital[:-1]
sharpe = np.sqrt(252) * np.mean(returns)/np.std(returns)
print(f"Annualized Sharpe Ratio: {sharpe:.2f}")


#########
def grading(train_data, test_data): 
    '''
    Grading Script
    '''
    weights = np.full(shape=(len(test_data.index),5), fill_value=0.0)
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

sharpe, capital, weights = grading(TRAIN, TEST)
#Sharpe gets printed to command line
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

