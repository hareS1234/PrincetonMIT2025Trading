import numpy as np
import pandas as pd
from numpy.linalg import inv

# Load the historical data (39000 rows = 1300 days * 30 ticks per day for 6 assets)
data = pd.read_csv('Case2.csv')  # Each column Asset_1 ... Asset_6 is a price series

# Split into training and test sets by day (130-day test period)
total_days = len(data) // 30
test_days = 130
train_days = total_days - test_days
train_ticks = train_days * 30
TRAIN = data.iloc[:train_ticks].reset_index(drop=True)
TEST  = data.iloc[train_ticks:].reset_index(drop=True)

# Define the dynamic allocation class
class Allocator:
    def __init__(self, train_data: pd.DataFrame):
        """
        Initialize the allocator with training data.
        Stores historical information and pre-computes any static parameters.
        """
        # Store training price history (tick-level) and compute daily close prices
        self.train_data = train_data.copy()
        self.n_assets = train_data.shape[1]
        # Compute daily closing prices from training data (price at tick 29 of each day)
        train_closes = self.train_data.iloc[29::30].reset_index(drop=True)
        # Compute daily realized volatility for training period:
        # realized_vol[d][j] = sqrt(sum of squared log returns within day d for asset j)
        vol_records = []
        for day in range(len(train_closes)):
            day_prices = self.train_data.iloc[day*30 : day*30 + 30]
            # Calculate realized vol for each asset j on this day
            log_returns = np.diff(np.log(day_prices), axis=0)  # 29 intraday returns per asset
            # Sum of squared intraday returns per asset:
            daily_realized_var = (log_returns ** 2).sum(axis=0)
            daily_realized_vol = np.sqrt(daily_realized_var)
            vol_records.append(daily_realized_vol)
        train_realized_vol = pd.DataFrame(vol_records, columns=train_data.columns)
        # Store daily history DataFrames for prices and volatilities
        self.daily_history = train_closes.copy()       # Daily closing price history
        self.daily_vol_history = train_realized_vol.copy()  # Daily intraday vol history
        # Portfolio weights from previous allocation (start with equal-weight baseline)
        self.last_weights = np.ones(self.n_assets) / self.n_assets
        # Intraday tracking
        self.tick_count = 0                        # intraday tick counter for current day
        self.intraday_prices = []                  # list to collect intraday prices for the current day

    def allocate_portfolio(self, asset_prices: np.ndarray) -> np.ndarray:
        """
        Decide portfolio weights for the next day, given current day's prices.
        This method is called at each tick (intraday price). It only changes the 
        allocation at end-of-day (tick 29), using accumulated data.
        
        Parameters:
        asset_prices (np.ndarray): Length-6 array of current prices for the 6 assets.
        
        Returns:
        np.ndarray: Length-6 array of portfolio weights for the next day.
        """
        # Record the current tick's prices
        prices = np.array(asset_prices, dtype=float)
        self.intraday_prices.append(prices)
        
        # If we have not yet reached end-of-day, keep last allocation (no rebalancing intraday)
        if self.tick_count < 29:
            self.tick_count += 1
            return self.last_weights
        
        # If this is the end-of-day tick (tick_count == 29):
        # 1. Update daily price and volatility history with today's data
        # Convert intraday_prices list to array for computations
        prices_matrix = np.vstack(self.intraday_prices)  # shape = (30, n_assets)
        # Compute realized volatility for today (using intraday log returns)
        log_ret_matrix = np.diff(np.log(prices_matrix), axis=0)       # shape = (29, n_assets)
        realized_var_today = np.sum(log_ret_matrix**2, axis=0)        # sum of squared returns per asset
        realized_vol_today = np.sqrt(realized_var_today)              # realized vol per asset for the day
        # Append today's closing prices (prices at tick 29) to daily history
        close_series = pd.Series(prices, index=self.daily_history.columns)
        self.daily_history = pd.concat([self.daily_history, close_series.to_frame().T], ignore_index=True)
        # Append today's realized volatility to vol history
        vol_series = pd.Series(realized_vol_today, index=self.daily_vol_history.columns)
        self.daily_vol_history = pd.concat([self.daily_vol_history, vol_series.to_frame().T], ignore_index=True)
        
        # 2. Compute new portfolio weights using recent window of historical data
        # Choose rolling window size (e.g., 90 days) or use all history if fewer days available
        window = 90
        num_days = len(self.daily_history)
        window_days = min(window, num_days)
        # Get the recent window of daily closing prices and realized vols
        recent_prices = self.daily_history.iloc[-window_days:]
        recent_vols   = self.daily_vol_history.iloc[-window_days:]
        # Compute daily returns over this window
        daily_returns = recent_prices.pct_change().dropna()  # DataFrame of shape (window_days-1, n_assets)
        if daily_returns.empty:
            # Not enough data to compute returns (e.g., first day) – keep last weights
            new_weights = self.last_weights
        else:
            # Calculate expected returns (mean of daily returns) and covariance over the window
            mu = daily_returns.mean().values                    # expected return vector (length 6)
            Sigma = daily_returns.cov().values                  # covariance matrix (6x6)
            # Incorporate intraday risk: add average realized variance of each asset to covariance diagonal
            avg_realized_var = (recent_vols**2).mean().values   # mean of daily realized variances for each asset
            for j in range(self.n_assets):
                Sigma[j, j] += avg_realized_var[j]
            
            # Solve for weights that maximize return per risk (tangency portfolio): w ∝ Σ⁻¹ μ
            try:
                inv_Sigma = inv(Sigma)
            except np.linalg.LinAlgError:
                # In case of singular matrix, add a tiny noise for stability
                inv_Sigma = inv(Sigma + 1e-8 * np.eye(self.n_assets))
            raw_weights = inv_Sigma.dot(mu)  # unconstrained optimal weights (not scaled)
            # Scale weights so that no asset exceeds weight bounds [-1, 1]
            max_w = np.max(np.abs(raw_weights))
            if max_w > 1:
                raw_weights = raw_weights / max_w  # scale down to within [-1,1]
            new_weights = np.clip(raw_weights, -1, 1)  # ensure bound constraints
            
        # 3. Reset intraday tracking for the next day
        self.last_weights = new_weights
        self.tick_count = 0
        self.intraday_prices = []
        return new_weights

# Grading function to test the performance on the test set
def grading(train_data: pd.DataFrame, test_data: pd.DataFrame):
    weights_output = np.zeros((len(test_data), train_data.shape[1]))
    allocator = Allocator(train_data)
    for i in range(len(test_data)):
        # Pass each tick (row) of test data to allocator
        current_prices = test_data.iloc[i, :].values
        weights_output[i, :] = allocator.allocate_portfolio(current_prices)
        # Check weight bounds
        if np.any(weights_output[i, :] < -1) or np.any(weights_output[i, :] > 1):
            raise Exception("Weights Outside of Bounds")
    # Simulate portfolio performance over test period
    capital = [1.0]  # start with capital = 1 (arbitrary unit)
    for i in range(len(test_data) - 1):
        # Use weights at day i to invest for period i->i+1
        price_today = test_data.iloc[i, :].values
        price_next  = test_data.iloc[i+1, :].values
        w = weights_output[i, :]
        # Calculate holdings (number of shares for each asset)
        shares = capital[-1] * w / price_today
        # Calculate capital after rebalancing (cash left after buying/selling shares)
        cash_balance = capital[-1] - np.dot(shares, price_today)
        # Update capital for next day: cash_balance + value of shares at next day's prices
        capital.append(cash_balance + np.dot(shares, price_next))
    capital = np.array(capital)
    returns = (capital[1:] - capital[:-1]) / capital[:-1]  # daily returns of portfolio
    sharpe = np.mean(returns) / (np.std(returns) if np.std(returns) != 0 else 1e-8)
    return sharpe, capital, weights_output

# Evaluate the algorithm on the train/test split
sharpe_ratio, capital_curve, weight_history = grading(TRAIN, TEST)
print(f"Out-of-sample Sharpe Ratio: {sharpe_ratio:.4f}")
