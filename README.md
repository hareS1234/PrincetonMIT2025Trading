**Dynamic Portfolio Allocation Algorithm**

**Introduction**

In this solution, we develop a dynamic portfolio allocation algorithm that uses both end-of-day and intraday price data to adjust a portfolio of 6 stocks daily. The goal is to maximize the Sharpe ratio over a 130-day out-of-sample test period. We use an equal-weight portfolio (each asset at 1/6 weight) as a baseline for comparison and allow the strategy to take long, short, or zero positions in each asset (weights ∈ [-1, 1]). The algorithm rebalances once per day at the closing price (tick 29) of each trading day, using intraday data (30 ticks per day) to inform risk-adjusted allocation decisions.

**Methodology**

1. Data Handling
We first split the historical price data (Case2.csv) into a training set and a 130-day test set (the last 130 days). Each day consists of 30 intraday price ticks for 6 assets. We extract daily closing prices (the price at tick 29) for each asset and use these to compute daily returns. Intraday price series for each day are used to estimate realized volatility within the day. This intraday volatility provides additional risk information beyond what daily returns capture.

2. Baseline and Rolling Window
The algorithm starts with an equal-weight baseline (1/6 per asset) on the first test day. Thereafter, it uses a rolling window of recent data (e.g., last 60–90 days) to dynamically update portfolio weights. At each day’s close, we update our history with that day’s prices and volatility, and then compute new weights for the next day. This rolling approach lets the model adapt to changing trends and volatilities in the market.

3. Risk-Return Optimization
For each day’s rebalancing, we apply a mean-variance optimization to maximize expected return for a given risk level, aiming to improve the Sharpe ratio. We estimate each asset’s expected return as the mean of its daily returns over the recent window, and estimate risk via the covariance matrix of those returns. To incorporate intraday information, we augment the covariance matrix’s diagonal with each asset’s realized intraday variance (average squared intraday return) over the window. This effectively increases the perceived risk of assets that experience high intraday volatility, even if their daily closing prices don’t reflect large changes. Research shows that volatility-targeting and scaling can improve portfolio Sharpe ratios by adjusting exposure based on risk.

4. Dynamic Weight Adjustment
Given the expected return vector μ and adjusted covariance matrix Σ for the window, we compute the tangency portfolio weights proportional to Σ⁻¹μ. These weights maximize the return-to-risk ratio (Sharpe) for that window’s estimates. We then scale this weight vector so that no asset’s weight exceeds 1 or –1 (respecting the allowed bounds). This scaling also controls leverage, ensuring the portfolio isn’t over-leveraged. The resulting weight vector is used for the next trading day’s positions. If the model’s outlook is close to the baseline (e.g., all assets have similar risk-adjusted returns), the weights will remain near the equal-weight distribution. If certain assets have significantly higher expected return-to-risk ratios, the algorithm will overweight those (up to the limits), and underweight or short assets with poor outlook.

5. Daily Rebalancing Workflow
During the trading day, the algorithm holds the weights decided at the previous close. We do not trade intraday – weights stay constant throughout each day’s 30 ticks. At the end of day (tick 29), the algorithm records that day’s price path, updates the statistical estimates (returns and volatilities), and computes new weights for the next day. This ensures that intraday information (like volatility spikes or trends) influences the next day’s allocation, but we avoid excessive turnover by only rebalancing once daily as required.

