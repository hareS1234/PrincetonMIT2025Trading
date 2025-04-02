# Portfolio Optimization 

## Overview

This project implements a daily portfolio allocation strategy that aims to maximize the Sharpe ratio using historical intraday price data for a set of assets.

The allocator rebalances the portfolio once per day based on the last tick of that day's prices. It is designed to be simple, interpretable, and robust to noise in return and risk estimation.

## Strategy Summary

- **Data Granularity**: 30 intraday ticks per day per asset. The allocator uses only the final tick of each day to simulate daily rebalancing.
- **Return Forecasting**: Combines an exponentially-weighted moving average (EWMA) of past returns with the long-term historical average.
- **Risk Estimation**: Uses the Ledoit-Wolf shrinkage estimator to stabilize the sample covariance matrix of asset returns.
- **Weight Optimization**: Optimizes for maximum Sharpe ratio, subject to:
  - Weight sum-to-one constraint
  - Volatility constraint (target daily volatility derived from 15% annualized)
  - Weight bounds of [-1, 1] per asset
- **Turnover Control**: Limits portfolio turnover to a maximum of 5% per day to reduce excessive trading and maintain realistic execution.

## Environment

This code is written in Python 3.10 and requires only the following libraries:

- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `matplotlib` (used for visualization only)

The implementation is compliant with environments that limit external dependencies.

## How to Use

1. Load a DataFrame of intraday price data with 30 ticks per day.
2. Split the data into a training and testing set.
3. Instantiate the `Allocator` with the training data.
4. Feed one row of price data at a time to `allocate_portfolio()`.
5. Store the returned weights and simulate performance.

The script includes a built-in grading function that simulates capital evolution and computes the Sharpe ratio on a test set.

## Customization

You can tune the following parameters for experimentation:

- Volatility target (`self.vol_target`)
- EWMA span (`span` in `.ewm(span=...)`)
- Maximum daily turnover
- Proportional weights for EWMA vs. historical average

## Visualization

The script optionally generates plots of:

- Capital over time
- Asset weight trajectories

These help evaluate portfolio stability and rebalancing behavior.

## License

This project is provided for educational and analytical purposes only.
