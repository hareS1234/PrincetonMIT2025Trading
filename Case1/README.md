# PrincetonMIT2025Trading
# UChicago Trading Competition 2025 – Princeton_MIT Team
# Case 1 Trading Bot

This repository contains an asynchronous trading bot developed for **Case 1** of the 2025 UChicago Trading Competition. The bot is designed to maximize P&L by trading three stocks and two ETFs in a high-frequency, real-time exchange environment. The team comprises of Clint Song, Jessica Choe, Jessie Wang, and Angelina Quan.

---

## Assets Traded

| Symbol | Description |
|--------|-------------|
| **APT** | Large-cap stock with structured earnings releases |
| **DLR** | Mid-cap stock driven by petition-based binary outcomes |
| **MKJ** | Small-cap stock with unstructured sentiment/news |
| **AKAV** | ETF holding 1 share each of APT, DLR, and MKJ |
| **AKIM** | Inverse ETF tracking daily movement of AKAV |

---

## Strategy Overview

This bot implements the following strategies:

### APT Earnings Trading
- Parses structured earnings announcements
- Calculates fair value using a constant P/E ratio
- Rapidly reacts to price discrepancies before market adjustment

### DLR Petition Probability Model
- Tracks daily signature count updates
- Estimates probability of petition success using cumulative signatures
- Values DLR using expected payout of binary outcome

### MKJ Sentiment Analyzer
- Parses unstructured news events
- Applies keyword-based sentiment scoring
- Dynamically updates fair value and spreads

### AKAV Arbitrage Engine
- Detects mispricings between ETF and constituent stocks
- Executes creation/redeem arbitrage trades based on fair value
- Considers redemption fees and liquidity constraints

### AKIM Volatility Hedge
- Trades AKIM near end-of-day to hedge AKAV exposure
- Accounts for volatility drag and inverse ETF decay
