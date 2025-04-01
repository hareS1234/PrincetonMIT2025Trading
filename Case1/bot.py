
from utcxchangelib import xchange_client
import asyncio
import numpy as np
import csv
import time
from collections import defaultdict, deque
from sklearn.linear_model import LinearRegression
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.stats import lognorm

MAX_ORDER_SIZE = 40
MAX_POSITION = 100
ETF_COMPONENTS = {"AKAV": ["APT", "DLR", "MKJ"]}
ORDER_TIMEOUT = 3  # seconds
order_lock = asyncio.Lock()

class BayesianDLR:
    def __init__(self):
        self.mu = 2.5
        self.sigma = 0.5

    def update(self, new_signatures):
        if not new_signatures:
            return
        clean_sigs = [max(0.1, x) for x in new_signatures[-5:]]
        log_sigs = np.log(clean_sigs)
        self.mu = 0.5 * self.mu + 0.5 * np.mean(log_sigs)
        self.sigma = 0.5 * self.sigma + 0.5 * np.std(log_sigs)

    def expected_signatures(self, days_left):
        return np.exp(self.mu + 0.5 * self.sigma**2) * days_left * 5

class Case1Client(xchange_client.XChangeClient):
    def __init__(self, host, username, password):
        super().__init__(host, username, password)
        self.fv = {'APT': 100.0, 'DLR': 50.0, 'MKJ': 50.0, 'AKAV': 200.0, 'AKIM': 100.0}
        self.positions = {k: 0 for k in self.fv}
        self.fade = {k: 0 for k in self.fv}
        self.pnl_window = []
        self.trade_log = []
        self.fv_log = defaultdict(list)
        self.open_order_timestamps = {}
        self.earnings_PE = None
        self.last_news_time = 0
        self.total_signatures = 0
        self.signature_history = []
        self.dlr_bayesian = BayesianDLR()
        self.mkj_news_log = deque(maxlen=10)
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.previous_akav = 200.0
        self.day_last_updated = -1

    async def start(self):
        asyncio.create_task(self.market_maker())
        asyncio.create_task(self.apt_news_trader())
        asyncio.create_task(self.dlr_petition_tracker())
        asyncio.create_task(self.mkj_sentiment_analyzer())
        asyncio.create_task(self.etf_arb_akav())
        asyncio.create_task(self.handle_akim_rebalance())
        asyncio.create_task(self.trade_akim())
        asyncio.create_task(self.cancel_stale_orders())
        asyncio.create_task(self.log_writer())
        asyncio.create_task(self.monitor_positions())
        asyncio.create_task(self.clear_position_periodically())
        asyncio.create_task(self.momentum_trader())
        await self.connect()

    async def place_safe_order(self, symbol, qty, side, price):
        async with order_lock:
            new_position = self.positions[symbol] + (qty if side == xchange_client.Side.BUY else -qty)
            if abs(new_position) > MAX_POSITION:
                return False
            try:
                order_id = await self.place_order(symbol, qty, side, price)
            except Exception as e:
                print(f"Order placement failed for {symbol}: {e}")
                return False
            if not order_id:
                return False
            self.open_order_timestamps[order_id] = time.time()
            return True

    async def market_maker(self):
        while True:
            for symbol in self.fv:
                if symbol == 'AKIM':
                    continue
                book = self.order_books[symbol]
                if book.bids and book.asks:
                    observed_spread = min(book.asks.keys()) - max(book.bids.keys())
                    volatility = self.get_volatility(symbol)
                else:
                    observed_spread = 1.0
                    volatility = 1.0
                fair = self.fv[symbol] - self.fade[symbol]
                inventory_adjustment = 0.15 * np.tanh(self.positions[symbol] / MAX_POSITION)
                spread = max(0.1, 0.5 * observed_spread) + 0.1 * abs(self.fade[symbol]) + 0.2 * volatility
                qty = min(MAX_ORDER_SIZE, 10)
                await self.place_safe_order(symbol, qty, xchange_client.Side.BUY, fair - spread - inventory_adjustment * 2)
                await self.place_safe_order(symbol, qty, xchange_client.Side.SELL, fair + spread - inventory_adjustment * 2)
            await asyncio.sleep(1)

    def get_volatility(self, symbol, window=10):
        prices = [fv for _, fv in self.fv_log[symbol][-window:]]
        return np.std(prices) if len(prices) >= window else 1.0

    async def apt_news_trader(self):
        while True:
            await asyncio.sleep(0.2)
            current_time = self.exchange_time().total_seconds()
            if self.earnings_PE and current_time - self.last_news_time < 3:
                fair = self.fv['APT']
                await self.place_safe_order("APT", 10, xchange_client.Side.BUY, fair - 0.2)
                await self.place_safe_order("APT", 10, xchange_client.Side.SELL, fair + 0.2)

    async def dlr_petition_tracker(self):
        while True:
            await asyncio.sleep(1.5)
            if len(self.signature_history) < 3:
                continue
            day = int(self.exchange_time().total_seconds() // 90)
            days_left = 10 - day
            self.dlr_bayesian.update(self.signature_history)
            expected = min(self.total_signatures + self.dlr_bayesian.expected_signatures(days_left), 100000)
            prob = min(max(expected / 100000, 0), 1)
            self.fv['DLR'] = 100 * prob

    async def mkj_sentiment_analyzer(self):
        while True:
            await asyncio.sleep(2)
            if self.mkj_news_log:
                weights = np.exp(np.linspace(-1, 0, len(self.mkj_news_log)))
                weights /= weights.sum()
                avg_score = np.dot(list(self.mkj_news_log), weights)
                volatility = self.get_volatility('MKJ')
                self.fv['MKJ'] = max(1.0, self.fv['MKJ'] + volatility * avg_score)

    async def etf_arb_akav(self):
        while True:
            await asyncio.sleep(1)
            comp_sum = sum([self.fv[s] for s in ETF_COMPONENTS['AKAV']])
            premium = self.fv['AKAV'] - comp_sum
            fee = 0.5
            qty = min(MAX_ORDER_SIZE, 10)
            if premium > fee:
                if all(abs(self.positions[s] + qty) <= MAX_POSITION for s in ETF_COMPONENTS['AKAV']):
                    await self.place_safe_order("AKAV", qty, xchange_client.Side.SELL, self.fv['AKAV'])
                    await asyncio.gather(*[self.place_safe_order(s, qty, xchange_client.Side.BUY, self.fv[s]) for s in ETF_COMPONENTS['AKAV']])
            elif premium < -fee:
                if all(abs(self.positions[s] - qty) <= MAX_POSITION for s in ETF_COMPONENTS['AKAV']):
                    await self.place_safe_order("AKAV", qty, xchange_client.Side.BUY, self.fv['AKAV'])
                    await asyncio.gather(*[self.place_safe_order(s, qty, xchange_client.Side.SELL, self.fv[s]) for s in ETF_COMPONENTS['AKAV']])

    async def handle_akim_rebalance(self):
        while True:
            seconds = self.exchange_time().total_seconds()
            remainder = max(0, 90 - (seconds % 90))
            await asyncio.sleep(remainder + 0.05)
            current_day = int(self.exchange_time().total_seconds() // 90)
            if current_day != self.day_last_updated:
                akav_return = (self.fv['AKAV'] / self.previous_akav) - 1
                self.fv['AKIM'] = max(1.0, self.fv['AKIM'] * (1 - akav_return))
                self.previous_akav = self.fv['AKAV']
                self.day_last_updated = current_day

    async def trade_akim(self):
        while True:
            await asyncio.sleep(1)
            akim_fair = self.fv['AKIM']
            spread = min(1.0, 0.2 * self.get_volatility('AKAV') + 0.1)
            await self.place_safe_order("AKIM", 10, xchange_client.Side.BUY, akim_fair - spread)
            await self.place_safe_order("AKIM", 10, xchange_client.Side.SELL, akim_fair + spread)

    async def monitor_positions(self):
        while True:
            await asyncio.sleep(3)
            for sym in self.positions:
                pos = self.positions[sym]
                self.fade[sym] = 0.5 * np.sign(pos) if abs(pos) > 50 else 0

    async def clear_position_periodically(self):
        while True:
            await asyncio.sleep(15)
            for sym, pos in self.positions.items():
                if abs(pos) < 20:
                    continue
                book = self.order_books[sym]
                if not book.bids or not book.asks:
                    continue
                price = max(book.bids.keys()) if pos > 0 else min(book.asks.keys())
                qty = min(abs(pos), book.bids[max(book.bids)] if pos > 0 else book.asks[min(book.asks)])
                await self.place_safe_order(sym, qty, xchange_client.Side.SELL if pos > 0 else xchange_client.Side.BUY, price)

    async def log_writer(self):
        while True:
            await asyncio.sleep(10)
            with open("trades.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "asset", "qty", "price", "side"])
                writer.writerows(self.trade_log)
            pnl = sum(self.pnl_window[-20:])
            std = np.std(self.pnl_window[-20:]) or 1
            print(f"[PnL] Last 20 trades: ${pnl:.2f}, Sharpe: {pnl/std:.2f}")

    async def momentum_trader(self):
        while True:
            for sym in ['APT', 'DLR', 'MKJ']:
                if len(self.fv_log[sym]) >= 5:
                    recent_prices = [fv for _, fv in self.fv_log[sym][-5:]]
                    prices = np.array(recent_prices)
                    smoothed = np.convolve(prices, np.ones(3)/3, mode='valid')
                    slope = np.polyfit(range(len(smoothed)), smoothed, 1)[0]
                    book = self.order_books[sym]
                    if book.bids and book.asks:
                        best_bid = max(book.bids.keys())
                        best_ask = min(book.asks.keys())
                        if slope > 0.1:
                            await self.place_safe_order(sym, 10, xchange_client.Side.BUY, best_ask)
                        elif slope < -0.1:
                            await self.place_safe_order(sym, 10, xchange_client.Side.SELL, best_bid)
            await asyncio.sleep(2)

    async def bot_handle_order_fill(self, order_id, qty, price):
        order = self.open_orders[order_id]
        sym = order[0].symbol
        side = order[0].side
        self.positions[sym] += qty if side == xchange_client.Side.BUY else -qty
        self.pnl_window.append((-1 if side == xchange_client.Side.BUY else 1) * qty * price)
        self.trade_log.append((time.time(), sym, qty, price, side.name))
        if not hasattr(order[0], 'qty_filled') or order[0].qty_filled == order[0].qty:
            self.open_order_timestamps.pop(order_id, None)

    async def bot_handle_book_update(self, symbol):
        book = self.order_books[symbol]
        if book.bids and book.asks:
            best_bid = max(book.bids.keys())
            best_ask = min(book.asks.keys())
            mid = (best_bid + best_ask) / 2
            self.fv[symbol] = mid
            self.fv_log[symbol].append((self.exchange_time().total_seconds(), mid))

    async def bot_handle_generic_msg(self, msg):
        text = msg.lower()
        if "apt earnings released" in text:
            val = float(text.split("$")[-1])
            if not self.earnings_PE:
                self.earnings_PE = self.fv['APT'] / val
            self.fv['APT'] = self.earnings_PE * val
            self.last_news_time = self.exchange_time().total_seconds()
        elif "signatures" in text:
            num = int(text.split()[2])
            self.total_signatures += num
            self.signature_history.append(num)
        elif "mkj" in text:
            score = self.sentiment_analyzer.polarity_scores(msg)['compound']
            if abs(score) > 0.7:
                score *= 2.0
            elif abs(score) < 0.3:
                return
            self.mkj_news_log.append(score)

async def main():
    SERVER = "dayof.uchicagotradingcompetition.com:3333"
    client = Case1Client(SERVER, "princeton", "nidoqueen-chansey-9131")
    await client.start()

if __name__ == "__main__":
    asyncio.run(main())