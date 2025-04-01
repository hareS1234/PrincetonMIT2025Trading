from utc_bot import UTCBot, start_bot
import proto.utc_bot as pb
import asyncio
import betterproto

class Case1Bot(UTCBot):
    def __init__(self):
        super().__init__()
        self.fv = {'APT': 100.0, 'DLR': 50.0, 'MKJ': 50.0, 'AKAV': 200.0, 'AKIM': 100.0}
        self.earnings_PE = None
        self.signatures = []
        self.total_signatures = 0

    async def handle_round_started(self):
        asyncio.create_task(self.market_maker())
        asyncio.create_task(self.apt_news_trader())
        asyncio.create_task(self.dlr_petition_tracker())
        asyncio.create_task(self.mkj_sentiment_analyzer())
        asyncio.create_task(self.etf_arb_engine())
        asyncio.create_task(self.akim_hedge_logic())

    async def handle_exchange_update(self, update: pb.FeedMessage):
        kind, msg = betterproto.which_one_of(update, "msg")

        if kind == "generic_msg":
            if "earnings" in msg.message.lower():
                await self.parse_earnings(msg.message)
            elif "signatures" in msg.message.lower():
                await self.parse_signature_update(msg.message)
            elif "MKJ" in msg.message:
                await self.handle_mkj_news(msg.message)

        elif kind == "market_snapshot_msg":
            for book_name, book in msg.books.items():
                if len(book.bids) and len(book.asks):
                    self.fv[book_name] = (book.bids[0].px + book.asks[0].px) / 2

    async def market_maker(self):
        while True:
            for symbol in self.fv:
                if symbol == 'AKIM':
                    continue
                fair = self.fv[symbol]
                spread = 0.5
                qty = 10
                await self.modify_order(f"{symbol}_bid", symbol, pb.OrderSpecType.LIMIT, pb.OrderSpecSide.BID, qty, fair - spread)
                await self.modify_order(f"{symbol}_ask", symbol, pb.OrderSpecType.LIMIT, pb.OrderSpecSide.ASK, qty, fair + spread)
            await asyncio.sleep(1)

    async def apt_news_trader(self):
        pass  # TODO: implement earnings-based price prediction and fast execution

    async def dlr_petition_tracker(self):
        pass  # TODO: model lognormal distribution and update fair value

    async def mkj_sentiment_analyzer(self):
        pass  # TODO: use keyword-based sentiment to adjust MKJ fair value

    async def etf_arb_engine(self):
        while True:
            akav_fv = self.fv['APT'] + self.fv['DLR'] + self.fv['MKJ']
            if abs(self.fv['AKAV'] - akav_fv) > 1:
                # TODO: place arbitrage trades depending on mispricing
                pass
            await asyncio.sleep(0.5)

    async def akim_hedge_logic(self):
        pass  # TODO: at end of day, assess AKAV change and trade AKIM accordingly

    async def parse_earnings(self, message):
        # Example message: "APT earnings released: $2.5"
        earnings = float(message.split("$")[-1])
        if self.earnings_PE is None:
            self.earnings_PE = self.fv['APT'] / earnings
        self.fv['APT'] = self.earnings_PE * earnings

    async def parse_signature_update(self, message):
        # Example message: "DLR received 1200 signatures"
        new_signatures = int(message.split()[2])
        self.total_signatures += new_signatures
        # TODO: use Bayesian update to revalue DLR

    async def handle_mkj_news(self, message):
        # TODO: sentiment analysis
        if any(word in message.lower() for word in ["lawsuit", "scandal"]):
            self.fv['MKJ'] -= 2
        elif any(word in message.lower() for word in ["acquisition", "expansion"]):
            self.fv['MKJ'] += 2

if __name__ == "__main__":
    start_bot(Case1Bot)