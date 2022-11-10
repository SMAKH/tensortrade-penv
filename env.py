from collections import deque
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensortrade.env.generic as generic
from gym.spaces import Space, Box
from tensortrade.env.default import TradingEnv
from tensortrade.feed import Stream, DataFeed

from features import fracdiff, macd, rsi
from metrics import maximum_drawdown, sharpe
from price_generators import StockPriceFromFile, ComputationalSignalManager

plt.style.use("seaborn")


class Simplex(Space):

    def __init__(self, m: int):
        assert m >= 1
        super().__init__(shape=(m + 1,), dtype=np.float32)
        self.m = m

    def sample(self) -> np.array:
        return np.random.dirichlet(alpha=(1 + self.m) * [1 + 3 * np.random.random()])

    def contains(self, x) -> bool:
        if len(x) != self.m + 1:
            return False
        if not np.testing.assert_almost_equal(sum(x), 1.0):
            return False
        return True


class Observer(generic.Observer):

    def __init__(self,
                 feed: DataFeed,
                 window_size,
                 number_of_assets,
                 num_assets,
                 min_periods,
                 **kwargs) -> None:
        super().__init__()
        self.feed = feed
        self.feed.compile()
        self.number_of_assets = number_of_assets
        self.window_size = window_size
        self.keys = None

        self.history = deque([], maxlen=self.window_size)
        self.state = None
        self.renderer_history = []
        self.min_periods = kwargs.get("min_periods", window_size)
        assert self.min_periods >= self.window_size

    @property
    def observation_space(self) -> Space:
        obs = self.feed.next()["obs"]
        self.keys = [k for k in obs]
        space = Box(-np.inf, np.inf, shape=(self.window_size, 174, self.number_of_assets), dtype=np.float32)
        self.feed.reset()
        return space

    def warmup(self) -> None:
        for _ in range(self.min_periods):
            if self.feed.has_next():
                obs = self.feed.next()["obs"]
                prices = obs["price_stream"]
                high_prices = obs["h_price"]
                low_prices = obs["l_price"]
                volumes = obs["volume"]
                signals = obs["signal_stream"]
                inputs = [prices, high_prices, low_prices, volumes]
                for i in range(85):
                    for j in range(2):
                        inputs.append(signals[i, j, :])
                obs = inputs
                self.history += [obs]

    def observe(self, env) -> np.array:
        data = self.feed.next()

        self.state = data["state"]

        if "renderer" in data.keys():
            self.renderer_history += [data["renderer"]]

        obs = data["obs"]
        lag = np.array(list(obs["lag_stream"]))
        prices = np.array(list(obs["price_stream"]))
        high_prices = np.array(list(obs["h_price"]))
        low_prices = np.array(list(obs["l_price"]))
        volumes = obs["volume"]
        signals = obs["signal_stream"]
        inputs = [prices / lag, high_prices / lag, low_prices / lag, volumes]
        for i in range(85):
            for j in range(2):
                inputs.append(signals[i, j, :])
        # obs=np.vstack(inputs)
        ##self.history += [[obs[k] for k in self.keys]]
        obs = inputs
        self.history += [obs]
        # print(np.array(self.history).shape)
        # print("......................nnnn")
        # return np.array(self.history,dtype=np.float32)
        return np.array(self.history, dtype=np.float32)

    def reset(self) -> None:
        self.renderer_history = []
        self.history = deque([], maxlen=self.window_size)
        self.feed.reset()
        self.warmup()


class PortfolioAllocation(generic.ActionScheme):

    def __init__(self, num_assets: int, commission: float, initial_cash=1):
        super().__init__()
        self.num_assets = num_assets
        self.commission = commission
        self.initial_cash = initial_cash
        self.max_cache = initial_cash
        self.p = self.initial_cash
        self.mu = None
        self.weights = np.array([1] + self.num_assets * [0])
        self.buffer = deque([self.p], maxlen=2)

    @property
    def action_space(self):
        return Simplex(m=self.num_assets)

    def update_mu(self, w, w_p, y) -> float:
        c = self.commission

        if self.mu is None:
            self.mu = c * abs(w_p[1:] - w[1:]).sum()

        ts = (w_p[1:] - self.mu * w[1:]).clip(min=0).sum()
        num = (1 - c * w_p[0]) - c * (2 - c) * ts
        den = 1 - c * w[0]

        self.mu = num / den

    def perform(self, env, action):
        np.testing.assert_almost_equal(action.sum(), 1.0, decimal=5)
        state = env.observer.state
        # y(t)= price relative vector of tth trading period
        # y(t+1)= percentage of price change relative to last step for each
        ##y1 = state["y(t+1)"]
        p = np.array(list(state["price_stream"]))
        l = np.array(list(state["lag_stream"]))
        y1 = p / l
        y1 = np.array([1] + list(y1))  # 1 is for cach that its price is constant. for example dollar
        w = (y1 * self.weights) / np.dot(y1, self.weights)
        w_p = action.copy()

        self.update_mu(w, w_p, y1)

        self.p *= self.mu * np.dot(y1, w_p)
        if self.p >= self.max_cache:
            self.max_cache = self.p
        self.weights = w_p
        self.buffer += [self.p]

    def reset(self):
        self.p = self.initial_cash
        self.max_cache = self.initial_cash
        self.weights = np.array([1] + self.num_assets * [0])
        self.buffer = deque([self.p], maxlen=2)


class Profit(generic.RewardScheme):

    def reward(self, env):
        buffer = env.action_scheme.buffer
        return buffer[1] - buffer[0]


class MaxStepsOrNoMoney(generic.Stopper):
    def __init__(self, min_profit: float, max_drawdown: float) -> None:
        super().__init__()
        self.min_profit = min_profit
        self.max_drawdown = max_drawdown

    def stop(self, env) -> bool:
        '''p = env.action_scheme.p
        return (p == 0) or not env.observer.feed.has_next()'''
        p = env.action_scheme.p
        if (p == 0) or not env.observer.feed.has_next():
            return True

        max_cach = env.action_scheme.max_cache
        initial_cache = env.action_scheme.initial_cash
        if ((p - initial_cache) / initial_cache) <= self.min_profit:
            return True
        if ((p - max_cach) / max_cach) <= self.max_drawdown:
            return True
        return False


class RebalanceInformer(generic.Informer):

    def info(self, env) -> dict:
        return {
            "step": self.clock.step,
        }


class ReallocationChart(generic.Renderer):

    def __init__(self, num_assets: int):
        super().__init__()
        self.num_assets = num_assets

    def render(self, env, **kwargs):
        history = env.observer.renderer_history
        df = pd.DataFrame(history)
        # df.to_csv("./histofy.csv")
        sr = round(sharpe(df.pv.pct_change()), 2)
        mdd = round(maximum_drawdown(df.pv), 2)
        num_assets = self.num_assets
        fig, axs = plt.subplots(num_assets + 2, 1, figsize=(3 * (num_assets + 2) + 5, 400), sharex=True)

        fig.suptitle(f"Portfolio Reallocation Chart (Sharpe: {sr}, MDD: {mdd}%)")

        x = list(range(len(history)))

        axs[0].set_ylabel(f"Cash")
        axs[0].set_xlim(0, len(history))

        ax2 = axs[0].twinx()
        ax2.plot(x, [row["weights"][0] for row in history], color="r")
        ax2.set_ylabel("Weight")
        ax2.set_ylim(0, 1)

        for i in range(num_assets):
            y1 = [row["lag_stream"][i] for row in history]
            y2 = [row["weights"][i + 1] for row in history]
            ax = axs[i + 1]
            ax.plot(x, y1)
            ax.set_ylabel(f"Asset {i + 1}")
            ax.set_xlim(0, len(history))

            ax2 = ax.twinx()
            ax2.plot(x, y2, color="r")
            ax2.set_ylabel("Weight")
            ax2.set_ylim(0, .5)

        df.pv.plot(ax=axs[-1])
        axs[-1].set_xlabel("Step")
        axs[-1].set_ylabel("Portfolio Value")

        fig.tight_layout()
        fig.savefig("charts/reallocation_chart.png")


def make_features(s: Stream[float]) -> List[Stream[float]]:
    return [
        fracdiff(s, d=0.6, window=25).lag().rename("fd"),
        macd(s, fast=20, slow=100, signal=50).lag().rename("macd"),
        rsi(s, period=20, use_multiplier=False).lag().rename("rsi")
    ]


def index(i: int):
    def f(x: np.array):
        return x[i]

    return f


def matrix_index(i: int, k: int):
    def f(x: np.array):
        return x[i, :, k]

    return f


def create_env_with_price_series(config: dict, price_stream: Stream[np.array], high_price_stream: Stream[np.array],
                                 low_price_stream: Stream[np.array]
                                 , volume_stream: Stream[np.array], signal_stream: Stream[np.array]):
    np.seterr(divide="ignore")
    action_scheme = PortfolioAllocation(
        num_assets=config["num_assets"],
        commission=config["commission"],
        initial_cash=config["initial_cash"]
    )

    total_steps = config["total_steps"] + config["min_periods"]
    m = config["num_assets"]
    ne = config["num_experts"]
    ##p_streams = [price_stream.apply(index(i), dtype="float").rename(f"p{i}") for i in range(m)]
    p_streams = [price_stream.rename("price_stream")]
    high_streams = high_price_stream.rename("h_price")
    low_streams = low_price_stream.rename(f"l_price")
    volume_streams = volume_stream.rename(f"volume")
    signal_stream = signal_stream.rename(f"signal_stream")

    lag_streams = p_streams[0].lag().rename("lag_stream")

    weights = Stream.sensor(action_scheme, lambda x: x.weights).rename("weights")
    pv = Stream.sensor(action_scheme, lambda x: x.p, dtype="float").rename("pv")

    obs_group = Stream.group(
        [p_streams[0], high_streams, low_streams, volume_streams, signal_stream, lag_streams]).rename("obs")
    state_group = Stream.group([p_streams[0], lag_streams]).rename("state")
    perform_group = Stream.group([lag_streams, weights, pv]).rename("renderer")

    feed = DataFeed([obs_group, state_group, perform_group])
    env = TradingEnv(
        action_scheme=action_scheme,
        reward_scheme=Profit(),
        observer=Observer(
            feed=feed,
            window_size=config["window_size"],
            number_of_assets=config["num_assets"],
            num_assets=config["num_assets"],
            min_periods=config["min_periods"]
        ),
        stopper=MaxStepsOrNoMoney(config["min_profit"], config["max_drawdown"]),
        informer=RebalanceInformer(),
        renderer=ReallocationChart(num_assets=config["num_assets"])
    )

    return env


def create_env(config: dict):
    start_date = config['start_date']
    end_date = config['end_date']
    path = './prices.csv'
    signal_path = './signals.csv'
    signals = pd.read_csv(signal_path)
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    env = create_env_with_price_series(
        config=config,
        price_stream=StockPriceFromFile(df, 'close', start_date, end_date),
        high_price_stream=StockPriceFromFile(df, 'high', start_date, end_date),
        low_price_stream=StockPriceFromFile(df, 'low', start_date, end_date),
        volume_stream=StockPriceFromFile(df, 'volume', start_date, end_date),
        signal_stream=ComputationalSignalManager(start_date, end_date, signals, config["num_assets"],
                                                 config["num_experts"])
    )
    return env
