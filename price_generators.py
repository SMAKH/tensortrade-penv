import numpy
import numpy as np
import pandas as pd
from scipy.stats import truncnorm
from tensortrade.feed import Stream


class MultiGBM(Stream[np.array]):

    def __init__(self, s0: np.array, drift: np.array, volatility: np.array, rho: np.array, n: int):
        super().__init__()
        self.n = n
        self.m = len(s0)
        self.i = 0

        self.dt = 1 / n

        self.s0 = s0.reshape(-1, 1)
        self.mu = drift.reshape(-1, 1)
        self.v = volatility.reshape(-1, 1)

        V = (self.v @ self.v.T) * rho
        self.A = np.linalg.cholesky(V)
        self.x = None

    def forward(self) -> np.array:
        self.i += 1
        if self.x is None:
            self.x = self.s0.flatten().astype(float)
            return self.x
        dw = np.random.normal(loc=0, scale=np.sqrt(self.dt), size=[self.m, 1])
        s = np.exp((self.mu - (1 / 2) * self.v ** 2) * self.dt + (self.A @ dw)).T
        s = s.flatten()
        self.x *= s
        return self.x

    def has_next(self):
        return self.i < self.n

    def reset(self):
        super().reset()
        self.i = 0
        self.x = None


def multi_corr_gbm(s0: np.array, drift: np.array, volatility: np.array, rho: np.array, n: int):
    m = len(s0)
    assert drift.shape == (m,)
    assert volatility.shape == (m,)
    assert rho.shape == (m, m)

    dt = 1 / n

    s0 = s0.reshape(-1, 1)  # Shape: (m, 1)
    mu = drift.reshape(-1, 1)  # Shape: (m, 1)
    v = volatility.reshape(-1, 1)  # Shape: (m, 1)

    V = (v @ v.T) * rho
    A = np.linalg.cholesky(V)  # Shape: (m, m)

    dW = np.random.normal(loc=0, scale=np.sqrt(dt), size=[m, n])  # Shape (m, n)

    S = np.exp((mu - (1 / 2) * v ** 2) * dt + (A @ dW)).T

    S = np.vstack([np.ones(m), S])

    S = s0.T * S.cumprod(0)

    return S


def make_multi_gbm_price_curve(n: int):
    rho = np.array([
        [1., -0.34372319, 0.23809065, -0.21918481],
        [-0.34372319, 1., -0.07774865, -0.17430333],
        [0.23809065, -0.07774865, 1., -0.17521052],
        [-0.21918481, -0.17430333, -0.17521052, 1.]
    ])
    s0 = np.array([50, 48, 45, 60])
    drift = np.array([0.13, 0.16, 0.10, 0.05])
    volatility = np.array([0.25, 0.20, 0.30, 0.15])

    P = multi_corr_gbm(s0, drift, volatility, rho, n)

    prices = pd.DataFrame(P).astype(float)
    prices.columns = ["p1", "p2", "p3", "p4"]

    # prices = prices.ewm(span=50).mean()

    return prices


def make_shifting_sine_price_curves(n: int, warmup: int = 0):
    n += 1

    slide = 2 * np.pi * (warmup / n)

    steps = n + warmup

    x = np.linspace(-slide, 2 * np.pi, num=steps)
    x = np.repeat(x, 4).reshape(steps, 4)

    s0 = np.array([50, 48, 45, 60]).reshape(1, 4)
    shift = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2]).reshape(1, 4)
    freq = np.array([1, 4, 3, 2]).reshape(1, 4)

    y = s0 + 25 * np.sin(freq * (x - shift))

    prices = pd.DataFrame(y, columns=["p1", "p2", "p3", "p4"])

    return prices


class MultiSinePriceCurves(Stream[np.array]):

    def __init__(self, s0: np.array, shift: np.array, freq: np.array, n: int, warmup: int = 0):
        super().__init__()
        self.s0 = s0
        self.shift = shift
        self.freq = freq

        self.steps = n + warmup + 1
        self.i = 0
        self.m = len(s0)

        self.x = np.linspace(-2 * np.pi * (warmup / n), 2 * np.pi, num=self.steps)

    def forward(self) -> np.array:
        rv = truncnorm.rvs(a=-10, b=10, size=self.m)
        v = self.s0 + 25 * np.sin(self.freq * (self.x[self.i] - self.shift)) + rv
        self.i += 1
        return v

    def has_next(self):
        return self.i < self.steps

    def reset(self):
        super().reset()
        self.i = 0


class StockPriceFromFile(Stream[np.array]):
    def __init__(self, prices, hloc, start_date='6/22/2015', end_date='12/31/2019'):
        super().__init__()
        self.start_date = start_date
        self.end_date = end_date
        self.data = prices
        self.hloc = hloc
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data = self.data[(self.data['date'] >= self.start_date) & (self.data['date'] <= self.end_date)]
        columns = ["date"]
        for col in self.data.columns[1:]:
            if col.split(":")[1] == self.hloc:
                columns.append(col)
        self.data = self.data[columns]
        self.number_of_assets = len(columns) - 1  # -1 is for date column
        self.index = 0
        self.max_steps = len(self.data.index)

    def forward(self):  # -> np.array:
        values = numpy.zeros([self.number_of_assets])
        row = self.data.iloc[self.index, 1:self.number_of_assets + 1]
        counter = 0
        '''for index,value in row.items():
           values[counter]=value
           counter+=1
        '''
        values = np.array(list(row))
        self.index += 1
        return values

    def has_next(self):
        return self.index < self.max_steps - 1

    def reset(self):
        super().reset()
        self.index = 0


class ComputationalSignalManager(Stream[np.array]):
    def __init__(self, start_date, end_date, data, number_of_symbols, number_of_experts):
        super().__init__()
        self.data = data
        self.data['start_date'] = pd.to_datetime(self.data['start_date'])
        self.data['close_date'] = pd.to_datetime(self.data['close_date'])
        self.start_date = start_date
        self.end_date = end_date
        self.date_range = pd.date_range(start_date, end_date, freq='D')
        self.max_steps = len(self.date_range)
        self.number_of_experts = number_of_experts
        self.number_of_symbols = number_of_symbols
        self.current_step = 0
        self.index = 0
        self.current_date = pd.to_datetime(start_date)

    def convert_signals_to_tensors(self, current_signals):
        z = np.zeros(shape=(self.number_of_experts, 2, self.number_of_symbols), dtype=np.float32)

        for s in current_signals.iterrows():
            sym_id = s[0][0]
            exp_id = s[0][1]
            profit = s[1][4]
            loss = s[1][5]
            z[exp_id, 0, sym_id] = profit
            z[exp_id, 1, sym_id] = loss
        return z

    def has_next(self):
        return self.index < self.max_steps - 1

    def reset(self):
        super().reset()
        self.index = 0
        self.current_date = pd.to_datetime(self.start_date)

    def forward(self) -> np.array:
        singnals = self.data[
            (self.data['start_date'] <= self.current_date) & (self.data['close_date'] >= self.current_date)]
        si = singnals.groupby(['symbol_id', 'expert_id']).mean()
        self.current_date += pd.Timedelta(days=1)
        self.index += 1
        return self.convert_signals_to_tensors(si)
