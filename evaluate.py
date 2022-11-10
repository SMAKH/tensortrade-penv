import json

import pandas as pd
import ray
import ray.rllib.agents.ppo as ppo

from env import create_env_with_price_series
from price_generators import StockPriceFromFile, ComputationalSignalManager
from __init__ import *


def main(random: bool = False):
    params = json.load(open("./data/tuned_params.json", "r"))

    config = params["config"]
    config["num_workers"] = 1
    config["explore"] = False
    config["env_config"]["total_steps"] = int(365)
    config["env_config"]["max_drawdown"] = -0.8
    config["env_config"]["min_profit"] = -0.1

    checkpoint = "/app/results/portfolio_allocation/PPO_TradingEnv_fc76e_00000_0_2022-02-13_01-26-13/checkpoint_000790/checkpoint-790"

    agent = ppo.PPOTrainer(env="TradingEnv", config=config)
    agent.restore(checkpoint)

    config = config["env_config"]
    config["validation"] = True
    config["start_date"] = config["validation_start_date"]
    config["end_date"] = config["validation_close_date"]
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
    done = False
    obs = env.reset()
    action = env.action_scheme.weights.copy()

    while not done:
        if not random:
            action = agent.compute_action(obs, prev_action=action)
        else:
            action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

    env.render()


if __name__ == "__main__":
    ray.init()

    try:
        main()
    finally:
        ray.shutdown()
