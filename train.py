import json

import click
import numpy as np
import pandas as pd
import ray
from ray import tune

from metrics import sharpe, maximum_drawdown

from __init__ import *


def on_episode_end(info):
    env = info["env"].vector_env.envs[0]
    history = env.observer.renderer_history
    pv = np.array([row["pv"] for row in history])
    returns = pd.Series(pv).pct_change()

    episode = info["episode"]
    episode.custom_metrics["sharpe"] = sharpe(returns.values)
    episode.custom_metrics["MDD"] = maximum_drawdown(pv)


@click.command()
def main():
    params = json.load(open("tensortrade-penv/data/tuned_params.json", "r"))

    config = params["config"].copy()
    config["callbacks"] = {
        "on_episode_end": on_episode_end
    }
    checkpoint = params["checkpoints"][0][0]

    analysis = tune.run(
        "PPO",
        checkpoint_score_attr="episode_reward_min",
        name="portfolio_allocation",
        config=config,
        stop={
            "training_iteration": 1000
        },
        checkpoint_freq=10,
        restore=checkpoint,
        checkpoint_at_end=True,
        local_dir="tensortrade-penv/results"
    )

    checkpoints = analysis.get_trial_checkpoints_paths(
        trial=analysis.get_best_trial(metric="episode_reward_min", mode="max"),
        metric="episode_reward_mean"
    )

    params["checkpoints"] = checkpoints
    json.dump(params, open("tensortrade-penv/data/trained_params.json", "w"), indent=4)


import sys

sys.setrecursionlimit(100000)
if __name__ == "__main__":
    ray.init()
    main()
    ray.shutdown()
