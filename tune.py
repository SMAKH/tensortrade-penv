import json

import click
import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining

from __init__ import *


@click.command()
@click.option("--num-samples", default=2, type=int)
@click.option("--num-workers", default=1, type=int)
def main(num_samples=1, num_workers=1):
    pbt = PopulationBasedTraining(
        time_attr="time_total_s",
        perturbation_interval=50,
        resample_probability=0.25,
        hyperparam_mutations={
            "lambda": tune.uniform(0.85, 1.0),
            "clip_param": tune.uniform(0.01, 0.5),
            "lr": [1e-2, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            "num_sgd_iter": tune.randint(1, 30),
            "sgd_minibatch_size": tune.randint(100, 300),
            "train_batch_size": tune.randint(1000, 1500),
            "rollout_fragment_length": [50, 100, 150],
        }
    )

    analysis = tune.run(
        "PPO",
        name="pbt_portfolio_reallocation",
        scheduler=pbt,
        num_samples=num_samples,
        metric="episode_reward_min",
        mode="max",
        config={
            "env": "TradingEnv",
            "env_config": {
                "total_steps": 200,
                "num_assets": 54,
                "num_experts": 85,
                "commission": 0.001,
                "time_cost": 0,
                "window_size": 60,
                "min_periods": 60,
                "min_profit": tune.choice([-0.2, -0.05, 0]),
                "max_drawdown": tune.choice([-0.3, -0.1, -0.05, 0]),
                "start_date": "6/22/2015",
                "end_date": "2/28/2019",
                "initial_cash": 1.0,
                "validation_start_date": "3/1/2019",
                "validation_start_close": "11/30/2019"
            },
            "kl_coeff": 1.0,
            "num_workers": 8,
            "num_gpus": 1,
            "rollout_fragment_length": 100,
            "observation_filter": tune.choice(["NoFilter", "MeanStdFilter"]),
            "batch_mode": tune.choice(["truncate_episodes", "complete_episodes"]),
            "framework": "torch",
            "model": {
                "custom_model": "reallocate",
                "custom_model_config": {
                    "num_assets": 54,
                    "num_experts": 85,
                    "num_features": 174,
                    "second_channels": 80,
                    "third_channels": 30,
                    "forth_channels": 10
                },
                "custom_action_dist": "dirichlet",
            },
            "num_sgd_iter": 10,
            "sgd_minibatch_size": 128,
            "lambda": tune.uniform(0.85, 1.0),
            "clip_param": tune.uniform(0.1, 0.5),
            "lr": tune.loguniform(1e-2, 1e-5),
            "train_batch_size": tune.randint(1000, 2000)
        },
        stop={
            "training_iteration": 500
        },
        checkpoint_at_end=True,
        local_dir="./results"
    )

    checkpoints = analysis.get_trial_checkpoints_paths(
        trial=analysis.get_best_trial(metric="episode_reward_min", mode="max"),
        metric="episode_reward_mean"
    )

    params = {
        "config": analysis.best_config,
        "checkpoints": checkpoints
    }

    json.dump(params, open("./data/tuned_params.json", "w"), indent=4)


if __name__ == "__main__":
    ray.init()

    main()

    ray.shutdown()
