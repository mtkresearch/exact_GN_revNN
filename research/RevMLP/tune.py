import os
import torch
from types import SimpleNamespace
import ray
from ray import train as ray_train, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.logger.aim import AimLoggerCallback
from ray.tune.search.hyperopt import HyperOptSearch
import yaml

from train import main_train
from utils import *


def tuning_callback(config):
    args = SimpleNamespace(**config)
    main_train(args, tuning=True)


def main():
    # These need to be False or Ray might break
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False

    args = create_argparser().parse_args()

    # Parse the arguments.
    parser = Parser(config=args)

    (
        exp_args,
        data_args,
        model_args,
        train_args,
        results_args,
    ) = parser.parse()

    # reconstruct the config dict for ray tune (considering default params)
    search_space = {
        "Experiment": vars(exp_args),
        "Data": vars(data_args),
        "Model": vars(model_args),
        "Train": vars(train_args),
        "Results": vars(results_args),
    }.copy()

    tune_config_file = os.path.join("tune_config", "tune_test.yaml")
    with open(tune_config_file) as f:
        tune_params = yaml.safe_load(f)

    set_tune_params(search_space, tune_params)

    ray.init(address="auto")

    tuner = tune.Tuner(
        tune.with_resources(tuning_callback, {"gpu": 1}),
        tune_config=tune.TuneConfig(
            num_samples=4,
            scheduler=ASHAScheduler(metric="mean_accuracy", mode="max"),
            search_alg=HyperOptSearch(search_space, metric="mean_accuracy", mode="max"),
        ),
        #param_space=search_space,
        run_config=ray_train.RunConfig(
            callbacks=[AimLoggerCallback()],
            name="ray_tune",
        ),
    )
    results = tuner.fit()

    print(results.get_best_result("mean_accuracy", mode="max"))


if __name__ == "__main__":
    main()
