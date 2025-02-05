from fastbreak.utils.cmd_utils import to_yaml_interface
from model.defaults import model_defaults
import argparse
from argparse import Namespace


class Parser:
    def __init__(self, config):
        self._config_args = config

    @property
    def experiment_args(self):
        return Namespace(**self._config_args.Experiment)

    @property
    def data_args(self):
        return Namespace(**self._config_args.Data)

    @property
    def model_args(self):
        return Namespace(**self._config_args.Model)

    @property
    def train_args(self):
        return Namespace(**self._config_args.Train)

    @property
    def results_args(self):
        return Namespace(**self._config_args.Results)

    def parse(self):
        return (
            self.experiment_args,
            self.data_args,
            self.model_args,
            self.train_args,
            self.results_args,
        )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


@to_yaml_interface(__file__)
def create_argparser(description="FastBreak Experiment"):
    defaults = {
        "Experiment": {
            "version": 1.0,
            "seed": 42,
            "description": "Dense",
            "device": True,
        },
        "Data": {
            "name": "synthetic",
            "input_size": 300,
            "output_size": 300,
            "batch_size": 20,
            "batch_size_test": 20,
        },
        "Train": {
            "Task": {"name": "labeller", "args": {"load_from": "random"}},
            "loss_function": {"name": "mse", "args": {}},
            "optimizer": {"name": "sgd", "args": {"lr": 0.1}},
            "args": {
                "num_epochs": 50,
                "steps_for_printing": 1,
                "save_checkpoint_every": False,
            },
        },
        "Results": {
            "output_dir": "",
            "format_strs": ["stdout", "csv", "json", "tensorboard"],
            "files": {},
        },
    }
    defaults.update(model_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    print(parser._defaults)
    return parser
