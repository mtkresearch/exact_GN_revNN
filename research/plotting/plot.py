import sys

import matplotlib.pyplot as plt
import numpy as np

from plotter import *

# RevMLP: Results & Analysis

# dir_sgd = "/proj/gpu_mtk53601/research/deq_experiments/alberto_experiment/research/inv_dense/results_old_weight_init_oldC/best/mnist512/sgd512"
dir_sgd = "/proj/gpu_mtk53548/Fastbreak/research/inv_dense/results/results_old_weight_init_oldC/best/mnist/sgd"
# dir_adam = "/proj/gpu_mtk53601/research/deq_experiments/alberto_experiment/research/inv_dense/results_old_weight_init_oldC/best/mnist512/adam512"
dir_adam = "/proj/gpu_mtk53548/Fastbreak/research/inv_dense/results/results_old_weight_init_oldC/best/mnist/adam"
# dir_ggn = "/proj/gpu_mtk53601/research/deq_experiments/alberto_experiment/research/inv_dense/results_old_weight_init_oldC/best/mnist512/ggn512"
dir_ggn = "/proj/gpu_mtk53548/Fastbreak/research/results_ggn_mlp25M"
# dir_ggn_rsvd = "/proj/gpu_mtk53601/research/deq_experiments/alberto_experiment/research/inv_dense/results_old_weight_init_oldC/best/mnist512/ggn512_rsvd"
dir_ggn_rsvd = "/proj/gpu_mtk53548/Fastbreak/research/inv_dense/results/results_old_weight_init_oldC/best/mnist/ggn_rsvd"
dir_tp = "/proj/gpu_mtk53548/Fastbreak/research/inv_dense/results/results_old_weight_init_oldC/best/mnist/tp"


experiments_sgd = ["sgd_experiment_lr_.10000000"]
experiments_adam = ["adam_experiment_lr_.00010000"]
# experiments_adam = ["adam_experiment_lr_.00100000"]
experiments_ggn = ["ggn_experiment_lr_.10000_rtol_.0100"]
# experiments_ggn = ["ggn_experiment_lr_1.00000_rtol_.0100"] # 512
experiments_ggn_rsvd = ["ggn_rsvd_experiment_lr_1.00000_rtol_.0010"]
# experiments_ggn_rsvd = ["ggn_experiment_lr_1.00000_rtol_.0100"] # 512
experiments_tp = ["tp"]

x_value = "time_mean"
# x_value = "epoch"
x_value = "iter"

############################################################################
# y_value should be equal to one of:
# "train_loss_mean", "trainacc_mean", "val_loss_mean", "val_acc_mean"
############################################################################

# Usage
# Change this to "train_loss", "train_acc", or "val_acc" as needed
y_value_base = (
    # "train_loss",
    # "train_acc",
    "val_loss",
    # "val_acc",
)
y_value, y_std = set_y_values(y_value_base)


tick_labels = {
    "iter": "Iterations",
    "epoch": "Epochs",
    "time_mean": "Time (s)",
    "train_loss_mean": "Train Loss",
    "train_acc_mean": "Train Accuracy",
    "train_loss": "Train Loss",
    "val_loss_mean": "Test Loss",
    "val_acc_mean": "Test Accuracy",
}

xlims_dict = {"time_mean": (0, 2000), "epoch": (1, 40), "iter": (0, 2000)}
ylims_dict = {
    "val_acc_mean": (0, 100),
    "val_loss_mean": (1.4, 3.0),
    "train_loss_mean": (0.01, 0.8),
    "train_acc_mean": (85, 101),
}


labels_dict = {
    "ggn_experiment_lr_1.00000_rtol_.0010": "GGN (Ours)",
    "ggn_experiment_lr_1.00000_rtol_.0100": "GGN",  # 512
    "ggn_experiment_lr_1.00000_rtol_.1000": "GGN",
    "ggn_experiment_lr_.10000_rtol_.0100": "GGN",
    "adam_experiment_lr_.00010000": "Adam",
    # "adam_experiment_lr_.00100000": "Adam",
    "sgd_experiment_lr_.10000000": "SGD",
    "ggn_rsvd_experiment_lr_1.00000_rtol_.0010": "RSVD-GGN (Ours)",
    # "ggn_experiment_lr_1.00000_rtol_.0100": "GGN (RSVD)", # 512
    "tp": "TP",
}

out_path = ""
out_filename = f"{y_value_base[0]}_{x_value}.png"

plot_args = {
    # "showlegend": True,
    # "figsize": (8, 4),
    "xlabel": tick_labels[x_value],
    "ylabel": tick_labels[y_value],
    "markersize": 10,
    "alpha": 0.6,
    "xlim": xlims_dict[x_value],
    "ylim": ylims_dict[y_value],
    "conference": "icml",
    "yscale": "log" if "loss" in y_value else "linear",
}

##############################################################################################################
# MODIFY EVERYTHING BELOW WITH CARE!!!!
##############################################################################################################

plotter = Plotter(**plot_args)


def plot_experiment(
    plotter, exp_dir, experiment, x_value, y_value, y_std=False, label=False
):
    values, label_dict = create_single_plot_dict(
        dir=exp_dir,
        experiment=experiment,
        values=[x_value, y_value, y_std] if y_std else [x_value, y_value],
        label=label if label else experiment,
        per_epoch=True
        if x_value.lower() == "epoch" or x_value.lower() == "time_mean"
        else False,
    )

    for name, value in values.items():
        if "time" in x_value.lower():
            x_axis_vals = value[x_value].cumsum()
        elif "epoch" in x_value.lower():
            x_axis_vals = range(1, value[x_value].iloc[-1] + 2)
        else:
            x_axis_vals = value[x_value]

        if y_std:
            plotter.add_fillbetween_plot(
                x_axis_vals,
                value[y_value] + value[y_std],
                value[y_value] - value[y_std],
                showlegend=False,
                alpha=0.4,
            )
            plotter.add_plot(
                x_axis_vals,
                value[y_value],
                label=label_dict[name],
                showlegend=False,
                alpha=0.8,
            )
        else:
            plotter.add_plot(
                x_axis_vals,
                value[y_value],
                label=label_dict[name],
            )


def plot_all_experiments(
    plotter, exp_dir, experiments, x_value, y_value, y_std, labels_dict=False
):
    for experiment in experiments:
        label = labels_dict[experiment] if labels_dict else experiment
        plot_experiment(plotter, exp_dir, experiment, x_value, y_value, y_std, label)


# plot_all_experiments(
#     plotter, dir_sgd, experiments_sgd, x_value, y_value, y_std, labels_dict
# )
# plot_all_experiments(
#     plotter, dir_adam, experiments_adam, x_value, y_value, y_std, labels_dict
# )
# plot_all_experiments(
#     plotter, dir_tp, experiments_tp, x_value, y_value, y_std, labels_dict
# )
plotter._update(boldlegend="bold")
plot_all_experiments(
    plotter, dir_ggn, experiments_ggn, x_value, y_value, y_std, labels_dict
)
# plot_all_experiments(
#     plotter, dir_ggn_rsvd, experiments_ggn_rsvd, x_value, y_value, y_std, labels_dict
# )


plotter.savefig(os.path.join(out_path, out_filename))
