import sys

import matplotlib.pyplot as plt
import numpy as np

from research.plotting.plotter import *

# RevMLP: Results & Analysis

dir_sgd = "/proj/gpu_mtk53548/Fastbreak/research/results_revmlp_cdim/mnist/sgd/"
dir_adam = "/proj/gpu_mtk53548/Fastbreak/research/results_revmlp_cdim/mnist/adam/"
dir_ggn = "/proj/gpu_mtk53548/Fastbreak/research/results_revmlp_cdim/mnist/ggn/"


## SGD Results
experiments_sgd = get_unique_experiment_prefixes(dir_sgd)


### Per-Iteration Results
_, full_dict_sgd, _ = create_plot_dict(dir_sgd, experiments_sgd)

passes_sgd, fails_sgd = get_passes_and_fails(full_dict_sgd)

x_list_sgd, pass_dict_sgd, labels_dict_sgd = create_plot_dict(
    dir_sgd,
    passes_sgd,
    y_value=["Running loss_mean", "Running acc_mean", "Val Loss_mean", "Val Acc_mean"],
)


# Rank the dataframes by the minimum validation loss at index 1
rankings_sgd = rank_by_min_val_loss(pass_dict_sgd, 1)
# Now `ranked_dict` is an ordered dictionary from the lowest to the highest validation loss at index 99

# Set up the figsize to be ICML dimensions
fig, axes = plt.subplots(
    figsize=(3.25 * 2, 2.0086104634371584 * 2), nrows=2, ncols=2
)  # Create fig and axes

k = 10

sgd_plots = Plotter(
    # showlegend=True,
    figsize=(8, 4),
    xlabel="Iterations",
    fontsize=14,
    markersize=10,
    alpha=0.6,
    xlim=(0, x_list_sgd.iloc[-1]),
    # ylim=(0,100),
    conference="icml",
)

rankings_subset_sgd = OrderedDict(list(rankings_sgd.items())[:k])

labels_sgd = {item: item for item in rankings_subset_sgd.keys()}

train_loss_sgd, train_acc_sgd, val_loss_sgd, val_acc_sgd = extract_columns_to_dict(
    rankings_subset_sgd
)

for (x_ind, y_ind), ax in np.ndenumerate(axes):
    if x_ind == 0 and y_ind == 0:
        sgd_plots.new_plot(fig=fig, ax=ax, yscale="log", ylabel="Train Loss")
        sgd_plots.iter_plot(
            x_list_sgd,
            train_loss_sgd,
            label=labels_sgd,
        )
        # ax.legend()
    if x_ind == 0 and y_ind == 1:
        sgd_plots.new_plot(fig=fig, ax=ax, yscale="linear", ylabel="Train Accuracy")
        sgd_plots.iter_plot(
            x_list_sgd,
            train_acc_sgd,
            label=labels_sgd,
        )
        pos = ax.get_position()
        # ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
        ax.legend(loc="center right", bbox_to_anchor=(1.25, 0.5))
    if x_ind == 1 and y_ind == 0:
        sgd_plots.new_plot(fig=fig, ax=ax, yscale="log", ylabel="Test Loss")
        sgd_plots.iter_plot(
            x_list_sgd,
            val_loss_sgd,
            label=labels_sgd,
        )
    if x_ind == 1 and y_ind == 1:
        sgd_plots.new_plot(fig=fig, ax=ax, yscale="linear", ylabel="Test Accuracy")
        sgd_plots.iter_plot(
            x_list_sgd,
            val_acc_sgd,
            label=labels_sgd,
        )

sgd_plots.savefig("sgd_tuning_iter.png")

### Per-Epoch Results
_, full_dict_sgd_epoch, _ = create_plot_dict(
    dir_sgd, experiments_sgd, x_value="Epoch", y_value=["Train Loss_mean"]
)
passes_sgd_epoch, fails_sgd_epoch = get_passes_and_fails(full_dict_sgd_epoch)

x_list_sgd_epoch, pass_dict_sgd_epoch, labels_dict_sgd = create_plot_dict(
    dir_sgd,
    passes_sgd_epoch,
    x_value="Epoch",
    y_value=["Train Loss_mean", "Train Acc_mean", "Val Loss_mean", "Val Acc_mean"],
)
rankings_sgd_epoch = rank_by_min_val_loss(pass_dict_sgd_epoch, 1)
fig, axes = plt.subplots(
    figsize=(3.25 * 2, 2.0086104634371584 * 2), nrows=2, ncols=2
)  # Create fig and axes

k = 10

sgd_epoch_plots = Plotter(
    # showlegend=True,
    figsize=(8, 4),
    xlabel="Epochs",
    fontsize=14,
    markersize=10,
    alpha=0.6,
    xlim=(0, x_list_sgd_epoch.iloc[-1]),
    # ylim=(0,100),
    conference="icml",
)

rankings_subset_sgd_epoch = OrderedDict(list(rankings_sgd_epoch.items())[:k])

labels_sgd_epoch = {item: item for item in rankings_subset_sgd_epoch.keys()}

(
    train_loss_sgd_epoch,
    train_acc_sgd_epoch,
    val_loss_sgd_epoch,
    val_acc_sgd_epoch,
) = extract_columns_to_dict(rankings_subset_sgd_epoch)

for (x_ind, y_ind), ax in np.ndenumerate(axes):
    if x_ind == 0 and y_ind == 0:
        sgd_epoch_plots.new_plot(fig=fig, ax=ax, yscale="log", ylabel="Train Loss")
        sgd_epoch_plots.iter_plot(
            x_list_sgd_epoch,
            train_loss_sgd_epoch,
            label=labels_sgd_epoch,
        )
        # ax.legend()
    if x_ind == 0 and y_ind == 1:
        sgd_epoch_plots.new_plot(
            fig=fig, ax=ax, yscale="linear", ylabel="Train Accuracy"
        )
        sgd_epoch_plots.iter_plot(
            x_list_sgd_epoch,
            train_acc_sgd_epoch,
            label=labels_sgd_epoch,
        )
        pos = ax.get_position()
        # ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
        ax.legend(loc="center right", bbox_to_anchor=(1.25, 0.5))
    if x_ind == 1 and y_ind == 0:
        sgd_epoch_plots.new_plot(fig=fig, ax=ax, yscale="log", ylabel="Test Loss")
        sgd_epoch_plots.iter_plot(
            x_list_sgd_epoch,
            val_loss_sgd_epoch,
            label=labels_sgd_epoch,
        )
    if x_ind == 1 and y_ind == 1:
        sgd_epoch_plots.new_plot(
            fig=fig, ax=ax, yscale="linear", ylabel="Test Accuracy"
        )
        sgd_epoch_plots.iter_plot(
            x_list_sgd_epoch,
            val_acc_sgd_epoch,
            label=labels_sgd_epoch,
        )

sgd_epoch_plots.savefig("sgd_tuning_epoch.png")

## Adam Results

experiments_adam = get_unique_experiment_prefixes(dir_adam)
### Per-Iteration Results
_, full_dict_adam, _ = create_plot_dict(dir_adam, experiments_adam)

passes_adam, fails_adam = get_passes_and_fails(full_dict_adam)

x_list_adam, pass_dict_adam, labels_dict_adam = create_plot_dict(
    dir_adam,
    passes_adam,
    y_value=["Running loss_mean", "Running acc_mean", "Val Loss_mean", "Val Acc_mean"],
)
# Rank the dataframes by the minimum validation loss at index 1
rankings_adam = rank_by_min_val_loss(pass_dict_adam, 1)

# Now `ranked_dict` is an ordered dictionary from the lowest to the highest validation loss at index 99
fig, axes = plt.subplots(
    figsize=(3.25 * 2, 2.0086104634371584 * 2), nrows=2, ncols=2
)  # Create fig and axes

k = 10

adam_plots = Plotter(
    # showlegend=True,
    figsize=(8, 4),
    xlabel="Iterations",
    fontsize=14,
    markersize=10,
    alpha=0.6,
    xlim=(0, x_list_adam.iloc[-1]),
    # ylim=(0,100),
    conference="icml",
)

rankings_subset_adam = OrderedDict(list(rankings_adam.items())[:k])

labels_adam = {item: item for item in rankings_subset_adam.keys()}

train_loss_adam, train_acc_adam, val_loss_adam, val_acc_adam = extract_columns_to_dict(
    rankings_subset_adam
)

for (x_ind, y_ind), ax in np.ndenumerate(axes):
    if x_ind == 0 and y_ind == 0:
        adam_plots.new_plot(fig=fig, ax=ax, yscale="log", ylabel="Train Loss")
        adam_plots.iter_plot(
            x_list_adam,
            train_loss_adam,
            label=labels_adam,
        )
        # ax.legend()
    if x_ind == 0 and y_ind == 1:
        adam_plots.new_plot(fig=fig, ax=ax, yscale="linear", ylabel="Train Accuracy")
        adam_plots.iter_plot(
            x_list_adam,
            train_acc_adam,
            label=labels_adam,
        )
        pos = ax.get_position()
        # ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
        ax.legend(loc="center right", bbox_to_anchor=(1.25, 0.5))
    if x_ind == 1 and y_ind == 0:
        adam_plots.new_plot(fig=fig, ax=ax, yscale="log", ylabel="Test Loss")
        adam_plots.iter_plot(
            x_list_adam,
            val_loss_adam,
            label=labels_adam,
        )
    if x_ind == 1 and y_ind == 1:
        adam_plots.new_plot(fig=fig, ax=ax, yscale="linear", ylabel="Test Accuracy")
        adam_plots.iter_plot(
            x_list_adam,
            val_acc_adam,
            label=labels_adam,
        )

adam_plots.savefig("adam_tuning_iter.png")

### Per-Epoch Results
_, full_dict_adam_epoch, _ = create_plot_dict(
    dir_adam, experiments_adam, x_value="Epoch", y_value=["Train Loss_mean"]
)
passes_adam_epoch, fails_adam_epoch = get_passes_and_fails(full_dict_adam_epoch)

x_list_adam_epoch, pass_dict_adam_epoch, labels_dict_adam = create_plot_dict(
    dir_adam,
    passes_adam_epoch,
    x_value="Epoch",
    y_value=["Train Loss_mean", "Train Acc_mean", "Val Loss_mean", "Val Acc_mean"],
)
rankings_adam_epoch = rank_by_min_val_loss(pass_dict_adam_epoch, 1)
fig, axes = plt.subplots(
    figsize=(3.25 * 2, 2.0086104634371584 * 2), nrows=2, ncols=2
)  # Create fig and axes

k = 10

adam_epoch_plots = Plotter(
    # showlegend=True,
    figsize=(8, 4),
    xlabel="Epochs",
    fontsize=14,
    markersize=10,
    alpha=0.6,
    xlim=(0, x_list_adam_epoch.iloc[-1]),
    # ylim=(0,100),
    conference="icml",
)

rankings_subset_adam_epoch = OrderedDict(list(rankings_adam_epoch.items())[:k])

labels_adam_epoch = {item: item for item in rankings_subset_adam_epoch.keys()}

(
    train_loss_adam_epoch,
    train_acc_adam_epoch,
    val_loss_adam_epoch,
    val_acc_adam_epoch,
) = extract_columns_to_dict(rankings_subset_adam_epoch)

for (x_ind, y_ind), ax in np.ndenumerate(axes):
    if x_ind == 0 and y_ind == 0:
        adam_epoch_plots.new_plot(fig=fig, ax=ax, yscale="log", ylabel="Train Loss")
        adam_epoch_plots.iter_plot(
            x_list_adam_epoch,
            train_loss_adam_epoch,
            label=labels_adam_epoch,
        )
        # ax.legend()
    if x_ind == 0 and y_ind == 1:
        adam_epoch_plots.new_plot(
            fig=fig, ax=ax, yscale="linear", ylabel="Train Accuracy"
        )
        adam_epoch_plots.iter_plot(
            x_list_adam_epoch,
            train_acc_adam_epoch,
            label=labels_adam_epoch,
        )
        pos = ax.get_position()
        # ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
        ax.legend(loc="center right", bbox_to_anchor=(1.25, 0.5))
    if x_ind == 1 and y_ind == 0:
        adam_epoch_plots.new_plot(fig=fig, ax=ax, yscale="log", ylabel="Test Loss")
        adam_epoch_plots.iter_plot(
            x_list_adam_epoch,
            val_loss_adam_epoch,
            label=labels_adam_epoch,
        )
    if x_ind == 1 and y_ind == 1:
        adam_epoch_plots.new_plot(
            fig=fig, ax=ax, yscale="linear", ylabel="Test Accuracy"
        )
        adam_epoch_plots.iter_plot(
            x_list_adam_epoch,
            val_acc_adam_epoch,
            label=labels_adam_epoch,
        )

adam_epoch_plots.savefig("adam_tuning_epoch.png")

## GGN Results

experiments_ggn = get_unique_experiment_prefixes(dir_ggn)
### Per-Iteration Results
_, full_dict_ggn, _ = create_plot_dict(dir_ggn, experiments_ggn)

passes_ggn, fails_ggn = get_passes_and_fails(full_dict_ggn)

x_list_ggn, pass_dict_ggn, labels_dict_ggn = create_plot_dict(
    dir_ggn,
    passes_ggn,
    y_value=["Running loss_mean", "Running acc_mean", "Val Loss_mean", "Val Acc_mean"],
)


# Rank the dataframes by the minimum validation loss at index 1
rankings_ggn = rank_by_min_val_loss(pass_dict_ggn, 1)

# Now `ranked_dict` is an ordered dictionary from the lowest to the highest validation loss at index 99
fig, axes = plt.subplots(
    figsize=(3.25 * 2, 2.0086104634371584 * 2), nrows=2, ncols=2
)  # Create fig and axes

k = 10

ggn_plots = Plotter(
    # showlegend=True,
    figsize=(8, 4),
    xlabel="Iterations",
    fontsize=14,
    markersize=10,
    alpha=0.6,
    xlim=(0, x_list_ggn.iloc[-1]),
    # ylim=(0,100),
    conference="icml",
)

rankings_subset_ggn = OrderedDict(list(rankings_ggn.items())[:k])

labels_ggn = {item: item for item in rankings_subset_ggn.keys()}

train_loss_ggn, train_acc_ggn, val_loss_ggn, val_acc_ggn = extract_columns_to_dict(
    rankings_subset_ggn
)

for (x_ind, y_ind), ax in np.ndenumerate(axes):
    if x_ind == 0 and y_ind == 0:
        ggn_plots.new_plot(fig=fig, ax=ax, yscale="log", ylabel="Train Loss")
        ggn_plots.iter_plot(
            x_list_ggn,
            train_loss_ggn,
            label=labels_ggn,
        )
        # ax.legend()
    if x_ind == 0 and y_ind == 1:
        ggn_plots.new_plot(fig=fig, ax=ax, yscale="linear", ylabel="Train Accuracy")
        ggn_plots.iter_plot(
            x_list_ggn,
            train_acc_ggn,
            label=labels_ggn,
        )
        pos = ax.get_position()
        # ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
        ax.legend(loc="center right", bbox_to_anchor=(1.25, 0.5))
    if x_ind == 1 and y_ind == 0:
        ggn_plots.new_plot(fig=fig, ax=ax, yscale="log", ylabel="Test Loss")
        ggn_plots.iter_plot(
            x_list_ggn,
            val_loss_ggn,
            label=labels_ggn,
        )
    if x_ind == 1 and y_ind == 1:
        ggn_plots.new_plot(fig=fig, ax=ax, yscale="linear", ylabel="Test Accuracy")
        ggn_plots.iter_plot(
            x_list_ggn,
            val_acc_ggn,
            label=labels_ggn,
        )

ggn_plots.savefig("ggn_tuning_iter.png")

### Per-Epoch Results
_, full_dict_ggn_epoch, _ = create_plot_dict(
    dir_ggn, experiments_ggn, x_value="Epoch", y_value=["Train Loss_mean"]
)
passes_ggn_epoch, fails_ggn_epoch = get_passes_and_fails(full_dict_ggn_epoch)

x_list_ggn_epoch, pass_dict_ggn_epoch, labels_dict_ggn = create_plot_dict(
    dir_ggn,
    passes_ggn_epoch,
    x_value="Epoch",
    y_value=["Train Loss_mean", "Train Acc_mean", "Val Loss_mean", "Val Acc_mean"],
)
rankings_ggn_epoch = rank_by_min_val_loss(pass_dict_ggn_epoch, 1)
fig, axes = plt.subplots(
    figsize=(3.25 * 2, 2.0086104634371584 * 2), nrows=2, ncols=2
)  # Create fig and axes

k = 10

ggn_epoch_plots = Plotter(
    # showlegend=True,
    figsize=(8, 4),
    xlabel="Epochs",
    fontsize=14,
    markersize=10,
    alpha=0.6,
    xlim=(0, x_list_ggn_epoch.iloc[-1]),
    # ylim=(0,100),
    conference="icml",
)

rankings_subset_ggn_epoch = OrderedDict(list(rankings_ggn_epoch.items())[:k])

labels_ggn_epoch = {item: item for item in rankings_subset_ggn_epoch.keys()}

(
    train_loss_ggn_epoch,
    train_acc_ggn_epoch,
    val_loss_ggn_epoch,
    val_acc_ggn_epoch,
) = extract_columns_to_dict(rankings_subset_ggn_epoch)

for (x_ind, y_ind), ax in np.ndenumerate(axes):
    if x_ind == 0 and y_ind == 0:
        ggn_epoch_plots.new_plot(fig=fig, ax=ax, yscale="log", ylabel="Train Loss")
        ggn_epoch_plots.iter_plot(
            x_list_ggn_epoch,
            train_loss_ggn_epoch,
            label=labels_ggn_epoch,
        )
        # ax.legend()
    if x_ind == 0 and y_ind == 1:
        ggn_epoch_plots.new_plot(
            fig=fig, ax=ax, yscale="linear", ylabel="Train Accuracy"
        )
        ggn_epoch_plots.iter_plot(
            x_list_ggn_epoch,
            train_acc_ggn_epoch,
            label=labels_ggn_epoch,
        )
        pos = ax.get_position()
        # ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
        ax.legend(loc="center right", bbox_to_anchor=(1.25, 0.5))
    if x_ind == 1 and y_ind == 0:
        ggn_epoch_plots.new_plot(fig=fig, ax=ax, yscale="log", ylabel="Test Loss")
        ggn_epoch_plots.iter_plot(
            x_list_ggn_epoch,
            val_loss_ggn_epoch,
            label=labels_ggn_epoch,
        )
    if x_ind == 1 and y_ind == 1:
        ggn_epoch_plots.new_plot(
            fig=fig, ax=ax, yscale="linear", ylabel="Test Accuracy"
        )
        ggn_epoch_plots.iter_plot(
            x_list_ggn_epoch,
            val_acc_ggn_epoch,
            label=labels_ggn_epoch,
        )

ggn_epoch_plots.savefig("ggn_tuning_epoch.png")


## Comparison of Best Runs
top_sgd_run = OrderedDict([list(rankings_sgd_epoch.items())[0]])
top_adam_run = OrderedDict([list(rankings_adam_epoch.items())[0]])
top_ggn_run = OrderedDict([list(rankings_ggn_epoch.items())[0]])

labels_sgd_top = {item: item for item in top_sgd_run.keys()}
labels_adam_top = {item: item for item in top_adam_run.keys()}
labels_ggn_top = {item: item for item in top_ggn_run.keys()}
(
    train_loss_sgd_top,
    train_acc_sgd_top,
    val_loss_sgd_top,
    val_acc_sgd_top,
) = extract_columns_to_dict(top_sgd_run)

(
    train_loss_adam_top,
    train_acc_adam_top,
    val_loss_adam_top,
    val_acc_adam_top,
) = extract_columns_to_dict(top_adam_run)

(
    train_loss_ggn_top,
    train_acc_ggn_top,
    val_loss_ggn_top,
    val_acc_ggn_top,
) = extract_columns_to_dict(top_ggn_run)

fig, axes = plt.subplots(
    figsize=(3.25 * 2, 2.0086104634371584 * 2), nrows=2, ncols=2
)  # Create fig and axes

k = 10

top_epoch_plots = Plotter(
    # showlegend=True,
    figsize=(8, 4),
    xlabel="Epochs",
    fontsize=14,
    markersize=10,
    alpha=0.6,
    xlim=(0, x_list_ggn_epoch.iloc[-1]),
    # ylim=(0,100),
    conference="icml",
)

for (x_ind, y_ind), ax in np.ndenumerate(axes):
    if x_ind == 0 and y_ind == 0:
        top_epoch_plots.new_plot(fig=fig, ax=ax, yscale="log", ylabel="Train Loss")
        top_epoch_plots.iter_plot(
            x_list_ggn_epoch,
            train_loss_ggn_top,
            label=labels_ggn_top,
        )
        top_epoch_plots.iter_plot(
            x_list_ggn_epoch,
            train_loss_adam_top,
            label=labels_adam_top,
        )
        top_epoch_plots.iter_plot(
            x_list_ggn_epoch,
            train_loss_sgd_top,
            label=labels_sgd_top,
        )
        # ax.legend()
    if x_ind == 0 and y_ind == 1:
        top_epoch_plots.new_plot(
            fig=fig, ax=ax, yscale="linear", ylabel="Train Accuracy"
        )
        top_epoch_plots.iter_plot(
            x_list_ggn_epoch,
            train_acc_ggn_top,
            label=labels_ggn_top,
        )
        top_epoch_plots.iter_plot(
            x_list_ggn_epoch,
            train_acc_adam_top,
            label=labels_adam_top,
        )
        top_epoch_plots.iter_plot(
            x_list_ggn_epoch,
            train_acc_sgd_top,
            label=labels_sgd_top,
        )
        pos = ax.get_position()
        # ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
        ax.legend(loc="center right", bbox_to_anchor=(1.25, 0.5))
    if x_ind == 1 and y_ind == 0:
        top_epoch_plots.new_plot(fig=fig, ax=ax, yscale="log", ylabel="Test Loss")
        top_epoch_plots.iter_plot(
            x_list_ggn_epoch,
            val_loss_ggn_top,
            label=labels_ggn_top,
        )
        top_epoch_plots.iter_plot(
            x_list_ggn_epoch,
            val_loss_adam_top,
            label=labels_adam_top,
        )
        top_epoch_plots.iter_plot(
            x_list_ggn_epoch,
            val_loss_sgd_top,
            label=labels_sgd_top,
        )
    if x_ind == 1 and y_ind == 1:
        top_epoch_plots.new_plot(
            fig=fig, ax=ax, yscale="linear", ylabel="Test Accuracy"
        )
        top_epoch_plots.iter_plot(
            x_list_ggn_epoch,
            val_acc_ggn_top,
            label=labels_ggn_top,
        )
        top_epoch_plots.iter_plot(
            x_list_ggn_epoch,
            val_acc_adam_top,
            label=labels_adam_top,
        )
        top_epoch_plots.iter_plot(
            x_list_ggn_epoch,
            val_acc_sgd_top,
            label=labels_sgd_top,
        )

top_epoch_plots.savefig("comparison_best.png")


x_list_ggn_epoch_time, _, _ = create_plot_dict(
    dir_ggn,
    passes_ggn_epoch,
    x_value="Time_mean",
    y_value=["Train Loss_mean"],
    per_epoch=True,
)

x_list_adam_epoch_time, _, _ = create_plot_dict(
    dir_adam,
    passes_adam_epoch,
    x_value="Time_mean",
    y_value=["Train Loss_mean"],
    per_epoch=True,
)

x_list_sgd_epoch_time, _, _ = create_plot_dict(
    dir_sgd,
    passes_sgd_epoch,
    x_value="Time_mean",
    y_value=["Train Loss_mean"],
    per_epoch=True,
)

fig, axes = plt.subplots(
    figsize=(3.25 * 2, 2.0086104634371584 * 2), nrows=2, ncols=2
)  # Create fig and axes

k = 10

top_epoch_plots = Plotter(
    # showlegend=True,
    figsize=(8, 4),
    xlabel="Wall clock Time (s)",
    fontsize=14,
    markersize=10,
    alpha=0.6,
    # xlim=(0, x_list_ggn_epoch_time.cumsum().iloc[-1]),
    # ylim=(0,100),
    conference="icml",
)


for (x_ind, y_ind), ax in np.ndenumerate(axes):
    if x_ind == 0 and y_ind == 0:
        top_epoch_plots.new_plot(fig=fig, ax=ax, yscale="log", ylabel="Train Loss")
        top_epoch_plots.iter_plot(
            x_list_ggn_epoch_time.cumsum(),
            train_loss_ggn_top,
            label=labels_ggn_top,
        )
        top_epoch_plots.iter_plot(
            x_list_adam_epoch_time.cumsum(),
            train_loss_adam_top,
            label=labels_adam_top,
        )
        top_epoch_plots.iter_plot(
            x_list_sgd_epoch_time.cumsum(),
            train_loss_sgd_top,
            label=labels_sgd_top,
        )
        # ax.legend()
    if x_ind == 0 and y_ind == 1:
        top_epoch_plots.new_plot(
            fig=fig, ax=ax, yscale="linear", ylabel="Train Accuracy"
        )
        top_epoch_plots.iter_plot(
            x_list_ggn_epoch_time.cumsum(),
            train_acc_ggn_top,
            label=labels_ggn_top,
        )
        top_epoch_plots.iter_plot(
            x_list_adam_epoch_time.cumsum(),
            train_acc_adam_top,
            label=labels_adam_top,
        )
        top_epoch_plots.iter_plot(
            x_list_sgd_epoch_time.cumsum(),
            train_acc_sgd_top,
            label=labels_sgd_top,
        )
        pos = ax.get_position()
        # ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
        ax.legend(loc="center right", bbox_to_anchor=(1.25, 0.5))
    if x_ind == 1 and y_ind == 0:
        top_epoch_plots.new_plot(fig=fig, ax=ax, yscale="log", ylabel="Test Loss")
        top_epoch_plots.iter_plot(
            x_list_ggn_epoch_time.cumsum(),
            val_loss_ggn_top,
            label=labels_ggn_top,
        )
        top_epoch_plots.iter_plot(
            x_list_adam_epoch_time.cumsum(),
            val_loss_adam_top,
            label=labels_adam_top,
        )
        top_epoch_plots.iter_plot(
            x_list_sgd_epoch_time.cumsum(),
            val_loss_sgd_top,
            label=labels_sgd_top,
        )
    if x_ind == 1 and y_ind == 1:
        top_epoch_plots.new_plot(
            fig=fig, ax=ax, yscale="linear", ylabel="Test Accuracy"
        )
        top_epoch_plots.iter_plot(
            x_list_ggn_epoch_time.cumsum(),
            val_acc_ggn_top,
            label=labels_ggn_top,
        )
        top_epoch_plots.iter_plot(
            x_list_adam_epoch_time.cumsum(),
            val_acc_adam_top,
            label=labels_adam_top,
        )
        top_epoch_plots.iter_plot(
            x_list_sgd_epoch_time.cumsum(),
            val_acc_sgd_top,
            label=labels_sgd_top,
        )

top_epoch_plots.savefig("comparison_best_time.png")
