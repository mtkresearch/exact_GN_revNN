from tueplots import bundles
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

import matplotlib.font_manager as font_manager


class Plotter(object):
    """
    Class that implements thin matplotlib wrapper for easy, reusable plotting
    """

    def __init__(self, *args, conference=None, **kwargs):
        """
        Arguments
        =========
        *args : Support for plot(y), plot(x, y), plot(x, y, 'b-o'). x, y and
                format string are passed through for plotting

        **kwargs: All kwargs are optional
          Plot Parameters:
          ----------------
            fig : figure instance for drawing plots
            ax : axes instance for drawing plots (If user wants to supply axes,
                 figure externally, both ax and fig must be supplied together)
            figSize : tuple of integers ~ width & height in inches
            dpi : dots per inch setting for figure
            label : Label for line plot as determined by *args, string
            color / c : Color of line plot, overrides format string in *args if
                        supplied. Accepts any valid matplotlib color
            linewidth / lw : Plot linewidth
            linestyle / ls : Plot linestyle ['-','--','-.',':','None',' ','']
            marker : '+', 'o', '*', 's', 'D', ',', '.', '<', '>', '^', '1', '2'
            markerfacecolor / mfc : Face color of marker
            markeredgewidth / mew :
            markeredgecolor / mec :
            markersize / ms : Size of markers
            markevery / mev : Mark every Nth marker
                              [None|integer|(startind, stride)]
            alpha : Opacity of line plot (0 - 1.0), default = 1.0
            title : Plot title, string
            xlabel : X-axis label, string
            ylabel : Y-axis label, string
            xlim : X-axis limits - tuple. eg: xlim=(0,10). Set to None for auto
            ylim : Y-axis limits - tuple. eg: ylim=(0,10). Set to None for auto
            xscale : Set x axis scale ['linear'|'log'|'symlog']
            yscale : Set y axis scale ['linear'|'log'|'symlog']
                Only supports basic xscale/yscale functionality. Use
                get_axes().set_xscale() if further customization is required
            grid : Display axes grid. ['on'|'off']. See grid() for more options
            colorcycle / cs: Set plot colorcycle to list of valid matplotlib
                             colors
            fontsize : Global fontsize for all plots

          Legend Parameters:
          ------------------
            showlegend : set to True to display legend
            fancybox : True by default. Enables rounded corners for legend box
            framealpha : Legend box opacity (0 - 1.0), default = 1.0
            loc : Location of legend box in plot, default = 'best'
            numpoints : number of markers in legend, default = 1.0
            ncol : number of columns for legend. default is 1
            markerscale : The relative size of legend markers vs. original.
                          If None, use rc settings.
            mode : if mode is “expand”, the legend will be horizontally
                   expanded to fill the axes area (or bbox_to_anchor)
            bbox_to_anchor : The bbox that the legend will be anchored. Tuple of
                             2 or 4 floats
        """
        if conference == "icml":
            mpl.rcParams.update(
                {
                    "font.family": "serif",
                    "text.latex.preamble": "\\usepackage{times} ",
                    "figure.figsize": (3.25, 2.0086104634371584),
                    "figure.constrained_layout.use": True,
                    "figure.autolayout": False,
                    "savefig.bbox": "tight",
                    "savefig.pad_inches": 0.015,
                    "font.size": 8,
                    "axes.labelsize": 8,
                    "legend.fontsize": 6,
                    "xtick.labelsize": 6,
                    "ytick.labelsize": 6,
                    "axes.titlesize": 8,
                }
            )

            # mpl.rcParams.update(bundles.icml2022(family="serif", usetex=False))

        self._default_kwargs = {
            "fig": None,
            "ax": None,
            "figsize": None,
            "dpi": mpl.rcParams["figure.dpi"],
            "showlegend": False,
            "fancybox": True,
            "loc": "best",
            "numpoints": 1,
            "yscale": "linear",
            "shaded": False,
        }
        # Dictionary of plot parameter aliases
        self.alias_dict = {
            "lw": "linewidth",
            "ls": "linestyle",
            "mfc": "markerfacecolor",
            "mew": "markeredgewidth",
            "mec": "markeredgecolor",
            "ms": "markersize",
            "mev": "markevery",
            "c": "color",
            "fs": "fontsize",
            "boldlegend": "legend.fontweight",
        }

        # List of all named plot parameters passable to plot method
        self._plot_kwargs = [
            "label",
            "linewidth",
            "linestyle",
            "marker",
            "markerfacecolor",
            "markeredgewidth",
            "markersize",
            "markeredgecolor",
            "markevery",
            "alpha",
            "color",
        ]
        self._plot_fillbetween_kwargs = [
            "label",
            "linewidth",
            "linestyle",
            "marker",
            "markerfacecolor",
            "markeredgewidth",
            "markeredgecolor",
            "markevery",
            "alpha",
            "color",
        ]
        self._legend_kwargs = [
            "fancybox",
            "loc",
            "framealpha",
            "numpoints",
            "ncol",
            "markerscale",
            "mode",
            "bbox_to_anchor",
        ]
        # Parameters that should only be passed to the plot once, then reset
        self._uniqueparams = ["color", "label", "marker", "linestyle", "colorcycle"]
        self._colorcycle = []
        # Mapping between plot parameter and corresponding axes function to call
        self._ax_funcs = {
            "xlabel": "set_xlabel",
            "ylabel": "set_ylabel",
            "xlim": "set_xlim",
            "ylim": "set_ylim",
            "title": "set_title",
            "colorcycle": "set_color_cycle",
            "grid": "grid",
            "xscale": "set_xscale",
            "yscale": "set_yscale",
        }

        self.kwargs = self._default_kwargs.copy()  # Prevent mutating dictionary
        self.args = []
        self.line_list = []  # List of all Line2D items that are plotted
        self.add_plot(*args, **kwargs)

    def add_plot(self, *args, **kwargs):
        """
        Add plot using supplied parameters and existing instance parameters

        Creates new Figure and Axes object if 'fig' and 'ax' parameters not
        supplied. Stores references to all Line2D objects plotted in
        self.line_list.
        Arguments
        =========
            *args : Supports format plot(y), plot(x, y), plot(x, y, 'b-'). x, y
                    and format string are passed through for plotting
            **kwargs : Plot parameters. Refer to __init__ docstring for details
        """
        self._update(*args, **kwargs)

        # Create figure and axes if needed
        if self.kwargs["fig"] is None:
            if not self.isnewargs:
                return  # Don't create fig, ax yet if no x, y data provided
            self.kwargs["fig"] = plt.figure(
                figsize=self.kwargs["figsize"], dpi=self.kwargs["dpi"]
            )
            self.kwargs["ax"] = self.kwargs["fig"].gca()
            self.kwargs["fig"].add_axes(self.kwargs["ax"])

        ax, fig = self.kwargs["ax"], self.kwargs["fig"]

        if self.kwargs["yscale"] != "log":
            ax.ticklabel_format(useOffset=False)  # Prevent offset notation in plots

        # Apply axes functions if present in kwargs
        for kwarg in self.kwargs:
            if kwarg in self._ax_funcs:
                # eg: f = getattr(ax,'set_title'); f('new title')
                func = getattr(ax, self._ax_funcs[kwarg])
                func(self.kwargs[kwarg])

        # Add plot only if new args passed to this instance
        if self.isnewargs:
            # Create updated name, value dict to pass to plot method
            plot_kwargs = {
                kwarg: self.kwargs[kwarg]
                for kwarg in self._plot_kwargs
                if kwarg in self.kwargs
            }

            (line,) = ax.plot(*self.args, **plot_kwargs)
            self.line_list.append(line)

        # Display legend if required
        if self.kwargs["showlegend"]:
            legend_kwargs = {
                kwarg: self.kwargs[kwarg]
                for kwarg in self._legend_kwargs
                if kwarg in self.kwargs
            }
            leg = ax.legend(**legend_kwargs)

        if "fontsize" in self.kwargs:
            self.set_fontsize(self.kwargs["fontsize"])

        self._delete_uniqueparams()  # Clear unique parameters from kwargs list

        if plt.isinteractive():  # Only redraw canvas in interactive mode
            self.redraw()

    def add_fillbetween_plot(self, *args, **kwargs):
        """
        Add plot using supplied parameters and existing instance parameters

        Creates new Figure and Axes object if 'fig' and 'ax' parameters not
        supplied. Stores references to all Line2D objects plotted in
        self.line_list.
        Arguments
        =========
            *args : Supports format plot(y), plot(x, y), plot(x, y, 'b-'). x, y
                    and format string are passed through for plotting
            **kwargs : Plot parameters. Refer to __init__ docstring for details
        """
        self._update(*args, **kwargs)

        # Create figure and axes if needed
        if self.kwargs["fig"] is None:
            if not self.isnewargs:
                return  # Don't create fig, ax yet if no x, y data provided
            self.kwargs["fig"] = plt.figure(
                figsize=self.kwargs["figsize"], dpi=self.kwargs["dpi"]
            )
            self.kwargs["ax"] = self.kwargs["fig"].gca()
            self.kwargs["fig"].add_axes(self.kwargs["ax"])

        ax, fig = self.kwargs["ax"], self.kwargs["fig"]

        if self.kwargs["yscale"] != "log":
            ax.ticklabel_format(useOffset=False)  # Prevent offset notation in plots

        # Apply axes functions if present in kwargs
        for kwarg in self.kwargs:
            if kwarg in self._ax_funcs:
                # eg: f = getattr(ax,'set_title'); f('new title')
                func = getattr(ax, self._ax_funcs[kwarg])
                func(self.kwargs[kwarg])

        # Add plot only if new args passed to this instance
        if self.isnewargs:
            # Create updated name, value dict to pass to plot method
            plot_kwargs = {
                kwarg: self.kwargs[kwarg]
                for kwarg in self._plot_fillbetween_kwargs
                if kwarg in self.kwargs
            }

            line = ax.fill_between(*self.args, **plot_kwargs)
            self.line_list.append(line)

        # Display legend if required
        if self.kwargs["showlegend"]:
            legend_kwargs = {
                kwarg: self.kwargs[kwarg]
                for kwarg in self._legend_kwargs
                if kwarg in self.kwargs
            }
            leg = ax.legend(**legend_kwargs)

        if "fontsize" in self.kwargs:
            self.set_fontsize(self.kwargs["fontsize"])

        self._delete_uniqueparams()  # Clear unique parameters from kwargs list

        if plt.isinteractive():  # Only redraw canvas in interactive mode
            self.redraw()

    def update_plot(self, **kwargs):
        """ "Update plot parameters (keyword arguments) and replot figure

        Usage:
            a = Plotter([1,2,3], [2,4,8], 'r-o', label='label 1')
            # Update title and xlabel string and redraw plot
            a.update_plot(title='Title', xlabel='xlabel')
        """
        self.add_plot(**kwargs)

    def new_plot(self, *args, **kwargs):
        """
        Plot new plot using Plotter object and default plot parameters

        Pass a named argument reset=True if all plotting parameters should
        be reset to original defaults
        """
        reset = kwargs["reset"] if "reset" in kwargs else False
        self._reset(reset=reset)
        if self._colorcycle:
            self.kwargs["colorcycle"] = self._colorcycle
        self.add_plot(*args, **kwargs)

    def iter_plot(self, x, y, mode="dict", **kwargs):
        """
        Plot multiple plots by iterating through x, y and parameter lists

        Arguments:
        ==========
          x : x values. 1D List/Array, Dictionary or Numpy 2D Array
          y : y values. Dictionary or 2D Python array (List of Lists where each
              sub-list is one set of y-data) or Numpy 2D Array
          mode : y, labels and other parameters should either be a Dictionary
                 or a 2D Numpy array/2D List where each row corresponds to a
                 single plot ['dict'|'array']
          **kwargs : Plot params as defined in __init__ documentation.
             Params can either be:
               scalars (same value applied to all plots),
               dictionaries (mode='dict', key[val] value applies to each plot)
               1D Lists/Numpy Arrays (mode='array', param[index] applies to each
               plot)
        """
        if mode.lower() == "dict":
            for key in y:
                loop_kwargs = {}
                for kwarg in kwargs:
                    try:  # Check if parameter is a dictionary
                        loop_kwargs[kwarg] = kwargs[kwarg][key]
                    except:
                        loop_kwargs[kwarg] = kwargs[kwarg]
                try:
                    x_loop = x[key]
                except:
                    x_loop = x
                self.add_plot(x_loop, y[key], **loop_kwargs)

        elif mode.lower() == "array":
            for ind in range(len(y)):
                loop_kwargs = {}
                for kwarg in kwargs:
                    # Do not iterate through tuple/string plot parameters
                    if isinstance(kwargs[kwarg], (str, tuple)):
                        loop_kwargs[kwarg] = kwargs[kwarg]
                    else:
                        try:  # Check if parameter is a 1-D List/Array
                            loop_kwargs[kwarg] = kwargs[kwarg][ind]
                        except:
                            loop_kwargs[kwarg] = kwargs[kwarg]
                try:
                    x_loop = x[ind][:]
                except:
                    x_loop = x
                self.add_plot(x_loop, y[ind], **loop_kwargs)
        else:
            print("Error! Incorrect mode specification. Ignoring method call")

    def autoscale(self, enable=True, axis="both", tight=None):
        """Autoscale the axis view to the data (toggle).

        Convenience method for simple axis view autoscaling. It turns
        autoscaling on or off, and then, if autoscaling for either axis is on,
        it performs the autoscaling on the specified axis or axes.

        Arguments
        =========
        enable: [True | False | None]
        axis: ['x' | 'y' | 'both']
        tight: [True | False | None]
        """
        ax = self.get_axes()
        ax.autoscale(enable=enable, axis=axis, tight=tight)
        # Reset xlim and ylim parameters to None if previously set to some value
        if "xlim" in self.kwargs and (axis == "x" or axis == "both"):
            self.kwargs.pop("xlim")
        if "ylim" in self.kwargs and (axis == "y" or axis == "both"):
            self.kwargs.pop("ylim")
        self.redraw()

    def grid(self, **kwargs):
        """Turn axes grid on or off

        Call signature: grid(self, b=None, which='major', axis='both', **kwargs)
        **kwargs are passed to linespec of grid lines (eg: linewidth=2)
        """
        self.get_axes().grid(**kwargs)
        self.redraw()

    def get_figure(self):
        """Returns figure instance of current plot"""
        return self.kwargs["fig"]

    def get_axes(self):
        """Returns axes instance for current plot"""
        return self.kwargs["ax"]

    def redraw(self):
        """
        Redraw plot. Use after custom user modifications of axes & fig objects
        """
        if plt.isinteractive():
            fig = self.kwargs["fig"]
            # Redraw figure if it was previously closed prior to updating it
            if not plt.fignum_exists(fig.number):
                fig.show()
            fig.canvas.draw()
        else:
            # print("redraw() is unsupported in non-interactive plotting mode!")
            pass

    def set_fontsize(self, font_size):
        """Updates global font size for all plot elements"""
        mpl.rcParams["font.size"] = font_size
        self.redraw()
        # TODO: Implement individual font size setting

    #        params = {'font.family': 'serif',
    #          'font.size': 16,
    #          'axes.labelsize': 18,
    #          'text.fontsize': 18,
    #          'legend.fontsize': 18,
    #          'xtick.labelsize': 18,
    #          'ytick.labelsize': 18,
    #          'text.usetex': True}
    #        mpl.rcParams.update(params)

    #    def set_font(self, family=None, weight=None, size=None):
    #        """ Updates global font properties for all plot elements
    #
    #        TODO: Font family and weight don't update dynamically"""
    #        if family is None:
    #            family = mpl.rcParams['font.family']
    #        if weight is None:
    #            weight = mpl.rcParams['font.weight']
    #        if size is None:
    #            size = mpl.rcParams['font.size']
    #        mpl.rc('font', family=family, weight=weight, size=size)
    #        self.redraw()

    def _delete_uniqueparams(self):
        """Delete plot parameters that are unique per plot

        Prevents unique parameters (eg: label) carrying over to future plots"""
        # Store colorcycle list prior to deleting from this instance
        if "colorcycle" in self.kwargs:
            self._colorcycle = self.kwargs["colorcycle"]

        for param in self._uniqueparams:
            self.kwargs.pop(param, None)

    def _update(self, *args, **kwargs):
        """Update instance variables args and kwargs with supplied values"""
        if args:
            self.args = args  # Args to be directly passed to plot command
            self.isnewargs = True
        else:
            self.isnewargs = False

        # Update self.kwargs with full parameter name of aliased plot parameter
        for alias in self.alias_dict:
            if alias in kwargs:
                self.kwargs[self.alias_dict[alias]] = kwargs.pop(alias)

        # Update kwargs dictionary
        for key in kwargs:
            self.kwargs[key] = kwargs[key]

    def _reset(self, reset=False):
        """Reset instance variables in preparation for new plots
        reset: True if current instance defaults for plotting parameters should
               be reset to Class defaults"""
        self.args = []
        self.line_list = []
        self.kwargs["fig"] = None
        self.kwargs["ax"] = None
        if reset:
            self.kwargs = self._default_kwargs.copy()

    def savefig(self, filename, dir=".", dpi=300):
        print("Saving: ", os.path.join(dir, filename))
        self.kwargs["fig"].savefig(os.path.join(dir, filename), dpi=dpi)


import pandas as pd
import os


def create_df(dir, experiment, per_epoch=False):
    # Define the directory where the experiment folders are located
    base_directory = dir

    # Initialize an empty DataFrame to hold all the results
    all_results_df = pd.DataFrame()

    # Loop over the directories in the base directory
    for folder_name in os.listdir(base_directory):
        # Construct the full folder path
        folder_path = os.path.join(base_directory, folder_name)
        # Check if it is a directory and follows the naming convention 'experiment_A_seedX'
        if os.path.isdir(folder_path) and folder_name.startswith(experiment):
            # Extract the seed number from the folder name
            seed = int(folder_name.split("seed_")[-1])
            # Construct the full path to the progress.csv file
            timestamp = os.listdir(folder_path)[0]
            csv_file_path = (
                os.path.join(folder_path, timestamp, "per_epoch", "progress.csv")
                if per_epoch
                else os.path.join(folder_path, timestamp, "progress.csv")
            )
            # Check if the progress.csv file exists
            if os.path.isfile(csv_file_path):
                # Read the CSV file into a DataFrame
                results_df = pd.read_csv(csv_file_path)
                # Add a suffix to each column name with the seed number, except for any columns you want to exclude
                results_df = results_df.add_suffix(f"_seed{seed}")
                # If there are columns that should not have the suffix, rename them back
                # For example, if 'epoch' should not have the suffix, uncomment the next line
                results_df = results_df.rename(
                    columns={"epoch_seed{0}".format(seed): "epoch"}
                )
                results_df = results_df.rename(
                    columns={"iter_seed{0}".format(seed): "iter"}
                )
                # Append the results to the main DataFrame
                all_results_df = pd.concat([all_results_df, results_df], axis=1)

                all_results_df = all_results_df.loc[
                    :, ~all_results_df.columns.duplicated()
                ].copy()

    # Calculate the mean across seed columns
    # Identify all unique metrics by removing the '_seedX' suffix and excluding 'Epoch' and 'Iter'
    metrics = set(
        col.split("_seed")[0]
        for col in all_results_df.columns
        if "_seed" in col and not col.startswith(("epoch", "iter"))
    )

    # For each metric, calculate the mean across its seed columns and add it to the DataFrame
    for metric in metrics:
        seed_columns = [
            col for col in all_results_df.columns if col.startswith(metric + "_seed")
        ]
        all_results_df[metric + "_mean"] = all_results_df[seed_columns].mean(axis=1)
        all_results_df[metric + "_std"] = all_results_df[seed_columns].std(axis=1)

    # Now all_results_df contains all the data from the CSV files with columns labeled by seed number
    return all_results_df


def get_unique_experiment_prefixes(directory):
    # Initialize a set to hold the unique prefixes
    unique_prefixes = set()

    # Loop over the folder names in the given directory
    for folder_name in os.listdir(directory):
        # Check if it is a directory
        if os.path.isdir(os.path.join(directory, folder_name)):
            # Split the folder name on '_seed' to extract the prefix
            parts = folder_name.split("_seed")
            if len(parts) > 1:
                # Add the prefix to the set of unique prefixes
                unique_prefixes.add(parts[0])

    # Convert the set to a list and return it
    return sorted(list(unique_prefixes))


def get_top_k_runs_per_iteration(metric_dict, iteration):
    """
    This function filters the top k experimental runs based on per iteration loss.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the experimental results.
    loss_column (str): The name of the column containing the per iteration loss values.
    k (int): The number of top runs to select.

    Returns:
    pandas.DataFrame: A DataFrame containing the top k runs.
    """
    df = pd.DataFrame.from_dict(metric_dict)

    idxs = list(df.rank(axis=1, method="min", ascending=True).iloc[iteration])

    return [int(idx - 1) for idx in idxs]


def get_passes_and_fails(train_results):
    fails, passes = [], []

    for name, value in train_results.items():
        if value.isnull().any():
            fails.append(name)
        else:
            passes.append(name)
    return passes, fails


def get_top_k_results(original_dict, indices, k=10):
    # Convert the dictionary to a list of items
    items = list(original_dict.items())

    # Sort the items based on the list of indices
    sorted_items = [items[i] for i in indices][:k]

    # Create a new dictionary from the sorted list of items
    sorted_dict = dict(sorted_items)

    return sorted_dict


def get_top_k_labels(original_labels, indices, k=10):
    # Sort the items based on the list of indices
    sorted_labels = [original_labels[i] for i in indices][:k]
    return sorted_labels


def get_best(train_results):
    fails, passes = [], []

    for name, value in train_results.items():
        if value.isnull().any():
            fails.append(name)
        else:
            passes.append(name)
    return passes, fails


from collections import OrderedDict
import pandas as pd


def rank_by_min_val_loss(data_dict, index):
    # Step 1: Extract the validation loss and the corresponding key into a list
    val_loss_list = [
        (key, df.loc[index, "val_loss_mean"]) for key, df in data_dict.items()
    ]

    # Step 2: Sort the list by the validation loss
    val_loss_list.sort(key=lambda x: x[1])

    # Step 3: Create an ordered dictionary based on the sorted list
    ordered_dict = OrderedDict((key, data_dict[key]) for key, _ in val_loss_list)

    return ordered_dict


def extract_columns_to_dict(ordered_dict):
    train_loss_dict = {}
    train_acc_dict = {}
    val_loss_dict = {}
    val_acc_dict = {}

    for key, df in ordered_dict.items():
        try:
            train_loss_dict[key] = df["train_loss_mean"]
            train_acc_dict[key] = df["train_acc_mean"]
        except KeyError:
            train_loss_dict[key] = df["train_loss_mean"]
            train_acc_dict[key] = df["Train Acc_mean"]

        val_loss_dict[key] = df["val_loss_mean"]
        val_acc_dict[key] = df["val_acc_mean"]

    return train_loss_dict, train_acc_dict, val_loss_dict, val_acc_dict


def get_passes_and_fails(train_results):
    fails, passes = [], []

    for name, value in train_results.items():
        if value.isnull().any().any():
            fails.append(name)
        else:
            passes.append(name)
    return passes, fails


def create_single_plot_dict(
    dir, experiment, values=["train_loss_mean"], label=False, per_epoch=False
):
    val_dict, labels_dict = {}, {}

    df = create_df(dir, experiment, per_epoch=per_epoch)
    val_dict[experiment] = df[values]
    labels_dict[experiment] = label if label else experiment

    return val_dict, labels_dict


def create_plot_dict(
    dir, experiments, x_value="Iter", y_value=["train_loss_mean"], per_epoch=False
):
    y_dict, labels_dict = {}, {}
    x_list = []

    if x_value == "Epoch":
        per_epoch = True

    for experiment in experiments:
        df = create_df(dir, experiment, per_epoch=per_epoch)
        y_dict[experiment] = df[y_value]
        labels_dict[experiment] = experiment
        x_list = df[x_value]
    return x_list, y_dict, labels_dict


def set_y_values(y_value_base):
    if isinstance(y_value_base, tuple):
        y_value_base = y_value_base[0]
    y_value = f"{y_value_base}_mean"
    y_std = (
        f"{y_value_base}_std"
        if y_value_base.startswith("train") or y_value_base.startswith("val")
        else False
    )
    return y_value, y_std


if __name__ == "__main__":
    dir = (
        "/proj/gpu_mtk53548/Fastbreak/research/inv_dense/results/mnist/20240112-202511"
    )

    import numpy as np

    epochs = np.arange(1, 51)  # 50 epochs for example
    sgd_train_acc = np.random.rand(50)

    plotter = Plotter(
        epochs,
        sgd_train_acc,
        label=r"$y = x^2$",
        showlegend=True,
        xlabel="x",
        ylabel="y",
        title="title",
        fontsize=14,
        markersize=10,
        alpha=0.6,
        conference="icml",
    )

    plotter.add_plot(
        epochs,
        0.15 * sgd_train_acc**3,
        label="$y = 0.15x^3$",
        c="c",
        ls="-",
        # marker="D",
        markersize=10,
    )

    plotter.savefig("test.png")
