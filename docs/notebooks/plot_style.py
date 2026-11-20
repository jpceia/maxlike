"""
Shared chart style of the notebooks: one categorical palette used in a fixed
order, a single-hue sequential ramp, a recessive grid and thin marks.
"""
import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
# the notebooks reuse the functions of the example scripts
sys.path.insert(0, os.path.join(HERE, "..", "examples"))
# run from a checkout without installing the package
sys.path.insert(0, os.path.join(HERE, "..", ".."))

# categorical slots, in this order (never cycled past the number needed)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SERIES = [BLUE, ORANGE, AQUA, YELLOW]
SURFACE = "#fcfcfb"
TEXT, TEXT_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# sequential ramp (magnitude), light to dark
SEQUENTIAL = LinearSegmentedColormap.from_list("seq_blue", [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf",
    "#184f95", "#0d366b"])


def apply():
    mpl.rcParams.update({
        "figure.figsize": (7.5, 4.2),
        "figure.dpi": 110,
        "savefig.dpi": 110,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_2,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.size": 10,
        "text.color": TEXT,
        "svg.fonttype": "none",
    })


def finish(ax, title=None, xlabel=None, ylabel=None, legend=False):
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if legend:
        ax.legend(loc="best")
    plt.tight_layout()
