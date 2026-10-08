"""Plot per-method L2 error histograms from a per-seed errors JSON file.

The JSON has a title, the seed list, and one error list per method (in seed
order, null for a seed that did not finish), in the order to be plotted:

    {"title": "Experiment 1", "seeds": [0, 1, ...],
     "errors": {"Algorithm 1": [0.41, null, ...], ...}}

All panels share bins and x/y limits. Run, for example:

    python plot_histograms.py experiment1_errors.json
"""

import argparse
import json
from math import ceil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, MultipleLocator

# Fixed colors keep each method's color the same across figures.
COLORS = {
    "Algorithm 1": "#4477aa",
    "Random MoM (tilde J=1000)": "#228844",
    "Random trimmed mean (tilde J=1000)": "#cc5577",
    "Random MoM (tilde J=100)": "#555599",
    "Random trimmed mean (tilde J=100)": "#bb6633",
    "Random MoM (tilde J=50)": "#007788",
    "Random trimmed mean (tilde J=50)": "#aa4499",
    "Random MoM (tilde J=20)": "#882255",
    "Random trimmed mean (tilde J=20)": "#ddaa33",
    "Random MoM (tilde J=10)": "#332288",
    "Random trimmed mean (tilde J=10)": "#44aa99",
    "Coordinate-wise MoM": "#ee9933",
    "Geometric MoM + HT": "#aa3377",
    "Sample mean": "#228833",
    "Sample mean + HT": "#228833",
    "Brute-force": "#ccbb44",
    "Projected Algorithm 1 (LS)": "#66ccee",
    "Projected Algorithm 1 (max)": "#ee6677",
    "Haar-projection LS": "#007788",
    "Haar-projection max": "#994455",
    "Haar-projection LS (r=10)": "#007788",
    "Haar-projection max (r=10)": "#994455",
    "Haar-projection LS (r=5)": "#332288",
    "Haar-projection max (r=5)": "#ddaa33",
    "Haar-projection LS (r=10, J=10)": "#007788",
    "Haar-projection max (r=10, J=10)": "#994455",
    "Haar-projection LS (r=5, J=10)": "#332288",
    "Haar-projection max (r=5, J=10)": "#ddaa33",
    "Haar-projection LS (r=5, J=20)": "#44aa99",
    "Haar-projection max (r=5, J=20)": "#aa4499",
}
FALLBACK_COLOR = "#bbbbbb"


def plot_histograms(title, seeds, errors, output, *, bin_width=0.1, xmax=None, ymax=None, columns=None):
    """Save one histogram panel per method to output.

    xmax defaults to the largest error rounded up to a multiple of 0.5 and is
    extended to a whole number of bins. ymax defaults to the smallest even
    number above the tallest bar. columns defaults to one row for up to four
    methods, otherwise two rows.
    """
    values = {name: [e for e in errs if e is not None] for name, errs in errors.items()}
    finished = [e for errs in values.values() for e in errs]
    if not finished:
        raise ValueError("no finished runs to plot")
    if bin_width <= 0:
        raise ValueError("bin_width must be positive")
    if xmax is None:
        xmax = max(0.5, 0.5 * ceil(max(finished) / 0.5))
    edges = bin_width * np.arange(ceil(xmax / bin_width - 1e-9) + 1)
    if max(finished) > edges[-1]:
        raise ValueError(f"xmax={xmax} is below the largest error {max(finished):.4g}")
    tallest = max(np.histogram(errs, edges)[0].max() for errs in values.values() if errs)
    ymax = 2 * ceil((tallest + 1) / 2) if ymax is None else ymax

    n = len(values)
    columns = columns or (n if n <= 4 else ceil(n / 2))
    rows = ceil(n / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.2 * rows + 0.5),
                             sharey=True, squeeze=False)
    for ax, (name, errs) in zip(axes.flat, values.items()):
        ax.hist(errs, edges, color=COLORS.get(name, FALLBACK_COLOR), edgecolor="white")
        ax.set_title(f"{name.replace('tilde J=1000', 'J=1000')} ({len(errs)}/{len(seeds)})", fontsize=10)
        ax.set_xlim(0, edges[-1])
        ax.set_ylim(0, ymax)
        ax.set_xlabel("L2 error")
        ax.xaxis.set_major_locator(MultipleLocator(max(2 * bin_width, 0.5 * ceil(edges[-1] / 4))))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="y", color="#efefef")
        ax.set_axisbelow(True)
    for ax in axes[:, 0]:
        ax.set_ylabel("Count")
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("errors_json", type=Path, help="per-seed errors JSON file")
    parser.add_argument("--output", type=Path,
                        help="PNG path (default: <name>_error_histograms.png for <name>_errors.json)")
    parser.add_argument("--bin-width", type=float, default=0.1, help="histogram bin width; ticks every two bins (default: 0.1)")
    parser.add_argument("--xmax", type=float, help="x-axis upper limit")
    parser.add_argument("--ymax", type=float, help="y-axis upper limit")
    parser.add_argument("--columns", type=int, help="panels per row")
    args = parser.parse_args()
    with open(args.errors_json) as file:
        results = json.load(file)
    output = args.output or args.errors_json.with_name(
        args.errors_json.stem.removesuffix("_errors") + "_error_histograms.png")
    plot_histograms(results["title"], results["seeds"], results["errors"], output,
                    bin_width=args.bin_width, xmax=args.xmax, ymax=args.ymax, columns=args.columns)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
