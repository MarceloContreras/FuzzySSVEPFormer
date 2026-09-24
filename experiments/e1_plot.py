"""Plot e1 noise reports for multiple datasets and methods.

Example:
    python experiments/e1_plot.py --results-dir results/e1 \
        --datasets nakanishi utec --methods ssvepformer f1ssvepformer_C \
        --output-dir results/e1/plots

Searches recursively for <method>_<dataset>_Noise.csv and
<method>_<dataset>_Noise_ITR.csv. Reads saved ITR values (bits/min).
Each Time gets a separate figure: metric rows and dataset columns.
Repetitions, including appended runs, are averaged within each subject;
error bands show +/- one standard error across subject means, not seeds.
Incomplete reports are rejected so missing results are not silently averaged.
Style follows plot_quality_noisy.py using native Matplotlib settings (no LaTeX required).
Dependencies: numpy, pandas, matplotlib. No training dependencies are needed.
"""

import argparse
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METADATA = ["Time", "Noise", "Repeat", "NoiseSeed"]
METRICS = {"acc": "Accuracy (\%)", "itr": "ITR (bits/min)"}


def discover_reports(root, datasets, methods=None, metrics=("acc", "itr")):
    """Require exactly one report per selected dataset/method/metric."""
    pattern = re.compile(
        r"(?P<method>.+)_(?P<dataset>nakanishi|utec)_Noise(?P<itr>_ITR)?\.csv",
        re.IGNORECASE,
    )
    found = {}
    for path in sorted(Path(root).rglob("*.csv")):
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        dataset = match["dataset"].lower()
        method = match["method"]
        metric = "itr" if match["itr"] else "acc"
        if dataset not in datasets or metric not in metrics:
            continue
        if methods is not None and method not in methods:
            continue
        key = (dataset, method, metric)
        if key in found:
            raise ValueError(f"Multiple reports for {key}: {found[key]} and {path}")
        found[key] = path
    if not found:
        raise ValueError(f"No matching e1 reports found under {root}.")
    methods = list(methods) if methods is not None else sorted({k[1] for k in found})
    missing = [
        (d, m, v)
        for d in datasets
        for m in methods
        for v in metrics
        if (d, m, v) not in found
    ]
    if missing:
        raise ValueError(
            f"Missing reports: {missing}. Use --methods/--metrics to select available reports."
        )
    return found, methods


def load_report(path, metric):
    df = pd.read_csv(path)
    if df.empty or not set(METADATA).issubset(df.columns):
        raise ValueError(f"{path}: expected nonempty e1 CSV with {METADATA}.")
    subjects = [c for c in df.columns if c not in METADATA]
    if len(subjects) < 2 or not all(c.isdigit() for c in subjects):
        raise ValueError(f"{path}: expected at least two numeric subject columns.")
    df = df.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError(f"{path}: incomplete report or nonfinite values.")
    if not df.Time.gt(0).all() or not df.Noise.ge(0).all():
        raise ValueError(f"{path}: Time must be positive and Noise nonnegative.")
    for column, minimum in (("Repeat", 1), ("NoiseSeed", 0)):
        if not (df[column].ge(minimum) & df[column].mod(1).eq(0)).all():
            raise ValueError(f"{path}: invalid {column} values.")
    values = df[subjects].to_numpy()
    if (values < 0).any() or (metric == "acc" and (values > 100).any()):
        raise ValueError(
            f"{path}: expected {'accuracy in [0, 100]' if metric == 'acc' else 'nonnegative ITR'}."
        )
    return df, subjects


def summarize_reports(reports, times=None, noise_levels=None):
    """Return descriptive statistics and require comparable grids per dataset."""
    summaries = []
    references = {}
    for (dataset, method, metric), path in reports.items():
        df, subjects = load_report(path, metric)
        if times is not None:
            mask = np.zeros(len(df), dtype=bool)
            for time in times:
                selected = np.isclose(df.Time, time, rtol=0, atol=1e-9)
                if not selected.any():
                    raise ValueError(f"{path}: missing requested Time={time}.")
                df.loc[selected, "Time"] = time
                mask |= selected
            df = df.loc[mask].copy()
        if noise_levels is not None:
            mask = np.zeros(len(df), dtype=bool)
            for noise in noise_levels:
                selected = np.isclose(df.Noise, noise, rtol=1e-9, atol=0)
                missing_times = sorted(set(df.Time) - set(df.loc[selected, "Time"]))
                if missing_times:
                    raise ValueError(
                        f"{path}: missing requested Noise={noise} for Time={missing_times}."
                    )
                df.loc[selected, "Noise"] = noise
                mask |= selected
            df = df.loc[mask].copy()
        # Include repeated rows so unequal or unfinished runs cannot be pooled silently.
        signature = (
            tuple(sorted(subjects)),
            sorted(df[METADATA].itertuples(index=False, name=None)),
        )
        if dataset in references and signature != references[dataset]:
            raise ValueError(
                f"{path}: subjects or Time/Noise/repetition grid differs within {dataset}."
            )
        references[dataset] = signature
        long = df.melt(
            id_vars=METADATA,
            value_vars=subjects,
            var_name="subject",
            value_name="value",
        )
        per_subject = long.groupby(["Time", "Noise", "subject"]).value.agg(
            ["mean", "size"]
        )
        grouped = per_subject.groupby(level=["Time", "Noise"])
        summary = (
            grouped["mean"]
            .agg(["mean", "std", "count"])
            .rename(columns={"count": "n_subjects"})
        )
        summary["se"] = summary["std"] / np.sqrt(summary["n_subjects"])
        summary["n_repetitions_min"] = grouped["size"].min()
        summary["n_repetitions_max"] = grouped["size"].max()
        summary = summary.reset_index()
        summary["dataset"], summary["method"], summary["metric"] = (
            dataset,
            method,
            metric,
        )
        summaries.append(summary)
    result = pd.concat(summaries, ignore_index=True)
    windows = [set(group.Time) for _, group in result.groupby("dataset")]
    if any(window != windows[0] for window in windows[1:]):
        raise ValueError(
            "Datasets have different Time windows; select common windows with --times."
        )
    return result


# Science-style typography and ticks, without a SciencePlots/LaTeX dependency.
PLOT_STYLE = {
    "font.family": "serif",
    "font.size": 10,
    "mathtext.fontset": "cm",
    "text.usetex": True,
    "axes.linewidth": 0.5,
    "axes.grid": False,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "legend.frameon": False,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}


def method_style(method, index):
    name = method.lower()
    if name == "ssvepformer":
        return "tab:red", "SSVEPformer"
    for prefix, color, label in (
        ("f1ssvepformer", "tab:green", "SSVEPformer-T1"),
        ("f2ssvepformer", "tab:blue", "SSVEPformer-IT2"),
    ):
        if name == prefix or name.startswith(prefix + "_"):
            variant = method[len(prefix) :].lstrip("_")
            return color, f"{label} ({variant})" if variant else label
    palette = [
        "tab:orange",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]
    return palette[index % len(palette)], method


@plt.rc_context(PLOT_STYLE)
def plot_reports(
    summary, datasets, methods, metrics, output_dir, formats=("png", "pdf"), dpi=300
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    styles = {
        method: method_style(method, index) for index, method in enumerate(methods)
    }
    line_styles = ["-", "--", "-.", ":"]
    color_counts = {}
    method_lines = {}
    for method in methods:
        color = styles[method][0]
        count = color_counts.get(color, 0)
        method_lines[method] = line_styles[count % len(line_styles)]
        color_counts[color] = count + 1
    saved = []
    for time, window in summary.groupby("Time", sort=True):
        fig, axes = plt.subplots(
            len(metrics),
            len(datasets),
            squeeze=False,
            figsize=(4 * len(datasets), 2.5 * len(metrics)),
            sharex="col",
            sharey=False,
        )
        for row, metric in enumerate(metrics):
            for col, dataset in enumerate(datasets):
                ax = axes[row, col]
                subset = window.loc[
                    (window.dataset == dataset) & (window.metric == metric)
                ]
                for index, method in enumerate(methods):
                    data = subset.loc[subset.method == method].sort_values("Noise")
                    x = data.Noise.to_numpy()
                    mean, se = data["mean"].to_numpy(), data.se.to_numpy()
                    color, label = styles[method]
                    ax.plot(
                        x,
                        mean,
                        label=label,
                        color=color,
                        linestyle=method_lines[method],
                        linewidth=1,
                        marker="o" if len(x) == 1 else None,
                        markersize=3,
                    )
                    ax.fill_between(
                        x, mean - se, mean + se, color=color, alpha=0.2, linewidth=0
                    )
                noises = sorted(subset.Noise.unique())
                positive = [n for n in noises if n > 0]
                if len(positive) == len(noises):
                    ax.set_xscale("log")
                else:
                    ax.set_xscale("symlog", linthresh=min(positive) if positive else 1)
                ax.set_xticks(noises, [f"{n:g}" for n in noises], rotation=0)
                ax.set_title(
                    "Nakanishi" if dataset == "nakanishi" else "UTEC",
                    fontstyle="italic",
                )
                ax.set_ylabel(METRICS[metric])
                ax.set_xlabel(r"Noise intensity [$\sigma$]")
                # Fit each panel to its own curves and SE bands with minimal padding.
                ax.margins(x=0, y=0.02)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.tight_layout()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(4, len(methods)),
            bbox_to_anchor=(0.5, 1.0),
            frameon=False,
        )
        for extension in formats:
            path = output_dir / f"e1_noise_{time:g}s.{extension}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            saved.append(path)
        plt.close(fig)
    summary.to_csv(output_dir / "e1_noise_summary.csv", index=False)
    return saved


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Search this folder and its subfolders for e1 CSVs.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=str.lower,
        choices=["nakanishi", "utec"],
        default=["nakanishi", "utec"],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        help="Exact method prefixes from filenames; default: all discovered methods.",
    )
    parser.add_argument("--metrics", nargs="+", choices=METRICS, default=["acc", "itr"])
    parser.add_argument(
        "--times",
        nargs="+",
        type=float,
        help="Signal windows in seconds; default: all.",
    )
    parser.add_argument(
        "--noise-levels",
        nargs="+",
        type=float,
        help="Noise standard deviations to include, e.g. 0 1 10 100; default: all.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/e1_plots"))
    parser.add_argument(
        "--formats", nargs="+", choices=["png", "pdf", "svg"], default=["png", "pdf"]
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive.")
    if args.noise_levels is not None and any(
        not np.isfinite(n) or n < 0 for n in args.noise_levels
    ):
        parser.error("--noise-levels must be finite and nonnegative.")
    for name in ("datasets", "methods", "metrics", "times", "formats", "noise_levels"):
        values = getattr(args, name)
        if values is not None and len(set(values)) != len(values):
            parser.error(f"--{name.replace('_', '-')} must not contain duplicates.")
    try:
        reports, methods = discover_reports(
            args.results_dir, args.datasets, args.methods, args.metrics
        )
        summary = summarize_reports(reports, args.times, args.noise_levels)
        saved = plot_reports(
            summary,
            args.datasets,
            methods,
            args.metrics,
            args.output_dir,
            args.formats,
            args.dpi,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    for path in saved:
        print(f"Saved {path}")
    print(f'Saved {args.output_dir / "e1_noise_summary.csv"}')


if __name__ == "__main__":
    main()
