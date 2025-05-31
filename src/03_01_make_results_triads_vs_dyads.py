from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

import scienceplots

plt.style.use("science")

from utils import get_formatted_p_value_stars, get_formatted_p_value
from parameters import MAX_DELTA_F

plot_parameters = {
    "pairs": {
        "dyads": {
            "color": "#1f77b4",
            "label": "Dyads",
        },
        "triads_1n": {
            "color": "#ff7f0e",
            "label": "Triads ($1^{st}$ neighbor)",
        },
        "triads_2n": {
            "color": "#2ca02c",
            "label": "Triads ($2^{nd}$ neighbor)",
        },
    },
    "metrics": {
        "gsi": {"label": "GSI", "limits": [0, 0.8], "table_label": "GSI"},
        "coherence": {"label": "CWC", "limits": [0, 1], "table_label": "CWC"},
        "delta_f": {
            "label": "$\\Delta f$ [Hz]",
            "limits": [0, 0.4],
            "table_label": "$\\Delta f$ [Hz]",
        },
        "rec": {
            "label": "\\%REC",
            "limits": [0, 1],
            "table_label": "\\%REC",
        },
        "det": {
            "label": "\\%DET",
            "limits": [0.5, 1],
            "table_label": "\\%DET",
        },
        "maxline": {
            "label": "MAXLINE",
            "limits": [0, 1],
            "table_label": "MAXLINE",
        },
        "lyapunov": {
            "label": "$l_{lyap}$ ($\\times 10^{-3}$)",
            "limits": [0, 1],
            "table_label": "$l_{lyap}$",
        },
        "determinism": {
            "label": "$D$",
            "limits": [0.4, 1],
            "table_label": "$D$",
        },
    },
}

if __name__ == "__main__":

    df_triads = pd.read_csv("../data/csv/gait_data_triads.csv")
    df_dyads = pd.read_csv("../data/csv/gait_data_dyads.csv")

    data_dyads = df_dyads[
        (df_dyads["delta_f"] < MAX_DELTA_F) & (df_dyads["interaction"] < 4)
    ]
    data_triads = df_triads[
        (df_triads["delta_f"] < MAX_DELTA_F) & (df_triads["formation"] != "following")
    ]

    data_triads_first_neighbors = data_triads[
        data_triads["pair"].isin(["left_center", "right_center"])
    ]
    data_triads_second_neighbors = data_triads[(data_triads["pair"] == "left_right")]

    all_values = {}
    metrics = [
        "delta_f",
        "gsi",
        "coherence",
        "rec",
        "det",
        "maxline",
        "determinism",
        "lyapunov",
    ]

    for metric in metrics:

        all_values[metric] = {}

        means = []
        stds = []
        stes = []
        values_for_p_values = []

        fig, ax = plt.subplots(figsize=(5, 3))

        for i, (label, data) in enumerate(
            zip(
                ["dyads", "triads_1n", "triads_2n"],
                [
                    data_dyads,
                    data_triads_first_neighbors,
                    data_triads_second_neighbors,
                ],
            )
        ):
            if metric in ["determinism", "lyapunov"]:
                values_1 = data[metric + "_1"].values
                values_2 = data[metric + "_2"].values
                values_metric = np.concatenate([values_1, values_2])
            else:
                values_metric = data[metric].values

            # remove NaN values
            values_metric = values_metric[~np.isnan(values_metric)]

            all_values[metric][label] = values_metric

            values_for_p_values.append(values_metric)

            mean = np.mean(values_metric)
            std = np.std(values_metric)
            ste = np.std(values_metric) / np.sqrt(len(values_metric))

            means.append(mean)
            stds.append(std)
            stes.append(ste)

            ax.bar(
                i,
                mean,
                yerr=ste,
                capsize=5,
                color=plot_parameters["pairs"][label]["color"],
            )

        ax.set_xticks(
            np.arange(len(plot_parameters["pairs"])),
            labels=[
                plot_parameters["pairs"]["dyads"]["label"],
                plot_parameters["pairs"]["triads_1n"]["label"],
                plot_parameters["pairs"]["triads_2n"]["label"],
            ],
        )
        # p_values

        _, p_value_dyads_triads_1n = ttest_ind(
            values_for_p_values[0],
            values_for_p_values[1],
        )
        _, p_value_dyads_triads_2n = ttest_ind(
            values_for_p_values[0],
            values_for_p_values[2],
        )

        max_val = max(means)
        y = max_val * 1.1
        dh = max_val * 0.05

        # brackets for dyads vs triads_1n
        ax.plot([0, 0, 1, 1], [y, y + dh, y + dh, y], color="gray", linewidth=1.5)
        ax.text(
            0.5,
            y + dh * 3,
            get_formatted_p_value_stars(p_value_dyads_triads_1n),
            ha="center",
            va="center",
            color="gray",
        )
        y = y + dh * 4
        # brackets for dyads vs triads_2n
        ax.plot([0, 0, 2, 2], [y, y + dh, y + dh, y], color="gray", linewidth=1.5)
        ax.text(
            1,
            y + dh * 3,
            get_formatted_p_value_stars(p_value_dyads_triads_2n),
            ha="center",
            va="center",
            color="gray",
        )

        if metric == "lyapunov":
            # format the axis to show only 1 decimal and put exponent in the axis label
            def scale_formatter(value, _):
                return f"{value * 1e3:.1f}"  # Multiply by 1e5 to scale

            ax.yaxis.set_major_formatter(FuncFormatter(scale_formatter))

        ax.set_ylim(plot_parameters["metrics"][metric]["limits"][0], float(y) + 8 * dh)

        ax.set_ylabel(plot_parameters["metrics"][metric]["label"])
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        # remove ticks
        ax.tick_params(bottom=False, left=False, right=False, top=False, which="both")

        plt.tight_layout()
        # plt.show()
        plt.savefig(f"../data/figures/dyads_vs_triads/{metric}.pdf")

    # MAKE TABLE
    with open("../data/tables/dyads_vs_triads/dyads_vs_triads_metrics.tex", "w") as f:
        f.write("\\begin{table}\n\n")
        f.write("\\centering\n")
        f.write(
            "\\caption{Mean value and standard deviation of synchronization metrics for dyads and triads. Student's $t$-test $p$-values are also shown.\\label{tab:coherence_interaction}}\n"
        )

        n_metrics = len(metrics)

        f.write(f"\\begin{{tabular}}{{l|{'c' * (n_metrics)}}}\n")
        f.write("\\toprule\n")
        f.write(
            "Metric & "
            + " & ".join(
                [
                    plot_parameters["metrics"][metric]["table_label"]
                    for metric in metrics
                ]
            )
            + " \\\\\n"
        )
        f.write("\\midrule\n")
        for pair in ["dyads", "triads_1n", "triads_2n"]:
            line = f"{plot_parameters['pairs'][pair]['label']} & "
            for metric in metrics:
                mean = np.mean(all_values[metric][pair])
                std = np.std(all_values[metric][pair])
                line += f"{mean:.2f} $\\pm$ {std:.2f} & "
            line = line[:-2] + "\\\\\n"
            f.write(line)
        # p_values
        f.write("\\midrule\n")
        line_1 = f"$p$-value for dyads vs triads ($1^{{st}}$ neighbor) & "
        line_2 = f"$p$-value for dyads vs triads ($2^{{nd}}$ neighbor) & "
        for metric in metrics:
            p_value_dyads_triads_1n = ttest_ind(
                all_values[metric]["dyads"],
                all_values[metric]["triads_1n"],
            )[1]
            p_value_dyads_triads_2n = ttest_ind(
                all_values[metric]["dyads"],
                all_values[metric]["triads_2n"],
            )[1]

            line_1 += get_formatted_p_value(p_value_dyads_triads_1n) + " & "
            line_2 += get_formatted_p_value(p_value_dyads_triads_2n) + " & "
        line_1 = line_1[:-2] + "\\\\\n"
        line_2 = line_2[:-2] + "\\\\\n"
        f.write(line_1)
        f.write(line_2)
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
