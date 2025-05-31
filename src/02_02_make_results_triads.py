import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import ImageGrid

import numpy as np
import pandas as pd

import scienceplots

plt.style.use("science")

from utils import (
    get_formatted_p_value,
    get_formatted_p_value_stars,
    pickle_load,
)
from scipy.stats import kruskal

from parameters import TRIADS_PARAMETERS, MAX_DELTA_F


def make_p_values_table(df_no_nan):
    with open("../data/tables/triads/p_values_triads_formations.tex", "w") as f:
        f.write("\\begin{table}\n")
        f.write("\\centering\n")
        f.write(
            "\\caption{$p$-values for the Kruskal-Wallis tests for the independence of means of $\\Delta f$, GSI and CWC for the different pairs in the triad, for all formations.}\n"
        )
        f.write("\\label{tab:p_values_triads_formations}\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write("Formation & $\\Delta f$ & GSI & CWC \\\\\n")
        f.write("\\midrule\n")

        for formation in ["v", "lambda", "abreast"]:
            p_values = []
            for metric in ["delta_f", "gsi", "coherence"]:

                data = df_no_nan[
                    (df_no_nan["formation"] == formation)
                    & (df_no_nan["delta_f"] < MAX_DELTA_F)
                ]
                positions = data["pair"].unique()
                values = [
                    data[data["pair"] == position][metric] for position in positions
                ]

                # remove nan values
                values = [v[~np.isnan(v)] for v in values]

                h, p = kruskal(*values)
                p_values.append(p)

            p_values = [get_formatted_p_value(p) for p in p_values]
            f.write(
                f"{TRIADS_PARAMETERS['formation'][formation]['label']} & ${p_values[0]}$ & ${p_values[1]}$ & ${p_values[2]}$ \\\\\n"
            )

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def make_counts_table(df):
    with open("../data/tables/triads/counts_triads.tex", "w") as f:
        f.write("\\begin{table}\n")
        f.write("\\centering\n")
        f.write("\\caption{Breakdown of the number of triads for each formation.}\n")
        f.write("\\label{tab:counts_triads_formations}\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\toprule\n")
        f.write("Formation & Count \\\\\n")
        f.write("\\midrule\n")

        for formation in ["v", "lambda", "abreast", "following"]:
            counts = len(df[(df["formation"] == formation)]) // 3

            f.write(
                f"{TRIADS_PARAMETERS['formation'][formation]['label']} & {counts} \\\\\n"
            )

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def make_counts_table_pairs(df_no_nan):
    with open("../data/tables/triads/counts_triads_pairs.tex", "w") as f:
        f.write("\\begin{table}\n")
        f.write("\\centering\n")
        f.write(
            "\\caption{Breakdown of the number of triads for each possible pair in the different formations. L is left, R is right, C is center, F is front, and B is back.}\n"
        )
        f.write("\\label{tab:counts_triads_pairs}\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\toprule\n")
        f.write("Formation & L--C & R--C & L--R & F--C & F--B & B--C \\\\\n")
        f.write("\\midrule\n")

        for formation in ["v", "lambda", "abreast", "following"]:
            counts = []
            for pair in [
                "left_center",
                "right_center",
                "left_right",
                "front_center",
                "front_back",
                "back_center",
            ]:

                if formation in ["v", "lambda", "abreast"] and pair in [
                    "front_center",
                    "back_center",
                    "front_back",
                ]:
                    counts.append("-")
                    continue
                if formation == "following" and pair in [
                    "left_center",
                    "right_center",
                    "left_right",
                ]:
                    counts.append("-")
                    continue

                counts.append(
                    len(
                        df_no_nan[
                            (df_no_nan["formation"] == formation)
                            & (df_no_nan["pair"] == pair)
                        ]
                    )
                )

            f.write(
                f"{TRIADS_PARAMETERS['formation'][formation]['label']} & {counts[0]} & {counts[1]} & {counts[2]} & {counts[3]} & {counts[4]} & {counts[5]} \\\\\n"
            )

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


if __name__ == "__main__":

    df = pd.read_csv("../data/csv/gait_data_triads.csv")
    # copy without nan values

    # increase the font size
    plt.rcParams.update({"font.size": 14})

    # ============================
    # Metric vs formation
    # ============================

    for metric in ["delta_f", "gsi", "coherence"]:

        fig, ax = plt.subplots(figsize=(6, 3))

        values_for_p_values = []
        means = []

        for formation in ["v", "lambda", "abreast"]:
            data = df[(df["formation"] == formation) & (df["delta_f"] < MAX_DELTA_F)]

            values_metric = data[metric]

            # remove nan values
            values_metric = values_metric[~np.isnan(values_metric)]

            values_for_p_values.append(values_metric)

            mean_metric = np.mean(values_metric)
            std_metric = np.std(values_metric)
            ste_metric = std_metric / np.sqrt(len(values_metric))

            means.append(mean_metric)

            ax.bar(
                TRIADS_PARAMETERS["formation"][formation]["label"],
                mean_metric,
                yerr=ste_metric,
                color=TRIADS_PARAMETERS["formation"][formation]["color"],
                capsize=5,
            )

        ax.set_ylabel(TRIADS_PARAMETERS["metrics"][metric]["label"])
        ax.grid(color="lightgray", linestyle="--")
        ax.set_ylim(TRIADS_PARAMETERS["metrics"][metric]["limits"])

        _, p_val = kruskal(*values_for_p_values)
        # bracket for p-value
        max_val = max(means)
        y = max_val * 1.1
        dh = max_val * 0.05
        ax.plot([0, 0, 2, 2], [y, y + dh, y + dh, y], color="gray", linewidth=1.5)
        ax.text(
            1,
            float(y) + 4 * dh,
            get_formatted_p_value_stars(p_val),
            ha="center",
            va="center",
            color="gray",
        )
        ax.set_ylim(0, float(y) + 8 * dh)

        plt.tight_layout()
        plt.savefig(f"../data/figures/triads/{metric}_formation.pdf")
        plt.close()
        # plt.show()

    # ============================
    # Metric vs position
    # ============================

    for metric in ["gsi", "coherence", "delta_f"]:

        fig, ax = plt.subplots(3, 1, figsize=(6, 10))

        for i, formation in enumerate(["v", "lambda", "abreast"]):

            data = df[(df["formation"] == formation) & (df["delta_f"] < MAX_DELTA_F)]

            values_for_p_values = []
            means = []
            for j, position in enumerate(["left_center", "right_center", "left_right"]):

                data_position = data[data["pair"] == position]

                values = data_position[metric]

                # remove nan values
                values = values[~np.isnan(values)]

                values_for_p_values.append(values)

                mean = np.mean(values)
                std = np.std(values)
                ste = std / np.sqrt(len(values))

                means.append(mean)

                ax[i].bar(
                    TRIADS_PARAMETERS["pair"][position]["label"],
                    mean,
                    yerr=ste,
                    color=TRIADS_PARAMETERS["pair"][position]["color"],
                    capsize=5,
                )

            ax[i].set_ylabel(TRIADS_PARAMETERS["metrics"][metric]["label"])
            ax[i].grid(color="lightgray", linestyle="--")
            ax[i].set_title(
                f"{TRIADS_PARAMETERS['formation'][formation]['label']} formation"
            )

            _, p_val = kruskal(*values_for_p_values)
            # bracket for p-value
            max_val = max(means)
            y = max_val * 1.3
            dh = max_val * 0.05
            ax[i].plot(
                [0, 0, 2, 2], [y, y + dh, y + dh, y], color="gray", linewidth=1.5
            )
            ax[i].text(
                1,
                float(y) + 4 * dh,
                get_formatted_p_value_stars(p_val),
                ha="center",
                va="center",
                color="gray",
            )
            ax[i].set_ylim(0, float(y) + 8 * dh)

        plt.tight_layout()
        plt.savefig(f"../data/figures/triads/{metric}_position.pdf")
        plt.close()

    # ============================
    # Scatter plot positions
    # ============================

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    for formation in ["v", "lambda", "abreast", "following"]:

        data = df[(df["formation"] == formation)]

        if formation in ["v", "lambda", "abreast"]:

            position_left_x = data[data["pair"] == "left_center"]["position_1_x"]
            position_left_y = data[data["pair"] == "left_center"]["position_1_y"]
            position_center_x = data[data["pair"] == "left_center"]["position_2_x"]
            position_center_y = data[data["pair"] == "left_center"]["position_2_y"]
            position_right_x = data[data["pair"] == "right_center"]["position_1_x"]
            position_right_y = data[data["pair"] == "right_center"]["position_1_y"]

            ax.scatter(
                position_left_x,
                position_left_y,
                color=TRIADS_PARAMETERS["formation"][formation]["color"],
                marker=TRIADS_PARAMETERS["formation"][formation]["marker"],
            )

            ax.scatter(
                position_center_x,
                position_center_y,
                color=TRIADS_PARAMETERS["formation"][formation]["color"],
                marker=TRIADS_PARAMETERS["formation"][formation]["marker"],
            )

            ax.scatter(
                position_right_x,
                position_right_y,
                color=TRIADS_PARAMETERS["formation"][formation]["color"],
                label=TRIADS_PARAMETERS["formation"][formation]["label"],
                marker=TRIADS_PARAMETERS["formation"][formation]["marker"],
            )

        elif formation == "following":

            position_front_x = data[data["pair"] == "front_center"]["position_1_x"]
            position_front_y = data[data["pair"] == "front_center"]["position_1_y"]
            position_back_x = data[data["pair"] == "back_center"]["position_1_x"]
            position_back_y = data[data["pair"] == "back_center"]["position_1_y"]
            position_center_x = data[data["pair"] == "front_center"]["position_2_x"]
            position_center_y = data[data["pair"] == "front_center"]["position_2_y"]

            ax.scatter(
                position_front_x,
                position_front_y,
                color=TRIADS_PARAMETERS["formation"][formation]["color"],
                marker=TRIADS_PARAMETERS["formation"][formation]["marker"],
            )

            ax.scatter(
                position_back_x,
                position_back_y,
                color=TRIADS_PARAMETERS["formation"][formation]["color"],
                marker=TRIADS_PARAMETERS["formation"][formation]["marker"],
            )

            ax.scatter(
                position_center_x,
                position_center_y,
                color=TRIADS_PARAMETERS["formation"][formation]["color"],
                label=TRIADS_PARAMETERS["formation"][formation]["label"],
                marker=TRIADS_PARAMETERS["formation"][formation]["marker"],
            )

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.legend()
        ax.grid(color="lightgray", linestyle="--")
        ax.set_aspect("equal")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)

    plt.tight_layout()
    plt.savefig("../data/figures/triads/scatter_positions.pdf")
    plt.close()

    # ============================
    # heatmap positions
    # ============================

    positions = pickle_load("../data/pickle/triad_positions.pkl")

    n_cells = 50
    x_min, x_max = -1.5, 1.5
    y_min, y_max = -1.5, 1.5
    x_bins = np.linspace(x_min, x_max, n_cells + 1)
    y_bins = np.linspace(y_min, y_max, n_cells + 1)

    fig = plt.figure(figsize=(12, 4))

    axes = ImageGrid(
        fig,
        111,  # as in plt.subplot(111)
        nrows_ncols=(1, 4),
        axes_pad=0.5,
        share_all=True,
        cbar_location="right",
        cbar_mode="single",
        cbar_size="7%",
        cbar_pad=0.15,
    )

    for i, (formation, label) in enumerate(
        zip(["v", "lambda", "abreast", "following"], "abcd")
    ):

        grid = np.zeros((n_cells, n_cells))

        for position_tuple in positions[formation]:
            for position in position_tuple:
                x = position[:, 0] / 1000
                y = position[:, 1] / 1000

                nx = np.ceil((x - x_min) / (x_max - x_min) * n_cells).astype("int")
                ny = np.ceil((y - y_min) / (y_max - y_min) * n_cells).astype("int")

                in_roi = np.logical_and(
                    np.logical_and(nx >= 0, nx < n_cells),
                    np.logical_and(ny >= 0, ny < n_cells),
                )
                nx = nx[in_roi]
                ny = ny[in_roi]

                grid[nx, ny] += 1

        grid /= np.max(grid)

        im = axes[i].imshow(
            grid.T,
            extent=[x_min, x_max, y_min, y_max],
            origin="lower",
            cmap="inferno_r",
        )
        axes[i].set_xlabel("x (m)")
        axes[i].set_ylabel("y (m)")
        axes[i].set_title(f"({label})", y=-0.4)
        # axes[i].grid(color="lightgray", linestyle="--")
        axes[i].set_aspect("equal")
        axes[i].set_xlim(-1.5, 1.5)
        axes[i].set_ylim(-1.5, 1.5)

    axes[0].cax.colorbar(im)
    # axes[0].cax.toggle_label(True)

    plt.tight_layout()
    plt.savefig(f"../data/figures/triads/heatmap_positions.pdf")
    plt.close()

    # formation_positions = positions[formation]

    fig, ax = plt.subplots(2, 3, figsize=(14, 7))

    # ============================
    # Determinism
    # ============================

    values_for_p_values = []
    means = []
    for formation in ["v", "lambda", "abreast"]:
        data = df[(df["formation"] == formation) & (df["delta_f"] < MAX_DELTA_F)]

        values_1 = data["determinism_1"]
        values_2 = data["determinism_2"]

        values_det = np.concatenate([values_1, values_2])

        # remove NaNs
        values_det = values_det[~np.isnan(values_det)]

        values_for_p_values.append(values_det)

        mean_metric = np.mean(values_det)
        std_metric = np.std(values_det)
        ste_metric = std_metric / np.sqrt(len(values_det))

        means.append(mean_metric)

        ax[0][0].bar(
            TRIADS_PARAMETERS["formation"][formation]["label"],
            mean_metric,
            yerr=ste_metric,
            color=TRIADS_PARAMETERS["formation"][formation]["color"],
            capsize=5,
        )

    ax[0][0].set_ylabel("$D$")
    ax[0][0].grid(color="lightgray", linestyle="--")
    ax[0][0].set_title("(a)", y=-0.4)
    ax[0][0].set_xlabel("Formation")

    _, p_val = kruskal(*values_for_p_values)
    # bracket for p-value
    max_val = max(means)
    y = max_val * 1.1
    dh = (max_val - TRIADS_PARAMETERS["metrics"]["determinism"]["limits"][0]) * 0.05
    ax[0][0].plot([0, 0, 2, 2], [y, y + dh, y + dh, y], color="gray", linewidth=1.5)
    ax[0][0].text(
        1,
        float(y) + 4 * dh,
        get_formatted_p_value_stars(p_val),
        ha="center",
        va="center",
        color="gray",
    )
    ax[0][0].set_ylim(
        TRIADS_PARAMETERS["metrics"]["determinism"]["limits"][0], float(y) + 8 * dh
    )

    # ============================
    # Lyapunov
    # ============================

    values_for_p_values = []
    means = []
    for formation in ["v", "lambda", "abreast"]:
        data = df[(df["formation"] == formation) & (df["delta_f"] < MAX_DELTA_F)]

        values_1 = data["lyapunov_1"]
        values_2 = data["lyapunov_2"]

        values_lyap = np.concatenate([values_1, values_2])

        # remove NaNs
        values_lyap = values_lyap[~np.isnan(values_lyap)]

        values_for_p_values.append(values_lyap)

        mean_metric = np.mean(values_lyap)
        std_metric = np.std(values_lyap)
        ste_metric = std_metric / np.sqrt(len(values_lyap))

        means.append(mean_metric)

        ax[0][1].bar(
            TRIADS_PARAMETERS["formation"][formation]["label"],
            mean_metric,
            yerr=ste_metric,
            color=TRIADS_PARAMETERS["formation"][formation]["color"],
            capsize=5,
        )

    ax[0][1].set_ylabel("$l_{lyap}$ ($\\times 10^{-3}$)")
    ax[0][1].grid(color="lightgray", linestyle="--")
    ax[0][1].set_title("(b)", y=-0.4)
    ax[0][1].set_xlabel("Formation")

    # format the axis to show only 1 decimal and put exponent in the axis label
    def scale_formatter(value, _):
        return f"{value * 1e3:.1f}"  # Multiply by 1e5 to scale

    ax[0][1].yaxis.set_major_formatter(FuncFormatter(scale_formatter))

    _, p_val = kruskal(*values_for_p_values)
    print("lyap", get_formatted_p_value(p_val))
    # bracket for p-value
    max_val = max(means)
    y = max_val * 1.1
    dh = (max_val - TRIADS_PARAMETERS["metrics"]["lyapunov"]["limits"][0]) * 0.05
    ax[0][1].plot([0, 0, 2, 2], [y, y + dh, y + dh, y], color="gray", linewidth=1.5)
    ax[0][1].text(
        1,
        float(y) + 4 * dh,
        get_formatted_p_value_stars(p_val),
        ha="center",
        va="center",
        color="gray",
    )
    ax[0][1].set_ylim(
        TRIADS_PARAMETERS["metrics"]["lyapunov"]["limits"][0], float(y) + 8 * dh
    )

    # ============================
    # CRQ
    # ============================

    for i, (metric, label) in enumerate(
        zip(["rec", "det", "maxline"], ["(c)", "(d)", "(e)"])
    ):

        values_for_p_values = []
        means = []
        for formation in ["v", "lambda", "abreast"]:
            data = df[(df["formation"] == formation) & (df["delta_f"] < MAX_DELTA_F)]

            values_metric = data[metric]

            # remove nan values
            values_metric = values_metric[~np.isnan(values_metric)]

            values_for_p_values.append(values_metric)

            mean_metric = np.mean(values_metric)
            std_metric = np.std(values_metric)
            ste_metric = std_metric / np.sqrt(len(values_metric))

            means.append(mean_metric)

            ax[1][i].bar(
                TRIADS_PARAMETERS["formation"][formation]["label"],
                mean_metric,
                yerr=ste_metric,
                color=TRIADS_PARAMETERS["formation"][formation]["color"],
                capsize=5,
            )

        ax[1][i].set_ylabel(TRIADS_PARAMETERS["metrics"][metric]["label"])
        ax[1][i].grid(color="lightgray", linestyle="--")
        ax[1][i].set_title(label, y=-0.4)
        ax[1][i].set_ylim(TRIADS_PARAMETERS["metrics"][metric]["limits"])
        ax[1][i].set_xlabel("Formation")

        _, p_val = kruskal(*values_for_p_values)
        print(metric, get_formatted_p_value(p_val))
        # bracket for p-value
        max_val = max(means)
        y = max_val * 1.1
        dh = (max_val - TRIADS_PARAMETERS["metrics"][metric]["limits"][0]) * 0.05
        ax[1][i].plot([0, 0, 2, 2], [y, y + dh, y + dh, y], color="gray", linewidth=1.5)
        ax[1][i].text(
            1,
            float(y) + 4 * dh,
            get_formatted_p_value_stars(p_val),
            ha="center",
            va="center",
            color="gray",
        )
        ax[1][i].set_ylim(
            TRIADS_PARAMETERS["metrics"][metric]["limits"][0], float(y) + 8 * dh
        )

    # remove the last subplot
    fig.delaxes(ax[0][2])

    plt.tight_layout()
    plt.savefig("../data/figures/triads/triad_nonlinear_metrics.pdf")
    plt.close()
    # plt.show()

    # ============================
    # P-values and counts

    make_p_values_table(df)
    make_counts_table(df)
    make_counts_table_pairs(df)
