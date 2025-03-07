from matplotlib import transforms
from matplotlib.patches import Ellipse
from matplotlib.legend_handler import HandlerPatch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import pandas as pd
import scikit_posthocs as sp

import scienceplots

plt.style.use("science")

from utils import (
    get_formatted_p_value,
    compute_pdf,
    compute_binned_values,
    get_formatted_p_value_stars,
    get_latex_scientific_notation,
)
from scipy.stats import kruskal, tukey_hsd, circmean, circvar, ttest_ind
from matplotlib.gridspec import GridSpec


def make_gait_stats_table(df_dyads, df_individuals):
    with open("../data/ieee/tables/dyads/gait_stats.tex", "w") as f:
        f.write("\\begin{table*}[!t]\n\n")
        f.write(
            "\\caption{Mean Value and Standard Deviation of Velocity $v$, Stride Frequency $f$, and Stride Length $l$ for Different Intensities of Interaction. Kruskal-Wallis $p$-Values for the Difference Between the Intensities of Interaction and Student's $t$-Test $p$-Values for the Difference between All Dyads and Individuals Are Also Shown\\label{tab:gait_stats}}\n"
        )
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\toprule\n")
        f.write(
            "Intensity of interaction & \\multicolumn{2}{c}{$v$ [m/s]} & \\multicolumn{2}{c}{$f$ [Hz]} & \\multicolumn{2}{c}{$l$ [m]} \\\\\n"
        )
        f.write("\\midrule\n")
        all_values_frequency = []
        all_values_velocity = []
        all_values_length = []

        # compute p-values
        for interaction in interactions:
            data = df_dyads[df_dyads["interaction"] == interaction]
            values_frequency = np.concatenate(
                [data["frequency_1"], data["frequency_2"]]
            )
            values_velocity = np.concatenate([data["velocity_1"], data["velocity_2"]])
            values_length = np.concatenate(
                [data["stride_length_1"], data["stride_length_2"]]
            )

            # remove NaNs
            values_frequency = values_frequency[~np.isnan(values_frequency)]
            values_velocity = values_velocity[~np.isnan(values_velocity)]
            values_length = values_length[~np.isnan(values_length)]

            all_values_frequency.append(values_frequency)
            all_values_velocity.append(values_velocity)
            all_values_length.append(values_length)

        # p-values for Kruskal-Wallis
        _, p_val_freq = kruskal(*all_values_frequency)
        _, p_val_vel = kruskal(*all_values_velocity)
        _, p_val_len = kruskal(*all_values_length)

        for interaction in interactions:
            data = df_dyads[df_dyads["interaction"] == interaction]
            values_frequency = np.concatenate(
                [data["frequency_1"], data["frequency_2"]]
            )
            values_velocity = np.concatenate([data["velocity_1"], data["velocity_2"]])
            values_length = np.concatenate(
                [data["stride_length_1"], data["stride_length_2"]]
            )

            # remove NaNs
            values_frequency = values_frequency[~np.isnan(values_frequency)]
            values_velocity = values_velocity[~np.isnan(values_velocity)]
            values_length = values_length[~np.isnan(values_length)]

            if interaction == 0:
                f.write(
                    f"{plot_data['interaction'][interaction]['label']} & "
                    f"${np.mean(values_velocity):.2f} \\pm {np.std(values_velocity):.2f}$ & \\multirow{{4}}{{*}}{{${get_formatted_p_value(p_val_vel)}$}} &"
                    f"${np.mean(values_frequency):.2f} \\pm {np.std(values_frequency):.2f}$ & \\multirow{{4}}{{*}}{{${get_formatted_p_value(p_val_freq)}$}} &"
                    f"${np.mean(values_length):.2f} \\pm {np.std(values_length):.2f}$ & \\multirow{{4}}{{*}}{{${get_formatted_p_value(p_val_len)}$}} \\\\\n"
                )
            else:
                f.write(
                    f"{plot_data['interaction'][interaction]['label']} & "
                    f"${np.mean(values_velocity):.2f} \\pm {np.std(values_velocity):.2f}$ & &"
                    f"${np.mean(values_frequency):.2f} \\pm {np.std(values_frequency):.2f}$ & &"
                    f"${np.mean(values_length):.2f} \\pm {np.std(values_length):.2f}$ & \\\\\n"
                )

        f.write("\\midrule\n")

        # averages
        flat_freq = np.concatenate(all_values_frequency)
        flat_vel = np.concatenate(all_values_velocity)
        flat_len = np.concatenate(all_values_length)

        f.write(
            f"All & "
            f"${np.mean(flat_vel):.2f} \\pm {np.std(flat_vel):.2f}$ & &"
            f"${np.mean(flat_freq):.2f} \\pm {np.std(flat_freq):.2f}$ & &"
            f"${np.mean(flat_len):.2f} \\pm {np.std(flat_len):.2f}$ &\\\\\n"
        )

        # individuals
        values_frequency_individuals = df_individuals["frequency"]
        values_velocity_individuals = df_individuals["velocity"]
        values_length_individuals = df_individuals["stride_length"]

        # remove NaNs
        values_frequency_individuals = values_frequency_individuals[
            ~np.isnan(values_frequency_individuals)
        ]
        values_velocity_individuals = values_velocity_individuals[
            ~np.isnan(values_velocity_individuals)
        ]
        values_length_individuals = values_length_individuals[
            ~np.isnan(values_length_individuals)
        ]

        f.write(
            f"Individuals & "
            f"${np.mean(values_velocity_individuals):.2f} \\pm {np.std(values_velocity_individuals):.2f}$ & & "
            f"${np.mean(values_frequency_individuals):.2f} \\pm {np.std(values_frequency_individuals):.2f}$ & &"
            f"${np.mean(values_length_individuals):.2f} \\pm {np.std(values_length_individuals):.2f}$  & \\\\\n"
        )

        f.write("\\midrule\n")

        # f.write(
        #     f"Kruskal-Wallis $p$-value & "
        #     f"${get_formatted_p_value(p_val_vel)}$ & "
        #     f"${get_formatted_p_value(p_val_freq)}$ & "
        #     f"${get_formatted_p_value(p_val_len)}$ \\\\\n"
        # )

        # all vs individuals
        _, p_val_freq = ttest_ind(flat_freq, values_frequency_individuals)
        _, p_val_vel = ttest_ind(flat_vel, values_velocity_individuals)
        _, p_val_len = ttest_ind(flat_len, values_length_individuals)

        f.write(
            f"Student's $t$-test $p$-value & "
            f"${get_formatted_p_value(p_val_vel)}$ & &"
            f"${get_formatted_p_value(p_val_freq)}$ & &"
            f"${get_formatted_p_value(p_val_len)}$ &\\\\\n"
        )

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")


def make_table_gsi_interaction(df_dyads):
    with open("../data/ieee/tables/dyads/gsi_interaction.tex", "w") as f:
        f.write("\\begin{table}[!t]\n")
        f.write(
            "\\caption{GSI for Different Intensities of Interaction. Kruskal-Wallis $p$-Values for the Difference Between the Intensities of Interaction and Student's $t$-Test $p$-values for the Difference Vetween All Dyads and the Baselines Are Also Shown\\label{tab:gsi_interaction}}\n"
        )
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\toprule\n")
        f.write("Intensity of interaction & \\multicolumn{2}{c}{GSI}  \\\\\n")
        f.write("\\midrule\n")

        # compute p-values
        all_values_gsi = []
        for interaction in interactions:
            data = df_dyads[
                (df_dyads["interaction"] == interaction)
                & (df_dyads["delta_f"] < max_delta_f)
            ]
            values_gsi = data["gsi"]

            # remove NaNs
            values_gsi = values_gsi[~np.isnan(values_gsi)]

            all_values_gsi.append(values_gsi)

        # p-values for Kruskal-Wallis
        _, p_val_gsi = kruskal(*all_values_gsi)

        for interaction in [0, 1, 2, 3, 4, 5]:

            data = df_dyads[
                (df_dyads["interaction"] == interaction)
                & (df_dyads["delta_f"] < max_delta_f)
            ]
            values_gsi = data["gsi"]

            # remove NaNs
            values_gsi = values_gsi[~np.isnan(values_gsi)]

            if interaction == 0:
                f.write(
                    f"{plot_data['interaction'][interaction]['label']} & "
                    f"${np.mean(values_gsi):.2f} \\pm {np.std(values_gsi):.2f}$ & \\multirow{{4}}{{*}}{{${get_formatted_p_value(p_val_gsi)}$}} \\\\\n"
                )
            else:
                f.write(
                    f"{plot_data['interaction'][interaction]['label']} & "
                    f"${np.mean(values_gsi):.2f} \\pm {np.std(values_gsi):.2f}$ & \\\\\n"
                )

            if interaction == 3:
                f.write("\\midrule\n")

        f.write("\\midrule\n")

        # averages
        flat_gsi = np.concatenate(all_values_gsi)

        f.write(
            f"All & " f"${np.mean(flat_gsi):.2f} \\pm {np.std(flat_gsi):.2f}$ & \\\\\n"
        )

        f.write("\\midrule\n")

        # all vs baseline
        values_gsi_baseline_1 = df_dyads[
            (df_dyads["interaction"] == 4) & (df_dyads["delta_f"] < max_delta_f)
        ]["gsi"]
        values_gsi_baseline_2 = df_dyads[
            (df_dyads["interaction"] == 5) & (df_dyads["delta_f"] < max_delta_f)
        ]["gsi"]

        # remove NaNs
        values_gsi_baseline_1 = values_gsi_baseline_1[~np.isnan(values_gsi_baseline_1)]
        values_gsi_baseline_2 = values_gsi_baseline_2[~np.isnan(values_gsi_baseline_2)]

        _, p_val_gsi_1 = ttest_ind(flat_gsi, values_gsi_baseline_1)
        _, p_val_gsi_2 = ttest_ind(flat_gsi, values_gsi_baseline_2)

        f.write(
            f"Student's $t$-test $p$-value for $B_r$ & "
            f"${get_formatted_p_value(p_val_gsi_1)}$ &\\\\\n"
        )

        f.write(
            f"Student's $t$-test $p$-value for $B_c$ & "
            f"${get_formatted_p_value(p_val_gsi_2)}$  &\\\\\n"
        )

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def make_table_coherence_interaction(df_dyads):
    with open("../data/ieee/tables/dyads/coherence_interaction.tex", "w") as f:
        f.write("\\begin{table}[!t]\n")
        f.write(
            "\\caption{CWC for Different Intensities of Interaction. Kruskal-Wallis $p$-Values for the Difference Between the Intensities of Interaction and Student's $t$-Test $p$-Values for the Difference between All Dyads and the Baselines Are Also Shown\\label{tab:coherence_interaction}}\n"
        )
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\toprule\n")
        f.write("Intensity of interaction & \\multicolumn{2}{c}{CWC} \\\\\n")
        f.write("\\midrule\n")

        # compute p-values
        all_values_coherence = []
        for interaction in interactions:
            data = df_dyads[
                (df_dyads["interaction"] == interaction)
                & (df_dyads["delta_f"] < max_delta_f)
            ]
            values_coherence = data["coherence"]

            # remove NaNs
            values_coherence = values_coherence[~np.isnan(values_coherence)]

            all_values_coherence.append(values_coherence)

        # p-values for Kruskal-Wallis
        _, p_val_coherence = kruskal(*all_values_coherence)

        for interaction in [0, 1, 2, 3, 4, 5]:

            data = df_dyads[
                (df_dyads["interaction"] == interaction)
                & (df_dyads["delta_f"] < max_delta_f)
            ]
            values_coherence = data["coherence"]

            # remove NaNs
            values_coherence = values_coherence[~np.isnan(values_coherence)]

            if interaction == 0:
                f.write(
                    f"{plot_data['interaction'][interaction]['label']} & "
                    f"${np.mean(values_coherence):.2f} \\pm {np.std(values_coherence):.2f}$ & \\multirow{{4}}{{*}}{{${get_formatted_p_value(p_val_coherence)}$}} \\\\\n"
                )
            else:
                f.write(
                    f"{plot_data['interaction'][interaction]['label']} & "
                    f"${np.mean(values_coherence):.2f} \\pm {np.std(values_coherence):.2f}$ & \\\\\n"
                )

            if interaction == 3:
                f.write("\\midrule\n")

        f.write("\\midrule\n")

        # averages
        flat_coherence = np.concatenate(all_values_coherence)

        f.write(
            f"All & "
            f"${np.mean(flat_coherence):.2f} \\pm {np.std(flat_coherence):.2f}$ & \\\\\n"
        )

        f.write("\\midrule\n")

        # all vs baseline
        values_coherence_baseline_1 = df_dyads[
            (df_dyads["interaction"] == 4) & (df_dyads["delta_f"] < max_delta_f)
        ]["coherence"]
        values_coherence_baseline_2 = df_dyads[
            (df_dyads["interaction"] == 5) & (df_dyads["delta_f"] < max_delta_f)
        ]["coherence"]

        # remove NaNs
        values_coherence_baseline_1 = values_coherence_baseline_1[
            ~np.isnan(values_coherence_baseline_1)
        ]
        values_coherence_baseline_2 = values_coherence_baseline_2[
            ~np.isnan(values_coherence_baseline_2)
        ]

        _, p_val_coherence_1 = ttest_ind(flat_coherence, values_coherence_baseline_1)
        _, p_val_coherence_2 = ttest_ind(flat_coherence, values_coherence_baseline_2)

        f.write(
            f"Student's $t$-test $p$-value for $B_r$ & "
            f"${get_formatted_p_value(p_val_coherence_1)}$  &\\\\\n"
        )

        f.write(
            f"Student's $t$-test $p$-value for $B_c$ & "
            f"${get_formatted_p_value(p_val_coherence_2)}$ &\\\\\n"
        )

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def make_table_gsi_coherence_contact(df_dyads):
    with open("../data/ieee/tables/dyads/gsi_coherence_contact.tex", "w") as f:
        f.write("\\begin{table}[!t]\n")
        f.write(
            "\\caption{GSI and CWC for Different Levels of Contact. Student's $t$-Test $p$-Values for the Difference Between the Levels of Contact Are Also Shown\\label{tab:sync_stats_contact}}\n"
        )
        f.write("\\centering\n")

        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write(
            "Contact state & \\multicolumn{2}{c}{GSI} & \\multicolumn{2}{c}{CWC} \\\\\n"
        )
        f.write("\\midrule\n")

        all_values_gsi = []
        all_values_coherence = []

        # compute p-values
        for contact in [0, 1]:
            data = df_dyads[
                (df_dyads["contact"] == contact) & (df_dyads["delta_f"] < max_delta_f)
            ]
            values_gsi = data["gsi"]
            values_coherence = data["coherence"]

            # remove NaNs
            values_gsi = values_gsi[~np.isnan(values_gsi)]
            values_coherence = values_coherence[~np.isnan(values_coherence)]

            all_values_gsi.append(values_gsi)
            all_values_coherence.append(values_coherence)

        # p-values for Student's t-test
        _, p_val_gsi = ttest_ind(*all_values_gsi)
        _, p_val_coherence = ttest_ind(*all_values_coherence)

        for contact in [0, 1]:
            data = df_dyads[
                (df_dyads["contact"] == contact) & (df_dyads["delta_f"] < max_delta_f)
            ]
            values_gsi = data["gsi"]
            values_coherence = data["coherence"]

            # remove NaNs
            values_gsi = values_gsi[~np.isnan(values_gsi)]
            values_coherence = values_coherence[~np.isnan(values_coherence)]

            if contact == 0:
                f.write(
                    f"{plot_data['contact'][contact]['label']} & "
                    f"${np.mean(values_gsi):.2f} \\pm {np.std(values_gsi):.2f}$ & \\multirow{{2}}{{*}}{{${get_formatted_p_value(p_val_gsi)}$}} &"
                    f"${np.mean(values_coherence):.2f} \\pm {np.std(values_coherence):.2f}$ & \\multirow{{2}}{{*}}{{${get_formatted_p_value(p_val_coherence)}$}}  \\\\\n"
                )
            else:
                f.write(
                    f"{plot_data['contact'][contact]['label']} & "
                    f"${np.mean(values_gsi):.2f} \\pm {np.std(values_gsi):.2f}$ & &"
                    f"${np.mean(values_coherence):.2f} \\pm {np.std(values_coherence):.2f}$ & \\\\\n"
                )

        f.write("\\midrule\n")

        # baseline-values
        values_gsi_baseline_1 = df_dyads[
            (df_dyads["interaction"] == 4) & (df_dyads["delta_f"] < max_delta_f)
        ]["gsi"]
        values_gsi_baseline_2 = df_dyads[
            (df_dyads["interaction"] == 5) & (df_dyads["delta_f"] < max_delta_f)
        ]["gsi"]
        values_coherence_baseline_1 = df_dyads[
            (df_dyads["interaction"] == 4) & (df_dyads["delta_f"] < max_delta_f)
        ]["coherence"]
        values_coherence_baseline_2 = df_dyads[
            (df_dyads["interaction"] == 5) & (df_dyads["delta_f"] < max_delta_f)
        ]["coherence"]

        # remove NaNs
        values_gsi_baseline_1 = values_gsi_baseline_1[~np.isnan(values_gsi_baseline_1)]
        values_gsi_baseline_2 = values_gsi_baseline_2[~np.isnan(values_gsi_baseline_2)]
        values_coherence_baseline_1 = values_coherence_baseline_1[
            ~np.isnan(values_coherence_baseline_1)
        ]
        values_coherence_baseline_2 = values_coherence_baseline_2[
            ~np.isnan(values_coherence_baseline_2)
        ]

        f.write(
            f"$B_r$ & "
            f"${np.mean(values_gsi_baseline_1):.2f} \\pm {np.std(values_gsi_baseline_1):.2f}$ & & "
            f"${np.mean(values_coherence_baseline_1):.2f} \\pm {np.std(values_coherence_baseline_1):.2f}$ & \\\\\n"
        )

        f.write(
            f"$B_c$ & "
            f"${np.mean(values_gsi_baseline_2):.2f} \\pm {np.std(values_gsi_baseline_2):.2f}$ & & "
            f"${np.mean(values_coherence_baseline_2):.2f} \\pm {np.std(values_coherence_baseline_2):.2f}$ & \\\\\n"
        )

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def make_tukey_table(df_dyads):

    with open("../data/ieee/tables/dyads/tukey.tex", "w") as f:

        all_values_gsi = []
        all_values_coherence = []

        for interaction in interactions_with_baseline:
            data = df_dyads[
                (df_dyads["interaction"] == interaction)
                & (df_dyads["delta_f"] < max_delta_f)
            ]
            values_gsi = data["gsi"]
            values_coherence = data["coherence"]

            # remove NaNs
            values_gsi = values_gsi[~np.isnan(values_gsi)]
            values_coherence = values_coherence[~np.isnan(values_coherence)]

            all_values_gsi.append(values_gsi)
            all_values_coherence.append(values_coherence)

        result_tukey_gsi = tukey_hsd(*all_values_gsi)
        result_tukey_coherence = tukey_hsd(*all_values_coherence)

        f.write("\\begin{table}[!t]\n")
        f.write("\\centering\n")
        f.write(
            "\\caption{Tukey's HSD Test for Pairwise Comparisons of GSI Between Different Intensities of Interaction.}\n"
        )
        f.write("\\label{tab:tukey_gsi}\n")
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\toprule\n")
        f.write(
            " & Baseline & Interaction 0 & Interaction 1 & Interaction 2 & Interaction 3 \\\\\n"
        )
        f.write("\\midrule\n")
        for i, interaction_i in enumerate(interactions_with_baseline):
            f.write(
                f"{plot_data['interaction'][interaction_i]['label']} & "
                + " & ".join(
                    [
                        (
                            f"${get_formatted_p_value(result_tukey_gsi.pvalue[i][j])}$"
                            if j > i
                            else "-"
                        )
                        for j in range(5)
                    ]
                )
                + " \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

        f.write("\\begin{table}[!t]\n")
        f.write("\\centering\n")
        f.write(
            "\\caption{Tukey's HSD Test for Pairwise Comparisons of Coherence Between Different Intensities of Interaction.}\n"
        )
        f.write("\\label{tab:tukey_coherence}\n")
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\toprule\n")
        f.write(
            " & Baseline & Interaction 0 & Interaction 1 & Interaction 2 & Interaction 3 \\\\\n"
        )
        f.write("\\midrule\n")

        for i, interaction_i in enumerate(interactions_with_baseline):
            f.write(
                f"{plot_data['interaction'][interaction_i]['label']} & "
                + " & ".join(
                    [
                        (
                            f"${get_formatted_p_value(result_tukey_coherence.pvalue[i][j])}$"
                            if j > i
                            else "-"
                        )
                        for j in range(5)
                    ]
                )
                + " \\\\\n"
            )

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def make_relative_phase_table(df_dyads):
    with open("../data/ieee/tables/dyads/relative_phase.tex", "w") as f:
        f.write("\\begin{table}[!t]\n")
        f.write(
            "\\caption{Circular Mean and Variance of the Relative Phase Between Pedestrians for Different Intensities of Interaction and Baseline\\label{tab:relative_phase}}\n"
        )
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\toprule\n")
        f.write("Intensity of interaction & Mean relative phase (°) & Variance \\\\\n")
        f.write("\\midrule\n")
        all_values_phase = []
        for interaction in [0, 1, 2, 3, 4, 5]:

            data = df_dyads[
                (df_dyads["interaction"] == interaction)
                & (df_dyads["delta_f"] < max_delta_f)
            ]
            values_phase = data["relative_phase"]

            # remove NaNs
            values_phase = values_phase[~np.isnan(values_phase)]

            mean_phase = np.rad2deg(circmean(values_phase, high=np.pi, low=-np.pi))  # type: ignore
            var_phase = circvar(values_phase, high=np.pi, low=-np.pi)  # type: ignore

            all_values_phase.append(values_phase)

            f.write(
                f"{plot_data['interaction'][interaction]['label']} & "
                f"${mean_phase:.2f}$ & "
                f"${var_phase:.2f}$ \\\\\n"
            )

            if interaction == 3:
                f.write("\\midrule\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def make_table_delta_f(df_dyads):
    with open("../data/ieee/tables/dyads/delta_f.tex", "w") as f:
        f.write("\\begin{table}[!t]\n")
        f.write(
            "\\caption{Mean and Standard Error of the Difference in Stride Frequency $\\Delta f$ Between Baseline Pairs of $B_r$ and $B_c$ as Well as Dyad Members for Different Intensities of Interaction. Kruskal-Wallis $p$-Value for the Difference Between the Intensities of Interaction and Student's $t$-Test $p$-Value for the Difference Between All Dyads and the Baseline Are Also Shown\\label{tab:delta_f}}\n"
        )
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\toprule\n")
        f.write(
            "Intensity of interaction & \\multicolumn{2}{c}{$\\Delta f$ [Hz]}  \\\\\n"
        )
        f.write("\\midrule\n")

        # compute the p-values
        values_pvalue = []
        for interaction in interactions_with_baseline:
            data = df_dyads[(df_dyads["interaction"] == interaction)]
            values_delta_f = data["delta_f"]

            # remove NaNs
            values_delta_f = values_delta_f[~np.isnan(values_delta_f)]

            if interaction <= 3:
                values_pvalue.append(values_delta_f)

        # p-value for Kruskal-Wallis
        _, p_val_dyads = kruskal(*values_pvalue)

        all_values_dyads = []
        for interaction in [0, 1, 2, 3, 4, 5]:

            data = df_dyads[(df_dyads["interaction"] == interaction)]
            values_delta_f = data["delta_f"]

            # remove NaNs
            values_delta_f = values_delta_f[~np.isnan(values_delta_f)]

            mean_delta_f = np.mean(values_delta_f)
            std_delta_f = np.std(values_delta_f)
            ste_delta_f = std_delta_f / np.sqrt(len(values_delta_f))

            if interaction <= 3:
                all_values_dyads.extend(values_delta_f)

            if interaction == 0:
                f.write(
                    f"{plot_data['interaction'][interaction]['label']} &"
                    f"${mean_delta_f:.2f} \\pm {ste_delta_f:.2f}$  & \\multirow{{4}}{{*}}{{${get_formatted_p_value(p_val_dyads)}$}} \\\\\n"
                )
            else:
                f.write(
                    f"{plot_data['interaction'][interaction]['label']} &"
                    f"${mean_delta_f:.2f} \\pm {ste_delta_f:.2f}$ & \\\\\n"
                )

            if interaction == 3:
                f.write("\\midrule\n")

        f.write("\\midrule\n")

        values_baseline_1 = df_dyads[df_dyads["interaction"] == 4]["delta_f"]
        values_baseline_2 = df_dyads[df_dyads["interaction"] == 5]["delta_f"]
        # remove NaNs
        values_baseline_1 = values_baseline_1[~np.isnan(values_baseline_1)]
        values_baseline_2 = values_baseline_2[~np.isnan(values_baseline_2)]

        _, p_val_baseline_1 = ttest_ind(all_values_dyads, values_baseline_1)
        f.write(
            f"Student's $t$-test $p$-value for $B_r$ & "
            f"${get_formatted_p_value(p_val_baseline_1)}$ \\\\"
        )

        _, p_val_baseline_2 = ttest_ind(all_values_dyads, values_baseline_2)
        f.write(
            f"Student's $t$-test $p$-value for $B_c$ & "
            f"${get_formatted_p_value(p_val_baseline_2)}$ \\\\"
        )

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def make_table_counts(df_dyads):
    with open("../data/ieee/tables/dyads/counts_dyads.tex", "w") as f:
        f.write("\\begin{table}[!t]\n")
        f.write(
            "\\caption{Breakdown of the Number of Dyads for (a) Different Intensities of Interaction and (b) Presence of Contact\\label{tab:counts_dyads}}\n"
        )
        f.write("\\centering\n")
        f.write("\\subfloat[]{")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lc}\n")
        f.write("\\toprule\n")
        f.write("Intensity of interaction & Count \\\\\n")
        f.write("\\midrule\n")

        for interaction in [0, 1, 2, 3]:

            data = df_dyads[(df_dyads["interaction"] == interaction)]

            f.write(
                f"{plot_data['interaction'][interaction]['label']} & "
                f"{len(data)} \\\\\n"
            )

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\label{tab:count_dyads_interaction}}\n\\hfil")
        f.write("\\subfloat[]{")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lc}\n")
        f.write("\\toprule\n")
        f.write("Contact & Count \\\\\n")
        f.write("\\midrule\n")
        for contact in [0, 1]:

            data = df_dyads[(df_dyads["contact"] == contact)]

            f.write(
                f"{plot_data['contact'][contact]['label']} & " f"{len(data)} \\\\\n"
            )

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\\label{tab:count_dyads_contact}}\n")

        f.write("\\end{table}\n")


def make_ssmd_table(df_dyads, metric, double=False, positive=False):
    means = []
    stds = []
    for interaction in interactions_with_baseline:
        data = df_dyads[
            (df_dyads["interaction"] == interaction)
            & (df_dyads["delta_f"] < max_delta_f)
        ]
        if double:
            values = np.concatenate([data[metric + "_1"], data[metric + "_2"]])
        else:
            values = data[metric]

        if positive:
            values = values[values > 0]

        # remove NaNs
        values = values[~np.isnan(values)]

        means.append(np.mean(values))
        stds.append(np.std(values))

    # compute SSMD
    ssmd = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i < j:
                ssmd[i, j] = np.mean(means[i] - means[j]) / np.sqrt(
                    stds[i] ** 2 + stds[j] ** 2
                )

    with open(f"../data/ieee/tables/dyads/ssmd_{metric}.tex", "w") as f:
        f.write("\\begin{table*}[!t]\n")
        f.write(
            f"\\caption{{SSMD for Pairwise Comparisons of the {plot_data['metrics'][metric]['table_label']} Between Different Intensities of Interaction\\label{{tab:ssmd_{metric}}}}}\n"
        )
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\toprule\n")
        f.write(
            " & ".join(
                [""]
                + [
                    plot_data["interaction"][i]["short_label"]
                    for i in interactions_with_baseline
                ]
            )
            + " \\\\\n"
        )
        f.write("\\midrule\n")
        for i, interaction_i in enumerate(interactions_with_baseline):
            f.write(
                f"{plot_data['interaction'][interaction_i]['short_label']} & "
                + " & ".join(
                    [
                        (
                            f"${get_latex_scientific_notation(ssmd[i][j])}$"
                            if j > i
                            else "-"
                        )
                        for j in range(6)
                    ]
                )
                + " \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")


def make_dunn_table(df_dyads, metric, double=False, positive=False):
    values_for_p_values = []
    for interaction in interactions_with_baseline:
        data = df_dyads[
            (df_dyads["interaction"] == interaction)
            & (df_dyads["delta_f"] < max_delta_f)
        ]
        if double:
            values = np.concatenate([data[metric + "_1"], data[metric + "_2"]])
        else:
            values = data[metric]

        if positive:
            values = values[values > 0]

        # remove NaNs
        values = values[~np.isnan(values)]

        values_for_p_values.append(values)

    result_dunn = sp.posthoc_dunn(values_for_p_values, p_adjust="holm")

    # print(result_dunn)

    with open(f"../data/ieee/tables/dyads/dunn_{metric}.tex", "w") as f:
        f.write("\\begin{table*}[!t]\n")
        f.write(
            f"\\caption{{Dunn Post-hoc Test for Pairwise Comparisons of the {plot_data['metrics'][metric]['table_label']} Between Baseline Pairs of $B_r$ and $B_c$ as Well as Different Intensities of Interaction. The $p$-Values Are Adjusted Using Holm--Bonferroni Correction\\label{{tab:dunn_{metric}}}}}\n"
        )
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\toprule\n")
        f.write(
            " & ".join(
                [""]
                + [
                    plot_data["interaction"][i]["short_label"]
                    for i in interactions_with_baseline
                ]
            )
            + " \\\\\n"
        )
        f.write("\\midrule\n")
        for i, interaction_i in enumerate(interactions_with_baseline):
            f.write(
                f"{plot_data['interaction'][interaction_i]['short_label']} & "
                + " & ".join(
                    [
                        (
                            f"${get_formatted_p_value(result_dunn[i+1][j+1])}$"
                            if j > i
                            else "-"
                        )
                        for j in range(6)
                    ]
                )
                + " \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")


def make_table_pearson_correlation(df_dyads, df_individuals):
    with open("../data/ieee/tables/dyads/pearson_correlation.tex", "w") as f:
        f.write("\\begin{table}[!t]\n")
        f.write(
            "\\caption{Pearson Correlation Coefficient $r_{vf}$ Between Velocity $v$ and Stride Frequency $f$ and $r_{vl}$ Between Velocity $v$ and Stride Length $l$ for Different Intensities of Interaction\\label{tab:pearson_correlation}}\n"
        )
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\toprule\n")
        f.write("Intensity of interaction & $r_{vf}$ & $r_{vl}$ \\\\\n")
        f.write("\\midrule\n")

        all_values_freq = []
        all_values_vel = []
        all_values_len = []

        for interaction in [0, 1, 2, 3]:

            data = df_dyads[df_dyads["interaction"] == interaction]

            freq = np.concatenate([data["frequency_1"], data["frequency_2"]])
            vel = np.concatenate([data["velocity_1"], data["velocity_2"]])
            length = np.concatenate([data["stride_length_1"], data["stride_length_2"]])

            # remove NaNs
            is_nan = np.isnan(freq) | np.isnan(vel) | np.isnan(length)
            freq = freq[~is_nan]
            vel = vel[~is_nan]
            length = length[~is_nan]

            all_values_freq.append(freq)
            all_values_vel.append(vel)
            all_values_len.append(length)

            r_fv, _ = pearsonr(freq, vel)
            r_lv, _ = pearsonr(length, vel)

            f.write(
                f"{plot_data['interaction'][interaction]['label']} & "
                f"${r_fv:.2f}$ & "
                f"${r_lv:.2f}$ \\\\\n"
            )

        f.write("\\midrule\n")
        # all values
        flat_freq = np.concatenate(all_values_freq)
        flat_vel = np.concatenate(all_values_vel)
        flat_len = np.concatenate(all_values_len)

        r_fv, _ = pearsonr(flat_freq, flat_vel)
        r_lv, _ = pearsonr(flat_len, flat_vel)

        f.write(f"All & " f"${r_fv:.2f}$ & " f"${r_lv:.2f}$ \\\\\n")

        # individuals
        freq_individuals = df_individuals["frequency"]
        vel_individuals = df_individuals["velocity"]
        length_individuals = df_individuals["stride_length"]

        # remove NaNs
        is_nan = (
            np.isnan(freq_individuals)
            | np.isnan(vel_individuals)
            | np.isnan(length_individuals)
        )
        freq_individuals = freq_individuals[~is_nan]
        vel_individuals = vel_individuals[~is_nan]
        length_individuals = length_individuals[~is_nan]

        r_fv, _ = pearsonr(freq_individuals, vel_individuals)
        r_lv, _ = pearsonr(length_individuals, vel_individuals)

        f.write(f"Individuals & " f"${r_fv:.2f}$ & " f"${r_lv:.2f}$ \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def fit_f(x, a, b):
    return a * x + b


if __name__ == "__main__":

    df_dyads = pd.read_csv("../data/csv/gait_data_dyads.csv")
    df_individuals = pd.read_csv("../data/csv/gait_data_individuals.csv")

    # =============================================================================
    # Parameters
    # =============================================================================

    interactions = [0, 1, 2, 3]
    interactions_with_baseline = [4, 5, 0, 1, 2, 3]
    interactions_with_close_baseline = [5, 0, 1, 2, 3]
    plot_data = {
        "interaction": {
            0: {
                "color": "blue",
                "label": "Interaction 0",
                "marker": "o",
                "short_label": "0",
            },
            1: {
                "color": "red",
                "label": "Interaction 1",
                "marker": "s",
                "short_label": "1",
            },
            2: {
                "color": "green",
                "label": "Interaction 2",
                "marker": "D",
                "short_label": "2",
            },
            3: {
                "color": "orange",
                "label": "Interaction 3",
                "marker": "^",
                "short_label": "3",
            },
            4: {
                "color": "gray",
                "label": "$B_r$",
                "marker": None,
                "short_label": "$B_r$",
            },
            5: {
                "color": "indigo",
                "label": "$B_c$",
                "marker": None,
                "short_label": "$B_c$",
            },
        },
        "contact": {
            0: {"color": "blue", "label": "No contact", "marker": "o"},
            1: {"color": "red", "label": "Contact", "marker": "s"},
            2: {"color": "purple", "label": "Baseline", "marker": "x"},
        },
        "individual": {
            "color": "turquoise",
            "label": "Individuals",
            "marker": "x",
        },
        "metrics": {
            "gsi": {"label": "GSI", "limits": [0, 0.8], "table_label": "GSI"},
            "coherence": {"label": "CWC", "limits": [0, 1], "table_label": "CWC"},
            "delta_f": {
                "label": "$\\Delta f$ [Hz]",
                "limits": [0, 0.4],
                "table_label": "difference in stride frequency $\\Delta f$",
            },
            "rec": {
                "label": "\\%REC",
                "limits": [0, 1],
                "table_label": "percentage of recurrence $\\%\\text{REC}$",
            },
            "det": {
                "label": "\\%DET",
                "limits": [0.7, 1],
                "table_label": "percentage of determinism $\\%\\text{DET}$",
            },
            "maxline": {
                "label": "MAXLINE",
                "limits": [0, 1],
                "table_label": "maximal line length $\\text{MAXLINE}$ ",
            },
            "lyapunov": {
                "label": "maximal Lyapunov exponent",
                "limits": [200 * 10 ** (-3), 1],
                "table_label": "maximal Lyapunov exponent $l_{lyap}$",
            },
            "determinism": {
                "label": "Determinism",
                "limits": [0.5, 1],
                "table_label": "determinism $D$",
            },
        },
    }

    ticks_baseline = [0, 1, 2, 3, 4, 5]

    max_delta_f = 10  # maximum difference in frequency between the two pedestrians

    # increase the font size
    plt.rcParams.update({"font.size": 14})

    # =============================================================================
    # Velocity vs. interaction
    # =============================================================================

    fig, ax = plt.subplots(figsize=(6, 4))

    min_vel = 0
    max_vel = 3
    n_bins = 16

    for interaction in interactions:
        data = df_dyads[df_dyads["interaction"] == interaction]

        values_velocity = np.concatenate([data["velocity_1"], data["velocity_2"]])

        pdf, bins = compute_pdf(values_velocity, min_vel, max_vel, n_bins)

        ax.plot(
            bins,
            pdf,
            label=plot_data["interaction"][interaction]["short_label"],
            linewidth=2,
            marker=plot_data["interaction"][interaction]["marker"],
            color=plot_data["interaction"][interaction]["color"],
        )

    # add individuals
    values_velocity_individuals = df_individuals["velocity"]
    pdf_individuals, bins_individuals = compute_pdf(
        values_velocity_individuals, min_vel, max_vel, n_bins
    )
    ax.plot(
        bins_individuals,
        pdf_individuals,
        label=plot_data["individual"]["label"],
        linewidth=2,
        marker=plot_data["individual"]["marker"],
        color=plot_data["individual"]["color"],
    )

    ax.legend()
    ax.set_xlabel("$v$ [m/s]")
    ax.set_ylabel("$p(v)$")
    ax.grid(color="lightgray", linestyle="--")

    # print mean velocity
    data_dyads = df_dyads[(df_dyads["interaction"] <= 3)]
    all_velocities = np.concatenate(
        [data_dyads["velocity_1"], data_dyads["velocity_2"]]
    )
    # interval containing X% of the data
    percentage = 90
    lower, upper = np.percentile(
        all_velocities, [50 - percentage / 2, 50 + percentage / 2]
    )
    print(f"Mean velocity: {np.mean(all_velocities):.2f} [{lower:.2f}, {upper:.2f}]")

    plt.tight_layout()
    # plt.show()
    plt.savefig("../data/ieee/figures/dyads/pdf_velocity.pdf")
    plt.close()

    # =============================================================================
    # Stride frequency vs. interaction
    # =============================================================================

    fig, ax = plt.subplots(figsize=(6, 4))

    min_freq = 0.2
    max_freq = 2
    n_bins = 16

    for interaction in interactions:
        data = df_dyads[df_dyads["interaction"] == interaction]
        values_freq = np.concatenate([data["frequency_1"], data["frequency_2"]])

        pdf, bins = compute_pdf(values_freq, min_freq, max_freq, n_bins)

        ax.plot(
            bins,
            pdf,
            label=plot_data["interaction"][interaction]["short_label"],
            linewidth=2,
            marker=plot_data["interaction"][interaction]["marker"],
            color=plot_data["interaction"][interaction]["color"],
        )

    # add individuals
    values_freq_individuals = df_individuals["frequency"]
    pdf_individuals, bins_individuals = compute_pdf(
        values_freq_individuals, min_freq, max_freq, n_bins
    )
    ax.plot(
        bins_individuals,
        pdf_individuals,
        label=plot_data["individual"]["label"],
        linewidth=2,
        marker=plot_data["individual"]["marker"],
        color=plot_data["individual"]["color"],
    )

    ax.legend()
    ax.set_xlabel("$f$ [Hz]")
    ax.set_ylabel("$p(f)$")
    ax.grid(color="lightgray", linestyle="--")

    plt.tight_layout()
    # plt.show()
    plt.savefig("../data/ieee/figures/dyads/pdf_frequency.pdf")
    plt.close()

    # =============================================================================
    # Gait length vs. interaction
    # =============================================================================

    fig, ax = plt.subplots(figsize=(6, 4))

    min_length = 0
    max_length = 3
    n_bins = 16

    for interaction in interactions:
        data = df_dyads[df_dyads["interaction"] == interaction]

        values_length = np.concatenate(
            [data["stride_length_1"], data["stride_length_2"]]
        )

        pdf, bins = compute_pdf(values_length, min_length, max_length, n_bins)

        ax.plot(
            bins,
            pdf,
            label=plot_data["interaction"][interaction]["short_label"],
            linewidth=2,
            marker=plot_data["interaction"][interaction]["marker"],
            color=plot_data["interaction"][interaction]["color"],
        )

    # add individuals
    values_length_individuals = df_individuals["stride_length"]
    pdf_individuals, bins_individuals = compute_pdf(
        values_length_individuals, min_length, max_length, n_bins
    )
    ax.plot(
        bins_individuals,
        pdf_individuals,
        label=plot_data["individual"]["label"],
        linewidth=2,
        marker=plot_data["individual"]["marker"],
        color=plot_data["individual"]["color"],
    )

    ax.legend()
    ax.set_xlabel("$l$ [m]")
    ax.set_ylabel("$p(l)$")
    ax.grid(color="lightgray", linestyle="--")

    plt.tight_layout()
    # plt.show()
    plt.savefig("../data/ieee/figures/dyads/pdf_length.pdf")
    plt.close()

    # =============================================================================
    # Stride frequency against velocity
    # =============================================================================
    fig, ax = plt.subplots(figsize=(6, 4))

    for interaction in interactions:

        data = df_dyads[df_dyads["interaction"] == interaction]

        f = np.concatenate([data["frequency_1"], data["frequency_2"]])
        v = np.concatenate([data["velocity_1"], data["velocity_2"]])

        v = v[~np.isnan(f)]
        f = f[~np.isnan(f)]

        # fit a linear model
        popt, _ = curve_fit(fit_f, v, f)

        ax.scatter(
            v,
            f,
            label=plot_data["interaction"][interaction]["short_label"],
            marker=plot_data["interaction"][interaction]["marker"],
            color=plot_data["interaction"][interaction]["color"],
            alpha=0.7,
            # linewidth=0.3,
            # edgecolor="black",
            s=5,
        )

        ax.plot(
            np.linspace(0, 3, 100),
            fit_f(np.linspace(0, 3, 100), *popt),
            color=plot_data["interaction"][interaction]["color"],
            linestyle="--",
            linewidth=2,
        )

    ax.legend(markerscale=3)
    ax.set_xlabel("$v$ [m/s]")
    ax.set_ylabel("$f$ [Hz]")
    ax.grid(color="lightgray", linestyle="--")

    plt.tight_layout()
    # plt.show()
    plt.savefig("../data/ieee/figures/dyads/f_vs_v.pdf")
    plt.close()

    # =============================================================================
    # Stride length against velocity
    # =============================================================================

    fig, ax = plt.subplots(figsize=(6, 4))

    for interaction in interactions:

        data = df_dyads[df_dyads["interaction"] == interaction]

        l = np.concatenate([data["stride_length_1"], data["stride_length_2"]])
        v = np.concatenate([data["velocity_1"], data["velocity_2"]])

        v = v[~np.isnan(l)]
        l = l[~np.isnan(l)]

        # fit a linear model
        popt, _ = curve_fit(fit_f, v, l)

        ax.scatter(
            v,
            l,
            label=plot_data["interaction"][interaction]["short_label"],
            marker=plot_data["interaction"][interaction]["marker"],
            color=plot_data["interaction"][interaction]["color"],
            alpha=0.7,
            # linewidth=0.3,
            # edgecolor="black",
            s=5,
        )

        ax.plot(
            np.linspace(0, 3, 100),
            fit_f(np.linspace(0, 3, 100), *popt),
            color=plot_data["interaction"][interaction]["color"],
            linestyle="--",
            linewidth=2,
        )

    ax.legend(markerscale=3)
    ax.set_xlabel("$v$ [m/s]")
    ax.set_ylabel("$l$ [m]")
    ax.grid(color="lightgray", linestyle="--")

    plt.tight_layout()
    # plt.show()
    plt.savefig("../data/ieee/figures/dyads/l_vs_v.pdf")
    plt.close()

    # =============================================================================
    # f1 vs f2
    # =============================================================================

    class HandlerEllipse(HandlerPatch):
        def __init__(self, markers=None, **kwargs):
            super().__init__(**kwargs)
            self.markers = markers

        def create_artists(
            self,
            legend,
            orig_handle,
            xdescent,
            ydescent,
            width,
            height,
            fontsize,
            trans,
        ):
            center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
            p = Ellipse(xy=center, width=width + xdescent, height=height + ydescent)
            self.update_prop(p, orig_handle, legend)
            p.set_transform(trans)

            # Add the marker inside the ellipse
            marker_x, marker_y = center
            marker = plt.Line2D(  # type: ignore
                [marker_x],
                [marker_y],
                marker=self.markers.get(orig_handle.get_label(), None),  # type: ignore
                color=orig_handle.get_edgecolor(),  # type: ignore
                markersize=fontsize / 4,
                linestyle="",
                transform=trans,
            )

            return [p, marker]

    fig, ax = plt.subplots(figsize=(6, 4))

    ellipses = []

    for i, interaction in enumerate(interactions_with_baseline):
        f1 = np.array(df_dyads[df_dyads["interaction"] == interaction]["frequency_1"])
        f2 = np.array(df_dyads[df_dyads["interaction"] == interaction]["frequency_2"])

        f1f2 = np.vstack((f1, f2))

        # remove NaNs
        f1f2 = f1f2[:, ~np.isnan(f1f2).any(axis=0)]

        covariance = np.cov(f1f2)
        eigenval, eigenvec = np.linalg.eig(covariance)
        # Get the largest eigenvalue
        largest_eigenval = max(eigenval)

        # Get the index of the largest eigenvector
        largest_eigenvec_ind_c = np.argwhere(eigenval == max(eigenval))[0][0]
        largest_eigenvec = eigenvec[:, largest_eigenvec_ind_c]

        # Get the smallest eigenvector and eigenvalue
        smallest_eigenval = min(eigenval)
        if largest_eigenvec_ind_c == 0:
            smallest_eigenvec = eigenvec[:, 1]
        else:
            smallest_eigenvec = eigenvec[:, 0]

        # Calculate the angle between the x-axis and the largest eigenvector
        angle = np.arctan2(largest_eigenvec[1], largest_eigenvec[0])
        if angle < 0:
            angle = angle + 2 * np.pi

        # Get the coordinates of the data mean
        avg_0 = np.nanmean(f1)
        avg_1 = np.nanmean(f2)

        # % Get the 95% confidence interval error ellipse
        chisquare_val = 2.4477
        X0 = avg_0
        Y0 = avg_1

        pearson = covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])

        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse(
            (0, 0),
            width=ell_radius_x * 2,
            height=ell_radius_y * 2,
            facecolor="none",
            edgecolor=plot_data["interaction"][interaction]["color"],
            label=plot_data["interaction"][interaction]["short_label"],
        )

        scale_x = np.sqrt(covariance[0, 0] * chisquare_val)
        mean_x = X0
        scale_y = np.sqrt(covariance[1, 1] * chisquare_val)
        mean_y = Y0

        transf = (
            transforms.Affine2D()
            .rotate_deg(45)
            .scale(scale_x, scale_y)
            .translate(mean_x, mean_y)  # type: ignore
        )
        ellipse.set_transform(transf + ax.transData)
        ax.add_patch(ellipse)

        ellipses.append(ellipse)

        if interaction <= 3:
            ax.scatter(
                f1,
                f2,
                marker=plot_data["interaction"][interaction]["marker"],  # type: ignore
                color=plot_data["interaction"][interaction]["color"],
                alpha=0.7,
                linewidth=0,
                s=8,
            )

    # plot the diagonal
    ax.plot([0, 1.6], [0, 1.6], color="black", linestyle="--")

    ax.legend(
        ellipses,
        [
            plot_data["interaction"][i]["short_label"]
            for i in interactions_with_baseline
        ],
        handler_map={
            Ellipse: HandlerEllipse(
                markers={
                    plot_data["interaction"][i]["short_label"]: plot_data[
                        "interaction"
                    ][i]["marker"]
                    for i in interactions_with_baseline
                }
            )
        },
        loc="right",
    )

    # ax.legend(markerscale=3)
    ax.set_xlabel("$f_i$ [Hz]")
    ax.set_ylabel("$f_j$ [Hz]")
    ax.grid(color="lightgray", linestyle="--")
    ax.set_aspect("equal")
    ax.set_xlim(0.5, 2.2)
    ax.set_ylim(0.5, 2)

    plt.tight_layout()
    # plt.show()
    plt.savefig("../data/ieee/figures/dyads/f1_vs_f2.pdf")
    plt.close()

    # =============================================================================
    # Delta f vs. interaction
    # =============================================================================

    fig, ax = plt.subplots(figsize=(6, 4))

    values_for_p_values = []
    means = []
    for i, interaction in enumerate(interactions_with_baseline):

        data = df_dyads[
            (df_dyads["interaction"] == interaction)
            # & (df_dyads["delta_f"] < max_delta_f)
        ]

        delta_f = np.array(data["delta_f"])

        # remove NaNs
        delta_f = delta_f[~np.isnan(delta_f)]

        if interaction <= 3:
            values_for_p_values.append(delta_f)

        mean_delta_f = np.nanmean(delta_f)
        std_delta_f = np.nanstd(delta_f)
        ste_delta_f = std_delta_f / np.sqrt(len(delta_f))

        means.append(mean_delta_f)

        ax.bar(
            i,
            mean_delta_f,
            yerr=ste_delta_f,
            color=plot_data["interaction"][interaction]["color"],
            capsize=5,
        )

    ax.set_xlabel("Intensity of interaction and baselines")
    ax.set_ylabel("$\\Delta f$ [Hz]")
    ax.set_xticks(ticks_baseline)
    ax.set_xticklabels(
        [plot_data["interaction"][i]["short_label"] for i in interactions_with_baseline]
    )
    ax.grid(color="lightgray", linestyle="--")

    # add p-values
    _, p_val = kruskal(*values_for_p_values)

    values_baseline_br = df_dyads[df_dyads["interaction"] == 4]["delta_f"]
    values_baseline_bc = df_dyads[df_dyads["interaction"] == 5]["delta_f"]
    # remove NaNs
    values_baseline_br = values_baseline_br[~np.isnan(values_baseline_br)]
    values_baseline_bc = values_baseline_bc[~np.isnan(values_baseline_bc)]

    all_values_dyads = np.concatenate(values_for_p_values)
    _, p_val_baseline_br = ttest_ind(all_values_dyads, values_baseline_br)
    _, p_val_baseline_bc = ttest_ind(all_values_dyads, values_baseline_bc)

    # bracket for p-value
    max_val = max(means)
    y = max_val * 1.1
    dh = max_val * 0.05
    ax.plot([2, 2, 5, 5], [y, y + dh, y + dh, y], color="gray", linewidth=1.5)
    ax.text(
        3.5,
        float(y) + 4 * dh,
        get_formatted_p_value_stars(p_val),
        ha="center",
        va="center",
        color="gray",
    )

    y = y + 6 * dh
    ax.plot(
        [0, 0, 3.5, 3.5],
        [y, y + dh, y + dh, y],
        color="gray",
        linewidth=1.5,
    )
    ax.text(
        (0 + 3.5) / 2,
        float(y) + 4 * dh,
        get_formatted_p_value_stars(p_val_baseline_br),
        ha="center",
        va="center",
        color="gray",
    )

    y = y + 6 * dh
    ax.plot(
        [1, 1, 3.5, 3.5],
        [y, y + dh, y + dh, y],
        color="gray",
        linewidth=1.5,
    )
    ax.text(
        (1 + 3.5) / 2,
        float(y) + 4 * dh,
        get_formatted_p_value_stars(p_val_baseline_bc),
        ha="center",
        va="center",
        color="gray",
    )

    ax.set_ylim(0, float(y) + 8 * dh)

    plt.tight_layout()
    # plt.show()
    plt.savefig("../data/ieee/figures/dyads/bar_plot_delta_f_interaction.pdf")
    plt.close()

    # =============================================================================
    # bar plot GSIs/CWC vs interaction
    # =============================================================================

    for metric in ["gsi", "coherence"]:
        fig, ax = plt.subplots(figsize=(6, 4))
        means = []
        values_for_p_values = []
        for j, interaction in enumerate(interactions_with_baseline):

            data = df_dyads[
                (df_dyads["interaction"] == interaction)
                & (df_dyads["delta_f"] < max_delta_f)
            ]

            values_metric = data[metric]

            # remove NaNs
            values_metric = values_metric[~np.isnan(values_metric)]

            if interaction <= 3:
                values_for_p_values.append(values_metric)

            mean_metric = np.mean(values_metric)
            std_metric = np.std(values_metric)
            ste_metric = std_metric / np.sqrt(len(values_metric))

            means.append(mean_metric)

            ax.bar(
                j,
                mean_metric,
                yerr=ste_metric,
                color=plot_data["interaction"][interaction]["color"],
                capsize=5,
            )

        ax.set_xlabel("Intensity of interaction and baselines")
        ax.set_ylabel(plot_data["metrics"][metric]["label"])
        ax.grid(color="lightgray", linestyle="--")
        ax.set_xticks(ticks_baseline)
        ax.set_xticklabels(
            [
                plot_data["interaction"][i]["short_label"]
                for i in interactions_with_baseline
            ]
        )

        # add p-values
        _, p_val = kruskal(*values_for_p_values)

        values_baseline_br = df_dyads[df_dyads["interaction"] == 4][metric]
        values_baseline_bc = df_dyads[df_dyads["interaction"] == 5][metric]
        # remove NaNs
        values_baseline_br = values_baseline_br[~np.isnan(values_baseline_br)]
        values_baseline_bc = values_baseline_bc[~np.isnan(values_baseline_bc)]

        all_values_dyads = np.concatenate(values_for_p_values)
        _, p_val_baseline_br = ttest_ind(all_values_dyads, values_baseline_br)
        _, p_val_baseline_bc = ttest_ind(all_values_dyads, values_baseline_bc)

        # bracket for p-value
        max_val = max(means)
        y = max_val * 1.1
        dh = max_val * 0.05
        ax.plot([2, 2, 5, 5], [y, y + dh, y + dh, y], color="gray", linewidth=1.5)
        ax.text(
            3.5,
            float(y) + 4 * dh,
            get_formatted_p_value_stars(p_val),
            ha="center",
            va="center",
            color="gray",
        )

        y = y + 6 * dh
        ax.plot(
            [0, 0, 3.5, 3.5],
            [y, y + dh, y + dh, y],
            color="gray",
            linewidth=1.5,
        )
        ax.text(
            (0 + 3.5) / 2,
            float(y) + 4 * dh,
            get_formatted_p_value_stars(p_val_baseline_br),
            ha="center",
            va="center",
            color="gray",
        )

        y = y + 6 * dh
        ax.plot(
            [1, 1, 3.5, 3.5],
            [y, y + dh, y + dh, y],
            color="gray",
            linewidth=1.5,
        )
        ax.text(
            (1 + 3.5) / 2,
            float(y) + 4 * dh,
            get_formatted_p_value_stars(p_val_baseline_bc),
            ha="center",
            va="center",
            color="gray",
        )

        ax.set_ylim(0, float(y) + 8 * dh)

        plt.tight_layout()
        # plt.show()
        plt.savefig(f"../data/ieee/figures/dyads/bar_plot_{metric}_interaction.pdf")
        plt.close()

    # =============================================================================
    # Polar histograms of the relative phase
    # =============================================================================

    fig, ax = plt.subplots(1, 6, figsize=(20, 4), subplot_kw={"projection": "polar"})

    for i, interaction in enumerate(interactions_with_baseline):

        data = df_dyads[
            (df_dyads["interaction"] == interaction)
            & (df_dyads["delta_f"] < max_delta_f)
        ]

        values_phase = data["relative_phase"]

        # remove NaNs
        values_phase = values_phase[~np.isnan(values_phase)]

        mean_phase = circmean(values_phase, high=np.pi, low=-np.pi)  # type: ignore

        ax[i].hist(
            values_phase,
            bins=64,
            density=True,
            color=plot_data["interaction"][interaction]["color"],
        )
        # show the mean
        ax[i].plot(
            [mean_phase, mean_phase],
            [0, 1],
            color="black",
            linestyle="--",
            linewidth=2,
        )
        ax[i].set_title(plot_data["interaction"][interaction]["short_label"], y=1.15)
        ax[i].set_xticks(np.array([-135, -90, -45, 0, 45, 90, 135, 180]) / 180 * np.pi)
        ax[i].set_ylim(0, 1)
        ax[i].set_thetalim(-np.pi, np.pi)

    plt.tight_layout()
    # plt.show()
    plt.savefig("../data/ieee/figures/dyads/polar_hist_relative_phase.pdf")
    plt.close()

    # =============================================================================
    # GSI wrt contact
    # =============================================================================

    for metric in ["gsi", "coherence"]:

        fig, ax = plt.subplots(figsize=(6, 4))

        for j, contact in enumerate([0, 1]):

            data = df_dyads[
                (df_dyads["contact"] == contact) & (df_dyads["delta_f"] < max_delta_f)
            ]

            values_metric = data[metric]

            mean_metric = np.mean(values_metric)
            std_metric = np.std(values_metric)
            ste_metric = std_metric / np.sqrt(len(values_metric))

            ax.bar(
                j,
                mean_metric,
                yerr=ste_metric,
                color=plot_data["contact"][contact]["color"],
                capsize=5,
            )

        # add baselines
        for j, interaction in enumerate([4, 5]):

            data = df_dyads[
                (df_dyads["interaction"] == interaction)
                & (df_dyads["delta_f"] < max_delta_f)
            ]

            values_metric = data[metric]

            mean_metric = np.mean(values_metric)
            std_metric = np.std(values_metric)
            ste_metric = std_metric / np.sqrt(len(values_metric))

            ax.bar(
                j + 2,
                mean_metric,
                yerr=ste_metric,
                color=plot_data["interaction"][interaction]["color"],
                capsize=5,
            )

        # add p-values
        data_contact_0 = df_dyads[df_dyads["contact"] == 0][metric]
        data_contact_1 = df_dyads[df_dyads["contact"] == 1][metric]

        # remove NaNs
        data_contact_0 = data_contact_0[~np.isnan(data_contact_0)]
        data_contact_1 = data_contact_1[~np.isnan(data_contact_1)]

        _, p_val = ttest_ind(
            data_contact_0,
            data_contact_1,
        )

        # bracket for p-value
        max_val = max(np.mean(data_contact_0), np.mean(data_contact_1))
        y = max_val * 1.2
        dh = max_val * 0.05
        ax.plot([0, 0, 1, 1], [y, y + dh, y + dh, y], color="gray", linewidth=1.5)
        ax.text(
            0.5,
            float(y) + 4 * dh,
            get_formatted_p_value_stars(p_val),
            ha="center",
            va="center",
            color="gray",
        )
        ax.set_ylim(0, float(y) + 8 * dh)

        ax.set_xlabel("Contact state and baselines")
        ax.set_ylabel(plot_data["metrics"][metric]["label"])
        ax.grid(color="lightgray", linestyle="--")
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["No contact", "Contact", "$B_r$", "$B_c$"])

        plt.tight_layout()
        # plt.show()
        plt.savefig(f"../data/ieee/figures/dyads/bar_plot_{metric}_contact.pdf")
        plt.close()

    # =============================================================================
    # DISTANCE ANALYSIS
    # =============================================================================

    # =============================================================================
    # Coherence vs. distance (all)
    # =============================================================================

    fig, ax = plt.subplots(figsize=(6, 4))

    n_bins = 8
    distance_min = 0.6
    distance_max = 2

    data = df_dyads[
        (df_dyads["delta_f"] < max_delta_f) & (df_dyads["interaction"] != 4)
    ]

    bin_centers, means, stds, errors, n_values = compute_binned_values(
        data["interpersonal_distance"],
        data["coherence"],
        distance_min,
        distance_max,
        n_bins,
    )

    ax.errorbar(
        bin_centers,
        means,
        yerr=errors,
        linewidth=2,
        color=plot_data["interaction"][4]["color"],
        capsize=2,
        marker=plot_data["interaction"][4]["marker"],
    )

    ax.set_xlabel("$\\delta$ [m]")
    ax.set_ylabel("CWC")
    ax.grid(color="lightgray", linestyle="--")
    ax.set_xlim(0.5, 2)

    plt.tight_layout()
    # plt.show()
    plt.savefig("../data/ieee/figures/dyads/distance_coherence_all.pdf")
    plt.close()

    # =============================================================================
    # Distance vs. interaction
    # =============================================================================

    fig, ax = plt.subplots(figsize=(6, 4))

    distance_min = 0.5
    distance_max = 2
    n_bins = 5

    for interaction in interactions_with_close_baseline:

        data = df_dyads[
            (df_dyads["interaction"] == interaction)
            & (df_dyads["delta_f"] < max_delta_f)
        ]

        pdf, bins = compute_pdf(
            data["interpersonal_distance"], distance_min, distance_max, n_bins
        )

        ax.plot(
            bins,
            pdf,
            label=plot_data["interaction"][interaction]["short_label"],
            linewidth=2,
            marker=plot_data["interaction"][interaction]["marker"],
            color=plot_data["interaction"][interaction]["color"],
        )

    ax.legend()
    ax.set_xlabel("$\\delta$ [m]")
    ax.set_ylabel("$p(\\delta)$")
    ax.grid(color="lightgray", linestyle="--")
    ax.set_xlim(0.5, 2)

    plt.tight_layout()
    # plt.show()
    plt.savefig("../data/ieee/figures/dyads/distance_coherence_pd.pdf")
    plt.close()

    # =============================================================================
    # Coherence vs. distance (wrt interaction)
    # =============================================================================

    fig, ax = plt.subplots(figsize=(6, 4))

    n_bins = 5

    for interaction in interactions_with_close_baseline:

        data = df_dyads[
            (df_dyads["interaction"] == interaction)
            & (df_dyads["delta_f"] < max_delta_f)
        ]

        bin_centers, means, stds, errors, n_values = compute_binned_values(
            data["interpersonal_distance"],
            data["coherence"],
            distance_min,
            distance_max,
            n_bins,
        )

        ax.errorbar(
            bin_centers,
            means,
            yerr=errors,
            linewidth=2,
            marker=plot_data["interaction"][interaction]["marker"],
            color=plot_data["interaction"][interaction]["color"],
            label=plot_data["interaction"][interaction]["short_label"],
            capsize=2,
        )

    ax.legend()
    ax.set_xlabel("$\\delta$ [m]")
    ax.set_ylabel("CWC")
    ax.grid(color="lightgray", linestyle="--")
    ax.set_xlim(0, 2)

    plt.tight_layout()
    # plt.show()
    plt.savefig("../data/ieee/figures/dyads/distance_coherence_interaction.pdf")
    plt.close()

    # =============================================================================
    # NON LINEAR ANALYSIS
    # =============================================================================

    # =============================================================================
    # Determism
    # =============================================================================

    fig, ax = plt.subplots(figsize=(6, 4))

    values_for_p_values = []
    means = []
    for i, interaction in enumerate(interactions_with_baseline):
        data = df_dyads[
            (df_dyads["interaction"] == interaction)
            & (df_dyads["delta_f"] < max_delta_f)
        ]

        values_1 = data["determinism_1"]
        values_2 = data["determinism_2"]

        values_det = np.concatenate([values_1, values_2])

        # remove NaNs
        values_det = values_det[~np.isnan(values_det)]

        if interaction <= 3:
            values_for_p_values.append(values_det)

        mean_det = np.nanmean(values_det)
        print(interaction, mean_det)
        std_det = np.nanstd(values_det)
        ste_det = std_det / np.sqrt(len(values_det))

        means.append(mean_det)

        ax.bar(
            i,
            mean_det,
            yerr=ste_det,
            color=plot_data["interaction"][interaction]["color"],
            capsize=5,
        )

    ax.set_xlabel("Intensity of interaction and baselines")
    ax.set_ylabel("$D$")
    ax.grid(color="lightgray", linestyle="--")
    ax.set_xticks(ticks_baseline)
    ax.set_xticklabels(
        [plot_data["interaction"][i]["short_label"] for i in interactions_with_baseline]
    )

    # add p-values
    _, p_val_det = kruskal(*values_for_p_values)

    values_baseline_br = np.concatenate(
        [
            df_dyads[
                (df_dyads["interaction"] == 4) & (df_dyads["delta_f"] < max_delta_f)
            ]["determinism_1"],
            df_dyads[
                (df_dyads["interaction"] == 4) & (df_dyads["delta_f"] < max_delta_f)
            ]["determinism_2"],
        ]
    )
    values_baseline_bc = np.concatenate(
        [
            df_dyads[
                (df_dyads["interaction"] == 5) & (df_dyads["delta_f"] < max_delta_f)
            ]["determinism_1"],
            df_dyads[
                (df_dyads["interaction"] == 5) & (df_dyads["delta_f"] < max_delta_f)
            ]["determinism_2"],
        ]
    )
    # remove NaNs
    values_baseline_br = values_baseline_br[~np.isnan(values_baseline_br)]
    values_baseline_bc = values_baseline_bc[~np.isnan(values_baseline_bc)]

    all_values_dyads = np.concatenate(values_for_p_values)
    _, p_val_baseline_br = ttest_ind(all_values_dyads, values_baseline_br)
    _, p_val_baseline_bc = ttest_ind(all_values_dyads, values_baseline_bc)

    # print(p_val_det)
    # bracket for p-value
    max_val = max(means)
    y = max_val * 1.1
    dh = (max_val - plot_data["metrics"]["determinism"]["limits"][0]) * 0.05

    ax.plot([2, 2, 5, 5], [y, y + dh, y + dh, y], color="gray", linewidth=1.5)
    ax.text(
        3.5,
        float(y) + 4 * dh,
        get_formatted_p_value_stars(p_val_det),
        ha="center",
        va="center",
        color="gray",
    )
    y = y + 6 * dh
    ax.plot(
        [0, 0, 3.5, 3.5],
        [y, y + dh, y + dh, y],
        color="gray",
        linewidth=1.5,
    )
    ax.text(
        (0 + 3.5) / 2,
        float(y) + 4 * dh,
        get_formatted_p_value_stars(p_val_baseline_br),
        ha="center",
        va="center",
        color="gray",
    )

    y = y + 6 * dh
    ax.plot(
        [1, 1, 3.5, 3.5],
        [y, y + dh, y + dh, y],
        color="gray",
        linewidth=1.5,
    )
    ax.text(
        (1 + 3.5) / 2,
        float(y) + 4 * dh,
        get_formatted_p_value_stars(p_val_baseline_bc),
        ha="center",
        va="center",
        color="gray",
    )
    ax.set_ylim(plot_data["metrics"]["determinism"]["limits"][0], float(y) + 8 * dh)

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"../data/ieee/figures/dyads/nla_determinism.pdf")

    # =============================================================================
    # Lyapunov exponents
    # =============================================================================

    fig, ax = plt.subplots(figsize=(6, 4))

    values_for_p_values = []
    means = []
    for i, interaction in enumerate(interactions_with_baseline):

        data = df_dyads[
            (df_dyads["interaction"] == interaction)
            & (df_dyads["delta_f"] < max_delta_f)
        ]

        values_1 = data["lyapunov_1"]
        values_2 = data["lyapunov_2"]

        values_lyap = np.concatenate([values_1, values_2])

        # remove NaNs
        values_lyap = values_lyap[~np.isnan(values_lyap)]

        percentage = (np.sum(values_lyap > 0)) / len(values_lyap)
        # print(f"Interaction {interaction}: {percentage * 100:.2f}% > 0")

        if interaction <= 3:
            values_for_p_values.append(values_lyap)

        mean_lyap = np.nanmean(values_lyap)
        std_lyap = np.nanstd(values_lyap)
        ste_lyap = std_lyap / np.sqrt(len(values_lyap))

        means.append(mean_lyap)

        ax.bar(
            i,
            mean_lyap,
            yerr=ste_lyap,
            color=plot_data["interaction"][interaction]["color"],
            capsize=5,
        )

    ax.set_xlabel("Intensity of interaction and baselines")

    # needs asmath for \times

    ax.set_ylabel("$l_{lyap}$ ($\\times 10^{-3}$)")
    ax.grid(color="lightgray", linestyle="--")
    ax.set_xticks(ticks_baseline)
    ax.set_xticklabels(
        [plot_data["interaction"][i]["short_label"] for i in interactions_with_baseline]
    )

    # format the axis to show only 1 decimal and put exponent in the axis label
    def scale_formatter(value, _):
        return f"{value * 1e3:.1f}"  # Multiply by 1e5 to scale

    ax.yaxis.set_major_formatter(FuncFormatter(scale_formatter))

    # add p-values
    _, p_val_lyap = kruskal(*values_for_p_values)

    values_baseline_br = np.concatenate(
        [
            df_dyads[
                (df_dyads["interaction"] == 4) & (df_dyads["delta_f"] < max_delta_f)
            ]["lyapunov_1"],
            df_dyads[
                (df_dyads["interaction"] == 4) & (df_dyads["delta_f"] < max_delta_f)
            ]["lyapunov_2"],
        ]
    )
    values_baseline_bc = np.concatenate(
        [
            df_dyads[
                (df_dyads["interaction"] == 5) & (df_dyads["delta_f"] < max_delta_f)
            ]["lyapunov_1"],
            df_dyads[
                (df_dyads["interaction"] == 5) & (df_dyads["delta_f"] < max_delta_f)
            ]["lyapunov_2"],
        ]
    )

    # remove NaNs
    values_baseline_br = values_baseline_br[~np.isnan(values_baseline_br)]
    values_baseline_bc = values_baseline_bc[~np.isnan(values_baseline_bc)]

    all_values_dyads = np.concatenate(values_for_p_values)
    _, p_val_baseline_br = ttest_ind(all_values_dyads, values_baseline_br)
    _, p_val_baseline_bc = ttest_ind(all_values_dyads, values_baseline_bc)

    # bracket for p-value
    max_val = max(means)
    y = max_val * 1.1
    dh = (max_val - plot_data["metrics"]["lyapunov"]["limits"][0]) * 0.05
    ax.plot([2, 2, 5, 5], [y, y + dh, y + dh, y], color="gray", linewidth=1.5)
    ax.text(
        3.5,
        float(y) + 4 * dh,
        get_formatted_p_value_stars(p_val_lyap),
        ha="center",
        va="center",
        color="gray",
    )
    y = y + 6 * dh
    ax.plot(
        [0, 0, 3.5, 3.5],
        [y, y + dh, y + dh, y],
        color="gray",
        linewidth=1.5,
    )
    ax.text(
        (0 + 3.5) / 2,
        float(y) + 4 * dh,
        get_formatted_p_value_stars(p_val_baseline_br),
        ha="center",
        va="center",
        color="gray",
    )

    y = y + 6 * dh
    ax.plot(
        [1, 1, 3.5, 3.5],
        [y, y + dh, y + dh, y],
        color="gray",
        linewidth=1.5,
    )
    ax.text(
        (1 + 3.5) / 2,
        float(y) + 4 * dh,
        get_formatted_p_value_stars(p_val_baseline_bc),
        ha="center",
        va="center",
        color="gray",
    )
    ax.set_ylim(plot_data["metrics"]["lyapunov"]["limits"][0], float(y) + 8 * dh)

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"../data/ieee/figures/dyads/nla_lyapunov.pdf")

    # =============================================================================
    # CRQ
    # =============================================================================

    # rec, det, maxline
    for metric in ["rec", "det", "maxline"]:

        fig, ax = plt.subplots(figsize=(6, 4))

        values_for_p_values = []
        means = []
        for i, interaction in enumerate(interactions_with_baseline):

            data = df_dyads[
                (df_dyads["interaction"] == interaction)
                & (df_dyads["delta_f"] < max_delta_f)
            ]

            values_metric = data[metric]

            # remove NaNs
            values_metric = values_metric[~np.isnan(values_metric)]

            if interaction <= 3:
                values_for_p_values.append(values_metric)

            mean_rec = np.nanmean(values_metric)
            std_rec = np.nanstd(values_metric)
            ste_rec = std_rec / np.sqrt(len(values_metric))

            means.append(mean_rec)

            ax.bar(
                i,
                mean_rec,
                yerr=ste_rec,
                color=plot_data["interaction"][interaction]["color"],
                capsize=5,
            )

        ax.set_xlabel("Intensity of interaction and baselines")
        ax.set_ylabel(plot_data["metrics"][metric]["label"])
        ax.grid(color="lightgray", linestyle="--")
        ax.set_xticks(ticks_baseline)
        ax.set_xticklabels(
            [
                plot_data["interaction"][i]["short_label"]
                for i in interactions_with_baseline
            ]
        )

        # add p-values
        _, p_val_rec = kruskal(*values_for_p_values)
        # bracket for p-value
        max_val = max(means)
        y = max_val * 1.1

        dh = (max_val - plot_data["metrics"][metric]["limits"][0]) * 0.05

        ax.plot([2, 2, 5, 5], [y, y + dh, y + dh, y], color="gray", linewidth=1.5)
        ax.text(
            3.5,
            float(y) + 4 * dh,
            get_formatted_p_value_stars(p_val_rec),
            ha="center",
            va="center",
            color="gray",
        )

        y = y + 6 * dh
        ax.plot(
            [0, 0, 3.5, 3.5],
            [y, y + dh, y + dh, y],
            color="gray",
            linewidth=1.5,
        )
        ax.text(
            (0 + 3.5) / 2,
            float(y) + 4 * dh,
            get_formatted_p_value_stars(p_val_baseline_br),
            ha="center",
            va="center",
            color="gray",
        )

        y = y + 6 * dh
        ax.plot(
            [1, 1, 3.5, 3.5],
            [y, y + dh, y + dh, y],
            color="gray",
            linewidth=1.5,
        )
        ax.text(
            (1 + 3.5) / 2,
            float(y) + 4 * dh,
            get_formatted_p_value_stars(p_val_baseline_bc),
            ha="center",
            va="center",
            color="gray",
        )

        ax.set_ylim(plot_data["metrics"][metric]["limits"][0], float(y) + 8 * dh)

        plt.tight_layout()
        # plt.show()
        plt.savefig(f"../data/ieee/figures/dyads/nla_{metric}.pdf")

    # =============================================================================
    # Prints
    # =============================================================================

    # print count number of stationary samples

    for interaction in interactions:
        data = df_dyads[(df_dyads["interaction"] == interaction)]

        stationary_1 = data["stationarity_1"]
        stationary_2 = data["stationarity_2"]

        # remove NaNs
        stationary_1 = stationary_1.dropna()
        stationary_2 = stationary_2.dropna()

        all_stationary = np.concatenate([stationary_1, stationary_2])

        print(
            f"Interaction {interaction}: {np.sum(all_stationary)} stationary samples out of {len(all_stationary)}"
        )

    # =============================================================================
    # Tables
    # =============================================================================

    make_table_counts(df_dyads)
    make_gait_stats_table(df_dyads, df_individuals)
    make_table_delta_f(df_dyads)
    make_relative_phase_table(df_dyads)
    make_table_gsi_interaction(df_dyads)
    make_table_coherence_interaction(df_dyads)
    make_tukey_table(df_dyads)
    make_table_gsi_coherence_contact(df_dyads)
    make_table_pearson_correlation(df_dyads, df_individuals)

    make_dunn_table(df_dyads, "delta_f")
    make_dunn_table(df_dyads, "gsi")
    make_dunn_table(df_dyads, "coherence")
    make_dunn_table(df_dyads, "rec")
    make_dunn_table(df_dyads, "det")
    make_dunn_table(df_dyads, "maxline")
    make_dunn_table(df_dyads, "lyapunov", double=True)  # positive=True)
    make_dunn_table(df_dyads, "determinism", double=True)

    make_ssmd_table(df_dyads, "delta_f")
    make_ssmd_table(df_dyads, "gsi")
    make_ssmd_table(df_dyads, "coherence")
    make_ssmd_table(df_dyads, "rec")
    make_ssmd_table(df_dyads, "det")
    make_ssmd_table(df_dyads, "maxline")
    make_ssmd_table(df_dyads, "lyapunov", double=True)  # positive=True)
    make_ssmd_table(df_dyads, "determinism", double=True)
