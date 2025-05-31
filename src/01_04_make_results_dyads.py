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

from parameters import DYADS_PARAMETERS, MAX_DELTA_F


def make_gait_stats_table(df_dyads, df_individuals):
    with open("../data/tables/dyads/gait_stats.tex", "w") as f:
        f.write("\\begin{table}\n\n")
        f.write("\\centering\n")
        f.write(
            "\\caption{Mean value and standard deviation of velocity $v$, stride frequency $f$, and stride length $l$ for different intensities of interaction. Kruskal-Wallis $p$-values for the difference between the intensities of interaction and Student's $t$-test $p$-values for the difference between all dyads and individuals are also shown.}\n"
        )
        f.write("\\label{tab:gait_stats}\n")
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
                    f"{DYADS_PARAMETERS['interaction'][interaction]['label']} & "
                    f"${np.mean(values_velocity):.2f} \\pm {np.std(values_velocity):.2f}$ & \\multirow{{4}}{{*}}{{${get_formatted_p_value(p_val_vel)}$}} &"
                    f"${np.mean(values_frequency):.2f} \\pm {np.std(values_frequency):.2f}$ & \\multirow{{4}}{{*}}{{${get_formatted_p_value(p_val_freq)}$}} &"
                    f"${np.mean(values_length):.2f} \\pm {np.std(values_length):.2f}$ & \\multirow{{4}}{{*}}{{${get_formatted_p_value(p_val_len)}$}} \\\\\n"
                )
            else:
                f.write(
                    f"{DYADS_PARAMETERS['interaction'][interaction]['label']} & "
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
        f.write("\\end{table}\n")


def make_table_gsi_interaction(df_dyads):
    with open("../data/tables/dyads/gsi_interaction.tex", "w") as f:
        f.write("\\begin{table}\n")
        f.write(
            "\\caption{GSI for different intensities of interaction and baselines. Kruskal-Wallis $p$-values for the difference between the intensities of interaction and Student's $t$-test $p$-values for the difference between all dyads and the baselines are also shown\\label{tab:coherence_interaction}}\n"
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
                & (df_dyads["delta_f"] < MAX_DELTA_F)
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
                & (df_dyads["delta_f"] < MAX_DELTA_F)
            ]
            values_gsi = data["gsi"]

            # remove NaNs
            values_gsi = values_gsi[~np.isnan(values_gsi)]

            if interaction == 0:
                f.write(
                    f"{DYADS_PARAMETERS['interaction'][interaction]['label']} & "
                    f"${np.mean(values_gsi):.2f} \\pm {np.std(values_gsi):.2f}$ & \\multirow{{4}}{{*}}{{${get_formatted_p_value(p_val_gsi)}$}} \\\\\n"
                )
            else:
                f.write(
                    f"{DYADS_PARAMETERS['interaction'][interaction]['label']} & "
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
            (df_dyads["interaction"] == 4) & (df_dyads["delta_f"] < MAX_DELTA_F)
        ]["gsi"]
        values_gsi_baseline_2 = df_dyads[
            (df_dyads["interaction"] == 5) & (df_dyads["delta_f"] < MAX_DELTA_F)
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
    with open("../data/tables/dyads/coherence_interaction.tex", "w") as f:
        f.write("\\begin{table}\n")
        f.write("\\centering\n")
        f.write(
            "\\caption{CWC for different intensities of interaction and baselines. Kruskal-Wallis $p$-values for the difference between the intensities of interaction and Student's $t$-test $p$-values for the difference between all dyads and the baselines are also shown\\label{tab:coherence_interaction}}\n"
        )
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\toprule\n")
        f.write(" & \\multicolumn{2}{c}{CWC} \\\\\n")
        f.write("\\midrule\n")

        # compute p-values
        values_pvalues = []
        for interaction in [0, 1, 2, 3]:
            data = df_dyads[
                (df_dyads["interaction"] == interaction)
                & (df_dyads["delta_f"] < MAX_DELTA_F)
            ]
            values_coherence = data["coherence"]

            # remove NaNs
            values_coherence = values_coherence[~np.isnan(values_coherence)]

            values_pvalues.append(values_coherence)

        all_values_coherence = np.concatenate(values_pvalues)

        # p-values for Kruskal-Wallis
        _, p_val_coherence = kruskal(*values_pvalues)

        for interaction in [4, 5, 0, 1, 2, 3]:

            data = df_dyads[
                (df_dyads["interaction"] == interaction)
                & (df_dyads["delta_f"] < MAX_DELTA_F)
            ]
            values_coherence = data["coherence"]

            # remove NaNs
            values_coherence = values_coherence[~np.isnan(values_coherence)]

            if interaction == 0:
                f.write(
                    f"{DYADS_PARAMETERS['interaction'][interaction]['label']} & "
                    f"${np.mean(values_coherence):.2f} \\pm {np.std(values_coherence):.2f}$ & \\multirow{{4}}{{*}}{{${get_formatted_p_value(p_val_coherence)}$}} \\\\\n"
                )
            else:
                f.write(
                    f"{DYADS_PARAMETERS['interaction'][interaction]['label']} & "
                    f"${np.mean(values_coherence):.2f} \\pm {np.std(values_coherence):.2f}$ & \\\\\n"
                )

            if interaction == 5:
                f.write("\\midrule\n")
                f.write(
                    f"All & "
                    f"${np.mean(all_values_coherence):.2f} \\pm {np.std(all_values_coherence) / np.sqrt(len(all_values_coherence)):.2f}$ & \\\\\n"
                )
                f.write("\\midrule\n")

        f.write("\\midrule\n")

        # all vs baseline
        values_coherence_baseline_1 = df_dyads[
            (df_dyads["interaction"] == 4) & (df_dyads["delta_f"] < MAX_DELTA_F)
        ]["coherence"]
        values_coherence_baseline_2 = df_dyads[
            (df_dyads["interaction"] == 5) & (df_dyads["delta_f"] < MAX_DELTA_F)
        ]["coherence"]

        # remove NaNs
        values_coherence_baseline_1 = values_coherence_baseline_1[
            ~np.isnan(values_coherence_baseline_1)
        ]
        values_coherence_baseline_2 = values_coherence_baseline_2[
            ~np.isnan(values_coherence_baseline_2)
        ]

        _, p_val_coherence_1 = ttest_ind(
            all_values_coherence, values_coherence_baseline_1
        )
        _, p_val_coherence_2 = ttest_ind(
            all_values_coherence, values_coherence_baseline_2
        )

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


def make_table_gsi_contact(df_dyads):
    with open("../data/tables/dyads/gsi_contact.tex", "w") as f:
        f.write("\\begin{table}\n")
        f.write("\\centering\n")
        f.write(
            "\\caption{GSI for different levels of contact. Student's $t$-test $p$-values for the difference between the levels of contact are also shown.}\n"
        )
        f.write("\\label{tab:gsi_contact}\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\toprule\n")
        f.write("Contact state & GSI (Mean $\\pm$ SD) & $p$-value \\\\ \n")
        f.write("\\midrule\n")

        all_values_gsi = []

        for contact in [0, 1]:
            values_gsi = df_dyads[(df_dyads["contact"] == contact)]["gsi"].dropna()
            all_values_gsi.append(values_gsi)

            f.write(
                f"{DYADS_PARAMETERS['contact'][contact]['label']} & "
                f"${values_gsi.mean():.2f} \pm {values_gsi.std():.2f}$ & "
                f"\\\\ \n"
            )

        _, p_val_gsi = ttest_ind(*all_values_gsi)
        f.write(f"\\midrule\n")
        f.write(f"p-value & & ${get_formatted_p_value(p_val_gsi)}$ \\\\ \n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def make_table_coherence_contact(df_dyads):
    with open("../data/tables/dyads/coherence_contact.tex", "w") as f:
        f.write("\\begin{table}\n")
        f.write("\\centering\n")
        f.write(
            "\\caption{Coherence for different levels of contact. Student's $t$-test $p$-values for the difference between the levels of contact are also shown.}\n"
        )
        f.write("\\label{tab:coherence_contact}\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\toprule\n")
        f.write("Contact state & CWC (Mean $\\pm$ SD) & $p$-value \\\\ \n")
        f.write("\\midrule\n")

        all_values_coherence = []

        for contact in [0, 1]:
            values_coherence = df_dyads[(df_dyads["contact"] == contact)][
                "coherence"
            ].dropna()
            all_values_coherence.append(values_coherence)

            f.write(
                f"{DYADS_PARAMETERS['contact'][contact]['label']} & "
                f"${values_coherence.mean():.2f} \pm {values_coherence.std():.2f}$ & "
                f"\\\\ \n"
            )

        _, p_val_coherence = ttest_ind(*all_values_coherence)
        f.write(f"\\midrule\n")
        f.write(f"p-value & & ${get_formatted_p_value(p_val_coherence)}$ \\\\ \n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def make_tukey_table(df_dyads):

    with open("../data/tables/dyads/tukey.tex", "w") as f:

        all_values_gsi = []
        all_values_coherence = []

        for interaction in interactions_with_baseline:
            data = df_dyads[
                (df_dyads["interaction"] == interaction)
                & (df_dyads["delta_f"] < MAX_DELTA_F)
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

        f.write("\\begin{table}\n")
        f.write("\\centering\n")
        f.write(
            "\\caption{Tukey's HSD test for pairwise comparisons of GSI between different intensities of interaction.}\n"
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
                f"{DYADS_PARAMETERS['interaction'][interaction_i]['label']} & "
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

        f.write("\\begin{table}\n")
        f.write("\\centering\n")
        f.write(
            "\\caption{Tukey's HSD test for pairwise comparisons of coherence between different intensities of interaction.}\n"
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
                f"{DYADS_PARAMETERS['interaction'][interaction_i]['label']} & "
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
    with open("../data/tables/dyads/relative_phase.tex", "w") as f:
        f.write("\\begin{table}\n")
        f.write("\\centering\n")
        f.write(
            "\\caption{Circular mean and variance of the relative phase between pedestrians for different intensities of interaction and baseline.}\n"
        )
        f.write("\\label{tab:relative_phase}\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\toprule\n")
        f.write("& Mean relative phase (°) & Variance \\\\\n")
        f.write("\\midrule\n")

        for interaction in [4, 5, 0, 1, 2, 3]:

            data = df_dyads[
                (df_dyads["interaction"] == interaction)
                & (df_dyads["delta_f"] < MAX_DELTA_F)
            ]
            values_phase = data["relative_phase"]

            # remove NaNs
            values_phase = values_phase[~np.isnan(values_phase)]

            mean_phase = np.rad2deg(circmean(values_phase, high=np.pi, low=-np.pi))  # type: ignore
            var_phase = circvar(values_phase, high=np.pi, low=-np.pi)  # type: ignore

            f.write(
                f"{DYADS_PARAMETERS['interaction'][interaction]['label']} & "
                f"${mean_phase:.2f}$ & "
                f"${var_phase:.2f}$ \\\\\n"
            )

            if interaction == 5:
                f.write("\\midrule\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def make_table_delta_f(df_dyads):
    with open("../data/tables/dyads/delta_f.tex", "w") as f:
        f.write("\\begin{table}\n")
        f.write("\\centering\n")
        f.write(
            "\\caption{Mean and standard error of the difference in stride frequency $\\Delta f$ between baseline pairs of $B_r$ and $B_c$ as well as dyad members for different intensities of interaction. Kruskal-Wallis $p$-value for the difference between the intensities of interaction and Student's $t$-test $p$-value for the difference between all dyads and the baseline are also shown.}\n"
        )
        f.write("\\label{tab:delta_f}\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\toprule\n")
        f.write("& \\multicolumn{2}{c}{$\\Delta f$ [Hz]}  \\\\\n")
        f.write("\\midrule\n")

        # compute the p-values
        values_pvalue = []
        for interaction in [0, 1, 2, 3]:
            data = df_dyads[(df_dyads["interaction"] == interaction)]
            values_delta_f = data["delta_f"]
            # remove NaNs
            values_delta_f = values_delta_f[~np.isnan(values_delta_f)]
            values_pvalue.append(values_delta_f)

        all_values_dyads = np.concatenate(values_pvalue)

        # p-value for Kruskal-Wallis
        _, p_val_dyads = kruskal(*values_pvalue)

        for interaction in [4, 5, 0, 1, 2, 3]:

            data = df_dyads[(df_dyads["interaction"] == interaction)]
            values_delta_f = data["delta_f"]

            # remove NaNs
            values_delta_f = values_delta_f[~np.isnan(values_delta_f)]

            mean_delta_f = np.mean(values_delta_f)
            std_delta_f = np.std(values_delta_f)
            ste_delta_f = std_delta_f / np.sqrt(len(values_delta_f))

            if interaction == 0:
                f.write(
                    f"{DYADS_PARAMETERS['interaction'][interaction]['label']} &"
                    f"${mean_delta_f:.2f} \\pm {ste_delta_f:.2f}$  & \\multirow{{4}}{{*}}{{${get_formatted_p_value(p_val_dyads)}$}} \\\\\n"
                )
            else:
                f.write(
                    f"{DYADS_PARAMETERS['interaction'][interaction]['label']} &"
                    f"${mean_delta_f:.2f} \\pm {ste_delta_f:.2f}$ & \\\\\n"
                )

            if interaction == 5:
                f.write("\\midrule\n")
                # all values together
                f.write(
                    f"All & "
                    f"${np.mean(all_values_dyads):.2f} \\pm {np.std(all_values_dyads) / np.sqrt(len(all_values_dyads)):.2f}$ & \\\\\n"
                )
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


def make_table_counts_dyads_interaction(df_dyads):
    with open("../data/tables/dyads/counts_dyads_interaction.tex", "w") as f:
        f.write("\\begin{table}\n")
        f.write("\\centering\n")
        f.write(
            "\\caption{Breakdown of the number of dyads for different intensities of interaction.}\n"
        )
        f.write("\\label{tab:counts_dyads_interaction}\n")
        f.write("\\begin{tabular}{lc}\n")
        f.write("\\toprule\n")
        f.write("Intensity of interaction & Count \\\\\n")
        f.write("\\midrule\n")

        for interaction in [0, 1, 2, 3]:

            data = df_dyads[(df_dyads["interaction"] == interaction)]

            f.write(
                f"{DYADS_PARAMETERS['interaction'][interaction]['label']} & "
                f"{len(data)} \\\\\n"
            )

        f.write("\\bottomrule\n")

        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def make_table_counts_dyads_contact(df_dyads):
    with open("../data/tables/dyads/counts_dyads_contact.tex", "w") as f:
        f.write("\\begin{table}\n")
        f.write("\\centering\n")
        f.write(
            "\\caption{Breakdown of the number of dyads with and without contact.}\n"
        )
        f.write("\\label{tab:counts_dyads_contacts}\n")
        f.write("\\begin{tabular}{lc}\n")
        f.write("\\toprule\n")
        f.write("Contact & Count \\\\\n")
        f.write("\\midrule\n")
        for contact in [0, 1]:

            data = df_dyads[(df_dyads["contact"] == contact)]

            f.write(
                f"{DYADS_PARAMETERS['contact'][contact]['label']} & "
                f"{len(data)} \\\\\n"
            )

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def make_ssmd_table(df_dyads, metric, double=False, positive=False):
    means = []
    stds = []
    for interaction in interactions_with_baseline:
        data = df_dyads[
            (df_dyads["interaction"] == interaction)
            & (df_dyads["delta_f"] < MAX_DELTA_F)
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

    with open(f"../data/tables/dyads/ssmd_{metric}.tex", "w") as f:
        f.write("\\begin{table}\n")
        f.write("\\centering\n")
        f.write(
            f"\\caption{{SSMD for pairwise comparisons of the {DYADS_PARAMETERS['metrics'][metric]['table_label']} the baselines and dyads with different intensities of interaction.}}\n"
        )
        f.write(f"\\label{{tab:ssmd_{metric}}}\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\toprule\n")
        f.write(
            " & ".join(
                [""]
                + [
                    DYADS_PARAMETERS["interaction"][i]["short_label"]
                    for i in interactions_with_baseline
                ]
            )
            + " \\\\\n"
        )
        f.write("\\midrule\n")
        for i, interaction_i in enumerate(interactions_with_baseline):
            f.write(
                f"{DYADS_PARAMETERS['interaction'][interaction_i]['short_label']} & "
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
        f.write("\\end{table}\n")


def make_dunn_table(df_dyads, metric, double=False, positive=False):
    values_for_p_values = []
    for interaction in interactions_with_baseline:
        data = df_dyads[
            (df_dyads["interaction"] == interaction)
            & (df_dyads["delta_f"] < MAX_DELTA_F)
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

    with open(f"../data/tables/dyads/dunn_{metric}.tex", "w") as f:
        f.write("\\begin{table}\n")
        f.write("\\centering\n")
        f.write(
            f"\\caption{{Dunn post-hoc test for pairwise comparisons of the {DYADS_PARAMETERS['metrics'][metric]['table_label']} between baseline pairs of $B_r$ and $B_c$ as well as different intensities of interaction. The $p$-values are adjusted using the Bonferroni correction.}}\n"
        )
        f.write(f"\\label{{tab:dunn_{metric}}}\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\toprule\n")
        f.write(
            " & ".join(
                [""]
                + [
                    DYADS_PARAMETERS["interaction"][i]["short_label"]
                    for i in interactions_with_baseline
                ]
            )
            + " \\\\\n"
        )
        f.write("\\midrule\n")
        for i, interaction_i in enumerate(interactions_with_baseline):
            f.write(
                f"{DYADS_PARAMETERS['interaction'][interaction_i]['short_label']} & "
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
        f.write("\\end{table}\n")


def make_table_pearson_correlation(df_dyads, df_individuals):
    with open("../data/tables/dyads/pearson_correlation.tex", "w") as f:
        f.write("\\begin{table}\n")
        f.write("\\centering\n")
        f.write(
            "\\caption{Pearson correlation coefficient $r_{vf}$ between velocity $v$ and stride frequency $f$ and $r_{vl}$ between velocity $v$ and stride length $l$ for different intensities of interaction.}\n"
        )
        f.write("\\label{tab:pearson_correlation}\n")
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
                f"{DYADS_PARAMETERS['interaction'][interaction]['label']} & "
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

    ticks_baseline = [0, 1, 2, 3, 4, 5]

    # increase the font size
    plt.rcParams.update({"font.size": 14})

    fig, ax = plt.subplots(1, 3, figsize=(18, 4))

    # =============================================================================
    # Velocity vs. interaction
    # =============================================================================

    min_vel = 0
    max_vel = 3
    n_bins = 16

    for interaction in interactions:
        data = df_dyads[df_dyads["interaction"] == interaction]

        values_velocity = np.concatenate([data["velocity_1"], data["velocity_2"]])

        pdf, bins = compute_pdf(values_velocity, min_vel, max_vel, n_bins)

        ax[0].plot(
            bins,
            pdf,
            label=DYADS_PARAMETERS["interaction"][interaction]["short_label"],
            linewidth=2,
            marker=DYADS_PARAMETERS["interaction"][interaction]["marker"],
            color=DYADS_PARAMETERS["interaction"][interaction]["color"],
        )

    # add individuals
    values_velocity_individuals = df_individuals["velocity"]
    pdf_individuals, bins_individuals = compute_pdf(
        values_velocity_individuals, min_vel, max_vel, n_bins
    )
    ax[0].plot(
        bins_individuals,
        pdf_individuals,
        label=DYADS_PARAMETERS["individual"]["label"],
        linewidth=2,
        marker=DYADS_PARAMETERS["individual"]["marker"],
        color=DYADS_PARAMETERS["individual"]["color"],
    )

    ax[0].legend()
    ax[0].set_xlabel("$v$ [m/s]")
    ax[0].set_ylabel("$p(v)$")
    ax[0].grid(color="lightgray", linestyle="--")
    ax[0].set_title("(a)", y=-0.4)

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

    # =============================================================================
    # Stride frequency vs. interaction
    # =============================================================================

    min_freq = 0.2
    max_freq = 2
    n_bins = 16

    for interaction in interactions:
        data = df_dyads[df_dyads["interaction"] == interaction]
        values_freq = np.concatenate([data["frequency_1"], data["frequency_2"]])

        pdf, bins = compute_pdf(values_freq, min_freq, max_freq, n_bins)

        ax[1].plot(
            bins,
            pdf,
            label=DYADS_PARAMETERS["interaction"][interaction]["short_label"],
            linewidth=2,
            marker=DYADS_PARAMETERS["interaction"][interaction]["marker"],
            color=DYADS_PARAMETERS["interaction"][interaction]["color"],
        )

    # add individuals
    values_freq_individuals = df_individuals["frequency"]
    pdf_individuals, bins_individuals = compute_pdf(
        values_freq_individuals, min_freq, max_freq, n_bins
    )
    ax[1].plot(
        bins_individuals,
        pdf_individuals,
        label=DYADS_PARAMETERS["individual"]["label"],
        linewidth=2,
        marker=DYADS_PARAMETERS["individual"]["marker"],
        color=DYADS_PARAMETERS["individual"]["color"],
    )

    ax[1].legend()
    ax[1].set_xlabel("$f$ [Hz]")
    ax[1].set_ylabel("$p(f)$")
    ax[1].grid(color="lightgray", linestyle="--")
    ax[1].set_title("(b)", y=-0.4)

    # =============================================================================
    # Gait length vs. interaction
    # =============================================================================

    min_length = 0
    max_length = 3
    n_bins = 16

    for interaction in interactions:
        data = df_dyads[df_dyads["interaction"] == interaction]

        values_length = np.concatenate(
            [data["stride_length_1"], data["stride_length_2"]]
        )

        pdf, bins = compute_pdf(values_length, min_length, max_length, n_bins)

        ax[2].plot(
            bins,
            pdf,
            label=DYADS_PARAMETERS["interaction"][interaction]["short_label"],
            linewidth=2,
            marker=DYADS_PARAMETERS["interaction"][interaction]["marker"],
            color=DYADS_PARAMETERS["interaction"][interaction]["color"],
        )

    # add individuals
    values_length_individuals = df_individuals["stride_length"]
    pdf_individuals, bins_individuals = compute_pdf(
        values_length_individuals, min_length, max_length, n_bins
    )
    ax[2].plot(
        bins_individuals,
        pdf_individuals,
        label=DYADS_PARAMETERS["individual"]["label"],
        linewidth=2,
        marker=DYADS_PARAMETERS["individual"]["marker"],
        color=DYADS_PARAMETERS["individual"]["color"],
    )

    ax[2].legend()
    ax[2].set_xlabel("$l$ [m]")
    ax[2].set_ylabel("$p(l)$")
    ax[2].grid(color="lightgray", linestyle="--")
    ax[2].set_title("(c)", y=-0.4)

    plt.tight_layout()
    # plt.show()
    plt.savefig("../data/figures/dyads/gait_stats.pdf")
    plt.close()

    # =============================================================================
    # Stride frequency against velocity
    # =============================================================================
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    for interaction in interactions:

        data = df_dyads[df_dyads["interaction"] == interaction]

        f = np.concatenate([data["frequency_1"], data["frequency_2"]])
        v = np.concatenate([data["velocity_1"], data["velocity_2"]])

        v = v[~np.isnan(f)]
        f = f[~np.isnan(f)]

        # fit a linear model
        popt, _ = curve_fit(fit_f, v, f)

        ax[0].scatter(
            v,
            f,
            label=DYADS_PARAMETERS["interaction"][interaction]["short_label"],
            marker=DYADS_PARAMETERS["interaction"][interaction]["marker"],
            color=DYADS_PARAMETERS["interaction"][interaction]["color"],
            alpha=0.7,
            # linewidth=0.3,
            # edgecolor="black",
            s=5,
        )

        ax[0].plot(
            np.linspace(0, 3, 100),
            fit_f(np.linspace(0, 3, 100), *popt),
            color=DYADS_PARAMETERS["interaction"][interaction]["color"],
            linestyle="--",
            linewidth=2,
        )

    ax[0].legend(markerscale=3)
    ax[0].set_xlabel("$v$ [m/s]")
    ax[0].set_ylabel("$f$ [Hz]")
    ax[0].grid(color="lightgray", linestyle="--")
    ax[0].set_title("(a)", y=-0.35)

    # =============================================================================
    # Stride length against velocity
    # =============================================================================

    for interaction in interactions:

        data = df_dyads[df_dyads["interaction"] == interaction]

        l = np.concatenate([data["stride_length_1"], data["stride_length_2"]])
        v = np.concatenate([data["velocity_1"], data["velocity_2"]])

        v = v[~np.isnan(l)]
        l = l[~np.isnan(l)]

        # fit a linear model
        popt, _ = curve_fit(fit_f, v, l)

        ax[1].scatter(
            v,
            l,
            label=DYADS_PARAMETERS["interaction"][interaction]["short_label"],
            marker=DYADS_PARAMETERS["interaction"][interaction]["marker"],
            color=DYADS_PARAMETERS["interaction"][interaction]["color"],
            alpha=0.7,
            # linewidth=0.3,
            # edgecolor="black",
            s=5,
        )

        ax[1].plot(
            np.linspace(0, 3, 100),
            fit_f(np.linspace(0, 3, 100), *popt),
            color=DYADS_PARAMETERS["interaction"][interaction]["color"],
            linestyle="--",
            linewidth=2,
        )

    ax[1].legend(markerscale=3)
    ax[1].set_xlabel("$v$ [m/s]")
    ax[1].set_ylabel("$l$ [m]")
    ax[1].grid(color="lightgray", linestyle="--")
    ax[1].set_title("(b)", y=-0.35)

    plt.tight_layout()
    # plt.show()
    plt.savefig("../data/figures/dyads/gait_stats_correlation.pdf")
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
            edgecolor=DYADS_PARAMETERS["interaction"][interaction]["color"],
            label=DYADS_PARAMETERS["interaction"][interaction]["short_label"],
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
                marker=DYADS_PARAMETERS["interaction"][interaction]["marker"],  # type: ignore
                color=DYADS_PARAMETERS["interaction"][interaction]["color"],
                alpha=0.7,
                linewidth=0,
                s=8,
            )

    # plot the diagonal
    ax.plot([0, 1.6], [0, 1.6], color="black", linestyle="--")

    ax.legend(
        ellipses,
        [
            DYADS_PARAMETERS["interaction"][i]["short_label"]
            for i in interactions_with_baseline
        ],
        handler_map={
            Ellipse: HandlerEllipse(
                markers={
                    DYADS_PARAMETERS["interaction"][i]["short_label"]: DYADS_PARAMETERS[
                        "interaction"
                    ][i]["marker"]
                    for i in interactions_with_baseline
                }
            )
        },
        loc="right",
    )

    # ax[0].legend(markerscale=3)
    ax.set_xlabel("$f_i$ [Hz]")
    ax.set_ylabel("$f_j$ [Hz]")
    ax.grid(color="lightgray", linestyle="--")
    ax.set_aspect("equal")
    ax.set_xlim(0.5, 2.2)
    ax.set_ylim(0.5, 2)
    # ax.set_title("(a)", y=-0.35)

    plt.tight_layout()
    # plt.show()
    plt.savefig("../data/figures/dyads/frequency_analysis_f1_f2.pdf")

    # =============================================================================
    # Delta f vs. interaction
    # =============================================================================

    fig, ax = plt.subplots(figsize=(6, 4))

    values_for_p_values = []
    means = []
    for i, interaction in enumerate(interactions_with_baseline):

        data = df_dyads[
            (df_dyads["interaction"] == interaction)
            # & (df_dyads["delta_f"] < MAX_DELTA_F)
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
            color=DYADS_PARAMETERS["interaction"][interaction]["color"],
            capsize=5,
        )

    # ax.set_xlabel("Intensity of interaction and baselines")
    ax.set_ylabel("$\\Delta f$ [Hz]")
    ax.set_xticks(ticks_baseline)
    ax.set_xticklabels(
        [
            DYADS_PARAMETERS["interaction"][i]["short_label"]
            for i in interactions_with_baseline
        ]
    )
    ax.grid(color="lightgray", linestyle="--")
    # ax.set_title("(b)", y=-0.35)

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

    # remove ticks
    ax.tick_params(bottom=False, left=False, right=False, top=False, which="both")

    plt.tight_layout()
    # plt.show()
    plt.savefig("../data/figures/dyads/frequency_analysis_delta_f.pdf")
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
                & (df_dyads["delta_f"] < MAX_DELTA_F)
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
                color=DYADS_PARAMETERS["interaction"][interaction]["color"],
                capsize=5,
            )

        # ax.set_xlabel("Intensity of interaction and baselines")
        ax.set_ylabel(DYADS_PARAMETERS["metrics"][metric]["label"])
        ax.grid(color="lightgray", linestyle="--")
        ax.set_xticks(ticks_baseline)
        ax.set_xticklabels(
            [
                DYADS_PARAMETERS["interaction"][i]["short_label"]
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

        # remove ticks
        ax.tick_params(bottom=False, left=False, right=False, top=False, which="both")

        plt.tight_layout()
        # plt.show()
        plt.savefig(f"../data/figures/dyads/bar_plot_{metric}_interaction.pdf")
        plt.close()

    # =============================================================================
    # Polar histograms of the relative phase
    # =============================================================================

    for i, interaction in enumerate(interactions_with_baseline):

        fig, ax = plt.subplots(figsize=(5, 4), subplot_kw={"projection": "polar"})

        data = df_dyads[
            (df_dyads["interaction"] == interaction)
            & (df_dyads["delta_f"] < MAX_DELTA_F)
        ]

        values_phase = data["relative_phase"]

        # remove NaNs
        values_phase = values_phase[~np.isnan(values_phase)]

        mean_phase = circmean(values_phase, high=np.pi, low=-np.pi)  # type: ignore

        ax.hist(
            values_phase,
            bins=64,
            density=True,
            color=DYADS_PARAMETERS["interaction"][interaction]["color"],
        )
        # show the mean
        ax.plot(
            [mean_phase, mean_phase],
            [0, 1],
            color="black",
            linestyle="--",
            linewidth=2,
        )
        # ax.set_title(DYADS_PARAMETERS["interaction"][interaction]["short_label"], y=1.1)
        ax.set_xticks(np.array([-135, -90, -45, 0, 45, 90, 135, 180]) / 180 * np.pi)
        ax.set_ylim(0, 1)
        ax.set_thetalim(-np.pi, np.pi)

        plt.tight_layout()
        # plt.show()
        plt.savefig(
            f"../data/figures/dyads/polar_no_title/polar_hist_relative_phase_{interaction}.pdf"
        )
        plt.close()

    # =============================================================================
    # GSI wrt contact
    # =============================================================================

    for metric in ["gsi", "coherence"]:

        fig, ax = plt.subplots(figsize=(6, 4))

        for j, contact in enumerate([0, 1]):

            data = df_dyads[
                (df_dyads["contact"] == contact) & (df_dyads["delta_f"] < MAX_DELTA_F)
            ]

            values_metric = data[metric]

            mean_metric = np.mean(values_metric)
            std_metric = np.std(values_metric)
            ste_metric = std_metric / np.sqrt(len(values_metric))

            ax.bar(
                j,
                mean_metric,
                yerr=ste_metric,
                color=DYADS_PARAMETERS["contact"][contact]["color"],
                capsize=5,
            )

        # add baselines
        for j, interaction in enumerate([4, 5]):

            data = df_dyads[
                (df_dyads["interaction"] == interaction)
                & (df_dyads["delta_f"] < MAX_DELTA_F)
            ]

            values_metric = data[metric]

            mean_metric = np.mean(values_metric)
            std_metric = np.std(values_metric)
            ste_metric = std_metric / np.sqrt(len(values_metric))

            ax.bar(
                j + 2,
                mean_metric,
                yerr=ste_metric,
                color=DYADS_PARAMETERS["interaction"][interaction]["color"],
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
        ax.set_ylabel(DYADS_PARAMETERS["metrics"][metric]["label"])
        ax.grid(color="lightgray", linestyle="--")
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["No contact", "Contact", "$B_r$", "$B_c$"])

        plt.tight_layout()
        # plt.show()
        plt.savefig(f"../data/figures/dyads/bar_plot_{metric}_contact.pdf")
        plt.close()

    # =============================================================================
    # DISTANCE ANALYSIS
    # =============================================================================

    # =============================================================================
    # Coherence vs. distance (all)
    # =============================================================================

    fig, axes = plt.subplots(figsize=(6, 3.5))

    n_bins = 8
    distance_min = 0.6
    distance_max = 2

    data = df_dyads[
        (df_dyads["delta_f"] < MAX_DELTA_F) & (df_dyads["interaction"] != 4)
    ]

    bin_centers, means, stds, errors, n_values = compute_binned_values(
        data["interpersonal_distance"],
        data["coherence"],
        distance_min,
        distance_max,
        n_bins,
    )

    axes.errorbar(
        bin_centers,
        means,
        yerr=errors,
        linewidth=2,
        color=DYADS_PARAMETERS["interaction"][4]["color"],
        capsize=2,
        marker=DYADS_PARAMETERS["interaction"][4]["marker"],
    )

    axes.set_xlabel("$\\delta$ [m]")
    axes.set_ylabel("CWC")
    axes.grid(color="lightgray", linestyle="--")
    axes.set_xlim(0.5, 2)
    # axes.set_title("(a)", y=-0.35)

    plt.tight_layout()
    # plt.show()
    plt.savefig("../data/figures/dyads/distance/coherence_distance_all.pdf")

    # =============================================================================
    # Distance vs. interaction
    # =============================================================================

    fig, axes = plt.subplots(figsize=(6, 3.5))

    distance_min = 0.5
    distance_max = 2
    n_bins = 5

    for interaction in interactions_with_close_baseline:

        data = df_dyads[
            (df_dyads["interaction"] == interaction)
            & (df_dyads["delta_f"] < MAX_DELTA_F)
        ]

        pdf, bins = compute_pdf(
            data["interpersonal_distance"], distance_min, distance_max, n_bins
        )

        axes.plot(
            bins,
            pdf,
            label=DYADS_PARAMETERS["interaction"][interaction]["short_label"],
            linewidth=2,
            marker=DYADS_PARAMETERS["interaction"][interaction]["marker"],
            color=DYADS_PARAMETERS["interaction"][interaction]["color"],
        )

    axes.legend()
    axes.set_xlabel("$\\delta$ [m]")
    axes.set_ylabel("$p(\\delta)$")
    axes.grid(color="lightgray", linestyle="--")
    axes.set_xlim(0.5, 2)
    # axes.set_title("(c)", y=-0.35)

    plt.tight_layout()
    # plt.show()
    plt.savefig("../data/figures/dyads/distance/distance_interaction.pdf")

    # =============================================================================
    # Coherence vs. distance (wrt interaction)
    # =============================================================================

    fig, axes = plt.subplots(figsize=(6, 3.5))

    n_bins = 5

    for interaction in interactions_with_close_baseline:

        data = df_dyads[
            (df_dyads["interaction"] == interaction)
            & (df_dyads["delta_f"] < MAX_DELTA_F)
        ]

        bin_centers, means, stds, errors, n_values = compute_binned_values(
            data["interpersonal_distance"],
            data["coherence"],
            distance_min,
            distance_max,
            n_bins,
        )

        axes.errorbar(
            bin_centers,
            means,
            yerr=errors,
            linewidth=2,
            marker=DYADS_PARAMETERS["interaction"][interaction]["marker"],
            color=DYADS_PARAMETERS["interaction"][interaction]["color"],
            label=DYADS_PARAMETERS["interaction"][interaction]["short_label"],
            capsize=2,
        )

    axes.legend()
    axes.set_xlabel("$\\delta$ [m]")
    axes.set_ylabel("CWC")
    axes.grid(color="lightgray", linestyle="--")
    axes.set_xlim(0, 2)
    # axes.set_title("(e)", y=-0.35)

    plt.tight_layout()
    # plt.show()
    plt.savefig("../data/figures/dyads/distance/coherence_distance_interaction.pdf")
    plt.close()

    # =============================================================================
    # Determism
    # =============================================================================

    fig, ax = plt.subplots(figsize=(6, 3.5))

    values_for_p_values = []
    means = []
    for i, interaction in enumerate(interactions_with_baseline):
        data = df_dyads[
            (df_dyads["interaction"] == interaction)
            & (df_dyads["delta_f"] < MAX_DELTA_F)
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
            color=DYADS_PARAMETERS["interaction"][interaction]["color"],
            capsize=5,
        )

    # ax.set_xlabel("Intensity of interaction and baselines")
    ax.set_ylabel("$D$")
    ax.grid(color="lightgray", linestyle="--")
    ax.set_xticks(ticks_baseline)
    ax.set_xticklabels(
        [
            DYADS_PARAMETERS["interaction"][i]["short_label"]
            for i in interactions_with_baseline
        ]
    )

    # add p-values
    _, p_val_det = kruskal(*values_for_p_values)

    values_baseline_br = np.concatenate(
        [
            df_dyads[
                (df_dyads["interaction"] == 4) & (df_dyads["delta_f"] < MAX_DELTA_F)
            ]["determinism_1"],
            df_dyads[
                (df_dyads["interaction"] == 4) & (df_dyads["delta_f"] < MAX_DELTA_F)
            ]["determinism_2"],
        ]
    )
    values_baseline_bc = np.concatenate(
        [
            df_dyads[
                (df_dyads["interaction"] == 5) & (df_dyads["delta_f"] < MAX_DELTA_F)
            ]["determinism_1"],
            df_dyads[
                (df_dyads["interaction"] == 5) & (df_dyads["delta_f"] < MAX_DELTA_F)
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
    dh = (max_val - DYADS_PARAMETERS["metrics"]["determinism"]["limits"][0]) * 0.05

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
    ax.set_ylim(
        DYADS_PARAMETERS["metrics"]["determinism"]["limits"][0], float(y) + 8 * dh
    )

    # remove ticks
    ax.tick_params(bottom=False, left=False, right=False, top=False, which="both")

    plt.tight_layout()
    # plt.show()
    plt.savefig("../data/figures/dyads/nonlinear_analysis/determinism.pdf")
    plt.close()

    # =============================================================================
    # Lyapunov exponents
    # =============================================================================

    fig, ax = plt.subplots(figsize=(6, 3.5))

    values_for_p_values = []
    means = []
    for i, interaction in enumerate(interactions_with_baseline):

        data = df_dyads[
            (df_dyads["interaction"] == interaction)
            & (df_dyads["delta_f"] < MAX_DELTA_F)
        ]

        values_1 = data["lyapunov_1"]
        values_2 = data["lyapunov_2"]

        values_lyap = np.concatenate([values_1, values_2])

        # remove NaNs
        values_lyap = values_lyap[~np.isnan(values_lyap)]

        percentage = (np.sum(values_lyap > 0)) / len(values_lyap)
        print(f"Interaction {interaction}: {percentage * 100:.2f}% > 0")

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
            color=DYADS_PARAMETERS["interaction"][interaction]["color"],
            capsize=5,
        )

    # ax.set_xlabel("Intensity of interaction and baselines")

    # needs asmath for \times

    ax.set_ylabel("$l_{lyap}$ ($\\times 10^{-3}$)")
    ax.grid(color="lightgray", linestyle="--")
    ax.set_xticks(ticks_baseline)
    ax.set_xticklabels(
        [
            DYADS_PARAMETERS["interaction"][i]["short_label"]
            for i in interactions_with_baseline
        ]
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
                (df_dyads["interaction"] == 4) & (df_dyads["delta_f"] < MAX_DELTA_F)
            ]["lyapunov_1"],
            df_dyads[
                (df_dyads["interaction"] == 4) & (df_dyads["delta_f"] < MAX_DELTA_F)
            ]["lyapunov_2"],
        ]
    )
    values_baseline_bc = np.concatenate(
        [
            df_dyads[
                (df_dyads["interaction"] == 5) & (df_dyads["delta_f"] < MAX_DELTA_F)
            ]["lyapunov_1"],
            df_dyads[
                (df_dyads["interaction"] == 5) & (df_dyads["delta_f"] < MAX_DELTA_F)
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
    dh = (max_val - DYADS_PARAMETERS["metrics"]["lyapunov"]["limits"][0]) * 0.05
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
    ax.set_ylim(DYADS_PARAMETERS["metrics"]["lyapunov"]["limits"][0], float(y) + 8 * dh)
    # ax.set_title("(b)", y=-0.35)

    # remove ticks
    ax.tick_params(bottom=False, left=False, right=False, top=False, which="both")

    plt.tight_layout()
    # plt.show()
    plt.savefig("../data/figures/dyads/nonlinear_analysis/lyapunov.pdf")
    plt.close()

    # =============================================================================
    # CRQ
    # =============================================================================

    # rec, det, maxline
    for j, metric in enumerate(["rec", "det", "maxline"]):

        fig, ax = plt.subplots(figsize=(6, 3.5))

        values_for_p_values = []
        means = []
        for i, interaction in enumerate(interactions_with_baseline):

            data = df_dyads[
                (df_dyads["interaction"] == interaction)
                & (df_dyads["delta_f"] < MAX_DELTA_F)
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
                color=DYADS_PARAMETERS["interaction"][interaction]["color"],
                capsize=5,
            )

        # ax.set_xlabel("Intensity of interaction and baselines")
        ax.set_ylabel(DYADS_PARAMETERS["metrics"][metric]["label"])
        ax.grid(color="lightgray", linestyle="--")
        ax.set_xticks(ticks_baseline)
        ax.set_xticklabels(
            [
                DYADS_PARAMETERS["interaction"][i]["short_label"]
                for i in interactions_with_baseline
            ]
        )

        # add p-values
        _, p_val_rec = kruskal(*values_for_p_values)
        # bracket for p-value
        max_val = max(means)
        y = max_val * 1.1

        dh = (max_val - DYADS_PARAMETERS["metrics"][metric]["limits"][0]) * 0.05

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

        ax.set_ylim(DYADS_PARAMETERS["metrics"][metric]["limits"][0], float(y) + 8 * dh)

        # remove ticks
        ax.tick_params(bottom=False, left=False, right=False, top=False, which="both")

        plt.tight_layout()
        # plt.show()
        plt.savefig(f"../data/figures/dyads/nonlinear_analysis/{metric}.pdf")
        plt.close()

    # =============================================================================
    # Prints
    # =============================================================================

    print("Number of dyads:")
    for interaction in interactions_with_baseline:
        data = df_dyads[df_dyads["interaction"] == interaction]
        print(f"Interaction {interaction}: {len(data)} dyads")

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

    make_table_counts_dyads_interaction(df_dyads)
    make_table_counts_dyads_contact(df_dyads)
    make_gait_stats_table(df_dyads, df_individuals)
    make_table_delta_f(df_dyads)
    make_relative_phase_table(df_dyads)
    make_table_gsi_interaction(df_dyads)
    make_table_coherence_interaction(df_dyads)
    make_tukey_table(df_dyads)
    make_table_gsi_contact(df_dyads)
    make_table_coherence_contact(df_dyads)
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
