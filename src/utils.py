import numpy as np
from statsmodels.tsa.stattools import adfuller

import pickle as pk

from parameters import (
    VEL_MIN,
    VEL_MAX,
    BOUNDARIES_ATC_CORRIDOR,
    BOUNDARIES_DIAMOR_CORRIDOR,
    SOCIAL_RELATIONS_EN,
    INTENSITIES_OF_INTERACTION_NUM,
    COLORS_SOC_REL,
    COLORS_INTERACTION,
    GROUP_BREADTH_MIN,
    GROUP_BREADTH_MAX,
    DAYS_ATC,
    DAYS_DIAMOR,
)

from pedestrians_social_binding.threshold import Threshold
from pedestrians_social_binding.trajectory_utils import (
    compute_interpersonal_distance,
    compute_depth_and_breadth,
    compute_trajectory_direction,
    compute_stride_parameters,
    compute_average_synchronization,
    compute_coherence,
    compute_lyapunov_exponent,
    check_determinism,
    compute_gait_residual,
    compute_phase_embedding,
    compute_optimal_delay,
    compute_optimal_embedding_dimension,
    compute_rqa,
)


def pickle_load(file_path: str):
    """Load the content of a pickle file

    Arguments:
    ----------
        file_path {str} -- The path to the file which will be unpickled

    Return
    ------
        obj -- The content of the pickle file
    """
    with open(file_path, "rb") as f:
        data = pk.load(f)
    return data


def pickle_save(file_path: str, data):
    """Save data to pickle file

    Arguments:
    ----------
        file_path {str} -- The path to the file where the data will be saved
        data {obj} -- The data to save
    """
    with open(file_path, "wb") as f:
        pk.dump(data, f)


def get_pedestrian_thresholds(env_name):
    """Return the thresholds for the pedestrians

    Arguments:
    ----------
        env_name {str} -- name of the environment

    Return
    ------
        thresholds_indiv {list} -- list of thresholds
    """

    thresholds_indiv = []
    thresholds_indiv += [
        Threshold("v", min=VEL_MIN, max=VEL_MAX)
    ]  # velocity in [0.5; 3]m/s
    # thresholds_indiv += [Threshold("d", min=D_MIN)]  # walk at least 5 m
    thresholds_indiv += [Threshold("n", min=16)]
    # thresholds_indiv += [Threshold("theta", max=THETA_MAX)]  # no 90° turns

    # corridor threshold for ATC
    if env_name == "atc:corridor":
        thresholds_indiv += [
            Threshold(
                "x", BOUNDARIES_ATC_CORRIDOR["xmin"], BOUNDARIES_ATC_CORRIDOR["xmax"]
            )
        ]
        thresholds_indiv += [
            Threshold(
                "y", BOUNDARIES_ATC_CORRIDOR["ymin"], BOUNDARIES_ATC_CORRIDOR["ymax"]
            )
        ]
    elif env_name == "diamor:corridor":
        thresholds_indiv += [
            Threshold(
                "x",
                BOUNDARIES_DIAMOR_CORRIDOR["xmin"],
                BOUNDARIES_DIAMOR_CORRIDOR["xmax"],
            )
        ]
        thresholds_indiv += [
            Threshold(
                "y",
                BOUNDARIES_DIAMOR_CORRIDOR["ymin"],
                BOUNDARIES_DIAMOR_CORRIDOR["ymax"],
            )
        ]

    return thresholds_indiv


def get_social_values(env_name):
    """Return the social values for the environment

    Arguments:
    ----------
        env_name {str} -- name of the environment

    Return
    ------
        social_values {list} -- list of social values
    """
    if "atc" in env_name:
        return "soc_rel", SOCIAL_RELATIONS_EN, [2, 1, 3, 4], COLORS_SOC_REL
    elif "diamor" in env_name:
        return (
            "interaction",
            INTENSITIES_OF_INTERACTION_NUM,
            [0, 1, 2, 3],
            COLORS_INTERACTION,
        )
    else:
        raise ValueError(f"Unknown env {env_name}")


def get_groups_thresholds():
    """Return the thresholds for the groups

    Return
    ------
        group_thresholds {list} -- list of thresholds
    """
    # threshold on the distance between the group members, max 4 m
    group_thresholds = [
        Threshold("delta", min=GROUP_BREADTH_MIN, max=GROUP_BREADTH_MAX)
    ]
    return group_thresholds


def get_all_days(env_name):
    """Return the days for the environment

    Arguments:
    ----------
        env_name {str} -- name of the environment

    Return
    ------
        days {list} -- list of days
    """
    if "atc" in env_name:
        return DAYS_ATC
    elif "diamor" in env_name:
        return DAYS_DIAMOR
    else:
        raise ValueError(f"Unknown env {env_name}")


def get_interaction(group):
    """Return the interaction type of the group

    Arguments:
    ----------
        group {Group} -- group

    Return
    ------
        conversation {int} -- 1 if the group is in a conversation
        contact {int} -- 1 if the group is in contact
    """

    id_A = group.get_members()[0].get_id()
    id_B = group.get_members()[1].get_id()

    conversation, contact = 0, 0
    if id_A in group.annotations["interactions"]:
        interaction_A = group.annotations["interactions"][id_A]["interaction_type"]
        if interaction_A[0] == 1:
            conversation = 1
        if interaction_A[3] == 1:
            contact = 1
    if id_B in group.annotations["interactions"]:
        interaction_B = group.annotations["interactions"][id_B]["interaction_type"]
        if interaction_B[0] == 1:
            conversation = 1
        if interaction_B[3] == 1:
            contact = 1
    return conversation, contact


def check_stationarity(x):
    """Check the stationarity of a time series.

    Arguments:
    ----------
        x {list} -- time series

    Return
    ------
        True if the p-value for the Augmented Dickey-Fuller is smaller than 0.05, else False
    """

    p_value_adf = adfuller(x, autolag="AIC")[1]

    return p_value_adf < 0.05


def compute_synchronisation_data_pair(
    traj_A,
    traj_B,
    sampling_time,
    power_threshold,
    n_fft,
    min_freq,
    max_freq,
    window_duration,
    simult=True,
    nonlinear_parameters_compute=False,
):
    """Compute the synchronization data between two pedestrians

    Arguments:
    ----------
        traj_A {np.array} -- trajectory of pedestrian A
        traj_B {np.array} -- trajectory of pedestrian B
        sampling_time {float} -- sampling time
        power_threshold {float} -- power threshold
        n_fft {int} -- number of points for the FFT
        min_freq {float} -- minimum frequency
        max_freq {float} -- maximum frequency
        window_duration {float} -- window duration
        simult {bool} -- if True, time values are synchronized

    Return
    ------
        vel_A {float} -- velocity of pedestrian A
        vel_B {float} -- velocity of pedestrian B
        direction_A {float} -- direction of pedestrian A
        direction_B {float} -- direction of pedestrian B
        d {float} -- interpersonal distance
        depth {float} -- depth of the group
        breadth {float} -- breadth of the group
        stride_frequency_A {float} -- stride frequency of pedestrian A
        stride_frequency_B {float} -- stride frequency of pedestrian B
        swaying_A {float} -- swaying of pedestrian A
        swaying_B {float} -- swaying of pedestrian B
        stride_length_A {float} -- stride length of pedestrian A
        stride_length_B {float} -- stride length of pedestrian B
        delta_f {float} -- difference in stride frequency
        mean_gsi_h {float} -- mean GSI
        mean_relative_phase_h {float} -- mean relative phase
        variance_relative_phase_h {float} -- variance of the relative phase
        mean_coherence {float} -- mean coherence
        stationarity_A {bool} -- stationarity of pedestrian A
        stationarity_B {bool} -- stationarity of pedestrian B
        lyapunov_A {float} -- Lyapunov exponent of pedestrian A
        lyapunov_B {float} -- Lyapunov exponent of pedestrian B
        determinism_A {float} -- determinism of pedestrian A
        determinism_B {float} -- determinism of pedestrian B
        rec {float} -- recurrence rate
        det {float} -- determinism
        maxline {float} -- longest diagonal line
        tau_A {int} -- optimal delay of pedestrian A
        m_A {int} -- optimal embedding dimension of pedestrian A
        tau_B {int} -- optimal delay of pedestrian B
        m_B {int} -- optimal embedding dimension of pedestrian B
    """

    vel_A = np.mean(np.linalg.norm(traj_A[:, 5:7], axis=1)) / 1000
    vel_B = np.mean(np.linalg.norm(traj_B[:, 5:7], axis=1)) / 1000

    d = np.mean(compute_interpersonal_distance(traj_A, traj_B)) / 1000

    if simult:
        depth_values, breadth_values = compute_depth_and_breadth(traj_A, traj_B)
        depth = np.mean(depth_values) / 1000
        breadth = np.mean(breadth_values) / 1000
    else:
        depth, breadth = None, None

    # ============================
    # Directions
    # ============================

    direction_A = compute_trajectory_direction(traj_A)
    direction_B = compute_trajectory_direction(traj_B)

    # ============================
    # Stride and swaying
    # ============================

    stride_frequency_A, swaying_A, stride_length_A = compute_stride_parameters(
        traj_A,
        power_threshold=power_threshold,
        n_fft=n_fft,
        min_f=min_freq,
        max_f=max_freq,
    )
    stride_frequency_B, swaying_B, stride_length_B = compute_stride_parameters(
        traj_B,
        power_threshold=power_threshold,
        n_fft=n_fft,
        min_f=min_freq,
        max_f=max_freq,
    )

    if stride_frequency_A is None or stride_frequency_B is None:
        delta_f = None
    else:
        delta_f = np.abs(stride_frequency_A - stride_frequency_B)

    if stride_length_A is not None:
        stride_length_A = stride_length_A / 1000
    if stride_length_B is not None:
        stride_length_B = stride_length_B / 1000

    # ============================
    # Synchronization with GSI
    # ============================

    mean_gsi_h, mean_relative_phase_h, variance_relative_phase_h, _ = (
        compute_average_synchronization(
            traj_A,
            traj_B,
            window_duration=window_duration,
            power_threshold=power_threshold,
            min_freq=min_freq,
            max_freq=max_freq,
        )
    )

    # ============================
    # Coherence
    # ============================
    gait_residual_A = compute_gait_residual(traj_A)
    gait_residual_B = compute_gait_residual(traj_B)

    if gait_residual_A is None or gait_residual_B is None:
        mean_coherence = None
    else:
        mean_coherence = compute_coherence(
            gait_residual_A,
            gait_residual_B,
            sampling_time=sampling_time,
            min_freq=min_freq,
            max_freq=max_freq,
        )

    # ============================
    # Stationarity
    # ============================
    if gait_residual_A is None:
        stationarity_A = None
    else:
        stationarity_A = check_stationarity(gait_residual_A)

    if gait_residual_B is None:
        stationarity_B = None
    else:
        stationarity_B = check_stationarity(gait_residual_B)

    # ============================
    # Lyapunov exponent and determinism
    # ============================
    if gait_residual_A is None:
        lyapunov_A = None
        determinism_A = None
    else:
        embedding_A = compute_phase_embedding(gait_residual_A, 4, 7)

        lyapunov_A = compute_lyapunov_exponent(
            embedding_A,
            n_iterations=5,
            n_neighbors=1,
            n_points=100,
            eps=0.03,
            theiler_window=5,
        )
        determinism_A = check_determinism(embedding_A, n_boxes=5, min_val_box=3)

    if gait_residual_B is None:
        lyapunov_B = None
        determinism_B = None
    else:
        embedding_B = compute_phase_embedding(gait_residual_B, 4, 7)
        lyapunov_B = compute_lyapunov_exponent(
            embedding_B,
            n_iterations=5,
            n_neighbors=1,
            n_points=100,
            eps=0.03,
            theiler_window=5,
        )
        determinism_B = check_determinism(embedding_B, n_boxes=5, min_val_box=3)

    # ============================
    # RQA
    # ============================

    if gait_residual_A is None or gait_residual_B is None:
        rec = None
        det = None
        maxline = None
    else:
        rec, det, maxline = compute_rqa(
            gait_residual_A,
            gait_residual_B,
            embedding_dimension=4,
            time_delay=7,
            epsilon=0.07,
        )

    # ============================
    # Optimal parameters
    # ============================

    if nonlinear_parameters_compute == False:
        tau_A, m_A, tau_B, m_B = None, None, None, None
    else:
        if gait_residual_A is not None:
            tau_A = compute_optimal_delay(gait_residual_A, max_tau=100)
            m_A = compute_optimal_embedding_dimension(
                gait_residual_A, max_dim=20, delay=tau_A, epsilon=0.07, threshold=0.01
            )
        else:
            tau_A = None
            m_A = None

        if gait_residual_B is not None:
            tau_B = compute_optimal_delay(gait_residual_B, max_tau=100)
            m_B = compute_optimal_embedding_dimension(
                gait_residual_B, max_dim=20, delay=tau_B, epsilon=0.07, threshold=0.01
            )
        else:
            tau_B = None
            m_B = None

    return (
        vel_A,
        vel_B,
        direction_A,
        direction_B,
        d,
        depth,
        breadth,
        stride_frequency_A,
        stride_frequency_B,
        swaying_A,
        swaying_B,
        stride_length_A,
        stride_length_B,
        delta_f,
        mean_gsi_h,
        mean_relative_phase_h,
        variance_relative_phase_h,
        mean_coherence,
        stationarity_A,
        stationarity_B,
        lyapunov_A,
        lyapunov_B,
        determinism_A,
        determinism_B,
        rec,
        det,
        maxline,
        tau_A,
        m_A,
        tau_B,
        m_B,
    )


def get_scientific_notation(f):
    """Get the scientific notation of a float

    Arguments:
    ----------
        f {float} -- The float to format

    Return
    ------
        v {str} -- The value part of the scientific notation
        exp {str} -- The exponent part of the scientific notation
    """
    scientific_notation = f"{f:.2e}"
    v, exp = scientific_notation.split("e")
    exp_sign = exp[0]
    exp_value = exp[1:]
    # remove leading 0
    exp_value = exp_value.lstrip("0")
    if exp_sign == "+":
        exp = exp_value
    else:
        exp = f"-{exp_value}"
    return v, exp


def get_latex_scientific_notation(f, threshold=None):
    """Get the scientific notation of a float (LaTeX format)

    Arguments:
    ----------
        f {float} -- The float to format
        threshold {float, optional} -- The threshold to use for the formatting, by default None

    Return
    ------
        str -- The formatted float
    """
    if threshold is not None and f > threshold:
        return f"{f:.2f}"
    if f == 0:
        return "0"
    v, exp = get_scientific_notation(f)
    if exp == "":
        return v
    if exp == "1":
        return f"{v} \\times 10"
    return f"{v} \\times 10^{{{exp}}}"


def get_formatted_p_value(p_value):
    """Format the p-value

    Arguments:
    ----------
        p_value {float} -- p-value

    Return
    ------
        formatted_p_value {str} -- formatted p-value
    """
    if p_value < 1e-4:
        formatted_p_value = f"\\mathbf{{< 10^{{-4}}}}"
    else:
        s = get_latex_scientific_notation(p_value)
        if p_value < 0.05:
            formatted_p_value = f"\\mathbf{{{s}}}"
        else:
            formatted_p_value = s
    return formatted_p_value


def get_formatted_p_value_stars(p_value):
    """Format the p-value with stars

    Arguments:
    ----------
        p_value {float} -- p-value

    Return
    ------
        formatted_p_value {str} -- formatted p-value
    """

    s = get_latex_scientific_notation(p_value)
    if p_value < 1e-4:
        return "$p < 10^{{-4}}****$"
    elif p_value < 1e-3:
        return "$p < 10^{{-3}}***$"
    elif p_value < 1e-2:
        return "$p < 10^{{-2}}**$"
    elif p_value < 0.05:
        return f"$p = {s}*$"
    else:
        return f"$p = {s}$"


def get_bins(vmin, vmax, n_bins):
    """Get bins for a given range and number of bins. A bin is defined by its edges and center.

    Arguments:
    ----------
        vmin {float} -- min value
        vmax {float} -- max value
        n_bins {int} -- number of bins

        Return
        ------
        bin_size {float} -- size of the bins
        pdf_edges {np.array} -- edges of the bins
        bin_centers {np.array} -- centers of the bins
    """
    bin_size = (vmax - vmin) / n_bins
    pdf_edges = np.linspace(vmin, vmax, n_bins + 1)
    bin_centers = 0.5 * (pdf_edges[0:-1] + pdf_edges[1:])
    return bin_size, pdf_edges, bin_centers


def compute_pdf(x, vmin, vmax, n_bins):
    """Compute the pdf of x

    Arguments:
    ----------
        x {np.array} -- x values
        vmin {float} -- min value
        vmax {float} -- max value
        n_bins {int} -- number of bins

        Return
        ------
        pdf {np.array} -- pdf of x
        bin_centers {np.array} -- centers of the bins
    """
    _, pdf_edges, bin_centers = get_bins(vmin, vmax, n_bins)
    pdf, _ = np.histogram(x, bins=pdf_edges, density=True)
    return pdf, bin_centers


def compute_binned_values(x_values, y_values, min_v, max_v, n_bins):
    """Compute binned values of y_values with respect to x_values

    Arguments:
    ----------
        x_values {np.array} -- x values
        y_values {np.array} -- y values
        min_v {float} -- min value
        max_v {float} -- max value
        n_bins {int} -- number of bins

    Return
    ------
        bin_centers {np.array} -- centers of the bins
        means {np.array} -- means of y_values over the bins
        stds {np.array} -- stds of y_values over the bins
        errors {np.array} -- errors of y_values over the bins
    """

    pdf_edges = np.linspace(min_v, max_v, n_bins + 1)
    # print(pdf_edges)
    bin_centers = 0.5 * (pdf_edges[0:-1] + pdf_edges[1:])

    indices = np.digitize(x_values, pdf_edges) - 1

    means = np.full(n_bins, np.nan)
    stds = np.full(n_bins, np.nan)
    errors = np.full(n_bins, np.nan)
    n_values = np.zeros(n_bins)

    for i in range(n_bins):
        if np.sum(indices == i) == 0:
            continue
        means[i] = np.nanmean(y_values[indices == i])
        stds[i] = np.nanstd(y_values[indices == i])
        errors[i] = stds[i] / np.sqrt(np.sum(indices == i))
        n_values[i] = np.sum(indices == i)

    return bin_centers, means, stds, errors, n_values
