from matplotlib.ticker import FuncFormatter
import numpy as np
from pedestrians_social_binding.environment import Environment
from pedestrians_social_binding.trajectory_utils import (
    compute_stride_frequency,
    smooth_trajectory_savitzy_golay,
    compute_gait_residual,
    compute_simultaneous_observations,
    compute_coherence,
    compute_phase_embedding,
    compute_optimal_delay,
    compute_optimal_embedding_dimension,
    get_box,
)

from scipy.signal import hilbert
from scipy.stats import entropy, circmean, circvar
from scipy.spatial.distance import cdist

from utils import get_pedestrian_thresholds

import matplotlib.pyplot as plt
import scienceplots


from pyrqa.analysis_type import Cross
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation

from pycwt import wct, cwt, xwt


plt.style.use("science")


color1 = "#0B60B0"
color2 = "#40A2D8"
color_diff = "#003049"
color_hist = "#fcbf49"


def plot_wavelet_analysis(
    t,
    s1,
    s2,
    dt,
    min_frequency_band=0.5,
    max_frequency_band=2.0,
    save_path=None,
    ax=None,
    labels=None,
):

    def custom_formatter(x, pos):
        if x.is_integer():
            return f"{int(x)}"
        else:
            return f"{x:.3f}"

    l = 4 * np.pi / (6 + np.sqrt(2 + 6**2))

    wave_1, _, freqs, coi, _, _ = cwt(s1, dt)
    wave_2, _, freqs, coi, _, _ = cwt(s2, dt)

    # Cross wavelet transform
    xwt_v, coi, _, xwt_sig = xwt(s1, s2, dt)
    xwt_sig = np.abs(xwt_v) / xwt_sig[:, None]

    coi_freq = 1 / (l * coi)

    # Wavelet coherence
    wct_v, _, _, _, _ = wct(s1, s2, dt)

    if ax is None:
        _, ax = plt.subplots(1, 4, figsize=(18, 4))

    pcol0 = ax[0].pcolormesh(
        t, freqs, np.abs(wave_1), cmap="jet", rasterized=True, linewidth=0
    )
    pcol0.set_edgecolor("face")
    if labels is not None:
        ax[0].set_title(labels[0], y=-0.35)

    pcol1 = ax[1].pcolormesh(
        t, freqs, np.abs(wave_2), cmap="jet", rasterized=True, linewidth=0
    )
    pcol1.set_edgecolor("face")
    if labels is not None:
        ax[1].set_title(labels[1], y=-0.35)

    pcol2 = ax[2].pcolormesh(
        t, freqs, np.abs(xwt_v), cmap="jet", rasterized=True, linewidth=0
    )
    pcol2.set_edgecolor("face")
    # ax[2].contour(t, xwt_period, xwt_sig, [1], colors="red", linewidths=2)
    # ax[2].set_title("Cross-Wavelet transform (XWT)")
    if labels is not None:
        ax[2].set_title(labels[2], y=-0.35)

    pcol3 = ax[3].pcolormesh(
        t,
        freqs,
        np.abs(wct_v),
        cmap="jet",
        rasterized=True,
        linewidth=0,
    )
    pcol3.set_edgecolor("face")

    # colorbar
    cbar = plt.colorbar(pcol3, ax=ax[3])
    cbar.set_label("Cross Wavelet magnitude")

    # ax[3].contour(t, wct_period, wct_mag, [1], colors="red", linewidths=2)
    # ax[3].set_title("Wavelet Coherence (WTC)")
    if labels is not None:
        ax[3].set_title(labels[3], y=-0.35)

    ticks = [int(2**i) if int(2**i) == 2**i else 2**i for i in range(-4, 4)]
    for i in range(4):
        ax[i].set_xlabel("$t$ [s]")
        ax[i].set_ylabel("$f$ [Hz]")
        ax[i].set_yscale("log", base=2)
        ax[i].set_yticks(ticks)
        ax[i].get_yaxis().set_major_formatter(FuncFormatter(custom_formatter))
        ax[i].set_ylim([1 / 16, 10])
        ax[i].plot(t, coi_freq, color="black", linestyle="--", linewidth=2)
        ax[i].fill_between(
            t, 1 / 16, coi_freq, alpha=0.5, hatch="//", facecolor="white"
        )
        # show frequency band
        ax[i].hlines(
            y=[min_frequency_band, max_frequency_band],
            xmin=t[0],
            xmax=t[-1],
            color="red",
            linestyle="--",
            linewidth=2,
        )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, transparent=True)
        plt.close()


def plot_gait_residuals(gait_residual_A, gait_residual_B, t, ax, labels=None):
    t_start = 0
    t_max = 30
    print()
    indx_start = np.argmin(np.abs(t - t_start))
    indx_max = np.argmin(np.abs(t - t_max))
    # gait residual
    ax.plot(
        t[indx_start:indx_max],
        gait_residual_A[indx_start:indx_max],
        "o-",
        markersize=2,
        color=color1,
    )
    ax.plot(
        t[indx_start:indx_max],
        gait_residual_B[indx_start:indx_max],
        "o-",
        markersize=2,
        color=color2,
    )
    ax.set_ylabel("$\\gamma$ [m]")
    ax.set_xlabel("Time [s]")
    if labels is not None:
        ax.set_title(labels[0], y=-0.5)
    ax.grid(color="gray", linestyle="--", linewidth=0.5)


def plot_gsi(gait_residual_A, gait_residual_B, t, axes, labels=None):

    t_start, t_end = 0, 5
    indx_start = np.argmin(np.abs(t - t_start))
    indx_end = np.argmin(np.abs(t - t_end))

    hilbert_A = hilbert(gait_residual_A)
    hilbert_B = hilbert(gait_residual_B)

    phase_A = np.angle(hilbert_A)  # type: ignore
    phase_A = (phase_A + np.pi) % (2 * np.pi) - np.pi
    phase_B = np.angle(hilbert_B)  # type: ignore
    phase_B = (phase_B + np.pi) % (2 * np.pi) - np.pi

    phase_diff = phase_A - phase_B
    phase_diff = phase_diff[indx_start:indx_end]
    phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi

    mean_dphi = circmean(phase_diff, high=np.pi, low=-np.pi)  # type: ignore
    variance_dphi = circvar(phase_diff, high=np.pi, low=-np.pi)  # type: ignore

    # histogram phase diff
    n_bins = 32
    hist, _ = np.histogram(phase_diff, bins=n_bins, range=(-np.pi, np.pi), density=True)
    ent = entropy(hist)
    gsi = (np.log(n_bins) - ent) / np.log(n_bins)

    axes[0].plot(
        t[indx_start:indx_end],
        phase_A[indx_start:indx_end],
        "o-",
        markersize=2,
        color=color1,
    )
    axes[0].plot(
        t[indx_start:indx_end],
        phase_B[indx_start:indx_end],
        "o-",
        markersize=2,
        color=color2,
    )
    axes[0].set_ylabel("$\\phi$ [rad]")
    axes[0].set_xlabel("t [s]")
    axes[0].set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    axes[0].set_yticklabels(["$-\pi$", "$-\pi/2$", "0", "$\pi/2$", "$\pi$"])
    if labels is not None:
        axes[0].set_title(labels[0], y=-0.5)
    axes[0].grid(color="gray", linestyle="--", linewidth=0.5)

    # phase diff
    axes[1].plot(
        t[indx_start:indx_end], phase_diff, "o-", markersize=2, color=color_diff
    )
    axes[1].set_ylabel("$\\Delta\\phi$ [rad]")
    axes[1].set_xlabel("t [s]")
    axes[1].set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    axes[1].set_yticklabels(["$-\pi$", "$-\pi/2$", "0", "$\pi/2$", "$\pi$"])
    if labels is not None:
        axes[1].set_title(labels[1], y=-0.5)
    axes[1].grid(color="gray", linestyle="--", linewidth=0.5)

    # histogram phase diff
    n_bins = 32
    hist, _ = np.histogram(phase_diff, bins=n_bins, range=(-np.pi, np.pi), density=True)
    ent = entropy(hist)
    gsi = (np.log(n_bins) - ent) / np.log(n_bins)

    axes[2].hist(
        phase_diff,
        bins=n_bins,
        color=color_hist,
        alpha=0.5,
        weights=np.ones_like(phase_diff) / len(phase_diff),
        range=(-np.pi, np.pi),
    )

    axes[2].vlines(mean_dphi, 0, 1, color="black", linestyle="--", linewidth=1)
    axes[2].text(
        mean_dphi + 0.1,
        0.55,
        f"$\\overline{{\\Delta\\phi}} = {mean_dphi:.2f}$ rad\n$\\sigma_{{\\Delta\\phi}}^2 = {variance_dphi:.2f}$",
        fontsize=12,
    )
    # show entropy
    axes[2].text(
        -3,
        0.85,
        f"Entropy: {ent:.2f}\nGSI: {gsi:.2f}",
        fontsize=12,
    )
    axes[2].set_ylabel("p($\Delta\phi$)")
    axes[2].set_xlabel("$\Delta\phi$ [rad]")
    axes[2].set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    axes[2].set_xticklabels(["$-\pi$", "$-\pi/2$", "0", "$\pi/2$", "$\pi$"])

    if labels is not None:
        axes[2].set_title(labels[2], y=-0.5)


def plot_determinism(p, n_boxes, min_val_box, ax, color):
    # n_dimensions = p.shape[1]
    n_dimensions = 3
    p = p[:, :n_dimensions]
    limit = np.zeros((n_dimensions, 2))
    for i in range(n_dimensions):
        limit[i, 0] = np.min(p[:, i])
        limit[i, 1] = np.max(p[:, i])

    first_point = p[0, :]
    current_box = get_box(p[0, :], limit, n_dimensions, n_boxes)

    box_directions = {}

    for j in range(1, len(p)):
        new_box = get_box(p[j, :], limit, n_dimensions, n_boxes)
        if not np.all(new_box == current_box):
            # reached a new box
            direction = (p[j, :] - first_point) / np.linalg.norm(p[j, :] - first_point)
            if tuple(current_box) not in box_directions:
                box_directions[tuple(current_box)] = []
            box_directions[tuple(current_box)].append(direction)

            first_point = p[j, :]  # update the first point
        current_box = new_box

    boxes = np.zeros([n_boxes for _ in range(n_dimensions)] + [3])
    for box, directions in box_directions.items():
        if len(directions) >= min_val_box:
            i, j, k = int(box[0]), int(box[1]), int(box[2])
            boxes[j, i, k] = np.mean(directions, axis=0)

    x = np.linspace(limit[0, 0], limit[0, 1], n_boxes + 1)
    y = np.linspace(limit[1, 0], limit[1, 1], n_boxes + 1)
    z = np.linspace(limit[2, 0], limit[2, 1], n_boxes + 1)

    center_x = (x[1:] + x[:-1]) / 2
    center_y = (y[1:] + y[:-1]) / 2
    center_z = (z[1:] + z[:-1]) / 2

    # plot
    X, Y, Z = np.meshgrid(center_x, center_y, center_z)

    ax.quiver(
        X,
        Y,
        Z,
        boxes[:, :, :, 0],
        boxes[:, :, :, 1],
        boxes[:, :, :, 2],
        length=0.05,
        # normalize=True,
        linewidth=1,
        color="black",
    )
    ax.plot(p[:, 0], p[:, 1], p[:, 2], color=color, alpha=0.5)

    # show grid
    for i in range(n_boxes + 1):
        for j in range(n_boxes + 1):
            ax.plot(
                [x[i], x[i]],
                [y[j], y[j]],
                [z[0], z[-1]],
                color="gray",
                alpha=0.3,
            )
            ax.plot(
                [x[0], x[-1]],
                [y[j], y[j]],
                [z[i], z[i]],
                color="gray",
                alpha=0.3,
            )
            ax.plot(
                [x[i], x[i]],
                [y[0], y[-1]],
                [z[j], z[j]],
                color="gray",
                alpha=0.3,
            )

    ticks = [-0.2, -0.1, 0, 0.1, 0.2]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)
    ax.minorticks_off()

    ax.set_xlabel("$x_i$")
    ax.set_ylabel("$x_{i+\\tau}$")
    ax.set_zlabel("$x_{i+2\\tau}$")
    ax.grid(False)

    return ax


def plot_recurrence_plot(
    gait_residual_A, gait_residual_B, embedding_A, embedding_B, eps, ax
):
    time_series_A = TimeSeries(gait_residual_A, embedding_dimension=4, time_delay=7)
    time_series_B = TimeSeries(gait_residual_B, embedding_dimension=4, time_delay=7)
    time_series = (time_series_A, time_series_B)
    settings = Settings(
        time_series,
        analysis_type=Cross,
        neighbourhood=FixedRadius(0.07),
        similarity_measure=EuclideanMetric,
        theiler_corrector=0,
    )
    computation = RQAComputation.create(settings)
    result = computation.run()
    rec = result.recurrence_rate
    det = result.determinism
    maxline = result.longest_diagonal_line

    distances = cdist(embedding_A, embedding_B, metric="euclidean")
    recurrences = distances < eps

    ax.imshow(recurrences, cmap="binary", origin="lower")
    ax.set_xlabel("Gait residual of pedestrian 1")
    ax.set_ylabel("Gait residual of pedestrian 2")
    # show values
    t = ax.text(
        0.8,
        0.9,
        f"\%REC: {rec:.2f}\n\%DET: {det:.2f}\nMAXLINE: {maxline}",
        fontsize=12,
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
    )
    t.set_bbox(dict(facecolor="white", alpha=0.8))


def plot_lyapunov_exponent(
    p,
    n_iterations=1000,
    n_neighbors=10,
    n_points=500,
    eps=0.07,
    theiler_window=400,
    ax=None,
):

    max_try = n_points * 10

    # for eps in [0.03, 0.07, 0.1]:
    for eps in [0.03]:

        distances = []
        j = 0

        n_try = 0

        while j < n_points and n_try < max_try:
            # random starting point in the embedding
            idx_reference_point = np.random.randint(0, p.shape[0])
            point = p[idx_reference_point]
            # find the nearest neighbors
            all_distances = np.linalg.norm(p - point, axis=1)
            all_distances[idx_reference_point] = np.inf  # exclude the reference point
            # id of the close enough neighbors
            neighbors = np.where(all_distances < eps)[0]
            n_try += 1
            # keep only the neighbors that are not too close to the reference point
            neighbors = neighbors[
                np.abs(neighbors - idx_reference_point) > theiler_window
            ]

            if len(neighbors) < n_neighbors:
                continue
            # compute the average distance at all iterations
            point_distances = np.zeros(n_iterations)
            for k in range(n_iterations):
                reference_trajectory_point = p[(idx_reference_point + k) % p.shape[0]]
                for neighbor in neighbors:
                    neighbor_trajectory_point = p[(neighbor + k) % p.shape[0]]
                    point_distances[k] += np.abs(
                        reference_trajectory_point[-1] - neighbor_trajectory_point[-1]
                    )
            point_distances /= len(neighbors)
            j += 1
            distances.append(point_distances)

        distances = np.array(distances)

        if len(distances) == 0:
            return None

        expansion_rate = np.nanmean(np.log(distances), axis=0)

        # fit a line with method of least squares
        A = np.vstack([np.arange(5), np.ones(5)]).T
        m, c = np.linalg.lstsq(A, expansion_rate[:5], rcond=None)[0]

        if ax is not None:
            ax.plot(expansion_rate)
            ax.plot(np.arange(5), m * np.arange(5) + c, label=f"$l_{{lyap}} = {m:.2f}$")
            ax.set_xlabel("Number of iterations")
            ax.set_ylabel("$\\log(E)$")
            ax.grid(color="gray", linestyle="--", linewidth=0.5)
            ax.legend(loc="lower right")


if __name__ == "__main__":

    sampling_time = 0.03
    smoothing_window_duration = 0.25  # seconds
    smoothing_window = int(smoothing_window_duration / sampling_time)
    power_threshold = 1e-4
    n_fft = 10000

    min_frequency_band = 0.5  # Hz, min frequency for coherence
    max_frequency_band = 2.0  # Hz, max frequency for coherence

    env_name = "diamor"
    env = Environment(
        env_name, data_dir="../../atc-diamor-pedestrians/data/formatted", raw=True
    )

    thresholds_indiv = get_pedestrian_thresholds(env_name)
    ped = env.get_pedestrians(ids=[10582900], thresholds=thresholds_indiv)[0]

    traj = smooth_trajectory_savitzy_golay(ped.trajectory, smoothing_window)

    compute_stride_frequency(
        ped.trajectory,
        power_threshold=power_threshold,
        n_fft=n_fft,
        min_f=min_frequency_band,
        max_f=max_frequency_band,
        save_plot=True,
        file_path="../data/figures/examples/periodogram.pdf",
    )
    # ----------------- FIRST ROW (dyad) -----------------

    group = env.get_groups(
        # size=2, ped_thresholds=thresholds_indiv, ids=[1132490011325100]
        size=2,
        ped_thresholds=thresholds_indiv,
        ids=[1451470014514900],
    )[0]

    traj_Ad = group.get_members()[0].get_trajectory()
    traj_Bd = group.get_members()[1].get_trajectory()

    traj_Ad, traj_Bd = compute_simultaneous_observations([traj_Ad, traj_Bd])
    t_d = traj_Ad[:-1, 0] - traj_Ad[0, 0]

    traj_Ad = smooth_trajectory_savitzy_golay(traj_Ad, smoothing_window)
    traj_Bd = smooth_trajectory_savitzy_golay(traj_Bd, smoothing_window)

    gait_residual_Ad = compute_gait_residual(traj_Ad)
    gait_residual_Bd = compute_gait_residual(traj_Bd)

    if gait_residual_Ad is None or gait_residual_Bd is None:
        raise ValueError("Gait residual is None")

    # ----------------- SECOND ROW (baseline) -----------------

    # ped_1 = env.get_pedestrians(ids=[11460100], thresholds=thresholds_indiv)[0]
    # ped_2 = env.get_pedestrians(ids=[14243400], thresholds=thresholds_indiv)[0]

    ped_1 = env.get_pedestrians(ids=[16361500], thresholds=thresholds_indiv)[0]
    ped_2 = env.get_pedestrians(ids=[14414700], thresholds=thresholds_indiv)[0]

    traj_Ab = ped_1.get_trajectory()
    traj_Bb = ped_2.get_trajectory()

    len_a = traj_Ab.shape[0]
    len_b = traj_Bb.shape[0]
    min_len = min(len_a, len_b)

    traj_Ab = traj_Ab[:min_len]
    traj_Bb = traj_Bb[:min_len]

    t_b = traj_Ab[:-1, 0] - traj_Ab[0, 0]

    traj_Ab = smooth_trajectory_savitzy_golay(traj_Ab, smoothing_window)
    traj_Bb = smooth_trajectory_savitzy_golay(traj_Bb, smoothing_window)

    gait_residual_Ab = compute_gait_residual(traj_Ab)
    gait_residual_Bb = compute_gait_residual(traj_Bb)

    if gait_residual_Ab is None or gait_residual_Bb is None:
        raise ValueError("Gait residual is None")

    # ----------------- GAIT RESIDUAL -----------------

    _, ax = plt.subplots(2, 1, figsize=(10, 5))

    plot_gait_residuals(
        gait_residual_Ad,
        gait_residual_Bd,
        t_d,
        ax[0],
        labels=["(a)"],
    )

    plot_gait_residuals(
        gait_residual_Ab,
        gait_residual_Bb,
        t_b,
        ax[1],
        labels=["(b)"],
    )

    plt.tight_layout()
    # plt.show()
    plt.savefig("../data/figures/examples/gait_residuals.pdf")
    plt.close()

    # ----------------- GSI -----------------

    _, ax = plt.subplots(2, 3, figsize=(10, 5))

    plot_gsi(
        gait_residual_Ad,
        gait_residual_Bd,
        t_d,
        ax[0],
        labels=["(a)", "(b)", "(c)"],
    )

    plot_gsi(
        gait_residual_Ab,
        gait_residual_Bb,
        t_b,
        ax[1],
        labels=["(d)", "(e)", "(f)"],
    )

    plt.tight_layout()
    # plt.show()
    plt.savefig("../data/figures/examples/gsi.pdf")
    plt.close()

    # ----------------- WAVELET -----------------

    _, ax = plt.subplots(2, 4, figsize=(13, 5))

    plot_wavelet_analysis(
        t_d,
        gait_residual_Ad,
        gait_residual_Bd,
        dt=sampling_time,
        min_frequency_band=min_frequency_band,
        max_frequency_band=max_frequency_band,
        ax=ax[0],
        labels=["(a)", "(b)", "(c)", "(d)"],
    )

    coherence_d = compute_coherence(gait_residual_Ad, gait_residual_Bd)

    t = ax[0][3].text(
        0.5,
        0.5,
        f"CWC: {coherence_d:.2f}",
        fontsize=12,
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax[0][3].transAxes,
    )
    t.set_bbox(dict(facecolor="white", alpha=0.6))

    plot_wavelet_analysis(
        t_b,
        gait_residual_Ab,
        gait_residual_Bb,
        dt=sampling_time,
        min_frequency_band=min_frequency_band,
        max_frequency_band=max_frequency_band,
        ax=ax[1],
        labels=["(e)", "(f)", "(g)", "(h)"],
    )

    coherence_b = compute_coherence(gait_residual_Ab, gait_residual_Bb)

    t = ax[1][3].text(
        0.5,
        0.5,
        f"CWC: {coherence_b:.2f}",
        fontsize=12,
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax[1][3].transAxes,
    )
    t.set_bbox(dict(facecolor="white", alpha=0.6))

    plt.tight_layout()
    # plt.show()
    plt.savefig("../data/figures/examples/wavelet.pdf")
    plt.close()

    # ----------------- NONLINEAR ANALYSIS PARAMETERS -----------------

    fig, ax = plt.subplots(1, 2, figsize=(8, 3))

    # optimal parameters
    compute_optimal_delay(gait_residual_Ad, max_tau=20, ax=ax[0])

    compute_optimal_embedding_dimension(
        gait_residual_Ad, max_dim=10, delay=7, epsilon=0.07, threshold=0.01, ax=ax[1]
    )

    ax[0].set_title("(a)", y=-0.4)
    ax[1].set_title("(b)", y=-0.4)

    plt.tight_layout()
    # plt.show()
    plt.savefig("../data/figures/examples/nonlinear_analysis_parameters.pdf")
    plt.close()

    # -----------------  PHASE EMBEDDING -----------------

    embedding_Ad = compute_phase_embedding(gait_residual_Ad, 4, 7)
    embedding_Bd = compute_phase_embedding(gait_residual_Bd, 4, 7)

    embedding_Ab = compute_phase_embedding(gait_residual_Ab, 4, 7)
    embedding_Bb = compute_phase_embedding(gait_residual_Bb, 4, 7)

    # plot 3D embedding
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(221, projection="3d")
    ax1.plot(embedding_Ad[:, 0], embedding_Ad[:, 1], embedding_Ad[:, 2], color=color1)
    ax1.set_xlabel("$x_i$")
    ax1.set_ylabel("$x_{i+\\tau}$")
    ax1.set_zlabel("$x_{i+2\\tau}$")
    ax1.set_title("(a)")

    ax2 = fig.add_subplot(222, projection="3d")
    ax2.plot(embedding_Bd[:, 0], embedding_Bd[:, 1], embedding_Bd[:, 2], color=color2)
    ax2.set_xlabel("$x_i$")
    ax2.set_ylabel("$x_{i+\\tau}$")
    ax2.set_zlabel("$x_{i+2\\tau}$")
    ax2.set_title("(b)")

    ax3 = fig.add_subplot(223, projection="3d")
    ax3.plot(embedding_Ab[:, 0], embedding_Ab[:, 1], embedding_Ab[:, 2], color=color1)
    ax3.set_xlabel("$x_i$")
    ax3.set_ylabel("$x_{i+\\tau}$")
    ax3.set_zlabel("$x_{i+2\\tau}$")
    ax3.set_title("(c)")

    ax4 = fig.add_subplot(224, projection="3d")
    ax4.plot(embedding_Bb[:, 0], embedding_Bb[:, 1], embedding_Bb[:, 2], color=color2)
    ax4.set_xlabel("$x_i$")
    ax4.set_ylabel("$x_{i+\\tau}$")
    ax4.set_zlabel("$x_{i+2\\tau}$")
    ax4.set_title("(d)")

    plt.tight_layout()

    # plt.show()
    plt.savefig("../data/figures/examples/phase_embedding.pdf")
    plt.close()

    # # ----------------- DETERMINISM -----------------

    fig = plt.figure(figsize=(6, 4))

    ax1 = fig.add_subplot(121, projection="3d")
    ax1 = plot_determinism(embedding_Ad, n_boxes=5, min_val_box=3, ax=ax1, color=color1)

    ax2 = fig.add_subplot(122, projection="3d")
    ax2 = plot_determinism(embedding_Ab, n_boxes=5, min_val_box=3, ax=ax2, color=color2)

    # check_determinism(embedding_Ad, n_boxes=5, min_val_box=3, ax=ax[0])
    # check_determinism(embedding_Ab, n_boxes=5, min_val_box=3, ax=ax[1])

    ax1.set_xlim(-0.2, 0.2)
    ax1.set_ylim(-0.2, 0.2)
    ax1.set_zlim(-0.2, 0.2)

    ax2.set_xlim(-0.2, 0.2)
    ax2.set_ylim(-0.2, 0.2)
    ax2.set_zlim(-0.2, 0.2)

    ax1.set_title("(a)", y=-0.25)
    ax2.set_title("(b)", y=-0.25)

    # ax1.set_aspect("equal")
    # ax2.set_aspect("equal")

    plt.tight_layout()
    # plt.show()
    plt.savefig("../data/figures/examples/determinism_phase_embedding.pdf")
    plt.close()

    # # ----------------- MAX LYAPUNOV EXPONENT -----------------

    fig, ax = plt.subplots(1, 2, figsize=(8, 3))

    plot_lyapunov_exponent(
        embedding_Ad,
        n_iterations=20,
        n_neighbors=1,
        n_points=100,
        eps=0.03,
        theiler_window=5,
        ax=ax[0],
    )

    plot_lyapunov_exponent(
        embedding_Ab,
        n_iterations=20,
        n_neighbors=1,
        n_points=100,
        eps=0.03,
        theiler_window=5,
        ax=ax[1],
    )

    ax[0].set_title("(a)", y=-0.3)
    ax[1].set_title("(b)", y=-0.3)

    plt.tight_layout()
    # plt.show()
    plt.savefig("../data/figures/examples/phase_space_lyapunov.pdf")
    plt.close()

    # # ----------------- CROSS RECURRENCE PLOT -----------------

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    plot_recurrence_plot(
        gait_residual_Ad,
        gait_residual_Bd,
        embedding_Ad,
        embedding_Bd,
        eps=0.07,
        ax=ax[0],
    )
    ax[0].set_title("(a)", y=-0.3)

    plot_recurrence_plot(
        gait_residual_Ab,
        gait_residual_Bb,
        embedding_Ab,
        embedding_Bb,
        eps=0.07,
        ax=ax[1],
    )
    ax[1].set_title("(b)", y=-0.3)

    plt.tight_layout()
    # plt.show()
    plt.savefig("../data/figures/examples/cross_recurrence_plot.pdf")
    plt.close()
