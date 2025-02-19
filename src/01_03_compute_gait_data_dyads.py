from pathlib import Path

import numpy as np
import pandas as pd

from pedestrians_social_binding.environment import Environment
from pedestrians_social_binding.trajectory_utils import (
    smooth_trajectory_savitzy_golay,
    compute_simultaneous_observations,
)

from utils import (
    get_pedestrian_thresholds,
    get_social_values,
    get_groups_thresholds,
    get_interaction,
    compute_synchronisation_data_pair,
    pickle_load,
)

from tqdm import tqdm


if __name__ == "__main__":
    env_name = "diamor"
    env = Environment(
        env_name, data_dir="../../atc-diamor-pedestrians/data/formatted", raw=True
    )
    (
        soc_binding_type,
        soc_binding_names,
        soc_binding_values,
        colors,
    ) = get_social_values(env_name)

    thresholds_indiv = get_pedestrian_thresholds(env_name)
    thresholds_groups = get_groups_thresholds()

    groups = env.get_groups(
        size=2,
        ped_thresholds=thresholds_indiv,
        group_thresholds=thresholds_groups,
    )

    fig_dir = Path("../data/figures/gait/article/stride/periodograms")

    # ============================
    # Seed
    # ============================

    np.random.seed(0)

    # ============================
    # Parameters
    # ============================

    sampling_time = 0.03
    smoothing_window_duration = 0.25  # seconds
    smoothing_window = int(smoothing_window_duration / sampling_time)
    power_threshold = 1e-4
    n_fft = 10000
    window_duration = 5  # window duration for synchronization

    min_frequency_band = 0.5  # Hz, min frequency for coherence
    max_frequency_band = 2.0  # Hz, max frequency for coherence

    # ============================
    # Data dictionary
    # ============================

    values = {
        "id_1": [],
        "id_2": [],
        "interaction": [],
        "contact": [],
        "frequency_1": [],
        "frequency_2": [],
        "swaying_1": [],
        "swaying_2": [],
        "stride_length_1": [],
        "stride_length_2": [],
        "direction_1": [],
        "direction_2": [],
        "velocity_1": [],
        "velocity_2": [],
        "gsi": [],
        "relative_phase": [],
        "delta_f": [],
        "coherence": [],
        "interpersonal_distance": [],
        "depth": [],
        "breadth": [],
        "stationarity_1": [],
        "stationarity_2": [],
        "lyapunov_1": [],
        "lyapunov_2": [],
        "determinism_1": [],
        "determinism_2": [],
        "rec": [],
        "det": [],
        "maxline": [],
        "m_1": [],
        "tau_A": [],
        "m_2": [],
        "tau_2": [],
    }

    # ==============================
    # Compute for each dyad
    # ==============================

    print("Computing for each dyad")
    for group in tqdm(groups):

        soc_binding = group.get_annotation(soc_binding_type)
        conversation, contact = get_interaction(group)

        if soc_binding not in soc_binding_values:
            continue

        traj_A = group.get_members()[0].get_trajectory()
        traj_B = group.get_members()[1].get_trajectory()

        traj_A, traj_B = compute_simultaneous_observations([traj_A, traj_B])

        if len(traj_A) == 0 or len(traj_B) == 0:
            continue

        if len(traj_A) <= smoothing_window or len(traj_B) <= smoothing_window:
            continue

        traj_A = smooth_trajectory_savitzy_golay(traj_A, smoothing_window)
        traj_B = smooth_trajectory_savitzy_golay(traj_B, smoothing_window)

        (
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
            _,
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
            m_A,
            tau_A,
            m_B,
            tau_B,
        ) = compute_synchronisation_data_pair(
            traj_A,
            traj_B,
            sampling_time,
            power_threshold,
            n_fft,
            min_frequency_band,
            max_frequency_band,
            window_duration,
        )

        values["id_1"].append(group.get_members()[0].get_id())
        values["id_2"].append(group.get_members()[1].get_id())

        values["interaction"].append(soc_binding)
        values["contact"].append(contact)

        values["direction_1"].append(direction_A)
        values["direction_2"].append(direction_B)

        values["velocity_1"].append(vel_A)
        values["velocity_2"].append(vel_B)
        values["frequency_1"].append(stride_frequency_A)
        values["frequency_2"].append(stride_frequency_B)
        values["swaying_1"].append(swaying_A)
        values["swaying_2"].append(swaying_B)
        values["stride_length_1"].append(stride_length_A)
        values["stride_length_2"].append(stride_length_B)

        values["delta_f"].append(delta_f)
        values["gsi"].append(mean_gsi_h)
        values["relative_phase"].append(mean_relative_phase_h)
        values["coherence"].append(mean_coherence)
        values["interpersonal_distance"].append(d)
        values["depth"].append(depth)
        values["breadth"].append(breadth)

        values["stationarity_1"].append(stationarity_A)
        values["stationarity_2"].append(stationarity_B)

        values["lyapunov_1"].append(lyapunov_A)
        values["lyapunov_2"].append(lyapunov_B)

        values["determinism_1"].append(determinism_A)
        values["determinism_2"].append(determinism_B)

        values["rec"].append(rec)
        values["det"].append(det)
        values["maxline"].append(maxline)

        values["m_1"].append(m_A)
        values["tau_A"].append(tau_A)
        values["m_2"].append(m_B)
        values["tau_2"].append(tau_B)

    # ==============================
    # Compute baseline (random individuals)
    # ==============================

    individuals = env.get_pedestrians(
        thresholds=thresholds_indiv,
        # no_groups=True,
    )

    used_pairs = []
    n_baseline = 1000  # number of baseline pairs
    count_baseline = 0
    print("Computing baseline")

    with tqdm(total=n_baseline) as pbar:
        while count_baseline < n_baseline:

            idx_A = np.random.randint(0, len(individuals))
            idx_B = np.random.randint(0, len(individuals))

            if idx_A == idx_B:  # cannot be the same individual
                continue

            ped_A = individuals[idx_A]
            ped_B = individuals[idx_B]

            traj_A = ped_A.get_trajectory()
            traj_B = ped_B.get_trajectory()

            if len(traj_A) == 0 or len(traj_B) == 0:
                continue

            len_a = traj_A.shape[0]
            len_b = traj_B.shape[0]
            min_len = min(len_a, len_b)

            traj_A = traj_A[:min_len]
            traj_B = traj_B[:min_len]

            if len(traj_A) <= smoothing_window or len(traj_B) <= smoothing_window:
                continue

            traj_A = smooth_trajectory_savitzy_golay(traj_A, smoothing_window)
            traj_B = smooth_trajectory_savitzy_golay(traj_B, smoothing_window)

            (
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                freq_A,
                freq_B,
                _,
                _,
                _,
                _,
                delta_f,
                mean_gsi_h,
                mean_relative_phase_h,
                _,
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
                m_A,
                tau_A,
                m_B,
                tau_B,
            ) = compute_synchronisation_data_pair(
                traj_A,
                traj_B,
                sampling_time,
                power_threshold,
                n_fft,
                min_frequency_band,
                max_frequency_band,
                window_duration,
                simult=False,
            )

            values["id_1"].append(ped_A.get_id())
            values["id_2"].append(ped_B.get_id())

            values["interaction"].append(4)
            values["contact"].append(2)

            values["direction_1"].append(-1)
            values["direction_2"].append(-1)

            values["velocity_1"].append(-1)
            values["velocity_2"].append(-1)
            values["frequency_1"].append(freq_A)
            values["frequency_2"].append(freq_B)
            values["swaying_1"].append(-1)
            values["swaying_2"].append(-1)
            values["stride_length_1"].append(-1)
            values["stride_length_2"].append(-1)

            values["delta_f"].append(delta_f)
            values["gsi"].append(mean_gsi_h)
            values["relative_phase"].append(mean_relative_phase_h)
            values["coherence"].append(mean_coherence)
            values["interpersonal_distance"].append(-1)
            values["depth"].append(-1)
            values["breadth"].append(-1)

            values["stationarity_1"].append(stationarity_A)
            values["stationarity_2"].append(stationarity_B)

            values["lyapunov_1"].append(lyapunov_A)
            values["lyapunov_2"].append(lyapunov_B)

            values["determinism_1"].append(determinism_A)
            values["determinism_2"].append(determinism_B)

            values["rec"].append(rec)
            values["det"].append(det)
            values["maxline"].append(maxline)

            values["m_1"].append(m_A)
            values["tau_A"].append(tau_A)
            values["m_2"].append(m_B)
            values["tau_2"].append(tau_B)

            count_baseline += 1
            pbar.update(1)

    # ==============================
    # Compute baseline (close individuals)
    # ==============================

    pairs = pickle_load("../data/pickle/close_enough_individuals_baseline.pkl")

    seen = set()
    for id_A, id_B in tqdm(pairs.keys()):
        for traj_A, traj_B in pairs[(id_A, id_B)]:

            if (min(id_A, id_B), max(id_A, id_B)) in seen:
                continue

            if len(traj_A) == 0 or len(traj_B) == 0:
                continue

            len_a = traj_A.shape[0]
            len_b = traj_B.shape[0]
            min_len = min(len_a, len_b)

            traj_A = traj_A[:min_len]
            traj_B = traj_B[:min_len]

            if len(traj_A) <= smoothing_window or len(traj_B) <= smoothing_window:
                continue

            seen.add((min(id_A, id_B), max(id_A, id_B)))

            traj_A = smooth_trajectory_savitzy_golay(traj_A, smoothing_window)
            traj_B = smooth_trajectory_savitzy_golay(traj_B, smoothing_window)

            (
                _,
                _,
                direction_A,
                direction_B,
                d,
                depth,
                breadth,
                freq_A,
                freq_B,
                _,
                _,
                _,
                _,
                delta_f,
                mean_gsi_h,
                mean_relative_phase_h,
                _,
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
                m_A,
                tau_A,
                m_B,
                tau_B,
            ) = compute_synchronisation_data_pair(
                traj_A,
                traj_B,
                sampling_time,
                power_threshold,
                n_fft,
                min_frequency_band,
                max_frequency_band,
                window_duration,
            )

            values["id_1"].append(-1)
            values["id_2"].append(-1)

            values["interaction"].append(5)
            values["contact"].append(2)

            values["direction_1"].append(direction_A)
            values["direction_2"].append(direction_B)

            values["velocity_1"].append(-1)
            values["velocity_2"].append(-1)
            values["frequency_1"].append(freq_A)
            values["frequency_2"].append(freq_B)
            values["swaying_1"].append(-1)
            values["swaying_2"].append(-1)
            values["stride_length_1"].append(-1)
            values["stride_length_2"].append(-1)

            values["delta_f"].append(delta_f)
            values["gsi"].append(mean_gsi_h)
            values["relative_phase"].append(mean_relative_phase_h)
            values["coherence"].append(mean_coherence)
            values["interpersonal_distance"].append(d)
            values["depth"].append(depth)
            values["breadth"].append(breadth)

            values["stationarity_1"].append(stationarity_A)
            values["stationarity_2"].append(stationarity_B)

            values["lyapunov_1"].append(lyapunov_A)
            values["lyapunov_2"].append(lyapunov_B)

            values["determinism_1"].append(determinism_A)
            values["determinism_2"].append(determinism_B)

            values["rec"].append(rec)
            values["det"].append(det)
            values["maxline"].append(maxline)

            values["m_1"].append(m_A)
            values["tau_A"].append(tau_A)
            values["m_2"].append(m_B)
            values["tau_2"].append(tau_B)

    # ==============================
    # save data to file using pandas
    # ==============================
    df = pd.DataFrame(values)
    df.to_csv("../data/csv/gait_article_dyads.csv", index=False)
