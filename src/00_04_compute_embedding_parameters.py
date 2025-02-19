from pathlib import Path

import numpy as np
import pandas as pd

from pedestrians_social_binding.environment import Environment
from pedestrians_social_binding.trajectory_utils import (
    smooth_trajectory_savitzy_golay,
    compute_simultaneous_observations,
    compute_gait_residual,
    compute_optimal_delay,
    compute_optimal_embedding_dimension,
)

from utils import (
    get_pedestrian_thresholds,
    get_social_values,
    get_groups_thresholds,
    get_interaction,
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

    taus, ms = [], []

    print("Computing for each dyad")
    # select 1000 groups randomly
    for group in tqdm(np.random.choice(groups, 1000)):

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

        gait_residual_A = compute_gait_residual(traj_A)
        gait_residual_B = compute_gait_residual(traj_B)

        if gait_residual_A is not None:
            tau_A = compute_optimal_delay(gait_residual_A, max_tau=20)
            if tau_A is not None:
                m_A = compute_optimal_embedding_dimension(
                    gait_residual_A,
                    max_dim=10,
                    delay=tau_A,
                    epsilon=0.07,
                    threshold=0.01,
                )
        else:
            tau_A = None
            m_A = None

        if gait_residual_B is not None:
            tau_B = compute_optimal_delay(gait_residual_B, max_tau=20)
            if tau_B is not None:
                m_B = compute_optimal_embedding_dimension(
                    gait_residual_B,
                    max_dim=10,
                    delay=tau_B,
                    epsilon=0.07,
                    threshold=0.01,
                )
        else:
            tau_B = None
            m_B = None

        taus.extend([tau_A, tau_B])
        ms.extend([m_A, m_B])

    taus = np.array(taus)
    ms = np.array(ms)

    print("mean tau", np.nanmean(taus, axis=0))
    print("mean m", np.nanmean(ms, axis=0))

    # save the results
    results = pd.DataFrame(
        {
            "tau": taus,
            "m": ms,
        }
    )
    results.to_csv(f"../data/csv/embedding_parameters.csv", index=False)
