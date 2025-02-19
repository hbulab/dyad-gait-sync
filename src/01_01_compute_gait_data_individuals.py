from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from pedestrians_social_binding.environment import Environment
from pedestrians_social_binding.trajectory_utils import (
    smooth_trajectory_savitzy_golay,
    compute_stride_parameters,
)

from utils import get_pedestrian_thresholds

from tqdm import tqdm


if __name__ == "__main__":
    env_name = "diamor"
    env = Environment(
        env_name, data_dir="../../atc-diamor-pedestrians/data/formatted", raw=True
    )

    thresholds_indiv = get_pedestrian_thresholds(env_name)

    individuals = env.get_pedestrians(
        thresholds=thresholds_indiv,
        no_groups=True,
    )

    # ============================
    # Parameters
    # ============================

    sampling_time = 0.03
    smoothing_window_duration = 0.25  # seconds
    smoothing_window = int(smoothing_window_duration / sampling_time)
    power_threshold = 1e-4
    n_fft = 10000
    min_frequency_band = 0.5  # Hz
    max_frequency_band = 2.0  # Hz

    # ============================
    # Data dictionary
    # ============================

    values = {
        "frequency": [],
        "swaying": [],
        "stride_length": [],
        "velocity": [],
    }

    # ============================
    # Compute data
    # =================

    velocities = []
    velocities_smoothed = []
    dt = []

    print("Computing for individuals")
    for pedestrian in tqdm(individuals[:100]):

        traj = pedestrian.get_trajectory()

        if len(traj) == 0:
            continue

        if len(traj) <= smoothing_window:
            continue

        traj_smoothed = smooth_trajectory_savitzy_golay(traj, smoothing_window)

        vel = np.mean(np.linalg.norm(traj_smoothed[:, 5:7], axis=1)) / 1000

        stride_frequency, swaying, stride_length = compute_stride_parameters(
            traj,
            power_threshold=power_threshold,
            n_fft=n_fft,
            min_f=min_frequency_band,
            max_f=max_frequency_band,
        )

        if stride_length is not None:
            stride_length = stride_length / 1000

        values["velocity"].append(vel)
        values["frequency"].append(stride_frequency)
        values["swaying"].append(swaying)
        values["stride_length"].append(stride_length)

    # save data to file using pandas
    df = pd.DataFrame(values)
    df.to_csv("../data/csv/gait_article_individuals.csv", index=False)
