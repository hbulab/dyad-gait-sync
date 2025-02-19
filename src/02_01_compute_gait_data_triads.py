from pedestrians_social_binding.environment import Environment
from pedestrians_social_binding.trajectory_utils import (
    smooth_trajectory_savitzy_golay,
    compute_simultaneous_observations,
    compute_center_of_mass,
    align_trajectories_at_origin,
)

from utils import (
    get_pedestrian_thresholds,
    compute_synchronisation_data_pair,
    pickle_save,
)

import numpy as np
from tqdm import tqdm

import pandas as pd

if __name__ == "__main__":
    env_name = "diamor"
    env = Environment(
        env_name, data_dir="../../atc-diamor-pedestrians/data/formatted", raw=True
    )

    thresholds_indiv = get_pedestrian_thresholds(env_name)

    groups = env.get_groups(
        size=3,
        ped_thresholds=thresholds_indiv,
    )

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

    max_angle = 135
    min_angle = -135

    # ============================
    # Data dictionary
    # ============================

    data = {
        "id": [],
        "pair": [],
        "formation": [],
        "frequency_1": [],
        "frequency_2": [],
        "gsi": [],
        "relative_phase": [],
        "delta_f": [],
        "coherence": [],
        "interpersonal_distance": [],
        "position_1_x": [],
        "position_1_y": [],
        "position_2_x": [],
        "position_2_y": [],
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
        "tau_1": [],
        "m_2": [],
        "tau_2": [],
    }

    all_positions = {
        "abreast": [],
        "following": [],
        "lambda": [],
        "v": [],
    }

    for group in tqdm(groups):

        group_id = group.get_id()
        members = group.get_members()
        traj_A, traj_B, traj_C = compute_simultaneous_observations(
            [m.trajectory for m in members]
        )
        traj_A = smooth_trajectory_savitzy_golay(traj_A, smoothing_window)
        traj_B = smooth_trajectory_savitzy_golay(traj_B, smoothing_window)
        traj_C = smooth_trajectory_savitzy_golay(traj_C, smoothing_window)

        traj_com = compute_center_of_mass([traj_A, traj_B, traj_C])

        _, [tra_A_aligned, tra_B_aligned, tra_C_aligned] = align_trajectories_at_origin(
            traj_com, [traj_A, traj_B, traj_C], axis="y"
        )

        avg_pos_A = np.mean(tra_A_aligned[:, 1:3], axis=0)
        avg_pos_B = np.mean(tra_B_aligned[:, 1:3], axis=0)
        avg_pos_C = np.mean(tra_C_aligned[:, 1:3], axis=0)

        avg_positions = [avg_pos_A, avg_pos_B, avg_pos_C]
        trajectories = [traj_A, traj_B, traj_C]

        # find the pedestrian in the center (horizontally)
        # sort by x position
        x_positions = [avg_pos_A[0], avg_pos_B[0], avg_pos_C[0]]
        y_positions = [avg_pos_A[1], avg_pos_B[1], avg_pos_C[1]]
        sorted_indices = np.argsort(x_positions)
        pos_center = avg_positions[sorted_indices[1]]
        pos_left = avg_positions[sorted_indices[0]]
        pos_right = avg_positions[sorted_indices[2]]

        traj_center = trajectories[sorted_indices[1]]
        traj_left = trajectories[sorted_indices[0]]
        traj_right = trajectories[sorted_indices[2]]

        # compute angle left-center-right
        center_to_left = pos_left - pos_center
        center_to_right = pos_right - pos_center
        angle_left_center = np.arctan2(center_to_left[1], center_to_left[0])
        angle_right_center = np.arctan2(center_to_right[1], center_to_right[0])
        angle = angle_right_center - angle_left_center
        # keep angle between -pi and pi
        if angle > np.pi:
            angle -= 2 * np.pi
        elif angle < -np.pi:
            angle += 2 * np.pi

        formation = None
        angle = np.rad2deg(angle)
        if angle > -160 and angle < -20:
            formation = "v"
        elif angle > 20 and angle < 160:
            formation = "lambda"
        elif angle < -160 or angle > 160:
            # abreast or following
            x_dist_max = np.max(x_positions) - np.min(x_positions)
            y_dist_max = np.max(y_positions) - np.min(y_positions)
            if x_dist_max > y_dist_max:
                formation = "abreast"
            else:
                formation = "following"

        if formation is not None:
            all_positions[formation].append(
                (tra_A_aligned[:, 1:3], tra_B_aligned[:, 1:3], tra_C_aligned[:, 1:3])
            )

        if formation in ["lambda", "v", "abreast"]:

            for (traj_1, traj_2), positions, pair in zip(
                [
                    [traj_left, traj_center],
                    [traj_right, traj_center],
                    [traj_left, traj_right],
                ],
                [
                    [pos_left, pos_center],
                    [pos_right, pos_center],
                    [pos_left, pos_right],
                ],
                ["left_center", "right_center", "left_right"],
            ):

                (
                    vel_1,
                    vel_2,
                    direction_1,
                    direction_2,
                    d,
                    depth,
                    breadth,
                    stride_frequency_1,
                    stride_frequency_2,
                    swaying_1,
                    swaying_2,
                    stride_length_1,
                    stride_length_2,
                    delta_f,
                    mean_gsi,
                    mean_relative_phase,
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
                    traj_1,
                    traj_2,
                    sampling_time,
                    power_threshold,
                    n_fft,
                    min_frequency_band,
                    max_frequency_band,
                    window_duration,
                )

                data["id"].append(group_id)
                data["pair"].append(pair)
                data["formation"].append(formation)
                data["frequency_1"].append(stride_frequency_1)
                data["frequency_2"].append(stride_frequency_2)
                data["gsi"].append(mean_gsi)
                data["relative_phase"].append(mean_relative_phase)
                data["delta_f"].append(delta_f)
                data["coherence"].append(mean_coherence)
                data["interpersonal_distance"].append(d)
                data["position_1_x"].append(positions[0][0] / 1000)
                data["position_1_y"].append(positions[0][1] / 1000)
                data["position_2_x"].append(positions[1][0] / 1000)
                data["position_2_y"].append(positions[1][1] / 1000)

                data["stationarity_1"].append(stationarity_A)
                data["stationarity_2"].append(stationarity_B)

                data["lyapunov_1"].append(lyapunov_A)
                data["lyapunov_2"].append(lyapunov_B)

                data["determinism_1"].append(determinism_A)
                data["determinism_2"].append(determinism_B)

                data["rec"].append(rec)
                data["det"].append(det)
                data["maxline"].append(maxline)

                data["m_1"].append(m_A)
                data["tau_1"].append(tau_A)
                data["m_2"].append(m_B)
                data["tau_2"].append(tau_B)

        elif formation in ["following"]:
            # find pedestrian in the center (vertically)
            sorted_indices = np.argsort(y_positions)

            traj_front = trajectories[sorted_indices[2]]
            traj_back = trajectories[sorted_indices[0]]
            traj_vert_center = trajectories[sorted_indices[1]]

            pos_front = avg_positions[sorted_indices[2]]
            pos_back = avg_positions[sorted_indices[0]]
            pos_vert_center = avg_positions[sorted_indices[1]]

            for (traj_1, traj_2), positions, pair in zip(
                [
                    [traj_front, traj_vert_center],
                    [traj_back, traj_vert_center],
                    [traj_front, traj_back],
                ],
                [
                    [pos_front, pos_vert_center],
                    [pos_back, pos_vert_center],
                    [pos_front, pos_back],
                ],
                ["front_center", "back_center", "front_back"],
            ):

                (
                    vel_1,
                    vel_2,
                    direction_1,
                    direction_2,
                    d,
                    depth,
                    breadth,
                    stride_frequency_1,
                    stride_frequency_2,
                    swaying_1,
                    swaying_2,
                    stride_length_1,
                    stride_length_2,
                    delta_f,
                    mean_gsi,
                    mean_relative_phase,
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
                    traj_1,
                    traj_2,
                    sampling_time,
                    power_threshold,
                    n_fft,
                    min_frequency_band,
                    max_frequency_band,
                    window_duration,
                )

                data["id"].append(group_id)
                data["pair"].append(pair)
                data["formation"].append(formation)
                data["frequency_1"].append(stride_frequency_1)
                data["frequency_2"].append(stride_frequency_2)
                data["gsi"].append(mean_gsi)
                data["relative_phase"].append(mean_relative_phase)
                data["delta_f"].append(delta_f)
                data["coherence"].append(mean_coherence)
                data["interpersonal_distance"].append(d)
                data["position_1_x"].append(positions[0][0] / 1000)
                data["position_1_y"].append(positions[0][1] / 1000)
                data["position_2_x"].append(positions[1][0] / 1000)
                data["position_2_y"].append(positions[1][1] / 1000)

                data["stationarity_1"].append(stationarity_A)
                data["stationarity_2"].append(stationarity_B)

                data["lyapunov_1"].append(lyapunov_A)
                data["lyapunov_2"].append(lyapunov_B)

                data["determinism_1"].append(determinism_A)
                data["determinism_2"].append(determinism_B)

                data["rec"].append(rec)
                data["det"].append(det)
                data["maxline"].append(maxline)

                data["m_1"].append(m_A)
                data["tau_1"].append(tau_A)
                data["m_2"].append(m_B)
                data["tau_2"].append(tau_B)

    # save data to file using pandas
    df = pd.DataFrame(data)
    df.to_csv("../data/csv/gait_data_triads.csv", index=False)

    # save positions
    pickle_save("../data/pickle/triad_positions.pkl", all_positions)
