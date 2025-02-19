from pedestrians_social_binding.environment import Environment
from pedestrians_social_binding.trajectory_utils import (
    get_trajectory_at_times,
    compute_interpersonal_distance,
    compute_continuous_sub_trajectories_using_time,
)

from utils import (
    get_social_values,
    get_all_days,
    get_pedestrian_thresholds,
)

from tqdm import tqdm
import numpy as np

from utils import pickle_save

if __name__ == "__main__":

    env_name = "diamor"
    env = Environment(
        env_name, data_dir="../../atc-diamor-pedestrians/data/formatted", raw=True
    )
    env_name_short = env_name.split(":")[0]

    (
        soc_binding_type,
        soc_binding_names,
        soc_binding_values,
        colors,
    ) = get_social_values(env_name)
    days = get_all_days(env_name)

    D_MAX = 2000
    T_MIN = 10

    thresholds_indiv = get_pedestrian_thresholds(env_name)

    trajectory_pairs = {}

    for day in days:
        print(f"Day {day}:")

        pedestrians = env.get_pedestrians(
            days=[day],
            no_groups=True,
        )

        for pedestrian in tqdm(pedestrians):

            id_ped = pedestrian.get_id()
            trajectory = pedestrian.get_trajectory()
            times = pedestrian.get_time()

            encountered = pedestrian.get_encountered_pedestrians(
                pedestrians,
                skip=[id_ped],
                # alone=True,
                # proximity_threshold=4000,
            )

            for i, p in enumerate(encountered):
                id_other = p.get_id()
                other_trajectory = p.get_trajectory()

                other_trajectory = np.array(
                    get_trajectory_at_times(other_trajectory, times)
                )

                d = compute_interpersonal_distance(
                    trajectory[:, 1:3], other_trajectory[:, 1:3]
                )

                close_enough = d < D_MAX

                trajectory_close_enough = trajectory[close_enough]
                sub_trajectories = compute_continuous_sub_trajectories_using_time(
                    trajectory_close_enough, max_gap=2
                )

                trajectory_close_enough_other = other_trajectory[close_enough]
                sub_trajectories_other = compute_continuous_sub_trajectories_using_time(
                    trajectory_close_enough_other, max_gap=2
                )

                for sub_traj, sub_traj_other in zip(
                    sub_trajectories, sub_trajectories_other
                ):
                    t_subtraj = sub_traj[-1, 0] - sub_traj[0, 0]
                    t_subtraj_other = sub_traj_other[-1, 0] - sub_traj_other[0, 0]

                    if t_subtraj < T_MIN or t_subtraj_other < T_MIN:
                        continue

                    if (id_ped, id_other) not in trajectory_pairs:
                        trajectory_pairs[(id_ped, id_other)] = []

                    trajectory_pairs[(id_ped, id_other)].append(
                        (sub_traj, sub_traj_other)
                    )

    pickle_save(
        f"../data/pickle/close_enough_individuals_baseline_02_18.pkl",
        trajectory_pairs,
    )
