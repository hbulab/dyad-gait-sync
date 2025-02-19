from pedestrians_social_binding.environment import Environment

from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches


import scienceplots

plt.style.use("science")

from utils import (
    get_pedestrian_thresholds,
)

if __name__ == "__main__":

    # load environment
    env_name = "diamor"
    env = Environment(
        env_name, data_dir="../../atc-diamor-pedestrians/data/formatted", raw=True
    )

    env_name_short = env_name.split(":")[0]

    thresholds_indiv = get_pedestrian_thresholds(env_name)

    # load sensor positions
    sensors_positions = pd.read_csv(f"../data/dataset/sensors_positions.csv")

    camera_positions = {
        "06": [50, 9],
        "08": [50, 5],
    }

    for day in ["06", "08"]:

        pedestrians = env.get_pedestrians(thresholds=thresholds_indiv, days=[day])

        # find the env boundaries
        mins_x, mins_y, maxs_x, maxs_y = [], [], [], []
        for pedestrian in pedestrians:
            trajectory = pedestrian.trajectory
            mins_x.append(np.min(trajectory[:, 1] / 1000))
            maxs_x.append(np.max(trajectory[:, 1] / 1000))
            mins_y.append(np.min(trajectory[:, 2] / 1000))
            maxs_y.append(np.max(trajectory[:, 2] / 1000))

        min_x = np.min(mins_x)
        max_x = np.max(maxs_x)
        min_y = np.min(mins_y)
        max_y = np.max(maxs_y)

        # create occupancy grid
        CELL_SIZE = 0.1

        n_bin_x = int(np.ceil((max_x - min_x) / CELL_SIZE) + 1)
        n_bin_y = int(np.ceil((max_y - min_y) / CELL_SIZE) + 1)
        grid = np.zeros((n_bin_x, n_bin_y))

        for pedestrian in pedestrians:
            trajectory = pedestrian.trajectory
            x = trajectory[:, 1] / 1000
            y = trajectory[:, 2] / 1000

            nx = np.ceil((x - min_x) / CELL_SIZE).astype("int")
            ny = np.ceil((y - min_y) / CELL_SIZE).astype("int")

            in_limit = (x >= min_x) & (x <= max_x) & (y >= min_y) & (y <= max_y)

            nx = nx[in_limit]
            ny = ny[in_limit]

            grid[nx, ny] += 1

        max_val = np.max(grid)
        grid /= max_val

        xi = np.linspace(min_x, max_x, n_bin_x)
        yi = np.linspace(min_y, max_y, n_bin_y)

        # transform to grid coordinates
        day_sensor_positions = sensors_positions[sensors_positions["day"] == int(day)]
        x_sensors = day_sensor_positions["x"].values / 1000 - min_x
        y_sensors = day_sensor_positions["y"].values / 1000 - min_y

        # plot
        fig, ax = plt.subplots(figsize=(6, 6))

        im = ax.imshow(
            grid.T,
            extent=(0, max_x - min_x, 0, max_y - min_y),
            origin="lower",
            cmap="inferno_r",
        )
        # colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.2)
        cbar = plt.colorbar(im, cax=cax)

        # plot sensors
        ax.scatter(
            x_sensors,
            y_sensors,
            c="steelblue",
            s=15,
            linewidth=0.5,
            edgecolors="k",
        )

        camera_fov = patches.Wedge(
            (camera_positions[day][0], camera_positions[day][1]),
            5,
            150,
            210,
            facecolor="magenta",
            edgecolor="k",
            linewidth=0.5,
        )
        ax.add_patch(camera_fov)

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

        ax.set_aspect("equal")

        plt.savefig(f"../data/figures/occupancy_grid_{day}.pdf", dpi=300)
        # plt.show()
        plt.close()
