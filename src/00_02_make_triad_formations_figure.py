import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.transforms import Bbox, IdentityTransform, TransformedBbox

import scienceplots

plt.style.use("science")


class AngleAnnotation(Arc):
    """
    Draws an arc between two vectors which appears circular in display space.
    """

    def __init__(
        self,
        xy,
        p1,
        p2,
        size=75,
        unit="points",
        ax=None,
        text="",
        textposition="inside",
        text_kw=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        xy, p1, p2 : tuple or array of two floats
            Center position and two points. Angle annotation is drawn between
            the two vectors connecting *p1* and *p2* with *xy*, respectively.
            Units are data coordinates.

        size : float
            Diameter of the angle annotation in units specified by *unit*.

        unit : str
            One of the following strings to specify the unit of *size*:

            * "pixels": pixels
            * "points": points, use points instead of pixels to not have a
              dependence on the DPI
            * "axes width", "axes height": relative units of Axes width, height
            * "axes min", "axes max": minimum or maximum of relative Axes
              width, height

        ax : `matplotlib.axes.Axes`
            The Axes to add the angle annotation to.   text : str
            The text to mark the angle with.

        textposition : {"inside", "outside", "edge"}
            Whether to show the text in- or outside the arc. "edge" can be used
            for custom positions anchored at the arc's edge.

        text_kw : dict
            Dictionary of arguments passed to the Annotation.

        **kwargs
            Further parameters are passed to `matplotlib.patches.Arc`. Use this
            to specify, color, linewidth etc. of the arc.

        """
        self.ax = ax or plt.gca()
        self._xydata = xy  # in data coordinates
        self.vec1 = p1
        self.vec2 = p2
        self.size = size
        self.unit = unit
        self.textposition = textposition
        # self.arrowhead = arrowhead
        # self.arrow_kw = arrow_kw or {}

        super().__init__(
            self._xydata,
            size,
            size,
            angle=0.0,
            theta1=self.theta1,
            theta2=self.theta2,
            **kwargs,
        )

        self.set_transform(IdentityTransform())
        self.ax.add_patch(self)

        self.kw = dict(
            ha="center",
            va="center",
            xycoords=IdentityTransform(),
            xytext=(0, 0),
            textcoords="offset points",
            annotation_clip=True,
        )
        self.kw.update(text_kw or {})
        self.text = ax.annotate(text, xy=self._center, **self.kw)  # type: ignore

    def get_size(self):
        factor = 1.0
        if self.unit == "points":
            factor = self.ax.figure.dpi / 72.0
        elif self.unit[:4] == "axes":
            b = TransformedBbox(Bbox.unit(), self.ax.transAxes)
            dic = {
                "max": max(b.width, b.height),
                "min": min(b.width, b.height),
                "width": b.width,
                "height": b.height,
            }
            factor = dic[self.unit[5:]]
        return self.size * factor

    def set_size(self, size):
        self.size = size

    def get_center_in_pixels(self):
        """return center in pixels"""
        return self.ax.transData.transform(self._xydata)

    def set_center(self, xy):
        """set center in data coordinates"""
        self._xydata = xy

    def get_theta(self, vec):
        vec_in_pixels = self.ax.transData.transform(vec) - self._center
        return np.rad2deg(np.arctan2(vec_in_pixels[1], vec_in_pixels[0]))

    def get_theta1(self):
        return self.get_theta(self.vec1)

    def get_theta2(self):
        return self.get_theta(self.vec2)

    def set_theta(self, angle):
        pass

    # Redefine attributes of the Arc to always give values in pixel space
    _center = property(get_center_in_pixels, set_center)
    theta1 = property(get_theta1, set_theta)
    theta2 = property(get_theta2, set_theta)
    width = property(get_size, set_size)
    height = property(get_size, set_size)

    # The following two methods are needed to update the text position.
    def draw(self, renderer):
        self.update_text()
        super().draw(renderer)

    def update_text(self):
        c = self._center
        s = self.get_size()
        angle_span = (self.theta2 - self.theta1) % 360
        angle = np.deg2rad(self.theta1 + angle_span / 2)
        r = s / 1.8
        if self.textposition == "inside":
            r = s / np.interp(angle_span, [60, 90, 135, 180], [3.3, 3.5, 3.8, 4])
        self.text.xy = c + r * np.array([np.cos(angle), np.sin(angle)])
        if self.textposition == "outside":

            def R90(a, r, w, h):
                if a < np.arctan(h / 2 / (r + w / 2)):
                    return np.sqrt((r + w / 2) ** 2 + (np.tan(a) * (r + w / 2)) ** 2)
                else:
                    c = np.sqrt((w / 2) ** 2 + (h / 2) ** 2)
                    T = np.arcsin(c * np.cos(np.pi / 2 - a + np.arcsin(h / 2 / c)) / r)
                    xy = r * np.array([np.cos(a + T), np.sin(a + T)])
                    xy += np.array([w / 2, h / 2])
                    return np.sqrt(np.sum(xy**2))

            def R(a, r, w, h):
                aa = (a % (np.pi / 4)) * ((a % (np.pi / 2)) <= np.pi / 4) + (
                    np.pi / 4 - (a % (np.pi / 4))
                ) * ((a % (np.pi / 2)) >= np.pi / 4)
                return R90(aa, r, *[w, h][:: int(np.sign(np.cos(2 * a)))])

            bbox = self.text.get_window_extent()
            X = R(angle, r, bbox.width, bbox.height)
            trans = self.ax.figure.dpi_scale_trans.inverted()
            offs = trans.transform(((X - s / 2), 0))[0] * 72
            self.text.set_position([offs * np.cos(angle), offs * np.sin(angle)])


import numpy as np


if __name__ == "__main__":

    plot_data = [
        {"label": "i", "color": "#eae2b7", "edge_color": "#817326"},
        {"label": "j", "color": "#fcbf49", "edge_color": "#805502"},
        {"label": "k", "color": "#f77f00", "edge_color": "#633300"},
    ]

    positions = {
        "v": [np.array([-0.4, 0.6]), np.array([-0.1, -0.4]), np.array([0.5, 0.3])],
        "lambda": [np.array([-0.5, -0.5]), np.array([0, 0.5]), np.array([0.7, -0.4])],
        "abreast": [np.array([-0.5, 0.1]), np.array([0, 0.05]), np.array([0.8, -0.1])],
        "following": [
            np.array([-0.1, 0.6]),
            np.array([0.1, -0]),
            np.array([0.2, -0.5]),
        ],
    }

    arrow_base = [
        np.array([0, 0.2]),
        np.array([0, -0.4]),
        np.array([0.1, 0.5]),
        np.array([0.5, 0]),
    ]

    dx_y = -0.7
    dy_x = -0.7

    plt.rcParams.update({"font.size": 14})
    plt.rc("text.latex", preamble=r"\usepackage{amsmath}")  # for bold math

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))

    for i, (formation, label) in enumerate(
        zip(["v", "lambda", "abreast", "following"], ["(a)", "(b)", "(c)", "(d)"])
    ):

        # dashed lines between the agents
        for j in range(2):
            ax[i].plot(
                [positions[formation][j][0], positions[formation][j + 1][0]],
                [positions[formation][j][1], positions[formation][j + 1][1]],
                color="black",
                linestyle="--",
                linewidth=2,
                zorder=-1,
            )

        for j, position in enumerate(positions[formation]):
            ax[i].scatter(
                position[0],
                position[1],
                color=plot_data[j]["color"],
                s=300,
                edgecolor=plot_data[j]["edge_color"],
                linewidth=1.5,
            )
            ax[i].text(
                position[0] + 0.1,
                position[1] + 0.05,
                f"$\\mathbf{{{plot_data[j]['label']}}}$",
            )

        # add angle annotation
        ji = positions[formation][0] - positions[formation][1]
        jk = positions[formation][2] - positions[formation][1]
        angle_ji = np.arctan2(ji[1], ji[0])
        angle_jk = np.arctan2(jk[1], jk[0])
        angle = angle_jk - angle_ji
        # keep angle between -pi and pi
        if angle > np.pi:
            angle -= 2 * np.pi
        elif angle < -np.pi:
            angle += 2 * np.pi

        if angle > 0:
            start = positions[formation][0]
            end = positions[formation][2]
        else:
            start = positions[formation][2]
            end = positions[formation][0]

        angle_inner = AngleAnnotation(
            positions[formation][1],
            start,
            end,
            ax=ax[i],
            size=50,
            color="#2a9d8f",
            linewidth=1.5,
        )

        angle_outer = AngleAnnotation(
            positions[formation][1],
            start,
            end,
            ax=ax[i],
            size=60,
            color="#2a9d8f",
            linewidth=1.5,
            text="$\\theta$",
            textposition="outside",
            text_kw={"fontsize": 20, "color": "#2a9d8f"},
        )

        ax[i].add_patch(angle_inner)
        ax[i].add_patch(angle_outer)

        # show dx, dy
        idx_min_x = np.argmin([positions[formation][j][0] for j in range(3)])
        idx_max_x = np.argmax([positions[formation][j][0] for j in range(3)])
        idx_min_y = np.argmin([positions[formation][j][1] for j in range(3)])
        idx_max_y = np.argmax([positions[formation][j][1] for j in range(3)])

        min_x = positions[formation][idx_min_x]
        max_x = positions[formation][idx_max_x]
        min_y = positions[formation][idx_min_y]
        max_y = positions[formation][idx_max_y]

        dx = max_x - min_x
        dy = max_y - min_y

        # dx
        dx_left = [min_x[0], dx_y]
        dx_right = [max_x[0], dx_y]
        ax[i].plot(
            [dx_left[0], dx_right[0]],
            [dx_left[1], dx_right[1]],
            color="black",
            linewidth=2,
            zorder=-1,
        )
        ax[i].text(
            (min_x[0] + max_x[0]) / 2,
            dx_y - 0.2,
            "$d_x$",
            fontsize=20,
            color="black",
        )
        # construction lines dx
        ax[i].plot(
            [min_x[0], min_x[0]],
            [min_x[1], dx_y],
            color="black",
            linestyle="--",
            linewidth=1,
            zorder=-1,
        )
        ax[i].plot(
            [max_x[0], max_x[0]],
            [max_x[1], dx_y],
            color="black",
            linestyle="--",
            linewidth=1,
            zorder=-1,
        )

        # dy
        dy_bottom = [min_y[1], dy_x]
        dy_top = [max_y[1], dy_x]
        ax[i].plot(
            [dy_bottom[1], dy_top[1]],
            [dy_bottom[0], dy_top[0]],
            color="black",
            linewidth=2,
            zorder=-1,
        )
        ax[i].text(
            dy_x - 0.25,
            (min_y[1] + max_y[1]) / 2,
            "$d_y$",
            fontsize=20,
            color="black",
        )
        # construction lines dy
        ax[i].plot(
            [dy_x, min_y[0]],
            [min_y[1], min_y[1]],
            color="black",
            linestyle="--",
            linewidth=1,
            zorder=-1,
        )
        ax[i].plot(
            [dy_x, max_y[0]],
            [max_y[1], max_y[1]],
            color="black",
            linestyle="--",
            linewidth=1,
            zorder=-1,
        )

        # add velocity vector (vertical)
        ax[i].scatter(
            arrow_base[i][0],
            arrow_base[i][1],
            color="black",
            s=50,
            edgecolor="black",
        )
        ax[i].arrow(
            arrow_base[i][0],
            arrow_base[i][1],
            0,
            0.3,
            head_width=0.05,
            head_length=0.05,
            fc="black",
            ec="black",
        )

        ax[i].set_title(label, y=-0.15)
        ax[i].axis("off")
        ax[i].set_xlim(-1, 1)
        ax[i].set_ylim(-1, 1)
        ax[i].set_aspect("equal")

    plt.tight_layout()
    plt.savefig("../data/figures/triads/triads_formations.pdf")
    plt.close()
