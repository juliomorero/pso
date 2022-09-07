import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np

from enum import IntEnum
from IPython.display import display
from matplotlib import cm
from scipy.stats import qmc
from scipy.spatial.distance import cdist


class Mode(IntEnum):
    NNVM = 0
    NNVM_CRAZINESS = 1
    CORNFIELD_VECTOR = 2
    ONLY_CORNFIELD = 3
    CURRENT = 4
    EXTRA = 5


class PSO:
    def __init__(
        self,
        mode: Mode,
        fitness_func=None,
        p_inc=2,
        g_inc=2,
        w=1,
        particles=32,
        dimensions=2,
        grid_center=None,
        grid_length=200,
        duration=500,
    ) -> None:
        self.mode = mode
        self.p_inc = p_inc
        self.g_inc = g_inc
        self.w = w
        self.particles = particles
        self.dimensions = dimensions
        self.grid_length = grid_length
        self.duration = duration

        self.grid_center = np.array([100, 100]) if grid_center is None else grid_center

        self.lower_lim = self.grid_center - self.grid_length / 2
        self.upper_lim = self.grid_center + self.grid_length / 2
        self.offset = self.grid_center - self.grid_length / 2

        def default_func(point: np.ndarray):
            return np.sqrt((point - 100) ** 2).sum(axis=1)

        self.fitness_func = default_func if fitness_func is None else fitness_func

        self.version = {
            Mode.NNVM: self.nnvm,
            Mode.NNVM_CRAZINESS: self.nnvm_craziness,
            Mode.CORNFIELD_VECTOR: self.cornfield_vector,
            Mode.ONLY_CORNFIELD: self.only_cornfield_vector,
            Mode.CURRENT: self.current,
            Mode.EXTRA: self.extra,
        }

        colormap = cm.get_cmap("rainbow")
        self.colors = colormap(np.linspace(0, 1, self.particles))

        self.reset()
        if self.mode > 1:
            plt.xlim(self.lower_lim[0], self.upper_lim[0])
            plt.ylim(self.lower_lim[1], self.upper_lim[1])
            plt.title(
                f"Gbest value: {self.gbest_val}\nGbest location: {self.gbest_loc}"
            )

            plt.scatter(self.X[:, 0], self.X[:, 1], color=self.colors)
            plt.scatter(*self.gbest_loc, color="green", marker="p")
            plt.scatter(*self.grid_center, color="black", marker="s")

            X, Y = np.mgrid[
                self.lower_lim[0] : self.upper_lim[0] : 1000j,
                self.lower_lim[1] : self.upper_lim[1] : 1000j,
            ]
            positions = np.vstack([X.ravel(), Y.ravel()]).T
            Z = self.fitness_func(positions).reshape(1000, 1000)
            contours = plt.contourf(X, Y, Z, alpha=0.5, zorder=-10, levels=50)
            plt.colorbar(contours, shrink=0.8, extend="both")

    def reset(self):
        np.random.seed(42)
        sampler = qmc.Sobol(d=self.dimensions, seed=42)
        self.X = sampler.random(self.particles) * self.grid_length + self.offset
        self.V = np.random.randn(self.particles, self.dimensions)

        # Initialize data
        self.pbest_loc = self.X.copy()
        self.pbest_val = self.fitness_func(self.pbest_loc)
        self.gbest_loc = self.pbest_loc[self.pbest_val.argmin(), :]
        self.gbest_val = self.pbest_val.min()
        self.gbest_t = 0

    def update_position_torus(self):
        self.X += self.V
        self.X -= self.offset
        self.X %= self.grid_length
        self.X += self.offset

    def update_position_grid(self):
        self.X += self.V
        self.X = np.clip(self.X, self.lower_lim, self.upper_lim)

    def toroidal_distance(self, point1, point2):
        xdiff = abs(point1[0] - point2[0])
        if xdiff > self.grid_length / 2:
            xdiff = self.grid_length - xdiff

        ydiff = abs(point1[1] - point2[1])
        if ydiff > self.grid_length / 2:
            ydiff = self.grid_length - ydiff

        return np.sqrt(xdiff * xdiff + ydiff * ydiff)

    def nearest_neighbor_velocity_matching(self):
        # calculate distances between particles
        distances = cdist(self.X, self.X, self.toroidal_distance)

        # remove distances between self
        np.fill_diagonal(distances, np.inf)

        # find closest neighbor
        nearest = distances.argmin(axis=1)

        # update velocity
        self.V = self.V[nearest, :]

    def craziness(self):
        self.V += np.random.rand(self.particles, self.dimensions) - 0.5

    def inequality_cornfield(self):
        new_vals = self.fitness_func(self.X)
        pbest_mask = self.pbest_val >= new_vals
        self.pbest_loc[pbest_mask, :] = self.X[pbest_mask, :]
        self.pbest_val[pbest_mask] = new_vals[pbest_mask]

        self.gbest_loc = self.pbest_loc[self.pbest_val.argmin(), :]

        pbest_val_min = self.pbest_val.min()
        if self.gbest_val > pbest_val_min:
            self.gbest_val = pbest_val_min
            self.gbest_t = self.t

        r1, r2 = np.random.rand(2)
        self.V += self.p_inc * r1 * np.where(self.X > self.pbest_loc, -1, 1)
        self.V += self.g_inc * r2 * np.where(self.X > self.gbest_loc, -1, 1)

    def difference_cornfield(self, p_inc: float, g_inc: float, w: float):
        new_vals = self.fitness_func(self.X)
        pbest_mask = self.pbest_val >= new_vals
        self.pbest_loc[pbest_mask, :] = self.X[pbest_mask, :]
        self.pbest_val[pbest_mask] = new_vals[pbest_mask]

        self.gbest_loc = self.pbest_loc[self.pbest_val.argmin(), :]

        pbest_val_min = self.pbest_val.min()
        if self.gbest_val > pbest_val_min:
            self.gbest_val = pbest_val_min
            self.gbest_t = self.t

        r1, r2 = np.random.rand(2)
        self.V *= w
        self.V += p_inc * r1 * (self.pbest_loc - self.X)
        self.V += g_inc * r2 * (self.gbest_loc - self.X)

    def plot(self):
        # plot animation
        plt.xlim(self.lower_lim[0], self.upper_lim[0])
        plt.ylim(self.lower_lim[1], self.upper_lim[1])
        plt.scatter(self.grid_center[0], self.grid_center[1], color="black", marker="s")
        plt.scatter(self.X[:, 0], self.X[:, 1], color=self.colors)

        if self.arrows:
            plt.quiver(
                self.X[:, 0],
                self.X[:, 1],
                self.V[:, 0],
                self.V[:, 1],
                color="blue",
                angles="xy",
                scale_units="xy",
                scale=1,
            )

        if self.mode > 1:
            plt.title(
                f"Gbest value: {self.gbest_val} - Gbest t: {self.gbest_t}\nGbest location: {self.gbest_loc}"
            )
            plt.scatter(self.gbest_loc[0], self.gbest_loc[1], color="green", marker="p")

    def run(self, arrows: bool = False):
        self.arrows = arrows

        slider = widgets.IntText(description="Iteration", disabled=True)
        play = widgets.Play(min=0, max=self.duration, interval=50)
        widgets.jslink((play, "value"), (slider, "value"))

        display(
            widgets.HBox([play, slider]),
            widgets.interactive_output(self.wrapper, {"t": play}),
        )

    def wrapper(self, t):
        self.t = t
        if t == 0:
            self.reset()

        self.plot()
        self.version[self.mode]()

    def nnvm(self):
        self.update_position_torus()
        self.nearest_neighbor_velocity_matching()

    def nnvm_craziness(self):
        self.update_position_torus()
        self.nearest_neighbor_velocity_matching()
        self.craziness()

    def cornfield_vector(self):
        self.update_position_grid()
        self.nearest_neighbor_velocity_matching()
        self.craziness()
        self.inequality_cornfield()

    def only_cornfield_vector(self):
        self.update_position_grid()
        self.inequality_cornfield()

    def current(self):
        self.update_position_grid()
        self.difference_cornfield(p_inc=2, g_inc=2, w=1)

    def extra(self):
        self.update_position_grid()
        self.difference_cornfield(self.p_inc, self.g_inc, self.w)
