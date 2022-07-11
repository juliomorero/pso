"""Adapted from https://machinelearningmastery.com/a-gentle-introduction-to-particle-swarm-optimization/"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import qmc

# Set seed and sobol generator
seed = 42
np.random.seed(seed)
sampler = qmc.Sobol(d=2, seed=seed)


def f(x, y):
    """
    Objective function -> Schaffer's F6
    Formula can be found in https://arxiv.org/pdf/1207.4318.pdf
    """

    num = np.sin(np.sqrt(x**2 + y**2)) ** 2 - 0.5
    den = (1 + 0.001 * (x**2 + y**2)) ** 2

    return 0.5 + (num / den)


# Compute and plot the function in 3D within [-100,100]x[-100,100]
max_range = 100
axis = np.linspace(-max_range, max_range, 1000)
x, y = np.array(np.meshgrid(axis, axis))
z = f(x, y)

# Find the global minimum
x_min = x.ravel()[z.argmin()]
y_min = y.ravel()[z.argmin()]
x_min = y_min = 0

# Hyper-parameter of the algorithm
c1 = c2 = 0.1
w = 0.7

# current_best 0.1 0.7 v=0 32 part

# Create particles
n_particles = 32
X = (sampler.random_base2(m=5)[:n_particles].T - 0.5) * 2 * max_range
V = np.random.randn(2, n_particles)
V = np.zeros((2, n_particles))
# V = np.random.rand(2, n_particles)

# Initialize data
pbest = X
pbest_obj = f(X[0], X[1])
gbest = pbest[:, pbest_obj.argmin()]
gbest_obj = pbest_obj.min()


def update():
    "Function to do one iteration of particle swarm optimization"
    global V, X, pbest, pbest_obj, gbest, gbest_obj
    # Update params
    r1, r2 = np.random.rand(2)
    V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest.reshape(-1, 1) - X)
    X = X + V
    obj = f(X[0], X[1])
    pbest[:, (pbest_obj >= obj)] = X[:, (pbest_obj >= obj)]
    pbest_obj = np.array([pbest_obj, obj]).min(axis=0)
    gbest = pbest[:, pbest_obj.argmin()]
    gbest_obj = pbest_obj.min()


# Set up base figure: The contour map
fig, ax = plt.subplots(figsize=(10, 10))
fig.set_tight_layout(True)
ax.set_xlim([-max_range, max_range])
ax.set_ylim([-max_range, max_range])

img = ax.imshow(
    z,
    extent=[-max_range, max_range, -max_range, max_range],
    origin="lower",
    cmap="viridis",
)
fig.colorbar(img, ax=ax, fraction=0.0455, pad=0.04)

ax.plot([x_min], [y_min], marker="x", markersize=5, color="white")
contours = ax.contourf(x, y, z, origin="lower")

p_plot = ax.scatter(X[0], X[1], marker="o", color="blue", alpha=0.5)
pbest_plot = ax.scatter(pbest[0], pbest[1], marker="o", color="black", alpha=0.5)
gbest_plot = plt.scatter([gbest[0]], [gbest[1]], marker="*", s=100, color="red")

plt.savefig("initial.png")


def animate(i):
    "Steps of PSO: algorithm update and show in plot"
    title = "Iteration {:02d}".format(i)
    # Update params
    update()
    # Set picture
    ax.set_title(title)
    pbest_plot.set_offsets(pbest.T)
    p_plot.set_offsets(X.T)
    gbest_plot.set_offsets(gbest.reshape(1, -1))
    return ax, pbest_plot, p_plot, gbest_plot


anim = FuncAnimation(
    fig, animate, frames=list(range(1, 50)), interval=500, blit=False, repeat=True
)
anim.save("pso.gif", dpi=120, writer="imagemagick")

print("PSO found best solution at f({})={}".format(gbest, gbest_obj))
print("Global optimal at f({})={}".format([x_min, y_min], f(x_min, y_min)))
