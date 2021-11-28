from typing import List, Optional, Tuple, Union

from scipy.linalg.decomp import _make_complex_eigvecs
from emgmm import GMM
import numpy as np
from matplotlib import animation as anime, axes
from matplotlib import pyplot as plt
from scipy.stats import norm
from matplotlib.patches import Ellipse

COLORS = [
    "magenta",
    "deepskyblue",
    "orange",
    "lime",
    "lightpink",
    "yellow",
    "green",
    "red",
    "powderblue",
    "tomato",
    "orange",
    "deepskyblue",
    "yellow",
    "blue",
]


class Plotter:
    def __init__(self, gmm):
        self.gmm: GMM = gmm
        self.dim: int = gmm.dim
        self.iterations: List[int] = []

    def fit_animate(
        self,
        maxiter: int = 64,
        rtol: float = 1e-8,
        atol: float = 1e-3,
        figsize: Optional[Tuple[int]] = None,
        axis: Optional[Tuple[int]] = None,
        file: Optional[str] = None,
        interval: int = 32
    ):
        if figsize is None:
            figsize = (12, 6)

        if axis is None:
            axis = (0, 1)

        self._init_plot(self.gmm, figsize, axis)

        def animate(i):
            self.gmm._EM_iterate()
            self.iterations.append(self.gmm.em_iterations)
            if np.allclose(
                self.gmm.hood_history[-1], self.gmm.hood_history[-2], rtol=rtol, atol=atol
            ):
                movie.event_source.stop()
                print("Converged")
            self.plot_result(axis=axis, show=False)

        movie = anime.FuncAnimation(
            self.fig, animate, frames=maxiter, interval=interval, blit=False, repeat=False
        )

        if file:
            import os
            _, extension = os.path.splitext(file)
            assert extension in ['.gif', '.mp4'], "File must have extension .gif or .mp4"
            if extension == '.gif':
                movie.save(file)
            elif extension == '.mp4':
                movie.save(file, writer='ffmpeg')
        else:
            plt.show()

    def _init_plot(self, gmm, figsize, axis):
        """Initialize plot attributes"""
        from matplotlib.colors import to_rgb

        self._plot_flag = True
        self.colors = np.array([to_rgb(c) for c in COLORS[: gmm.k]])
        self.nc = len(self.colors)
        self.centroid_colors = [self.colors[i % self.nc] for i in range(len(gmm.components))]

        self.centroid_kwargs = dict(marker="h", s=200, c=self.centroid_colors, edgecolor="k")
        dummy = [np.zeros(len(self.centroid_colors))] * 2

        # Handle 1d case
        if gmm.dim > 1:
            X = gmm.X[:, axis].T
        elif gmm.dim == 1:
            X = np.column_stack([gmm.X, np.zeros_like(gmm.X)]).T

        self.fig = plt.figure(figsize=figsize)
        # Upper left
        self.left = self.fig.add_subplot(221)
        self.left.set_title("GMM-Components")
        self.left_scatter = self.left.scatter(*X)
        self.left_centroids = self.left.scatter(*dummy, **self.centroid_kwargs)
        if gmm.dim == 1:
            self.xrange_leftplot = np.linspace(X[0].min() - gmm.X_std, X[0].max() + gmm.X_std, 512)
            zeros = np.zeros_like(self.xrange_leftplot)
            self.left_gs = [
                self.left.plot(self.xrange_leftplot, zeros, color=c)[0] for c in self.centroid_colors
            ]
        # Upper right
        self.right = self.fig.add_subplot(222)
        self.right.set_title("Soft clustering")
        self.right_scatter = self.right.scatter(*X)
        self.right_centroids = self.right.scatter(*dummy, **self.centroid_kwargs, zorder=32)
        # Whole lower
        self.lower = self.fig.add_subplot(212)
        self.lower.set_title("Log-Likelihood")
        self.lower.set_xlabel("Iterations")
        self.lower.grid()
        self.iterations = list(range(len(self.gmm.hood_history[1:])))
        self.lower_plot = self.lower.plot(self.iterations, self.gmm.hood_history[1:])[0]

    def plot_result(
        self, figsize: tuple = (12, 6), axis: list = [0, 1], show=True
    ) -> Union[None, axes.Axes]:
        """
        Plots GMM result. If data is more that two axis', you can select
        which axis' to plot in axis parameter
        """
        assert len(axis) == 2, "Length of axis must be 2"

        # Handle 1d case
        if self.dim > 1:
            centroids = self.gmm.components["mean"][:, axis].T
            X = self.gmm.X[:, axis].T
        elif self.dim == 1:
            # Effectively give y-values (zeros) so I can plot them
            centroids = np.column_stack(
                [self.gmm.components["mean"], np.zeros_like(self.gmm.components["mean"])]
            ).T
            X = np.column_stack([self.gmm.X, np.zeros_like(self.gmm.X)]).T

        self._plot_left(X, centroids, axis)
        self._plot_right(X, centroids)
        self._plot_lower(X)

        plt.suptitle(f"Iteration {self.gmm.em_iterations}")

        self.fig.tight_layout()
        if show:
            self.show()

    def _plot_left(self, X: np.ndarray, centroids: np.ndarray, axis: axes.Axes):
        """Method to plot upper left subplot (Ellipsis plot)"""
        self.left_centroids.set_offsets(centroids.T)
        if self.dim <= 1:
            self._plot_left_1d()
        else:
            self._plot_left_2d(axis)

    def _plot_left_1d(self):
        old_lim = self.left.get_ylim()
        max_y = old_lim[1]
        for left_plot, c in zip(self.left_gs, self.gmm.components):
            pdf = np.log(c["mix"] * norm.pdf(self.xrange_leftplot, c["mean"], c["cov"][0][0]) + 1.1)
            left_plot.set_ydata(pdf)
            max_y = max(pdf.max(), max_y)
        self.left.set_ylim([old_lim[0], max_y])

    def _plot_left_2d(self, axis):
        self.left.patches.clear()
        for i, c in enumerate(self.gmm.components):
            self._draw_ellipse(
                c["mean"][axis],
                c["cov"][axis],
                self.left,
                alpha=c["mix"],
                color=self.colors[i % self.nc],
            )

    def _plot_right(self, X, centroids):
        """Method to plot upper right subplot (Soft clustering)"""
        self.right_scatter.set_color(np.clip(self.gmm.weights @ self.colors, 0, 1))
        self.right_centroids.set_offsets(centroids.T)

    def _plot_lower(self, X):
        """Method to plot lower subplot (Likelihood graph)"""
        # if not self.lower_x: return
        self.lower_plot.set_data(self.iterations, self.gmm.hood_history[1:])
        self.lower.relim()
        self.lower.autoscale()

    def show(self):
        """Ensures that _plot_flag gets assigned correctly"""
        plt.show()

    @staticmethod
    def _draw_ellipse(position, covariance, ax, **kwargs):
        """
        Source:
        https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html

        Draw an ellipse with a given position and covariance

        Expects 2D covariance
        """
        # Convert covariance to principal axes
        U, S, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(S)

        # Draw the Ellipse
        # Multiple draws for one covariance to express contours
        # print(1/np.linalg.norm(S))
        for nsig in range(1, 4):
            ax.add_patch(
                Ellipse(xy=position, width=nsig * width, height=nsig * height, angle=angle, **kwargs)
            )