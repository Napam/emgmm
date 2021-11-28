from typing import List, Optional, Tuple
from emgmm import GMM
import numpy as np 
from matplotlib import animation as anime
from matplotlib import pyplto as plt 

class Plotter:
    def __init__(self, gmm):
        self.gmm = gmm

    def fit_animate(
        self, 
        gmm: GMM, 
        maxiter: int = 64,
        rtol: float = 1e-8,
        atol: float = 1e-3,
        figsize: Optional[Tuple[int]] = None,
        axis: Optional[Tuple[int]] = None
    ):
        if figsize is None:
            figsize = (12, 6)
        
        if axis is None:
            axis = (0, 1)

        self._init_plot(figsize, axis)
        def animate(i):
            self._EM_iterate()
            if np.allclose(gmm.hood_history[-1], gmm.hood_history[-2], rtol=rtol, atol=atol):
                movie.event_source.stop()
                print("Converged")
            self.plot_result(axis=axis, show=False)
        
        movie = anime.FuncAnimation(
            self.fig, animate, frames=maxiter, interval=16, blit=False, repeat=False
        )

        # movie = anime.FuncAnimation(self.fig, animate, frames=30, interval=128, blit=False, repeat=False)
        # movie.save('GMM.gif', writer='PillowWriter')
        # movie = anime.FuncAnimation(self.fig, animate, frames=40, interval=128, blit=False, repeat=False)
        # movie.save('GMM.mp4', writer='ffmpeg')

        plt.show()

    def _init_plot(self, figsize, axis) -> tuple:
        """Initialize plot attributes"""
        from matplotlib.colors import to_rgb

        self._plot_flag = True
        self.colors = [
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
        self.colors = np.array([to_rgb(c) for c in self.colors[: self.k]])
        self.nc = len(self.colors)
        self.centroid_colors = [self.colors[i % self.nc] for i in range(len(self.components))]

        self.centroid_kwargs = dict(marker="h", s=200, c=self.centroid_colors, edgecolor="k")
        dummy = [np.zeros(len(self.centroid_colors))] * 2

        # Handle 1d case
        if self.dim > 1:
            X = self.X[:, axis].T
        elif self.dim == 1:
            X = np.column_stack([self.X, np.zeros_like(self.X)]).T

        self.fig = plt.figure(figsize=figsize)
        # Upper left
        self.left = self.fig.add_subplot(221)
        self.left.set_title("GMM-Components")
        self.left_scatter = self.left.scatter(*X)
        self.left_centroids = self.left.scatter(*dummy, **self.centroid_kwargs, zorder=32)
        if self.dim == 1:
            self.xrange_leftplot = np.linspace(X[0].min() - self.X_std, X[0].max() + self.X_std, 512)
            zeros = np.zeros_like(self.xrange_leftplot)
            self.left_gs = [self.left.plot(self.xrange_leftplot, zeros, color=c)[0] for c in self.centroid_colors]
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
        self.lower_x = []
        self.lower_plot = self.lower.plot(self.lower_x, self.hood_history[1:])[0]