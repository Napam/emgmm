"""
Written by Naphat Amundsen
04/03/2020
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from typing import Union
from matplotlib.patches import Ellipse
from matplotlib import axes

class GMM:
    """General use GMM algorithm with built in viz tools"""

    def __init__(self, k: int = 3, init_covariance: Union[float, str] = "auto") -> None:
        """
        k: number of centroids / components

        init_covarince: initial value for diagonal elements in covariance matrix. If
                        'auto' is given as argument (default), the value will be the overall
                        variance of the data
        """
        self.k = k
        self.init_covariance = init_covariance
        self._plot_flag = False

    def _init_dtype(self, dim) -> None:
        return np.dtype([("mean", float, dim), ("cov", float, (dim, dim)), ("mix", float)])

    def __repr__(self):
        return f"GMM(k={self.k})"

    def __str__(self):
        return f"GMM(k={self.k})"

    def _prepare_before_fit(self, X: np.ndarray) -> None:
        """
        Prepares object attributes and such before fitting
        """
        self.X = X
        self.N, self.dim = X.shape
        self.X_std = np.std(X)

        if self.init_covariance == "auto":
            self.init_covariance = np.var(X)

        # Initialize component placeholders
        self.components = np.empty(self.k, dtype=self._init_dtype(self.dim))

        # Pick random points as initial mean positions
        # Reshape to handle case for 1-dim data
        self.components["mean"] = X[np.random.choice(range(self.N), self.k, replace=False)].reshape(
            *self.components["mean"].shape
        )
        # Initialize covariance matrices with scaled identity matrices
        self.components["cov"] = np.repeat(
            self.init_covariance * np.eye(self.dim)[np.newaxis, ...], self.k, axis=0
        )
        # Initialize uniform mixing weights
        self.components["mix"] = np.full(self.k, 1 / self.k)

        # Weight for each data point, columns are respective to components
        self.weights = np.empty((self.N, self.k))

        self.hood_history = []
        # Calculate starting weights in order to calculate initial likelihood
        self._E_step()  # This automatically logs likelihood

        # EM iterations
        self.em_iterations = 0

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """ 'Returns log likelihoods for each data point"""
        hood = np.zeros(X.shape[0])
        for component in self.components:
            hood += component["mix"] * mvn.pdf(x=X, mean=component["mean"], cov=component["cov"])
        return np.log(hood)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Returns log likelihoods for each data point"""
        return self.predict_proba(X)

    def _E_step(self) -> None:
        """
        E-step: Calculates the weights for each datapoint with
        respect to each component while assuming model parameters are correct
        """
        # C for component
        for i, c in enumerate(self.components):
            self.weights[:, i] = c["mix"] * mvn.pdf(x=self.X, mean=c["mean"], cov=c["cov"])
        w_axis_sum = self.weights.sum(axis=1)

        # Log the log-likelihood
        self.hood_history.append(np.log(w_axis_sum).sum())

        # Row-wise division
        self.weights /= w_axis_sum.reshape(-1, 1)

    def _M_step(self) -> None:
        """
        M-step: Updates the mean, covariance and priors of the
        components.
        """
        # C for component
        for i, c in enumerate(self.components):
            # Vector of weights for component
            w = self.weights[:, i]
            w_sum = w.sum()

            # Update component mean
            c["mean"] = w @ self.X / w_sum
            # Update component covariance
            D = self.X - c["mean"]
            c["cov"] = D.T @ (D * w.reshape(-1, 1)) / w_sum
            # Update component mixing probability
            c["mix"] = w_sum / self.N

    def get_labels(self) -> np.ndarray:
        """Returns hard labels"""
        return np.argmax(self.weights, axis=1)

    def _EM_iterate(self) -> None:
        """Do one EM iteration and save log-likelihood"""
        # _prepare_before_fit method should have been invoked once
        # before using this method. _prepare_before_fit will call
        # the E_step method once
        self._M_step()
        self._E_step()
        self.em_iterations += 1

    def fit(self, X, maxiter: int = 420, rtol: float = 1e-8, atol: float = 1e-3) -> None:
        """
        Fit the thing
        """
        self._prepare_before_fit(X)

        for i in range(maxiter):
            self._EM_iterate()
            if np.allclose(self.hood_history[-1], self.hood_history[-2], rtol=rtol, atol=atol):
                break
        return self

    def fit_animate(
        self,
        X,
        maxiter: int = 420,
        rtol: float = 1e-8,
        atol: float = 1e-3,
        figsize: tuple = (12, 6),
        axis: list = [0, 1],
    ) -> None:
        """
        Fit while visualizing
        """
        from matplotlib import animation as anime

        self._prepare_before_fit(X)
        self._init_plot(figsize, axis)

        ALL = np.array([self.left, self.right, self.lower])

        def animate(i):
            # [ax.clear() for ax in ALL]
            self._EM_iterate()
            if np.allclose(self.hood_history[-1], self.hood_history[-2], rtol=rtol, atol=atol):
                movie.event_source.stop()
                print("Converged")
            self.plot_result(axis=axis, show=False)

        movie = anime.FuncAnimation(
            self.fig, animate, frames=maxiter, interval=32, blit=False, repeat=False
        )
        # movie = anime.FuncAnimation(self.fig, animate, frames=30, interval=128, blit=False, repeat=False)
        # movie.save('GMM.gif', writer='PillowWriter')
        # movie = anime.FuncAnimation(self.fig, animate, frames=40, interval=128, blit=False, repeat=False)
        # movie.save('GMM.mp4', writer='ffmpeg')
        plt.show()
        self._plot_flag = False
        return self

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
        dummypoints = np.zeros(len(self.centroid_colors))

        self.fig = plt.figure(figsize=figsize)
        # Upper left
        self.left = self.fig.add_subplot(221)
        self.left.set_title("GMM-Components")
        self.left_scatter = self.left.scatter(*self.X[:,axis].T)
        self.left_centroids = self.left.scatter(dummypoints, dummypoints, **self.centroid_kwargs)
        # Upper right
        self.right = self.fig.add_subplot(222)
        self.right.set_title("Soft clustering")
        self.right_scatter = self.right.scatter(*self.X[:,axis].T)
        self.right_centroids = self.right.scatter(dummypoints, dummypoints, **self.centroid_kwargs, zorder=32)
        # Whole lower
        self.lower = self.fig.add_subplot(212)
        self.lower.set_title("Log-Likelihood")
        self.lower.set_xlabel("Iterations")
        self.lower.grid()
        self.lower_x = []
        self.lower_plot = self.lower.plot(self.lower_x, self.hood_history[1:])[0]

    def _plot_left(self, X: np.ndarray, centroids: np.ndarray, axis: axes.Axes):
        """Method to plot upper left subplot (Ellipsis plot)"""
        self.left_centroids.set_offsets(centroids.T)

        # Handle 1d case
        self.left.patches.clear()
        if self.dim > 1:
            for i, c in enumerate(self.components):
                self._draw_ellipse(
                    c["mean"][axis],
                    c["cov"][axis],
                    self.left,
                    alpha=c["mix"],
                    color=self.colors[i % self.nc],
                )
        else:
            xrange = np.linspace(X[0].min() - self.X_std, X[0].max() + self.X_std, 200)
            for i, c in enumerate(self.components):
                self.left.plot(
                    xrange,
                    np.log(c["mix"] * norm.pdf(xrange, c["mean"], c["cov"][0][0]) + 1.1),
                    color=self.colors[i % self.nc]
                )

    def _plot_right(self, X, centroids):
        """Method to plot upper right subplot (Soft clustering)"""
        self.right_scatter.set_color(np.clip(self.weights @ self.colors, 0, 1))
        self.right_centroids.set_offsets(centroids.T)

    def _plot_lower(self, X):
        """Method to plot lower subplot (Likelihood graph)"""
        # if not self.lower_x: return
        self.lower_x.append(self.em_iterations)
        self.lower_plot.set_data(self.lower_x, self.hood_history[1:])
        self.lower.relim()
        self.lower.autoscale()
        

    def plot_result(
        self, figsize: tuple = (12, 6), axis: list = [0, 1], show=True
    ) -> Union[None, "Axes"]:
        """
        Plots GMM result. If data is more that two axis', you can select
        which axis' to plot in axis parameter
        """
        assert len(axis) == 2, "Length of axis must be 2"
        if self._plot_flag == False:
            self._init_plot(figsize)

        # Handle 1d case
        if self.dim > 1:
            centroids = self.components["mean"][:, axis].T
            X = self.X[:, axis].T
        elif self.dim == 1:
            # Effectively give y-values (zeros) so I can plot them
            centroids = np.column_stack(
                [self.components["mean"], np.zeros_like(self.components["mean"])]
            ).T
            X = np.column_stack([self.X, np.zeros_like(self.X)]).T

        self._plot_left(X, centroids, axis)
        self._plot_right(X, centroids)
        self._plot_lower(X)

        plt.suptitle(f"Iteration {self.em_iterations}")

        self.fig.tight_layout()
        if show:
            self.show()

    def show(self):
        """Ensures that _plot_flag gets assigned correctly"""
        self._plot_flag = False
        plt.show()

    def get_mise(self, validation_data: np.ndarray) -> float:
        """Approximation of mean integrated square error"""
        second_term = np.zeros(validation_data.shape[0])

        for c in self.components:
            second_term += c["mix"] * mvn.pdf(x=validation_data, mean=c["mean"], cov=c["cov"])
        second_term = 2 / validation_data.shape[0] * second_term.sum()

        cum = 0
        for ci in self.components:
            for cj in self.components:
                cum += (
                    ci["mix"]
                    * cj["mix"]
                    * mvn.pdf(x=ci["mean"], mean=cj["mean"], cov=ci["cov"] + cj["cov"])
                )
        cum = cum - second_term
        return cum

    def get_bic(self) -> float:
        """Returns BIC value"""
        d = self.dim
        penalty = (self.k * (d + (d + 1) * d / 2) + (self.k - 1)) / 2 * np.log(self.N)
        return self.hood_history[-1] - penalty

    def get_aic(self) -> float:
        """Returns AIC value"""
        d = self.dim
        penalty = 2 * (self.k * (d + (d + 1) * d / 2) + (self.k - 1)) / self.N
        return -2 * self.hood_history[-1] / self.N - penalty


if __name__ == "__main__":
    np.random.seed(420)
    from sklearn.datasets import load_iris

    # data=load_iris()['data'][:,3].reshape(-1,1)
    data = load_iris()["data"]

    axis = [2, 1]
    k = 3

    gmm = GMM(k)
    # gmm.fit(data)
    gmm.fit_animate(data, axis=axis)
    # gmm.plot_result(axis=[0,3])
