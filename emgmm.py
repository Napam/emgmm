"""
Written by Naphat Amundsen
04/03/2020
"""
import numpy as np
from scipy.stats import multivariate_normal as mvn
from typing import Optional, Tuple, Union


class GMM:
    """General use GMM algorithm with built in viz tools"""

    def __init__(self, k: int = 3, init_covariance: Union[float, str] = "auto") -> None:
        """
        k: number of gaussians

        init_covarince: initial value for diagonal elements in covariance matrix. If
                        'auto' is given as argument (default), the value will be the overall
                        variance of the data
        """
        self.k: int = k
        self.init_covariance: Union[float, str] = init_covariance
        self._plotter = None  # For plotter instance, if needed

    def _init_dtype(self, dim) -> None:
        return np.dtype([("mean", float, (dim,)), ("cov", float, (dim, dim)), ("mix", float)])

    def __repr__(self):
        return f"GMM(k={self.k}, init_covariance={self.init_covariance})"

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
        """Returns log likelihoods for each data point"""
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
        # c for component
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

    def fit(self, X: np.ndarray, maxiter: int = 128, rtol: float = 1e-8, atol: float = 1e-4) -> None:
        """
        Fit model to training data

        X: training data
        maxiter: maximum number of EM iterations
        rtol: ratio change limit between log likelihood of previous and current iteration (to determine convergence)
        atol: difference limit between log likelihood of previous and current iteration (to determine convergence)
        """
        self._prepare_before_fit(X)

        for i in range(maxiter):
            self._EM_iterate()
            if np.allclose(self.hood_history[-1], self.hood_history[-2], rtol=rtol, atol=atol):
                break
        return self

    def _get_plotter(self):
        if self._plotter is None:
            from plotter import Plotter

            self._plotter: Plotter = Plotter(self)

        return self._plotter

    def fit_animate(self, X, **kwargs) -> "GMM":
        """
        Fit while visualizing

        X: training data
        maxiter: maximum number of EM iterations
        rtol: ratio change limit between log likelihood of previous and current iteration (to determine convergence)
        atol: difference limit between log likelihood of previous and current iteration (to determine convergence)
        figsize: size of figure
        axis: which features to visualize
        file: filename if you want to save to .gif or .mp4. E.g. 'video.mp4' or 'video.gif'
        interval: wait interval in milliseconds between frames
        """
        self._prepare_before_fit(X)
        plotter = self._get_plotter()
        plotter.fit_animate(**kwargs)

        return self

    def plot_result(
        self,
        figsize: Optional[Tuple[float, float]] = None,
        axis: Optional[Tuple[int, int]] = None,
        show: bool = True,
    ) -> "matplotlib.figure":
        """
        Plots GMM result. If data is more that two axis', you can select
        which axis' to plot in axis parameter

        figsize: passed on to plt.figure(figsize)
        axis: which data columns to use
        show: to do plt.show() or not
        """
        if figsize is None:
            figsize = (12, 6)

        if axis is None:
            axis = (0, 1)

        assert len(axis) == 2, "Length of axis must be 2"
        plotter = self._get_plotter()
        plotter._init_plot(self, figsize, axis)
        return plotter.plot_result(figsize, axis, show)

    def show(self) -> None:
        """Will work after plot_result"""
        self._get_plotter().show()

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
        """Returns BIC value based on last log likelihood"""
        d = self.dim
        penalty = (self.k * (d + (d + 1) * d / 2) + (self.k - 1)) / 2 * np.log(self.N)
        return self.hood_history[-1] - penalty

    def get_aic(self) -> float:
        """Returns AIC value based on last log likelihood"""
        d = self.dim
        penalty = 2 * (self.k * (d + (d + 1) * d / 2) + (self.k - 1)) / self.N
        return -2 * self.hood_history[-1] / self.N - penalty


if __name__ == "__main__":
    np.random.seed(420)

    # data = np.loadtxt('iris.txt')[:, 3].reshape(-1, 1)
    data = np.loadtxt("iris.txt")

    axis = [2, 1]
    gmm = GMM(k=3)

    gmm.fit_animate(data, axis=axis, interval=64)
    # gmm.fit(data)
    # gmm.plot_result(axis=[1, 3])
