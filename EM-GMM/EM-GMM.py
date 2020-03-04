'''
Written by Naphat Amundsen
04/03/2020
'''
import numpy as np 
from matplotlib import pyplot as plt 
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from typing import Union
from matplotlib.patches import Ellipse

class GMM:
    '''General use GMM algorithm with built in viz tools'''
    def __init__(self, k: int = 3, init_covariance: Union[float, str]='auto') -> None:
        '''
        k: number of centroids / components
        '''
        self.k = k 
        self.init_covariance = init_covariance
        self._plot_flag=False

    def _init_dtype(self, dim):
        return np.dtype([
            ('mean', float, dim),
            ('cov', float, (dim,dim)),
            ('mix', float)
        ])

    def _prepare_before_fit(self, X: np.ndarray) -> None:
        '''
        Prepares object attributes and such before fitting
        '''
        self.X = X
        self.N, self.dim = X.shape
        self.X_std = np.std(X)

        if self.init_covariance == 'auto':
            self.init_covariance = np.var(X)

        # Initialize component placeholders
        self.components = np.empty(self.k, dtype=self._init_dtype(self.dim))

        # Pick random points as initial mean positions
        # Reshape to handle case for 1-dim data
        self.components['mean'] = \
            X[np.random.choice(range(self.N), self.k, replace=False)].reshape(*self.components['mean'].shape)
        # Initialize covariance matrices with scaled identity matrices
        self.components['cov'] = np.repeat(self.init_covariance*np.eye(self.dim)[np.newaxis,...], self.k, axis=0) 
        # Initialize uniform mixing weights
        self.components['mix'] = np.full(self.k, 1/self.k)

        #Weight for each data point, columns are respective to components
        self.weights = np.empty((self.N, self.k))  

        self.hood_history = []
        # Calculate starting weights in order to 
        # calculate initial likelihood
        self.E_step()
        self.calc_hood()

        #EM iterations
        self.em_iterations = 0 

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ''''Returns log likelihoods for each data point'''
        hood = np.zeros(X.shape[0])
        for component in self.components:
            hood += component['mix'] * mvn.pdf(x=X, mean=component['mean'], cov=component['cov'])
        return np.log(hood)
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        '''Returns log likelihoods for each data point'''
        return self.predict_proba(X)

    def calc_hood(self) -> None:
        '''
        Calculate the GMM log-likelihood 
        '''
        self.hood_history.append(self.predict_proba(self.X).sum())

    def E_step(self) -> None:
        '''
        E-step: Calculates the weights for each datapoint with 
        respect to each component. 
        '''
        # C for component
        for i, c in enumerate(self.components):
            self.weights[:,i] = c['mix'] * mvn.pdf(x=self.X, mean=c['mean'], cov=c['cov'])
        
        # Row-wise division 
        self.weights /= self.weights.sum(axis=1).reshape(-1,1)

    def M_step(self) -> None:
        '''
        M-step: Updates the mean, covariance and priors of the 
        components. 
        '''
        # C for component
        for i, c in enumerate(self.components):
            # Vector of weights for component
            w = self.weights[:,i]
            w_sum = w.sum()            

            # Update component mean
            c['mean'] = w@self.X/w_sum
            # Update component covariance
            D = self.X - c['mean']
            c['cov'] = D.T@(D*w.reshape(-1,1))/w_sum
            # Update component mixing probability
            c['mix'] = w_sum/self.N

    def get_labels(self) -> np.ndarray:
        '''Returns hard labels'''
        return np.argmax(self.weights, axis=1)
        
    def EM_iterate(self) -> None:
        '''Do one EM iteration and save log-likelihood'''
        self.E_step()
        self.M_step()
        self.calc_hood()
        self.em_iterations += 1

    def fit(self, X, maxiter: int=420, rtol: float=1e-8, atol: float=1e-3) -> None:
        '''
        Fit the thing
        '''
        self._prepare_before_fit(X)

        for i in range(maxiter):
            self.EM_iterate()
            if np.allclose(self.hood_history[-1], self.hood_history[-2], rtol=rtol, atol=atol): 
                break
    
    def fit_animate(self, X, maxiter: int=420, rtol: float=1e-8, atol: float=1e-3, 
                    figsize:tuple = (12,6), axis: list=[0,1]) -> None:
        '''
        Fit while visualizing
        '''
        from matplotlib import animation as anime
        self._prepare_before_fit(X)
        self._init_plot(figsize)

        ALL = np.array([self.left, self.right, self.lower])
        def animate(i):
            [ax.clear() for ax in ALL]    
            self.EM_iterate()
            if np.allclose(self.hood_history[-1], self.hood_history[-2], rtol=rtol, atol=atol): 
                movie.event_source.stop()
                print('Converged')
            self.plot_result(axis=axis, show=False)

        movie = anime.FuncAnimation(self.fig, animate, frames=maxiter, interval=120, blit=False, repeat=False)
        plt.show()
        self._plot_flag = False        

    def get_mise(self, validation_data: np.ndarray) -> float:
        '''Approximation of mean integrated square error'''
        second_term = np.zeros(validation_data.shape[0])
       
        for c in self.components:
            second_term += c['mix'] * mvn.pdf(x=validation_data, mean=c['mean'], cov=c['cov']) 
        
        second_term = 2/validation_data.shape[0] * second_term.sum()

        cum = 0
        for ci in self.components:
            for cj in self.components:
                cum += ci['mix'] * cj['mix'] * mvn.pdf(x=ci['mean'], mean=cj['mean'], cov=ci['cov'] + cj['cov'])

        cum = cum - second_term
        return cum

    @staticmethod
    def draw_ellipse(position, covariance, ax, **kwargs):
        '''
        Source:
        https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html
        
        Draw an ellipse with a given position and covariance

        Expects 2D covariance
        '''
        # Convert covariance to principal axes
        U, S, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(S)
        
        # Draw the Ellipse
        # Multiple draws for one covariance to express contours
        for nsig in range(1, 4):
            ax.add_patch(Ellipse(
                xy=position, 
                width=nsig * width, 
                height=nsig * height,
                angle=angle, 
                **kwargs)
            )

    def _init_plot(self, figsize) -> tuple:
        '''Initialize plot attributes'''
        self._plot_flag = True
        self.colors = ['tomato', 'orange', 'deepskyblue', 'yellow', 'red', 'blue']
        self.nc = len(self.colors)
        self.centroid_colors = [self.colors[i%self.nc] for i in range(len(self.components))]
        self.centroid_kwargs = dict(marker='h', s=200, c=self.centroid_colors, edgecolor='k')
        self.X_kwargs = dict(edgecolor='gray')

        self.fig = plt.figure(figsize=figsize)
        # Upper left
        self.left = self.fig.add_subplot(221)
        # Upper right
        self.right = self.fig.add_subplot(222)
        # Whole lower
        self.lower = self.fig.add_subplot(212)

    def plot_result(self, figsize:tuple = (12,6), axis: list=[0,1], show=True) -> Union[None, 'Axes']:
        '''
        Plots GMM result. If data is more that two axis', you can select
        which axis' to plot in axis parameter
        '''
        assert len(axis) == 2, 'Length of axis must be 2'
        if self._plot_flag == False: 
            self._init_plot(figsize)

        # Handle 1d case
        if self.dim > 1:
            centroids = self.components['mean'][:,axis].T
            X = self.X[:,axis].T
        else:
            # Effectively give y-values (zeros) so I can plot them
            centroids = np.column_stack([self.components['mean'], np.zeros_like(self.components['mean'])]).T
            X = np.column_stack([self.X, np.zeros_like(self.X)]).T

        centroid_colors = [self.colors[j%self.nc] for j in range(len(centroids.T))]
        # EM subplot
        self.left.scatter(*X, **self.X_kwargs)
        self.left.scatter(*centroids, **self.centroid_kwargs)

        # Handle 1d case
        if self.dim > 1:
            for i, c in enumerate(self.components):
                self.draw_ellipse(c['mean'][axis], c['cov'][axis], self.left, alpha=c['mix'], color=self.colors[i%self.nc])        
        else:
            for i, c in enumerate(self.components):
                xrange = np.linspace(X[0].min()-self.X_std, X[0].max()+self.X_std, 420)
                self.left.plot(xrange, norm.pdf(xrange, c['mean'], c['cov'][0][0])+0.5, color=self.colors[i%self.nc])

        self.left.set_title('GMM-Components')

        # Categorized data point subplot (hard labels)
        labels = self.get_labels()
        self.right.scatter(*X, c=[self.colors[j%self.nc] for j in labels], **self.X_kwargs)
        self.right.scatter(*centroids, **self.centroid_kwargs)
        self.right.set_title('Hard clustering')

        # Likelihood subplot
        self.lower.grid()
        self.lower.plot(np.arange(1,len(self.hood_history)), self.hood_history[1:])
        self.lower.set_title('Log-Likelihood')
        self.lower.set_xlabel('Iterations')

        plt.suptitle(f'Iteration {self.em_iterations}')
        
        self.fig.tight_layout()
        if show: 
            self._plot_flag = False
            plt.show()

if __name__ == '__main__':
    from sklearn.datasets import load_iris
    data=load_iris()['data'][:,3].reshape(-1,1)
    data=load_iris()['data']
    print(data.shape)
    
    k = 3
    JohnWick = GMM(k)
    JohnWick.fit_animate(data, axis=[2,3])
    # JohnWick.plot_result(axis=[0,3])
