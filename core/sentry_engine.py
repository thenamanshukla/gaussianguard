import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class KernelSentry:
    def __init__(self, length_scale=1.0, noise_level=0.1):
        """
        The Sentry Engine. Uses Gaussian Process Regression to model 
        and predict normal boundaries of a time-series signal.
        """
        # Define an RBF kernel with allowable bounds for optimization
        self.kernel = C(1.0, (1e-2, 1e2)) * RBF(length_scale=length_scale, length_scale_bounds=(1e-1, 1e2))
        
        # alpha represents the measurement noise variance (sigma_n^2)
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=noise_level**2,
            n_restarts_optimizer=10,
            random_state=42
        )

    def learn(self, X, y):
        """
        Condition the GP on observed training data. 
        Optimizes hyperparameters using Log-Marginal Likelihood maximization.
        """
        self.gp.fit(X, y)
        print(f"─── [KernelSentry] Calibration Complete ───")
        print(f"Optimized Architecture Parameters: {self.gp.kernel_}\n")

    def check_integrity(self, X_new):
        """
        Calculates the conditioned posterior mean and standard deviation profiles.
        """
        # return_std=True forces the execution of the conditioned variance matrix formula
        y_mean, y_std = self.gp.predict(X_new, return_std=True)
        return y_mean, y_std