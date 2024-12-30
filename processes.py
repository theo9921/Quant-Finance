'''
Library to simulate different SDEs
@Author: T.T. Ouzounellis Kavlakonis
@Date: 29Dec24
'''

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, ks_2samp
from sklearn.metrics import mean_absolute_error, mean_squared_error

class BaseOUProcess:
    def __init__(self, T, n_steps, n_simulations):
        self.T = T
        self.n_steps = n_steps
        self.n_simulations = n_simulations
        self.dt = T / n_steps
        self.time = np.linspace(0, T, n_steps + 1)
        self.B = None

    def plot_paths(self, max_paths=None, title='Process Paths'):
        if self.B is None:
            raise ValueError("Simulation not yet run. Call simulate() first.")
        mean_B = np.mean(self.B, axis=1)
        std_B = np.std(self.B, axis=1)
        plt.figure(figsize=(12, 6))
        plt.plot(self.time, self.B[:, :max_paths] if max_paths else self.B, alpha=0.6)
        plt.plot(self.time, mean_B, color='red', label='Mean', linewidth=2)
        plt.fill_between(self.time, mean_B - std_B, mean_B + std_B, color='black', alpha=0.2, label='1 Std Dev')
        plt.plot(self.time, mean_B - std_B, color='black', linestyle='--', linewidth=1)
        plt.plot(self.time, mean_B + std_B, color='black', linestyle='--', linewidth=1)
        plt.title(title)
        plt.xlabel('Time (Years)')
        plt.ylabel('Basis (B)')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_distribution_at_final_time(self, title='Distribution at Final Time Step'):
        if self.B is None:
            raise ValueError("Simulation not yet run. Call simulate() first.")
        final_values = self.B[-1, :]
        plt.figure(figsize=(8, 6))
        plt.hist(final_values, bins=30, density=True, alpha=0.7, color='gray')
        plt.axvline(np.mean(final_values), color='black', linestyle='--', label='Mean')
        plt.axvline(np.mean(final_values) - np.std(final_values), color='black', linestyle='--', label='1 Std Dev')
        plt.axvline(np.mean(final_values) + np.std(final_values), color='black', linestyle='--')
        plt.title(title)
        plt.xlabel('Basis (B)')
        plt.ylabel('Density')
        plt.legend()
        plt.grid()
        plt.show()

class OrnsteinUhlenbeckProcess(BaseOUProcess):
    def __init__(self, mu, theta, sigma, B0, T, n_steps, n_simulations):
        super().__init__(T, n_steps, n_simulations)
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.B0 = B0

    def simulate(self):
        """
        Simulate the Ornstein-Uhlenbeck process.
        """
        B = np.zeros((self.n_steps + 1, self.n_simulations))
        B[0, :] = self.B0
        for t in range(1, self.n_steps + 1):
            dW = np.random.normal(0, np.sqrt(self.dt), self.n_simulations)
            B[t, :] = B[t - 1, :] + self.theta * (self.mu - B[t - 1, :]) * self.dt + self.sigma * dW
        self.B = B
        return self.time, self.B

    @staticmethod
    def calibrate(B_data, dt):
        """
        Calibrate the Ornstein-Uhlenbeck process to historical data.
        """
        def log_likelihood(params):
            mu, theta, sigma = params
            N = len(B_data)
            m_t = B_data[:-1] + theta * (mu - B_data[:-1]) * dt
            v_t = sigma**2 * dt
            return -0.5 * np.sum(np.log(2 * np.pi * v_t) + ((B_data[1:] - m_t)**2) / v_t)

        initial_params = [np.mean(B_data), 0.1, np.std(B_data)]
        bounds = [(-np.inf, np.inf), (0, np.inf), (0, np.inf)]
        result = minimize(lambda params: -log_likelihood(params), initial_params, bounds=bounds)
        if result.success:
            return result.x
        else:
            raise RuntimeError("Optimization failed during calibration.")

class LagAdjustedOrnsteinUhlenbeckProcess(BaseOUProcess):
    def __init__(self, mu, theta, sigma, alpha, B0, I, tau, T, n_steps, n_simulations):
        super().__init__(T, n_steps, n_simulations)
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.alpha = alpha
        self.B0 = B0
        self.I = I
        self.tau = tau

    def simulate(self):
        """
        Simulate the Lag-Adjusted Ornstein-Uhlenbeck process.
        """
        B = np.zeros((self.n_steps + 1, self.n_simulations))
        B[0, :] = self.B0
        if self.tau >= self.n_steps:
            raise ValueError("Lag tau cannot exceed or equal the total number of time steps.")
        for t in range(1, self.n_steps + 1):
            dW = np.random.normal(0, np.sqrt(self.dt), self.n_simulations)
            if t > self.tau:
                delta_lagged_index = self.I[t - 1] - self.I[t - 1 - self.tau]
            else:
                delta_lagged_index = 0
            B[t, :] = (B[t - 1, :] +
                       self.theta * (self.mu - B[t - 1, :]) * self.dt +
                       self.alpha * delta_lagged_index * self.dt +
                       self.sigma * dW)
        self.B = B
        return self.time, self.B

    @staticmethod
    def calibrate(B_data, I_data, tau, dt):
        """
        Calibrate the Lag-Adjusted Ornstein-Uhlenbeck process.
        """
        def log_likelihood(params):
            mu, theta, sigma, alpha = params
            N = len(B_data)
            lagged_I = np.roll(I_data, tau)
            lagged_I[:tau] = I_data[0]
            delta_lagged_I = I_data[:-1] - lagged_I[:-1]
            m_t = B_data[:-1] + theta * (mu - B_data[:-1]) * dt + alpha * delta_lagged_I * dt
            v_t = sigma**2 * dt
            return -0.5 * np.sum(np.log(2 * np.pi * v_t) + ((B_data[1:] - m_t)**2) / v_t)

        initial_params = [np.mean(B_data), 0.1, np.std(B_data), 0.5]
        bounds = [(-np.inf, np.inf), (0, np.inf), (0, np.inf), (-np.inf, np.inf)]
        result = minimize(lambda params: -log_likelihood(params), initial_params, bounds=bounds)
        if result.success:
            return result.x
        else:
            raise RuntimeError("Optimization failed during calibration.")
