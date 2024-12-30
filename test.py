'''
Library to evaluate different models
@Author: T.T. Ouzounellis Kavlakonis
@Date: 29Dec24
'''

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import mean_absolute_error, mean_squared_error

class ModelFitEvaluator:
    def __init__(self, observed, simulated):
        """
        Initialize the evaluator with observed and simulated values.

        Parameters:
        - observed: Array of observed data.
        - simulated: Array of simulated data (from the model).
        """
        self.observed = np.asarray(observed)
        self.simulated = np.asarray(simulated)
        if len(self.observed) != len(self.simulated):
            raise ValueError("Observed and simulated data must have the same length.")

    def compute_aic(self):
        """
        Compute the Akaike Information Criterion (AIC).

        Returns:
        - AIC value.
        """
        residuals = self.observed - self.simulated
        n = len(self.observed)
        residual_variance = np.var(residuals)
        log_likelihood = -0.5 * n * (np.log(2 * np.pi * residual_variance) + 1)
        k = 1  # Assuming one parameter for simplicity; adjust based on your model
        aic = -2 * log_likelihood + 2 * k
        return aic

    def compute_bic(self):
        """
        Compute the Bayesian Information Criterion (BIC).

        Returns:
        - BIC value.
        """
        residuals = self.observed - self.simulated
        n = len(self.observed)
        residual_variance = np.var(residuals)
        log_likelihood = -0.5 * n * (np.log(2 * np.pi * residual_variance) + 1)
        k = 1  # Assuming one parameter for simplicity; adjust based on your model
        bic = -2 * log_likelihood + k * np.log(n)
        return bic

    def compute_mae(self):
        """
        Compute the Mean Absolute Error (MAE).

        Returns:
        - MAE value.
        """
        return mean_absolute_error(self.observed, self.simulated)

    def compute_rmse(self):
        """
        Compute the Root Mean Squared Error (RMSE).

        Returns:
        - RMSE value.
        """
        return np.sqrt(mean_squared_error(self.observed, self.simulated))

    def ks_test(self):
        """
        Perform the Kolmogorov-Smirnov test to compare the distribution of residuals to a normal distribution.

        Returns:
        - KS test statistic and p-value.
        """
        residuals = self.observed - self.simulated
        standardized_residuals = (residuals - np.mean(residuals)) / np.std(residuals)
        normal_sample = np.random.normal(0, 1, len(standardized_residuals))
        ks_stat, p_value = ks_2samp(standardized_residuals, normal_sample)
        return ks_stat, p_value

    def summary(self):
        """
        Print a summary of all metrics.
        """
        print("Model Fit Metrics:")

        # AIC
        aic = self.compute_aic()
        print(f"AIC: {aic:.4f}")

        # BIC
        bic = self.compute_bic()
        print(f"BIC: {bic:.4f}")

        # MAE
        mae = self.compute_mae()
        print(f"MAE: {mae:.4f}")

        # RMSE
        rmse = self.compute_rmse()
        print(f"RMSE: {rmse:.4f}")

        # KS Test
        ks_stat, p_value = self.ks_test()
        print(f"KS Test Statistic: {ks_stat:.4f}, p-value: {p_value:.4f}")

class ParameterStabilityTester:
    def __init__(self, calibration_function, data, window_size, step_size, **kwargs):
        """
        Initialize the parameter stability tester.

        Parameters:
        - calibration_function: Function to calibrate parameters; should return a list or array of parameters.
        - data: The full dataset used for rolling calibration.
        - window_size: Number of observations in each rolling window.
        - step_size: Number of observations to step forward for each window.
        - kwargs: Additional arguments to pass to the calibration function.
        """
        self.calibration_function = calibration_function
        self.data = data
        self.window_size = window_size
        self.step_size = step_size
        self.kwargs = kwargs

    def test_stability(self):
        """
        Perform rolling calibration to test parameter stability.

        Returns:
        - A numpy array where each row contains calibrated parameters for a window.
        """
        num_windows = (len(self.data) - self.window_size) // self.step_size + 1
        parameters = []

        for i in range(num_windows):
            start_idx = i * self.step_size
            end_idx = start_idx + self.window_size
            window_data = self.data[start_idx:end_idx]

            calibrated_params = self.calibration_function(window_data, **self.kwargs)
            parameters.append(calibrated_params)

        return np.array(parameters)

    def summary(self, parameter_names):
        """
        Print a summary of parameter stability.

        Parameters:
        - parameter_names: List of parameter names corresponding to the calibrated parameters.
        """
        parameters = self.test_stability()
        print("Parameter Stability Summary:")
        for i, param_name in enumerate(parameter_names):
            param_values = parameters[:, i]
            print(f"{param_name}: Mean = {np.mean(param_values):.4f}, Std Dev = {np.std(param_values):.4f}")

# Example usage
if __name__ == "__main__":
    # Example Model Fit Evaluation
    observed = np.random.normal(0, 1, 100)
    simulated = np.random.normal(0, 1, 100)

    evaluator = ModelFitEvaluator(observed, simulated)
    evaluator.summary()

    # Example Parameter Stability Testing
    def example_calibration(data, **kwargs):
        return [np.mean(data), np.std(data)]  # Example: mean and standard deviation

    data = np.random.normal(0, 1, 1000)
    stability_tester = ParameterStabilityTester(example_calibration, data, window_size=100, step_size=50)
    stability_tester.summary(parameter_names=["Mean", "Std Dev"])
