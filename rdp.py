# We provide privacy analysis in rdp.py
from scipy.optimize import root_scalar, minimize_scalar
import numpy as np
import math

def RDP_DP(alpha, gamma, delta):
    # If an algorithm satisfies (optimal_epsilon, delta)-DP, then it satisfies (alpha, gamma)-RDP.
    # Function to find epsilon given alpha, gamma, and delta
    # Following Theorem 3 in https://arxiv.org/pdf/2008.06529.pdf
    if alpha<=1:
        print("Error: alpha <= 1")
    def objective(epsilon):
        def gamma_alpha(p):
            M_value = (p ** alpha) * ((p - delta) ** (1 - alpha)) + ((1 - p) ** alpha) * ((math.exp(epsilon) - p + delta) ** (1 - alpha))
            gamma = epsilon + (1 / (alpha - 1)) * np.log(M_value)
            return gamma

        res = minimize_scalar(lambda p: gamma_alpha(p), bounds=(delta, 1), method='bounded')
        if not res.success:
            raise ValueError("Optimization did not converge inside gamma calculation.")

        gamma_calc = res.fun
        return gamma_calc - gamma

    # Use root_scalar to find the epsilon that makes objective(epsilon) = 0
    result = root_scalar(objective, bracket=[0, 100], method='brentq')  # Assuming epsilon is in the range [0, 100]

    if not result.converged:
        raise ValueError("Root finding did not converge.")

    optimal_epsilon = result.root

    return optimal_epsilon

def DP_RDP(epsilon, delta, gamma):
    # If an algorithm satisfies (optimal_alpha, gamma)-RDP, then it satisfies (epsilon, delta)-DP.
    # Function to find alpha given epsilon, delta, and gamma
    # Following in Theorem 21 in https://arxiv.org/pdf/1905.09982.pdf
    def objective(alpha):
        epsilon_calc = gamma + np.log((alpha - 1) / alpha) - (np.log(delta) + np.log(alpha)) / (alpha - 1)
        return epsilon_calc - epsilon
    # print("f(1+1e-7):", objective(1+1e-7))
    # print("f(1e100):", objective(1e100))
    result = root_scalar(objective, bracket=[1+1e-7, 100], method='brentq')# Assuming result is in the range [1+1e-7, 100]
    if not result.converged:
        raise ValueError("Root finding did not converge.")

    optimal_alpha = result.root
    return optimal_alpha


def subsample(epsilon, delta, rate):
    # Amplification of privacy by subsampling: If an algorithm satisfies (epsilon_sample, delta_sample)-DP, then the subsampled algorithm with sampling rate "rate" satisfies (epsilon, delta)-DP
    delta_sample = delta/rate
    epsilon_sample = np.log(((np.exp(epsilon) - 1) / rate) + 1)
    return epsilon_sample, delta_sample

if __name__ == "__main__":
    # Desired epsilon and delta
    # epsilon 1, 2, 4, 8
    epsilon = 8
    # delta MIT-D: 1/1561 MIT-G: 1/2953 AGNEWS: 1/120000 DBPEDIA: 1/49999 TREC: 1/5452
    delta = 1/120000
    # T_max MIT-D: 20 MIT-G: 20 AGNEWS: 100 DBPEDIA: 100 TREC: 15
    T_max = 100
    # subsample rate MIT-D: 80/1561 MIT-G: 80/2953 AGNEWS: 20/30000 DBPEDIA: 20/3558 TREC: 80/835
    sample_rate = 20/30000
    # MIT-D: 0.1 MIT-G: 0.1 AGNEWS: 0.01 DBPEDIA:0.1 for 8 (0.05) TREC:0.5
    gamma = 0.01
    # number of DP-testing and DP-avg
    T_update = 3
    budget_radius = 0.05
    budget_test = 0.05/T_update
    budget_avg = 0.9/T_update

    assert budget_radius + T_update*(budget_test+budget_avg) == 1

    alpha_iteration = DP_RDP(epsilon,delta,gamma)
    # Each iteration (T_max in total) should satisfy (alpha_iteration, gamma_iteration)-RDP
    gamma_iteration = gamma/T_max
    epsilon_iteration = RDP_DP(alpha_iteration, gamma_iteration, delta)
    # Amplify by subsampling
    eps_noise, delta_noise = subsample(epsilon_iteration,delta,sample_rate)
    # Uniform Gaussian analysis
    sigma = np.sqrt(2 * np.log(1.25/delta_noise))/eps_noise
    print(f"sigma for Uniform Gaussian: {sigma}")
    # DP-radius, DP-testing, DP-avg should satisfy (alpha_noise, gamma)-RDP
    alpha_noise = DP_RDP(eps_noise,delta_noise,gamma)
    print(f"DP-radius + DP-testing + DP-avg should satisfy: ({alpha_noise}, {gamma})-RDP.")
    # DP-radius
    print(f"DP-radius satisfies: ({alpha_noise}, {gamma*budget_radius})-RDP.")
    # DP-testing
    print(f"Each DP-testing satisfies: ({alpha_noise}, {gamma*budget_test})-RDP.")
    epsilon_gaussian = RDP_DP(alpha_noise, gamma*budget_test, delta)
    sigma = np.sqrt(2 * np.log(1.25/delta))/epsilon_gaussian
    print(f"sigma for DP-testing: {sigma}")
    # DP-avg
    print(f"DP-avg satisfies: ({alpha_noise}, {gamma*budget_avg})-RDP.")
    epsilon_gaussian = RDP_DP(alpha_noise, gamma*budget_avg, delta)
    sigma = np.sqrt(2 * np.log(1.25/delta))/epsilon_gaussian
    print(f"sigma for DP-avg: {sigma}")




