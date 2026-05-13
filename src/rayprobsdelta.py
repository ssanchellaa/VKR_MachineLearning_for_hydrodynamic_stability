# rayleigh_probability_delta.py
# Detailed probability assessment for Rayleigh-Benard problem

import matplotlib.pyplot as plt
import numpy as np
import data_utils as du
import neural_utils as nu
import copy

def compute_probabilities(input_data, network_params):
    """Compute probability of symmetric solution."""
    prob_list = []
    for idx in range(len(input_data)):
        output_probs = nu.forward_pass(network_params[0], network_params[1], input_data[idx])[-1]
        prob_list.append(output_probs[0])
    return np.array(prob_list)

def generate_boundary_data(col_means, col_stds, num_points=30):
    """Generate random points near bifurcation boundary."""
    np.random.seed(21256)
    rayleigh_vals = np.random.normal(1520, 20, num_points)
    log_pert_vals = np.random.uniform(-16, -2, num_points)
    perturbation_vals = 10 ** log_pert_vals
    
    raw_data = np.array([rayleigh_vals, perturbation_vals])
    raw_data_copy = copy.copy(raw_data.T)
    
    scaled_data = du.apply_scaling(raw_data.T, col_means, col_stds)
    return scaled_data, raw_data_copy

def generate_constant_perturbation_data(pert_exp, col_means, col_stds, num_points=30):
    """Generate data with fixed perturbation amplitude."""
    rayleigh_vals = np.linspace(1490, 1540, num_points)
    perturbation_vals = 10 ** (pert_exp * np.ones(num_points))
    
    raw_data = np.array([rayleigh_vals, perturbation_vals])
    raw_data_copy = copy.copy(raw_data.T)
    
    scaled_data = du.apply_scaling(raw_data.T, col_means, col_stds)
    return scaled_data, raw_data_copy

def classify_data(network_params, input_data):
    """Run classification on probe data."""
    asymmetric_count = 0
    class_labels = np.ones(len(input_data))
    
    for idx in range(len(input_data)):
        predicted = nu.index_of_max(nu.forward_pass(network_params[0], network_params[1], input_data[idx])[-1])
        if predicted == 1:
            asymmetric_count += 1
            class_labels[idx] = 0
            
    print(asymmetric_count, "/", len(input_data))
    return class_labels

if __name__ == "__main__":
    # Note: This script expects pre-loaded parameters: params, means, stds
    
    print('\nperturbation probability assessment ...')
    
    # Generate probability boundary
    num_samples = 200000
    # scaled_data, raw_data = generate_boundary_data(means, stds, num_samples)
    # scaled_data = scaled_data.tolist()
    # prob_data = compute_probabilities(scaled_data, params)
    
    # Cumulative distribution for perturbation = 1e-3.5
    num_cdf = 200
    # scaled_cdf1, raw_cdf1 = generate_constant_perturbation_data(-3.5, means, stds, num_cdf)
    # scaled_cdf1 = scaled_cdf1.tolist()
    # cdf_values1 = compute_probabilities(scaled_cdf1, params)
    
    # Cumulative distribution for perturbation = 1e-15
    # scaled_cdf2, raw_cdf2 = generate_constant_perturbation_data(-15, means, stds, num_cdf)
    # scaled_cdf2 = scaled_cdf2.tolist()
    # cdf_values2 = compute_probabilities(scaled_cdf2, params)
    
    # Multi-panel plot
    # fig = plt.figure(figsize=(8.8, 4.2))
    # ax1 = plt.subplot(222)
    # ax2 = plt.subplot(224)
    # ax3 = plt.subplot(121)
    #
    # ax3.scatter(raw_data[:, 0], np.log10(raw_data[:, 1]), 2,
    #             c=prob_data, marker='.', cmap='twilight', edgecolor='none')
    # ax3.set_title("predicted bifurcation boundary")
    # ax3.set_ylabel("logarithm of hot wall perturbation")
    # ax3.set_xlabel("Rayleigh number")
    #
    # ax1.plot(raw_cdf1[:, 0], cdf_values1)
    # ax1.set_title("perturbation of 1e-3.5")
    # ax1.set_ylabel("probability")
    # ax1.set_xlabel("Rayleigh number")
    #
    # ax2.plot(raw_cdf2[:, 0], cdf_values2)
    # ax2.set_title("perturbation of 1e-15")
    # ax2.set_ylabel("probability")
    # ax2.set_xlabel("Rayleigh number")
    #
    # plt.tight_layout()
    # plt.show()
    # plt.savefig('rayprobsdelta_output.png')
    
    print("Script requires pre-loaded parameters: params, means, stds")