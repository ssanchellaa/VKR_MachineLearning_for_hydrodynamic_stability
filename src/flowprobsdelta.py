# channel_probability.py
# Binary classification probability assessment for channel flow

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import data_utils as du
import neural_utils as nu
import copy

def compute_probabilities(input_data, network_params):
    """Compute probability of symmetric flow for each input."""
    prob_list = []
    for idx in range(len(input_data)):
        output_probs = nu.forward_pass(network_params[0], network_params[1], input_data[idx])[-1]
        prob_list.append(output_probs[0])  # Probability of symmetric flow
    return np.array(prob_list)

def generate_boundary_data(col_means, col_stds, num_points=30):
    """Generate random points near bifurcation boundary."""
    np.random.seed(21256)
    reynolds_vals = np.random.uniform(160, 240, num_points)
    log_pert_vals = np.random.uniform(-5.5, -2, num_points)
    
    visc_vals = 1 / reynolds_vals
    pert_vals = 10 ** log_pert_vals
    
    raw_data = np.array([visc_vals, pert_vals])
    raw_data_copy = copy.copy(raw_data.T)
    
    # Standardize data
    scaled_data = du.apply_scaling(raw_data.T, col_means, col_stds)
    return scaled_data, raw_data_copy

def generate_constant_perturbation_data(pert_exp, col_means, col_stds, num_points=30):
    """Generate data with constant perturbation amplitude."""
    reynolds_vals = np.linspace(205, 220, num_points)
    visc_vals = 1 / reynolds_vals
    
    pert_vals = 10 ** (pert_exp * np.ones(num_points))
    
    raw_data = np.array([visc_vals, pert_vals])
    raw_data_copy = copy.copy(raw_data.T)
    
    # Standardize data
    scaled_data = du.apply_scaling(raw_data.T, col_means, col_stds)
    return scaled_data, raw_data_copy

def classify_network(network_params, input_data):
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
    # Load trained network parameters
    with open('params_refined_lr005.npy', 'rb') as f:
        beta_weights = np.load(f)
        gamma_weights = np.load(f)
        col_means = np.load(f)
        col_stds = np.load(f)
    
    network_params = [beta_weights, gamma_weights]
    print('\nperturbation probability assessment ...')
    
    # Generate probability boundary
    num_samples = 200000
    scaled_data, raw_data = generate_boundary_data(col_means, col_stds, num_samples)
    scaled_data = scaled_data.tolist()
    
    prob_data = compute_probabilities(scaled_data, network_params)
    
    # CDF for perturbation = 1e-3.5
    num_cdf = 200
    scaled_cdf1, raw_cdf1 = generate_constant_perturbation_data(-3.5, col_means, col_stds, num_cdf)
    scaled_cdf1 = scaled_cdf1.tolist()
    cdf_values1 = compute_probabilities(scaled_cdf1, network_params)
    
    # CDF for perturbation = 1e-15
    scaled_cdf2, raw_cdf2 = generate_constant_perturbation_data(-15, col_means, col_stds, num_cdf)
    scaled_cdf2 = scaled_cdf2.tolist()
    cdf_values2 = compute_probabilities(scaled_cdf2, network_params)
    
    # Create multi-panel figure
    fig = plt.figure(figsize=(8.8, 4.2))
    ax_boundary = plt.subplot(121)
    ax_cdf1 = plt.subplot(222)
    ax_cdf2 = plt.subplot(224)
    
    # Boundary plot
    ax_boundary.scatter(1/raw_data[:, 0], np.log10(raw_data[:, 1]), 2,
                        c=prob_data, cmap='twilight', edgecolor='none')
    ax_boundary.set_title("predicted bifurcation boundary")
    ax_boundary.set_ylabel("logarithm of inlet perturbation")
    ax_boundary.set_xlabel("Reynolds number")
    
    # CDF plots
    ax_cdf1.plot(1/raw_cdf1[:, 0], cdf_values1)
    ax_cdf1.set_title("perturbation of 1e-3.5")
    ax_cdf1.set_ylabel("probability")
    ax_cdf1.set_xlabel("Reynolds number")
    
    ax_cdf2.plot(1/raw_cdf2[:, 0], cdf_values2)
    ax_cdf2.set_title("perturbation of 1e-15")
    ax_cdf2.set_ylabel("probability")
    ax_cdf2.set_xlabel("Reynolds number")
    
    plt.tight_layout()
    plt.savefig('flowprobs_refined_lr005.png')
    plt.show()