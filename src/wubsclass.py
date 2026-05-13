# wubs_classification.py
# Boundary classification for Elder/Wubs problem using shallow neural network


import matplotlib.pyplot as plt
import numpy as np
import data_utils as du
import neural_utils as nu
import copy

def generate_boundary_data1(col_means, col_stds, num_points=30):
    """Generate first set of boundary probe points."""
    np.random.seed(21256)
    rayleigh_vals = np.random.normal(2.9e9, 2e7, num_points)
    prandtl_vals = np.random.uniform(10, 1100, num_points)
    
    raw_data = np.array([rayleigh_vals, prandtl_vals])
    raw_data_copy = copy.copy(raw_data.T)
    
    scaled_data = du.apply_scaling(raw_data.T, col_means, col_stds)
    return scaled_data, raw_data_copy

def generate_boundary_data2(col_means, col_stds, num_points=30):
    """Generate second set of boundary probe points (zoom region)."""
    np.random.seed(21257)
    rayleigh_vals = np.random.normal(2.99e9, 3e7, num_points)
    prandtl_vals = np.random.normal(160, 50, num_points)
    
    raw_data = np.array([rayleigh_vals, prandtl_vals])
    raw_data_copy = copy.copy(raw_data.T)
    
    scaled_data = du.apply_scaling(raw_data.T, col_means, col_stds)
    return scaled_data, raw_data_copy

def classify_probe_data(network_params, input_data):
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
    
    print('\nbifurcation boundary generation ...')
    
    num_samples = 200000
    
    # Generate and classify first dataset
    # scaled_data1, raw_data1 = generate_boundary_data1(means, stds, num_samples)
    # scaled_data1 = scaled_data1.tolist()
    # labels1 = classify_probe_data(params, scaled_data1)
    
    # Generate and classify second dataset (zoom)
    # scaled_data2, raw_data2 = generate_boundary_data2(means, stds, num_samples)
    # scaled_data2 = scaled_data2.tolist()
    # labels2 = classify_probe_data(params, scaled_data2)
    
    # Plot results
    # fig, (ax1, ax4) = plt.subplots(1, 2, figsize=(8.6, 3.8))
    # ax1.scatter(raw_data1[:, 0], raw_data1[:, 1], 2,
    #             c=labels1, marker='.', cmap='cool_r', edgecolor='none')
    # ax1.set_xlabel("Rayleigh number")
    # ax1.set_ylabel("Prandtl number")
    # ax1.set_title("predicted bifurcation boundary")
    #
    # ax4.scatter(raw_data2[:, 0], raw_data2[:, 1], 2,
    #             c=labels2, marker='.', cmap='cool_r', edgecolor='none')
    # ax4.set_title("zoom of boundary")
    # ax4.set_xlabel("Rayleigh number")
    # plt.show()
    # plt.savefig('wubs_boundary.png')
    
    print("Script requires pre-loaded parameters: params, means, stds")