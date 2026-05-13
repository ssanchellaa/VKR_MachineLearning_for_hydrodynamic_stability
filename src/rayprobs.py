# rayleigh_probability.py
# Probability assessment for Rayleigh-Benard bifurcation boundary

import matplotlib.pyplot as plt
import numpy as np
import data_utils as du
import neural_utils as nu
import copy

def compute_probabilities(input_data, network_params):
    """Compute probability of base (symmetric) solution."""
    prob_list = []
    for idx in range(len(input_data)):
        output_probs = nu.forward_pass(network_params[0], network_params[1], input_data[idx])[-1]
        prob_list.append(output_probs[0])
    return np.array(prob_list)

def generate_probe_data(col_means, col_stds, num_points=30):
    """Generate random probe points near boundary."""
    np.random.seed(21256)
    rayleigh_vals = np.random.normal(1520, 20, num_points)
    log_pert_vals = np.random.uniform(-16, -2, num_points)
    perturbation_vals = 10 ** log_pert_vals
    
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
    # Note: This script expects 'params', 'means', 'stds', 'yesdata' to be defined
    # before execution. These would typically come from a trained model.
    
    print('\nbifurcation probability assessment ...')
    num_samples = 200000
    
    # Assuming means and stds are defined (would come from loaded model)
    # scaled_data, raw_data = generate_probe_data(means, stds, num_samples)
    # scaled_data = scaled_data.tolist()
    # prob_data = compute_probabilities(scaled_data, params)
    
    # fig, (ax1, ax4) = plt.subplots(1, 2, figsize=(8.6, 3.8))
    # ax1.scatter(raw_data[:, 0], np.log10(raw_data[:, 1]), 5,
    #             c=yesdata, cmap='cool_r', edgecolor='none')
    # ax1.set_xlabel("Rayleigh number")
    # ax1.set_ylabel("logarithm of hot wall perturbation")
    # ax1.set_title("training data classifications")
    # ax4.scatter(raw_data[:, 0], np.log10(raw_data[:, 1]), 2,
    #             c=prob_data, marker='.', cmap='twilight', edgecolor='none')
    # ax4.set_title("predicted bifurcation boundary")
    # ax4.set_xlabel("Rayleigh number")
    # plt.show()
    
    print("Script requires pre-loaded parameters: params, means, stds, yesdata")