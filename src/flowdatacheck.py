# channel_validation.py
# Validation of neural network model on refined grid


import matplotlib.pyplot as plt
import numpy as np
from typing import List, Any, Tuple
import copy
import data_utils as du
import neural_utils as nu

VectorType = list[float]

#-----------------  data loading
def load_grid6_data(num_points=51):
    """Load flow data from grid6 results."""
    with open('./data/grid6_flowresults.txt', mode='rt', encoding='utf8') as f:
        viscosities = []
        perturbations = []
        symmetry_labels = []
        it_vals = []
        ib_vals = []
        print(__name__)
        for line_idx in range(num_points):
            try:
                line_content = f.readline()
                (visc, pert, label, b, t, _) = line_content.split(",", 5)
                if float(pert) < 0.01 and 1/float(visc) > 167:
                    viscosities.append(float(visc))
                    perturbations.append(float(pert))
                    symmetry_labels.append(int(label))
                    ib_vals.append(int(b))
                    it_vals.append(int(t))
            except Exception:
                print('irregular spacing in line#', line_idx)
                pass
    return viscosities, perturbations, symmetry_labels, it_vals, ib_vals

#-----------------  neural network functions
def forward_propagation(beta_weights, gamma_weights, input_vector):
    """Feed input through network, return all layer outputs."""
    layer_outputs = []
    
    # Hidden layer with bias
    augmented_input = input_vector + [1]
    hidden_activations = nu.hidden_layer_output(beta_weights, augmented_input)
    layer_outputs.append(hidden_activations.tolist())
    
    # Output layer
    final_input = np.concatenate((hidden_activations, np.array([1])))
    final_output = nu.softmax_layer_output(gamma_weights, final_input)
    layer_outputs.append(final_output.tolist())
    
    return layer_outputs

def encode_label(y_val: int) -> VectorType:
    """Convert label to one-hot encoding."""
    return [1, 0] if y_val == 1 else [0, 1]

def decode_label(y_vec: VectorType) -> int:
    """Convert one-hot to integer."""
    return 1 if y_vec == [1, 0] else 0

def validate_predictions(net_params, input_data, target_data):
    """Validate predictions and compute statistics."""
    correct_count = 0
    correct_flags = np.zeros(len(target_data))
    tp = fp = tn = fn = 0
    
    for idx in range(len(target_data)):
        predicted = nu.index_of_max(forward_propagation(net_params[0], net_params[1], input_data[idx])[-1])
        actual = nu.index_of_max(target_data[idx])
        
        # Debug print for specific index
        if idx == 17:
            print(forward_propagation(net_params[0], net_params[1], input_data[idx])[-1])
            print(target_data[idx])
        
        label_names = ["out", "in"]
        print(idx, label_names[predicted], label_names[actual])
        
        if predicted == actual:
            correct_count += 1
            correct_flags[idx] = 1
            if predicted == 1:
                tp += 1
            else:
                tn += 1
        else:
            if predicted == 1:
                fp += 1
            else:
                fn += 1
                
    print(correct_count, "/", len(target_data))
    perf_stats = [tp, fp, tn, fn]
    return correct_flags, perf_stats

#--------------------- main execution
if __name__ == "__main__":
    # Load trained network parameters
    with open('paramsT.npy', 'rb') as f:
        beta_weights = np.load(f)
        gamma_weights = np.load(f)
        col_means = np.load(f)
        col_stds = np.load(f)
    
    network_params = [beta_weights, gamma_weights]
    print('\nflow data validation ...')
    
    # Load validation data
    viscosities, perturbations, labels, it_vals, ib_vals = load_grid6_data()
    
    raw_data = np.array([np.array(viscosities), np.array(perturbations)])
    
    # Standardize data using training parameters
    scaled_data = du.apply_scaling(raw_data.T, col_means, col_stds)
    scaled_data = scaled_data.tolist()
    
    # Encode targets
    encoded_targets = [encode_label(y) for y in labels]
    
    # Validate
    correct_flags, perf_stats = validate_predictions(network_params, scaled_data, encoded_targets)
    
    # Reload data for plotting
    viscosities, perturbations, labels, it_vals, ib_vals = load_grid6_data()
    raw_data = np.array([np.array(viscosities), np.array(perturbations)])
    raw_data = raw_data.T
    
    fig, (ax1, ax4) = plt.subplots(1, 2, figsize=(8.6, 3.8))
    ax1.scatter(1/raw_data[:, 0], np.log10(raw_data[:, 1]), c=labels, cmap='cool_r')
    ax1.set_xlabel("Reynolds number")
    ax1.set_xlim([200, 240])
    ax1.set_ylabel("logarithm of inlet perturbation")
    ax1.set_title("validation data classification")
    
    ax4.scatter(1/raw_data[:, 0], np.log10(raw_data[:, 1]), c=correct_flags, cmap='winter')
    ax4.set_title("correct model predictions")
    ax4.set_xlim([200, 240])
    plt.show()
    
    tp_count, fp_count, tn_count, fn_count = perf_stats
    
    precision_val = tp_count / (tp_count + fp_count)
    recall_val = tp_count / (tp_count + fn_count)
    f1_val = 2 * precision_val * recall_val / (precision_val + recall_val)
    
    print("classification accuracy statistics ...")
    print(f"precision {precision_val:7.4f} | recall {recall_val:7.4f} | F1 score {f1_val:7.4f}")