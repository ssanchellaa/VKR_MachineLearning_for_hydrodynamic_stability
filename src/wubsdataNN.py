# wubs_training.py
# Binary classification for Elder/Wubs problem using shallow neural network
# Original author: David Silvester (c) 23/12/2024
# Modified for educational purposes

import matplotlib.pyplot as plt
import numpy as np
import data_utils as du
import neural_utils as nu
from typing import List, Any, Tuple

VectorType = list[float]

def load_wubs_data(num_lines=111):
    """Load Wubs problem simulation data."""
    with open('./data/grid32x_testresults.txt', mode='rt', encoding='utf8') as f:
        rayleigh_vals = []
        prandtl_vals = []
        bifurcation_labels = []
        mean_ke = []
        diff_ke = []
        
        print(__name__)
        for line_idx in range(num_lines):
            try:
                line_content = f.readline()
                (ra_val, pr_val, label, mke_val, dke_val, _) = line_content.split(",", 5)
                if float(pr_val) > 1 and float(ra_val) > 1200:
                    rayleigh_vals.append(float(ra_val))
                    prandtl_vals.append(float(pr_val))
                    mean_ke.append(float(mke_val))
                    diff_ke.append(float(dke_val))
                    bifurcation_labels.append(int(label))
            except Exception:
                print('irregular spacing in line#', line_idx)
                pass
    
    # Refine classification based on kinetic energy difference
    diff_array = np.array(diff_ke)
    refined_labels = (diff_array > 0.0006) * 1
    refined_labels = refined_labels.tolist()
    
    return rayleigh_vals, prandtl_vals, bifurcation_labels, mean_ke, diff_ke

def initialize_network():
    """Initialize network with Xavier initialization."""
    np.random.seed(80724)
    input_dim = 2
    hidden_neurons = 32
    output_dim = 2
    
    var_beta = 2/(input_dim + hidden_neurons)
    
    beta_weights = np.random.normal(0, var_beta, (hidden_neurons, input_dim + 1))
    gamma_weights = np.random.uniform(-1, 1, (output_dim, hidden_neurons + 1))
    
    return [beta_weights, gamma_weights]

def encode_label(y_val: int) -> VectorType:
    """Convert to one-hot encoding."""
    return [1, 0] if y_val == 1 else [0, 1]

def decode_label(y_vec: VectorType) -> int:
    """Convert from one-hot encoding."""
    return 1 if y_vec == [1, 0] else 0

def evaluate_performance(net_params, input_data, target_data):
    """Evaluate classification performance."""
    correct_count = 0
    correct_flags = np.zeros(len(target_data))
    tp = fp = tn = fn = 0
    
    for idx in range(len(target_data)):
        predicted = nu.index_of_max(nu.forward_pass(net_params[0], net_params[1], input_data[idx])[-1])
        actual = nu.index_of_max(target_data[idx])
        label_map = ["no", "yes"]
        print(idx, label_map[predicted], label_map[actual])
        
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
from time import time

if __name__ == "__main__":
    print('\nTraining Wubs data neural network using cross-entropy loss ...')
    
    # Load raw data
    rayleigh_vals, prandtl_vals, labels, mean_ke, diff_ke = load_wubs_data()
    
    print('\nWubs problem data results ...')
    print(f'{len(labels)} experiments with {sum(labels)} nonstationary results\n')
    
    raw_data = np.array([np.array(rayleigh_vals), np.array(prandtl_vals)])
    
    # Standardize data
    col_means, col_stds = du.get_scaling_params(raw_data.T)
    scaled_data = du.standardize(raw_data.T)
    print(f"mean Ra is {col_means[0]:9.6f} | mean Pr is {col_means[1]:9.6f}\n")
    
    scaled_data = scaled_data.tolist()
    
    # Encode targets
    encoded_targets = [encode_label(y) for y in labels]
    
    # Initialize and train network
    initial_params = initialize_network()
    num_iterations = 10000
    learning_rate = 0.15
    
    start_time = time()
    loss_history, trained_params, final_grad = nu.train_logistic_network(
        initial_params, scaled_data, encoded_targets, num_iterations, learning_rate)
    elapsed_time = time() - start_time
    
    print("cross-entropy minimisation")
    print("final gradient vector is ")
    print(final_grad)
    print(f"{num_iterations + 1} iterations, learning rate is {learning_rate:7.4f}")
    print(f"Elapsed time is {elapsed_time:6.3f} seconds")
    
    # Plot convergence
    iteration_indices = list(range(num_iterations + 1))
    fit_start = int(num_iterations / 4)
    
    plt.ion()
    fig, (ax1, ax4) = plt.subplots(1, 2, figsize=(8.6, 3.8))
    ax1.plot(loss_history, '-b')
    ax1.set_xlabel("iterations")
    ax1.set_ylabel("loss functional")
    ax1.set_title("complete training history")
    ax4.plot(iteration_indices[fit_start + 1:], loss_history[fit_start:], '-b')
    ax4.set_xlabel("iterations")
    ax4.set_ylabel("loss functional")
    ax4.set_title("final training history ")
    plt.show()
    plt.savefig('wubs_training.png')
    
    # Plot training results
    correct_flags, perf_stats = evaluate_performance(trained_params, scaled_data, encoded_targets)
    
    # Reload data for plotting
    rayleigh_vals, prandtl_vals, labels, mean_ke, diff_ke = load_wubs_data()
    raw_data = np.array([np.array(rayleigh_vals), np.array(prandtl_vals)])
    raw_data = raw_data.T
    
    scaled_arr = np.array(scaled_data)
    original_data = du.reverse_scaling(scaled_arr, col_means, col_stds)
    decoded_labels = [decode_label(y) for y in encoded_targets]
    
    fig, (ax1, ax4) = plt.subplots(1, 2, figsize=(8.6, 3.8))
    ax1.scatter(raw_data[:, 0], raw_data[:, 1], 8,
                c=labels, cmap='cool_r', edgecolor='none')
    ax1.set_xlabel("Rayleigh number")
    ax1.set_ylabel("Prandtl number")
    ax1.set_title("colour shows flow stability")
    
    ax4.scatter(raw_data[:, 0], raw_data[:, 1], 8,
                c=correct_flags, cmap='bwr', edgecolor='none')
    ax4.set_title("correct model predictions")
    ax4.set_xlabel("Rayleigh number")
    plt.show()
    plt.savefig('wubs_scatter.png')
    
    tp_count, fp_count, tn_count, fn_count = perf_stats
    
    precision_val = du.compute_precision(tp_count, fp_count, tn_count, fn_count)
    recall_val = du.compute_recall(tp_count, fp_count, tn_count, fn_count)
    f1_val = du.compute_f1(tp_count, fp_count, tn_count, fn_count)
    
    print("classification accuracy statistics ...")
    print(f"precision {precision_val:7.4f} | recall {recall_val:7.4f} | F1 score {f1_val:7.4f}")
    
    assert precision_val >= 0.9
    assert recall_val >= 0.95