# channel_training.py
# Binary classification for channel flow using shallow neural network


import matplotlib.pyplot as plt
import numpy as np
import data_utils as du
import neural_utils as nu
from typing import List, Any, Tuple
import copy

VectorType = list[float]

#-----------------  data loading functions
def load_coarse_data(num_lines=106):
    """Load coarse grid flow data."""
    with open('./data/channel_flow_labelsX.txt', mode='rt', encoding='utf8') as f:
        viscosities = []
        perturbations = []
        symmetry_labels = []
        it_vals = []
        ib_vals = []
        print(__name__)
        for line_idx in range(num_lines):
            try:
                line_content = f.readline()
                (reynolds, pert, label, b, t, _) = line_content.split(",", 5)
                if float(pert) < 0.0101 and float(reynolds) > 159.0:
                    viscosities.append(1/float(reynolds))
                    perturbations.append(float(pert))
                    symmetry_labels.append(int(label))
                    ib_vals.append(int(b))
                    it_vals.append(int(t))
            except Exception:
                print('irregular spacing in line#', line_idx)
                pass
    return viscosities, perturbations, symmetry_labels, it_vals, ib_vals

def load_refined_data(num_lines=1477):
    """Load refined grid flow data."""
    with open('./data/channel_flow_labels.txt', mode='rt', encoding='utf8') as f:
        viscosities = []
        perturbations = []
        symmetry_labels = []
        it_vals = []
        ib_vals = []
        print(__name__)
        for line_idx in range(num_lines):
            try:
                line_content = f.readline()
                (reynolds, pert, label, b, t, _) = line_content.split(",", 5)
                if float(pert) < 0.0101 and float(reynolds) > 159.0:
                    viscosities.append(1/float(reynolds))
                    perturbations.append(float(pert))
                    symmetry_labels.append(int(label))
                    ib_vals.append(int(b))
                    it_vals.append(int(t))
            except Exception:
                print('irregular spacing in line#', line_idx)
                pass
    return viscosities, perturbations, symmetry_labels, it_vals, ib_vals

def load_informed_data(num_lines=596):
    """Load informed (grid5) flow data."""
    with open('./data/grid5_flowresults.txt', mode='rt', encoding='utf8') as f:
        viscosities = []
        perturbations = []
        symmetry_labels = []
        it_vals = []
        ib_vals = []
        print(__name__)
        for line_idx in range(num_lines):
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

# ------ network initialization
def initialize_network():
    """Initialize neural network weights using Xavier initialization."""
    np.random.seed(212421)
    input_dim = 2
    hidden_neurons = 32
    output_dim = 2
    
    var_beta = 2/(input_dim + hidden_neurons)
    var_gamma = 2/(output_dim + hidden_neurons)
    
    # Xavier initialization
    beta_weights = np.random.normal(0, var_beta, (hidden_neurons, input_dim + 1))
    gamma_weights = np.random.normal(0, var_beta, (output_dim, hidden_neurons + 1))
    
    # Override output layer with uniform initialization
    gamma_weights = np.random.uniform(-1, 1, (output_dim, hidden_neurons + 1))
    
    return [beta_weights, gamma_weights]

#-----------------  encoding/decoding functions
def encode_label(y_val: int) -> VectorType:
    """Convert integer label to one-hot encoding."""
    return [1, 0] if y_val == 1 else [0, 1]  # [asymmetric, symmetric]

def decode_label(y_vec: VectorType) -> int:
    """Convert one-hot encoding back to integer."""
    return 1 if y_vec == [1, 0] else 0

def evaluate_predictions(net_params, input_data, target_data):
    """Evaluate classification performance on dataset."""
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
    stats = [tp, fp, tn, fn]
    return correct_flags, stats

#--------------------- main execution
from time import time

if __name__ == "__main__":
    split_ratio = 0  # 0.25 for test split
    
    print('\nTraining flow data neural network using cross-entropy loss ...')
    
    # Load appropriate dataset (uncomment desired option)
    # viscosities, perturbations, labels, it_vals, ib_vals = load_informed_data()
    # viscosities, perturbations, labels, it_vals, ib_vals = load_coarse_data()
    viscosities, perturbations, labels, it_vals, ib_vals = load_refined_data()
    
    print('\nflow data results ...')
    print(f'{len(labels)} experiments with {sum(labels)} bifurcated flows')
    
    raw_data = np.array([np.array(viscosities), np.array(perturbations)])
    
    # Standardize data
    col_means, col_stds = du.get_scaling_params(raw_data.T)
    scaled_data = du.standardize(raw_data.T)
    print(f"mean Re is {1/col_means[0]:9.6f} | mean perturbation is {col_means[1]:9.6f}\n")
    
    scaled_data = scaled_data.tolist()
    
    # Encode targets
    encoded_targets = [encode_label(y) for y in labels]
    
    # Split data if needed
    train_features, test_features, train_targets, test_targets = du.train_test_partition(
        scaled_data, encoded_targets, split_ratio)
    
    # Initialize network
    initial_params = initialize_network()
    num_iterations = 5000
    learning_rate = 0.05
    
    start_time = time()
    loss_history, trained_params, final_grad = nu.train_logistic_network(
        initial_params, train_features, train_targets, num_iterations, learning_rate)
    elapsed_time = time() - start_time
    
    print("\ncross-entropy minimisation")
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
    plt.savefig('flow_training_temp.png')
    plt.show()
    
    # Save parameters
    with open('params.npy', 'wb') as f:
        np.save(f, trained_params[0])
        np.save(f, trained_params[1])
        np.save(f, col_means)
        np.save(f, col_stds)
    print("converged NN parameters saved in params.npy")
    
    # Plot training results
    correct_flags, perf_stats = evaluate_predictions(trained_params, train_features, train_targets)
    
    train_features_arr = np.array(train_features)
    original_train = du.reverse_scaling(train_features_arr, col_means, col_stds)
    decoded_train = [decode_label(y) for y in train_targets]
    
    fig, (ax1, ax4) = plt.subplots(1, 2, figsize=(8.6, 3.8))
    ax1.scatter(1/original_train[:, 0], np.log10(original_train[:, 1]), 5,
                c=decoded_train, cmap='cool_r', edgecolor='none')
    ax1.set_xlabel("Reynolds number")
    ax1.set_ylabel("logarithm of inlet perturbation")
    ax1.set_title("colour shows flow classification")
    ax4.scatter(1/original_train[:, 0], np.log10(original_train[:, 1]), 5,
                c=correct_flags, cmap='copper', edgecolor='none')
    ax4.set_title("correct model classifications")
    ax4.set_xlabel("Reynolds number")
    plt.savefig('flow_scatter_temp.png')
    plt.show()
    
    if len(test_features) > 0:
        correct_flags, perf_stats = evaluate_predictions(trained_params, test_features, test_targets)
        
        test_features_arr = np.array(test_features)
        original_test = du.reverse_scaling(test_features_arr, col_means, col_stds)
        decoded_test = [decode_label(y) for y in test_targets]
        
        fig, (ax1, ax4) = plt.subplots(1, 2, figsize=(8.6, 3.8))
        ax1.scatter(1/original_test[:, 0], np.log10(original_test[:, 1]), 6,
                    c=decoded_test, cmap='cool_r', edgecolor='none')
        ax1.set_xlabel("Reynolds number")
        ax1.set_ylabel("logarithm of inlet perturbation")
        ax1.set_title("colour shows flow classification")
        ax4.scatter(1/original_test[:, 0], np.log10(original_test[:, 1]), 6,
                    c=correct_flags, cmap='copper', edgecolor='none')
        ax4.set_title("correct model classifications")
        ax4.set_xlabel("Reynolds number")
        plt.savefig('flow_test_temp.png')
        plt.show()
        
        tp_count, fp_count, tn_count, fn_count = perf_stats
        print("binary confusion matrix  ...")
        print(perf_stats)
        
        precision_val = du.compute_precision(tp_count, fp_count, tn_count, fn_count)
        recall_val = du.compute_recall(tp_count, fp_count, tn_count, fn_count)
        f1_val = du.compute_f1(tp_count, fp_count, tn_count, fn_count)
        
        print("classification accuracy statistics ...")
        print(f"precision {precision_val:7.4f} | recall {recall_val:7.4f} | F1 score {f1_val:7.4f}")
        
        assert precision_val >= 0.9
        assert recall_val >= 0.95
    else:
        print(f'Network trained for all {len(labels)} data points\n')