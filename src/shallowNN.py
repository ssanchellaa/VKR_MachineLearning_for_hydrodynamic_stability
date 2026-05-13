# neural_utils.py
# Core neural network utilities for binary classification


import numpy as np
import ipywidgets as widgets
from tqdm.notebook import tqdm
from typing import List, Any, Tuple

VectorType = list[float]

#----------------  activation functions
def index_of_max(arr: list) -> int:
    """Returns index of largest element."""
    return max(range(len(arr)), key=lambda idx: arr[idx])

def stable_softmax(vec: VectorType) -> VectorType:
    """Numerically stable softmax function."""
    exp_vec = np.exp(vec - vec.max())
    norm_factor = np.sum(exp_vec)
    return exp_vec / norm_factor

def softmax_on_layer(layer_matrix):
    """Apply softmax to each row of a matrix."""
    rows, cols = layer_matrix.shape
    exp_mat = np.exp(layer_matrix - layer_matrix.max())
    row_sums = np.sum(exp_mat, axis=1)
    return [exp_mat / np.outer(row_sums, np.ones(cols))]

def softmax_layer_output(weight_matrix: VectorType, activation_input: VectorType) -> VectorType:
    """Compute softmax output given weights and input (includes bias)."""
    return stable_softmax(weight_matrix @ activation_input)

def sigmoid(x: VectorType) -> VectorType:
    """Sigmoid activation function."""
    return 1.0 / (1 + np.exp(-x))

def hidden_layer_output(weights: VectorType, inputs: VectorType) -> VectorType:
    """Compute hidden layer output with sigmoid activation."""
    return sigmoid(weights @ inputs)

#-----------------  feedforward network
def forward_pass(beta_weights: List[VectorType],
                 gamma_weights: List[VectorType],
                 input_vector: VectorType) -> List[VectorType]:
    """
    Forward propagation through two-layer neural network.
    Returns outputs from both hidden and output layers.
    """
    layer_outputs: List[VectorType] = []
    
    # Hidden layer with bias term
    augmented_input = input_vector + [1]
    hidden_activations = hidden_layer_output(beta_weights, augmented_input)
    layer_outputs.append(hidden_activations.tolist())
    
    # Output layer with softmax
    final_input = np.concatenate((hidden_activations, np.array([1])))
    final_output = softmax_layer_output(gamma_weights, final_input)
    layer_outputs.append(final_output.tolist())
    
    return layer_outputs

def compute_gradients(beta_weights: List[VectorType],
                      gamma_weights: List[VectorType],
                      input_vector: VectorType,
                      target_vector: VectorType) -> List[List[VectorType]]:
    """
    Compute gradients of cross-entropy loss with respect to all weights.
    Returns [hidden_grads, output_grads].
    """
    # Forward pass
    hidden_acts, output_acts = forward_pass(beta_weights, gamma_weights, input_vector)
    output_array = np.array(output_acts)
    hidden_array = np.array(hidden_acts)
    
    # Output layer gradients
    output_deltas = output_array - target_vector
    
    # Output weight gradients
    hidden_with_bias = np.concatenate((hidden_array, [1]))
    output_weight_grads = np.outer(output_deltas, hidden_with_bias)
    
    # Hidden layer gradients (sigmoid derivative)
    hidden_deltas = hidden_array * (1 - hidden_array) * (gamma_weights[:, 0:-1].T @ output_deltas)
    
    # Hidden weight gradients
    input_with_bias = input_vector + [1]
    hidden_weight_grads = np.outer(hidden_deltas, input_with_bias)
    
    return [hidden_weight_grads, output_weight_grads]

def loss_gradient(x_val: float, y_val: float,
                  network_params: List[List[VectorType]]) -> List[List[VectorType]]:
    """Compute gradient vector for a single data point."""
    h_grad, o_grad = compute_gradients(network_params[0], network_params[1], x_val, y_val)
    return np.concatenate((h_grad.flatten(), o_grad.flatten()))

def cross_entropy_loss(x_val: float, y_val: VectorType,
                       network_params: List[List[VectorType]]) -> float:
    """Compute cross-entropy loss for a single prediction."""
    _, output = forward_pass(network_params[0], network_params[1], x_val)
    likelihoods = -np.log(np.array(output) + 1e-33) * np.array(y_val)
    return np.sum(likelihoods)

def gradient_update(params_vec: VectorType, grad_vec: VectorType, step: float) -> VectorType:
    """Perform parameter update using gradient descent."""
    assert len(params_vec) == len(grad_vec)
    return params_vec + step * grad_vec

def train_logistic_network(init_params, inputs_data, targets_data,
                           num_iterations, learn_rate):
    """
    Train shallow neural network using stochastic gradient descent.
    Returns training history, final parameters, and final gradient.
    """
    beta_shape = init_params[0].shape
    gamma_shape = init_params[1].shape
    n_beta = beta_shape[0] * beta_shape[1]
    n_gamma = gamma_shape[0] * gamma_shape[1]
    
    loss_history = []
    
    for epoch in tqdm(range(num_iterations)):
        for x_pt, y_pt in zip(inputs_data, targets_data):
            grad_vector = np.array(loss_gradient(x_pt, y_pt, init_params))
            flat_params = np.concatenate((init_params[0].flatten(), init_params[1].flatten()))
            updated_params = gradient_update(flat_params, grad_vector, -learn_rate)
            
            # Reshape back to matrix form
            new_beta = updated_params[0:n_beta].reshape(beta_shape)
            new_gamma = updated_params[n_beta:].reshape(gamma_shape)
            init_params = [new_beta, new_gamma]
        
        # Compute and record loss
        batch_losses = [cross_entropy_loss(x_pt, y_pt, init_params) 
                        for (x_pt, y_pt) in zip(inputs_data, targets_data)]
        total_loss = np.sum(np.array(batch_losses))
        loss_history.append(total_loss)
        print(epoch, total_loss)
    
    return loss_history, init_params, grad_vector