import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score, ConfusionMatrixDisplay
import logging
import time
import json
import pickle
import os
import csv
import itertools
from sklearn.datasets import load_breast_cancer
import unittest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global configurations
RANDOM_STATE = 42

class Optimizer:
    """
    Base class for optimizers. Defines the fundamental parameters and methods
    that should be implemented by specific optimizers like SGD, Adam, and RMSprop.
    """
    def __init__(self, learning_rate=0.01, momentum=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initializes the optimizer with the given hyperparameters.

        Parameters:
            learning_rate (float): Learning rate of the optimizer.
            momentum (float): Momentum factor for optimizers that use momentum.
            beta1 (float): First moment vector for Adam.
            beta2 (float): Second moment vector for Adam.
            epsilon (float): Small number to prevent division by zero.
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_velocities = None
        self.bias_velocities = None
        self.m_weights = None  # Adam: First moment estimates for weights
        self.v_weights = None  # Adam: Second moment estimates for weights
        self.m_biases = None   # Adam: First moment estimates for biases
        self.v_biases = None   # Adam: Second moment estimates for biases

    def initialize(self, weights, biases):
        """
        Initializes the velocity and moment estimates based on the weight and bias arrays.

        Parameters:
            weights (list of np.ndarray): List of weight matrices of the network.
            biases (list of np.ndarray): List of bias vectors of the network.
        """
        self.weight_velocities = [np.zeros_like(w, dtype=np.float32) for w in weights]
        self.bias_velocities = [np.zeros_like(b, dtype=np.float32) for b in biases]
        self.m_weights = [np.zeros_like(w, dtype=np.float32) for w in weights]
        self.v_weights = [np.zeros_like(w, dtype=np.float32) for w in weights]
        self.m_biases = [np.zeros_like(b, dtype=np.float32) for b in biases]
        self.v_biases = [np.zeros_like(b, dtype=np.float32) for b in biases]

    def update(self, weights, biases, weight_gradients, bias_gradients, t=None):
        """
        Updates the weights and biases based on gradients.
        Must be implemented by subclasses.

        Parameters:
            weights (list of np.ndarray): Current weights of the network.
            biases (list of np.ndarray): Current biases of the network.
            weight_gradients (list of np.ndarray): Gradients of the weights.
            bias_gradients (list of np.ndarray): Gradients of the biases.
            t (int, optional): Iteration step, relevant for Adam.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses should implement this!")

class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer with momentum.
    """
    def update(self, weights, biases, weight_gradients, bias_gradients, t=None):
        """
        Updates the weights and biases using SGD with momentum.

        Parameters:
            weights (list of np.ndarray): Current weights of the network.
            biases (list of np.ndarray): Current biases of the network.
            weight_gradients (list of np.ndarray): Gradients of the weights.
            bias_gradients (list of np.ndarray): Gradients of the biases.
            t (int, optional): Iteration step, not used in SGD.

        Returns:
            tuple: Updated weights and biases.
        """
        updated_weights = []
        updated_biases = []
        for i in range(len(weights)):
            # Update weight velocities
            self.weight_velocities[i] = self.momentum * self.weight_velocities[i] - self.learning_rate * weight_gradients[i]
            # Update weights
            updated_weights.append(weights[i] + self.weight_velocities[i])
            # Update bias velocities
            self.bias_velocities[i] = self.momentum * self.bias_velocities[i] - self.learning_rate * bias_gradients[i]
            # Update biases
            updated_biases.append(biases[i] + self.bias_velocities[i])
        return updated_weights, updated_biases

class Adam(Optimizer):
    """
    Adam optimizer, an adaptive learning rate optimization algorithm.
    """
    def update(self, weights, biases, weight_gradients, bias_gradients, t=None):
        """
        Updates the weights and biases using the Adam optimization algorithm.

        Parameters:
            weights (list of np.ndarray): Current weights of the network.
            biases (list of np.ndarray): Current biases of the network.
            weight_gradients (list of np.ndarray): Gradients of the weights.
            bias_gradients (list of np.ndarray): Gradients of the biases.
            t (int, optional): Iteration step for bias correction.

        Returns:
            tuple: Updated weights and biases.
        """
        updated_weights = []
        updated_biases = []
        for i in range(len(weights)):
            # Update first and second moments for weights
            self.m_weights[i] = self.beta1 * self.m_weights[i] + (1 - self.beta1) * weight_gradients[i]
            self.v_weights[i] = self.beta2 * self.v_weights[i] + (1 - self.beta2) * weight_gradients[i]**2
            # Bias-corrected moments
            m_hat = self.m_weights[i] / (1 - self.beta1**t)
            v_hat = self.v_weights[i] / (1 - self.beta2**t)
            # Update weights
            updated_weights.append(weights[i] - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon))
            
            # Update first and second moments for biases
            self.m_biases[i] = self.beta1 * self.m_biases[i] + (1 - self.beta1) * bias_gradients[i]
            self.v_biases[i] = self.beta2 * self.v_biases[i] + (1 - self.beta2) * bias_gradients[i]**2
            # Bias-corrected moments for biases
            m_hat_b = self.m_biases[i] / (1 - self.beta1**t)
            v_hat_b = self.v_biases[i] / (1 - self.beta2**t)
            # Update biases
            updated_biases.append(biases[i] - self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon))
        return updated_weights, updated_biases

class RMSprop(Optimizer):
    """
    RMSprop optimizer, another adaptive learning rate optimization algorithm.
    """
    def update(self, weights, biases, weight_gradients, bias_gradients, t=None):
        """
        Updates the weights and biases using the RMSprop optimization algorithm.

        Parameters:
            weights (list of np.ndarray): Current weights of the network.
            biases (list of np.ndarray): Current biases of the network.
            weight_gradients (list of np.ndarray): Gradients of the weights.
            bias_gradients (list of np.ndarray): Gradients of the biases.
            t (int, optional): Iteration step, not used in RMSprop.

        Returns:
            tuple: Updated weights and biases.
        """
        updated_weights = []
        updated_biases = []
        for i in range(len(weights)):
            # Update second moments for weights
            self.v_weights[i] = self.beta2 * self.v_weights[i] + (1 - self.beta2) * weight_gradients[i]**2
            # Update weights
            updated_weights.append(weights[i] - self.learning_rate * weight_gradients[i] / (np.sqrt(self.v_weights[i]) + self.epsilon))
            
            # Update second moments for biases
            self.v_biases[i] = self.beta2 * self.v_biases[i] + (1 - self.beta2) * bias_gradients[i]**2
            # Update biases
            updated_biases.append(biases[i] - self.learning_rate * bias_gradients[i] / (np.sqrt(self.v_biases[i]) + self.epsilon))
        return updated_weights, updated_biases

def load_data_from_csv(filepath, target_column='label', test_size=0.2, random_state=42, scale_features=True):
    """
    Loads data from a CSV file, processes the target column, and splits the data into training and testing sets.

    Parameters:
        filepath (str): Path to the CSV file.
        target_column (str): Name of the target column.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied to the data before applying the split.
        scale_features (bool): Whether to normalize the features.

    Returns:
        tuple: X_train, X_test, y_train, y_test after splitting.
    """
    data = pd.read_csv(filepath)
    target = data[target_column]
    features = data.drop(columns=[target_column])

    X = features.values.astype(np.float32)
    y = target.values

    # Process target column
    if len(np.unique(y)) > 2:  # Multi-class classification
        y = LabelEncoder().fit_transform(y)
        y = np.eye(len(np.unique(y)))[y]  # One-Hot Encoding
    else:  # Binary classification
        y = y.astype(np.float32).reshape(-1, 1)

    if scale_features:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

    # Handle missing values
    if np.isnan(X).any():
        X = np.nan_to_num(X)
    if np.isnan(y).any():
        y = np.nan_to_num(y)

    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def test_label_shapes():
    """
    Tests whether the labels have the expected shape.
    """
    X, y = create_dummy_data(num_samples=100, num_features=10, num_classes=2)
    assert y.shape[1] == 1, f"Expected shape (100, 1), but got: {y.shape}"

def save_dataset_to_csv(X, y, filepath):
    """
    Saves features and labels to a CSV file.

    Parameters:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.
        filepath (str): Path to the output file.
    """
    # If y is One-Hot Encoded, convert it to one-dimensional labels
    if len(y.shape) > 1 and y.shape[1] > 1:
        y = np.argmax(y, axis=1)  # Take the index of the maximum value for each row

    data = pd.DataFrame(X)
    data['label'] = y
    data.to_csv(filepath, index=False)
    logger.info(f"Dataset saved to {filepath}")

def check_for_nan(data):
    """
    Checks whether there are any missing values in the data.

    Parameters:
        data (np.ndarray or list): Data structure to check.

    Raises:
        ValueError: If missing values (NaN) are found.
        TypeError: If the data type is not supported.
    """
    if isinstance(data, np.ndarray):
        if np.isnan(data).any():
            raise ValueError("Input data contains NaN values.")
    elif isinstance(data, list):
        for row in data:
            if isinstance(row, list):
                if any(np.isnan(x) for x in row):
                    raise ValueError("Input data contains NaN values.")
            elif np.isnan(row):
                raise ValueError("Input data contains NaN values.")
    else:
        raise TypeError("Data type not supported.")

def handle_overflow(x):
    """
    Ensures that no overflow occurs during exponential operations.

    Parameters:
        x (float): Input value.

    Returns:
        float: Result of the exponential operation or Inf/0.
    """
    try:
        return np.exp(x)
    except OverflowError:
        return np.inf if x > 0 else 0.0

def sigmoid(x):
    """
    Sigmoid activation function with overflow protection.

    Parameters:
        x (np.ndarray): Input values.

    Returns:
        np.ndarray: Sigmoid output.
    """
    clipped_x = np.clip(x, -500, 500)  # Limit input values
    return 1 / (1 + np.exp(-clipped_x))

def relu(x):
    """
    ReLU activation function.

    Parameters:
        x (np.ndarray): Input values.

    Returns:
        np.ndarray: ReLU output.
    """
    return np.maximum(0, x)

def softmax(x):
    """
    Softmax activation function.

    Parameters:
        x (np.ndarray): Input values.

    Returns:
        np.ndarray: Softmax output.
    """
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def sigmoid_derivative(x):
    """
    Derivative of the Sigmoid function.

    Parameters:
        x (np.ndarray): Input values.

    Returns:
        np.ndarray: Derivative of Sigmoid.
    """
    s = sigmoid(x)
    return s * (1 - s)

def relu_derivative(x):
    """
    Derivative of the ReLU function.

    Parameters:
        x (np.ndarray): Input values.

    Returns:
        np.ndarray: Derivative of ReLU.
    """
    return np.where(x > 0, 1, 0).astype(float)

def softmax_derivative(x):
    """
    Derivative of the Softmax function.

    Parameters:
        x (np.ndarray): Input values.

    Returns:
        np.ndarray: Jacobian matrix of the Softmax function.
    """
    s = softmax(x)
    jacobian_matrix = np.diag(s.flatten()) - np.outer(s, s)
    return jacobian_matrix

def plot_confusion_matrix(y_true, y_pred, labels):
    """
    Creates and displays a confusion matrix.

    Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        labels (list): List of label names.
    """
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=labels)
    plt.show()

def early_stopping(history, patience=5):
    """
    Checks whether training should be stopped early based on loss stagnation.

    Parameters:
        history (dict): History of training metrics.
        patience (int): Number of epochs with no improvement after which training will be stopped.

    Returns:
        bool: True if training should be stopped early, else False.
    """
    if len(history['loss']) > patience and all(history['loss'][-i] >= history['loss'][-i-1] for i in range(1, patience+1)):
        return True
    return False

class Network:
    """
    Class representing a simple neural network with training and testing capabilities,
    as well as serialization methods for JSON and Pickle.
    """
    def __init__(self, num_nodes, V_max, R_harmony, activation_function='sigmoid', learning_rate=0.1, momentum=0.9, output_activation='sigmoid', optimizer_method='SGD'):
        """
        Initializes the network with given hyperparameters and initializes weights and biases.

        Parameters:
            num_nodes (list of int): Number of neurons in each layer (including input and output).
            V_max (float): Maximum voltage or activation threshold (specific to the model).
            R_harmony (float): Harmony factor (specific to the model).
            activation_function (str): Activation function for hidden layers ('sigmoid' or 'relu').
            learning_rate (float): Learning rate for the optimizer.
            momentum (float): Momentum factor for the optimizer.
            output_activation (str): Activation function for the output layer ('sigmoid' or 'softmax').
            optimizer_method (str): Optimizer method ('SGD', 'Adam', or 'RMSprop').

        Raises:
            ValueError: If an unsupported activation function is chosen.
        """
        self.num_nodes = num_nodes
        self.V_max = V_max
        self.R_harmony = R_harmony
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.output_activation = output_activation
        self.optimizer_method = optimizer_method

        # Initialize weights and biases
        self.weights = [np.random.uniform(-1, 1, size=(num_nodes[i], num_nodes[i+1])).astype(np.float32) for i in range(len(num_nodes) - 1)]
        self.biases = [np.zeros(nodes, dtype=np.float32) for nodes in num_nodes[1:]]

        # Select and initialize the optimizer
        self.optimizer = self.get_optimizer(optimizer_method, learning_rate, momentum)
        self.optimizer.initialize(self.weights, self.biases)

        # Select activation function and its derivative
        if activation_function == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation_function == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        else:
            raise ValueError("Unsupported activation function.")

        # Select output layer activation function and its derivative
        if output_activation == 'sigmoid':
            self.output_activation_func = sigmoid
            self.output_activation_derivative = sigmoid_derivative
        elif output_activation == 'softmax':
            self.output_activation_func = softmax
            self.output_activation_derivative = softmax_derivative
        else:
            raise ValueError("Unsupported output activation function.")

        self.trial_results = []       # Stores results of different trials
        self.metrics_history = {}     # History of metrics

    def get_optimizer(self, method, learning_rate, momentum):
        """
        Returns the corresponding optimizer based on the method.

        Parameters:
            method (str): Name of the optimizer ('SGD', 'Adam', 'RMSprop').
            learning_rate (float): Learning rate.
            momentum (float): Momentum factor.

        Returns:
            Optimizer: Instance of the chosen optimizer.

        Raises:
            ValueError: If an unsupported optimizer method is chosen.
        """
        if method == 'SGD':
            return SGD(learning_rate=learning_rate, momentum=momentum)
        elif method == 'Adam':
            return Adam(learning_rate=learning_rate)
        elif method == 'RMSprop':
            return RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError("Unsupported optimizer.")

    def validate_dimensions(self, x, y):
        """
        Checks whether the dimensions of x and y match the network parameters.

        Parameters:
            x (np.ndarray): Input data.
            y (np.ndarray): Target values.

        Raises:
            ValueError: If dimensions do not match.
        """
        if x.shape[1] != self.num_nodes[0]:
            raise ValueError(f"Input dimension {x.shape[1]} does not match expected {self.num_nodes[0]}.")
        if y.ndim == 1 and self.num_nodes[-1] > 1:
            raise ValueError(f"Target shape {y.shape} does not match number of output neurons {self.num_nodes[-1]}.")
        if y.ndim > 1 and y.shape[1] != self.num_nodes[-1]:
            raise ValueError(f"Target dimension {y.shape[1]} does not match number of output neurons {self.num_nodes[-1]}.")

    def feedforward(self, input_data):
        """
        Performs a feedforward pass through the network.

        Parameters:
            input_data (np.ndarray): Input data for the network.

        Returns:
            tuple: List of activations per layer and list of z-values (net inputs) per layer.
        """
        input_data = input_data.reshape(1, -1)  # Ensure input has the correct shape
        activations = [input_data]
        z_values = []

        for i in range(len(self.weights)):
            # Calculate net input (z) for the current layer
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            z_values.append(z)

            # Apply activation function
            if i == len(self.weights) - 1:  # Last layer
                activation = self.output_activation_func(z)
            else:
                activation = self.activation(z)
            activations.append(activation)

        return activations, z_values

    def backpropagation(self, activations, z_values, y):
        """
        Performs backpropagation to compute gradients of weights and biases.

        Parameters:
            activations (list of np.ndarray): Activations per layer.
            z_values (list of np.ndarray): Net inputs per layer.
            y (np.ndarray): Target values.

        Returns:
            tuple: Gradients of weights and biases.
        """
        # Calculate error in the output layer
        delta = (activations[-1] - y) * self.output_activation_derivative(z_values[-1])
        weight_gradients = []
        bias_gradients = []

        # Iterate backwards through the layers
        for i in reversed(range(len(self.weights))):
            if activations[i].ndim == 1:
                activations_t = activations[i].reshape(-1, 1)
            else:
                activations_t = activations[i].T
            # Calculate gradients for weights and biases
            weight_gradient = np.dot(activations_t, delta)
            bias_gradient = np.sum(delta, axis=0)

            # Insert gradients at the beginning of the list
            weight_gradients.insert(0, weight_gradient)
            bias_gradients.insert(0, bias_gradient)

            if i > 0:
                # Calculate error for the previous layer
                delta = np.dot(delta, self.weights[i].T) * self.activation_derivative(z_values[i-1])

        return weight_gradients, bias_gradients

    def update_weights(self, weight_gradients, bias_gradients, t=None):
        """
        Updates the weights and biases using the selected optimizer.

        Parameters:
            weight_gradients (list of np.ndarray): Gradients of the weights.
            bias_gradients (list of np.ndarray): Gradients of the biases.
            t (int, optional): Iteration step, relevant for certain optimizers like Adam.
        """
        updated_weights, updated_biases = self.optimizer.update(self.weights, self.biases, weight_gradients, bias_gradients, t)
        self.weights = updated_weights
        self.biases = updated_biases

    def validate_labels(self, y_true, y_pred):
        """
        Checks whether the labels and predictions are in the correct format.

        Parameters:
            y_true (np.ndarray or list): True labels.
            y_pred (np.ndarray or list): Predicted labels.

        Raises:
            ValueError: If labels are not array-like.
        """
        if not isinstance(y_true, (np.ndarray, list)):
            raise ValueError(f"'y_true' must be an array-like object. Received: {y_true}")
        if not isinstance(y_pred, (np.ndarray, list)):
            raise ValueError(f"'y_pred' must be an array-like object. Received: {y_pred}")

    def train_step(self, X_batch, y_batch, use_backpropagation=True, t=None):
        """
        Performs a training step over a batch of data.

        Parameters:
            X_batch (np.ndarray): Batch of input data.
            y_batch (np.ndarray): Batch of target values.
            use_backpropagation (bool): Whether to use backpropagation.
            t (int, optional): Iteration step for the optimizer.

        Returns:
            float: Average loss of the batch.
        """
        total_loss = 0
        for x, y in zip(X_batch, y_batch):
            x = x.reshape(1, -1)  # Reshape input to the correct form

            # Check and adjust target values
            if isinstance(y, (int, float)):
                y = np.array([[y]], dtype=np.float32)
            elif isinstance(y, np.ndarray) and y.ndim == 1:
                y = y.reshape(1, -1)
            
            # Forward pass
            activations, z_values = self.feedforward(x)
            self.validate_labels(y, activations[-1])  # Validate labels
            # Calculate loss (Mean Squared Error)
            loss = mean_squared_error(y, activations[-1])
            total_loss += loss

            if use_backpropagation:
                # Backward pass
                weight_gradients, bias_gradients = self.backpropagation(activations, z_values, y)
                # Update weights and biases
                self.update_weights(weight_gradients, bias_gradients, t)
        return total_loss / len(X_batch)

    def adjust_learning_rate(self, epoch, initial_lr, decay_rate=0.9):
        """
        Dynamically adjusts the learning rate based on the current epoch.

        Parameters:
            epoch (int): Current epoch.
            initial_lr (float): Initial learning rate.
            decay_rate (float): Rate at which the learning rate decays per epoch.
        """
        self.learning_rate = initial_lr * (decay_rate ** epoch)
        self.optimizer.learning_rate = self.learning_rate  # Update optimizer's learning rate

    def train(self, X_train, y_train, epochs, batch_size, use_backpropagation=True, learning_rate_decay=False, early_stopping_patience=None):
        """
        Trains the neural network over the specified number of epochs and batches.

        Parameters:
            X_train (np.ndarray): Training input data.
            y_train (np.ndarray): Training target values.
            epochs (int): Number of epochs to train.
            batch_size (int): Size of each batch.
            use_backpropagation (bool): Whether to use backpropagation.
            learning_rate_decay (bool): Whether to dynamically adjust the learning rate.
            early_stopping_patience (int, optional): Number of epochs with no improvement after which training will be stopped.

        Returns:
            dict: History of training metrics (e.g., loss).
        """
        history = {'loss': []}
        start_time = time.time()
        initial_lr = self.learning_rate
        num_batches = len(X_train) // batch_size

        for epoch in range(epochs):
            epoch_start_time = time.time()
            total_loss = 0
            # Shuffle training data
            shuffled_indices = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[shuffled_indices]
            y_train_shuffled = y_train[shuffled_indices]

            for i in range(num_batches):
                start_index = i * batch_size
                end_index = (i + 1) * batch_size
                X_batch = X_train_shuffled[start_index:end_index]
                y_batch = y_train_shuffled[start_index:end_index]

                # Perform a training step
                batch_loss = self.train_step(X_batch, y_batch, use_backpropagation, epoch + 1)
                total_loss += batch_loss

            # Calculate average loss for the epoch
            avg_loss = total_loss / num_batches
            history['loss'].append(avg_loss)

            logger.info(f"Epoch: {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}")

            # Dynamic learning rate adjustment
            if learning_rate_decay:
                self.adjust_learning_rate(epoch, initial_lr)
                logger.debug(f"Learning Rate adjusted to: {self.learning_rate:.4f}")

            # Early Stopping
            if early_stopping_patience and early_stopping(history, early_stopping_patience):
                logger.info("Early stopping triggered due to stagnation in loss.")
                break

            epoch_end_time = time.time()
            logger.debug(f"Epoch {epoch + 1}/{epochs}  Time: {epoch_end_time - epoch_start_time:.2f}s")

        end_time = time.time()
        logger.info(f"Training completed. Total training time: {end_time - start_time:.2f}s")
        return history

    def log_neuron_activations(self, inputs):
        """
        Logs the activations of the neurons for the given inputs.

        Parameters:
            inputs (list of np.ndarray): List of input data.
        """
        for i, input_data in enumerate(inputs):
            activations, _ = self.feedforward(input_data)
            for layer_idx, activation in enumerate(activations[1:]):
                if layer_idx == len(activations[1:]) - 1:
                    logger.debug(f"Input {i + 1}, Output Layer: Activations: {activation}")
                else:
                    logger.debug(f"Input {i + 1}, Layer {layer_idx + 1}: Activations: {activation}")

    def predict(self, X):
        """
        Returns the network's predictions for the input data.

        Parameters:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Network predictions.
        """
        predictions = []
        for x in X:
            activations, _ = self.feedforward(x)
            predictions.append(activations[-1])
        predictions = np.array(predictions)

        # Reduce to 2D (n_samples, 1) for regression
        if self.output_activation == "sigmoid" and predictions.ndim == 3:
            predictions = predictions.squeeze(axis=-1)
        elif predictions.ndim == 1:  # If 1D, reshape to (n_samples, 1)
            predictions = predictions.reshape(-1, 1)
        return predictions

    def test_network(self, test_data):
        """
        Tests the network's performance using the test data.

        Parameters:
            test_data (list of tuples): List of (input, target) pairs.

        Returns:
            dict: Metrics of the network's performance (e.g., accuracy, MSE).
        """
        X_test, y_test = zip(*test_data)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        predictions = self.predict(X_test)

        # Validate labels
        for y_true, y_pred in zip(y_test, predictions):
            self.validate_labels(y_true, y_pred)

        if self.output_activation == "softmax":
            # Classification: Calculate accuracy
            predicted_labels = np.argmax(predictions, axis=1)
            true_labels = np.argmax(y_test, axis=1)
            return {'accuracy': accuracy_score(true_labels, predicted_labels)}
        else:
            # Regression: Calculate MSE, MAE, and R²-Score
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            return {'mean_squared_error': mse, 'mean_absolute_error': mae, 'r2_score': r2}

    def visualize(self):
        """
        Visualizes the network architecture, including connections between neurons.
        """
        if len(self.num_nodes) < 3:
            logger.warning("No visualization available due to insufficient layers.")
            return

        plt.figure(figsize=(10, 6))
        layer_sizes = self.num_nodes
        for layer_idx, nodes in enumerate(layer_sizes):
            y = np.linspace(0, 1, nodes)
            x = np.full(nodes, layer_idx)
            plt.scatter(x, y, s=300, label=f'Layer {layer_idx + 1}', zorder=2)

            if layer_idx > 0:
                prev_y = np.linspace(0, 1, layer_sizes[layer_idx-1])
                prev_x = np.full(layer_sizes[layer_idx-1], layer_idx - 1)
                weights = self.weights[layer_idx-1]
                # Draw connections between neurons of the previous and current layer
                for i in range(len(prev_y)):
                    for j in range(len(y)):
                        plt.plot([prev_x[i], x[j]], [prev_y[i], y[j]], color='gray', alpha=0.2, zorder=1)
        plt.title("Network Architecture")
        plt.xlabel("Layer")
        plt.ylabel("Neuron Position")
        plt.yticks([])
        plt.xticks(range(len(layer_sizes)), labels=[f'Layer {i+1}' for i in range(len(layer_sizes))])
        plt.legend()
        plt.show()

    def save_network(self, filepath):
        """
        Saves the network as a Pickle file.

        Parameters:
            filepath (str): Path to the output file.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Network saved to: {filepath}")

    @staticmethod
    def load_network(filepath):
        """
        Loads a network from a Pickle file.

        Parameters:
            filepath (str): Path to the Pickle file.

        Returns:
            Network: Loaded network object.
        """
        with open(filepath, 'rb') as f:
            network = pickle.load(f)
        logger.info(f"Network loaded from: {filepath}")
        return network

    def to_json(self, filepath):
        """
        Saves the network as a JSON file. Includes structure and weights/biases.

        Parameters:
            filepath (str): Path to the output file.
        """
        network_data = {
            'num_nodes': self.num_nodes,
            'V_max': self.V_max,
            'R_harmony': self.R_harmony,
            'activation_function': self.activation_function,
            'learning_rate': self.learning_rate,
            'momentum': self.momentum,
            'output_activation': self.output_activation,
            'optimizer_method': self.optimizer_method,
            'weights': [w.astype(np.float64).tolist() for w in self.weights],
            'biases': [b.astype(np.float64).tolist() for b in self.biases]
        }
        with open(filepath, 'w') as f:
            json.dump(network_data, f, indent=4)
        logger.info(f"Network saved as JSON to: {filepath}")

    @staticmethod
    def from_json(filepath):
        """
        Loads a network from a JSON file.

        Parameters:
            filepath (str): Path to the JSON file.

        Returns:
            Network: Loaded network object.
        """
        with open(filepath, 'r') as f:
            network_data = json.load(f)
        network = Network(
            num_nodes=network_data['num_nodes'],
            V_max=network_data['V_max'],
            R_harmony=network_data['R_harmony'],
            activation_function=network_data['activation_function'],
            learning_rate=network_data['learning_rate'],
            momentum=network_data['momentum'],
            output_activation=network_data['output_activation'],
            optimizer_method=network_data['optimizer_method']
        )
        network.weights = [np.array(w, dtype=np.float64) for w in network_data['weights']]
        network.biases = [np.array(b, dtype=np.float64) for b in network_data['biases']]
        logger.info(f"Network loaded from JSON file: {filepath}")
        return network

    def plot_training_history(self, history):
        """
        Visualizes the training progress (Loss and Accuracy).

        Parameters:
            history (dict): History of training metrics.
        """
        epochs = range(1, len(history['loss']) + 1)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history['loss'], 'b-o', label='Loss')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, test_data):
        """
        Creates the confusion matrix for classification, provided that Softmax is used as the output activation.

        Parameters:
            test_data (list of tuples): List of (input, target) pairs.
        """
        if self.output_activation != "softmax":
            logger.warning("Confusion matrix is only applicable for classification.")
            return

        X_test, y_test = zip(*test_data)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        predictions = self.predict(X_test)
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(y_test, axis=1)

        plot_confusion_matrix(true_labels, predicted_labels, labels=list(range(y_test.shape[1])))

    def save_trial_results(self, filepath):
        """
        Saves the results of various trials as a CSV file.

        Parameters:
            filepath (str): Path to the output CSV file.
        """
        if not self.trial_results:
            logger.warning("No training results available to save.")
            return

        fieldnames = self.trial_results[0].keys()
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.trial_results)
        logger.info(f"Trial results saved to: {filepath}")

    def run_trials(self, X_train, y_train, X_test, y_test, num_trials=5, epochs=10, batch_size=32, use_backpropagation=True, learning_rate_decay=False, early_stopping_patience=None, hyperparameter_space={}):
        """
        Conducts multiple training runs with different hyperparameters and saves the results.

        Parameters:
            X_train (np.ndarray): Training input data.
            y_train (np.ndarray): Training target values.
            X_test (np.ndarray): Test input data.
            y_test (np.ndarray): Test target values.
            num_trials (int): Number of trials to conduct.
            epochs (int): Number of epochs per trial.
            batch_size (int): Batch size per trial.
            use_backpropagation (bool): Whether to use backpropagation.
            learning_rate_decay (bool): Whether to dynamically adjust the learning rate.
            early_stopping_patience (int, optional): Number of epochs with no improvement for early stopping.
            hyperparameter_space (dict, optional): Hyperparameter space for the trials.
        """
        if not hyperparameter_space:
            hyperparameter_space = {
                'learning_rate': [0.01, 0.001],
                'momentum': [0.9, 0.99],
                'optimizer_method': ['SGD', 'Adam']
            }
        keys = hyperparameter_space.keys()
        values = hyperparameter_space.values()
        combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
        logger.info(f"Number of hyperparameter combinations: {len(combinations)}")

        for i, params in enumerate(combinations):
            logger.info(f"Starting Trial {i+1} with params: {params}")
            trial_start_time = time.time()
            # Set hyperparameters for this trial

            if 'learning_rate' in params and self.learning_rate != params['learning_rate']:
                self.learning_rate = params['learning_rate']

            if 'momentum' in params and self.momentum != params['momentum']:
                self.momentum = params['momentum']

            if 'optimizer_method' in params and self.optimizer_method != params['optimizer_method']:
                self.optimizer_method = params['optimizer_method']

            # Update the optimizer based on new hyperparameters
            self.optimizer = self.get_optimizer(self.optimizer_method, self.learning_rate, self.momentum)
            self.optimizer.initialize(self.weights, self.biases)

            # Train the network
            history = self.train(X_train, y_train, epochs, batch_size, use_backpropagation, learning_rate_decay, early_stopping_patience)

            # Test the network
            test_data = list(zip(X_test, y_test))
            metrics = self.test_network(test_data)

            trial_end_time = time.time()
            trial_duration = trial_end_time - trial_start_time
            # Save the results
            trial_results = {
                'trial_number': i+1,
                'learning_rate': self.learning_rate,
                'momentum': self.momentum,
                'optimizer_method': self.optimizer_method,
                'epochs': epochs,
                'batch_size': batch_size,
                'training_time': f"{trial_duration:.2f}s",
                'loss': history['loss'][-1] if history['loss'] else None
            }

            if isinstance(metrics, dict):
                trial_results.update(metrics)

            self.trial_results.append(trial_results)
            logger.info(f"Trial {i+1} finished. Results: {trial_results}")

            # Visualize the results of every second trial
            if (i+1) % 2 == 0:
                self.plot_training_history(history)
                self.plot_confusion_matrix(test_data)

        logger.info("All trials completed.")

def create_dummy_data(num_samples=1000, num_features=10, num_classes=2):
    """
    Generates dummy data for testing purposes.

    Parameters:
        num_samples (int): Number of samples.
        num_features (int): Number of features.
        num_classes (int): Number of classes.

    Returns:
        tuple: Feature matrix and labels.
    """
    X = np.random.rand(num_samples, num_features).astype(np.float32)
    if num_classes > 2:
        y = np.random.randint(0, num_classes, num_samples)
        y = np.eye(num_classes)[y]  # One-Hot Encoding
    else:
        y = np.random.randint(0, num_classes, num_samples).astype(np.float32).reshape(-1, 1)
    return X, y

class TestNetwork(unittest.TestCase):
    """
    Test class for the Network object, using the unittest framework.
    """

    def test_label_shapes(self):
        """
        Tests whether the labels have the expected shape.
        """
        X, y = create_dummy_data(num_samples=10, num_features=10, num_classes=2)
        self.assertEqual(y.shape[1], 1, f"Expected shape (100, 1), but got: {y.shape}")

    def test_network_training_dummy_data(self):
        """
        Tests training the network with dummy data.
        """
        # Generate dummy data for testing
        X_train, y_train = create_dummy_data(num_samples=100, num_features=10, num_classes=2)
        X_test, y_test = create_dummy_data(num_samples=50, num_features=10, num_classes=2)

        # Initialize network
        num_nodes = [X_train.shape[1], 20, 1]  # Input, Hidden, Output
        net = Network(num_nodes=num_nodes, V_max=1, R_harmony=0.8, activation_function='relu',
                      learning_rate=0.01, momentum=0.9, output_activation='sigmoid', optimizer_method='SGD')

        # Train with the generated data
        history = net.train(X_train, y_train, epochs=5, batch_size=10)

        # Test the network with test data
        test_data = list(zip(X_test, y_test))
        metrics = net.test_network(test_data)

        # Check if test metrics are correct
        self.assertIsInstance(metrics, dict, "test_network returned wrong type")
        if net.output_activation == "softmax":
            self.assertIn('accuracy', metrics, "Accuracy not in the test results")
        else:
            self.assertIn('mean_squared_error', metrics, "MSE not in test results")
            self.assertIn('mean_absolute_error', metrics, "MAE not in test results")
            self.assertIn('r2_score', metrics, "R2 not in test results")

        # Check if loss history is correct
        self.assertGreater(len(history['loss']), 0, "No losses in training history")
        self.assertIsNotNone(history['loss'][-1], "Last loss is None")

    def test_network_training_breast_cancer_data(self):
        """
        Tests training and testing the network with the breast cancer dataset.
        """
        # Load breast cancer data
        cancer = load_breast_cancer()
        X, y = cancer.data, cancer.target
        X = X.astype(np.float32)
        y = y.astype(np.float32).reshape(-1, 1)  # Target values as column vector
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalize data
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Initialize network
        num_nodes = [X_train.shape[1], 10, 1]  # Input, Hidden, Output
        net = Network(num_nodes=num_nodes, V_max=1, R_harmony=0.8, activation_function='relu',
                      learning_rate=0.01, momentum=0.9, output_activation='sigmoid', optimizer_method='Adam')

        # Train
        history = net.train(X_train, y_train, epochs=10, batch_size=32)

        # Test
        test_data = list(zip(X_test, y_test))
        metrics = net.test_network(test_data)

        # Check if test metrics are correct
        self.assertIsInstance(metrics, dict, "test_network returned wrong type")
        if net.output_activation == "softmax":
            self.assertIn('accuracy', metrics, "Accuracy not in test results")
        else:
            self.assertIn('mean_squared_error', metrics, "MSE not in test results")
            self.assertIn('mean_absolute_error', metrics, "MAE not in test results")
            self.assertIn('r2_score', metrics, "R2 not in test results")
        self.assertGreater(len(history['loss']), 0, "No losses in training history")
        self.assertIsNotNone(history['loss'][-1], "Last loss is None")

    def test_network_from_to_json(self):
        """
        Tests saving and loading the network as a JSON file.
        """
        # Generate dummy data for testing
        X_train, y_train = create_dummy_data(num_samples=100, num_features=10, num_classes=2)
        num_nodes = [X_train.shape[1], 20, 1]  # Input, Hidden, Output

        # Create a network and train it
        net = Network(num_nodes=num_nodes, V_max=1, R_harmony=0.8, activation_function='relu',
                      learning_rate=0.01, momentum=0.9, output_activation='sigmoid')
        net.train(X_train, y_train, epochs=1, batch_size=10)

        # Save network as JSON
        json_filepath = 'test_network.json'
        net.to_json(json_filepath)
        self.assertTrue(os.path.exists(json_filepath))

        # Load network from JSON
        loaded_net = Network.from_json(json_filepath)

        # Check if parameters and structures match
        self.assertEqual(net.num_nodes, loaded_net.num_nodes)
        self.assertEqual(net.V_max, loaded_net.V_max)
        self.assertEqual(net.R_harmony, loaded_net.R_harmony)
        self.assertEqual(net.activation_function, loaded_net.activation_function)
        self.assertEqual(net.learning_rate, loaded_net.learning_rate)
        self.assertEqual(net.momentum, loaded_net.momentum)
        self.assertEqual(net.output_activation, loaded_net.output_activation)
        self.assertEqual(net.optimizer_method, loaded_net.optimizer_method)

        # Use numerical tolerance when comparing weights and biases
        for w1, w2 in zip(net.weights, loaded_net.weights):
            np.testing.assert_array_almost_equal(w1, w2, decimal=7)
        for b1, b2 in zip(net.biases, loaded_net.biases):
            np.testing.assert_array_almost_equal(b1, b2, decimal=7)

        # Delete JSON file after test
        os.remove(json_filepath)

    def test_network_from_to_pickle(self):
        """
        Tests saving and loading the network as a Pickle file.
        """
        # Generate dummy data for testing
        X_train, y_train = create_dummy_data(num_samples=100, num_features=10, num_classes=2)
        num_nodes = [X_train.shape[1], 20, 1]  # Input, Hidden, Output

        # Create a network and train it
        net = Network(num_nodes=num_nodes, V_max=1, R_harmony=0.8, activation_function='relu',
                      learning_rate=0.01, momentum=0.9, output_activation='sigmoid')
        net.train(X_train, y_train, epochs=1, batch_size=10)

        # Save network as Pickle
        pickle_filepath = 'test_network.pkl'
        net.save_network(pickle_filepath)
        self.assertTrue(os.path.exists(pickle_filepath))

        # Load network from Pickle
        loaded_net = Network.load_network(pickle_filepath)

        # Check if parameters and structures match
        self.assertEqual(net.num_nodes, loaded_net.num_nodes)
        self.assertEqual(net.V_max, loaded_net.V_max)
        self.assertEqual(net.R_harmony, loaded_net.R_harmony)
        self.assertEqual(net.activation_function, loaded_net.activation_function)
        self.assertEqual(net.learning_rate, loaded_net.learning_rate)
        self.assertEqual(net.momentum, loaded_net.momentum)
        self.assertEqual(net.output_activation, loaded_net.output_activation)
        self.assertEqual(net.optimizer_method, loaded_net.optimizer_method)
        for w1, w2 in zip(net.weights, loaded_net.weights):
            np.testing.assert_array_equal(w1, w2)
        for b1, b2 in zip(net.biases, loaded_net.biases):
            np.testing.assert_array_equal(b1, b2)

        # Delete Pickle file after test
        os.remove(pickle_filepath)

    def test_run_trials_with_dummy_data(self):
        """
        Tests running multiple training trials with dummy data.
        """
        # Generate dummy data for testing
        X_train, y_train = create_dummy_data(num_samples=100, num_features=10, num_classes=2)
        X_test, y_test = create_dummy_data(num_samples=50, num_features=10, num_classes=2)
        num_nodes = [X_train.shape[1], 20, 1]  # Input, Hidden, Output
        net = Network(num_nodes=num_nodes, V_max=1, R_harmony=0.8, activation_function='relu',
                      learning_rate=0.01, momentum=0.9, output_activation='sigmoid')

        # Run trials
        net.run_trials(X_train, y_train, X_test, y_test, num_trials=2, epochs=2, batch_size=10, hyperparameter_space={
            'learning_rate': [0.01, 0.001],
            'momentum': [0.9, 0.99],
            'optimizer_method': ['SGD', 'Adam']
        })
        
        # Check if trial results were generated
        self.assertGreater(len(net.trial_results), 0, "No training results available from trials.")

        # Check if CSV file was created
        csv_filepath = 'trial_results.csv'
        net.save_trial_results(csv_filepath)
        self.assertTrue(os.path.exists(csv_filepath))
        os.remove(csv_filepath)

if __name__ == '__main__':
    unittest.main()
