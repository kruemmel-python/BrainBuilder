# Modular and Extensible Neural Network in Python

A powerful, modular, and extensible neural network implemented in Python. This project provides comprehensive tools for creating, training, and evaluating neural networks with various optimization algorithms and activation functions. It includes features for data processing, visualization, as well as extensive unit tests to ensure functionality.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Loading Data](#loading-data)
  - [Initializing the Network](#initializing-the-network)
  - [Training](#training)
  - [Making Predictions](#making-predictions)
  - [Saving and Loading the Network](#saving-and-loading-the-network)
  - [Visualization](#visualization)
  - [Running Trials](#running-trials)
- [Optimizers](#optimizers)
- [Activation Functions](#activation-functions)
- [Glossary](#glossary)
- [Unit Tests](#unit-tests)
- [Example Project: Iris Dataset Classification](#example-project-iris-dataset-classification)
- [License](#license)

## Introduction

This project implements a modular and extensible neural network in Python. Neural networks are a type of machine learning designed to recognize patterns in data and make predictions. This network is suitable for both classification and regression tasks and offers various optimization algorithms and activation functions to meet different requirements.

## Features

- **Modular Architecture**: Flexibly define the number of layers and neurons per layer.
- **Various Optimizers**: Support for SGD (with Momentum), Adam, and RMSprop.
- **Various Activation Functions**: Sigmoid, ReLU, and Softmax.
- **Data Processing**: Load data from CSV files, scaling, one-hot encoding, and handling missing values.
- **Visualization**: Display network architecture, training progress, and confusion matrix.
- **Saving and Loading**: Networks can be saved and loaded as JSON or Pickle files.
- **Trials**: Conduct multiple training runs with different hyperparameters to optimize performance.
- **Early Stopping**: Automatically stop training when the loss function stagnates.
- **Comprehensive Unit Tests**: Ensure functionality and stability of the network.
- **Extensibility**: Easily extendable with additional features and optimizers.

## Prerequisites

This project requires basic knowledge of Python. Knowledge of machine learning and neural networks is beneficial but not strictly necessary. For beginners, we recommend the following resources:

- [Introduction to Machine Learning with Python](https://scikit-learn.org/stable/tutorial/index.html)
- [Neural Networks Explained](https://www.deeplearning.ai/)

## Installation

Ensure you have Python 3.7 or higher installed. Install the required dependencies using `pip`:

```bash
pip install numpy pandas matplotlib scikit-learn
```

The `unittest` module is used for unit tests and is included with Python by default.

## Usage

### Loading Data

Use the `load_data_from_csv` function to load data from a CSV file and split it into training and testing sets.

```python
from your_module import load_data_from_csv

X_train, X_test, y_train, y_test = load_data_from_csv(
    filepath='data.csv',
    target_column='target',
    test_size=0.2,
    random_state=42,
    scale_features=True
)
```

### Initializing the Network

Initialize the network by defining its structure and hyperparameters.

```python
from your_module import Network

num_nodes = [X_train.shape[1], 20, 1]  # Input, Hidden Layer, Output
network = Network(
    num_nodes=num_nodes,
    V_max=1.0,
    R_harmony=0.8,
    activation_function='relu',
    learning_rate=0.01,
    momentum=0.9,
    output_activation='sigmoid',
    optimizer_method='SGD'
)
```

### Training

Train the network with the training data.

```python
history = network.train(
    X_train,
    y_train,
    epochs=100,  # Number of complete passes through the training data
    batch_size=32,  # Number of samples processed before updating the model
    use_backpropagation=True,  # Use the backpropagation algorithm to compute gradients
    learning_rate_decay=True,  # Adjust the learning rate during training
    early_stopping_patience=10  # Stop training if loss does not improve for 10 epochs
)
```

*Note: The learning rate controls how much the weights are adjusted during each update. A too high learning rate can make training unstable, while a too low learning rate can lead to slow convergence.*

### Making Predictions

Use the trained network to make predictions.

```python
predictions = network.predict(X_test)
```

### Saving and Loading the Network

#### Saving as Pickle

```python
network.save_network('network.pkl')
```

#### Loading from Pickle

```python
from your_module import Network

loaded_network = Network.load_network('network.pkl')
```

#### Saving as JSON

```python
network.to_json('network.json')
```

#### Loading from JSON

```python
loaded_network = Network.from_json('network.json')
```

### Visualization

#### Visualize Network Architecture

```python
network.visualize()
```

#### Plot Training Progress

```python
network.plot_training_history(history)
```

#### Create Confusion Matrix (for Classification)

```python
network.plot_confusion_matrix(list(zip(X_test, y_test)))
```

### Running Trials

Conduct multiple training runs with different hyperparameters to find the best settings.

```python
network.run_trials(
    X_train, y_train,
    X_test, y_test,
    num_trials=5,
    epochs=50,
    batch_size=32,
    hyperparameter_space={
        'learning_rate': [0.01, 0.001],
        'momentum': [0.9, 0.99],
        'optimizer_method': ['SGD', 'Adam']
    }
)
```

Save the trial results as a CSV file:

```python
network.save_trial_results('trial_results.csv')
```

## Optimizers

The network supports various optimization algorithms that update the weights and biases during training:

- **SGD (Stochastic Gradient Descent)**: With Momentum to accelerate training.
- **Adam**: Adaptive Moment Estimation, combines Momentum and RMSprop.
- **RMSprop**: Adaptive learning rates based on the moving average of squared gradients.

### Choosing an Optimizer

When initializing the network, you can choose the desired optimizer:

```python
network = Network(
    num_nodes=num_nodes,
    optimizer_method='Adam',
    learning_rate=0.001
)
```

## Activation Functions

The network offers various activation functions to introduce non-linearities:

- **Sigmoid**: Suitable for binary classification tasks.
- **ReLU (Rectified Linear Unit)**: Promotes fast convergence and avoids the vanishing gradient problem.
- **Softmax**: Suitable for multi-class classification tasks.

### Choosing an Activation Function

When initializing the network, you can select the desired activation function:

```python
network = Network(
    activation_function='relu',
    output_activation='softmax'
)
```

## Glossary

- **Optimizer**: Algorithms that adjust the weights and biases of a neural network during training.
- **Momentum**: A parameter that helps accelerate training and avoid local minima by considering past gradients.
- **Softmax**: An activation function that converts outputs into probabilities that sum to 1, commonly used in the output layer for multi-class classification.
- **One-Hot Encoding**: A method for encoding categorical variables where each class is represented as a vector with a 1 in the position corresponding to the class and 0s elsewhere.

## Unit Tests

The project includes comprehensive unit tests to ensure functionality. Run the tests with the following command:

```bash
python your_program.py
```

The tests cover various aspects, including:

- Checking label formats
- Training and testing with dummy data
- Saving and loading the network as JSON and Pickle
- Running trials

## Example Project: Iris Dataset Classification

Here is a simple example of how to use the network for classifying the Iris dataset.

### Step 1: Loading Data

```python
from your_module import load_data_from_csv
from sklearn.datasets import load_iris
import pandas as pd

# Load the Iris dataset and save it as a CSV file
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df.to_csv('iris.csv', index=False)

# Load the data
X_train, X_test, y_train, y_test = load_data_from_csv(
    filepath='iris.csv',
    target_column='species',
    test_size=0.2,
    random_state=42,
    scale_features=True
)
```

### Step 2: Initializing the Network

```python
from your_module import Network

num_nodes = [X_train.shape[1], 10, 3]  # Input, Hidden Layer, Output
network = Network(
    num_nodes=num_nodes,
    V_max=1.0,
    R_harmony=0.8,
    activation_function='relu',
    learning_rate=0.01,
    momentum=0.9,
    output_activation='softmax',
    optimizer_method='Adam'
)
```

### Step 3: Training

```python
history = network.train(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    use_backpropagation=True,
    learning_rate_decay=True,
    early_stopping_patience=5
)
```

### Step 4: Making Predictions and Evaluation

```python
predictions = network.predict(X_test)
metrics = network.test_network(list(zip(X_test, y_test)))
print(metrics)
```

### Step 5: Visualization

```python
network.plot_training_history(history)
network.plot_confusion_matrix(list(zip(X_test, y_test)))
```

### Step 6: Saving and Loading the Network

```python
# Saving
network.save_network('iris_network.pkl')
network.to_json('iris_network.json')

# Loading
loaded_network = Network.load_network('iris_network.pkl')
loaded_network_json = Network.from_json('iris_network.json')
```

This example walks you through the entire process of data preparation, network initialization, training, making predictions, evaluation, and saving the network. It provides a practical introduction and facilitates understanding of each step.

## License

This project is licensed under the MIT License. For more information, see the [LICENSE](LICENSE) file.

---
