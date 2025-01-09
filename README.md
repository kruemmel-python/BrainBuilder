# Modular and Extensible Neural Network in Python

A powerful, modular, and extensible neural network implemented in Python. This project provides comprehensive tools for creating, training, and evaluating neural networks with various optimization algorithms and activation functions. It includes features for data processing, visualization, as well as extensive unit tests to ensure functionality.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Creating and Loading the Dataset](#creating-and-loading-the-dataset)
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
- [Example Project: Breast Cancer Dataset Classification](#example-project-breast-cancer-dataset-classification)
- [License](#license)

## Introduction

Early detection of breast cancer significantly increases the chances of successful treatment and survival. This project leverages the power of neural networks to create an AI model specifically trained to detect breast cancer using the **Breast Cancer Wisconsin (Diagnostic) Dataset** from scikit-learn. By analyzing various features extracted from cell nuclei in breast tumor samples, the neural network learns to distinguish between malignant and benign tumors.

The primary goal of this project is to develop a reliable and efficient AI tool that can assist medical professionals in diagnosing breast cancer, thereby enhancing diagnostic accuracy and facilitating timely intervention. The modular and extensible architecture of the network allows for flexibility in experimentation with different configurations, optimization algorithms, and activation functions, making it a valuable resource for both learners and developers in the field of machine learning and medical diagnostics.

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

### Creating and Loading the Dataset

This project uses the **Breast Cancer Wisconsin (Diagnostic) Dataset** from scikit-learn. The dataset contains features of cell nuclei extracted from breast tumor samples, along with the target variable indicating whether the tumor is malignant (1) or benign (0).

#### Creating and Saving the Dataset as CSV

The dataset is loaded and saved as a CSV file for ease of use.

```python
import pandas as pd
from sklearn.datasets import load_breast_cancer

# Load the Breast Cancer dataset from sklearn
breast_cancer = load_breast_cancer()

# Convert the dataset to a Pandas DataFrame
data = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)

# Add the target variable (class label)
data['target'] = breast_cancer.target

# Display the first few rows of the DataFrame
print(data.head())

# Save the dataset as a CSV file
data.to_csv("breast_cancer.csv", index=False)
```

#### Loading and Preparing the Data

Use the `load_data_from_csv` function to load the saved dataset, scale the features, and split the dataset into training and testing sets.

```python
from your_module import load_data_from_csv

X_train, X_test, y_train, y_test = load_data_from_csv(
    filepath='breast_cancer.csv',
    target_column='target',
    test_size=0.2,
    random_state=42,
    scale_features=True
)
```

**What the Network Does with the Data:**

- **Data Loading**: The network reads the CSV file containing breast cancer data.
- **Data Preprocessing**: Features are scaled using Min-Max scaling to normalize the input data.
- **Training**: The neural network is trained on the training set to learn patterns that distinguish malignant from benign tumors.
- **Prediction**: After training, the network can predict the probability of a tumor being malignant or benign based on input features.
- **Evaluation**: The network's performance is evaluated using metrics such as accuracy, mean squared error (MSE), mean absolute error (MAE), and R² score.

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
    optimizer_method='Adam'
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
    early_stopping_patience=10  # Early stopping if loss does not improve for 10 epochs
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

## Example Project: Breast Cancer Dataset Classification

This example demonstrates how to use the neural network to train an AI model that detects breast cancer based on the provided CSV dataset. The model is trained to distinguish between malignant and benign tumors and is saved for future use.

### Step 1: Creating and Loading the Dataset

First, load the Breast Cancer dataset from scikit-learn, convert it to a CSV file, and then load it into the neural network.

```python
import pandas as pd
from your_module import load_data_from_csv

# Load the Breast Cancer dataset and save it as a CSV file
from sklearn.datasets import load_breast_cancer

breast_cancer = load_breast_cancer()
data = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
data['target'] = breast_cancer.target
data.to_csv("breast_cancer.csv", index=False)

# Load and prepare the data
X_train, X_test, y_train, y_test = load_data_from_csv(
    filepath='breast_cancer.csv',
    target_column='target',
    test_size=0.2,
    random_state=42,
    scale_features=True
)
```

### Step 2: Initializing the Network

Initialize the neural network with the desired architecture and hyperparameters.

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
    optimizer_method='Adam'
)
```

### Step 3: Training

Train the neural network using the training data.

```python
history = network.train(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    use_backpropagation=True,
    learning_rate_decay=True,
    early_stopping_patience=10
)
```

### Step 4: Making Predictions and Evaluation

After training, use the network to make predictions on the test set and evaluate its performance.

```python
predictions = network.predict(X_test)
metrics = network.test_network(list(zip(X_test, y_test)))
print(metrics)
```

### Step 5: Visualization

Visualize the training progress and the confusion matrix to understand the network's performance.

```python
network.plot_training_history(history)
network.plot_confusion_matrix(list(zip(X_test, y_test)))
```

### Step 6: Saving and Loading the Network

Save the trained network for future use and load it when needed.

```python
# Saving the network
network.save_network('breast_cancer_network.pkl')
network.to_json('breast_cancer_network.json')

# Loading the network
loaded_network = Network.load_network('breast_cancer_network.pkl')
loaded_network_json = Network.from_json('breast_cancer_network.json')
```

**What the Network Learns:**

The neural network learns to identify patterns in the features of breast tumor samples that are indicative of malignancy or benignity. By training on the dataset, the model adjusts its weights and biases to minimize the loss function, thereby improving its ability to make accurate predictions on unseen data.

**Saving the Model:**

After training, the model is saved in both Pickle and JSON formats. This allows for easy storage and retrieval of the trained model for future predictions without the need to retrain.

## License

This project is licensed under the MIT License. For more information, see the [LICENSE](LICENSE) file.

