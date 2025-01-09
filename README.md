# Neural Network Training and Evaluation

This repository contains a Python implementation of a neural network for training and evaluation. The network supports various activation functions, optimizers, and includes functionalities for data handling, visualization, and hyperparameter tuning.

## Features

- **Data Loading and Preprocessing**: Functions to load data from CSV files, handle missing values, and preprocess data.
- **Activation Functions**: Implementations of sigmoid, ReLU, and softmax activation functions.
- **Optimizers**: Support for SGD, Momentum, Adam, and RMSprop optimizers.
- **Training and Evaluation**: Functions to train the network, evaluate its performance, and visualize training history.
- **Hyperparameter Tuning**: Run multiple trials with different hyperparameters to find the best configuration.
- **Model Saving and Loading**: Save and load the network model in both pickle and JSON formats.

## Requirements

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Logging
- JSON
- Pickle
- OS
- CSV
- Random
- Itertools

## Installation

To install the required packages, run:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Usage

### Data Loading and Preprocessing

```python
X_train, X_test, y_train, y_test = load_data_from_csv('data.csv', target_column='label')
check_for_nan(X_train)
check_for_nan(y_train)
```

### Network Initialization and Training

```python
input_dim = X_train.shape[1]
num_nodes = [input_dim, 20, 1]  # Example: 1 hidden layer with 20 neurons, 1 output neuron

network = Network(num_nodes, V_max=0.5, R_harmony=0.1, activation_function='relu', learning_rate=0.01, momentum=0.9, output_activation='sigmoid', optimizer_method='Adam')
network.validate_dimensions(X_train, y_train)
history = network.train(X_train, y_train, epochs=10, batch_size=32, use_backpropagation=True, learning_rate_decay=True, early_stopping_patience=3)
network.plot_training_history(history)
```

### Network Evaluation

```python
test_data = list(zip(X_test, y_test))
metrics = network.test_network(test_data)
logger.info(f"Metrics after training: {metrics}")
```

### Visualization

```python
network.visualize()
network.log_neuron_activations(X_test[:5])  # Log activations for the first 5 test data points
network.plot_confusion_matrix(test_data)
```

### Model Saving and Loading

```python
network.save_network('my_network.pkl')
loaded_network = Network.load_network('my_network.pkl')
loaded_metrics = loaded_network.test_network(test_data)
logger.info(f"Loaded network metrics: {loaded_metrics}")

network.to_json("my_network.json")
loaded_network_json = Network.from_json("my_network.json")
loaded_metrics_json = loaded_network_json.test_network(test_data)
logger.info(f"Loaded JSON network metrics: {loaded_metrics_json}")
```

### Hyperparameter Tuning

```python
hyperparameter_space = {
    'learning_rate': [0.01, 0.001, 0.0001],
    'momentum': [0.9, 0.8, 0.7],
    'optimizer_method': ['SGD', 'Adam', 'RMSprop']
}

network.run_trials(X_train, y_train, X_test, y_test, epochs=5, batch_size=32, use_backpropagation=True, learning_rate_decay=True, early_stopping_patience=2, hyperparameter_space=hyperparameter_space)
network.save_trial_results('trial_results.csv')
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Acknowledgments

- Inspired by various neural network implementations and tutorials.
- Special thanks to the open-source community for their contributions to the libraries used in this project.
