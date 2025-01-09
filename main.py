import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, ConfusionMatrixDisplay
import logging
import time
import json
import pickle
import os
import csv
import random
import itertools
from sklearn.datasets import load_breast_cancer

# Konfiguration des Loggings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Globale Konfigurationen
RANDOM_STATE = 42


def load_data_from_csv(filepath, target_column='label', test_size=0.2, random_state=42, scale_features=True):
    data = pd.read_csv(filepath)
    target = data[target_column]
    features = data.drop(columns=[target_column])

    X = features.values.astype(np.float32)
    y = target.values

    # Zielspalte verarbeiten
    if len(np.unique(y)) > 2:  # Mehrklassenklassifikation
        y = LabelEncoder().fit_transform(y)
        y = np.eye(len(np.unique(y)))[y]  # One-Hot-Encoding
    else:  # Binärklassifikation
        y = y.astype(np.float32).reshape(-1, 1)

    if scale_features:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def test_label_shapes():
    X, y = create_dummy_data(num_samples=100, num_features=10, num_classes=2)
    assert y.shape[1] == 1, f"Erwartete Form (100, 1), aber erhalten: {y.shape}"



def save_dataset_to_csv(X, y, filepath):
    """
    Speichert Features und Labels in einer CSV-Datei.
    """
    # Falls y One-Hot-Encoded ist, konvertiere es zu eindimensionalen Labels
    if len(y.shape) > 1 and y.shape[1] > 1:
        y = np.argmax(y, axis=1)  # Maximalwert (Index) für jede Zeile nehmen

    data = pd.DataFrame(X)
    data['label'] = y
    data.to_csv(filepath, index=False)
    logger.info(f"Datensatz gespeichert in {filepath}")

def check_for_nan(data):
    """Überprüft, ob in den Daten fehlende Werte vorhanden sind."""
    if isinstance(data, np.ndarray):
        if np.isnan(data).any():
           raise ValueError("Die Eingabedaten enthalten NaN-Werte.")
    elif isinstance(data, list):
         for row in data:
              if isinstance(row, list):
                  if any(np.isnan(x) for x in row):
                      raise ValueError("Die Eingabedaten enthalten NaN-Werte.")
              elif np.isnan(row):
                raise ValueError("Die Eingabedaten enthalten NaN-Werte.")
    else:
        raise TypeError("Datentyp wird nicht unterstützt.")

def handle_overflow(x):
    """Sicherstellt, dass bei exponentiellen Operationen kein Overflow auftritt."""
    try:
        return np.exp(x)
    except OverflowError:
        return np.inf if x > 0 else 0.0

def sigmoid(x):
    """Sigmoid-Aktivierungsfunktion mit Schutz vor Overflow."""
    clipped_x = np.clip(x, -500, 500)  # Eingabewerte begrenzen
    return 1 / (1 + np.exp(-clipped_x))


def relu(x):
    """ReLU-Aktivierungsfunktion."""
    return np.maximum(0, x)

def softmax(x):
    """Softmax-Aktivierungsfunktion."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def sigmoid_derivative(x):
    """Ableitung der Sigmoid-Funktion."""
    s = sigmoid(x)
    return s * (1 - s)

def relu_derivative(x):
    """Ableitung der ReLU-Funktion."""
    return (x > 0).astype(float)

def softmax_derivative(x):
    """Ableitung der Softmax-Funktion (falls notwendig)."""
    s = softmax(x)
    return s * (1 - s)

def plot_confusion_matrix(y_true, y_pred, labels):
    """Erstellt und zeigt die Konfusionsmatrix."""
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=labels)
    plt.show()

class Optimizer:
    def __init__(self, method='SGD', learning_rate=0.01, momentum=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.method = method
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_velocities = None
        self.bias_velocities = None
        self.m_weights = None  # Adam
        self.v_weights = None  # Adam
        self.m_biases = None  # Adam
        self.v_biases = None  # Adam

    def initialize(self, weights, biases):
        self.weight_velocities = [np.zeros_like(w, dtype=np.float32) for w in weights] if self.method in ['SGD', 'Momentum'] else None
        self.bias_velocities = [np.zeros_like(b, dtype=np.float32) for b in biases] if self.method in ['SGD', 'Momentum'] else None
        self.m_weights = [np.zeros_like(w, dtype=np.float32) for w in weights] if self.method in ['Adam', 'RMSprop'] else None
        self.v_weights = [np.zeros_like(w, dtype=np.float32) for w in weights] if self.method in ['Adam', 'RMSprop'] else None
        self.m_biases = [np.zeros_like(b, dtype=np.float32) for b in biases] if self.method in ['Adam', 'RMSprop'] else None
        self.v_biases = [np.zeros_like(b, dtype=np.float32) for b in biases] if self.method in ['Adam', 'RMSprop'] else None

    def update(self, weights, biases, weight_gradients, bias_gradients, t=None):
        updated_weights = []
        updated_biases = []

        for i in range(len(weights)):
            if self.method == 'SGD':
                self.weight_velocities[i] = self.momentum * self.weight_velocities[i] - self.learning_rate * weight_gradients[i]
                updated_weights.append(weights[i] + self.weight_velocities[i])
                self.bias_velocities[i] = self.momentum * self.bias_velocities[i] - self.learning_rate * bias_gradients[i]
                updated_biases.append(biases[i] + self.bias_velocities[i])

            elif self.method == 'Momentum':
                self.weight_velocities[i] = self.momentum * self.weight_velocities[i] - self.learning_rate * weight_gradients[i]
                updated_weights.append(weights[i] + self.weight_velocities[i])
                self.bias_velocities[i] = self.momentum * self.bias_velocities[i] - self.learning_rate * bias_gradients[i]
                updated_biases.append(biases[i] + self.bias_velocities[i])

            elif self.method == 'Adam':
                self.m_weights[i] = self.beta1 * self.m_weights[i] + (1 - self.beta1) * weight_gradients[i]
                self.v_weights[i] = self.beta2 * self.v_weights[i] + (1 - self.beta2) * weight_gradients[i]**2
                m_hat = self.m_weights[i] / (1 - self.beta1**t)
                v_hat = self.v_weights[i] / (1 - self.beta2**t)
                updated_weights.append(weights[i] - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon))
                self.m_biases[i] = self.beta1 * self.m_biases[i] + (1 - self.beta1) * bias_gradients[i]
                self.v_biases[i] = self.beta2 * self.v_biases[i] + (1 - self.beta2) * bias_gradients[i]**2
                m_hat_b = self.m_biases[i] / (1 - self.beta1**t)
                v_hat_b = self.v_biases[i] / (1 - self.beta2**t)
                updated_biases.append(biases[i] - self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon))

            elif self.method == 'RMSprop':
                self.v_weights[i] = self.beta2 * self.v_weights[i] + (1 - self.beta2) * weight_gradients[i]**2
                updated_weights.append(weights[i] - self.learning_rate * weight_gradients[i] / (np.sqrt(self.v_weights[i]) + self.epsilon))
                self.v_biases[i] = self.beta2 * self.v_biases[i] + (1 - self.beta2) * bias_gradients[i]**2
                updated_biases.append(biases[i] - self.learning_rate * bias_gradients[i] / (np.sqrt(self.v_biases[i]) + self.epsilon))

        return updated_weights, updated_biases

def early_stopping(history, patience=5):
    """Prüft, ob das Training frühzeitig beendet werden soll."""
    if len(history['loss']) > patience and all(history['loss'][-i] >= history['loss'][-i-1] for i in range(1, patience+1)):
        return True
    return False

class Network:
    def __init__(self, num_nodes, V_max, R_harmony, activation_function='sigmoid', learning_rate=0.1, momentum=0.9, output_activation='sigmoid', optimizer_method='SGD'):
        """Initialisiert das Netzwerk."""
        self.num_nodes = num_nodes
        self.V_max = V_max
        self.R_harmony = R_harmony
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.output_activation = output_activation
        self.optimizer_method = optimizer_method

        self.weights = [np.random.uniform(-1, 1, size=(num_nodes[i], num_nodes[i+1])).astype(np.float32) for i in range(len(num_nodes) - 1)]
        self.biases = [np.zeros(nodes, dtype=np.float32) for nodes in num_nodes[1:]]

        self.optimizer = Optimizer(method=optimizer_method, learning_rate=self.learning_rate, momentum=self.momentum)
        self.optimizer.initialize(self.weights, self.biases)

        if activation_function == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation_function == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        else:
            raise ValueError("Aktivierungsfunktion nicht unterstützt.")

        if output_activation == 'sigmoid':
            self.output_activation_func = sigmoid
            self.output_activation_derivative = sigmoid_derivative
        elif output_activation == 'softmax':
            self.output_activation_func = softmax
            self.output_activation_derivative = softmax_derivative
        else:
            raise ValueError("Output Aktivierungsfunktion nicht unterstützt.")

        self.trial_results = []
        self.metrics_history = {}

    def validate_dimensions(self, x, y):
        """Überprüft, ob die Dimensionen von x und y mit den Netzwerkparametern übereinstimmen."""
        if x.shape[1] != self.num_nodes[0]:
            raise ValueError(f"Die Eingabedimension {x.shape[1]} stimmt nicht mit der erwarteten {self.num_nodes[0]} überein.")
        if y.ndim == 1 and self.num_nodes[-1] > 1:
            raise ValueError(f"Die Zielgröße {y.shape} passt nicht zu den Ausgabeneuronen {self.num_nodes[-1]}.")
        if y.ndim > 1 and y.shape[1] != self.num_nodes[-1]:
            raise ValueError(f"Die Zielgröße {y.shape[1]} passt nicht zu den Ausgabeneuronen {self.num_nodes[-1]}.")

    def feedforward(self, input_data):
        """Führt einen Feedforward-Pass durch."""
        input_data = input_data.reshape(1, -1)  # Sicherstellen, dass die Eingabe die richtige Form hat
        activations = [input_data]
        z_values = []

        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            z_values.append(z)

            if i == len(self.weights) - 1:  # Letzte Schicht
                activation = self.output_activation_func(z)
            else:
                activation = self.activation(z)
            activations.append(activation)

        return activations, z_values

    def backpropagation(self, activations, z_values, y):
        """Führt die Backpropagation durch."""

        delta = (activations[-1] - y) * self.output_activation_derivative(z_values[-1])
        weight_gradients = []
        bias_gradients = []

        for i in reversed(range(len(self.weights))):
            weight_gradient = np.dot(activations[i].T, delta)
            bias_gradient = np.sum(delta, axis=0)

            weight_gradients.insert(0, weight_gradient)
            bias_gradients.insert(0, bias_gradient)

            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.activation_derivative(z_values[i-1])

        return weight_gradients, bias_gradients

    def update_weights(self, weight_gradients, bias_gradients, t=None):
        """Aktualisiert die Gewichte und Biases mithilfe des ausgewählten Optimierers."""
        updated_weights, updated_biases = self.optimizer.update(self.weights, self.biases, weight_gradients, bias_gradients, t)
        self.weights = updated_weights
        self.biases = updated_biases

    def validate_labels(self, y_true, y_pred):
        """Prüft, ob Labels und Vorhersagen im richtigen Format vorliegen."""
        if not isinstance(y_true, (np.ndarray, list)):
            raise ValueError(f"'y_true' muss ein array-ähnliches Objekt sein. Erhalten: {y_true}")
        if not isinstance(y_pred, (np.ndarray, list)):
            raise ValueError(f"'y_pred' muss ein array-ähnliches Objekt sein. Erhalten: {y_pred}")

    def train_step(self, X_batch, y_batch, use_backpropagation=True, t=None):
        total_loss = 0
        for x, y in zip(X_batch, y_batch):
            x = x.reshape(1, -1)  # Eingabe in die richtige Form bringen

            # Zielwerte überprüfen und anpassen
            if isinstance(y, (int, float)):
                y = np.array([[y]], dtype=np.float32)
            elif isinstance(y, np.ndarray) and y.ndim == 1:
                y = y.reshape(1, -1)

            activations, z_values = self.feedforward(x)
            self.validate_labels(y, activations[-1])  # Validierung der Labels
            loss = mean_squared_error(y, activations[-1])
            total_loss += loss

            if use_backpropagation:
                weight_gradients, bias_gradients = self.backpropagation(activations, z_values, y)
                self.update_weights(weight_gradients, bias_gradients, t)
        return total_loss / len(X_batch)



    def adjust_learning_rate(self, epoch, initial_lr, decay_rate=0.9):
        """Passt die Lernrate dynamisch an."""
        self.learning_rate = initial_lr * (decay_rate ** epoch)
        self.optimizer.learning_rate = self.learning_rate  # Update Optimizer

    def train(self, X_train, y_train, epochs, batch_size, use_backpropagation=True, learning_rate_decay=False, early_stopping_patience=None):
        """Trainiert das Netzwerk."""
        history = {'loss': []}
        start_time = time.time()
        initial_lr = self.learning_rate
        num_batches = len(X_train) // batch_size

        for epoch in range(epochs):
            epoch_start_time = time.time()
            total_loss = 0
            shuffled_indices = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[shuffled_indices]
            y_train_shuffled = y_train[shuffled_indices]

            for i in range(num_batches):
                start_index = i * batch_size
                end_index = (i + 1) * batch_size
                X_batch = X_train_shuffled[start_index:end_index]
                y_batch = y_train_shuffled[start_index:end_index]

                batch_loss = self.train_step(X_batch, y_batch, use_backpropagation, epoch + 1)
                total_loss += batch_loss

            avg_loss = total_loss / num_batches
            history['loss'].append(avg_loss)

            logger.info(f"Epoch: {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}")

            # Dynamische Lernratenanpassung
            if learning_rate_decay:
                self.adjust_learning_rate(epoch, initial_lr)
                logger.debug(f"Learning Rate angepasst: {self.learning_rate:.4f}")

            # Early Stopping
            if early_stopping_patience and early_stopping(history, early_stopping_patience):
                logger.info("Frühzeitiges Beenden des Trainings aufgrund von Stagnation im Verlust.")
                break

            epoch_end_time = time.time()
            logger.debug(f"Epoch {epoch + 1}/{epochs}  Time: {epoch_end_time - epoch_start_time:.2f}s")

        end_time = time.time()
        logger.info(f"Training abgeschlossen. Gesamte Trainingszeit: {end_time - start_time:.2f}s")
        return history

    def log_neuron_activations(self, inputs):
        """Loggt die Aktivierungen der Neuronen für die angegebenen Eingaben."""
        for i, input_data in enumerate(inputs):
            activations, _ = self.feedforward(input_data)
            for layer_idx, activation in enumerate(activations[1:]):
                if layer_idx == len(activations[1:]) - 1:
                    logger.debug(f"Input {i + 1}, Output Layer: Activations: {activation}")
                else:
                    logger.debug(f"Input {i + 1}, Layer {layer_idx + 1}: Activations: {activation}")

    def predict(self, X):
        """Gibt die Vorhersagen des Netzwerks für die Eingabedaten zurück."""
        predictions = []
        for x in X:
            activations, _ = self.feedforward(x)
            predictions.append(activations[-1])
        predictions = np.array(predictions)

        # Reduzieren auf 2D (n_samples, 1) für Regression
        if self.output_activation == "sigmoid" and predictions.ndim == 3:
            predictions = predictions.squeeze(axis=-1)
        elif predictions.ndim == 1:  # Wenn 1D, in (n_samples, 1) umwandeln
            predictions = predictions.reshape(-1, 1)
        return predictions

    def test_network(self, test_data):
        """Testet die Leistung des Netzwerks."""
        X_test, y_test = zip(*test_data)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        predictions = self.predict(X_test)

        for y_true, y_pred in zip(y_test, predictions):
            self.validate_labels(y_true, y_pred)

        if self.output_activation == "softmax":
            predicted_labels = np.argmax(predictions, axis=1)
            true_labels = np.argmax(y_test, axis=1)
            return {'accuracy': accuracy_score(true_labels, predicted_labels)}
        else:
            mse = mean_squared_error(y_test, predictions)
            return {'mean_squared_error': mse}

    def visualize(self):
        """Visualisiert das Netzwerk."""
        if len(self.num_nodes) < 3:
            logger.warning("Keine Visualisierung, da es zu wenig Layer gibt")
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
                for i in range(len(prev_y)):
                    for j in range(len(y)):
                        plt.plot([prev_x[i], x[j]], [prev_y[i], y[j]], color='gray', alpha=0.2, zorder=1)  # Linien zwischen Neuronen
        plt.title("Netzwerk Architektur")
        plt.xlabel("Layer")
        plt.ylabel("Neuron Position")
        plt.yticks([])
        plt.xticks(range(len(layer_sizes)), labels=[f'Layer {i+1}' for i in range(len(layer_sizes))])
        plt.legend()
        plt.show()

    def save_network(self, filepath):
        """Speichert das Netzwerk als Pickle-Datei."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Netzwerk gespeichert unter: {filepath}")

    @staticmethod
    def load_network(filepath):
        """Lädt ein Netzwerk aus einer Pickle-Datei."""
        with open(filepath, 'rb') as f:
            network = pickle.load(f)
        logger.info(f"Netzwerk geladen von: {filepath}")
        return network

    def to_json(self, filepath):
        """Speichert das Netzwerk als JSON-Datei."""
        network_data = {
            'num_nodes': self.num_nodes,
            'V_max': self.V_max,
            'R_harmony': self.R_harmony,
            'activation_function': self.activation_function,
            'learning_rate': self.learning_rate,
            'momentum': self.momentum,
            'output_activation': self.output_activation,
            'optimizer_method': self.optimizer_method,
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases]
        }
        with open(filepath, 'w') as f:
            json.dump(network_data, f, indent=4)
        logger.info(f"Netzwerk als JSON gespeichert unter: {filepath}")

    @staticmethod
    def from_json(filepath):
        """Lädt ein Netzwerk aus einer JSON-Datei."""
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
        network.weights = [np.array(w, dtype=np.float32) for w in network_data['weights']]
        network.biases = [np.array(b, dtype=np.float32) for b in network_data['biases']]
        logger.info(f"Netzwerk von JSON geladen von: {filepath}")
        return network

    def plot_training_history(self, history):
        """Visualisiert den Lernverlauf (Loss und Genauigkeit)."""
        epochs = range(1, len(history['loss']) + 1)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history['loss'], 'b-o', label='Loss')
        plt.title('Verlust über die Epochen')
        plt.xlabel('Epoche')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, test_data):
        """Erstellt die Konfusionsmatrix für die Klassifikation."""
        if self.output_activation != "softmax":
            logger.warning("Konfusionsmatrix nur für Klassifikation.")
            return

        X_test, y_test = zip(*test_data)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        predictions = self.predict(X_test)
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(y_test, axis=1)

        plot_confusion_matrix(true_labels, predicted_labels, labels=list(range(y_test.shape[1])))

    def save_trial_results(self, filepath):
        """Speichert die Ergebnisse der verschiedenen Trials als CSV."""
        if not self.trial_results:
            logger.warning("Keine Trainingsergebnisse zum Speichern vorhanden.")
            return

        fieldnames = self.trial_results[0].keys()
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.trial_results)
        logger.info(f"Trainingsergebnisse gespeichert in: {filepath}")

    def run_trials(self, X_train, y_train, X_test, y_test, num_trials=5, epochs=10, batch_size=32, use_backpropagation=True, learning_rate_decay=False, early_stopping_patience=None, hyperparameter_space={}):
        """Führt mehrere Trainingsläufe mit unterschiedlichen Hyperparametern durch."""

        if not hyperparameter_space:
            hyperparameter_space = {
                'learning_rate': [0.01, 0.001],
                'momentum': [0.9, 0.99],
                'optimizer_method': ['SGD', 'Adam']
            }
        keys = hyperparameter_space.keys()
        values = hyperparameter_space.values()
        combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
        logger.info(f"Anzahl der Hyperparameter-Kombinationen: {len(combinations)}")

        for i, params in enumerate(combinations):
            logger.info(f"Starting Trial {i+1} with params: {params}")
            trial_start_time = time.time()
            # Setze die Hyperparameter für diese Trial
            
            
            
            if 'learning_rate' in params and self.learning_rate != params['learning_rate']:
                self.learning_rate = params['learning_rate']
                
            if 'momentum' in params and self.momentum != params['momentum']:
                self.momentum = params['momentum']
                
            if 'optimizer_method' in params and self.optimizer_method != params['optimizer_method']:
                self.optimizer_method = params['optimizer_method']
            
            
            self.optimizer = Optimizer(method=self.optimizer_method, learning_rate=self.learning_rate, momentum=self.momentum)
            self.optimizer.initialize(self.weights, self.biases)

            # Trainiere das Netzwerk
            history = self.train(X_train, y_train, epochs, batch_size, use_backpropagation, learning_rate_decay, early_stopping_patience)

            # Teste das Netzwerk
            test_data = list(zip(X_test, y_test))
            metrics = self.test_network(test_data)

            trial_end_time = time.time()
            trial_duration = trial_end_time - trial_start_time
            # Speichere die Ergebnisse
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

            if (i+1) % 2 == 0:
                self.plot_training_history(history)
                self.plot_confusion_matrix(test_data)

        logger.info("Alle Trials abgeschlossen.")

def create_dummy_data(num_samples=1000, num_features=10, num_classes=2):
    """Erzeugt Dummy-Daten für Tests."""
    X = np.random.rand(num_samples, num_features).astype(np.float32)
    if num_classes > 2:
        y = np.random.randint(0, num_classes, num_samples)
        y = np.eye(num_classes)[y]  # One-Hot-Encoding
    else:
        y = np.random.randint(0, num_classes, num_samples).astype(np.float32).reshape(-1, 1)
    return X, y


def main():
    # Test der Data Loading Funktionen
    X, y = create_dummy_data(num_samples=1000, num_features=10, num_classes=2)

    # Speichern und Laden von Dummy-Daten
    save_dataset_to_csv(X, y, 'dummy_data.csv')
    X_train, X_test, y_train, y_test = load_data_from_csv('dummy_data.csv', target_column='label')

    # Überprüfung auf NaN
    check_for_nan(X_train)
    check_for_nan(y_train)

    # Dynamische Anpassung der Netzwerkarchitektur
    input_dim = X_train.shape[1]  # Anzahl der Eingabefeatures
    num_nodes = [input_dim, 20, 1]  # Beispiel: 1 versteckte Schicht mit 20 Neuronen, 1 Ausgabeneuron

    # Erstellung und Training eines Netzwerks
    network = Network(num_nodes, V_max=0.5, R_harmony=0.1, activation_function='relu', learning_rate=0.01, momentum=0.9, output_activation='sigmoid', optimizer_method='Adam')
    network.validate_dimensions(X_train, y_train)  # Validierung der Dimensionen
    history = network.train(X_train, y_train, epochs=10, batch_size=32, use_backpropagation=True, learning_rate_decay=True, early_stopping_patience=3)
    network.plot_training_history(history)

    # Testen des Netzwerks
    test_data = list(zip(X_test, y_test))
    metrics = network.test_network(test_data)
    logger.info(f"Metrics nach dem Training: {metrics}")

    # Visualisieren des Netzwerks
    network.visualize()

    # Loggen der Neuronaktivierungen (Beispiel)
    network.log_neuron_activations(X_test[:5])  # Erste 5 Testdatenpunkte

    # Konfusionsmatrix anzeigen
    network.plot_confusion_matrix(test_data)

    # Speichern und Laden des Netzwerks
    network.save_network('my_network.pkl')
    loaded_network = Network.load_network('my_network.pkl')
    loaded_metrics = loaded_network.test_network(test_data)
    logger.info(f"Geladenes Netzwerk Metrics: {loaded_metrics}")

    # JSON Speichern und Laden
    network.to_json("my_network.json")
    loaded_network_json = Network.from_json("my_network.json")
    loaded_metrics_json = loaded_network_json.test_network(test_data)
    logger.info(f"Geladenes JSON Netzwerk Metrics: {loaded_metrics_json}")


    # Durchführung von Trials mit Hyperparameter Tuning
    X, y = create_dummy_data(num_samples=1000, num_features=10, num_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Beispiel Hyperparameter Space
    hyperparameter_space = {
        'learning_rate': [0.01, 0.001, 0.0001],
        'momentum': [0.9, 0.8, 0.7],
        'optimizer_method': ['SGD', 'Adam', 'RMSprop']
    }

    network.run_trials(X_train, y_train, X_test, y_test, epochs=5, batch_size=32, use_backpropagation=True, learning_rate_decay=True, early_stopping_patience=2, hyperparameter_space=hyperparameter_space)
    network.save_trial_results('trial_results.csv')


if __name__ == '__main__':
    main()
