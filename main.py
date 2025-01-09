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

# Konfiguration des Loggings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Globale Konfigurationen
RANDOM_STATE = 42

class Optimizer:
    """
    Basisklasse für Optimierer. Definiert die grundlegenden Parameter und Methoden,
    die von spezifischen Optimierern wie SGD, Adam und RMSprop implementiert werden müssen.
    """
    def __init__(self, learning_rate=0.01, momentum=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initialisiert den Optimierer mit den gegebenen Hyperparametern.

        Parameters:
            learning_rate (float): Lernrate des Optimierers.
            momentum (float): Momentum-Faktor für Optimierer, die Momentum verwenden.
            beta1 (float): Erstes Momenten-Vektor für Adam.
            beta2 (float): Zweites Momenten-Vektor für Adam.
            epsilon (float): Kleine Zahl zur Vermeidung von Division durch Null.
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_velocities = None
        self.bias_velocities = None
        self.m_weights = None  # Adam: Erste Momenten-Schätzung für Gewichte
        self.v_weights = None  # Adam: Zweite Momenten-Schätzung für Gewichte
        self.m_biases = None   # Adam: Erste Momenten-Schätzung für Biases
        self.v_biases = None   # Adam: Zweite Momenten-Schätzung für Biases

    def initialize(self, weights, biases):
        """
        Initialisiert die Geschwindigkeits- und Momentenschätzungen basierend auf den Gewichtungs- und Bias-Arrays.

        Parameters:
            weights (list of np.ndarray): Liste der Gewichtungsmatrizen des Netzwerks.
            biases (list of np.ndarray): Liste der Bias-Vektoren des Netzwerks.
        """
        self.weight_velocities = [np.zeros_like(w, dtype=np.float32) for w in weights]
        self.bias_velocities = [np.zeros_like(b, dtype=np.float32) for b in biases]
        self.m_weights = [np.zeros_like(w, dtype=np.float32) for w in weights]
        self.v_weights = [np.zeros_like(w, dtype=np.float32) for w in weights]
        self.m_biases = [np.zeros_like(b, dtype=np.float32) for b in biases]
        self.v_biases = [np.zeros_like(b, dtype=np.float32) for b in biases]

    def update(self, weights, biases, weight_gradients, bias_gradients, t=None):
        """
        Aktualisiert die Gewichte und Biases basierend auf den Gradienten.
        Muss von Unterklassen implementiert werden.

        Parameters:
            weights (list of np.ndarray): Aktuelle Gewichte des Netzwerks.
            biases (list of np.ndarray): Aktuelle Biases des Netzwerks.
            weight_gradients (list of np.ndarray): Gradienten der Gewichte.
            bias_gradients (list of np.ndarray): Gradienten der Biases.
            t (int, optional): Iterationsschritt, relevant für Adam.

        Raises:
            NotImplementedError: Wenn die Methode nicht von einer Unterklasse implementiert wird.
        """
        raise NotImplementedError("Subclasses should implement this!")

class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) Optimierer mit Momentum.
    """
    def update(self, weights, biases, weight_gradients, bias_gradients, t=None):
        """
        Aktualisiert die Gewichte und Biases mittels SGD mit Momentum.

        Parameters:
            weights (list of np.ndarray): Aktuelle Gewichte des Netzwerks.
            biases (list of np.ndarray): Aktuelle Biases des Netzwerks.
            weight_gradients (list of np.ndarray): Gradienten der Gewichte.
            bias_gradients (list of np.ndarray): Gradienten der Biases.
            t (int, optional): Iterationsschritt, nicht verwendet in SGD.

        Returns:
            tuple: Aktualisierte Gewichte und Biases.
        """
        updated_weights = []
        updated_biases = []
        for i in range(len(weights)):
            # Update der Gewichtsgeschwindigkeit
            self.weight_velocities[i] = self.momentum * self.weight_velocities[i] - self.learning_rate * weight_gradients[i]
            # Aktualisiere die Gewichte
            updated_weights.append(weights[i] + self.weight_velocities[i])
            # Update der Bias-Geschwindigkeit
            self.bias_velocities[i] = self.momentum * self.bias_velocities[i] - self.learning_rate * bias_gradients[i]
            # Aktualisiere die Biases
            updated_biases.append(biases[i] + self.bias_velocities[i])
        return updated_weights, updated_biases

class Adam(Optimizer):
    """
    Adam Optimierer, eine adaptive Lernratenoptimierungsmethode.
    """
    def update(self, weights, biases, weight_gradients, bias_gradients, t=None):
        """
        Aktualisiert die Gewichte und Biases mittels Adam-Algorithmus.

        Parameters:
            weights (list of np.ndarray): Aktuelle Gewichte des Netzwerks.
            biases (list of np.ndarray): Aktuelle Biases des Netzwerks.
            weight_gradients (list of np.ndarray): Gradienten der Gewichte.
            bias_gradients (list of np.ndarray): Gradienten der Biases.
            t (int, optional): Iterationsschritt für Bias-Korrektur.

        Returns:
            tuple: Aktualisierte Gewichte und Biases.
        """
        updated_weights = []
        updated_biases = []
        for i in range(len(weights)):
            # Update der ersten und zweiten Momenten für Gewichte
            self.m_weights[i] = self.beta1 * self.m_weights[i] + (1 - self.beta1) * weight_gradients[i]
            self.v_weights[i] = self.beta2 * self.v_weights[i] + (1 - self.beta2) * weight_gradients[i]**2
            # Bias-Korrektur
            m_hat = self.m_weights[i] / (1 - self.beta1**t)
            v_hat = self.v_weights[i] / (1 - self.beta2**t)
            # Aktualisierung der Gewichte
            updated_weights.append(weights[i] - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon))
            
            # Update der ersten und zweiten Momenten für Biases
            self.m_biases[i] = self.beta1 * self.m_biases[i] + (1 - self.beta1) * bias_gradients[i]
            self.v_biases[i] = self.beta2 * self.v_biases[i] + (1 - self.beta2) * bias_gradients[i]**2
            # Bias-Korrektur für Biases
            m_hat_b = self.m_biases[i] / (1 - self.beta1**t)
            v_hat_b = self.v_biases[i] / (1 - self.beta2**t)
            # Aktualisierung der Biases
            updated_biases.append(biases[i] - self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon))
        return updated_weights, updated_biases

class RMSprop(Optimizer):
    """
    RMSprop Optimierer, eine weitere adaptive Lernratenoptimierungsmethode.
    """
    def update(self, weights, biases, weight_gradients, bias_gradients, t=None):
        """
        Aktualisiert die Gewichte und Biases mittels RMSprop-Algorithmus.

        Parameters:
            weights (list of np.ndarray): Aktuelle Gewichte des Netzwerks.
            biases (list of np.ndarray): Aktuelle Biases des Netzwerks.
            weight_gradients (list of np.ndarray): Gradienten der Gewichte.
            bias_gradients (list of np.ndarray): Gradienten der Biases.
            t (int, optional): Iterationsschritt, nicht verwendet in RMSprop.

        Returns:
            tuple: Aktualisierte Gewichte und Biases.
        """
        updated_weights = []
        updated_biases = []
        for i in range(len(weights)):
            # Update der zweiten Momenten für Gewichte
            self.v_weights[i] = self.beta2 * self.v_weights[i] + (1 - self.beta2) * weight_gradients[i]**2
            # Aktualisierung der Gewichte
            updated_weights.append(weights[i] - self.learning_rate * weight_gradients[i] / (np.sqrt(self.v_weights[i]) + self.epsilon))
            
            # Update der zweiten Momenten für Biases
            self.v_biases[i] = self.beta2 * self.v_biases[i] + (1 - self.beta2) * bias_gradients[i]**2
            # Aktualisierung der Biases
            updated_biases.append(biases[i] - self.learning_rate * bias_gradients[i] / (np.sqrt(self.v_biases[i]) + self.epsilon))
        return updated_weights, updated_biases

def load_data_from_csv(filepath, target_column='label', test_size=0.2, random_state=42, scale_features=True):
    """
    Lädt Daten aus einer CSV-Datei, verarbeitet die Zielspalte und teilt die Daten in Trainings- und Testsets auf.

    Parameters:
        filepath (str): Pfad zur CSV-Datei.
        target_column (str): Name der Zielspalte.
        test_size (float): Anteil der Daten für das Testset.
        random_state (int): Zufallsstate für die Reproduzierbarkeit.
        scale_features (bool): Ob die Features normalisiert werden sollen.

    Returns:
        tuple: X_train, X_test, y_train, y_test nach der Aufteilung.
    """
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

    # Fehlende Werte behandeln
    if np.isnan(X).any():
        X = np.nan_to_num(X)
    if np.isnan(y).any():
        y = np.nan_to_num(y)

    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def test_label_shapes():
    """
    Testet, ob die Labels die erwartete Form haben.
    """
    X, y = create_dummy_data(num_samples=100, num_features=10, num_classes=2)
    assert y.shape[1] == 1, f"Erwartete Form (100, 1), aber erhalten: {y.shape}"

def save_dataset_to_csv(X, y, filepath):
    """
    Speichert Features und Labels in einer CSV-Datei.

    Parameters:
        X (np.ndarray): Feature-Matrix.
        y (np.ndarray): Labels.
        filepath (str): Pfad zur Ausgabedatei.
    """
    # Falls y One-Hot-Encoded ist, konvertiere es zu eindimensionalen Labels
    if len(y.shape) > 1 and y.shape[1] > 1:
        y = np.argmax(y, axis=1)  # Maximalwert (Index) für jede Zeile nehmen

    data = pd.DataFrame(X)
    data['label'] = y
    data.to_csv(filepath, index=False)
    logger.info(f"Datensatz gespeichert in {filepath}")

def check_for_nan(data):
    """
    Überprüft, ob in den Daten fehlende Werte vorhanden sind.

    Parameters:
        data (np.ndarray or list): Datenstruktur zum Überprüfen.

    Raises:
        ValueError: Wenn fehlende Werte (NaN) gefunden werden.
        TypeError: Wenn der Datentyp nicht unterstützt wird.
    """
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
    """
    Sicherstellt, dass bei exponentiellen Operationen kein Overflow auftritt.

    Parameters:
        x (float): Eingabewert.

    Returns:
        float: Ergebnis der exponentiellen Operation oder Inf/0.
    """
    try:
        return np.exp(x)
    except OverflowError:
        return np.inf if x > 0 else 0.0

def sigmoid(x):
    """
    Sigmoid-Aktivierungsfunktion mit Schutz vor Overflow.

    Parameters:
        x (np.ndarray): Eingabewerte.

    Returns:
        np.ndarray: Sigmoid-Ausgabe.
    """
    clipped_x = np.clip(x, -500, 500)  # Eingabewerte begrenzen
    return 1 / (1 + np.exp(-clipped_x))

def relu(x):
    """
    ReLU-Aktivierungsfunktion.

    Parameters:
        x (np.ndarray): Eingabewerte.

    Returns:
        np.ndarray: ReLU-Ausgabe.
    """
    return np.maximum(0, x)

def softmax(x):
    """
    Softmax-Aktivierungsfunktion.

    Parameters:
        x (np.ndarray): Eingabewerte.

    Returns:
        np.ndarray: Softmax-Ausgabe.
    """
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def sigmoid_derivative(x):
    """
    Ableitung der Sigmoid-Funktion.

    Parameters:
        x (np.ndarray): Eingabewerte.

    Returns:
        np.ndarray: Ableitung der Sigmoid-Funktion.
    """
    s = sigmoid(x)
    return s * (1 - s)

def relu_derivative(x):
    """
    Ableitung der ReLU-Funktion.

    Parameters:
        x (np.ndarray): Eingabewerte.

    Returns:
        np.ndarray: Ableitung der ReLU-Funktion.
    """
    return np.where(x > 0, 1, 0).astype(float)

def softmax_derivative(x):
    """
    Ableitung der Softmax-Funktion.

    Parameters:
        x (np.ndarray): Eingabewerte.

    Returns:
        np.ndarray: Jacobian-Matrix der Softmax-Funktion.
    """
    s = softmax(x)
    jacobian_matrix = np.diag(s.flatten()) - np.outer(s, s)
    return jacobian_matrix

def plot_confusion_matrix(y_true, y_pred, labels):
    """
    Erstellt und zeigt die Konfusionsmatrix.

    Parameters:
        y_true (array-like): Wahre Labels.
        y_pred (array-like): Vorhergesagte Labels.
        labels (list): Liste der Label-Namen.
    """
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=labels)
    plt.show()

def early_stopping(history, patience=5):
    """
    Prüft, ob das Training frühzeitig beendet werden soll basierend auf der Stagnation des Verlusts.

    Parameters:
        history (dict): Verlauf der Trainingsmetriken.
        patience (int): Anzahl der Epochen ohne Verbesserung, nach denen gestoppt wird.

    Returns:
        bool: True, wenn frühzeitig gestoppt werden soll, sonst False.
    """
    if len(history['loss']) > patience and all(history['loss'][-i] >= history['loss'][-i-1] for i in range(1, patience+1)):
        return True
    return False

class Network:
    """
    Klasse zur Darstellung eines einfachen neuronalen Netzwerks mit Trainings- und Testfunktionen,
    sowie Speichermethoden für JSON und Pickle.
    """
    def __init__(self, num_nodes, V_max, R_harmony, activation_function='sigmoid', learning_rate=0.1, momentum=0.9, output_activation='sigmoid', optimizer_method='SGD'):
        """
        Initialisiert das Netzwerk mit den gegebenen Hyperparametern und initialisiert die Gewichte und Biases.

        Parameters:
            num_nodes (list of int): Anzahl der Neuronen in jedem Layer (inkl. Eingabe und Ausgabe).
            V_max (float): Maximale Spannung oder Aktivierungsgrenze (spezifisch für das Modell).
            R_harmony (float): Harmoniefaktor (spezifisch für das Modell).
            activation_function (str): Aktivierungsfunktion für die versteckten Layer ('sigmoid' oder 'relu').
            learning_rate (float): Lernrate für den Optimierer.
            momentum (float): Momentum-Faktor für den Optimierer.
            output_activation (str): Aktivierungsfunktion für den Ausgabelayer ('sigmoid' oder 'softmax').
            optimizer_method (str): Optimierer-Methode ('SGD', 'Adam' oder 'RMSprop').

        Raises:
            ValueError: Wenn eine nicht unterstützte Aktivierungsfunktion gewählt wird.
        """
        self.num_nodes = num_nodes
        self.V_max = V_max
        self.R_harmony = R_harmony
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.output_activation = output_activation
        self.optimizer_method = optimizer_method

        # Initialisierung der Gewichte und Biases
        self.weights = [np.random.uniform(-1, 1, size=(num_nodes[i], num_nodes[i+1])).astype(np.float32) for i in range(len(num_nodes) - 1)]
        self.biases = [np.zeros(nodes, dtype=np.float32) for nodes in num_nodes[1:]]

        # Auswahl und Initialisierung des Optimierers
        self.optimizer = self.get_optimizer(optimizer_method, learning_rate, momentum)
        self.optimizer.initialize(self.weights, self.biases)

        # Auswahl der Aktivierungsfunktion und ihrer Ableitung
        if activation_function == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation_function == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        else:
            raise ValueError("Aktivierungsfunktion nicht unterstützt.")

        # Auswahl der Ausgabeschicht-Aktivierungsfunktion und ihrer Ableitung
        if output_activation == 'sigmoid':
            self.output_activation_func = sigmoid
            self.output_activation_derivative = sigmoid_derivative
        elif output_activation == 'softmax':
            self.output_activation_func = softmax
            self.output_activation_derivative = softmax_derivative
        else:
            raise ValueError("Output Aktivierungsfunktion nicht unterstützt.")

        self.trial_results = []       # Speicherung der Ergebnisse verschiedener Trials
        self.metrics_history = {}     # Verlauf der Metriken

    def get_optimizer(self, method, learning_rate, momentum):
        """
        Gibt den entsprechenden Optimierer basierend auf der Methode zurück.

        Parameters:
            method (str): Name des Optimierers ('SGD', 'Adam', 'RMSprop').
            learning_rate (float): Lernrate.
            momentum (float): Momentum-Faktor.

        Returns:
            Optimizer: Instanz des gewählten Optimierers.

        Raises:
            ValueError: Wenn eine nicht unterstützte Optimierer-Methode gewählt wird.
        """
        if method == 'SGD':
            return SGD(learning_rate=learning_rate, momentum=momentum)
        elif method == 'Adam':
            return Adam(learning_rate=learning_rate)
        elif method == 'RMSprop':
            return RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError("Optimierer nicht unterstützt.")

    def validate_dimensions(self, x, y):
        """
        Überprüft, ob die Dimensionen von x und y mit den Netzwerkparametern übereinstimmen.

        Parameters:
            x (np.ndarray): Eingabedaten.
            y (np.ndarray): Zielwerte.

        Raises:
            ValueError: Wenn die Dimensionen nicht übereinstimmen.
        """
        if x.shape[1] != self.num_nodes[0]:
            raise ValueError(f"Die Eingabedimension {x.shape[1]} stimmt nicht mit der erwarteten {self.num_nodes[0]} überein.")
        if y.ndim == 1 and self.num_nodes[-1] > 1:
            raise ValueError(f"Die Zielgröße {y.shape} passt nicht zu den Ausgabeneuronen {self.num_nodes[-1]}.")
        if y.ndim > 1 and y.shape[1] != self.num_nodes[-1]:
            raise ValueError(f"Die Zielgröße {y.shape[1]} passt nicht zu den Ausgabeneuronen {self.num_nodes[-1]}.")

    def feedforward(self, input_data):
        """
        Führt einen Feedforward-Pass durch das Netzwerk.

        Parameters:
            input_data (np.ndarray): Eingabedaten für das Netzwerk.

        Returns:
            tuple: Liste der Aktivierungen pro Layer und Liste der z-Werte (net inputs) pro Layer.
        """
        input_data = input_data.reshape(1, -1)  # Sicherstellen, dass die Eingabe die richtige Form hat
        activations = [input_data]
        z_values = []

        for i in range(len(self.weights)):
            # Berechnung des net inputs (z) für das aktuelle Layer
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            z_values.append(z)

            # Anwendung der Aktivierungsfunktion
            if i == len(self.weights) - 1:  # Letzte Schicht
                activation = self.output_activation_func(z)
            else:
                activation = self.activation(z)
            activations.append(activation)

        return activations, z_values

    def backpropagation(self, activations, z_values, y):
        """
        Führt die Backpropagation durch, um die Gradienten der Gewichte und Biases zu berechnen.

        Parameters:
            activations (list of np.ndarray): Aktivierungen pro Layer.
            z_values (list of np.ndarray): net inputs pro Layer.
            y (np.ndarray): Zielwerte.

        Returns:
            tuple: Gradienten der Gewichte und Biases.
        """
        # Berechnung des Fehlers in der Ausgabeschicht
        delta = (activations[-1] - y) * self.output_activation_derivative(z_values[-1])
        weight_gradients = []
        bias_gradients = []

        # Rückwärts durch die Layer iterieren
        for i in reversed(range(len(self.weights))):
            if activations[i].ndim == 1:
                activations_t = activations[i].reshape(-1, 1)
            else:
                activations_t = activations[i].T
            # Berechnung der Gradienten für Gewichte und Biases
            weight_gradient = np.dot(activations_t, delta)
            bias_gradient = np.sum(delta, axis=0)

            # Einfügen der Gradienten an der Anfang der Listen
            weight_gradients.insert(0, weight_gradient)
            bias_gradients.insert(0, bias_gradient)

            if i > 0:
                # Berechnung des Fehlers für das vorherige Layer
                delta = np.dot(delta, self.weights[i].T) * self.activation_derivative(z_values[i-1])

        return weight_gradients, bias_gradients

    def update_weights(self, weight_gradients, bias_gradients, t=None):
        """
        Aktualisiert die Gewichte und Biases mithilfe des ausgewählten Optimierers.

        Parameters:
            weight_gradients (list of np.ndarray): Gradienten der Gewichte.
            bias_gradients (list of np.ndarray): Gradienten der Biases.
            t (int, optional): Iterationsschritt, relevant für bestimmte Optimierer wie Adam.
        """
        updated_weights, updated_biases = self.optimizer.update(self.weights, self.biases, weight_gradients, bias_gradients, t)
        self.weights = updated_weights
        self.biases = updated_biases

    def validate_labels(self, y_true, y_pred):
        """
        Prüft, ob Labels und Vorhersagen im richtigen Format vorliegen.

        Parameters:
            y_true (np.ndarray or list): Wahre Labels.
            y_pred (np.ndarray or list): Vorhergesagte Labels.

        Raises:
            ValueError: Wenn die Labels nicht array-ähnlich sind.
        """
        if not isinstance(y_true, (np.ndarray, list)):
            raise ValueError(f"'y_true' muss ein array-ähnliches Objekt sein. Erhalten: {y_true}")
        if not isinstance(y_pred, (np.ndarray, list)):
            raise ValueError(f"'y_pred' muss ein array-ähnliches Objekt sein. Erhalten: {y_pred}")

    def train_step(self, X_batch, y_batch, use_backpropagation=True, t=None):
        """
        Führt einen Trainingsschritt über einen Batch von Daten durch.

        Parameters:
            X_batch (np.ndarray): Batch von Eingabedaten.
            y_batch (np.ndarray): Batch von Zielwerten.
            use_backpropagation (bool): Ob Backpropagation verwendet werden soll.
            t (int, optional): Iterationsschritt für den Optimierer.

        Returns:
            float: Durchschnittlicher Verlust des Batches.
        """
        total_loss = 0
        for x, y in zip(X_batch, y_batch):
            x = x.reshape(1, -1)  # Eingabe in die richtige Form bringen

            # Zielwerte überprüfen und anpassen
            if isinstance(y, (int, float)):
                y = np.array([[y]], dtype=np.float32)
            elif isinstance(y, np.ndarray) and y.ndim == 1:
                y = y.reshape(1, -1)
            
            # Forward Pass
            activations, z_values = self.feedforward(x)
            self.validate_labels(y, activations[-1])  # Validierung der Labels
            # Berechnung des Verlusts (Mean Squared Error)
            loss = mean_squared_error(y, activations[-1])
            total_loss += loss

            if use_backpropagation:
                # Backward Pass
                weight_gradients, bias_gradients = self.backpropagation(activations, z_values, y)
                # Update der Gewichte und Biases
                self.update_weights(weight_gradients, bias_gradients, t)
        return total_loss / len(X_batch)

    def adjust_learning_rate(self, epoch, initial_lr, decay_rate=0.9):
        """
        Passt die Lernrate dynamisch an, basierend auf der aktuellen Epoche.

        Parameters:
            epoch (int): Aktuelle Epoche.
            initial_lr (float): Anfangs-Lernrate.
            decay_rate (float): Abnahmerate der Lernrate pro Epoche.
        """
        self.learning_rate = initial_lr * (decay_rate ** epoch)
        self.optimizer.learning_rate = self.learning_rate  # Update Optimizer

    def train(self, X_train, y_train, epochs, batch_size, use_backpropagation=True, learning_rate_decay=False, early_stopping_patience=None):
        """
        Trainiert das neuronale Netzwerk über die angegebenen Epochen und Batches.

        Parameters:
            X_train (np.ndarray): Trainings-Eingabedaten.
            y_train (np.ndarray): Trainings-Zielwerte.
            epochs (int): Anzahl der Epochen zum Trainieren.
            batch_size (int): Größe der Batches.
            use_backpropagation (bool): Ob Backpropagation verwendet werden soll.
            learning_rate_decay (bool): Ob die Lernrate dynamisch angepasst werden soll.
            early_stopping_patience (int, optional): Anzahl der Epochen ohne Verbesserung für frühzeitiges Stoppen.

        Returns:
            dict: Verlauf der Trainingsmetriken (z.B. Verlust).
        """
        history = {'loss': []}
        start_time = time.time()
        initial_lr = self.learning_rate
        num_batches = len(X_train) // batch_size

        for epoch in range(epochs):
            epoch_start_time = time.time()
            total_loss = 0
            # Shuffling der Trainingsdaten
            shuffled_indices = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[shuffled_indices]
            y_train_shuffled = y_train[shuffled_indices]

            for i in range(num_batches):
                start_index = i * batch_size
                end_index = (i + 1) * batch_size
                X_batch = X_train_shuffled[start_index:end_index]
                y_batch = y_train_shuffled[start_index:end_index]

                # Durchführung eines Trainingsschritts
                batch_loss = self.train_step(X_batch, y_batch, use_backpropagation, epoch + 1)
                total_loss += batch_loss

            # Durchschnittlichen Verlust für die Epoche berechnen
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
        """
        Loggt die Aktivierungen der Neuronen für die angegebenen Eingaben.

        Parameters:
            inputs (list of np.ndarray): Liste von Eingabedaten.
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
        Gibt die Vorhersagen des Netzwerks für die Eingabedaten zurück.

        Parameters:
            X (np.ndarray): Eingabedaten.

        Returns:
            np.ndarray: Vorhersagen des Netzwerks.
        """
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
        """
        Testet die Leistung des Netzwerks anhand der Testdaten.

        Parameters:
            test_data (list of tuples): Liste von (Eingabe, Zielwert) Paaren.

        Returns:
            dict: Metriken der Netzwerkleistung (z.B. Accuracy, MSE).
        """
        X_test, y_test = zip(*test_data)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        predictions = self.predict(X_test)

        # Validierung der Labels
        for y_true, y_pred in zip(y_test, predictions):
            self.validate_labels(y_true, y_pred)

        if self.output_activation == "softmax":
            # Klassifikation: Berechnung der Genauigkeit
            predicted_labels = np.argmax(predictions, axis=1)
            true_labels = np.argmax(y_test, axis=1)
            return {'accuracy': accuracy_score(true_labels, predicted_labels)}
        else:
            # Regression: Berechnung von MSE, MAE und R²-Score
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            return {'mean_squared_error': mse, 'mean_absolute_error': mae, 'r2_score': r2}

    def visualize(self):
        """
        Visualisiert die Architektur des Netzwerks, einschließlich der Verbindungen zwischen den Neuronen.
        """
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
                # Zeichne Verbindungen zwischen Neuronen der vorherigen und aktuellen Schicht
                for i in range(len(prev_y)):
                    for j in range(len(y)):
                        plt.plot([prev_x[i], x[j]], [prev_y[i], y[j]], color='gray', alpha=0.2, zorder=1)
        plt.title("Netzwerk Architektur")
        plt.xlabel("Layer")
        plt.ylabel("Neuron Position")
        plt.yticks([])
        plt.xticks(range(len(layer_sizes)), labels=[f'Layer {i+1}' for i in range(len(layer_sizes))])
        plt.legend()
        plt.show()

    def save_network(self, filepath):
        """
        Speichert das Netzwerk als Pickle-Datei.

        Parameters:
            filepath (str): Pfad zur Ausgabedatei.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Netzwerk gespeichert unter: {filepath}")

    @staticmethod
    def load_network(filepath):
        """
        Lädt ein Netzwerk aus einer Pickle-Datei.

        Parameters:
            filepath (str): Pfad zur Pickle-Datei.

        Returns:
            Network: Geladenes Netzwerk-Objekt.
        """
        with open(filepath, 'rb') as f:
            network = pickle.load(f)
        logger.info(f"Netzwerk geladen von: {filepath}")
        return network

    def to_json(self, filepath):
        """
        Speichert das Netzwerk als JSON-Datei. Umfasst die Struktur und die Gewichte/Biases.

        Parameters:
            filepath (str): Pfad zur Ausgabedatei.
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
        logger.info(f"Netzwerk als JSON gespeichert unter: {filepath}")

    @staticmethod
    def from_json(filepath):
        """
        Lädt ein Netzwerk aus einer JSON-Datei.

        Parameters:
            filepath (str): Pfad zur JSON-Datei.

        Returns:
            Network: Geladenes Netzwerk-Objekt.
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
        logger.info(f"Netzwerk von JSON geladen von: {filepath}")
        return network

    def plot_training_history(self, history):
        """
        Visualisiert den Lernverlauf (Verlust) über die Epochen.

        Parameters:
            history (dict): Verlauf der Trainingsmetriken.
        """
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
        """
        Erstellt die Konfusionsmatrix für die Klassifikation, sofern Softmax als Output-Aktivierung gewählt wurde.

        Parameters:
            test_data (list of tuples): Liste von (Eingabe, Zielwert) Paaren.
        """
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
        """
        Speichert die Ergebnisse der verschiedenen Trials als CSV-Datei.

        Parameters:
            filepath (str): Pfad zur Ausgabedatei.
        """
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
        """
        Führt mehrere Trainingsläufe mit unterschiedlichen Hyperparametern durch und speichert die Ergebnisse.

        Parameters:
            X_train (np.ndarray): Trainings-Eingabedaten.
            y_train (np.ndarray): Trainings-Zielwerte.
            X_test (np.ndarray): Test-Eingabedaten.
            y_test (np.ndarray): Test-Zielwerte.
            num_trials (int): Anzahl der Trials zu führen.
            epochs (int): Anzahl der Epochen pro Trial.
            batch_size (int): Größe der Batches pro Trial.
            use_backpropagation (bool): Ob Backpropagation verwendet werden soll.
            learning_rate_decay (bool): Ob die Lernrate dynamisch angepasst werden soll.
            early_stopping_patience (int, optional): Anzahl der Epochen ohne Verbesserung für frühzeitiges Stoppen.
            hyperparameter_space (dict, optional): Raum der Hyperparameter für die Trials.
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

            # Aktualisiere den Optimierer basierend auf den neuen Hyperparametern
            self.optimizer = self.get_optimizer(self.optimizer_method, self.learning_rate, self.momentum)
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

            # Visualisiere die Ergebnisse jedes zweiten Trials
            if (i+1) % 2 == 0:
                self.plot_training_history(history)
                self.plot_confusion_matrix(test_data)

        logger.info("Alle Trials abgeschlossen.")

def create_dummy_data(num_samples=1000, num_features=10, num_classes=2):
    """
    Erzeugt Dummy-Daten für Tests.

    Parameters:
        num_samples (int): Anzahl der Proben.
        num_features (int): Anzahl der Merkmale.
        num_classes (int): Anzahl der Klassen.

    Returns:
        tuple: Feature-Matrix und Labels.
    """
    X = np.random.rand(num_samples, num_features).astype(np.float32)
    if num_classes > 2:
        y = np.random.randint(0, num_classes, num_samples)
        y = np.eye(num_classes)[y]  # One-Hot-Encoding
    else:
        y = np.random.randint(0, num_classes, num_samples).astype(np.float32).reshape(-1, 1)
    return X, y

class TestNetwork(unittest.TestCase):
    """
    Testklasse für das Network-Objekt, verwendet das unittest-Framework.
    """

    def test_label_shapes(self):
        """
        Testet, ob die Labels die erwartete Form haben.
        """
        X, y = create_dummy_data(num_samples=10, num_features=10, num_classes=2)
        self.assertEqual(y.shape[1], 1, f"Erwartete Form (100, 1), aber erhalten: {y.shape}")

    def test_network_training_dummy_data(self):
        """
        Testet das Training des Netzwerks mit Dummy-Daten.
        """
        # Dummy-Daten für den Test generieren
        X_train, y_train = create_dummy_data(num_samples=100, num_features=10, num_classes=2)
        X_test, y_test = create_dummy_data(num_samples=50, num_features=10, num_classes=2)

        # Netzwerk initialisieren
        num_nodes = [X_train.shape[1], 20, 1]  # Eingabe, Hidden, Ausgabe
        net = Network(num_nodes=num_nodes, V_max=1, R_harmony=0.8, activation_function='relu',
                      learning_rate=0.01, momentum=0.9, output_activation='sigmoid', optimizer_method='SGD')

        # Training mit den generierten Daten
        history = net.train(X_train, y_train, epochs=5, batch_size=10)

        # Test des Netzwerks mit Testdaten
        test_data = list(zip(X_test, y_test))
        metrics = net.test_network(test_data)

        # Überprüfen, ob die Testmetriken korrekt sind
        self.assertIsInstance(metrics, dict, "Test_network returned wrong type")
        if net.output_activation == "softmax":
            self.assertIn('accuracy', metrics, "Accuracy not in the test results")
        else:
            self.assertIn('mean_squared_error', metrics, "MSE not in test results")
            self.assertIn('mean_absolute_error', metrics, "MAE not in test results")
            self.assertIn('r2_score', metrics, "R2 not in test results")

        # Überprüfen, ob der Verlustverlauf korrekt ist
        self.assertGreater(len(history['loss']), 0, "Keine Losses im Trainingsverlauf")
        self.assertIsNotNone(history['loss'][-1], "Letzter Loss ist None")

    def test_network_training_breast_cancer_data(self):
        """
        Testet das Training und Testen des Netzwerks mit dem Brustkrebs-Datensatz.
        """
        # Brustkrebsdaten laden
        cancer = load_breast_cancer()
        X, y = cancer.data, cancer.target
        X = X.astype(np.float32)
        y = y.astype(np.float32).reshape(-1, 1)  # Zielwerte als Spaltenvektor
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Daten normalisieren
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Netzwerk initialisieren
        num_nodes = [X_train.shape[1], 10, 1]  # Eingabe, Hidden, Ausgabe
        net = Network(num_nodes=num_nodes, V_max=1, R_harmony=0.8, activation_function='relu',
                      learning_rate=0.01, momentum=0.9, output_activation='sigmoid', optimizer_method='Adam')

        # Training
        history = net.train(X_train, y_train, epochs=10, batch_size=32)

        # Test
        test_data = list(zip(X_test, y_test))
        metrics = net.test_network(test_data)

        # Überprüfen, ob die Testmetriken korrekt sind
        self.assertIsInstance(metrics, dict, "Test_network returned wrong type")
        if net.output_activation == "softmax":
            self.assertIn('accuracy', metrics, "Accuracy not in test results")
        else:
            self.assertIn('mean_squared_error', metrics, "MSE not in test results")
            self.assertIn('mean_absolute_error', metrics, "MAE not in test results")
            self.assertIn('r2_score', metrics, "R2 not in test results")
        self.assertGreater(len(history['loss']), 0, "Keine Losses im Trainingsverlauf")
        self.assertIsNotNone(history['loss'][-1], "Letzter Loss ist None")

    def test_network_from_to_json(self):
        """
        Testet das Speichern und Laden des Netzwerks als JSON-Datei.
        """
        # Dummy-Daten für den Test generieren
        X_train, y_train = create_dummy_data(num_samples=100, num_features=10, num_classes=2)
        num_nodes = [X_train.shape[1], 20, 1]  # Eingabe, Hidden, Ausgabe

        # Erstelle ein Netzwerk und trainiere es
        net = Network(num_nodes=num_nodes, V_max=1, R_harmony=0.8, activation_function='relu',
                      learning_rate=0.01, momentum=0.9, output_activation='sigmoid')
        net.train(X_train, y_train, epochs=1, batch_size=10)

        # Speichere Netzwerk als JSON
        json_filepath = 'test_network.json'
        net.to_json(json_filepath)
        self.assertTrue(os.path.exists(json_filepath))

        # Lade Netzwerk aus JSON
        loaded_net = Network.from_json(json_filepath)

        # Überprüfe, ob die Parameter und Strukturen übereinstimmen
        self.assertEqual(net.num_nodes, loaded_net.num_nodes)
        self.assertEqual(net.V_max, loaded_net.V_max)
        self.assertEqual(net.R_harmony, loaded_net.R_harmony)
        self.assertEqual(net.activation_function, loaded_net.activation_function)
        self.assertEqual(net.learning_rate, loaded_net.learning_rate)
        self.assertEqual(net.momentum, loaded_net.momentum)
        self.assertEqual(net.output_activation, loaded_net.output_activation)
        self.assertEqual(net.optimizer_method, loaded_net.optimizer_method)

        # Verwende numerische Toleranz bei den Vergleichen der Gewichte und Biases
        for w1, w2 in zip(net.weights, loaded_net.weights):
            np.testing.assert_array_almost_equal(w1, w2, decimal=7)
        for b1, b2 in zip(net.biases, loaded_net.biases):
            np.testing.assert_array_almost_equal(b1, b2, decimal=7)

        # Lösche JSON-Datei nach dem Test
        os.remove(json_filepath)

    def test_network_from_to_pickle(self):
        """
        Testet das Speichern und Laden des Netzwerks als Pickle-Datei.
        """
        # Dummy-Daten für den Test generieren
        X_train, y_train = create_dummy_data(num_samples=100, num_features=10, num_classes=2)
        num_nodes = [X_train.shape[1], 20, 1]  # Eingabe, Hidden, Ausgabe
        
        # Erstelle ein Netzwerk und trainiere es
        net = Network(num_nodes=num_nodes, V_max=1, R_harmony=0.8, activation_function='relu',
                      learning_rate=0.01, momentum=0.9, output_activation='sigmoid')
        net.train(X_train, y_train, epochs=1, batch_size=10)

        # Speichere Netzwerk als Pickle
        pickle_filepath = 'test_network.pkl'
        net.save_network(pickle_filepath)
        self.assertTrue(os.path.exists(pickle_filepath))

        # Lade Netzwerk aus Pickle
        loaded_net = Network.load_network(pickle_filepath)

        # Überprüfe, ob die Parameter und Strukturen übereinstimmen
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

        # Lösche Pickle-Datei nach dem Test
        os.remove(pickle_filepath)

    def test_run_trials_with_dummy_data(self):
        """
        Testet das Ausführen mehrerer Trainings-Trials mit Dummy-Daten.
        """
        # Dummy-Daten für den Test generieren
        X_train, y_train = create_dummy_data(num_samples=100, num_features=10, num_classes=2)
        X_test, y_test = create_dummy_data(num_samples=50, num_features=10, num_classes=2)
        num_nodes = [X_train.shape[1], 20, 1]  # Eingabe, Hidden, Ausgabe
        net = Network(num_nodes=num_nodes, V_max=1, R_harmony=0.8, activation_function='relu',
                      learning_rate=0.01, momentum=0.9, output_activation='sigmoid')

        # Führe Trials aus
        net.run_trials(X_train, y_train, X_test, y_test, num_trials=2, epochs=2, batch_size=10, hyperparameter_space={
            'learning_rate': [0.01, 0.001],
            'momentum': [0.9, 0.99],
            'optimizer_method': ['SGD', 'Adam']
        })
        
        # Überprüfen, ob Ergebnisse der Trials erzeugt wurden
        self.assertGreater(len(net.trial_results), 0, "Keine Trainingsergebnisse der Trials vorhanden")

        # Überprüfen, ob CSV Datei erzeugt wurde
        csv_filepath = 'trial_results.csv'
        net.save_trial_results(csv_filepath)
        self.assertTrue(os.path.exists(csv_filepath))
        os.remove(csv_filepath)

if __name__ == '__main__':
    unittest.main()
