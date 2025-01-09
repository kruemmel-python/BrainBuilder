# Neuronales Netzwerk: Training und Evaluierung

Dieses Repository enthält eine Python-Implementierung eines neuronalen Netzwerks für Training und Evaluierung. Das Netzwerk unterstützt verschiedene Aktivierungsfunktionen, Optimierer und bietet Funktionen zur Datenverarbeitung, Visualisierung und Hyperparameter-Tuning.

## Funktionen

- **Datenladen und -vorverarbeitung**: Funktionen zum Laden von Daten aus CSV-Dateien, Handhabung fehlender Werte und Datenvorverarbeitung.
- **Aktivierungsfunktionen**: Implementierungen der Sigmoid-, ReLU- und Softmax-Aktivierungsfunktionen.
- **Optimierer**: Unterstützung für SGD, Momentum, Adam und RMSprop Optimierer.
- **Training und Evaluierung**: Funktionen zum Trainieren des Netzwerks, Bewerten seiner Leistung und Visualisieren des Trainingsverlaufs.
- **Hyperparameter-Tuning**: Führen Sie mehrere Versuche mit unterschiedlichen Hyperparametern durch, um die beste Konfiguration zu finden.
- **Modellspeicherung und -laden**: Speichern und Laden des Netzwerkmodells in Pickle- und JSON-Formaten.

## Anforderungen

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

Um die erforderlichen Pakete zu installieren, führen Sie aus:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Verwendung

### Datenladen und -vorverarbeitung

```python
X_train, X_test, y_train, y_test = load_data_from_csv('data.csv', target_column='label')
check_for_nan(X_train)
check_for_nan(y_train)
```

### Netzwerkinitialisierung und -training

```python
input_dim = X_train.shape[1]
num_nodes = [input_dim, 20, 1]  # Beispiel: 1 versteckte Schicht mit 20 Neuronen, 1 Ausgabeneuron

network = Network(num_nodes, V_max=0.5, R_harmony=0.1, activation_function='relu', learning_rate=0.01, momentum=0.9, output_activation='sigmoid', optimizer_method='Adam')
network.validate_dimensions(X_train, y_train)
history = network.train(X_train, y_train, epochs=10, batch_size=32, use_backpropagation=True, learning_rate_decay=True, early_stopping_patience=3)
network.plot_training_history(history)
```

### Netzwerkevaluierung

```python
test_data = list(zip(X_test, y_test))
metrics = network.test_network(test_data)
logger.info(f"Metriken nach dem Training: {metrics}")
```

### Visualisierung

```python
network.visualize()
network.log_neuron_activations(X_test[:5])  # Aktivierungen für die ersten 5 Testdatenpunkte protokollieren
network.plot_confusion_matrix(test_data)
```

### Modellspeicherung und -laden

```python
network.save_network('my_network.pkl')
loaded_network = Network.load_network('my_network.pkl')
loaded_metrics = loaded_network.test_network(test_data)
logger.info(f"Metriken des geladenen Netzwerks: {loaded_metrics}")

network.to_json("my_network.json")
loaded_network_json = Network.from_json("my_network.json")
loaded_metrics_json = loaded_network_json.test_network(test_data)
logger.info(f"Metriken des geladenen JSON-Netzwerks: {loaded_metrics_json}")
```

### Hyperparameter-Tuning

```python
hyperparameter_space = {
    'learning_rate': [0.01, 0.001, 0.0001],
    'momentum': [0.9, 0.8, 0.7],
    'optimizer_method': ['SGD', 'Adam', 'RMSprop']
}

network.run_trials(X_train, y_train, X_test, y_test, epochs=5, batch_size=32, use_backpropagation=True, learning_rate_decay=True, early_stopping_patience=2, hyperparameter_space=hyperparameter_space)
network.save_trial_results('trial_results.csv')
```

## Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert. Weitere Details finden Sie in der [LICENSE](LICENSE)-Datei.

## Mitwirkende

Beiträge sind willkommen! Bitte öffnen Sie ein Issue oder senden Sie einen Pull Request.

## Danksagungen

- Inspiriert von verschiedenen Implementierungen und Tutorials zu neuronalen Netzwerken.
- Besonderer Dank gilt der Open-Source-Community für ihre Beiträge zu den in diesem Projekt verwendeten Bibliotheken.
