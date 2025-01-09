# Modulares und Erweiterbares Neuronales Netzwerk in Python

Ein leistungsfähiges, modulares und erweiterbares neuronales Netzwerk, implementiert in Python. Dieses Projekt bietet umfassende Werkzeuge zur Erstellung, Schulung und Evaluierung von neuronalen Netzwerken mit verschiedenen Optimierungsalgorithmen und Aktivierungsfunktionen. Es umfasst Funktionen zur Datenverarbeitung, Visualisierung sowie umfangreiche Unit-Tests zur Sicherstellung der Funktionalität.

## Inhaltsverzeichnis

- [Einführung](#einführung)
- [Features](#features)
- [Voraussetzungen](#voraussetzungen)
- [Installation](#installation)
- [Verwendung](#verwendung)
  - [Datensatz erstellen und laden](#datensatz-erstellen-und-laden)
  - [Netzwerk initialisieren](#netzwerk-initialisieren)
  - [Training](#training)
  - [Vorhersagen treffen](#vorhersagen-treffen)
  - [Speichern und Laden des Netzwerks](#speichern-und-laden-des-netzwerks)
  - [Visualisierung](#visualisierung)
  - [Durchführen von Trials](#durchführen-von-trials)
- [Optimierer](#optimierer)
- [Aktivierungsfunktionen](#aktivierungsfunktionen)
- [Glossar](#glossar)
- [Unit-Tests](#unit-tests)
- [Beispielprojekt: Klassifikation des Breast Cancer Datensatzes](#beispielprojekt-klassifikation-des-breast-cancer-datensatzes)
- [Lizenz](#lizenz)

## Einführung

Dieses Projekt implementiert ein modulares und erweiterbares neuronales Netzwerk in Python. Neuronale Netzwerke sind eine Art von maschinellem Lernen, das darauf abzielt, Muster in Daten zu erkennen und Vorhersagen zu treffen. In diesem spezifischen Projekt wird das Netzwerk zur Vorhersage von Brustkrebs eingesetzt, basierend auf dem **Breast Cancer Wisconsin (Diagnostic) Datensatz** von scikit-learn. Das Netzwerk eignet sich sowohl für Klassifikations- als auch für Regressionsaufgaben und bietet verschiedene Optimierungsalgorithmen und Aktivierungsfunktionen, um unterschiedliche Anforderungen zu erfüllen.

## Features

- **Modulare Architektur**: Flexibel definierbare Anzahl der Schichten und Neuronen pro Schicht.
- **Verschiedene Optimierer**: Unterstützung für SGD (mit Momentum), Adam und RMSprop.
- **Verschiedene Aktivierungsfunktionen**: Sigmoid, ReLU und Softmax.
- **Datenverarbeitung**: Laden von Daten aus CSV-Dateien, Skalierung, One-Hot-Encoding und Umgang mit fehlenden Werten.
- **Visualisierung**: Darstellung der Netzwerkarchitektur, Trainingsverlauf und Konfusionsmatrix.
- **Speicherung und Laden**: Netzwerke können als JSON oder Pickle gespeichert und geladen werden.
- **Trials**: Durchführung mehrerer Trainingsläufe mit unterschiedlichen Hyperparametern zur Optimierung der Leistung.
- **Frühes Stoppen**: Automatisches Stoppen des Trainings bei Stagnation der Verlustfunktion.
- **Umfassende Unit-Tests**: Sicherstellung der Funktionalität und Stabilität des Netzwerks.
- **Erweiterbarkeit**: Leicht erweiterbar um zusätzliche Funktionen und Optimierer.

## Voraussetzungen

Dieses Projekt setzt Grundkenntnisse in Python voraus. Kenntnisse in maschinellem Lernen und neuronalen Netzwerken sind von Vorteil, aber nicht zwingend erforderlich. Für Anfänger empfehlen wir folgende Ressourcen:

- [Einführung in maschinelles Lernen mit Python](https://scikit-learn.org/stable/tutorial/index.html)
- [Neuronale Netzwerke erklärt](https://www.deeplearning.ai/)

## Installation

Stellen Sie sicher, dass Sie Python 3.7 oder höher installiert haben. Installieren Sie die benötigten Abhängigkeiten mit `pip`:

```bash
pip install numpy pandas matplotlib scikit-learn
```

Für die Unit-Tests wird das `unittest`-Modul verwendet, das standardmäßig mit Python geliefert wird.

## Verwendung

### Datensatz erstellen und laden

Dieses Projekt verwendet den **Breast Cancer Wisconsin (Diagnostic) Datensatz** aus scikit-learn. Der Datensatz enthält Merkmale von Zellkernen, die aus Brusttumorproben extrahiert wurden, sowie die Zielvariable, die angibt, ob der Tumor bösartig (1) oder gutartig (0) ist.

#### Datensatz erstellen und als CSV speichern

```python
import pandas as pd
from sklearn.datasets import load_breast_cancer

# Laden des Breast Cancer Datensatzes aus sklearn
breast_cancer = load_breast_cancer()

# Umwandlung des Datensatzes in einen Pandas DataFrame
data = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)

# Hinzufügen der Zielvariable (Klassenbezeichnung)
data['target'] = breast_cancer.target

# Anzeige der ersten paar Zeilen des DataFrame
print(data.head())

# Optional: Speichern des Datensatzes als CSV
data.to_csv("breast_cancer.csv", index=False)
```

#### Daten laden und vorbereiten

Verwenden Sie die Funktion `load_data_from_csv`, um den gespeicherten Datensatz zu laden, die Merkmale zu skalieren und den Datensatz in Trainings- und Testsets aufzuteilen.

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

### Netzwerk initialisieren

Initialisieren Sie das Netzwerk, indem Sie die Struktur und Hyperparameter definieren.

```python
from your_module import Network

num_nodes = [X_train.shape[1], 20, 1]  # Eingabe, versteckte Schicht, Ausgabe
netzwerk = Network(
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

Trainieren Sie das Netzwerk mit den Trainingsdaten.

```python
history = netzwerk.train(
    X_train,
    y_train,
    epochs=100,  # Anzahl der vollständigen Durchläufe durch die Trainingsdaten
    batch_size=32,  # Anzahl der Beispiele, die gleichzeitig verarbeitet werden
    use_backpropagation=True,  # Verwendung des Backpropagation-Algorithmus zur Berechnung der Gradienten
    learning_rate_decay=True,  # Anpassung der Lernrate während des Trainings
    early_stopping_patience=10  # Frühes Stoppen, wenn sich die Verlustfunktion 10 Epochen lang nicht verbessert
)
```

*Hinweis: Die Lernrate steuert, wie stark die Gewichte bei jedem Update angepasst werden. Eine zu hohe Lernrate kann das Training instabil machen, während eine zu niedrige Lernrate zu langsamer Konvergenz führt.*

### Vorhersagen treffen

Verwenden Sie das trainierte Netzwerk, um Vorhersagen zu treffen.

```python
vorhersagen = netzwerk.predict(X_test)
```

### Speichern und Laden des Netzwerks

#### Speichern als Pickle

```python
netzwerk.save_network('netzwerk.pkl')
```

#### Laden aus Pickle

```python
from your_module import Network

geladenes_netzwerk = Network.load_network('netzwerk.pkl')
```

#### Speichern als JSON

```python
netzwerk.to_json('netzwerk.json')
```

#### Laden aus JSON

```python
geladenes_netzwerk = Network.from_json('netzwerk.json')
```

### Visualisierung

#### Netzwerkarchitektur visualisieren

```python
netzwerk.visualize()
```

#### Trainingsverlauf plotten

```python
netzwerk.plot_training_history(history)
```

#### Konfusionsmatrix erstellen (bei Klassifikation)

```python
netzwerk.plot_confusion_matrix(list(zip(X_test, y_test)))
```

### Durchführen von Trials

Führen Sie mehrere Trainingsläufe mit unterschiedlichen Hyperparametern durch, um die besten Einstellungen zu finden.

```python
netzwerk.run_trials(
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

Speichern Sie die Ergebnisse der Trials als CSV:

```python
netzwerk.save_trial_results('trial_ergebnisse.csv')
```

## Optimierer

Das Netzwerk unterstützt verschiedene Optimierungsalgorithmen, die die Gewichte und Biases während des Trainings aktualisieren:

- **SGD (Stochastic Gradient Descent)**: Mit Momentum zur Beschleunigung des Trainings.
- **Adam**: Adaptive Moment Estimation, kombiniert Momentum und RMSprop.
- **RMSprop**: Adaptive Lernraten, basierend auf den Mittelwerten der Quadrate der Gradienten.

### Auswahl des Optimierers

Beim Initialisieren des Netzwerks können Sie den gewünschten Optimierer wählen:

```python
netzwerk = Network(
    num_nodes=num_nodes,
    optimizer_method='Adam',
    learning_rate=0.001
)
```

## Aktivierungsfunktionen

Das Netzwerk bietet verschiedene Aktivierungsfunktionen zur Einführung von Nichtlinearitäten:

- **Sigmoid**: Für binäre Klassifikationsaufgaben geeignet.
- **ReLU (Rectified Linear Unit)**: Fördert schnelle Konvergenz und vermeidet das Vanishing-Gradient-Problem.
- **Softmax**: Für mehrklassige Klassifikationsaufgaben geeignet.

### Auswahl der Aktivierungsfunktion

Beim Initialisieren des Netzwerks können Sie die gewünschte Aktivierungsfunktion wählen:

```python
netzwerk = Network(
    activation_function='relu',
    output_activation='softmax'
)
```

## Glossar

- **Optimizer (Optimierer)**: Algorithmen, die die Gewichte und Biases eines neuronalen Netzwerks während des Trainings anpassen.
- **Momentum**: Ein Parameter, der hilft, das Training zu beschleunigen und lokale Minima zu vermeiden, indem vergangene Gradienten berücksichtigt werden.
- **Softmax**: Eine Aktivierungsfunktion, die die Ausgabe in Wahrscheinlichkeiten umwandelt, die sich zu 1 summieren, häufig verwendet in der Ausgabeschicht bei Mehrklassenklassifikation.
- **One-Hot Encoding**: Eine Methode zur Kodierung kategorialer Variablen, bei der jede Klasse als Vektor mit einer 1 an der Stelle der Klasse und 0en an allen anderen Stellen dargestellt wird.

## Unit-Tests

Das Projekt enthält umfassende Unit-Tests, um die Funktionalität sicherzustellen. Führen Sie die Tests mit folgendem Befehl aus:

```bash
python your_program.py
```

Die Tests decken verschiedene Aspekte ab, einschließlich:

- Überprüfung der Label-Formate
- Training und Testen mit Dummy-Daten
- Speichern und Laden des Netzwerks als JSON und Pickle
- Durchführung von Trials

## Beispielprojekt: Klassifikation des Breast Cancer Datensatzes

Hier zeigen wir ein einfaches Beispiel, wie das Netzwerk zur Klassifikation des **Breast Cancer Wisconsin (Diagnostic) Datensatzes** verwendet werden kann.

### Schritt 1: Datensatz erstellen und laden

```python
import pandas as pd
from your_module import load_data_from_csv

# Laden des Breast Cancer Datensatzes und Speichern als CSV
from sklearn.datasets import load_breast_cancer

breast_cancer = load_breast_cancer()
data = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
data['target'] = breast_cancer.target
data.to_csv("breast_cancer.csv", index=False)

# Daten laden und vorbereiten
X_train, X_test, y_train, y_test = load_data_from_csv(
    filepath='breast_cancer.csv',
    target_column='target',
    test_size=0.2,
    random_state=42,
    scale_features=True
)
```

### Schritt 2: Netzwerk initialisieren

```python
from your_module import Network

num_nodes = [X_train.shape[1], 20, 1]  # Eingabe, versteckte Schicht, Ausgabe
netzwerk = Network(
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

### Schritt 3: Training

```python
history = netzwerk.train(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    use_backpropagation=True,
    learning_rate_decay=True,
    early_stopping_patience=10
)
```

### Schritt 4: Vorhersagen treffen und Evaluation

```python
vorhersagen = netzwerk.predict(X_test)
metriken = netzwerk.test_network(list(zip(X_test, y_test)))
print(metriken)
```

### Schritt 5: Visualisierung

```python
netzwerk.plot_training_history(history)
netzwerk.plot_confusion_matrix(list(zip(X_test, y_test)))
```

### Schritt 6: Netzwerk speichern und laden

```python
# Speichern
netzwerk.save_network('breast_cancer_netzwerk.pkl')
netzwerk.to_json('breast_cancer_netzwerk.json')

# Laden
geladenes_netzwerk = Network.load_network('breast_cancer_netzwerk.pkl')
geladenes_netzwerk_json = Network.from_json('breast_cancer_netzwerk.json')
```

Dieses Beispiel führt Sie durch den gesamten Prozess der Datenvorbereitung, Netzwerkinitialisierung, des Trainings, der Vorhersage, der Evaluation und der Speicherung des Netzwerks. Es bietet einen praktischen Einstieg und erleichtert das Verständnis der einzelnen Schritte.

## Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert. Weitere Informationen finden Sie in der [LICENSE](LICENSE)-Datei.

---
