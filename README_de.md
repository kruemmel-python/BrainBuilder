# Modulares und Erweiterbares Neuronales Netzwerk zur Brustkrebs-Erkennung in Python

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

Die frühzeitige Erkennung von Brustkrebs erhöht erheblich die Chancen auf eine erfolgreiche Behandlung und das Überleben der Patienten. Dieses Projekt nutzt die Leistungsfähigkeit neuronaler Netzwerke, um ein KI-Modell zu erstellen, das speziell darauf trainiert ist, Brustkrebs anhand des **Breast Cancer Wisconsin (Diagnostic) Datensatzes** von scikit-learn zu erkennen. Durch die Analyse verschiedener Merkmale, die aus Zellkernen von Brusttumorproben extrahiert wurden, lernt das neuronale Netzwerk, zwischen bösartigen und gutartigen Tumoren zu unterscheiden.

Das Hauptziel dieses Projekts ist die Entwicklung eines zuverlässigen und effizienten KI-Werkzeugs, das medizinische Fachkräfte bei der Diagnose von Brustkrebs unterstützt, die diagnostische Genauigkeit verbessert und eine rechtzeitige Intervention erleichtert. Die modulare und erweiterbare Architektur des Netzwerks ermöglicht Flexibilität bei der Experimentierung mit verschiedenen Konfigurationen, Optimierungsalgorithmen und Aktivierungsfunktionen, was es zu einer wertvollen Ressource sowohl für Lernende als auch für Entwickler im Bereich des maschinellen Lernens und der medizinischen Diagnostik macht.

## Features

- **Modulare Architektur**: Flexibel definierbare Anzahl der Schichten und Neuronen pro Schicht.
- **Verschiedene Optimierer**: Unterstützung für SGD (mit Momentum), Adam und RMSprop.
- **Verschiedene Aktivierungsfunktionen**: Sigmoid, ReLU und Softmax.
- **Datenverarbeitung**: Laden von Daten aus CSV-Dateien, Skalierung, One-Hot-Encoding und Umgang mit fehlenden Werten.
- **Visualisierung**: Darstellung der Netzwerkarchitektur, Trainingsverlauf und Konfusionsmatrix.
- **Speicherung und Laden**: Netzwerke können als JSON oder Pickle-Dateien gespeichert und geladen werden.
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

Der Datensatz wird geladen und als CSV-Datei gespeichert, um die Nutzung und Verarbeitung zu erleichtern.

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

# Speichern des Datensatzes als CSV-Datei
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

**Was das Netzwerk mit den Daten macht:**

- **Datenladen**: Das Netzwerk liest die CSV-Datei, die die Brustkrebsdaten enthält.
- **Datenvorverarbeitung**: Die Merkmale werden mithilfe von Min-Max-Skalierung normalisiert, um die Eingabedaten zu standardisieren.
- **Training**: Das neuronale Netzwerk wird mit dem Trainingssatz trainiert, um Muster zu erkennen, die bösartige von gutartigen Tumoren unterscheiden.
- **Vorhersage**: Nach dem Training kann das Netzwerk die Wahrscheinlichkeit vorhersagen, ob ein Tumor bösartig oder gutartig ist, basierend auf den Eingabemerkmalen.
- **Evaluation**: Die Leistung des Netzwerks wird mithilfe von Metriken wie Genauigkeit, mittlerer quadratischer Fehler (MSE), mittlerer absoluter Fehler (MAE) und R²-Score bewertet.

### Netzwerk initialisieren

Initialisieren Sie das Netzwerk, indem Sie die Struktur und Hyperparameter definieren.

```python
from your_module import Network

num_nodes = [X_train.shape[1], 20, 1]  # Eingabe, versteckte Schicht, Ausgabe
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

Trainieren Sie das Netzwerk mit den Trainingsdaten.

```python
history = network.train(
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
predictions = network.predict(X_test)
```

### Speichern und Laden des Netzwerks

#### Speichern als Pickle

```python
network.save_network('network.pkl')
```

#### Laden aus Pickle

```python
from your_module import Network

loaded_network = Network.load_network('network.pkl')
```

#### Speichern als JSON

```python
network.to_json('network.json')
```

#### Laden aus JSON

```python
loaded_network = Network.from_json('network.json')
```

### Visualisierung

#### Netzwerkarchitektur visualisieren

```python
network.visualize()
```

#### Trainingsverlauf plotten

```python
network.plot_training_history(history)
```

#### Konfusionsmatrix erstellen (bei Klassifikation)

```python
network.plot_confusion_matrix(list(zip(X_test, y_test)))
```

### Durchführen von Trials

Führen Sie mehrere Trainingsläufe mit unterschiedlichen Hyperparametern durch, um die besten Einstellungen zu finden.

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

Speichern Sie die Ergebnisse der Trials als CSV:

```python
network.save_trial_results('trial_results.csv')
```

## Optimierer

Das Netzwerk unterstützt verschiedene Optimierungsalgorithmen, die die Gewichte und Biases während des Trainings aktualisieren:

- **SGD (Stochastic Gradient Descent)**: Mit Momentum zur Beschleunigung des Trainings.
- **Adam**: Adaptive Moment Estimation, kombiniert Momentum und RMSprop.
- **RMSprop**: Adaptive Lernraten, basierend auf den Mittelwerten der Quadrate der Gradienten.

### Auswahl des Optimierers

Beim Initialisieren des Netzwerks können Sie den gewünschten Optimierer wählen:

```python
network = Network(
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
network = Network(
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

Dieses Beispiel demonstriert, wie das neuronale Netzwerk verwendet wird, um ein KI-Modell zu trainieren, das Brustkrebs erkennt. Der Datensatz wird aus einer CSV-Datei geladen, das Netzwerk wird initialisiert und trainiert, und das Modell wird zur Vorhersage und Bewertung verwendet. Anschließend wird das trainierte Modell gespeichert.

### Schritt 1: Datensatz erstellen und laden

Zuerst wird der Breast Cancer Datensatz aus scikit-learn geladen, in eine CSV-Datei konvertiert und dann in das neuronale Netzwerk geladen.

```python
import pandas as pd
from your_module import load_data_from_csv

# Laden des Breast Cancer Datensatzes und Speichern als CSV-Datei
from sklearn.datasets import load_breast_cancer

breast_cancer = load_breast_cancer()
data = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
data['target'] = breast_cancer.target
data.to_csv("breast_cancer.csv", index=False)

# Laden und Vorbereiten der Daten
X_train, X_test, y_train, y_test = load_data_from_csv(
    filepath='breast_cancer.csv',
    target_column='target',
    test_size=0.2,
    random_state=42,
    scale_features=True
)
```

### Schritt 2: Netzwerk initialisieren

Initialisieren Sie das neuronale Netzwerk mit der gewünschten Architektur und den Hyperparametern.

```python
from your_module import Network

num_nodes = [X_train.shape[1], 20, 1]  # Eingabe, versteckte Schicht, Ausgabe
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

### Schritt 3: Training

Trainieren Sie das neuronale Netzwerk mit den Trainingsdaten.

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

### Schritt 4: Vorhersagen treffen und Evaluation

Nach dem Training verwenden Sie das Netzwerk, um Vorhersagen auf dem Testsatz zu treffen und die Leistung zu bewerten.

```python
predictions = network.predict(X_test)
metrics = network.test_network(list(zip(X_test, y_test)))
print(metrics)
```

### Schritt 5: Visualisierung

Visualisieren Sie den Trainingsverlauf und die Konfusionsmatrix, um die Leistung des Netzwerks besser zu verstehen.

```python
network.plot_training_history(history)
network.plot_confusion_matrix(list(zip(X_test, y_test)))
```

### Schritt 6: Speichern und Laden des Netzwerks

Speichern Sie das trainierte Netzwerk für die zukünftige Nutzung und laden Sie es bei Bedarf wieder.

```python
# Speichern des Netzwerks
network.save_network('breast_cancer_network.pkl')
network.to_json('breast_cancer_network.json')

# Laden des Netzwerks
loaded_network = Network.load_network('breast_cancer_network.pkl')
loaded_network_json = Network.from_json('breast_cancer_network.json')
```

**Was das Netzwerk lernt:**

Das neuronale Netzwerk lernt, Muster in den Merkmalen der Brusttumorproben zu erkennen, die auf eine Bösartigkeit oder Gutartigkeit des Tumors hinweisen. Durch das Training mit dem Datensatz passt das Modell seine Gewichte und Biases an, um die Verlustfunktion zu minimieren und dadurch genauere Vorhersagen auf unbekannten Daten zu ermöglichen.

**Speichern des Modells:**

Nach dem Training wird das Modell sowohl im Pickle- als auch im JSON-Format gespeichert. Dies ermöglicht eine einfache Speicherung und Wiederverwendung des trainierten Modells für zukünftige Vorhersagen, ohne dass eine erneute Schulung erforderlich ist.

## Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert. Weitere Informationen finden Sie in der [LICENSE](LICENSE)-Datei.

---
