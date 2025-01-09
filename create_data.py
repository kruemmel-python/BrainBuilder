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