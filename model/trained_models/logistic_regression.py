import os
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Cargar dataset
df = pd.read_csv(
    r"C:\Users\Tupot\OneDrive\Escritorio\twitter_profiles_cleaned.csv"
)

# Variable objetivo
target = "label"

# Features y target
# Eliminar columnas de texto
columns_to_drop = [
    "name",
    "screen_name",
    "description",
    "location",
    "created_at"
]

X = df.drop(columns=columns_to_drop + [target])

# Mantener solo columnas numéricas
X = X.select_dtypes(include=["number"])
y = df[target]

# División train/test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Entrenamiento del modelo
logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(X_train, y_train)

# Evaluación rápida
y_pred = logistic_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

joblib.dump(
    logistic_model,
    "logistic_regression.pkl"
)

print("Modelo guardado correctamente.")