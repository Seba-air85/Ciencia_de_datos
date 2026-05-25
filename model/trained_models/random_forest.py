import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Cargar dataset
df = pd.read_csv(
    r"C:\Users\Tupot\OneDrive\Escritorio\twitter_profiles_cleaned.csv"
)

# Variable objetivo
target = "label"

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
random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest_model.fit(X_train, y_train)

# Evaluación
y_pred = random_forest_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Guardar modelo
joblib.dump(
    random_forest_model,
    r"C:\Users\Tupot\OneDrive\Escritorio\trained_models\random_forest.pkl"
)

print("Modelo guardado correctamente.")