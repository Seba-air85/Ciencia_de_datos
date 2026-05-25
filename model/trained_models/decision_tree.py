import joblib
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
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
decision_tree_model = DecisionTreeClassifier(
    random_state=42
)

decision_tree_model.fit(X_train, y_train)

# Evaluación
y_pred = decision_tree_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Guardar modelo
joblib.dump(
    decision_tree_model,
    r"C:\Users\Tupot\OneDrive\Escritorio\trained_models\decision_tree.pkl"
)

print("Modelo guardado correctamente.")