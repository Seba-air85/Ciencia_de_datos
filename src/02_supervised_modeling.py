# Instalar Optuna en caso de que la sesión de Colab no lo tenga
!pip install -q optuna xgboost lightgbm

import pandas as pd
import numpy as np
import os
import warnings
import joblib

# Optuna para optimización bayesiana
import optuna

# Herramientas de Scikit-Learn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Modelos clásicos y avanzados
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING) # Evita que Colab se llene de texto por cada intento
print("✅ Todas las librerías cargadas y listas")

# Montar Drive (si no está montado)
from google.colab import drive
drive.mount('/content/drive')



# ============================================================================
# LOCALIZAR Y CARGAR EL CSV
# ============================================================================

print("="*60)
print("BUSCANDO TU ARCHIVO CSV")
print("="*60)

# Buscar en Drive
csv_files = []
for root, dirs, files in os.walk('/content/drive/MyDrive'):
    for file in files:
        if file.endswith('.csv') and ('twitter' in file.lower() or 'profile' in file.lower()):
            filepath = os.path.join(root, file)
            size = os.path.getsize(filepath)
            csv_files.append((filepath, size))
            print(f"📄 Encontrado: {filepath} ({size:,} bytes)")

if csv_files:
    # Usar el archivo más grande (dataset completo)
    filepath = csv_files[0][0]
    df = pd.read_csv(filepath)
    print(f"\n✅ Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"\n📊 Columnas disponibles:")
    print(df.columns.tolist())
else:
    # Si no encuentra, subir manualmente
    print("❌ No se encontró el archivo. Por favor, súbelo manualmente:")
    from google.colab import files
    uploaded = files.upload()
    filename = list(uploaded.keys())[0]
    df = pd.read_csv(filename)
    print(f"✅ Dataset cargado: {df.shape}")

# ============================================================================
# PREPARACIÓN DE DATOS (MANTENIENDO TU LÓGICA DE TWITTER)
# ============================================================================

# Seleccionar tus columnas de características
feature_columns = ['followers_count', 'friends_count', 'post_count', 'has_location']
if 'is_real_location' in df.columns:
    feature_columns.append('is_real_location')

# Rellenar nulos de forma segura en las columnas seleccionadas
for col in feature_columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(0)

X = df[feature_columns].values
y = df['label'].values

# Dividir en entrenamiento y prueba (80% / 20%) de forma estratificada
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Datos listos. Entrenamiento: {X_train.shape[0]} muestras. Prueba: {X_test.shape[0]} muestras.")

# ============================================================================
# DETECTAR LA "TRAMPA" EN LOS DATOS (CORRELACIÓN)
# ============================================================================
print("🕵️‍♂️ Analizando correlación con la variable 'label'...")
correlaciones = df[feature_columns + ['label']].corr()['label'].sort_values(ascending=False)
print(correlaciones)

print("\n💡 Si alguna variable tiene una correlación cercana a 1.0 o -1.0 (ej. 0.98),")
print("esa es la variable 'trampa' que está inflando tus modelos. ¡Debes eliminarla de feature_columns!")

def objective(trial, X, y):
    steps = []

    # 1. PASO OBLIGATORIO PARA TUS DATOS: Escalado estándar
    steps.append(('scaler', StandardScaler()))

    # 2. PCA CONDICIONAL: Optuna decide si ayuda a reducir ruido en las métricas de Twitter
    use_pca = trial.suggest_categorical("use_pca", [True, False])
    if use_pca:
        pca_variance = trial.suggest_categorical("pca_variance", [0.85, 0.90, 0.95])
        steps.append(('pca', PCA(n_components=pca_variance, random_state=42)))

    # 3. SELECCIÓN DE MODELO: Elige entre tus modelos y los algoritmos top del profesor
    classifier_name = trial.suggest_categorical("classifier", ["LogisticRegression", "RandomForest", "SVM", "XGBoost", "LightGBM"])

    if classifier_name == "LogisticRegression":
        c_lr = trial.suggest_float("lr_C", 1e-3, 1e2, log=True)
        model = LogisticRegression(C=c_lr, max_iter=1000, class_weight='balanced', random_state=42)

    elif classifier_name == "RandomForest":
        rf_n_estimators = trial.suggest_int("rf_n_estimators", 50, 250)
        rf_max_depth = trial.suggest_int("rf_max_depth", 5, 20) # Reducido un poco para evitar tu overfitting previo
        model = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth, class_weight='balanced', random_state=42, n_jobs=-1)

    elif classifier_name == "SVM":
        svm_c = trial.suggest_float("svm_C", 0.1, 50.0, log=True)
        svm_kernel = trial.suggest_categorical("svm_kernel", ["rbf", "linear"])
        model = SVC(C=svm_c, kernel=svm_kernel, class_weight='balanced', random_state=42)

    elif classifier_name == "XGBoost":
        xgb_n_estimators = trial.suggest_int("xgb_n_estimators", 50, 150)
        xgb_lr = trial.suggest_float("xgb_learning_rate", 0.01, 0.2, log=True)
        model = XGBClassifier(n_estimators=xgb_n_estimators, learning_rate=xgb_lr, random_state=42, eval_metric='logloss', n_jobs=-1)

    else: # LightGBM
        lgb_n_estimators = trial.suggest_int("lgb_n_estimators", 50, 150)
        lgb_lr = trial.suggest_float("lgb_learning_rate", 0.01, 0.2, log=True)
        model = LGBMClassifier(n_estimators=lgb_n_estimators, learning_rate=lgb_lr, class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)

    # Acoplamos el clasificador elegido al pipeline
    steps.append(('classifier', model))
    pipeline = Pipeline(steps)

    # 4. VALIDACIÓN CRUZADA: Evaluamos el pipeline completo de forma segura
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Cambiamos 'accuracy' por 'f1' para proteger el rendimiento si tus clases están desbalanceadas (Bots vs Reales)
    score = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)

    return score.mean()

print("🚀 Iniciando la búsqueda del mejor Pipeline e Hiperparámetros con Optuna...")

# Creamos el estudio buscando maximizar el F1-Score promedio de la Validación Cruzada
study = optuna.create_study(direction="maximize")
study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=30)

print("\n" + "="*60)
print(f"👑 PIPELINE GANADOR ENCONTRADO:")
print(f"📊 Mejor F1-Score (CV): {study.best_value:.4f}")
print(f"⚙️ Parámetros óptimos: {study.best_params}")
print("="*60 + "\n")

# Guardar los mejores parámetros tal como hace tu profesor
os.makedirs("outputs/models", exist_ok=True)
joblib.dump(study.best_params, "outputs/models/best_params.pkl")
print("💾 Parámetros guardados en 'outputs/models/best_params.pkl'")

# ============================================================================
# EVALUACIÓN FINAL CON EL CONJUNTO DE PRUEBA (DATOS NO VISTOS) - CORREGIDO
# ============================================================================
print("🔄 Reconstruyendo el mejor pipeline para la evaluación final...")

# Definir la ruta base de Google Drive de manera explícita
BASE_PATH = '/content/drive/MyDrive/twitter_project'
os.makedirs(f"{BASE_PATH}/outputs/models", exist_ok=True)
os.makedirs(f"{BASE_PATH}/data/processed", exist_ok=True)

best_params = study.best_params
final_steps = [('scaler', StandardScaler())]

# 1. Reconstruir el paso de PCA si Optuna determinó que ayudaba
if best_params["use_pca"]:
    final_steps.append(('pca', PCA(n_components=best_params["pca_variance"], random_state=42)))

# 2. Reconstruir el clasificador ganador con sus parámetros óptimos
clf_name = best_params["classifier"]
if clf_name == "LogisticRegression":
    final_model = LogisticRegression(C=best_params["lr_C"], max_iter=1000, class_weight='balanced', random_state=42)
elif clf_name == "RandomForest":
    final_model = RandomForestClassifier(n_estimators=best_params["rf_n_estimators"], max_depth=best_params["rf_max_depth"], class_weight='balanced', random_state=42, n_jobs=-1)
elif clf_name == "SVM":
    final_model = SVC(C=best_params["svm_C"], kernel=best_params["svm_kernel"], class_weight='balanced', random_state=42)
elif clf_name == "XGBoost":
    final_model = XGBClassifier(n_estimators=best_params["xgb_n_estimators"], learning_rate=best_params["xgb_learning_rate"], random_state=42, eval_metric='logloss', n_jobs=-1)
elif clf_name == "LightGBM":
    final_model = LGBMClassifier(n_estimators=best_params["lgb_n_estimators"], learning_rate=best_params["lgb_learning_rate"], class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)

final_steps.append(('classifier', final_model))
best_pipeline = Pipeline(final_steps)

# 3. Entrenar en todo el conjunto de entrenamiento limpio
best_pipeline.fit(X_train, y_train)

# 4. Predecir en el conjunto de prueba real
y_pred_test = best_pipeline.predict(X_test)

# 5. Mostrar reporte de métricas realistas
print("\n📊 REPORTE DE CLASIFICACIÓN FINAL (Datos de Prueba):")
print(classification_report(y_test, y_pred_test, target_names=['Real Account', 'Bot']))

# ============================================================================
# GUARDADO SEGURO DE ARTEFACTOS EN GOOGLE DRIVE
# ============================================================================
print("\n" + "="*60)
print("💾 GUARDANDO DATOS Y MODELOS EN GOOGLE DRIVE...")
print("="*60)

# Guardar los mejores parámetros e hiperparámetros
joblib.dump(best_params, f"{BASE_PATH}/outputs/models/best_params.pkl")
print("✅ Hiperparámetros guardados en: outputs/models/best_params.pkl")

# Guardar el pipeline entrenado (el "cerebro" del modelo)
joblib.dump(best_pipeline, f"{BASE_PATH}/outputs/models/best_pipeline_model.pkl")
print("✅ Pipeline final guardado en: outputs/models/best_pipeline_model.pkl")

# Exportar los conjuntos de datos de prueba a formato CSV
# Convertimos los arreglos a DataFrames para poder usar .to_csv()
pd.DataFrame(X_test).to_csv(f"{BASE_PATH}/data/processed/X_test.csv", index=False)
pd.DataFrame(y_test).to_csv(f"{BASE_PATH}/data/processed/y_test.csv", index=False)

print("✅ X_test.csv guardado con éxito en: data/processed/")
print("✅ y_test.csv guardado con éxito en: data/processed/")

