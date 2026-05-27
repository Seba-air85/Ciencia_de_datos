"""
Main execution script for Twitter Profile Cleaning and Machine Learning Pipeline.
Coordinates preprocessing, feature engineering, model training,
evaluation, hyperparameter optimization, and result saving.
"""

import traceback
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# LIMPIEZA Y PREPROCESAMIENTO

from src.ingenieria_variable import generar_indicadores_perfil
from src.validacion_geografica import procesar_geografia_completa
from src.procesamiento_texto import consolidar_ubicaciones

# PREPROCESSING

from src.data_preprocessing import preprocess_data

# MODELOS

from src.model_training import (
    train_logistic_regression,
    train_random_forest,
    train_decision_tree,
    train_svm
)

# EVALUACIÓN

from src.model_evaluation import evaluate_model

# OPTIMIZACIÓN

from src.hyperparameter_tuning import optimize_random_forest


def main():
    """Main pipeline orchestrator."""

    print("=" * 70)
    print(" TWITTER PROFILE MACHINE LEARNING PIPELINE")
    print("=" * 70)

    try:

        # 1. CARGA DEL DATASET

        print("\n FASE 1: Cargando dataset")

        raw_dir = Path("data/raw")
        csv_files = list(raw_dir.glob("*.csv"))

        if not csv_files:
            print(f" Error: No se encontró ningún CSV en {raw_dir}")
            return

        csv_file = csv_files[0]

        print(f" Dataset encontrado: {csv_file.name}")

        df = pd.read_csv(csv_file)

        print(f" Dataset cargado correctamente")
        print(f" Filas: {df.shape[0]}")
        print(f" Columnas: {df.shape[1]}")

        # 2. INGENIERÍA DE VARIABLES

        print("\n FASE 2: Ingeniería de variables")

        df = generar_indicadores_perfil(df)

        print(" Indicadores generados")

        # 3. VALIDACIÓN GEOGRÁFICA

        print("\n FASE 3: Validación geográfica")

        df = procesar_geografia_completa(
            df,
            columna_origen='location'
        )

        print(" Validación geográfica completada")

        # 4. CONSOLIDACIÓN DE TEXTO

        print("\n FASE 4: Consolidación de ubicaciones")

        for ciudad in ['new york', 'australia', 'london']:

            df = consolidar_ubicaciones(
                df,
                'location_clean',
                ciudad,
                ciudad
            )

        print(" Consolidación completada")

        # 5. LIMPIEZA FINAL

        print("\n FASE 5: Limpieza final")

        df = df.drop(columns=['location'], errors='ignore')

        df = df.rename(
            columns={'location_clean': 'location'}
        )

        print(" Limpieza final completada")

        # 6. PREPROCESAMIENTO ML

        print("\n FASE 6: Preprocesamiento para ML")

        X, y = preprocess_data(df)

        print(" Preprocesamiento completado")

        # 7. TRAIN / TEST SPLIT

        print("\n FASE 7: Separación train/test")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        print(" Datos separados correctamente")

        # 8. ENTRENAMIENTO DE MODELOS

        print("\n FASE 8: Entrenamiento de modelos")

        logistic_model = train_logistic_regression(
            X_train,
            y_train
        )

        random_forest_model = train_random_forest(
            X_train,
            y_train
        )

        decision_tree_model = train_decision_tree(
            X_train,
            y_train
        )

        svm_model = train_svm(
            X_train,
            y_train
        )

        print(" Modelos entrenados correctamente")

        # 9. EVALUACIÓN DE MODELOS

        print("\n FASE 9: Evaluación de modelos")

        print("\n Logistic Regression")
        evaluate_model(
            logistic_model,
            X_test,
            y_test
        )

        print("\n Random Forest")
        evaluate_model(
            random_forest_model,
            X_test,
            y_test
        )

        print("\n Decision Tree")
        evaluate_model(
            decision_tree_model,
            X_test,
            y_test
        )

        print("\n SVM")
        evaluate_model(
            svm_model,
            X_test,
            y_test
        )

        print("\n Evaluación completada")

        # 10. OPTIMIZACIÓN DE HIPERPARÁMETROS

        print("\n FASE 10: Optimización de Random Forest")

        best_rf_model = optimize_random_forest(
            X_train,
            y_train
        )

        print(" Optimización completada")

        # 11. GUARDADO DEL DATASET FINAL

        print("\n FASE 11: Guardando dataset procesado")

        processed_dir = Path("data/processed")

        processed_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path = processed_dir / "twitter_profiles_final.csv"

        df.to_csv(
            output_path,
            index=False
        )

        print(f" Dataset guardado en:")
        print(f" {output_path}")

        # FINAL
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETADO EXITOSAMENTE")
        print("=" * 70)

    except Exception as e:

        print("\n ERROR EN EL PIPELINE")
        print(f" {e}")

        traceback.print_exc()


if __name__ == "__main__":
    main()
