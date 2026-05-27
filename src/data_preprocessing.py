#data_preprocessing.py
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    """
    Carga el dataset desde un archivo CSV.
    """
    return pd.read_csv(path)


def preprocess_data(df):
    """
    Limpieza y transformación del dataset.
    """

    # Copia de seguridad
    df = df.copy()

    # Eliminar duplicados
    df = df.drop_duplicates()

    # Rellenar valores nulos
    numeric_cols = ["followers_count", "friends_count", "post_count"]

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    categorical_cols = ["lang", "location_clean", "is_real_location"]

    for col in categorical_cols:
        df[col] = df[col].fillna("unknown")

    # Variables booleanas a enteros
    bool_cols = [
        "default_profile_image",
        "profile_use_background_image",
        "verified"
    ]

    for col in bool_cols:
        df[col] = df[col].astype(int)

    # Encoding de variables categóricas
    encoder = LabelEncoder()

    for col in ["lang", "location_clean", "is_real_location"]:
        df[col] = encoder.fit_transform(df[col].astype(str))

    return df


def split_data(df, target="label"):
    """
    Divide el dataset en entrenamiento y prueba.
    """

    X = df.drop(columns=[target, "name", "screen_name", "description", "location", "created_at"])
    y = df[target]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
