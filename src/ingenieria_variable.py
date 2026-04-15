import pandas as pd


# Funciones para ingeniería de variables y generación de indicadores
def generar_indicadores_perfil(df):

    # Indicador de ubicación
    df['has_location'] = df['location'].apply(
        lambda x: 0 if pd.isna(x) or str(x).strip() == "" else 1
    )
    return df