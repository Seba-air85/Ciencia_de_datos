import pandas as pd
import re

def consolidar_ubicaciones(df, columna, termino_busqueda, nombre_estandar):

    # 1. Identificar variaciones (case-insensitive)
    mask = df[columna].str.contains(termino_busqueda, case=False, na=False)
    variaciones = df.loc[mask, columna].unique()
    
    # 2. Reemplazar todas las variaciones por el nombre estándar
    for v in variaciones:
        df[columna] = df[columna].replace(v, nombre_estandar)
    
    print(f"Consolidación exitosa: '{termino_busqueda}' ahora tiene {df[columna].value_counts()[nombre_estandar]} registros.")
    return df

def limpiar_texto_ubicacion(df, columna_origen='location'):
    
    def clean_logic(text):
        if pd.isna(text):
            return text
        text = text.lower()
        # Mantiene letras, tildes, ñ, espacios y comas
        text = re.sub(r'[^a-zA-Záéíóúñ\s,]', '', text)
        return text.strip()

    df[f'{columna_origen}_clean'] = df[columna_origen].apply(clean_logic)
    return df