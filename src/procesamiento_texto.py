import pandas as pd


# Funciones para procesamiento de texto y consolidación de ubicaciones
def consolidar_ubicaciones(df, columna, termino_busqueda, nombre_estandar):

    mask = df[columna].str.contains(termino_busqueda, case=False, na=False)
    variaciones = df.loc[mask, columna].unique()
    
    for v in variaciones:
        df[columna] = df[columna].replace(v, nombre_estandar)
    
    print(f"Consolidación exitosa: '{termino_busqueda}' ahora tiene {df[columna].value_counts()[nombre_estandar]} registros.")
    return df