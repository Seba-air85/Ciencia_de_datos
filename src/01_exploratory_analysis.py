import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargamos el dataset
df = pd.read_csv('../data/raw/twitter_profiles_original.csv')

#Se muestra la informacion del dataset
print(f"Dimensiones del dataset: {df.shape}")
df.head()

df.isna().sum()

#Con esto nos encargamos de 'verified'
df['verified'] = df['verified'].fillna(False)

#Eliminamos las cuentas sin fechas
df = df.dropna(subset=['created_at'])

#Se eliminan los NaN por 0
cols_conteo = ['followers_count', 'friends_count', 'post_count']
df[cols_conteo] = df[cols_conteo].fillna(0)

#Se agrega un texto predeterminado a los Nan
df['description'] = df['description'].fillna('Usuario no entregó descripción de perfil')

#Verificamos que se hayan eliminado los nulos correspondientes
df.isna().sum()

location_counts = df['location'].value_counts()
locations_appearing_once = location_counts[location_counts == 2] #Si la ubicacion no la comparten al menos 3 personas, no sera relevante

locations_to_remove = locations_appearing_once.index
df['location'] = df['location'].replace(locations_to_remove, np.nan)

print("Valores nulos después de eliminar locaciones de una sola aparición:")
print(df['location'].isna().sum())

location_counts = df['location'].value_counts()
locations_appearing_once = location_counts[location_counts == 1]

print(f"El número de locaciones que aparecen solo una vez es: {len(locations_appearing_once)}")

# Importas tu herramienta (asegúrate de que la carpeta src sea tratada como módulo)
from src.procesamiento_texto import consolidar_ubicaciones

# Unificas New York
df = consolidar_ubicaciones(df, 'location', 'new york', 'new york')

# Unificas Australia
df = consolidar_ubicaciones(df, 'location', 'australia', 'australia')

# Unificas Italia
df = consolidar_ubicaciones(df, 'location', 'italy', 'italy')

# Verificas el resultado
display(df['location'].value_counts().head(10))

from src.ingenieria_variables import generar_indicadores_perfil

df = generar_indicadores_perfil(df)

from src.validacion_geografica import limpiar_texto_regex

df['location_clean'] = df['location'].apply(limpiar_texto_regex)

# Al final del Notebook de Limpieza
df.to_csv('../data/processed/twitter_profiles_cleaned.csv', index=False)
