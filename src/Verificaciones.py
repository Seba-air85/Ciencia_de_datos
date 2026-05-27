import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Leemos el archivo final de la carpeta processed
df = pd.read_csv('../data/processed/twitter_profiles_cleaned.csv')

print(f"Dataset listo para graficar: {df.shape[0]} registros y {df.shape[1]} columnas.")

#Grafico Distribución de bots vs reales
sns.countplot(data=df, x='label')
plt.title('Distribución de cuentas reales vs bots')
plt.xticks([0,1], ['Real (0)', 'Bot (1)'])
plt.show()

#Grafico Seguidores según tipo de cuenta
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x='label', y='followers_count')
plt.title('Seguidores según tipo de cuenta')
plt.xticks([0,1], ['Real', 'Bot'])
plt.yscale('log')   # importante por valores extremos
plt.show()

#Grafico Seguidos (friends_count) por tipo
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x='label', y='friends_count')
plt.title('Seguidos según tipo de cuenta')
plt.xticks([0,1], ['Real', 'Bot'])
plt.yscale('log')
plt.show()

#Grafico Cantidad de publicaciones
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x='label', y='post_count')
plt.title('Cantidad de publicaciones por tipo de cuenta')
plt.xticks([0,1], ['Real', 'Bot'])
plt.yscale('log')
plt.show()

from src.ingenieria_variable import generar_indicadores_perfil
df = generar_indicadores_perfil(df)

#Grafico ¿Tener ubicación ayuda a detectar bots?
sns.countplot(data=df, x='has_location', hue='label')
plt.title('Presencia de ubicación según tipo de cuenta')
plt.xticks([0,1], ['Sin ubicación', 'Con ubicación'])
plt.show()

from src.validacion_geografica import procesar_geografia_completa
df = procesar_geografia_completa(df)

#Grafico Ubicación real vs falsa
sns.countplot(data=df, x='is_real_location', hue='label')
plt.title('Ubicación válida según tipo de cuenta')
plt.xticks([0,1], ['No válida', 'Válida'])
plt.show()

#Grafico Idioma más frecuente
top_lang = df['lang'].value_counts().head(10).index

plt.figure(figsize=(10,5))
sns.countplot(data=df[df['lang'].isin(top_lang)], x='lang', hue='label')
plt.title('Top idiomas según tipo de cuenta')
plt.xticks(rotation=45)
plt.show()

#Grafico Correlación numérica
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlación entre variables numéricas')
plt.show()
