"""
Geographical validation module using Geopy.
Includes text cleaning to remove emojis and noise, and API validation for real locations.
"""

import pandas as pd
import re
from geopy.geocoders import Nominatim
from geopy.exc import GeopyError

# Configuración del geolocalizador con un user_agent específico para el proyecto
geolocator = Nominatim(user_agent="duoc_twitter_cleaner_v2")

def limpiar_texto_regex(text):
    """
    Cleans text by removing emojis, special characters, and extra whitespace.
    
    Parameters
    ----------
    text : str
        The raw location string from Twitter.
    """
    if pd.isna(text):
        return text
    
    # Pasamos a minúsculas para estandarizar
    text = str(text).lower()
    
    # ELIMINACIÓN DE EMOTES: Filtramos para dejar solo letras, tildes, ñ y comas
    # Esto elimina cualquier símbolo especial, emoji o icono de Twitter
    text = re.sub(r'[^a-zA-ZáéíóúñÁÉÍÓÚÑ\s,]', '', text)
    
    return text.strip()

def validar_ubicacion_real(location_name):
    """
    Validates if a string represents a real geographical location using Nominatim API.
    """
    try:
        # Ignoramos cadenas muy cortas que suelen ser ruido
        if not isinstance(location_name, str) or len(location_name.strip()) < 3:
            return False

        # Consulta a la API con un timeout generoso para evitar bloqueos
        location = geolocator.geocode(location_name, timeout=10)
        return location is not None
    except (GeopyError, Exception):
        # En caso de error de conexión o límite de tasa, marcamos como False
        return False

def procesar_geografia_completa(df, columna_origen='location'):
    """
    Orchestrates the cleaning and validation of all unique locations in the dataset.
    """
    print("🧹 Iniciando limpieza profunda de texto (eliminando emojis)...")
    df['location_clean'] = df[columna_origen].apply(limpiar_texto_regex)
    
    # Extraemos solo valores únicos para no repetir consultas innecesarias a la API
    unique_locations = df['location_clean'].dropna().unique()
    location_dict = {}
    
    print(f"🌍 Validando {len(unique_locations)} ubicaciones únicas con Geopy...")
    print("⚠️  Aviso: Este proceso puede demorar dependiendo de la conexión.")
    
    # Iteramos por el diccionario de únicas (Optimización de red)
    for loc in unique_locations:
        location_dict[loc] = validar_ubicacion_real(loc)
        
    # Mapeamos los resultados de vuelta al DataFrame original
    df['is_real_location'] = df['location_clean'].map(location_dict)
    
    return df