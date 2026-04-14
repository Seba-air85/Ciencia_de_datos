import pandas as pd
import re
import time
from geopy.geocoders import Nominatim
from geopy.exc import GeopyError

# Configuración del geolocalizador
# Se define fuera de la función para no reiniciarlo en cada llamada
geolocator = Nominatim(user_agent="duoc_student_project_geochecker")

def limpiar_texto_regex(text):
   
    if pd.isna(text):
        return text
    text = str(text).lower()
    # Mantiene caracteres alfabéticos y comas para ayudar a la API de mapas
    text = re.sub(r'[^a-zA-Záéíóúñ\s,]', '', text)
    return text.strip()

def validar_ubicacion_real(location_name):
    
    try:
        # Validaciones preventivas para ahorrar tiempo de API
        if not isinstance(location_name, str) or len(location_name.strip()) < 3:
            return False

        location = geolocator.geocode(location_name, timeout=10)
        return location is not None
    except (GeopyError, Exception) as e:
        # En caso de error de red o timeout, retorna False por seguridad
        return False

def procesar_geografia_completa(df, columna_origen='location'):
    
    print("Iniciando limpieza de texto...")
    df['location_clean'] = df[columna_origen].apply(limpiar_texto_regex)
    
    # Optimización: Solo validamos ubicaciones únicas para no saturar la API
    unique_locations = df['location_clean'].dropna().unique()
    location_dict = {}
    
    print(f"Validando {len(unique_locations)} ubicaciones únicas con Geopy (esto tomará tiempo)...")
    
    for loc in unique_locations:
        location_dict[loc] = validar_ubicacion_real(loc)
        # Respetar políticas de uso de la API gratuita
        time.sleep(1) 
    
    df['is_real_location'] = df['location_clean'].map(location_dict)
    print("✅ Proceso geográfico finalizado.")
    return df