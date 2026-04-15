import pandas as pd
from src.ingenieria_variable import generar_indicadores_perfil
from src.validacion_geografica import procesar_geografia_completa
from src.procesamiento_texto import consolidar_ubicaciones

def main():
    # Carga de datos
    print("Cargando datos crudos...")
    df = pd.read_csv('data/raw/twitter_profiles.csv')

    # Ingeniería de Variables
    print("Generando indicadores de perfil (has_location, etc)...")
    df = generar_indicadores_perfil(df)

    # Procesamiento Geográfico (Puede demorar dependiendo del número de ubicaciones únicas)
    print("Iniciando validación geográfica de ubicaciones...")
    df = procesar_geografia_completa(df)

    # Consolidación de categorías manuales
    print("Estandarizando ubicaciones frecuentes...")
    df = consolidar_ubicaciones(df, 'location', 'new york', 'new york')
    df = consolidar_ubicaciones(df, 'location', 'australia', 'australia')
    df = consolidar_ubicaciones(df, 'location', 'london', 'london')
     # Aquí se podrían agregar más términos de búsqueda y estandarización según el análisis previo
    
    # Verificación final de resultados
    print("Verificando resultados finales...")
    print(df[['location', 'location_clean', 'is_real_location']].head(10))

    # Guardar el dataset final limpio
    print("Guardando dataset procesado...")
    df.to_csv('data/processed/twitter_profiles_final.csv', index=False)
    
    print("¡Proceso completado con éxito!")

if __name__ == "__main__":
    main()