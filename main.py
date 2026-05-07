import pandas as pd
from pathlib import Path
from src.ingenieria_variable import generar_indicadores_perfil
from src.validacion_geografica import procesar_geografia_completa
from src.procesamiento_texto import consolidar_ubicaciones

def main():
    try:
        print("--- 🚀 Iniciando Pipeline de Limpieza ---")
        print("Cargando datos crudos...")

        # Búsqueda dinámica para evitar el FileNotFoundError
        raw_dir = Path("data/raw")
        csv_files = list(raw_dir.glob("*.csv"))
        
        if not csv_files:
            print(f"❌ Error: No se encontró ningún archivo CSV en {raw_dir}")
            return
            
        csv_file = csv_files[0]
        print(f"📁 Archivo encontrado: {csv_file.name}")
        df = pd.read_csv(csv_file)

        # Ingeniería de Variables
        print("\nGenerando indicadores de perfil (has_location, etc)...")
        df = generar_indicadores_perfil(df)

        # Procesamiento Geográfico (Puede demorar dependiendo del número de ubicaciones únicas)
        print("\nIniciando validación geográfica de ubicaciones...")
        df = procesar_geografia_completa(df, columna_origen='location')

        # Consolidación de categorías manuales
        print("\nEstandarizando ubicaciones frecuentes...")
        # CORRECCIÓN: Aplicamos la consolidación a 'location_clean', no a 'location'
        df = consolidar_ubicaciones(df, 'location_clean', 'new york', 'new york')
        df = consolidar_ubicaciones(df, 'location_clean', 'australia', 'australia')
        df = consolidar_ubicaciones(df, 'location_clean', 'london', 'london')
        # Aquí se podrían agregar más términos de búsqueda y estandarización según el análisis previo
        
        # CORRECCIÓN: Eliminamos la columna sucia original para evitar confusión en el dataset final
        df = df.drop(columns=['location'], errors='ignore')

        # Renombramos la limpia para que quede como la oficial
        df = df.rename(columns={'location_clean': 'location'})

        # Verificación final de resultados
        print("\nVerificando resultados finales...")
        print(df[['location', 'is_real_location']].head(10))

        # Guardar el dataset final limpio
        print("\nGuardando dataset procesado...")
        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(processed_dir / 'twitter_profiles_final.csv', index=False)
        
        print("¡Proceso completado con éxito!")

    except FileNotFoundError as e:
        print(f"\n❌ FATAL ERROR: El pipeline falló: {e}")

if __name__ == "__main__":
    main()