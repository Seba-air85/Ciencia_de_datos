"""
Main execution script for Twitter Profile Cleaning Pipeline.
Coordinates variable engineering, emoji removal, and geographic validation.
"""

import pandas as pd
import traceback
from pathlib import Path
from src.ingenieria_variable import generar_indicadores_perfil
from src.validacion_geografica import procesar_geografia_completa
from src.procesamiento_texto import consolidar_ubicaciones

def main():
    """Main pipeline orchestrator."""
    print("="*60)
    print("🐦 PIPELINE DE LIMPIEZA: PERFILES DE TWITTER")
    print("="*60)
    
    try:
        # ============ 1. EXTRACCIÓN ============
        raw_dir = Path("data/raw")
        csv_files = list(raw_dir.glob("*.csv"))
        
        if not csv_files:
            print(f"❌ Error: No se encontró el dataset en {raw_dir}")
            return
            
        csv_file = csv_files[0]
        print(f"📥 Cargando datos: {csv_file.name}")
        df = pd.read_csv(csv_file)

        # ============ 2. AUDITORÍA (Placeholder) ============
        # Fase pendiente para el equipo: Implementar audit.py
        print("\n🔍 Fase 2: Auditoría de integridad (Pendiente)")

        # ============ 3. INGENIERÍA DE VARIABLES ============
        print("\n🛠️  Fase 3: Generando indicadores de perfil")
        # Crea flags como 'has_location' para el análisis inicial
        df = generar_indicadores_perfil(df)

        # ============ 4. VALIDACIÓN GEOGRÁFICA ============
        print("\n🌍 Fase 4: Validación geográfica y limpieza de emojis")
        # Aquí eliminamos los emojis y validamos contra mapas reales
        df = procesar_geografia_completa(df, columna_origen='location')

        # ============ 5. CONSOLIDACIÓN DE TEXTO ============
        print("\n✍️  Fase 5: Estandarización de nombres frecuentes")
        # Unificamos variaciones de ciudades importantes (ej. London, Londres)
        for ciudad in ['new york', 'australia', 'london']:
            df = consolidar_ubicaciones(df, 'location_clean', ciudad, ciudad)
        
        # ============ 6. OPTIMIZACIÓN Y LIMPIEZA FINAL ============
        print("\n⚙️  Fase 6: Estructuración del dataset final")
        # Eliminamos la columna original con ruido y renombramos la limpia
        df = df.drop(columns=['location'], errors='ignore')
        df = df.rename(columns={'location_clean': 'location'})
        
        # Guardado en processed
        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        output_path = processed_dir / 'twitter_profiles_final.csv'
        df.to_csv(output_path, index=False)
        
        print("\n" + "="*60)
        print("✅ PROCESO COMPLETADO EXITOSAMENTE")
        print(f"📊 Dataset final guardado en: {output_path}")
        print("="*60)

    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()