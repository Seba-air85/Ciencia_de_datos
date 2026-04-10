# Twitter Profiles Preparación

## Descripción general
Este repositorio contiene un preprocesamiento de ciencia de datos inicial. Esta diseñado para limpiar, transformar y modificar el data set **Twitter Profiles** creando un algoritmo/patron para la identificación de perfiles falsos en twitter.

## Estructura del proyecto
```text
ciencia_de_datos/
├── data/
│   ├── raw/                  # Dataset sin limpiar
│   └── processed/            # Dataset procesado y listo para el análisis
├── docs/                     # Reporte técnico
├── notebooks/                # Análisis exploratorio de los datos hecho en Jupyter notebooks
├── src/                      # Código fuente para transformadores personalizados y auditoría
│   ├── __init__.py
│   ├── audit.py              # 
│   └── transformers.py       # 
├── main.py                   # Script Main
├── requirements.txt          # Librerías de python necesarias
└── README.md                 # Instrucciones del proyecto
```

## Instrucciones de configuración
Para replicar localmente el ambiente, seguir los siguientes pasos en su terminal:

### 1. Clonar el repositorio:

```bash
git clone https://github.com/Seba-air85/Ciencia_de_datos.git
cd Ciencia_de_datos
```
### 2. Crear y activar el ambiente virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar las dependencias necesarias:
```bash
pip install -r requirements.txt
```

## Ejecutar
Para ejecutar el proceso automatizado de ETL y preprocesamiento:

```bash
python main.py
```