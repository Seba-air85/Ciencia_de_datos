# Reporte Técnico: Preparación de datos - Perfiles de Twitter falsos 🤖

**Equipo:** Sebastian Curi, Sebastian Aird, Luis Hurtubia
**Asignatura:** SCY1101 - Programación para la Ciencia de Datos
**Fecha:** [14-04-2026]

---

## 1. Resumen ejecutivo 
Este proyecto tiene el objetivo de procesar el dataset *Twitter Profiles* para transformar datos crudos en información apta para un futuro modelo de detección de perfiles falsos. Este proceso se realizó utilizando herramientas de manipulación de datos, tales como: La inspección de la estructura, Diagnostico de Nulos, La estrategia de limpieza de datos(Numérica y categórica). Además, se integraron técnicas de validación externa para asegurar la calidad y veracidad de la información procesada.

## 2. Análisis Exploratorio Inicial (EDA)
*Resumen del Dataset:*
El conjunto de datos original cuenta con 8,223 registros y 14 columnas, incluyendo variables de perfil (seguidores, seguidos, cantidad de publicaciones) y metadatos geográficos.

Patrones clave identificados:

* Alta dispersión geográfica: La variable 'location' presenta una gran cantidad de valores únicos y "ruido" (emojis, frases sin sentido, ubicaciones ficticias).

* Asimetría en métricas sociales: Los perfiles sospechosos muestran una relación desproporcionada de seguidos frente a seguidores (estrategia de "follow-back").

* Inconsistencia en metadatos: Existe una correlación preliminar entre la falta de información en el perfil y la etiqueta de "bot".

## 3. Metodología de Transformación
Para asegurar la calidad de los datos, se implementaron las siguientes estrategias:

* Ingeniería de Variables: Se creó el indicador binario 'has_location'. Esta decisión se basa en la regla de negocio de que un usuario legítimo tiende a personalizar su perfil para generar confianza, mientras que las cuentas automatizadas suelen omitir este paso.

* Limpieza Estructurada (Regex): Implementación de expresiones regulares para normalizar cadenas de texto, eliminando caracteres especiales y emojis que generaban ruido en el procesamiento.

* Control de Ruido por Umbral: Se anularon (NaN) las ubicaciones con una frecuencia de aparición unitaria. Esto permite centrar el análisis en grupos geográficos con representatividad poblacional real.

* Estandarización Manual: Unificación de variantes textuales ("NYC" -> "New York") para consolidar categorías y mejorar la precisión de las visualizaciones.

## 4. Resultados y Validación Técnica
La integridad del proceso se aseguró mediante:

* Validación Externa (API Geopy): Se utilizó el servicio de Nominatim para verificar la existencia real de las ubicaciones procesadas. Esto permitió distinguir entre usuarios que viven en lugares reales y bots que utilizan texto aleatorio.

* Análisis de Correlación: Uso de una matriz de calor (Heatmap) para validar estadísticamente que las variables transformadas tienen una relación significativa con la naturaleza de la cuenta.

## 5. Conclusiones y Recomendaciones
* Conclusión: El preprocesamiento de la ubicación resultó ser el factor más determinante para la clasificación. Se confirmó que la omisión de datos geográficos y el bajo volumen de publicaciones son "huellas digitales" claras de perfiles automatizados o tambien llamados 'bots'.

* Eficacia de la Limpieza: La combinación de Regex y validación por API redujo drásticamente el ruido del dataset, permitiendo que las clases (Real vs Bot) sean más fáciles de separar para un futuro modelo de Machine Learning.

* Recomendación: Para futuras etapas, se sugiere integrar el análisis de la fecha de creación de la cuenta, ya que los bots suelen crearse de forma masiva en periodos cortos de tiempo.