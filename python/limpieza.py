# ----------------------------------------------
# Manipulación de datos
import pandas as pd
import numpy as np
# Conexión a base de datos
from pymongo import MongoClient
# Análisis estadístico
from scipy.stats import zscore
import ast

"""
1. CONEXIÓN A LA BASE DE DATOS MONGODB
 --------------------------------------
Se establece conexión con el servidor MongoDB local en el puerto por defecto (27017)
Base de datos llamada 'proyecto'
Colección llamada 'Airbnbs'
"""
cliente = MongoClient('mongodb://localhost:27017')
db = cliente["proyecto"]
coleccion = db["Airbnbs"]
"""
2. CARGA DE DATOS A DATAFRAME
-----------------------------
Convertimos los documentos de MongoDB en un DataFrame de pandas para facilitar el análisis
Primero convertimos el cursor a lista para evitar problemas de timeout en consultas grandes
"""

df = pd.DataFrame(list(coleccion.find()))

"""
3. EXPLORACIÓN INICIAL DE DATOS
-------------------------------
head(6) muestra las primeras 6 filas
Mostramos las primeras filas para entender la estructura de los datos
"""
print("\n" + "="*60)
print("EXPLORACIÓN INICIAL DEL DATAFRAME - PRIMERAS 6 FILAS")
print("="*60 + "\n")
print(df.head(6))  

"""
4. DETECCIÓN DE VALORES FALTANTES
--------------------------------
Calculamos y mostramos la cantidad de valores nulos por columna
Esto nos ayuda a identificar qué variables requieren tratamiento
isna() detecta valores nulos, sum() los cuenta
"""

print("\n" + "="*60)
print("ANÁLISIS DE VALORES FALTANTES (MISSING VALUES)")
print("="*60 + "\n")


print("Cantidad de valores nulos por columna:")
print("-"*45)
print(df.isna().sum())  
print("-"*45)


"""
5. TRATAMIENTO DE VALORES FALTANTES
-----------------------------------
Estrategia de imputación:
Para variables numéricas: usamos la mediana (robusta a outliers)
Para variables categóricas: usamos la moda (valor más frecuente)
Para porcentajes: hacemos limpieza especial antes de imputar
"""

print("\n" + "="*60)
print("IMPUTACIÓN DE VALORES FALTANTES")
print("="*60 + "\n")

# Imputación de variables numéricas con la mediana
df['bathrooms'].fillna(df['bathrooms'].median(), inplace=True)  # inplace=True modifica el DataFrame directamente
df['bedrooms'].fillna(df['bedrooms'].median(), inplace=True)
df['beds'].fillna(df['beds'].median(), inplace=True)


df['neighbourhood'].fillna(df['neighbourhood'].mode()[0], inplace=True)
df['zipcode'].fillna(df['zipcode'].mode()[0], inplace=True)


# Imputación para puntuaciones de reseñas
df['review_scores_rating'].fillna(df['review_scores_rating'].median(), inplace=True)
df['amenities'] = df['amenities'].str.replace(r'[{}"]', '', regex=True).str.split(',')

"""
Verificación post-imputación
Deberíamos ver 0 valores nulos en las columnas tratadas
"""
print("\n" + "="*60)
print("VERIFICACIÓN POST-IMPUTACIÓN")
print("="*60 + "\n")
print("Valores faltantes después del tratamiento:")
print("-"*45)
print(df.isna().sum())  
print("-"*45)

"""
6. NORMALIZACIÓN DE VARIABLES NUMÉRICAS
-------------------------------------- 
Definimos las columnas a normalizar
Z-score es adecuado porque:
- Mantiene la forma de la distribución original
- Es interpretable (número de desviaciones estándar desde la media)
- Maneja relativamente bien los outliers
"""

print("\n" + "="*60)
print("NORMALIZACIÓN CON Z-SCORE")
print("="*60 + "\n")


columnas_estandarizar = [   
    'bathrooms',           
    'bedrooms',            
    'beds',                
    'review_scores_rating' 
]

df[columnas_estandarizar] = df[columnas_estandarizar].astype(float)


df[columnas_estandarizar] = df[columnas_estandarizar].apply(zscore)

# Mostramos el resultado de la normalización
print("\nDatos normalizados (primeras 5 filas):")
print("-"*60)
print(df[columnas_estandarizar].head())
print(df.isnull().sum())  
print("-"*60)

"""
7. INSERTAR DATOS LIMPIOS EN MONGODB
---------------------------------
Volvemos a convertir el DataFrame a documentos y los insertamos en MongoDB
"""

print("\n" + "="*60)
print("INSERTAR DATOS LIMPIOS EN MONGODB")
print("="*60 + "\n")

for index, row in df.iterrows():
    doc = row.drop('_id').to_dict()  # evita modificar el _id
    coleccion.update_one({'_id': row['_id']}, {'$set': doc})
