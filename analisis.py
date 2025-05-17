# Manipulación de datos
import pandas as pd
import numpy as np
# Conexión a base de datos
from pymongo import MongoClient
from scipy.stats import chi2_contingency

cliente = MongoClient('mongodb://localhost:27017')
db = cliente["proyecto"]
coleccion = db["Airbnbs"]
df = pd.DataFrame(list(coleccion.find()))
columnas_numericas = [
    'log_price',
    'accommodates',
    'bathrooms',
    'bedrooms',
    'beds',
    'review_scores_rating',
    'number_of_reviews'
]

columnas_categoricas = ['property_type', 'room_type', 'bed_type', 'city', 'amenities']

"""
# 7. ANÁLISIS ESTADÍSTICO
# ----------------------
# Calculamos medidas de tendencia central
# Calculamos estadísticas descriptivas completas
# Añadimos medidas de forma de la distribución
"""
print("\n" + "="*60)
print("ESTADÍSTICAS DESCRIPTIVAS")
print("="*60 + "\n")


print("\nMedidas de tendencia central:")
print("-"*45)
print("Media:")
print(df[columnas_numericas].mean()) # mean() calcula la media aritmética

print("\nModa:")
print(df[columnas_numericas].mode().iloc[0]) # mode() devuelve la moda, iloc[0] toma la primera

print("\nMediana:")
print(df[columnas_numericas].median()) # median() calcula la mediana

desc_stats = df[columnas_numericas].describe().transpose() # describe() resume estadísticas, transpose() para mejor visualización
desc_stats['skewness'] = df[columnas_numericas].skew() # skewness mide asimetría
desc_stats['kurtosis'] = df[columnas_numericas].kurt() # kurtosis mide el "pico" de la distribución

columnas_categoricas = ['property_type', 'room_type', 'bed_type', 'city']

print("\nModa de columnas categóricas:")
print("-"*45)
for col in columnas_categoricas:
    print(f"{col}: {df[col].mode().iloc[0]}")

print("\nResumen estadístico completo:")
print("-"*60)
print(desc_stats)
print("-"*60)

correlacion = df[columnas_numericas].corr()
print("\nMatriz de correlación:")
print("-"*60)
print(correlacion.loc['log_price'])  # Muestra solo la fila de log_price
print("-"*60)


