import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient

# Configuración inicial
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

# Conexión a MongoDB y carga de datos
cliente = MongoClient('mongodb://localhost:27017')
db = cliente["proyecto"]
coleccion = db["Airbnbs"]
df = pd.DataFrame(list(coleccion.find()))

# Limpieza de datos - asegurar que las columnas numéricas sean correctas
numeric_cols = ['log_price', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 
               'number_of_reviews', 'review_scores_rating']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')


plt.figure(figsize=(15, 10))
variables = ['log_price', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'review_scores_rating']


## --------------------------------------------
## 1. Distribución de precios (combinación de histograma y boxplot)
## --------------------------------------------
plt.figure(figsize=(14, 7))
sns.histplot(df['log_price'], kde=True, bins=30, color='royalblue', 
             edgecolor='white', linewidth=0.5)
plt.title('Distribución de Precios en Airbnb (Escala Logarítmica)', fontsize=16, pad=20)
plt.xlabel('Logaritmo del Precio (USD)', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.grid(True, alpha=0.3)
plt.axvline(df['log_price'].mean(), color='red', linestyle='--', 
            label=f'Media: {df["log_price"].mean():.2f}')
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 8))
corr = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", 
            mask=mask, vmin=-1, vmax=1, linewidths=0.5,
            annot_kws={"size": 11}, cbar_kws={"shrink": .8})
plt.title('Matriz de Correlación entre Variables Numéricas', fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)
plt.tight_layout()
plt.show()

# CAMBIO: Boxplot a Violinplot para tipo de alojamiento
plt.figure(figsize=(12, 7))
room_order = df.groupby('room_type')['log_price'].median().sort_values().index
sns.violinplot(x='room_type', y='log_price', data=df, order=room_order, 
            palette='Set2', width=0.6, inner='quartile')
plt.title('Distribución de Precios por Tipo de Alojamiento', fontsize=16, pad=20)
plt.xlabel('Tipo de alojamiento', fontsize=12)
plt.ylabel('Logaritmo del Precio (USD)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 7))

# Calcular la mediana de precios por vecindario y obtener los 10 más caros
top_neigh_prices = df.groupby('neighbourhood')['log_price'].median().nlargest(10).sort_values(ascending=False)

# Crear gráfico de barras horizontales
bars = plt.barh(top_neigh_prices.index, top_neigh_prices.values, 
                color=sns.color_palette("Set3", len(top_neigh_prices)))

# Añadir los valores en cada barra
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.05, bar.get_y() + bar.get_height()/2, 
             f'{width:.2f}', 
             ha='left', va='center', fontsize=11)

# Personalización del gráfico
plt.title('Top 10 Vecindarios con Precios Más Altos (Mediana)', fontsize=16, pad=20)
plt.xlabel('Logaritmo del Precio (USD)', fontsize=12)
plt.ylabel('Vecindario', fontsize=12)
plt.grid(True, axis='x', alpha=0.3)
plt.gca().invert_yaxis()  # Mostrar el más caro arriba
plt.tight_layout()
plt.show()


# Configuración de estilo
plt.style.use('seaborn-v0_8')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12



# Limpieza de datos - asegurar que las columnas numéricas sean correctas
numeric_cols = ['log_price', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 
               'number_of_reviews', 'review_scores_rating']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 1. Gráfico de violín: Distribución de precios por tipo de habitación
plt.figure(figsize=(14, 8))
sns.violinplot(x='room_type', y='log_price', data=df, 
               palette='viridis', inner='quartile', 
               order=df.groupby('room_type')['log_price'].median().sort_values().index)
plt.title('Distribución de Precios por Tipo de Habitación', fontsize=16)
plt.xlabel('Tipo de Habitación', fontsize=14)
plt.ylabel('Logaritmo del Precio (USD)', fontsize=14)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('violin_room_type.png')
plt.show()

# 2. Gráfico de violín: Distribución de precios por ciudad
plt.figure(figsize=(14, 8))
sns.violinplot(x='city', y='log_price', data=df, 
               palette='plasma', inner='quartile',
               order=df.groupby('city')['log_price'].median().sort_values().index)
plt.title('Distribución de Precios por Ciudad', fontsize=16)
plt.xlabel('Ciudad', fontsize=14)
plt.ylabel('Logaritmo del Precio (USD)', fontsize=14)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('violin_city.png')
plt.show()



# 4. Gráfico de violín: Distribución de precios por capacidad de huéspedes
plt.figure(figsize=(14, 8))
# Filtrar para mostrar solo hasta 8 huéspedes (para mejor visualización)
df_filtered = df[df['accommodates'] <= 8].copy()
sns.violinplot(x='accommodates', y='log_price', data=df_filtered, 
               palette='cividis', inner='quartile')
plt.title('Distribución de Precios por Capacidad de Huéspedes', fontsize=16)
plt.xlabel('Número de Huéspedes', fontsize=14)
plt.ylabel('Logaritmo del Precio (USD)', fontsize=14)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('violin_accommodates.png')
plt.show()

# 5. Gráfico de violín: Comparación de precios y ratings por tipo de habitación
plt.figure(figsize=(16, 10))

# Crear subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Ordenar por mediana de precio
room_order = df.groupby('room_type')['log_price'].median().sort_values().index

# 6. Gráfico de barras: Precio promedio por ciudad y tipo de habitación
plt.figure(figsize=(16, 10))

# Calcular precio promedio por ciudad y tipo de habitación
city_room_price = df.groupby(['city', 'room_type'])['log_price'].mean().reset_index()

# Crear gráfico de barras agrupadas
sns.barplot(x='city', y='log_price', hue='room_type', data=city_room_price, 
            palette='Set2')

plt.title('Precio Promedio por Ciudad y Tipo de Habitación', fontsize=16)
plt.xlabel('Ciudad', fontsize=14)
plt.ylabel('Logaritmo del Precio Promedio (USD)', fontsize=14)
plt.legend(title='Tipo de Habitación', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('barras_ciudad_tipo.png')
plt.show()


