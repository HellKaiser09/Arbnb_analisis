import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient
import matplotlib.ticker as ticker

# Configuración de estilo para un aspecto más profesional
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Conexión a MongoDB y carga de datos
cliente = MongoClient('mongodb://localhost:27017')
db = cliente["proyecto"]
coleccion = db["Airbnbs"]
df = pd.DataFrame(list(coleccion.find()))

# Seleccionar solo las columnas numéricas para la correlación
columnas_numericas = ['log_price','accommodates', 'bathrooms','review_scores_rating']

# Diccionario para nombres más descriptivos en español
nombres_descriptivos = {
    'log_price': 'Precio (log)',
    'accommodates': 'Capacidad',
    'bathrooms': 'Baños',
    'review_scores_rating': 'Puntuación'
}

# Asegurarse de que las columnas son numéricas
for col in columnas_numericas:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Calcular la matriz de correlación
matriz_correlacion = df[columnas_numericas].corr()
mask = np.triu(np.ones_like(matriz_correlacion, dtype=bool))

# Crear el mapa de calor con aspecto profesional
plt.figure(figsize=(10, 8))

# Crear el mapa de calor con etiquetas mejoradas
ax = sns.heatmap(
     matriz_correlacion, 
     annot=True,
     cmap='RdBu_r',  # Paleta de colores profesional rojo-azul
     fmt=".2f", 
     mask=mask, 
     vmin=-1, 
     vmax=1,
     linewidths=0.5,
     cbar_kws={"shrink": 0.8, "label": "Coeficiente de Correlación"})

 # Renombrar las etiquetas de los ejes
ax.set_xticklabels([nombres_descriptivos.get(col, col) for col in matriz_correlacion.columns], rotation=45, ha='right')
ax.set_yticklabels([nombres_descriptivos.get(col, col) for col in matriz_correlacion.columns], rotation=0)

# Añadir títulos y subtítulos
plt.title('Matriz de Correlación de Variables', fontsize=16, pad=20, fontweight='bold')
plt.figtext(0.5, 0.01, 'Análisis de datos de Airbnb', ha='center', fontsize=10, fontstyle='italic')

# Añadir una nota explicativa
plt.figtext(0.5, 0.96, 
            'Los valores indican la fuerza y dirección de la relación entre variables (-1 a 1)',
            ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

 # Ajustar márgenes
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

 # Guardar la imagen en alta resolución
plt.savefig('matriz_correlacion_profesional.png', dpi=300, bbox_inches='tight')

 # Mostrar el gráfico
plt.show()


# Gráfico de regresión mejorado para presentación profesional
plt.figure(figsize=(12, 8))

plot_data = df.dropna(subset=['log_price', 'accommodates'])

# Crear el gráfico de regresión con estilo mejorado
ax = sns.regplot(
    x='accommodates', 
    y='log_price', 
    data=plot_data,
    scatter_kws={
        'alpha': 0.6, 
        's': 80, 
        'color': '#2C7FB8',
        'edgecolor': 'white',
    },
    line_kws={
        'color': '#D7301F', 
        'linewidth': 2.5,
        'linestyle': '-'
    },
    ci=95  # Intervalo de confianza del 95%
)

# Añadir títulos y etiquetas con formato profesional
plt.title('Impacto de la Capacidad en el Precio de Alojamientos', 
          fontsize=18, 
          pad=20, 
          fontweight='bold')

plt.xlabel('Capacidad (número de huéspedes)', fontsize=14, labelpad=10)
plt.ylabel('Precio (escala logarítmica)', fontsize=14, labelpad=10)

# Añadir anotación explicativa
r_squared = plot_data[['accommodates', 'log_price']].corr().iloc[0,1]**2
plt.annotate(
    f'R² = {r_squared:.3f}',
    xy=(0.05, 0.95),
    xycoords='axes fraction',
    fontsize=12,
    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8)
)

# Añadir nota explicativa
plt.figtext(
    0.5, 0.01, 
    'La línea roja muestra la tendencia general. Las áreas sombreadas representan el intervalo de confianza del 95%.',
    ha='center', 
    fontsize=10, 
    fontstyle='italic'
)

# Mejorar el aspecto de los ejes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, linestyle='--', alpha=0.3)

# Ajustar los límites de los ejes para mejor visualización
x_min, x_max = plot_data['accommodates'].min(), plot_data['accommodates'].max()
plt.xlim(x_min - 0.5, x_max + 0.5)

# Ajustar márgenes
plt.tight_layout(rect=[0, 0.03, 1, 0.97])

# Guardar la imagen en alta resolución
plt.savefig('regresion_capacidad_precio.png', dpi=300, bbox_inches='tight')

# Mostrar el gráfico
plt.show()
