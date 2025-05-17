# 🏡 Análisis de Datos de Airbnb
Este proyecto consiste en el análisis exploratorio y modelado predictivo de precios de alojamiento utilizando un dataset de Airbnb. Se trabajó con un archivo `.csv` importado a MongoDB para realizar procesos de limpieza, transformación, análisis estadístico y visualización de los datos, con el objetivo de identificar los factores más relevantes que afectan los precios de renta.

---
## 🧪 Herramientas utilizadas

- **Python**: pandas, numpy, matplotlib, seaborn, scikit-learn  
- **Bases de datos**: MongoDB  
- **Modelado estadístico**: Regresión lineal, Ridge, Lasso  
- **Visualización**: Gráficos de distribución, correlación, importancia de variables, etc

## 🔍 Análisis Exploratorio de Datos

Durante el análisis exploratorio se llevaron a cabo los siguientes pasos:
- **Tratamiento de valores faltantes**, especialmente en variables como `host_response_rate`.
- **Conversión y estandarización de tipos de datos** para facilitar el modelado numérico.
- **Identificación de outliers** mediante análisis de asimetría y curtosis.
- **Visualización de patrones** en variables clave: `accommodates`, `bathrooms`, `bedrooms`, `beds`, `room_type`, `property_type`.
---

## 📊 Hallazgos del análisis exploratorio
- Las características físicas como número de habitaciones, baños y tipo de propiedad son **los principales determinantes del precio**.
- Variables de percepción (reviews, ratings, etc.) tienen **bajo impacto** en el precio.
- **La ubicación geográfica** (ciudad y vecindario) influye notablemente en los precios, siendo San Francisco una de las más caras.
- **Las propiedades completas** tienen precios significativamente más altos que habitaciones privadas o compartidas.

### 🔸 Visualizaciones del análisis:
![importancia_![heatmap_correlaciones](https://github.com/user-attachments/assets/365d95fc-28c3-4621-b7af-b4c01b6c6a9a)
variables_ridge](https://github.com/user-attachments/assets/4bb93d25-0a4a-4453-85f5-4abb58a67bee)
![matriz_correlacion_profesional](https://github.com/user-attachments/assets/1822624f-f39a-46ad-a846-6e581a08526e)
![distribucion_precios_alojamiento](https://github.com/user-attachments/assets/0c3742f5-5638-46fc-8564-607d2df0a694)
![Top10 vecindarios](https://github.com/user-attachments/assets/13847024-9b3f-4bb2-82a8-bb5b8a3d1c38)
![Distribucion_precios_ciudad](https://github.com/user-attachments/assets/8577c5db-278e-4480-ba22-83b23cc58e7b)
![dis![precio_promedio](https://github.com/user-attachments/assets/876a7061-54b4-42af-9f23-1dc7da6797a5)
tribucion_precios_huesped](https://github.com/user-attachments/assets/8b6cd47d-62f9-440b-87ca-3acbce7b2778)

---
## 📈 Modelado Predictivo

Se aplicaron modelos de regresión para estimar el precio en escala logarítmica:

### 🔹 Regresión Lineal (OLS)
- **R²** = 0.622  
- **RMSE** = 0.441  
- **MAE** = 0.322  
- Variables significativas: `accommodates`, `bathrooms`, `room_type`, `city`, `neigh_price_group`.

### 🔹 Modelos Regularizados
- **Ridge**: mejor alpha = 61.36  
- **Lasso**: mejor alpha = 0.001  
- Los coeficientes identificados por Lasso y Ridge fueron consistentes con el modelo OLS.

> Se seleccionó el modelo **Ridge** para interpretar la importancia de variables por su buen equilibrio entre regularización e interpretabilidad.

---

## 🧠 Visualizaciones del modelo

### Verificación de modelos y desempeño

![validacion_cruzada](https://github.com/user-attachments/assets/65a5f228-1e97-48d6-89dd-a5472267082a)
![curva_aprendizaje](https://github.com/user-attachments/assets/cd6c4564-d269-4629-b9ac-b1ab552154d0)
![comparacion_rmse](https://github.com/user-attachments/assets/c09e3f75-2684-41fd-9363-912440fab0f2)
![evaluacion_predicciones](https://github.com/user-attachments/assets/3d525bf3-fe80-4975-b3dd-a8fe914624d4)

## 📂 Estructura del proyecto
📁 Airbnb_Analysis
│
├── data/
│ └── train.csv
│
├── python_analisis/
│ ├── importacion.py
│ ├── limpieza.py
│ └── analisis.py
│ ├── modelo.py
│ ├── graficoDEA.py
│ └── graficos.py
│
├── images/
│ └── (gráficos exportados)
│
└── README.md
