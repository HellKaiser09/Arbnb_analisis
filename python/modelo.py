#=== Importación de librerías ===
import pandas as pd
import numpy as np
from pymongo import MongoClient
import re

# Modelos y métricas de machine learning
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Análisis estadístico
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns

# === Conexión a MongoDB y carga de datos ===
cliente = MongoClient('mongodb://localhost:27017')
db = cliente["proyecto"]
coleccion = db["Airbnbs"]
df = pd.DataFrame(list(coleccion.find()))  # Convertimos la colección de MongoDB en un DataFrame de pandas

# === Selección de variables numéricas relevantes ===
columnas_numericas = ['accommodates', 'bathrooms', 'review_scores_rating']

# === Clasificación de barrios por nivel de precio promedio (en terciles) ===
neigh_price_mean = df.groupby('neighbourhood')['log_price'].mean()  # Precio promedio por vecindario
price_bins = pd.qcut(neigh_price_mean, q=3, labels=["bajo", "medio", "alto"])  # Agrupación en terciles
neigh_group_map = price_bins.to_dict()
df['neigh_price_group'] = df['neighbourhood'].map(neigh_group_map)  # Asignación de grupo a cada fila

# === Normalización de los amenities (comodidades) ===
# Mapeo de comodidades con nombres estandarizados
amenity_mapping = {
    r'wifi|wireless internet': 'Internet',
    r'air conditioning|a/c': 'Aire Acondicionado',
    r'kitchen': 'Cocina',
    r'heating': 'Calefaccion',
    r'family|kid friendly': 'Familiar',
    r'essentials': 'Basicos',
    r'hair dryer': 'Secador Pelo',
    r'iron': 'Plancha',
    r'parking': 'Estacionamiento',
    r'pool': 'Piscina'
}

# Función para estandarizar la lista de amenities
def normalize_amenities(amenity_list):
    normalized = set()
    for amenity in amenity_list:
        amenity_lower = amenity.lower()
        matched = False
        for pattern, standard_name in amenity_mapping.items():
            if re.search(pattern, amenity_lower):
                normalized.add(standard_name)
                matched = True
                break
        if not matched:
            normalized.add(amenity)
    return list(normalized)

df['amenities_normalized'] = df['amenities'].apply(normalize_amenities)

# === Creación de variables dummy para amenities ===
# Expandimos y convertimos las amenities normalizadas en variables binarias
amenities_dummies = pd.get_dummies(df['amenities_normalized'].explode()).groupby(level=0).sum()

# Filtramos las amenities que aparecen en un rango intermedio de frecuencia (5%-95%)
amenity_counts = amenities_dummies.sum()
relevant_amenities = amenity_counts[(amenity_counts > 0.05 * len(df)) & (amenity_counts < 0.95 * len(df))].index
amenities_dummies = amenities_dummies[relevant_amenities]

# === Codificación de variables categóricas ===
columnas_categoricas = ['room_type', 'city', 'neigh_price_group']
dummies = pd.get_dummies(df[columnas_categoricas], drop_first=True, prefix=columnas_categoricas)

# === Construcción del dataset final (X) y la variable objetivo (y) ===
X = pd.concat([df[columnas_numericas], dummies, amenities_dummies], axis=1)

# Eliminamos variables redundantes o problemáticas (manualmente seleccionadas)
X = X.drop(columns=[
    'city_NYC', 'city_LA', 'city_DC', 'Bathtub', 'Bed linens', 'Calefaccion',
    'Dog(s)', 'First aid kit', 'Piscina', 'Plancha', 'Refrigerator', 'Stove',
    'Washer', 'Hot water', 'Oven', 'Pets allowed', 'Wheelchair accessible',
    'Dishes and silverware', 'translation missing: en.hosting_amenity_49',
    'Hot tub', 'Microwave'
], errors='ignore')  # Ignora errores si alguna columna no existe

# Variable objetivo
y = df['log_price']

# Conversión a float para asegurar compatibilidad con los modelos
X = X.astype(float)
y = y.astype(float)

# === División del dataset en entrenamiento y prueba ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalado de las variables para regularización
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Regresión Lineal ===
reg = LinearRegression().fit(X_train, y_train)
y_pred = reg.predict(X_test)

# Métricas del modelo
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n[Regresión Lineal]")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R²: {r2}")
print(f"Sesgo (Intercepto): {reg.intercept_}")

# === Modelos con regularización (Ridge y Lasso) ===
ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 100)).fit(X_train_scaled, y_train)
lasso_cv = LassoCV(alphas=np.logspace(-3, 3, 100)).fit(X_train_scaled, y_train)

print(f"\n[Regularización]")
print(f"Mejor alpha Ridge: {ridge_cv.alpha_}")
print(f"Mejor alpha Lasso: {lasso_cv.alpha_}")

# Coeficientes obtenidos por Lasso
lasso_coefs = pd.Series(lasso_cv.coef_, index=X.columns)
variables_eliminadas = lasso_coefs[lasso_coefs == 0].index.tolist()

print("\n[Coeficientes Lasso]")
print(lasso_coefs)
print("\n[Variables eliminadas por Lasso (coef = 0)]")
print(variables_eliminadas)

# === Modelo OLS para análisis estadístico completo ===
X_const = sm.add_constant(X)
modelo_ols = sm.OLS(y, X_const).fit()

print("\n[Modelo OLS]")
print(modelo_ols.summary())

# RMSE sobre todo el conjunto
y_pred_ols = modelo_ols.predict(X_const)
rmse_ols = np.sqrt(mean_squared_error(y, y_pred_ols))
print(f"RMSE completo OLS: {rmse_ols}")

# === Cálculo del Factor de Inflación de la Varianza (VIF) ===
print("\n[Análisis VIF]")
X_vif = sm.add_constant(X)
vif_data = pd.DataFrame({
    "feature": X_vif.columns,
    "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
})
print(vif_data.sort_values(by="VIF", ascending=False).head(15))

# === Evaluación del modelo Ridge ===
ridge_pred = ridge_cv.predict(X_test_scaled)
ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))
ridge_mae = mean_absolute_error(y_test, ridge_pred)
ridge_r2 = r2_score(y_test, ridge_pred)

print("\n[Regresión Ridge]")
print(f"MSE: {mean_squared_error(y_test, ridge_pred)}")
print(f"RMSE: {ridge_rmse}")
print(f"MAE: {ridge_mae}")
print(f"R²: {ridge_r2}")

# Obtener los coeficientes de Ridge (esto es lo que faltaba)
ridge_coefs = pd.Series(ridge_cv.coef_, index=X.columns)

# === Comparación de coeficientes entre OLS, Ridge y Lasso ===
ols_coefs = modelo_ols.params.drop("const")
coef_comparacion = pd.DataFrame({
    'OLS': ols_coefs,
    'Ridge': ridge_coefs,
    'Lasso': lasso_coefs
})
coef_comparacion['abs_OLS'] = coef_comparacion['OLS'].abs()
coef_comparacion_sorted = coef_comparacion.sort_values(by='abs_OLS', ascending=False).drop(columns='abs_OLS')

print("\n[Comparación Coeficientes OLS vs Ridge vs Lasso] (Top 20)")
print(coef_comparacion_sorted.head(20))

# === Análisis de correlación entre log_price y amenities ===
df_corr = pd.concat([df[['log_price']], amenities_dummies], axis=1)
corr_matrix = df_corr.corr()
corr_with_price = corr_matrix.loc['log_price'].drop('log_price').sort_values()

# === Mapa de calor completo de correlaciones ===
plt.figure(figsize=(20, 16))
mask = np.triu(corr_matrix)  # Evita duplicados en la matriz simétrica
sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', vmin=-1, vmax=1,
            linewidths=0.5, cbar_kws={"shrink": .8})
plt.title('Mapa de Calor de Correlación entre log_price y Amenities', fontsize=18)
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig('heatmap_correlaciones.png')
plt.close()

# === Cálculo de RMSE para Lasso ===
lasso_pred = lasso_cv.predict(X_test_scaled)
lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_pred))
lasso_mae = mean_absolute_error(y_test, lasso_pred)
lasso_r2 = r2_score(y_test, lasso_pred)

print("\n[Regresión Lasso]")
print(f"MSE: {mean_squared_error(y_test, lasso_pred)}")
print(f"RMSE: {lasso_rmse}")
print(f"MAE: {lasso_mae}")
print(f"R²: {lasso_r2}")

# === DataFrame para comparar RMSE de modelos ===
df_comparacion = pd.DataFrame({
    'modelo': ['OLS', 'Ridge', 'Lasso'],
    'rmse': [rmse, ridge_rmse, lasso_rmse]
})

# === Gráfico de barras ===
plt.figure(figsize=(8, 5))
sns.barplot(data=df_comparacion, x='modelo', y='rmse', palette='Set2')

# Agregar etiquetas con valores redondeados
for i, row in df_comparacion.iterrows():
    plt.text(i, row.rmse + 0.01, round(row.rmse, 4), ha='center')

plt.title('Comparación de RMSE entre Modelos')
plt.xlabel('Modelo')
plt.ylabel('Error Raíz Cuadrático Medio (RMSE)')
plt.tight_layout()
plt.savefig('comparacion_rmse.png')
plt.close()

# ============= NUEVOS GRÁFICOS SOLICITADOS =============

# === 1. Valores Reales vs. Predicciones ===
# Creamos una figura con 2 subplots (2 filas, 2 columnas)
fig, axs = plt.subplots(2, 2, figsize=(16, 14))

# Gráfico de dispersión (valores reales vs. predichos)
axs[0, 0].scatter(y_test, y_pred, alpha=0.5)
axs[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axs[0, 0].set_xlabel('Valores Reales (log_price)')
axs[0, 0].set_ylabel('Valores Predichos (log_price)')
axs[0, 0].set_title('Valores Reales vs. Predichos (Regresión Lineal)')

# Añadir cuadro de texto con métricas
metrics_text = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}'
axs[0, 0].text(0.05, 0.95, metrics_text, transform=axs[0, 0].transAxes, 
              fontsize=12, verticalalignment='top', 
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Cálculo de residuos
residuos = y_test - y_pred

# Histograma de residuos
axs[0, 1].hist(residuos, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
axs[0, 1].axvline(x=0, color='r', linestyle='--', alpha=0.7)
axs[0, 1].set_xlabel('Residuos')
axs[0, 1].set_ylabel('Frecuencia')
axs[0, 1].set_title('Distribución de Residuos')

# Gráfico de residuos vs. valores predichos
axs[1, 0].scatter(y_pred, residuos, alpha=0.5)
axs[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.7)
axs[1, 0].set_xlabel('Valores Predichos')
axs[1, 0].set_ylabel('Residuos')
axs[1, 0].set_title('Residuos vs. Valores Predichos')

# Gráfico Q-Q plot para verificar normalidad de residuos
import scipy.stats as stats
stats.probplot(residuos, dist="norm", plot=axs[1, 1])
axs[1, 1].set_title('Q-Q Plot de Residuos')

plt.tight_layout()
plt.savefig('evaluacion_predicciones.png')
plt.close()

# === 2. Curva de Aprendizaje ===
# Definimos tamaños de entrenamiento
train_sizes = np.linspace(0.1, 1.0, 10)

# Calculamos curvas de aprendizaje
train_sizes, train_scores, val_scores = learning_curve(
    LinearRegression(), X, y, train_sizes=train_sizes, cv=5,
    scoring='neg_mean_squared_error', n_jobs=-1)

# Convertimos MSE negativo a RMSE positivo
train_rmse = np.sqrt(-train_scores).mean(axis=1)
val_rmse = np.sqrt(-val_scores).mean(axis=1)

# Creamos el gráfico
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_rmse, 'o-', color='r', label='Error de Entrenamiento')
plt.plot(train_sizes, val_rmse, 'o-', color='g', label='Error de Validación')
plt.fill_between(train_sizes, train_rmse, val_rmse, alpha=0.1, color='g', 
                 label='Brecha de Generalización')

# Añadimos líneas de referencia y anotaciones
plt.axhline(y=val_rmse[-1], color='blue', linestyle='--', alpha=0.5, 
           label=f'RMSE Final: {val_rmse[-1]:.4f}')
plt.annotate('Brecha de\nGeneralización', 
            xy=(train_sizes[5], (train_rmse[5] + val_rmse[5])/2),
            xytext=(train_sizes[5]-0.1, (train_rmse[5] + val_rmse[5])/2 + 0.05),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
            fontsize=12)

plt.title('Curva de Aprendizaje (Regresión Lineal)', fontsize=14)
plt.xlabel('Tamaño del Conjunto de Entrenamiento', fontsize=12)
plt.ylabel('RMSE', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('curva_aprendizaje.png')
plt.close()

# === 3. Validación Cruzada ===
# Configuramos la validación cruzada con 5 folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_rmse = []
fold_indices = []

# Realizamos validación cruzada manual para obtener RMSE por fold
for i, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
    y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Entrenamos el modelo
    model = LinearRegression().fit(X_fold_train, y_fold_train)
    
    # Predecimos y calculamos RMSE
    y_fold_pred = model.predict(X_fold_val)
    fold_rmse.append(np.sqrt(mean_squared_error(y_fold_val, y_fold_pred)))
    fold_indices.append(f'Fold {i+1}')

# Calculamos estadísticas
mean_rmse = np.mean(fold_rmse)
std_rmse = np.std(fold_rmse)

# Creamos el gráfico
plt.figure(figsize=(10, 6))
bars = plt.bar(fold_indices, fold_rmse, color='skyblue', alpha=0.7, edgecolor='black')
plt.axhline(y=mean_rmse, color='r', linestyle='--', 
           label=f'RMSE Promedio: {mean_rmse:.4f} ± {std_rmse:.4f}')

# Añadimos valores en cada barra
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.4f}', ha='center', va='bottom', fontsize=10)

plt.title('Validación Cruzada - RMSE por Fold', fontsize=14)
plt.xlabel('Fold', fontsize=12)
plt.ylabel('RMSE', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('validacion_cruzada.png')
plt.close()

# === 4. Importancia de Variables ===
# Obtenemos los coeficientes del modelo Ridge
coefs = pd.Series(ridge_cv.coef_, index=X.columns)
coefs = coefs.sort_values(ascending=False)

# Seleccionamos las 20 variables más importantes (en valor absoluto)
top_coefs = coefs.abs().sort_values(ascending=False).head(20).index
coefs_plot = coefs[top_coefs]

# Creamos el gráfico
plt.figure(figsize=(12, 10))
colors = ['green' if c > 0 else 'red' for c in coefs_plot]
bars = plt.barh(range(len(coefs_plot)), coefs_plot, color=colors, alpha=0.7)
plt.yticks(range(len(coefs_plot)), coefs_plot.index)
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)

# Añadimos valores en cada barra
for i, bar in enumerate(bars):
    width = bar.get_width()
    label_x = width + 0.01 if width >= 0 else width - 0.08
    plt.text(label_x, i, f'{width:.4f}', va='center', fontsize=9,
             color='black' if width >= 0 else 'white')

plt.title('Importancia de Variables - Modelo Ridge (Top 20)', fontsize=14)
plt.xlabel('Coeficiente', fontsize=12)
plt.ylabel('Variable', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('importancia_variables_ridge.png')
plt.close()

# Mostramos un mensaje de finalización
print("\n[Análisis Completado]")
print("Se han generado los siguientes gráficos:")
print("1. evaluacion_predicciones.png - Valores Reales vs. Predicciones")
print("2. curva_aprendizaje.png - Curva de Aprendizaje")
print("3. validacion_cruzada.png - Validación Cruzada")
print("4. importancia_variables.png - Importancia de Variables")
print("5. comparacion_rmse.png - Comparación de RMSE entre modelos")
print("6. heatmap_correlaciones.png - Mapa de calor de correlaciones")


