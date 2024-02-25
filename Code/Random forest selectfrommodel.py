# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 23:48:20 2024

@author: carlo
"""

# Importar las bibliotecas necesarias
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
import graphviz
import time

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier  # Importar el clasificador de árbol de decisión
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline

from sklearn.feature_selection import SelectFromModel

# Directorios para datos fraudulentos y legales
fraudulent_directory = 'C:/Users/carlo/OneDrive/Escritorio/TFM/Datasets/Final/Fraudulento/Flujos'
legal_directory = 'C:/Users/carlo/OneDrive/Escritorio/TFM/Datasets/Final/Legal/Flujos'

# Función para cargar datos CSV y mapear la columna 'Label'
def load_data(directory, label_value):
    dataframes = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            # Mapea la etiqueta según el valor proporcionado
            df['Label'] = label_value
            dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)

# Carga datos fraudulentos y legales
data_fraudulent = load_data(fraudulent_directory, label_value=1)  # Etiqueta 1 para tráfico fraudulento
data_legal = load_data(legal_directory, label_value=0)  # Etiqueta 0 para tráfico legal

# Concatena los DataFrames fraudulentos y legales
data = pd.concat([data_fraudulent, data_legal], ignore_index=True)

# Elimina filas con valores infinitos
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()

# Seleccionar características relevantes
features = ['Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 
                    'TotLen Bwd Pkts', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 
                    'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow IAT Mean',
                    'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std', 
                    'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Tot','Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 
                    'Bwd PSH Flags', 'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s', 'Bwd Pkts/s', 'Pkt Len Min', 
                    'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt', 'SYN Flag Cnt', 
                    'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg', 'Fwd Seg Size Avg', 
                    'Bwd Seg Size Avg','Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts', 
                    'Init Bwd Win Byts', 'Fwd Act Data Pkts', 'Active Mean', 'Active Std', 'Active Max', 
                    'Active Min']

X = data[features]
y = data['Label']

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
'''
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'criterion': ['gini', 'entropy']
}
'''
param_grid = {
    'n_estimators': [150],
    'max_depth': [20],
    'criterion': ['entropy']
}

# Definir el modelo Random Forest
modelo_rf = RandomForestClassifier()

# Definir las estrategias de sobre muestreo y submuestreo
over_sampling_strategy = SMOTE(sampling_strategy=0.5)  # Ajusta la proporción de la clase minoritaria al 50% de la clase mayoritaria
under_sampling_strategy = RandomUnderSampler(sampling_strategy=0.8)  # Ajusta la proporción de la clase mayoritaria al 80% de la clase minoritaria

# Crear el pipeline con las técnicas de sobre muestreo y submuestreo junto con el modelo Random Forest
pipeline = Pipeline([('over_sampling', over_sampling_strategy),
                     ('under_sampling', under_sampling_strategy),
                     ('model', modelo_rf)])

# Inicializar el objeto GridSearchCV
grid_search = GridSearchCV(estimator=modelo_rf, param_grid=param_grid, cv=5, scoring='accuracy')

# Entrenar el modelo utilizando validación cruzada
grid_search.fit(X_train, y_train)

# Obtener los mejores hiperparámetros encontrados
print("Mejores hiperparámetros encontrados:")
print(grid_search.best_params_)

# Definir el selector de características basado en el modelo RandomForestClassifier
selector = SelectFromModel(RandomForestClassifier(**grid_search.best_params_))

# Crear el pipeline con el selector de características y el modelo Random Forest
pipeline = Pipeline([('selector', selector),
                     ('model', modelo_rf)])

# Entrenar el modelo con los datos de entrenamiento
pipeline.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
predicciones_rf = pipeline.predict(X_test)

# Calcular la precisión del modelo
precision_rf = accuracy_score(y_test, predicciones_rf)
print(f"Precisión del modelo Random Forest después de la selección de características: {precision_rf}")

# Obtener las probabilidades de predicción para la clase positiva
probs_rf = pipeline.predict_proba(X_test)[:, 1]

# Calcular la tasa de falsos positivos, la tasa de verdaderos positivos y los umbrales utilizando la curva ROC
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, probs_rf)

# Calcular el área bajo la curva ROC
roc_auc_rf = auc(fpr_rf, tpr_rf)

# Graficar la curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, color='darkorange', lw=2, label=f'Área bajo la curva ROC = {roc_auc_rf:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - Random Forest')
plt.legend(loc="lower right")
plt.show()

# Obtener las características seleccionadas por el selector
features_selected = X.columns[selector.get_support()]

# Imprimir las características seleccionadas
print("Características seleccionadas:")
print(features_selected)


# Crear un DataFrame con las características seleccionadas y sus importancias
selected_features_df = pd.DataFrame({'Feature': features_selected, 'Importance': np.ones(len(features_selected))})

# Visualizar las características seleccionadas
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=selected_features_df)
plt.title('Características Seleccionadas')
plt.xlabel('Importancia')
plt.ylabel('Característica')
plt.show()
