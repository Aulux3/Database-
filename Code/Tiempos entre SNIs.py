# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 16:44:00 2024

@author: carlo
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Directorios para datos fraudulentos y legales
fraudulent_directory = 'C:/Users/carlo/OneDrive/Escritorio/TFM/Datasets/Final/Fraudulento/CSVs'
legal_directory = 'C:/Users/carlo/OneDrive/Escritorio/TFM/Datasets/Final/Legal/CSVs'

def procesar_directorio(directorio_csv):
    # Lista para almacenar las medias de los tiempos entre SNI por intervalo
    tiempos_entre_snis_intervalos = []
    
    for indice, archivo_csv in enumerate(os.listdir(directorio_csv)):
       if archivo_csv.endswith('.csv'):
        dtype_mapping = {'Server Name': str, 'JA3': str, 'JA3S': str}
        # Obtener la ruta completa del archivo
        archivo_path = os.path.join(directorio_csv, archivo_csv)

        # Lee el archivo CSV
        dtype_mapping = {'Server Name': str, 'JA3': str, 'JA3S': str}
        df = pd.read_csv(archivo_path, dtype=dtype_mapping)

        # Filtrar las filas con valores válidos en la columna 'Server Name'
        df = df.dropna(subset=['Server Name'])

        # Convertir la columna 'Time' a formato float (segundos)
        df['Time'] = df['Time'].astype(float)

        # Obtener el tiempo máximo y mínimo del archivo
        tiempo_minimo = df['Time'].min()
        tiempo_maximo = df['Time'].max()

        # Establecer el primer intervalo
        intervalo = 60  # intervalo de 1 minuto en segundos
        tiempo_intervalo = tiempo_minimo
        tiempo_final = tiempo_maximo

        # Iterar sobre cada intervalo
        while tiempo_intervalo < tiempo_final:
            # Filtrar los datos dentro del intervalo actual
            datos_intervalo = df[(df['Time'] >= tiempo_intervalo) & (df['Time'] < tiempo_intervalo + intervalo)]

            # Calcular los tiempos entre SNI dentro del intervalo
            tiempos_entre_snis = datos_intervalo['Time'].diff().dropna().tolist()

            # Calcular la media de los tiempos entre SNI dentro del intervalo
            if tiempos_entre_snis:
                media_tiempo_entre_snis = sum(tiempos_entre_snis) / len(tiempos_entre_snis)
            else:
                media_tiempo_entre_snis = 0

            # Agregar la media a la lista
            tiempos_entre_snis_intervalos.append(media_tiempo_entre_snis)

            # Mover al siguiente intervalo
            tiempo_intervalo += intervalo

    return tiempos_entre_snis_intervalos

# Procesar directorio de datos fraudulentos
tiempos_entre_snis_intervalos_fraudulento = procesar_directorio(fraudulent_directory)
#print("Medias de tiempos entre SNI por intervalo en datos fraudulentos:", tiempos_entre_snis_intervalos_fraudulento)

# Procesar directorio de datos legales
tiempos_entre_snis_intervalos_legal = procesar_directorio(legal_directory)
#print("Medias de tiempos entre SNI por intervalo en datos legales:", tiempos_entre_snis_intervalos_legal)

# Función para calcular la entropía
def calcular_entropia(datos):
    # Calcular la frecuencia de cada valor en los datos
    valores, conteos = np.unique(datos, return_counts=True)
    # Calcular la probabilidad de cada valor
    probabilidades = conteos / len(datos)
    # Calcular la entropía utilizando la fórmula de Shannon
    entropia = -np.sum(probabilidades * np.log2(probabilidades))
    return entropia

# Función para calcular la entropía de Rényi
def calcular_entropia_renyi(datos, alpha):
    # Calcular la frecuencia de cada valor en los datos
    valores, conteos = np.unique(datos, return_counts=True)
    # Calcular la probabilidad de cada valor
    probabilidades = conteos / len(datos)
    # Calcular la entropía de Rényi utilizando la fórmula correspondiente
    entropia = 1 / (1 - alpha) * np.log2(np.sum(probabilidades ** alpha))
    return entropia

# Función para calcular la entropía de Tsallis
def calcular_entropia_tsallis(datos, q):
    # Calcular la frecuencia de cada valor en los datos
    valores, conteos = np.unique(datos, return_counts=True)
    # Calcular la probabilidad de cada valor
    probabilidades = conteos / len(datos)
    # Calcular la entropía de Tsallis utilizando la fórmula correspondiente
    if q == 1:
        entropia = -np.sum(probabilidades * np.log2(probabilidades))
    else:
        entropia = 1 / (q - 1) * (np.sum(probabilidades ** q) - 1)
    return entropia

# Calcular entropías para los datos fraudulentos
entropia_shannon_fraudulento = calcular_entropia(tiempos_entre_snis_intervalos_fraudulento)
entropia_renyi_fraudulento = calcular_entropia_renyi(tiempos_entre_snis_intervalos_fraudulento, alpha=0.5)  # Puedes ajustar alpha según tus necesidades
entropia_tsallis_fraudulento = calcular_entropia_tsallis(tiempos_entre_snis_intervalos_fraudulento, q=2.0)  # Puedes ajustar q según tus necesidades

# Calcular entropías para los datos legales
entropia_shannon_legal = calcular_entropia(tiempos_entre_snis_intervalos_legal)
entropia_renyi_legal = calcular_entropia_renyi(tiempos_entre_snis_intervalos_legal, alpha=0.5)  # Puedes ajustar alpha según tus necesidades
entropia_tsallis_legal = calcular_entropia_tsallis(tiempos_entre_snis_intervalos_legal, q=2.0)  # Puedes ajustar q según tus necesidades

# Graficar las entropías
categorias = ['Shannon', 'Rényi (α=0.5)', 'Tsallis (q=2.0)']
entropias_fraudulento = [entropia_shannon_fraudulento, entropia_renyi_fraudulento, entropia_tsallis_fraudulento]
entropias_legal = [entropia_shannon_legal, entropia_renyi_legal, entropia_tsallis_legal]

plt.figure(figsize=(10, 6))
plt.bar(categorias, entropias_fraudulento, color='red', alpha=0.5, label='Fraudulento')
plt.bar(categorias, entropias_legal, color='blue', alpha=0.5, label='Legal')
plt.title('Entropías de los tiempos entre SNI por intervalo')
plt.xlabel('Tipo de Entropía')
plt.ylabel('Valor de Entropía')
plt.legend()
plt.show()
