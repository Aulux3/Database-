# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 21:41:09 2024

@author: carlo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Directorios para datos fraudulentos y legales
fraudulent_directory = 'C:/Users/carlo/OneDrive/Escritorio/TFM/Datasets/Final/Fraudulento/Spotify'
legal_directory = 'C:/Users/carlo/OneDrive/Escritorio/TFM/Datasets/Final/Legal/Spotify'

# Valores a probar para alpha en Entropía de Rényi
alphas_renyi_a_probar = [0.1, 0.25, 0.5, 0.75, 1.25, 1.5, 1.75, 2.0]

# Valores a probar para q en Entropía de Tsallis
qs_tsallis_a_probar = [0.0, 0.25, 0.5, 0.75, 1.25, 1.5, 1.75, 2.0]

def calcular_renyi_entropia(frecuencia, alpha):
    result = np.sum(frecuencia**alpha, axis=0)  # Suma a lo largo del eje 0 (si frecuencia es 2D)
    if result == 0:
        return 0  # Puedes ajustar esto según lo que sea apropiado en tu contexto
    return 1 / (1 - alpha) * np.log2(result)

# Modificar la función para calcular la Entropía de Server Names usando Rényi
def calcular_renyi_entropia_server_names(df, alpha):
    server_names_freq = df['Server Name'].value_counts(normalize=True)
    entropia_server_names_renyi = calcular_renyi_entropia(server_names_freq, alpha)
    return entropia_server_names_renyi

# Modificar la función para calcular la Entropía de JA3 usando Rényi
def calcular_renyi_entropia_JA3(df, alpha):
    JA3_freq = df['JA3'].value_counts(normalize=True)
    entropia_JA3_renyi = calcular_renyi_entropia(JA3_freq, alpha)
    return entropia_JA3_renyi

# Modificar la función para calcular la Entropía de JA3S usando Rényi
def calcular_renyi_entropia_JA3S(df, alpha):
    JA3S_freq = df['JA3S'].value_counts(normalize=True)
    entropia_JA3S_renyi = calcular_renyi_entropia(JA3S_freq, alpha)
    return entropia_JA3S_renyi

# Modificar la función para calcular la Entropía de Protocolos usando Rényi
def calcular_renyi_entropia_protocolos(df, alpha):
    protocolos_freq = df['Protocol'].value_counts(normalize=True)
    entropia_protocolos_renyi = calcular_renyi_entropia(protocolos_freq, alpha)
    return entropia_protocolos_renyi

# Modificar la función para calcular la Entropía de IPs usando Rényi
def calcular_renyi_entropia_ips(df, alpha):
    ip_origen_freq = df['Source'].value_counts(normalize=True)
    ip_destino_freq = df['Destination'].value_counts(normalize=True)
    ip_combinadas_freq = df[['Source', 'Destination']].stack().value_counts(normalize=True)

    entropia_ip_origen_renyi = calcular_renyi_entropia(ip_origen_freq, alpha)
    entropia_ip_destino_renyi = calcular_renyi_entropia(ip_destino_freq, alpha)
    entropia_ip_combinadas_renyi = calcular_renyi_entropia(ip_combinadas_freq, alpha)

    return entropia_ip_origen_renyi, entropia_ip_destino_renyi, entropia_ip_combinadas_renyi

# Función para Calcular la Entropía de Tsallis
def calcular_entropia_tsallis(frecuencia, q):
    if q == 1:
        return -np.sum(frecuencia * np.log2(frecuencia))
    else:
        return 1 / (q - 1) * (np.sum(frecuencia ** q) - 1)

# Modificar la función para calcular la Entropía de Server Names usando Tsallis
def calcular_tsallis_entropia_server_names(df, q):
    server_names_freq = df['Server Name'].value_counts(normalize=True)
    entropia_server_names = calcular_entropia_tsallis(server_names_freq, q)
    return entropia_server_names

# Modificar la función para calcular la Entropía de JA3 usando Tsallis
def calcular_tsallis_entropia_JA3(df, q):
    JA3_freq = df['JA3'].value_counts(normalize=True)
    entropia_JA3 = calcular_entropia_tsallis(JA3_freq, q)
    return entropia_JA3

# Modificar la función para calcular la Entropía de JA3S usando Tsallis
def calcular_tsallis_entropia_JA3S(df, q):
    JA3S_freq = df['JA3S'].value_counts(normalize=True)
    entropia_JA3S = calcular_entropia_tsallis(JA3S_freq, q)
    return entropia_JA3S

# Modificar la función para calcular la Entropía de Protocolos usando Tsallis
def calcular_tsallis_entropia_protocolos(df, q):
    protocolos_freq = df['Protocol'].value_counts(normalize=True)
    entropia_protocolos = calcular_entropia_tsallis(protocolos_freq, q)
    return entropia_protocolos

def procesar_directorio_entropias_renyi(directorio_csv, tipo_trafico, alpha):
    entropias_ip_origen_renyi_lista = []
    entropias_ip_destino_renyi_lista = []
    entropias_ip_combinadas_renyi_lista = []
    entropias_server_names_renyi_lista = []
    entropias_JA3_renyi_lista = []
    entropias_JA3S_renyi_lista = []
    entropias_protocolos_renyi_lista = []

    for indice, archivo_csv in enumerate(os.listdir(directorio_csv)):
        if archivo_csv.endswith('.csv'):
            dtype_mapping = {'Server Name': str, 'JA3': str, 'JA3S': str}

            # Lee el archivo CSV con los tipos de datos especificados
            df = pd.read_csv(os.path.join(directorio_csv, archivo_csv), dtype=dtype_mapping)

            # Calcula entropías usando Rényi
            (
                entropias_ip_origen_renyi,
                entropias_ip_destino_renyi,
                entropias_ip_combinadas_renyi,
                entropias_server_names_renyi,
                entropias_JA3_renyi,
                entropias_JA3S_renyi,
                entropias_protocolos_renyi
            ) = procesar_archivo_entropias_renyi(df, alpha)

            # Guarda entropías en listas
            entropias_ip_origen_renyi_lista.extend(entropias_ip_origen_renyi)
            entropias_ip_destino_renyi_lista.extend(entropias_ip_destino_renyi)
            entropias_ip_combinadas_renyi_lista.extend(entropias_ip_combinadas_renyi)
            entropias_server_names_renyi_lista.extend(entropias_server_names_renyi)
            entropias_JA3_renyi_lista.extend(entropias_JA3_renyi)
            entropias_JA3S_renyi_lista.extend(entropias_JA3S_renyi)
            entropias_protocolos_renyi_lista.extend(entropias_protocolos_renyi)

    return (
        entropias_ip_origen_renyi_lista,
        entropias_ip_destino_renyi_lista,
        entropias_ip_combinadas_renyi_lista,
        entropias_server_names_renyi_lista,
        entropias_JA3_renyi_lista,
        entropias_JA3S_renyi_lista,
        entropias_protocolos_renyi_lista,
    )

def procesar_archivo_entropias_renyi(df, alpha):
    entropias_ip_origen_renyi = []
    entropias_ip_destino_renyi = []
    entropias_ip_combinadas_renyi = []
    entropias_server_names_renyi = []
    entropias_JA3_renyi = []
    entropias_JA3S_renyi = []
    entropias_protocolos_renyi = []

    # Convertir la columna 'Time' a datetime
    df['Time'] = pd.to_datetime(df['Time'], unit='s')

    # Dividir el dataframe en intervalos de 1 minuto
    for _, intervalo_df in df.groupby(pd.Grouper(key='Time', freq='1Min')):
        entropia_ips_origen_renyi, entropia_ips_destino_renyi, entropia_ips_combinadas_renyi = calcular_renyi_entropia_ips(intervalo_df, alpha)
        entropia_server_names_renyi = calcular_renyi_entropia_server_names(intervalo_df, alpha)
        entropia_JA3_renyi = calcular_renyi_entropia_JA3(intervalo_df, alpha)
        entropia_JA3S_renyi = calcular_renyi_entropia_JA3S(intervalo_df, alpha)
        entropia_protocolos_renyi = calcular_renyi_entropia_protocolos(intervalo_df, alpha)

        # Almacenar entropías como valores únicos
        entropias_ip_origen_renyi.append(entropia_ips_origen_renyi)
        entropias_ip_destino_renyi.append(entropia_ips_destino_renyi)
        entropias_ip_combinadas_renyi.append(entropia_ips_combinadas_renyi)
        entropias_server_names_renyi.append(entropia_server_names_renyi)
        entropias_JA3_renyi.append(entropia_JA3_renyi)
        entropias_JA3S_renyi.append(entropia_JA3S_renyi)
        entropias_protocolos_renyi.append(entropia_protocolos_renyi)

    return (
        entropias_ip_origen_renyi,
        entropias_ip_destino_renyi,
        entropias_ip_combinadas_renyi,
        entropias_server_names_renyi,
        entropias_JA3_renyi,
        entropias_JA3S_renyi,
        entropias_protocolos_renyi
    )

# Modificar la función para calcular la Entropía de IPs usando Tsallis
def calcular_tsallis_entropia_ips(df, q):
    ip_origen_freq = df['Source'].value_counts(normalize=True)
    ip_destino_freq = df['Destination'].value_counts(normalize=True)
    ip_combinadas_freq = df[['Source', 'Destination']].stack().value_counts(normalize=True)

    entropia_ip_origen = calcular_entropia_tsallis(ip_origen_freq, q)
    entropia_ip_destino = calcular_entropia_tsallis(ip_destino_freq, q)
    entropia_ip_combinadas = calcular_entropia_tsallis(ip_combinadas_freq, q)

    return entropia_ip_origen, entropia_ip_destino, entropia_ip_combinadas

def procesar_archivo_entropias_tsallis(df, q):
    entropias_ip_origen_tsallis = []
    entropias_ip_destino_tsallis = []
    entropias_ip_combinadas_tsallis = []
    entropias_server_names_tsallis = []
    entropias_JA3_tsallis = []
    entropias_JA3S_tsallis = []
    entropias_protocolos_tsallis = []

    # Convertir la columna 'Time' a datetime
    df['Time'] = pd.to_datetime(df['Time'], unit='s')

    # Dividir el dataframe en intervalos de 1 minuto
    for _, intervalo_df in df.groupby(pd.Grouper(key='Time', freq='1Min')):
        entropia_ips_origen_tsallis, entropia_ips_destino_tsallis, entropia_ips_combinadas_tsallis = calcular_tsallis_entropia_ips(intervalo_df, q)
        entropia_server_names_tsallis = calcular_tsallis_entropia_server_names(intervalo_df, q)
        entropia_JA3_tsallis = calcular_tsallis_entropia_JA3(intervalo_df, q)
        entropia_JA3S_tsallis = calcular_tsallis_entropia_JA3S(intervalo_df, q)
        entropia_protocolos_tsallis = calcular_tsallis_entropia_protocolos(intervalo_df, q)

        # Almacenar entropías como valores únicos
        entropias_ip_origen_tsallis.append(entropia_ips_origen_tsallis)
        entropias_ip_destino_tsallis.append(entropia_ips_destino_tsallis)
        entropias_ip_combinadas_tsallis.append(entropia_ips_combinadas_tsallis)
        entropias_server_names_tsallis.append(entropia_server_names_tsallis)
        entropias_JA3_tsallis.append(entropia_JA3_tsallis)
        entropias_JA3S_tsallis.append(entropia_JA3S_tsallis)
        entropias_protocolos_tsallis.append(entropia_protocolos_tsallis)

    return (
        entropias_ip_origen_tsallis,
        entropias_ip_destino_tsallis,
        entropias_ip_combinadas_tsallis,
        entropias_server_names_tsallis,
        entropias_JA3_tsallis,
        entropias_JA3S_tsallis,
        entropias_protocolos_tsallis
    )

def procesar_directorio_entropias_tsallis(directorio_csv, tipo_trafico, q):
    entropias_ip_origen_tsallis_lista = []
    entropias_ip_destino_tsallis_lista = []
    entropias_ip_combinadas_tsallis_lista = []
    entropias_server_names_tsallis_lista = []
    entropias_JA3_tsallis_lista = []
    entropias_JA3S_tsallis_lista = []
    entropias_protocolos_tsallis_lista = []

    for indice, archivo_csv in enumerate(os.listdir(directorio_csv)):
        if archivo_csv.endswith('.csv'):
            dtype_mapping = {'Server Name': str, 'JA3': str, 'JA3S': str}

            # Lee el archivo CSV con los tipos de datos especificados
            df = pd.read_csv(os.path.join(directorio_csv, archivo_csv), dtype=dtype_mapping)

            # Calcula entropías usando Tsallis
            (
                entropias_ip_origen_tsallis,
                entropias_ip_destino_tsallis,
                entropias_ip_combinadas_tsallis,
                entropias_server_names_tsallis,
                entropias_JA3_tsallis,
                entropias_JA3S_tsallis,
                entropias_protocolos_tsallis
            ) = procesar_archivo_entropias_tsallis(df, q)

            # Guarda entropías en listas
            entropias_ip_origen_tsallis_lista.extend(entropias_ip_origen_tsallis)
            entropias_ip_destino_tsallis_lista.extend(entropias_ip_destino_tsallis)
            entropias_ip_combinadas_tsallis_lista.extend(entropias_ip_combinadas_tsallis)
            entropias_server_names_tsallis_lista.extend(entropias_server_names_tsallis)
            entropias_JA3_tsallis_lista.extend(entropias_JA3_tsallis)
            entropias_JA3S_tsallis_lista.extend(entropias_JA3S_tsallis)
            entropias_protocolos_tsallis_lista.extend(entropias_protocolos_tsallis)

    return (
        entropias_ip_origen_tsallis_lista,
        entropias_ip_destino_tsallis_lista,
        entropias_ip_combinadas_tsallis_lista,
        entropias_server_names_tsallis_lista,
        entropias_JA3_tsallis_lista,
        entropias_JA3S_tsallis_lista,
        entropias_protocolos_tsallis_lista,
    )


def graficar_resultados(valores, resultados_fraudulento, resultados_legal, titulo, etiqueta_eje):
    plt.plot(valores, resultados_fraudulento, label='Fraudulento', marker='o')
    plt.plot(valores, resultados_legal, label='Legal', marker='o')
    plt.title(titulo)
    plt.xlabel(etiqueta_eje)
    plt.ylabel('Entropía')  # Ajusta según la métrica que estás usando
    plt.legend()
    plt.grid(True)
    plt.show()
    

def realizar_pruebas_entropias():
    resultados_renyi = []

    # Pruebas automáticas para Entropía de Rényi
    for alpha_renyi in alphas_renyi_a_probar:
        (
            entropias_ips_origen_fraudulento_renyi,
            entropias_ips_destino_fraudulento_renyi,
            entropias_ip_combinadas_fraudulento_renyi,
            entropias_server_names_fraudulento_renyi,
            entropias_JA3_fraudulento_renyi,
            entropias_JA3S_fraudulento_renyi,
            entropias_protocolos_fraudulento_renyi
        ) = procesar_directorio_entropias_renyi(fraudulent_directory, 'Fraudulento', alpha_renyi)

        (
            entropias_ips_origen_legal_renyi,
            entropias_ips_destino_legal_renyi,
            entropias_ip_combinadas_legal_renyi,
            entropias_server_names_legal_renyi,
            entropias_JA3_legal_renyi,
            entropias_JA3S_legal_renyi,
            entropias_protocolos_legal_renyi
        ) = procesar_directorio_entropias_renyi(legal_directory, 'Legal', alpha_renyi)

        # Puedes realizar análisis o almacenar los resultados como desees
        # Aquí, solo se almacena la media de las entropías de IPs de origen
        media_entropias_ips_origen_fraudulento = np.mean(entropias_ips_origen_fraudulento_renyi)
        media_entropias_ips_origen_legal = np.mean(entropias_ips_origen_legal_renyi)
        
        media_entropias_ips_destino_fraudulento = np.mean(entropias_ips_destino_fraudulento_renyi)
        media_entropias_ips_destino_legal = np.mean(entropias_ips_destino_legal_renyi)
        
        media_entropias_ips_combinadas_fraudulento = np.mean(entropias_ip_combinadas_fraudulento_renyi)
        media_entropias_ips_combinadas_legal = np.mean(entropias_ip_combinadas_legal_renyi)
        
        media_entropias_server_names_fraudulento = np.mean(entropias_server_names_fraudulento_renyi)
        media_entropias_server_names_legal = np.mean(entropias_server_names_legal_renyi)
        
        media_entropias_JA3_fraudulento = np.mean(entropias_JA3_fraudulento_renyi)
        media_entropias_JA3_legal = np.mean(entropias_JA3_legal_renyi)
        
        media_entropias_JA3S_fraudulento = np.mean(entropias_JA3S_fraudulento_renyi)
        media_entropias_JA3S_legal = np.mean(entropias_JA3S_legal_renyi)
        
        media_entropias_protocolos_fraudulento = np.mean(entropias_protocolos_fraudulento_renyi)
        media_entropias_protocolos_legal = np.mean(entropias_protocolos_legal_renyi)

        resultados_renyi.append({
            'alpha_renyi': alpha_renyi,
            'media_entropias_ips_origen_fraudulento': media_entropias_ips_origen_fraudulento,
            'media_entropias_ips_origen_legal': media_entropias_ips_origen_legal,
            'media_entropias_ips_destino_fraudulento': media_entropias_ips_destino_fraudulento,
            'media_entropias_ips_destino_legal': media_entropias_ips_destino_legal,
            'media_entropias_ips_combinadas_fraudulento': media_entropias_ips_combinadas_fraudulento,
            'media_entropias_ips_combinadas_legal': media_entropias_ips_combinadas_legal,
            'media_entropias_server_names_fraudulento': media_entropias_server_names_fraudulento,
            'media_entropias_server_names_legal': media_entropias_server_names_legal,
            'media_entropias_JA3_fraudulento': media_entropias_JA3_fraudulento,
            'media_entropias_JA3_legal': media_entropias_JA3_legal,
            'media_entropias_JA3S_fraudulento': media_entropias_JA3S_fraudulento,
            'media_entropias_JA3S_legal': media_entropias_JA3S_legal,  # Corrección en esta línea
            'media_entropias_protocolos_fraudulento': media_entropias_protocolos_fraudulento,
            'media_entropias_protocolos_legal': media_entropias_protocolos_legal
})

    resultados_tsallis = []

    # Pruebas automáticas para Entropía de Tsallis
    for q_tsallis in qs_tsallis_a_probar:
        (
            entropias_ips_origen_fraudulento_tsallis,
            entropias_ips_destino_fraudulento_tsallis,
            entropias_ip_combinadas_fraudulento_tsallis,
            entropias_server_names_fraudulento_tsallis,
            entropias_JA3_fraudulento_tsallis,
            entropias_JA3S_fraudulento_tsallis,
            entropias_protocolos_fraudulento_tsallis
        ) = procesar_directorio_entropias_tsallis(fraudulent_directory, 'Fraudulento', q_tsallis)

        (
            entropias_ips_origen_legal_tsallis,
            entropias_ips_destino_legal_tsallis,
            entropias_ip_combinadas_legal_tsallis,
            entropias_server_names_legal_tsallis,
            entropias_JA3_legal_tsallis,
            entropias_JA3S_legal_tsallis,
            entropias_protocolos_legal_tsallis
        ) = procesar_directorio_entropias_tsallis(legal_directory, 'Legal', q_tsallis)

        # Puedes realizar análisis o almacenar los resultados como desees
        # Aquí, solo se almacena la media de las entropías de IPs de origen
        media_entropias_ips_origen_fraudulento = np.mean(entropias_ips_origen_fraudulento_tsallis)
        media_entropias_ips_origen_legal = np.mean(entropias_ips_origen_legal_tsallis)
        
        media_entropias_ips_destino_fraudulento = np.mean(entropias_ips_destino_fraudulento_tsallis)
        media_entropias_ips_destino_legal = np.mean(entropias_ips_destino_legal_tsallis)
        
        media_entropias_ips_combinadas_fraudulento = np.mean(entropias_ip_combinadas_fraudulento_tsallis)
        media_entropias_ips_combinadas_legal = np.mean(entropias_ip_combinadas_legal_tsallis)
        
        media_entropias_server_names_fraudulento = np.mean(entropias_server_names_fraudulento_tsallis)
        media_entropias_server_names_legal = np.mean(entropias_server_names_legal_tsallis)
        
        media_entropias_JA3_fraudulento = np.mean(entropias_JA3_fraudulento_tsallis)
        media_entropias_JA3_legal = np.mean(entropias_JA3_legal_tsallis)
        
        media_entropias_JA3S_fraudulento = np.mean(entropias_JA3S_fraudulento_tsallis)
        media_entropias_JA3S_legal = np.mean(entropias_JA3S_legal_tsallis)
        
        media_entropias_protocolos_fraudulento = np.mean(entropias_protocolos_fraudulento_tsallis)
        media_entropias_protocolos_legal = np.mean(entropias_protocolos_legal_tsallis)
        
        # ...
        
        resultados_tsallis.append({
            'q_tsallis': q_tsallis,
            'media_entropias_ips_origen_fraudulento': media_entropias_ips_origen_fraudulento,
            'media_entropias_ips_origen_legal': media_entropias_ips_origen_legal,
            'media_entropias_ips_destino_fraudulento': media_entropias_ips_destino_fraudulento,
            'media_entropias_ips_destino_legal': media_entropias_ips_destino_legal,
            'media_entropias_ips_combinadas_fraudulento': media_entropias_ips_combinadas_fraudulento,
            'media_entropias_ips_combinadas_legal': media_entropias_ips_combinadas_legal,
            'media_entropias_server_names_fraudulento': media_entropias_server_names_fraudulento,
            'media_entropias_server_names_legal': media_entropias_server_names_legal,
            'media_entropias_JA3_fraudulento': media_entropias_JA3_fraudulento,
            'media_entropias_JA3_legal': media_entropias_JA3_legal,
            'media_entropias_JA3S_fraudulento': media_entropias_JA3S_fraudulento,
            'media_entropias_JA3S_legal': media_entropias_JA3S_legal,
            'media_entropias_protocolos_fraudulento': media_entropias_protocolos_fraudulento,
            'media_entropias_protocolos_legal': media_entropias_protocolos_legal
        })
    # Graficar resultados de Entropía de Rényi
    graficar_resultados(alphas_renyi_a_probar, 
                        [r['media_entropias_ips_origen_fraudulento'] for r in resultados_renyi],
                        [r['media_entropias_ips_origen_legal'] for r in resultados_renyi], 
                        'Entropía de Rényi - IPs de Origen', 'Alpha')
    
    graficar_resultados(alphas_renyi_a_probar, 
                        [r['media_entropias_ips_destino_fraudulento'] for r in resultados_renyi],
                        [r['media_entropias_ips_destino_legal'] for r in resultados_renyi], 
                        'Entropía de Rényi - IPs de Destino', 'Alpha')
    
    graficar_resultados(alphas_renyi_a_probar, 
                        [r['media_entropias_ips_combinadas_fraudulento'] for r in resultados_renyi],
                        [r['media_entropias_ips_combinadas_legal'] for r in resultados_renyi], 
                        'Entropía de Rényi - IPs Combinadas', 'Alpha')
    
    graficar_resultados(alphas_renyi_a_probar, 
                        [r['media_entropias_server_names_fraudulento'] for r in resultados_renyi],
                        [r['media_entropias_server_names_legal'] for r in resultados_renyi], 
                        'Entropía de Rényi - Nombres de Servidor', 'Alpha')
    
    graficar_resultados(alphas_renyi_a_probar, 
                        [r['media_entropias_JA3_fraudulento'] for r in resultados_renyi],
                        [r['media_entropias_JA3_legal'] for r in resultados_renyi], 
                        'Entropía de Rényi - JA3', 'Alpha')
    
    graficar_resultados(alphas_renyi_a_probar, 
                        [r['media_entropias_JA3S_fraudulento'] for r in resultados_renyi],
                        [r['media_entropias_JA3S_legal'] for r in resultados_renyi], 
                        'Entropía de Rényi - JA3S', 'Alpha')
    
    graficar_resultados(alphas_renyi_a_probar, 
                        [r['media_entropias_protocolos_fraudulento'] for r in resultados_renyi],
                        [r['media_entropias_protocolos_legal'] for r in resultados_renyi], 
                        'Entropía de Rényi - Protocolos', 'Alpha')
    
    # Graficar resultados de Entropía de Tsallis
    graficar_resultados(qs_tsallis_a_probar, 
                        [r['media_entropias_ips_origen_fraudulento'] for r in resultados_tsallis],
                        [r['media_entropias_ips_origen_legal'] for r in resultados_tsallis], 
                        'Entropía de Tsallis - IPs de Origen', 'q')
    
    graficar_resultados(qs_tsallis_a_probar, 
                        [r['media_entropias_ips_destino_fraudulento'] for r in resultados_tsallis],
                        [r['media_entropias_ips_destino_legal'] for r in resultados_tsallis], 
                        'Entropía de Tsallis - IPs de Destino', 'q')
    
    graficar_resultados(qs_tsallis_a_probar, 
                        [r['media_entropias_ips_combinadas_fraudulento'] for r in resultados_tsallis],
                        [r['media_entropias_ips_combinadas_legal'] for r in resultados_tsallis], 
                        'Entropía de Tsallis - IPs Combinadas', 'q')
    
    graficar_resultados(qs_tsallis_a_probar, 
                        [r['media_entropias_server_names_fraudulento'] for r in resultados_tsallis],
                        [r['media_entropias_server_names_legal'] for r in resultados_tsallis], 
                        'Entropía de Tsallis - Nombres de Servidor', 'q')
    
    graficar_resultados(qs_tsallis_a_probar, 
                        [r['media_entropias_JA3_fraudulento'] for r in resultados_tsallis],
                        [r['media_entropias_JA3_legal'] for r in resultados_tsallis], 
                        'Entropía de Tsallis - JA3', 'q')
    
    graficar_resultados(qs_tsallis_a_probar, 
                        [r['media_entropias_JA3S_fraudulento'] for r in resultados_tsallis],
                        [r['media_entropias_JA3S_legal'] for r in resultados_tsallis], 
                        'Entropía de Tsallis - JA3S', 'q')
    
    graficar_resultados(qs_tsallis_a_probar, 
                        [r['media_entropias_protocolos_fraudulento'] for r in resultados_tsallis],
                        [r['media_entropias_protocolos_legal'] for r in resultados_tsallis], 
                        'Entropía de Tsallis - Protocolos', 'q')


    return resultados_renyi, resultados_tsallis

resultados_renyi, resultados_tsallis = realizar_pruebas_entropias()

# Puedes imprimir o analizar los resultados obtenidos
print("Resultados Entropía de Rényi:")
print(resultados_renyi)

print("\nResultados Entropía de Tsallis:")
print(resultados_tsallis)

def graficar_todas_entropias_tsallis(qs_vals, resultados_tsallis):
    fig, axs = plt.subplots(4, 2, figsize=(16, 16))
    fig.suptitle('Entropía de Tsallis - Todas las Categorías', fontsize=16)

    # Etiquetas y valores para cada gráfica
    etiquetas = ['IPs de Origen', 'IPs de Destino', 'IPs Combinadas', 'Nombres de Servidor', 'JA3', 'JA3S', 'Protocolos']
    y_vals_fraudulento = [
        [r['media_entropias_ips_origen_fraudulento'] for r in resultados_tsallis],
        [r['media_entropias_ips_destino_fraudulento'] for r in resultados_tsallis],
        [r['media_entropias_ips_combinadas_fraudulento'] for r in resultados_tsallis],
        [r['media_entropias_server_names_fraudulento'] for r in resultados_tsallis],
        [r['media_entropias_JA3_fraudulento'] for r in resultados_tsallis],
        [r['media_entropias_JA3S_fraudulento'] for r in resultados_tsallis],
        [r['media_entropias_protocolos_fraudulento'] for r in resultados_tsallis]
    ]
    y_vals_legal = [
        [r['media_entropias_ips_origen_legal'] for r in resultados_tsallis],
        [r['media_entropias_ips_destino_legal'] for r in resultados_tsallis],
        [r['media_entropias_ips_combinadas_legal'] for r in resultados_tsallis],
        [r['media_entropias_server_names_legal'] for r in resultados_tsallis],
        [r['media_entropias_JA3_legal'] for r in resultados_tsallis],
        [r['media_entropias_JA3S_legal'] for r in resultados_tsallis],
        [r['media_entropias_protocolos_legal'] for r in resultados_tsallis]
    ]

    # Iterar sobre las subgráficas
    for i in range(4):
        for j in range(2):
            ax = axs[i, j]
            
            if i * 2 + j < len(etiquetas):
                ax.plot(qs_vals, y_vals_fraudulento[i * 2 + j], label='Fraudulento', marker='o')
                ax.plot(qs_vals, y_vals_legal[i * 2 + j], label='Legal', marker='o')
                ax.set_title(etiquetas[i * 2 + j])
                ax.set_xlabel('q')
                ax.set_ylabel('Entropía')
                ax.legend()
                ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
def graficar_todas_entropias_tsallis3(qs_vals, resultados_tsallis):
    fig, axs = plt.subplots(4, 2, figsize=(16, 16))
    fig.suptitle('Entropía de Tsallis - Todas las Categorías', fontsize=16)

    # Etiquetas y valores para cada gráfica
    etiquetas = ['IPs de Origen', 'IPs de Destino', 'IPs Combinadas', 'Nombres de Servidor', 'JA3', 'JA3S', 'Protocolos']
    y_vals_fraudulento = [
        [r['media_entropias_ips_origen_fraudulento'] for r in resultados_tsallis],
        [r['media_entropias_ips_destino_fraudulento'] for r in resultados_tsallis],
        [r['media_entropias_ips_combinadas_fraudulento'] for r in resultados_tsallis],
        [r['media_entropias_server_names_fraudulento'] for r in resultados_tsallis],
        [r['media_entropias_JA3_fraudulento'] for r in resultados_tsallis],
        [r['media_entropias_JA3S_fraudulento'] for r in resultados_tsallis],
        [r['media_entropias_protocolos_fraudulento'] for r in resultados_tsallis]
    ]
    y_vals_legal = [
        [r['media_entropias_ips_origen_legal'] for r in resultados_tsallis],
        [r['media_entropias_ips_destino_legal'] for r in resultados_tsallis],
        [r['media_entropias_ips_combinadas_legal'] for r in resultados_tsallis],
        [r['media_entropias_server_names_legal'] for r in resultados_tsallis],
        [r['media_entropias_JA3_legal'] for r in resultados_tsallis],
        [r['media_entropias_JA3S_legal'] for r in resultados_tsallis],
        [r['media_entropias_protocolos_legal'] for r in resultados_tsallis]
    ]

    # Iterar sobre las subgráficas
    for i, (etiqueta, y_fraudulento, y_legal) in enumerate(zip(etiquetas, y_vals_fraudulento, y_vals_legal)):
        ax = axs[i // 2, i % 2]  # Índices de fila y columna
        ax.plot(qs_vals, y_fraudulento, label='Fraudulento', marker='o')
        ax.plot(qs_vals, y_legal, label='Legal', marker='o')
        ax.set_title(etiqueta)
        ax.set_xlabel('q')
        ax.set_ylabel('Entropía')
        ax.legend()
        ax.grid(True)

    # Eliminar subgráficas adicionales
    for i in range(len(etiquetas), 4 * 2):
        fig.delaxes(axs.flatten()[i])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# Llamada a la función
graficar_todas_entropias_tsallis(qs_tsallis_a_probar, resultados_tsallis)



def graficar_todas_entropias_tsalli3s(qs_vals, resultados_tsallis):
    fig, axs = plt.subplots(4, 2, figsize=(16, 16))
    fig.suptitle('Entropía de Tsallis - Todas las Categorías', fontsize=16)

    # Etiquetas y valores para cada gráfica
    etiquetas = ['IPs de Origen', 'IPs de Destino', 'IPs Combinadas', 'Nombres de Servidor', 'JA3', 'JA3S', 'Protocolos']
    y_vals_fraudulento = [
        [r['media_entropias_ips_origen_fraudulento'] for r in resultados_tsallis],
        [r['media_entropias_ips_destino_fraudulento'] for r in resultados_tsallis],
        [r['media_entropias_ips_combinadas_fraudulento'] for r in resultados_tsallis],
        [r['media_entropias_server_names_fraudulento'] for r in resultados_tsallis],
        [r['media_entropias_JA3_fraudulento'] for r in resultados_tsallis],
        [r['media_entropias_JA3S_fraudulento'] for r in resultados_tsallis],
        [r['media_entropias_protocolos_fraudulento'] for r in resultados_tsallis]
    ]
    y_vals_legal = [
        [r['media_entropias_ips_origen_legal'] for r in resultados_tsallis],
        [r['media_entropias_ips_destino_legal'] for r in resultados_tsallis],
        [r['media_entropias_ips_combinadas_legal'] for r in resultados_tsallis],
        [r['media_entropias_server_names_legal'] for r in resultados_tsallis],
        [r['media_entropias_JA3_legal'] for r in resultados_tsallis],
        [r['media_entropias_JA3S_legal'] for r in resultados_tsallis],
        [r['media_entropias_protocolos_legal'] for r in resultados_tsallis]
    ]

    # Iterar sobre las subgráficas
    for i in range(7):
        ax = axs[i // 2, i % 2]  # Índices de fila y columna

        ax.plot(qs_vals, y_vals_fraudulento[i], label='Fraudulento', marker='o')
        ax.plot(qs_vals, y_vals_legal[i], label='Legal', marker='o')
        ax.set_title(etiquetas[i])
        ax.set_xlabel('q')
        ax.set_ylabel('Entropía')
        ax.legend()
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# Llamada a la función
graficar_todas_entropias_tsallis(qs_tsallis_a_probar, resultados_tsallis)




