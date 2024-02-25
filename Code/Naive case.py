# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 10:46:59 2024

@author: carlo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold,  GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix, roc_auc_score, f1_score, precision_recall_curve
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
from sklearn.ensemble import RandomForestClassifier

# Directorios para datos fraudulentos y legales
fraudulent_directory = 'C:/Users/carlo/OneDrive/Escritorio/TFM/Datasets/Final/Fraudulento/CSVs'
legal_directory = 'C:/Users/carlo/OneDrive/Escritorio/TFM/Datasets/Final/Legal/CSVs'

def calcular_entropia(frecuencia):
    return -np.sum(frecuencia * np.log2(frecuencia))

def calcular_entropia_server_names(df):
    server_names_freq = df['Server Name'].value_counts(normalize=True)
    entropia_server_names = calcular_entropia(server_names_freq)
    return entropia_server_names

def calcular_entropia_JA3(df):
    JA3_freq = df['JA3'].value_counts(normalize=True)
    entropia_JA3 = calcular_entropia(JA3_freq)
    return entropia_JA3

def calcular_entropia_JA3S(df):
    JA3S_freq = df['JA3S'].value_counts(normalize=True)
    entropia_JA3S = calcular_entropia(JA3S_freq)
    return entropia_JA3S

def calcular_entropia_protocolos(df):
    protocolos_freq = df['Protocol'].value_counts(normalize=True)
    entropia_protocolos = calcular_entropia(protocolos_freq)
    return entropia_protocolos

def calcular_entropia_ips(df):
    ip_origen_freq = df['Source'].value_counts(normalize=True)
    ip_destino_freq = df['Destination'].value_counts(normalize=True)
    ip_combinadas_freq = df[['Source', 'Destination']].stack().value_counts(normalize=True)

    entropia_ip_origen = calcular_entropia(ip_origen_freq)
    entropia_ip_destino = calcular_entropia(ip_destino_freq)
    entropia_ip_combinadas = calcular_entropia(ip_combinadas_freq)

    return entropia_ip_origen, entropia_ip_destino, entropia_ip_combinadas

def procesar_archivo_entropias(df):
    entropias_ips_origen = []
    entropias_ips_destino = []
    entropias_ips_combinadas = []
    entropias_server_names = []
    entropias_JA3 = []
    entropias_JA3S = []
    entropias_protocolos = []

    # Convertir la columna 'Time' a datetime
    df['Time'] = pd.to_datetime(df['Time'], unit='s')

    # Dividir el dataframe en intervalos de 1 minuto
    for _, intervalo_df in df.groupby(pd.Grouper(key='Time', freq='1Min')):
        entropia_ip_origen, entropia_ip_destino, entropia_ip_combinadas = calcular_entropia_ips(intervalo_df)
        entropia_server_names = calcular_entropia_server_names(intervalo_df)
        entropia_JA3 = calcular_entropia_JA3(intervalo_df)
        entropia_JA3S = calcular_entropia_JA3S(intervalo_df)
        entropia_protocolos = calcular_entropia_protocolos(intervalo_df)

        entropias_ips_origen.append(entropia_ip_origen)
        entropias_ips_destino.append(entropia_ip_destino)
        entropias_ips_combinadas.append(entropia_ip_combinadas)
        entropias_server_names.append(entropia_server_names)
        entropias_JA3.append(entropia_JA3)
        entropias_JA3S.append(entropia_JA3S)
        entropias_protocolos.append(entropia_protocolos)

    return (
        entropias_ips_origen,
        entropias_ips_destino,
        entropias_ips_combinadas,
        entropias_server_names,
        entropias_JA3,
        entropias_JA3S,
        entropias_protocolos
    )

def procesar_directorio_entropias(directorio_csv, tipo_trafico):
    entropias_ips_origen_lista = []
    entropias_ips_destino_lista = []
    entropias_ips_combinadas_lista = []
    entropias_server_names_lista = []
    entropias_JA3_lista = []
    entropias_JA3S_lista = []
    entropias_protocolos_lista = []

    for indice, archivo_csv in enumerate(os.listdir(directorio_csv)):
        if archivo_csv.endswith('.csv'):
            dtype_mapping = {'Server Name': str, 'JA3': str, 'JA3S': str}

            df = pd.read_csv(os.path.join(directorio_csv, archivo_csv), dtype=dtype_mapping)

            entropias_ips_origen, entropias_ips_destino, entropias_ips_combinadas, \
            entropias_server_names, entropias_JA3, entropias_JA3S, entropias_protocolos = procesar_archivo_entropias(df)

            entropias_ips_origen_lista.extend(entropias_ips_origen)
            entropias_ips_destino_lista.extend(entropias_ips_destino)
            entropias_ips_combinadas_lista.extend(entropias_ips_combinadas)
            entropias_server_names_lista.extend(entropias_server_names)
            entropias_JA3_lista.extend(entropias_JA3)
            entropias_JA3S_lista.extend(entropias_JA3S)
            entropias_protocolos_lista.extend(entropias_protocolos)

    return (
        entropias_ips_origen_lista,
        entropias_ips_destino_lista,
        entropias_ips_combinadas_lista,
        entropias_server_names_lista,
        entropias_JA3_lista,
        entropias_JA3S_lista,
        entropias_protocolos_lista
    )

# Procesar directorios por separado - Fraudulento
(entropias_ips_origen_fraudulento, entropias_ips_destino_fraudulento, entropias_ips_combinadas_fraudulento,
 entropias_server_names_fraudulento, entropias_JA3_fraudulento, entropias_JA3S_fraudulento, entropias_protocolos_fraudulento) = procesar_directorio_entropias(fraudulent_directory, 'Fraudulento')

# Procesar directorios por separado - Legal
(entropias_ips_origen_legal, entropias_ips_destino_legal, entropias_ips_combinadas_legal,
 entropias_server_names_legal, entropias_JA3_legal, entropias_JA3S_legal, entropias_protocolos_legal) = procesar_directorio_entropias(legal_directory, 'Legal')

# Verificar dimensiones de las variables
print("Dimensiones de las variables:")
print("Entropías IPs Origen (Fraudulento):", np.shape(entropias_ips_origen_fraudulento))
print("Entropías IPs Destino (Fraudulento):", np.shape(entropias_ips_destino_fraudulento))
print("Entropías IPs Combinadas (Fraudulento):", np.shape(entropias_ips_combinadas_fraudulento))
print("Entropías Server Names (Fraudulento):", np.shape(entropias_server_names_fraudulento))
print("Entropías JA3 (Fraudulento):", np.shape(entropias_JA3_fraudulento))
print("Entropías JA3S (Fraudulento):", np.shape(entropias_JA3S_fraudulento))
print("Entropías Protocolos (Fraudulento):", np.shape(entropias_protocolos_fraudulento))

print("\nEntropías IPs Origen (Legal):", np.shape(entropias_ips_origen_legal))
print("Entropías IPs Destino (Legal):", np.shape(entropias_ips_destino_legal))
print("Entropías IPs Combinadas (Legal):", np.shape(entropias_ips_combinadas_legal))
print("Entropías Server Names (Legal):", np.shape(entropias_server_names_legal))
print("Entropías JA3 (Legal):", np.shape(entropias_JA3_legal))
print("Entropías JA3S (Legal):", np.shape(entropias_JA3S_legal))
print("Entropías Protocolos (Legal):", np.shape(entropias_protocolos_legal))

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

# Configurar alpha para la Entropía de Rényi
alpha_renyi = 0.25  # Puedes ajustar este valor según tus necesidades

# Procesar directorios por separado usando Entropía de Rényi
(
    entropias_ips_origen_fraudulento_renyi,
    entropias_ips_destino_fraudulento_renyi,
    entropias_ips_combinadas_fraudulento_renyi,
    entropias_server_names_fraudulento_renyi,
    entropias_JA3_fraudulento_renyi,
    entropias_JA3S_fraudulento_renyi,
    entropias_protocolos_fraudulento_renyi
) = procesar_directorio_entropias_renyi(fraudulent_directory, 'Fraudulento', alpha_renyi)

(
    entropias_ips_origen_legal_renyi,
    entropias_ips_destino_legal_renyi,
    entropias_ips_combinadas_legal_renyi,
    entropias_server_names_legal_renyi,
    entropias_JA3_legal_renyi,
    entropias_JA3S_legal_renyi,
    entropias_protocolos_legal_renyi
) = procesar_directorio_entropias_renyi(legal_directory, 'Legal', alpha_renyi)


## Verificar dimensiones de las variables
print("Dimensiones de las variables:")
print("Entropías Renyi IPs Origen (Fraudulento):", np.shape(entropias_ips_origen_fraudulento_renyi))
print("Entropías Renyi IPs Destino (Fraudulento):", np.shape(entropias_ips_destino_fraudulento_renyi))
print("Entropías Renyi IPs Combinadas (Fraudulento):", np.shape(entropias_ips_combinadas_fraudulento_renyi))
print("Entropías Renyi Server Names (Fraudulento):", np.shape(entropias_server_names_fraudulento_renyi))
print("Entropías Renyi JA3 (Fraudulento):", np.shape(entropias_JA3_fraudulento_renyi))
print("Entropías Renyi JA3S (Fraudulento):", np.shape(entropias_JA3S_fraudulento_renyi))
print("Entropías Renyi Protocolos (Fraudulento):", np.shape(entropias_protocolos_fraudulento_renyi))

print("\nEntropías Renyi IPs Origen (Legal):", np.shape(entropias_ips_origen_legal_renyi))
print("Entropías Renyi IPs Destino (Legal):", np.shape(entropias_ips_destino_legal_renyi))
print("Entropías Renyi IPs Combinadas (Legal):", np.shape(entropias_ips_combinadas_legal_renyi))
print("Entropías Renyi Server Names (Legal):", np.shape(entropias_server_names_legal_renyi))
print("Entropías Renyi JA3 (Legal):", np.shape(entropias_JA3_legal_renyi))
print("Entropías Renyi JA3S (Legal):", np.shape(entropias_JA3S_legal_renyi))
print("Entropías Renyi Protocolos (Legal):", np.shape(entropias_protocolos_legal_renyi))

# Función para Calcular la Entropía de Tsallis:
def calcular_entropia_tsallis(frecuencia, q):
    if q == 1:
        return -np.sum(frecuencia * np.log2(frecuencia))
    else:
        return 1 / (q - 1) * (1 - (np.sum(frecuencia ** q)))

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

            # Calcula entropías usando Rényi
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

# Configurar q para la Entropía de Rényi
q_tsallis = 0.5  # Puedes ajustar este valor según tus necesidades

# Procesar directorios por separado usando Entropía de Rényi
(
    entropias_ips_origen_fraudulento_tsallis,
    entropias_ips_destino_fraudulento_tsallis,
    entropias_ips_combinadas_fraudulento_tsallis,
    entropias_server_names_fraudulento_tsallis,
    entropias_JA3_fraudulento_tsallis,
    entropias_JA3S_fraudulento_tsallis,
    entropias_protocolos_fraudulento_tsallis
) = procesar_directorio_entropias_tsallis(fraudulent_directory, 'Fraudulento', q_tsallis)

(
    entropias_ips_origen_legal_tsallis,
    entropias_ips_destino_legal_tsallis,
    entropias_ips_combinadas_legal_tsallis,
    entropias_server_names_legal_tsallis,
    entropias_JA3_legal_tsallis,
    entropias_JA3S_legal_tsallis,
    entropias_protocolos_legal_tsallis
) = procesar_directorio_entropias_tsallis(legal_directory, 'Legal', q_tsallis)


## Verificar dimensiones de las variables
print("Dimensiones de las variables:")
print("Entropías Tsallis IPs Origen (Fraudulento):", np.shape(entropias_ips_origen_fraudulento_tsallis))
print("Entropías Tsallis IPs Destino (Fraudulento):", np.shape(entropias_ips_destino_fraudulento_tsallis))
print("Entropías Tsallis IPs Combinadas (Fraudulento):", np.shape(entropias_ips_combinadas_fraudulento_tsallis))
print("Entropías Tsallis Server Names (Fraudulento):", np.shape(entropias_server_names_fraudulento_tsallis))
print("Entropías Tsallis JA3 (Fraudulento):", np.shape(entropias_JA3_fraudulento_tsallis))
print("Entropías Tsallis JA3S (Fraudulento):", np.shape(entropias_JA3S_fraudulento_tsallis))
print("Entropías Tsallis Protocolos (Fraudulento):", np.shape(entropias_protocolos_fraudulento_tsallis))

print("\nEntropías Tsallis IPs Origen (Legal):", np.shape(entropias_ips_origen_legal_tsallis))
print("Entropías Tsallis IPs Destino (Legal):", np.shape(entropias_ips_destino_legal_tsallis))
print("Entropías Tsallis IPs Combinadas (Legal):", np.shape(entropias_ips_combinadas_legal_tsallis))
print("Entropías Tsallis Server Names (Legal):", np.shape(entropias_server_names_legal_tsallis))
print("Entropías Tsallis JA3 (Legal):", np.shape(entropias_JA3_legal_tsallis))
print("Entropías Tsallis JA3S (Legal):", np.shape(entropias_JA3S_legal_tsallis))
print("Entropías Tsallis Protocolos (Legal):", np.shape(entropias_protocolos_legal_tsallis))
'''
Caso Knife
'''
# Define el umbral de decisión
umbral = 2.5
clasificaciones = []
# Entropías de tráfico legal
X_legal_knife = entropias_ips_combinadas_legal_renyi
y_legal_knife = np.zeros(len(X_legal_knife), dtype=int)  # Etiqueta 0 para tráfico legal

# Entropías de tráfico fraudulento
X_fraudulento_knife = entropias_ips_combinadas_fraudulento_renyi
y_fraudulento_knife = np.ones(len(X_fraudulento_knife), dtype=int)  # Etiqueta 1 para tráfico fraudulento

# Combina conjuntos de datos
X_knife = np.concatenate((X_legal_knife, X_fraudulento_knife), axis=0)
y_knife = np.concatenate((y_legal_knife, y_fraudulento_knife))

# Paso 1: Iterar sobre la lista X_knife y clasificar los valores
for valor in X_knife:
    if valor < umbral:
        clasificaciones.append(1)  # Clasificar como fraude
    else:
        clasificaciones.append(0)  # Clasificar como legal

# Paso 2: Comparar las clasificaciones obtenidas con las etiquetas y_knife
correctos = 0
total = len(y_knife)

for pred, etiqueta in zip(clasificaciones, y_knife):
    if pred == etiqueta:
        correctos += 1
        
print("Número de correctos:", correctos)

# Paso 3: Calcular la precisión de clasificación
precision = (correctos / total) * 100

# Definir la cantidad de clasificaciones correctas e incorrectas
incorrectos = total - correctos

# Etiquetas para las barras
etiquetas = ['Correctos', 'Incorrectos']

# Alturas de las barras
alturas = [correctos, incorrectos]

# Colores para las barras
colores = ['green', 'red']

# Crear gráfico de barras
plt.figure(figsize=(8, 6))
plt.bar(etiquetas, alturas, color=colores)

# Añadir etiquetas y título
plt.xlabel('Clasificación')
plt.ylabel('Cantidad')
plt.title('Clasificación de tráfico con umbral de decisión de {:.1f}'.format(umbral))

# Mostrar porcentaje de precisión en la parte superior de las barras
for i in range(len(etiquetas)):
    plt.text(i, alturas[i] + 0.5, '{:.1f}%'.format((alturas[i] / total) * 100), ha='center', va='bottom')

# Mostrar gráfico
plt.show()

#print("Clasificaciones:", clasificaciones)
#print("Etiquetas verdaderas:", y_knife)
print("Precisión de clasificación:", precision)

'''
Aplicamos SVM y Random Forest con las características calculadas
'''
# Combina características para tráfico legal
X_legal = np.column_stack((entropias_ips_origen_legal, entropias_ips_destino_legal, entropias_ips_combinadas_legal,
                           entropias_server_names_legal, entropias_JA3_legal, entropias_JA3S_legal, entropias_protocolos_legal,
                           entropias_ips_origen_legal_renyi, entropias_ips_destino_legal_renyi, entropias_ips_combinadas_legal_renyi,
                           entropias_server_names_legal_renyi, entropias_JA3_legal_renyi, entropias_JA3S_legal_renyi, entropias_protocolos_legal_renyi,
                           entropias_ips_origen_legal_tsallis, entropias_ips_destino_legal_tsallis, entropias_ips_combinadas_legal_tsallis,
                           entropias_server_names_legal_tsallis, entropias_JA3_legal_tsallis, entropias_JA3S_legal_tsallis, entropias_protocolos_legal_tsallis))

y_legal = np.zeros(X_legal.shape[0])  # Etiqueta 0 para tráfico legal

# Combina características para tráfico fraudulento
X_fraudulento = np.column_stack((entropias_ips_origen_fraudulento, entropias_ips_destino_fraudulento, entropias_ips_combinadas_fraudulento,
                                 entropias_server_names_fraudulento, entropias_JA3_fraudulento, entropias_JA3S_fraudulento, entropias_protocolos_fraudulento,
                                 entropias_ips_origen_fraudulento_renyi, entropias_ips_destino_fraudulento_renyi, entropias_ips_combinadas_fraudulento_renyi,
                                 entropias_server_names_fraudulento_renyi, entropias_JA3_fraudulento_renyi, entropias_JA3S_fraudulento_renyi, entropias_protocolos_fraudulento_renyi,
                                 entropias_ips_origen_fraudulento_tsallis, entropias_ips_destino_fraudulento_tsallis, entropias_ips_combinadas_fraudulento_tsallis,
                                 entropias_server_names_fraudulento_tsallis, entropias_JA3_fraudulento_tsallis, entropias_JA3S_fraudulento_tsallis, entropias_protocolos_fraudulento_tsallis))

y_fraudulento = np.ones(X_fraudulento.shape[0])  # Etiqueta 1 para tráfico fraudulento

# Combina conjuntos de datos
X = np.concatenate((X_legal, X_fraudulento), axis=0)
y = np.concatenate((y_legal, y_fraudulento))

column_names = [
    'IPs_Origen_Shannon', 'IPs_Destino_Shannon', 'IPs_Combinadas_Shannon',
    'Server_Names_Shannon', 'JA3_Shannon', 'JA3S_Shannon', 'Protocolos_Shannon',
    'IPs_Origen_Renyi', 'IPs_Destino_Renyi', 'IPs_Combinadas_Renyi',
    'Server_Names_Renyi', 'JA3_Renyi', 'JA3S_Renyi', 'Protocolos_Renyi',
    'IPs_Origen_Tsallis', 'IPs_Destino_Tsallis', 'IPs_Combinadas_Tsallis',
    'Server_Names_Tsallis', 'JA3_Tsallis', 'JA3S_Tsallis', 'Protocolos_Tsallis'
]

# Crea un DataFrame de Pandas con tus datos y nombres de columna
df_legal = pd.DataFrame(X_legal, columns=column_names)
df_legal['Label'] = y_legal  # Añade la columna de etiquetas

df_fraudulento = pd.DataFrame(X_fraudulento, columns=column_names)
df_fraudulento['Label'] = y_fraudulento  # Añade la columna de etiquetas

# Concatena ambos DataFrames
df_combined = pd.concat([df_legal, df_fraudulento], axis=0)

# Selecciona las características relevantes para X y la etiqueta para y
features =  [
            'IPs_Combinadas_Renyi',
]
X = df_combined[features]
y = df_combined['Label']

# Dividir datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Entrenamiento y evaluación del modelo SVM
# Define los hiperparámetros a ajustar y sus rangos de valores
param_grid_svm = {'C': [0.1, 1, 10, 100],
                  'gamma': [0.001, 0.01, 0.1, 1],
                  'kernel': ['linear', 'rbf', 'poly']}

# Inicializa el clasificador SVM
svm = SVC()

# Inicializa el objeto GridSearchCV
grid_search_svm = GridSearchCV(estimator=svm, param_grid=param_grid_svm, cv=5, scoring='accuracy')

# Entrena el modelo utilizando validación cruzada
grid_search_svm.fit(X_train, y_train)

# Muestra la precisión de la validación cruzada
print("Precisión de la validación cruzada para SVM:", grid_search_svm.best_score_)

# Muestra los mejores hiperparámetros encontrados
print("Mejores hiperparámetros encontrados para SVM:")
print(grid_search_svm.best_params_)

# Evalúa el modelo en el conjunto de prueba
best_svm = grid_search_svm.best_estimator_
y_pred_svm = best_svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("Precisión en el conjunto de prueba para SVM:", accuracy_svm)

# Obtener el informe de clasificación
report_svm = classification_report(y_test, y_pred_svm)

# Calcula las probabilidades de predicción para la clase positiva
probs_svm = best_svm.decision_function(X_test)

# Calcula la tasa de falsos positivos, la tasa de verdaderos positivos y los umbrales utilizando la curva ROC
fpr_svm, tpr_svm, thresholds_svm = roc_curve(y_test, probs_svm)

# Calcula el área bajo la curva ROC
roc_auc_svm = auc(fpr_svm, tpr_svm)

# Grafica la curva ROC para SVM
plt.figure(figsize=(8, 6))
plt.plot(fpr_svm, tpr_svm, color='darkorange', lw=2, label=f'Área bajo la curva ROC = {roc_auc_svm:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - Modelo SVM')
plt.legend(loc="lower right")
plt.show()

# Calcula y muestra la matriz de confusión
cm_svm = confusion_matrix(y_test, y_pred_svm)
cm_percent_svm = cm_svm.astype('float') / cm_svm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8, 6))
sns.heatmap(cm_percent_svm, annot=True, fmt='.2%', cmap='Blues', xticklabels=['Legal', 'Fraudulento'], yticklabels=['Legal', 'Fraudulento'])
plt.title('Matriz de Confusión - Modelo SVM')
plt.xlabel('Predicciones')
plt.ylabel('Reales')
plt.show()
print("Matriz de Confusión - Modelo SVM:")
print(cm_svm)
print("Informe de Clasificación - Modelo SVM:")
print(report_svm)

f1_svm = f1_score(y_test, y_pred_svm)

# Muestra el f1-score
print("F1-score para SVM:", f1_svm)

# 2. Entrenamiento y evaluación del modelo Random Forest con búsqueda de hiperparámetros
# Define los hiperparámetros a ajustar y sus rangos de valores
param_grid_rf = {'n_estimators': [50, 100, 150],
                 'max_depth': [None, 10, 20],
                 'min_samples_split': [2, 5, 10],
                 'min_samples_leaf': [1, 2, 4]}

# Inicializa el clasificador Random Forest
rf = RandomForestClassifier(random_state=42)

# Inicializa el objeto GridSearchCV
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, scoring='accuracy')

# Entrena el modelo utilizando validación cruzada
grid_search_rf.fit(X_train, y_train)

# Muestra la precisión de la validación cruzada
print("Precisión de la validación cruzada para Random Forest:", grid_search_rf.best_score_)

# Muestra los mejores hiperparámetros encontrados
print("Mejores hiperparámetros encontrados para Random Forest:")
print(grid_search_rf.best_params_)

# Evalúa el modelo en el conjunto de prueba
best_rf = grid_search_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Precisión en el conjunto de prueba para Random Forest:", accuracy_rf)

# Obtener el informe de clasificación
report_rf = classification_report(y_test, y_pred_rf)

# Calcula las probabilidades de predicción para la clase positiva
probs_rf = best_rf.predict_proba(X_test)

# Calcula la tasa de falsos positivos, la tasa de verdaderos positivos y los umbrales utilizando la curva ROC
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, probs_rf[:, 1])

# Calcula el área bajo la curva ROC
roc_auc_rf = auc(fpr_rf, tpr_rf)

# Grafica la curva ROC para Random Forest
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, color='darkorange', lw=2, label=f'Área bajo la curva ROC = {roc_auc_rf:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - Modelo Random Forest')
plt.legend(loc="lower right")
plt.show()

# Calcula y muestra la matriz de confusión
cm_rf = confusion_matrix(y_test, y_pred_rf)
cm_percent_rf = cm_rf.astype('float') / cm_rf.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8, 6))
sns.heatmap(cm_percent_rf, annot=True, fmt='.2%', cmap='Blues', xticklabels=['Legal', 'Fraudulento'], yticklabels=['Legal', 'Fraudulento'])
plt.title('Matriz de Confusión - Modelo Random Forest')
plt.xlabel('Predicciones')
plt.ylabel('Reales')
plt.show()
print("Matriz de Confusión - Modelo Random Forest:")
print(cm_rf)
print("Informe de Clasificación - Modelo Random Forest:")
print(report_rf)

# Calcula el f1-score
f1_rf = f1_score(y_test, y_pred_rf)

# Muestra el f1-score
print("F1-score para Random Forest:", f1_rf)

