# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 16:56:34 2024

@author: carlo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import seaborn as sns
import time

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

def procesar_directorio(directorio_csv, tipo_trafico):
    resultados = []
    tiempos_entre_snis_hist = []
    dominios_por_intervalo_hist = []
    dominios_unicos_por_intervalo_hist = []

    for indice, archivo_csv in enumerate(os.listdir(directorio_csv)):
        if archivo_csv.endswith('.csv'):
            dtype_mapping = {'Server Name': str, 'JA3': str, 'JA3S': str}

            # Lee el archivo CSV con los tipos de datos especificados
            df = pd.read_csv(os.path.join(directorio_csv, archivo_csv), dtype=dtype_mapping)
            
            df['Time'] = pd.to_datetime(df['Time'], unit='s')
            df_reset = df[df['Server Name'].notnull()].reset_index(drop=True)
            df_reset['Tiempo_Entre_SNI'] = df_reset['Time'].diff()

            tiempos_entre_snis_hist.append(df_reset['Tiempo_Entre_SNI'].dropna().dt.total_seconds())

            df_reset = df_reset.dropna(subset=['Tiempo_Entre_SNI'])

            media_tiempo_entre_sni = df_reset['Tiempo_Entre_SNI'].mean()
            tiempo_entre_sni_segundos = df_reset['Tiempo_Entre_SNI'].dt.total_seconds().astype(float)
            varianza_tiempo_entre_sni = tiempo_entre_sni_segundos.var(ddof=1)
            media_tiempo_entre_sni_segundos = media_tiempo_entre_sni.total_seconds()
            coeficiente_variacion = (varianza_tiempo_entre_sni / (media_tiempo_entre_sni_segundos)**2)
            '''
            print(f"Archivo CSV #{indice + 1} (Tráfico {tipo_trafico})")
            print(f"Media de Tiempo entre SNIs: {media_tiempo_entre_sni_segundos} segundos\n")
            print(f"Varianza de Tiempo entre SNIs: {varianza_tiempo_entre_sni}\n")
            print(f"Coeficiente de Variación (C^2): {coeficiente_variacion}\n")
        
            if 0 < coeficiente_variacion < 0.7:
                clasificacion = "Se puede modelar mediante una distribución de tipo Erlang-m (C^2 = 1/m) (suma de exponenciales con tendencia al teorema del límite central)."
            elif 0.7 <= coeficiente_variacion < 1.3:
                clasificacion = "Comportamiento Poissoniano, se puede modelar como exponencial."
            else:
                clasificacion = "Se puede modelar mediante una distribución de tipo hiperexponencial (mixtura de exponenciales).
            print(f"Archivo CSV #{indice + 1} (Tráfico {tipo_trafico})")
            print(f"Clasificación: {clasificacion}\n")
            '''
            ventana_temporal = pd.to_timedelta('1 minute')
            df_reset['Intervalo'] = df_reset['Time'].dt.round(ventana_temporal)
            conteo_por_intervalo = df_reset[df_reset['Server Name'].notnull()].groupby('Intervalo')['Server Name'].count().reset_index()
            conteo_por_intervalo.columns = ['Intervalo', 'Dominios_Por_Ventana']
            '''
            plt.figure(figsize=(10, 6))
            plt.plot(conteo_por_intervalo['Intervalo'], conteo_por_intervalo['Dominios_Por_Ventana'], marker='o', linestyle='-')
            plt.title(f'Número de dominios por Intervalo de Tiempo - Archivo #{indice + 1} ({tipo_trafico} Trafico)')
            plt.xlabel('Intervalo de Tiempo')
            plt.ylabel('Número dominios')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.yticks(range(0, int(conteo_por_intervalo['Dominios_Por_Ventana'].max()) + 5, 5))
            plt.show()
            '''
            conteo_por_intervalo_unicos = df_reset[df_reset['Server Name'].notnull()].groupby(pd.Grouper(key='Time', freq=ventana_temporal))['Server Name'].nunique().reset_index()
            conteo_por_intervalo_unicos.columns = ['Intervalo', 'Dominios_Por_Ventana']
            '''
            plt.figure(figsize=(10, 6))
            plt.plot(conteo_por_intervalo_unicos['Intervalo'], conteo_por_intervalo_unicos['Dominios_Por_Ventana'], marker='o', linestyle='-')
            plt.title(f'Número de dominios únicos por Intervalo de Tiempo - Archivo #{indice + 1} ({tipo_trafico} Trafico)')
            plt.xlabel('Intervalo de Tiempo')
            plt.ylabel('Número de dominios únicos')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.yticks(range(0, int(conteo_por_intervalo_unicos['Dominios_Por_Ventana'].max()) + 5, 5))
            plt.show()
            '''
            resultados.append({
                'Nombre_Archivo': archivo_csv,
                'Media_Tiempo_Entre_SNIs': media_tiempo_entre_sni.total_seconds(),
                'Varianza_Tiempo_Entre_SNIs': varianza_tiempo_entre_sni,
                'Coeficiente_Variacion': coeficiente_variacion,
                #'Clasificacion': clasificacion,
                'Tipo_Trafico': tipo_trafico
            })

            dominios_por_intervalo_hist.append(conteo_por_intervalo['Dominios_Por_Ventana'])
            dominios_unicos_por_intervalo_hist.append(conteo_por_intervalo_unicos['Dominios_Por_Ventana'])

    return resultados, tiempos_entre_snis_hist, dominios_por_intervalo_hist, dominios_unicos_por_intervalo_hist

def procesar_directorio2(directorio_csv, tipo_trafico):
    ips_origen = []
    ips_destino = []
    server_name = []
    ja3 = []

    for indice, archivo_csv in enumerate(os.listdir(directorio_csv)):
        if archivo_csv.endswith('.csv'):
            dtype_mapping = {'Server Name': str, 'JA3': str, 'JA3S': str}

            # Lee el archivo CSV con los tipos de datos especificados
            df = pd.read_csv(os.path.join(directorio_csv, archivo_csv), dtype=dtype_mapping)
            
            # Hacer una copia del DataFrame original
            df_copy = df.copy()
            
            # Eliminar filas con valores vacíos en las columnas 'Server Name' y 'JA3' de la copia
            df_copy = df_copy.dropna(subset=['Server Name', 'JA3'])

            # Agregar los datos a las listas correspondientes desde la copia del DataFrame
            ips_origen.extend(df['Source'])
            ips_destino.extend(df['Destination'])
            server_name.extend(df_copy['Server Name'])
            ja3.extend(df_copy['JA3'])

    return ips_origen, ips_destino, server_name, ja3

# Procesar directorios por separado
ips_origen_fraudulento, ips_destino_fraudulento, server_name_fraudulento, ja3_fraudulento = procesar_directorio2(fraudulent_directory, 'Fraudulento')
ips_origen_legal, ips_destino_legal,server_name_legal, ja3_legal = procesar_directorio2(legal_directory, 'Legal')

# Crear la figura y los ejes
fig, axs = plt.subplots(2, 2, figsize=(18, 10))

# Graficar el histograma para las IPs de origen
axs[0, 0].hist(ips_origen_fraudulento, color='red', alpha=0.5, label='Fraudulento')
axs[0, 0].hist(ips_origen_legal, color='blue', alpha=0.5, label='Legal')
axs[0, 0].set_title('IPs de Origen')
axs[0, 0].legend()

# Graficar el histograma para las IPs de destino
axs[0, 1].hist(ips_destino_fraudulento, color='red', alpha=0.5, label='Fraudulento')
axs[0, 1].hist(ips_destino_legal, color='blue', alpha=0.5, label='Legal')
axs[0, 1].set_title('IPs de Destino')
axs[0, 1].legend()

# Graficar el histograma para las columnas Server name
axs[1, 0].hist(server_name_fraudulento, color='red', alpha=0.5, label='Fraudulento')
axs[1, 0].hist(server_name_legal, color='blue', alpha=0.5, label='Legal')
axs[1, 0].set_title('Server names')
axs[1, 0].legend()

# Graficar el histograma para las columnas JA3
axs[1, 1].hist(ja3_fraudulento, color='red', alpha=0.5, label='Fraudulento')
axs[1, 1].hist(ja3_legal, color='blue', alpha=0.5, label='Legal')
axs[1, 1].set_title('JA3')
axs[1, 1].legend()

# Configuración adicional
for ax in axs.flat:
    ax.set_xlabel('Valores')
    ax.set_ylabel('Frecuencia')
    ax.set_xticks([])  # Eliminar marcas en el eje x

# Ajustar el diseño de los subgráficos
plt.tight_layout()

# Mostrar la figura
plt.show()

# Procesar directorios por separado
resultados_fraudulento, tiempos_entre_snis_hist_fraudulento, dominios_por_intervalo_hist_fraudulento, dominios_unicos_por_intervalo_hist_fraudulento = procesar_directorio(fraudulent_directory, 'Fraudulento')
resultados_legal, tiempos_entre_snis_hist_legal, dominios_por_intervalo_hist_legal, dominios_unicos_por_intervalo_hist_legal = procesar_directorio(legal_directory, 'Legal')

# Crear DataFrames con los resultados
df_resultados_fraudulento = pd.DataFrame(resultados_fraudulento)
df_resultados_legal = pd.DataFrame(resultados_legal)

# Calcular la media de los resultados
media_Tiempo_Entre_SNI_fraudulento = df_resultados_fraudulento['Media_Tiempo_Entre_SNIs'].mean()
varianza_resultados_fraudulento = df_resultados_fraudulento['Varianza_Tiempo_Entre_SNIs'].mean()
coef_variacion_resultados_fraudulento = df_resultados_fraudulento['Coeficiente_Variacion'].mean()

media_Tiempo_Entre_SNI_legal = df_resultados_legal['Media_Tiempo_Entre_SNIs'].mean()
varianza_resultados_legal = df_resultados_legal['Varianza_Tiempo_Entre_SNIs'].mean()
coef_variacion_resultados_legal = df_resultados_legal['Coeficiente_Variacion'].mean()

# Imprimir los resultados finales
print(f"Media de Tiempo entre SNIs (Promedio de todos los archivos - Fraudulento): {media_Tiempo_Entre_SNI_fraudulento} segundos")
print(f"Varianza de Tiempo entre SNIs (Promedio de todos los archivos - Fraudulento): {varianza_resultados_fraudulento}")
print(f"Coeficiente de Variación (C^2) (Promedio de todos los archivos - Fraudulento): {coef_variacion_resultados_fraudulento}")

print(f"Media de Tiempo entre SNIs (Promedio de todos los archivos - Legal): {media_Tiempo_Entre_SNI_legal} segundos")
print(f"Varianza de Tiempo entre SNIs (Promedio de todos los archivos - Legal): {varianza_resultados_legal}")
print(f"Coeficiente de Variación (C^2) (Promedio de todos los archivos - Legal): {coef_variacion_resultados_legal}")

'''
# Clasificación de todo el directorio
def obtener_clasificacion_general(coef_variacion):
    if 0 < coef_variacion < 0.7:
        return "Se puede modelar mediante una distribución de tipo Erlang-m (C^2 = 1/m) (suma de exponenciales con tendencia al teorema del límite central)."
    elif 0.7 <= coef_variacion < 1.3:
        return "Comportamiento Poissoniano, se puede modelar como exponencial."
    else:
        return "Se puede modelar mediante una distribución de tipo hiperexponencial (mixtura de exponenciales)."

clasificacion_total_fraudulento = obtener_clasificacion_general(coef_variacion_resultados_fraudulento)
clasificacion_total_legal = obtener_clasificacion_general(coef_variacion_resultados_legal)

print(f"Clasificación total - Fraudulento: {clasificacion_total_fraudulento}\n")
print(f"Clasificación total - Legal: {clasificacion_total_legal}\n")
'''


# Crear la figura y los ejes
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Graficar el histograma para el tráfico fraudulento en la primera subgráfica
axs[0].hist(tiempos_entre_snis_hist_fraudulento, bins=100, edgecolor='blue', alpha=0.5)
axs[0].set_title('Tiempos entre SNIs - Tráfico Fraudulento')
axs[0].set_xlabel('Tiempo (segundos)')
axs[0].set_ylabel('Frecuencia')

# Graficar el histograma para el tráfico legal en la segunda subgráfica
axs[1].hist(tiempos_entre_snis_hist_legal, bins=100, edgecolor='blue', alpha=0.5)
axs[1].set_title('Tiempos entre SNIs - Tráfico Legal')
axs[1].set_xlabel('Tiempo (segundos)')
axs[1].set_ylabel('Frecuencia')

# Mostrar la figura
plt.tight_layout()
plt.show()

# Calcular los promedios de dominios y dominios únicos por intervalo - Fraudulento
promedios_dominios_fraudulento = []
promedios_dominios_unicos_fraudulento = []

for i in range(20):
    datos_intervalo_dominios = [intervalo[i] for intervalo in dominios_por_intervalo_hist_fraudulento if len(intervalo) > i and intervalo[i] > 0]
    datos_intervalo_dominios_unicos = [intervalo[i] for intervalo in dominios_unicos_por_intervalo_hist_fraudulento if len(intervalo) > i and intervalo[i] > 0]

    if datos_intervalo_dominios:
        promedio_dominios_fraudulento = np.mean(datos_intervalo_dominios)
        promedios_dominios_fraudulento.append(promedio_dominios_fraudulento)
    else:
        promedios_dominios_fraudulento.append(np.nan)

    if datos_intervalo_dominios_unicos:
        promedio_dominios_unicos_fraudulento = np.mean(datos_intervalo_dominios_unicos)
        promedios_dominios_unicos_fraudulento.append(promedio_dominios_unicos_fraudulento)
    else:
        promedios_dominios_unicos_fraudulento.append(np.nan)
'''
# Imprimir los resultados
print("\nPromedios de Dominios por Intervalo - Fraudulento:")
for i, promedio in enumerate(promedios_dominios_fraudulento):
    print(f"Intervalo {i + 1}: {promedio:.2f} dominios")

print("\nPromedios de Dominios Únicos por Intervalo - Fraudulento:")
for i, promedio_unico in enumerate(promedios_dominios_unicos_fraudulento):
    print(f"Intervalo {i + 1}: {promedio_unico:.2f} dominios únicos")
'''
    
promedios_dominios_legal = []
promedios_dominios_unicos_legal = []

for i in range(20):
    datos_intervalo_dominios = [intervalo[i] for intervalo in dominios_por_intervalo_hist_legal if len(intervalo) > i and intervalo[i] > 0]
    datos_intervalo_dominios_unicos = [intervalo[i] for intervalo in dominios_unicos_por_intervalo_hist_legal if len(intervalo) > i and intervalo[i] > 0]

    if datos_intervalo_dominios:
        promedio_dominios_legal = np.mean(datos_intervalo_dominios)
        promedios_dominios_legal.append(promedio_dominios_legal)
    else:
        promedios_dominios_legal.append(np.nan)

    if datos_intervalo_dominios_unicos:
        promedio_dominios_unicos_legal = np.mean(datos_intervalo_dominios_unicos)
        promedios_dominios_unicos_legal.append(promedio_dominios_unicos_legal)
    else:
        promedios_dominios_unicos_legal.append(np.nan)

'''
# Imprimir los resultados
print("\nPromedios de Dominios por Intervalo - Legal:")
for i, promedio in enumerate(promedios_dominios_legal):
    print(f"Intervalo {i + 1}: {promedio:.2f} dominios")

print("\nPromedios de Dominios Únicos por Intervalo - Legal:")
for i, promedio_unico in enumerate(promedios_dominios_unicos_legal):
    print(f"Intervalo {i + 1}: {promedio_unico:.2f} dominios únicos")
'''
    
# Remplazar NaN con 0 en los promedios
promedios_dominios_fraudulento = np.nan_to_num(promedios_dominios_fraudulento, nan=0)
promedios_dominios_unicos_fraudulento = np.nan_to_num(promedios_dominios_unicos_fraudulento, nan=0)

promedios_dominios_legal = np.nan_to_num(promedios_dominios_legal, nan=0)
promedios_dominios_unicos_legal = np.nan_to_num(promedios_dominios_unicos_legal, nan=0)
'''
# Ahora, puedes graficar los promedios como se mencionó en la respuesta anterior
# Graficar los promedios de dominios por intervalo - Fraudulento
plt.figure(figsize=(10, 6))
plt.plot(range(1, 21), promedios_dominios_fraudulento, marker='o', linestyle='-', label='Promedio de Dominios')
plt.title('Dominios por Intervalo - Fraudulento')
plt.xlabel('Intervalo')
plt.ylabel('Promedio de Dominios')
plt.xticks(range(0, 21, 5))  # Establecer el intervalo del eje x
plt.legend()
plt.grid(True)
plt.show()

# Graficar los promedios de dominios únicos por intervalo - Fraudulento
plt.figure(figsize=(10, 6))
plt.plot(range(1, 21), promedios_dominios_unicos_fraudulento, marker='o', linestyle='-', label='Promedio de Dominios Únicos')
plt.title('Dominios Únicos por Intervalo - Fraudulento')
plt.xlabel('Intervalo')
plt.ylabel('Promedio de Dominios Únicos')
plt.xticks(range(0, 21, 5))  # Establecer el intervalo del eje x
plt.legend()
plt.grid(True)
plt.show()

# Graficar los promedios de dominios por intervalo - Legal
plt.figure(figsize=(10, 6))
plt.plot(range(1, 21), promedios_dominios_legal, marker='o', linestyle='-', label='Promedio de Dominios')
plt.title('Dominios por Intervalo - Legal')
plt.xlabel('Intervalo')
plt.ylabel('Promedio de Dominios')
plt.xticks(range(0, 21, 5))  # Establecer el intervalo del eje x
plt.legend()
plt.grid(True)
plt.show()

# Graficar los promedios de dominios únicos por intervalo - Legal
plt.figure(figsize=(10, 6))
plt.plot(range(1, 21), promedios_dominios_unicos_legal, marker='o', linestyle='-', label='Promedio de Dominios Únicos')
plt.title('Dominios Únicos por Intervalo - Legal')
plt.xlabel('Intervalo')
plt.ylabel('Promedio de Dominios Únicos')
plt.xticks(range(0, 21, 5))  # Establecer el intervalo del eje x
plt.legend()
plt.grid(True)
plt.show()
'''
# Crear una figura y subgráficas
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Graficar los promedios de dominios por intervalo - Fraudulento
axs[0].plot(range(1, 21), promedios_dominios_fraudulento, marker='o', linestyle='-', label='Promedio de Dominios')
axs[0].set_title('Dominios por Intervalo - Fraudulento')
axs[0].set_xlabel('Intervalos de 1 minuto')
axs[0].set_ylabel('Promedio de Dominios')
axs[0].set_xticks(range(0, 21, 5))  # Establecer el intervalo del eje x
axs[0].legend()
axs[0].grid(True)

# Graficar los promedios de dominios por intervalo - Legal
axs[1].plot(range(1, 21), promedios_dominios_legal, marker='o', linestyle='-', label='Promedio de Dominios')
axs[1].set_title('Dominios por Intervalo - Legal')
axs[1].set_xlabel('Intervalos de 1 minuto')
axs[1].set_ylabel('Promedio de Dominios')
axs[1].set_xticks(range(0, 21, 5))  # Establecer el intervalo del eje x
axs[1].legend()
axs[1].grid(True)

# Mostrar la figura
plt.tight_layout()
plt.show()

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

# Define la función para graficar la densidad de las entropías
def plot_entropia_density(ax, entropias_fraudulento, entropias_legal, titulo):
    sns.kdeplot(entropias_fraudulento, ax=ax, label='Fraudulento', shade=True)
    sns.kdeplot(entropias_legal, ax=ax, label='Legal', shade=True)
    ax.set_title(titulo)
    ax.set_xlabel('Entropía')
    ax.set_ylabel('Densidad')
    ax.legend()

# Configurar el subplot
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Graficar la densidad de las entropías de IP origen
plot_entropia_density(axs[0], entropias_ips_origen_fraudulento, entropias_ips_origen_legal, 'Entropías de IP Origen')

# Graficar la densidad de las entropías de IP destino
plot_entropia_density(axs[1], entropias_ips_destino_fraudulento, entropias_ips_destino_legal, 'Entropías de IP Destino')

# Graficar la densidad de las entropías combinadas
plot_entropia_density(axs[2], entropias_ips_combinadas_fraudulento, entropias_ips_combinadas_legal, 'Entropías Combinadas')

plt.tight_layout()
plt.show()

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

# Define la función para graficar la densidad de las entropías
def plot_entropia_density_renyi(ax, entropias_fraudulento, entropias_legal, titulo):
    sns.kdeplot(entropias_fraudulento, ax=ax, label='Fraudulento', shade=True)
    sns.kdeplot(entropias_legal, ax=ax, label='Legal', shade=True)
    ax.set_title(titulo)
    ax.set_xlabel('Entropía')
    ax.set_ylabel('Densidad')
    ax.legend()

# Configurar el subplot
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Graficar la densidad de las entropías de IP origen
plot_entropia_density_renyi(axs[0], entropias_ips_origen_fraudulento_renyi, entropias_ips_origen_legal_renyi, 'Entropías de IP Origen')

# Graficar la densidad de las entropías de IP destino
plot_entropia_density_renyi(axs[1], entropias_ips_destino_fraudulento_renyi, entropias_ips_destino_legal_renyi, 'Entropías de IP Destino')

# Graficar la densidad de las entropías combinadas
plot_entropia_density_renyi(axs[2], entropias_ips_combinadas_fraudulento_renyi, entropias_ips_combinadas_legal, 'Entropías Combinadas')

plt.tight_layout()
plt.show()

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

# Define la función para graficar la densidad de las entropías
def plot_entropia_density_tsallis(ax, entropias_fraudulento, entropias_legal, titulo):
    sns.kdeplot(entropias_fraudulento, ax=ax, label='Fraudulento', shade=True)
    sns.kdeplot(entropias_legal, ax=ax, label='Legal', shade=True)
    ax.set_title(titulo)
    ax.set_xlabel('Entropía')
    ax.set_ylabel('Densidad')
    ax.legend()

# Configurar el subplot
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Graficar la densidad de las entropías de IP origen
plot_entropia_density_tsallis(axs[0], entropias_ips_origen_fraudulento_tsallis, entropias_ips_origen_legal_tsallis, 'Entropías de IP Origen')

# Graficar la densidad de las entropías de IP destino
plot_entropia_density_tsallis(axs[1], entropias_ips_destino_fraudulento_tsallis, entropias_ips_destino_legal_tsallis, 'Entropías de IP Destino')

# Graficar la densidad de las entropías combinadas
plot_entropia_density_tsallis(axs[2], entropias_ips_combinadas_fraudulento_tsallis, entropias_ips_combinadas_legal, 'Entropías Combinadas')

plt.tight_layout()
plt.show()

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
Aplicamos SVM y Random Forest con las características calculadas
'''
# Combina características para tráfico legal
'''
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
'''

X_legal = np.column_stack((entropias_ips_origen_legal_renyi, entropias_ips_destino_legal_renyi, entropias_ips_combinadas_legal_renyi,
                           entropias_ips_origen_legal_tsallis, entropias_ips_destino_legal_tsallis, entropias_ips_combinadas_legal_tsallis))
y_legal = np.zeros(X_legal.shape[0])  # Etiqueta 0 para tráfico legal

X_fraudulento = np.column_stack((entropias_ips_origen_fraudulento_renyi, entropias_ips_destino_fraudulento_renyi, entropias_ips_combinadas_fraudulento_renyi,
                                 entropias_ips_origen_fraudulento_tsallis, entropias_ips_destino_fraudulento_tsallis, entropias_ips_combinadas_fraudulento_tsallis))
y_fraudulento = np.ones(X_fraudulento.shape[0])  # Etiqueta 1 para tráfico fraudulento

'''
X_legal = np.column_stack((entropias_ips_origen_legal, entropias_ips_destino_legal, entropias_ips_combinadas_legal))
y_legal = np.zeros(X_legal.shape[0])  # Etiqueta 0 para tráfico legal

X_fraudulento = np.column_stack((entropias_ips_origen_fraudulento, entropias_ips_destino_fraudulento, entropias_ips_combinadas_fraudulento))
y_fraudulento = np.ones(X_fraudulento.shape[0])  # Etiqueta 1 para tráfico fraudulento
'''
# Dividir datos de prueba y entrenamiento para conexiones fraudulentas

X_fraudulento_train, X_fraudulento_test, y_fraudulento_train, y_fraudulento_test = train_test_split(X_fraudulento, y_fraudulento, test_size=0.2, random_state=42)

# Dividir datos de prueba y entrenamiento para conexiones legales
X_legal_train, X_legal_test, y_legal_train, y_legal_test = train_test_split(X_legal, y_legal, test_size=0.2, random_state=42)

# Concatenar datos de entrenamiento y prueba para conexiones fraudulentas y legales
X_train = np.concatenate((X_fraudulento_train, X_legal_train), axis=0)
y_train = np.concatenate((y_fraudulento_train, y_legal_train), axis=0)
X_test = np.concatenate((X_fraudulento_test, X_legal_test), axis=0)
y_test = np.concatenate((y_fraudulento_test, y_legal_test), axis=0)

'''
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
'''
# Selecciona las características relevantes para X y la etiqueta para y
'''
features =  [
    'IPs_Origen_Renyi', 'IPs_Destino_Renyi', 'IPs_Combinadas_Renyi',
]
'''
#X = df_combined[features]
#y = df_combined['Label']

# Dividir datos en conjunto de entrenamiento y prueba
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Entrenamiento y evaluación del modelo SVM
# Define los hiperparámetros a ajustar y sus rangos de valores
'''
param_grid_svm = {'C': [0.1, 1, 10, 100],
                  'gamma': [0.001, 0.01, 0.1, 1],
                  'kernel': ['linear', 'rbf', 'poly']}
'''

param_grid_svm = {'C': [10],
                  'gamma': [1],
                  'kernel':['rbf']}
'''
param_grid_svm = {'C': [100],
                  'gamma': [1],
                  'kernel':['rbf']}
'''
# Inicializa el clasificador SVM
svm = SVC()

# Entrenamiento y evaluación del modelo SVM con búsqueda de hiperparámetros
start_time_svm_train = time.time()

# Inicializa el objeto GridSearchCV
grid_search_svm = GridSearchCV(estimator=svm, param_grid=param_grid_svm, cv=5, scoring='accuracy')

# Entrena el modelo utilizando validación cruzada
grid_search_svm.fit(X_train, y_train)

end_time_svm_train = time.time()
svm_training_time = end_time_svm_train - start_time_svm_train

# Muestra el tiempo de entrenamiento para SVM
print("Tiempo de entrenamiento para SVM:", svm_training_time, "segundos")

# Muestra la precisión de la validación cruzada
print("Precisión de la validación cruzada para SVM:", grid_search_svm.best_score_)

# Muestra los mejores hiperparámetros encontrados
print("Mejores hiperparámetros encontrados para SVM:")
print(grid_search_svm.best_params_)

# Evalúa el modelo en el conjunto de prueba
best_svm = grid_search_svm.best_estimator_
start_time_svm_eval = time.time()
y_pred_svm = best_svm.predict(X_test)
end_time_svm_eval = time.time()
svm_evaluation_time = (end_time_svm_eval - start_time_svm_eval)* 1e3

# Muestra el tiempo de evaluación para SVM
print("Tiempo de evaluación para SVM:", svm_evaluation_time, "milisegundos")

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
'''
param_grid_rf = {'n_estimators': [50, 100, 150],
                 'max_depth': [None, 10, 20],
                 'min_samples_split': [2, 5, 10],
                 'min_samples_leaf': [1, 2, 4]}
'''
param_grid_rf = {'n_estimators': [150],
                 'max_depth': [None],
                 'min_samples_split': [5],
                 'min_samples_leaf': [1]}
'''
param_grid_rf = {'n_estimators': [150],
                 'max_depth': [None],
                 'min_samples_split': [10],
                 'min_samples_leaf': [1]}
'''
# Inicializa el clasificador Random Forest
rf = RandomForestClassifier(random_state=42)

# Entrenamiento y evaluación del modelo Random Forest con búsqueda de hiperparámetros
start_time_rf_train = time.time()

# Inicializa el objeto GridSearchCV
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, scoring='accuracy')

# Entrena el modelo utilizando validación cruzada
grid_search_rf.fit(X_train, y_train)

end_time_rf_train = time.time()
rf_training_time = end_time_rf_train - start_time_rf_train

# Muestra el tiempo de entrenamiento para Random Forest
print("Tiempo de entrenamiento para Random Forest:", rf_training_time, "segundos")


# Muestra la precisión de la validación cruzada
print("Precisión de la validación cruzada para Random Forest:", grid_search_rf.best_score_)

# Muestra los mejores hiperparámetros encontrados
print("Mejores hiperparámetros encontrados para Random Forest:")
print(grid_search_rf.best_params_)

# Evalúa el modelo en el conjunto de prueba
best_rf = grid_search_rf.best_estimator_
start_time_rf_eval = time.time()
y_pred_rf = best_rf.predict(X_test)
end_time_rf_eval = time.time()
rf_evaluation_time = (end_time_rf_eval - start_time_rf_eval)* 1e3

# Muestra el tiempo de evaluación para Random Forest
print("Tiempo de evaluación para Random Forest:", rf_evaluation_time, "milisegundos")

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

# Inicializa contadores para contar el número de intervalos clasificados como "Fraudulento" y "Legal" por cada modelo
svm_fraudulentos = 0
svm_legales = 0
rf_fraudulentos = 0
rf_legales = 0

# Itera sobre los intervalos de las conexiones fraudulentas
for intervalo in X_fraudulento_test:
    # Realiza la predicción para el intervalo utilizando el modelo SVM
    prediccion_svm = best_svm.predict(intervalo.reshape(1, -1))  # reshape para convertir en formato de una muestra
    # Realiza la predicción para el intervalo utilizando el modelo Random Forest
    prediccion_rf = best_rf.predict(intervalo.reshape(1, -1))  # reshape para convertir en formato de una muestra
    
    # Incrementa los contadores según la predicción de cada modelo
    if prediccion_svm == 1:
        svm_fraudulentos += 1
    else:
        svm_legales += 1
    
    if prediccion_rf == 1:
        rf_fraudulentos += 1
    else:
        rf_legales += 1

# Itera sobre los intervalos de las conexiones legales
for intervalo in X_legal_test:
    # Realiza la predicción para el intervalo utilizando el modelo SVM
    prediccion_svm = best_svm.predict(intervalo.reshape(1, -1))  # reshape para convertir en formato de una muestra
    # Realiza la predicción para el intervalo utilizando el modelo Random Forest
    prediccion_rf = best_rf.predict(intervalo.reshape(1, -1))  # reshape para convertir en formato de una muestra
    
    # Incrementa los contadores según la predicción de cada modelo
    if prediccion_svm == 1:
        svm_fraudulentos += 1
    else:
        svm_legales += 1
    
    if prediccion_rf == 1:
        rf_fraudulentos += 1
    else:
        rf_legales += 1

# Etiquetas para las barras
categorias = ['Fraudulento', 'Legal']

# Alturas de las barras para SVM
svm_resultados = [svm_fraudulentos, svm_legales]

# Alturas de las barras para Random Forest
rf_resultados = [rf_fraudulentos, rf_legales]

# Índices para las posiciones de las barras
indices = np.arange(len(categorias))
ancho_barra = 0.35

# Graficar los resultados
fig, ax = plt.subplots()
barra_svm = ax.bar(indices - ancho_barra/2, svm_resultados, ancho_barra, label='SVM', color='blue')
barra_rf = ax.bar(indices + ancho_barra/2, rf_resultados, ancho_barra, label='Random Forest', color='orange')
'''
# Añadir etiquetas a las barras
for barra in [barra_svm, barra_rf]:
    for idx, altura in enumerate(barra):
        etiqueta = categorias[idx]
        ax.text(altura.get_x() + altura.get_width() / 2, altura.get_height(), etiqueta,
                ha='center', va='bottom', fontsize=10, color='black')
'''
# Etiquetas, título y leyenda
ax.set_xlabel('Categoría')
ax.set_ylabel('Cantidad de Intervalos')
ax.set_title('Predicciones de Intervalos por Modelo')
ax.set_xticks(indices)
ax.set_xticklabels(categorias)
ax.legend()

# Mostrar la gráfica
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
plt.show()
#DETECCIÓN POR INTERVALOS DE 1 MINUTO
# Inicializa los contadores para contar el número de intervalos clasificados como "Fraudulento" y "Legal" por cada modelo
svm_fraudulentos_f = 0
svm_legales_f = 0
rf_fraudulentos_f = 0
rf_legales_f = 0

svm_fraudulentos_l = 0
svm_legales_l = 0
rf_fraudulentos_l = 0
rf_legales_l = 0

# Itera sobre los intervalos de las conexiones fraudulentas
for intervalo in X_fraudulento_test[168:188]:
    # Realiza la predicción para el intervalo utilizando el modelo SVM
    prediccion_svm = best_svm.predict(intervalo.reshape(1, -1))  # reshape para convertir en formato de una muestra
    # Realiza la predicción para el intervalo utilizando el modelo Random Forest
    prediccion_rf = best_rf.predict(intervalo.reshape(1, -1))  # reshape para convertir en formato de una muestra
    
    # Incrementa los contadores según la predicción de cada modelo
    if prediccion_svm == 1:
        svm_fraudulentos_f += 1
    else:
        svm_legales_f += 1
    
    if prediccion_rf == 1:
        rf_fraudulentos_f += 1
    else:
        rf_legales_f += 1

# Itera sobre los intervalos de las conexiones legales
for intervalo in X_legal_test[168:188]:
    # Realiza la predicción para el intervalo utilizando el modelo SVM
    prediccion_svm = best_svm.predict(intervalo.reshape(1, -1))  # reshape para convertir en formato de una muestra
    # Realiza la predicción para el intervalo utilizando el modelo Random Forest
    prediccion_rf = best_rf.predict(intervalo.reshape(1, -1))  # reshape para convertir en formato de una muestra
    
    # Incrementa los contadores según la predicción de cada modelo
    if prediccion_svm == 1:
        svm_fraudulentos_l += 1
    else:
        svm_legales_l += 1
    
    if prediccion_rf == 1:
        rf_fraudulentos_l += 1
    else:
        rf_legales_l += 1

# Etiquetas para las barras
categorias = ['Fraudulento', 'Legal']

# Alturas de las barras para SVM - Conjunto Fraudulento
svm_resultados_f = [svm_fraudulentos_f, svm_legales_f]

# Alturas de las barras para Random Forest - Conjunto Fraudulento
rf_resultados_f = [rf_fraudulentos_f, rf_legales_f]

# Alturas de las barras para SVM - Conjunto Legal
svm_resultados_l = [svm_fraudulentos_l, svm_legales_l]

# Alturas de las barras para Random Forest - Conjunto Legal
rf_resultados_l = [rf_fraudulentos_l, rf_legales_l]

# Índices para las posiciones de las barras
indices = np.arange(len(categorias))
ancho_barra = 0.35

# Graficar los resultados para el conjunto Fraudulento
fig, axs = plt.subplots(2, figsize=(8, 8))
fig.suptitle('Clasificación de una conexión dividida en 20 intervalos por modelo')

# Gráfico para el conjunto Fraudulento
axs[0].bar(indices - ancho_barra/2, svm_resultados_f, ancho_barra, label='SVM', color='blue')
axs[0].bar(indices + ancho_barra/2, rf_resultados_f, ancho_barra, label='Random Forest', color='orange')
axs[0].set_title('Conjunto Fraudulento')
axs[0].set_xlabel('Categoría')
axs[0].set_ylabel('Cantidad de Intervalos de 1 minuto')
axs[0].set_xticks(indices)
axs[0].set_xticklabels(categorias)
axs[0].set_yticks(np.arange(0, max(max(svm_resultados_f), max(rf_resultados_f)) + 5, 5))  # Establece el intervalo del eje y
axs[0].legend()

# Gráfico para el conjunto Legal
axs[1].bar(indices - ancho_barra/2, svm_resultados_l, ancho_barra, label='SVM', color='blue')
axs[1].bar(indices + ancho_barra/2, rf_resultados_l, ancho_barra, label='Random Forest', color='orange')
axs[1].set_title('Conjunto Legal')
axs[1].set_xlabel('Categoría')
axs[1].set_ylabel('Cantidad de Intervalos de 1 minuto')
axs[1].set_xticks(indices)
axs[1].set_xticklabels(categorias)
axs[1].set_yticks(np.arange(0, max(max(svm_resultados_l), max(rf_resultados_l)) + 5, 5))  # Establece el intervalo del eje y
axs[1].legend()

plt.tight_layout()
plt.show()

# Realizar predicciones para el conjunto de datos fraudulentos
predicciones_svm_fraudulento = best_svm.predict(X_fraudulento_test[168:188])
predicciones_rf_fraudulento = best_rf.predict(X_fraudulento_test[168:188])

# Realizar predicciones para el conjunto de datos legales
predicciones_svm_legal = best_svm.predict(X_legal_test[168:188])
predicciones_rf_legal = best_rf.predict(X_legal_test[168:188])

# Calcular la precisión de cada modelo en sus respectivos conjuntos de prueba
precision_svm_fraudulento = accuracy_score(y_fraudulento_test[168:188], predicciones_svm_fraudulento)
precision_rf_fraudulento = accuracy_score(y_fraudulento_test[168:188], predicciones_rf_fraudulento)

precision_svm_legal = accuracy_score(y_legal_test[168:188], predicciones_svm_legal)
precision_rf_legal = accuracy_score(y_legal_test[168:188], predicciones_rf_legal)

# Determinar la clasificación general en función de la precisión
clasificacion_general_svm_fraudulento = 'Fraudulento' if precision_svm_fraudulento >= 0.9 else 'Legal'
clasificacion_general_rf_fraudulento = 'Fraudulento' if precision_rf_fraudulento >= 0.9 else 'Legal'

clasificacion_general_svm_legal = 'Legal' if precision_svm_legal >= 0.9 else 'Fraudulento'
clasificacion_general_rf_legal = 'Legal' if precision_rf_legal >= 0.9 else 'Fraudulento'

print("Clasificación general del conjunto fraudulento por SVM:", clasificacion_general_svm_fraudulento)
print("Clasificación general del conjunto fraudulento por Random Forest:", clasificacion_general_rf_fraudulento)
print("Clasificación general del conjunto legal por SVM:", clasificacion_general_svm_legal)
print("Clasificación general del conjunto legal por Random Forest:", clasificacion_general_rf_legal)

# Predicciones de SVM y Random Forest para el conjunto de datos fraudulentos
predicciones_svm_f = best_svm.predict(X_fraudulento_test[168:188])
predicciones_rf_f = best_rf.predict(X_fraudulento_test[168:188])

# Predicciones de SVM y Random Forest para el conjunto de datos legales
predicciones_svm_l = best_svm.predict(X_legal_test[168:188])
predicciones_rf_l = best_rf.predict(X_legal_test[168:188])

# Crear figuras y ejes para los gráficos
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Gráfico para el conjunto de datos fraudulentos
axs[0].plot(predicciones_svm_f, label='SVM', marker='o')
axs[0].plot(predicciones_rf_f, label='Random Forest', marker='o')
axs[0].set_title('Predicciones para el conjunto de datos fraudulentos')
axs[0].set_xlabel('Intervalos de 1 minuto')
axs[0].set_ylabel('Predicción')
axs[0].set_xticks(range(20))
axs[0].set_yticks([0, 1])
axs[0].set_yticklabels(['Legal', 'Fraudulento'])  # Etiquetas para el eje y
axs[0].legend()

# Gráfico para el conjunto de datos legales
axs[1].plot(predicciones_svm_l, label='SVM', marker='o')
axs[1].plot(predicciones_rf_l, label='Random Forest', marker='o')
axs[1].set_title('Predicciones para el conjunto de datos legales')
axs[1].set_xlabel('Intervalos de 1 minuto')
axs[1].set_ylabel('Predicción')
axs[1].set_xticks(range(20))
axs[1].set_yticks([0, 1])
axs[1].set_yticklabels(['Legal', 'Fraudulento'])  # Etiquetas para el eje y
axs[1].legend()

# Ajustar espacio entre subgráficos
plt.tight_layout()

# Mostrar los gráficos
plt.show()