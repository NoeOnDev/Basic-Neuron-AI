import os
import shutil
import pandas as pd
import numpy as np
from graficas import plot_evolucion_error, plot_comparacion_y, plot_evolucion_pesos

def recrear_directorios():
    directorios = ['graficas_epocas', 'grafica_evolucion_error', 'grafica_evolucion_pesos']
    for directorio in directorios:
        if os.path.exists(directorio):
            shutil.rmtree(directorio)
        os.makedirs(directorio)

def cargar_datos(ruta_archivo):
    conjunto_datos = pd.read_csv(ruta_archivo)
    x1 = conjunto_datos['x1'].values
    x2 = conjunto_datos['x2'].values
    x3 = conjunto_datos['x3'].values
    x4 = conjunto_datos['x4'].values
    yd = conjunto_datos['y'].values

    x = np.column_stack((x1, x2, x3, x4))
    y = yd

    return x, y

def predecir(x, w, b):
    return np.dot(x, w) + b

def error_cuadratico_medio(y_verdadero, y_predicho):
    return np.mean((y_verdadero - y_predicho) ** 2)

def entrenar(x, y, w, b, tasa_aprendizaje, epocas):
    m = x.shape[0]
    historial_costos = []
    historial_pesos = []

    for epoca in range(epocas):
        y_predicho = predecir(x, w, b)
        
        error = y_predicho - y
        
        dw = (2/m) * np.dot(x.T, error)
        db = (2/m) * np.sum(error)
        
        w -= tasa_aprendizaje * dw
        b -= tasa_aprendizaje * db
        
        costo = error_cuadratico_medio(y, y_predicho)
        
        print(f"Época {epoca} - Costo: {costo}")
        print(f"Pesos: {w}")
        print(f"Sesgo: {b}")
        
        if np.isnan(costo) or np.isinf(costo):
            print("Error: Costo no válido (NaN o infinito). Deteniendo el entrenamiento.")
            break
        
        historial_costos.append(costo)
        historial_pesos.append(np.append(w, b))
        
        if epoca == 0 or epoca == epocas // 2 or epoca == epocas - 1:
            plot_comparacion_y(y, y_predicho, epoca)
    
    return w, b, historial_costos, historial_pesos

def ejecutar_entrenamiento(ruta_archivo, tasa_aprendizaje, epocas):
    recrear_directorios()
    x, y = cargar_datos(ruta_archivo)

    np.random.seed(0)
    w = np.random.rand(4)
    b = np.random.rand()

    w, b, historial_costos, historial_pesos = entrenar(x, y, w, b, tasa_aprendizaje, epocas)

    y_predicho = predecir(x, w, b)
    costo_final = error_cuadratico_medio(y, y_predicho)

    plot_evolucion_error(epocas, historial_costos)
    plot_evolucion_pesos(epocas, historial_pesos)
    
    return w, b, costo_final
