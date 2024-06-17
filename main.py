import os
import shutil
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tkinter import Tk, Label, Entry, Button, Text, END
import tkinter.messagebox as messagebox
from prettytable import PrettyTable

def recrear_directorios():
    directorios = ['graficas_epocas', 'grafica_evolucion_error', 'grafica_evolucion_pesos']
    for directorio in directorios:
        if os.path.exists(directorio):
            shutil.rmtree(directorio)
        os.makedirs(directorio)

conjunto_datos = pd.read_excel('2024.05.22 dataset 8A.xlsx')

x1 = conjunto_datos['x1'].values
x2 = conjunto_datos['x2'].values
x3 = conjunto_datos['x3'].values
x4 = conjunto_datos['x4'].values
yd = conjunto_datos['y'].values

x = np.column_stack((x1, x2, x3, x4))
y = yd

escalador = StandardScaler()
x = escalador.fit_transform(x)

np.random.seed(0)
w = np.random.rand(4)
b = np.random.rand()

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
        historial_costos.append(costo)
        historial_pesos.append(np.append(w, b))
        
        if epoca == 0 or epoca == epocas // 2 or epoca == epocas - 1:
            print(f'Epoca {epoca}: Costo {costo}')
            plot_comparacion_y(y, y_predicho, epoca)
    
    return w, b, historial_costos, historial_pesos

def plot_evolucion_error(epocas, historial_costos):
    plt.figure(figsize=(10, 6))
    plt.plot(range(epocas), historial_costos, label='Costo')
    plt.xlabel('Épocas')
    plt.ylabel('Error Cuadrático Medio')
    plt.title('Evolución del Error a lo largo de las Épocas')
    plt.legend()
    plt.savefig('grafica_evolucion_error/evolucion_error.png')
    plt.close()

def plot_comparacion_y(y, y_predicho, epoca):
    plt.figure(figsize=(10, 6))
    plt.plot(y, label='y-Deseada')
    plt.plot(y_predicho, label='y-Calculada')
    plt.xlabel('ID de Muestra')
    plt.ylabel('Valor')
    plt.title(f'Comparación entre y-Deseada y y-Calculada (Época {epoca})')
    plt.legend()
    plt.savefig(f'graficas_epocas/comparacion_epoca_{epoca}.png')
    plt.close()

def plot_evolucion_pesos(epocas, historial_pesos):
    historial_pesos = np.array(historial_pesos)
    plt.figure(figsize=(10, 6))
    for i in range(historial_pesos.shape[1]):
        if i < historial_pesos.shape[1] - 1:
            plt.plot(range(epocas), historial_pesos[:, i], label=f'Peso {i+1} (x{i+1})')
        else:
            plt.plot(range(epocas), historial_pesos[:, i], label='Sesgo')
    plt.xlabel('Épocas')
    plt.ylabel('Peso')
    plt.title('Evolución de los Pesos a lo largo de las Épocas')
    plt.legend()
    plt.savefig('grafica_evolucion_pesos/evolucion_pesos.png')
    plt.close()

def mostrar_tabla_pesos(w, b):
    tabla = PrettyTable()
    tabla.field_names = ["Característica", "Peso"]
    for i, peso in enumerate(w):
        tabla.add_row([f'x{i+1}', peso])
    tabla.add_row(['Sesgo', b])
    
    text_resultado.delete(1.0, END)
    text_resultado.insert(END, tabla)

def ejecutar_entrenamiento():
    try:
        recrear_directorios()
        tasa_aprendizaje = float(entry_tasa_aprendizaje.get())
        epocas = int(entry_epocas.get())
        
        global w, b
        w, b, historial_costos, historial_pesos = entrenar(x, y, w, b, tasa_aprendizaje, epocas)

        y_predicho = predecir(x, w, b)
        costo_final = error_cuadratico_medio(y, y_predicho)

        print(f'Costo Final: {costo_final}')
        print(f'Pesos: {w}')
        print(f'Sesgo: {b}')

        pesos_df = pd.DataFrame({
            'Caracteristica': ['x1', 'x2', 'x3', 'x4'],
            'Peso': w
        })

        sesgo_df = pd.DataFrame({
            'Caracteristica': ['sesgo'],
            'Peso': [b]
        })

        pesos_finales_df = pd.concat([pesos_df, sesgo_df])

        print(pesos_finales_df)

        plot_evolucion_error(epocas, historial_costos)
        plot_evolucion_pesos(epocas, historial_pesos)
        
        mostrar_tabla_pesos(w, b)

    except ValueError:
        messagebox.showerror("Entrada no válida", "Por favor, introduce valores numéricos válidos para la tasa de aprendizaje y las épocas.")

root = Tk()
root.title("Entrenamiento de Modelo")
root.geometry("450x440")

root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_rowconfigure(2, weight=1)
root.grid_rowconfigure(3, weight=1)
root.grid_rowconfigure(4, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

Label(root, text="Tasa de Aprendizaje:").grid(row=0, column=0, sticky="e", padx=(0, 10), pady=(10, 10))
entry_tasa_aprendizaje = Entry(root)
entry_tasa_aprendizaje.grid(row=0, column=1, sticky="w", padx=(20, 100))

Label(root, text="Épocas:").grid(row=1, column=0, sticky="e", padx=(100, 10), pady=(10, 10))
entry_epocas = Entry(root)
entry_epocas.grid(row=1, column=1, sticky="w", padx=(20, 100))

button_ejecutar = Button(root, text="Ejecutar", command=ejecutar_entrenamiento)
button_ejecutar.grid(row=2, column=0, columnspan=2, pady=20)

text_resultado = Text(root, height=20, width=100)
text_resultado.grid(row=3, column=0, columnspan=2, padx=20, pady=20)

root.mainloop()
