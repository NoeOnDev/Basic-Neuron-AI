import os
import shutil
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tkinter import Tk, Label, Entry, Button, END, ttk, font
import tkinter.messagebox as messagebox
from graficas import plot_evolucion_error, plot_comparacion_y, plot_evolucion_pesos

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

def mostrar_tabla_pesos(w, b):
    for item in tree.get_children():
        tree.delete(item)
    
    for i, peso in enumerate(w):
        tree.insert('', 'end', values=(f'x{i+1}', peso))
    tree.insert('', 'end', values=('Sesgo', b))

def ejecutar_entrenamiento():
    try:
        recrear_directorios()
        tasa_aprendizaje_str = entry_tasa_aprendizaje.get()
        epocas_str = entry_epocas.get()
        
        if not tasa_aprendizaje_str or not epocas_str:
            raise ValueError("Los campos no pueden estar vacíos")

        tasa_aprendizaje = float(tasa_aprendizaje_str)
        epocas = int(epocas_str)

        if not 0 <= tasa_aprendizaje <= 1:
            raise ValueError("La tasa de aprendizaje debe estar entre 0 y 1")

        if epocas <= 0:
            raise ValueError("Las épocas deben ser un número positivo")

        np.random.seed(0)
        w = np.random.rand(4)
        b = np.random.rand()
        
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

    except ValueError as e:
        messagebox.showerror("Entrada no válida", str(e))

root = Tk()
root.title("Entrenamiento de Modelo")
root.geometry("450x440")

font_size = 11
style = ttk.Style()
style.configure("Treeview.Heading", font=("Helvetica", font_size))

root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_rowconfigure(2, weight=1)
root.grid_rowconfigure(3, weight=1)
root.grid_rowconfigure(4, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

Label(root, text="Tasa de Aprendizaje:", font=("Helvetica", font_size)).grid(row=0, column=0, sticky="e", padx=(0, 10), pady=(10, 10))
entry_tasa_aprendizaje = Entry(root)
entry_tasa_aprendizaje.grid(row=0, column=1, sticky="w", padx=(20, 100))

Label(root, text="Épocas:", font=("Helvetica", font_size)).grid(row=1, column=0, sticky="e", padx=(100, 10), pady=(10, 10))
entry_epocas = Entry(root)
entry_epocas.grid(row=1, column=1, sticky="w", padx=(20, 100))

button_ejecutar = Button(root, text="Ejecutar", font=("Helvetica", font_size), command=ejecutar_entrenamiento)
button_ejecutar.grid(row=2, column=0, columnspan=2, pady=20)

tree = ttk.Treeview(root, columns=("Característica", "Peso"), show='headings', height=10, style='Treeview')
tree.heading("Característica", text="Característica")
tree.heading("Peso", text="Peso")
tree.grid(row=3, column=0, columnspan=2, padx=20, pady=20)

root.mainloop()
