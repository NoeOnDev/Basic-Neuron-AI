import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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
        
        if epoca % 100 == 0:
            print(f'Epoca {epoca}: Costo {costo}')
    
    return w, b, historial_costos, historial_pesos

tasa_aprendizaje = 0.1
epocas = 1000

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

plt.figure(figsize=(10, 6))
plt.plot(range(epocas), historial_costos, label='Costo')
plt.xlabel('Épocas')
plt.ylabel('Error Cuadrático Medio')
plt.title('Evolución del Error a lo largo de las Épocas')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(y, label='y-Deseada')
plt.plot(y_predicho, label='y-Calculada')
plt.xlabel('ID de Muestra')
plt.ylabel('Valor')
plt.title('Comparación entre y-Deseada y y-Calculada')
plt.legend()
plt.show()

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
plt.show()
