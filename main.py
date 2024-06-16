import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

conjunto_datos = pd.read_excel('2024.05.22 dataset 8A.xlsx')

x1 = conjunto_datos['x1'].values
x2 = conjunto_datos['x2'].values
x3 = conjunto_datos['x3'].values
x4 = conjunto_datos['x4'].values
yd = conjunto_datos['y'].values

X = np.column_stack((x1, x2, x3, x4))
y = yd

escalador = StandardScaler()
X = escalador.fit_transform(X)

np.random.seed(0)
w = np.random.rand(4)
b = np.random.rand()

def predecir(X, w, b):
    return np.dot(X, w) + b

def error_cuadratico_medio(y_verdadero, y_predicho):
    return np.mean((y_verdadero - y_predicho) ** 2)

def entrenar(X, y, w, b, tasa_aprendizaje, epocas):
    m = X.shape[0]
    
    for epoca in range(epocas):
        y_predicho = predecir(X, w, b)
        
        error = y_predicho - y
        
        dw = (2/m) * np.dot(X.T, error)
        db = (2/m) * np.sum(error)
        
        w -= tasa_aprendizaje * dw
        b -= tasa_aprendizaje * db
        
        if epoca % 100 == 0:
            costo = error_cuadratico_medio(y, y_predicho)
            print(f'Epoca {epoca}: Costo {costo}')
    
    return w, b

tasa_aprendizaje = 0.1
epocas = 1000

w, b = entrenar(X, y, w, b, tasa_aprendizaje, epocas)

y_predicho = predecir(X, w, b)

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