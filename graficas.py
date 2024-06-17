import matplotlib.pyplot as plt
import numpy as np

def plot_evolucion_error(epocas, historial_costos):
    plt.figure(figsize=(10, 6))
    plt.plot(range(epocas), historial_costos, label='Error')
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
