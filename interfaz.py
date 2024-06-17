from tkinter import Tk, Label, Entry, Button, END, ttk, font
import tkinter.messagebox as messagebox
import modelo

def mostrar_tabla_pesos(tree, w, b):
    for item in tree.get_children():
        tree.delete(item)
    
    for i, peso in enumerate(w):
        tree.insert('', 'end', values=(f'x{i+1}', peso))
    tree.insert('', 'end', values=('Sesgo', b))

def validar_entradas(tasa_aprendizaje_str, epocas_str):
    if not tasa_aprendizaje_str or not epocas_str:
        raise ValueError("Los campos no pueden estar vacíos")

    tasa_aprendizaje = float(tasa_aprendizaje_str)
    epocas = int(epocas_str)

    if not 0 <= tasa_aprendizaje <= 1:
        raise ValueError("La tasa de aprendizaje debe estar entre 0 y 1")

    if epocas <= 0:
        raise ValueError("Las épocas deben ser un número positivo")

    return tasa_aprendizaje, epocas

def ejecutar_entrenamiento(tree, entry_tasa_aprendizaje, entry_epocas):
    try:
        tasa_aprendizaje_str = entry_tasa_aprendizaje.get()
        epocas_str = entry_epocas.get()
        
        tasa_aprendizaje, epocas = validar_entradas(tasa_aprendizaje_str, epocas_str)
        
        w, b, costo_final = modelo.ejecutar_entrenamiento('2024.05.22 dataset 8A.xlsx', tasa_aprendizaje, epocas)

        mostrar_tabla_pesos(tree, w, b)

        messagebox.showinfo("Resultado", f"Entrenamiento completado\nCosto Final: {costo_final}")

    except ValueError as e:
        messagebox.showerror("Entrada no válida", str(e))

def main():
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

    tree = ttk.Treeview(root, columns=("Característica", "Peso"), show='headings', height=10, style='Treeview')
    tree.heading("Característica", text="Característica")
    tree.heading("Peso", text="Peso")
    tree.grid(row=3, column=0, columnspan=2, padx=20, pady=20)

    button_ejecutar = Button(root, text="Ejecutar", font=("Helvetica", font_size), command=lambda: ejecutar_entrenamiento(tree, entry_tasa_aprendizaje, entry_epocas))
    button_ejecutar.grid(row=2, column=0, columnspan=2, pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()
