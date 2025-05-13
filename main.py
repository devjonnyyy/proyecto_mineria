import tkinter as tk
from tkinter import filedialog, messagebox
from dataset_analysis import analizar_csv
import multilayer_perceptron_training as mpt
import sign_language_images_to_csv as slitc
import sign_language_prediction_single_file as slpsf
import sign_language_prediction_real_time as slprt
import onnxruntime as rt

sess = None

def run_csv_analysis():
    csv_path = filedialog.askopenfilename(
        title="Selecciona archivo CSV para análisis",
        filetypes=[("CSV files", "*.csv")]
    )
    if csv_path:
        analizar_csv(csv_path)


# Funciones para los botones
def make_new_csv():
    folder = filedialog.askdirectory(title="Selecciona la carpeta con las imágenes")
    if folder:
        slitc.process_images_from_folders(folder)
        messagebox.showinfo("Éxito", "Archivo CSV creado correctamente.")

def create_and_train_ANN():
    csv_path = filedialog.askopenfilename(title="Selecciona archivo CSV", filetypes=[("CSV files", "*.csv")])
    if csv_path:
        test_size = simple_input("¿Cuántos datos deseas usar para prueba? (ej. 0.2)")
        if test_size:
            mpt.create_and_train(csv_path, float(test_size))
            messagebox.showinfo("Éxito", "Modelo ANN creado.")

def load_ANN():
    global sess
    onnx_path = filedialog.askopenfilename(title="Selecciona archivo ONNX", filetypes=[("ONNX files", "*.onnx")])
    if onnx_path:
        sess = rt.InferenceSession(onnx_path)
        messagebox.showinfo("Modelo cargado", "Modelo ANN cargado exitosamente.")

def test_ANN():
    if sess is None:
        messagebox.showwarning("Error", "Debes cargar un modelo primero.")
        return
    path = filedialog.askdirectory(title="Selecciona carpeta de prueba")
    file = filedialog.askopenfilename(title="Selecciona imagen de prueba", filetypes=[("Imágenes", "*.png;*.jpg;*.jpeg")])
    if path and file:
        slitc.process_image_from_route(path, file.split("/")[-1])
        result = slpsf.predict_single_file(sess, './predict_landmarks.csv')
        messagebox.showinfo("Resultado", f"Predicción: {result}")

def test_ANN_real_time():
    if sess is None:
        messagebox.showwarning("Error", "Debes cargar un modelo primero.")
        return
    slprt.get_ANN(sess)

def simple_input(prompt):
    input_window = tk.Toplevel(root)
    input_window.title("Entrada")
    tk.Label(input_window, text=prompt).pack(padx=10, pady=10)
    entry = tk.Entry(input_window)
    entry.pack(pady=5)

    def submit():
        input_window.result = entry.get()
        input_window.destroy()

    tk.Button(input_window, text="Aceptar", command=submit).pack(pady=10)
    root.wait_window(input_window)
    return getattr(input_window, 'result', None)

# GUI principal
root = tk.Tk()
root.title("Traductor de Lenguaje de Señas")
root.geometry("400x400")

options = [
    ("Crear CSV desde imágenes", make_new_csv),
    ("Entrenar modelo ANN", create_and_train_ANN),
    ("Cargar modelo ANN", load_ANN),
    ("Probar modelo con imagen", test_ANN),
    ("Probar modelo en tiempo real", test_ANN_real_time),
    ("Análisis del CSV", run_csv_analysis)
]

for (text, func) in options:
    btn = tk.Button(root, text=text, command=func, width=40, pady=10)
    btn.pack(pady=5)

exit_btn = tk.Button(root, text="Salir", command=root.quit, fg="white", bg="red", width=40, pady=10)
exit_btn.pack(pady=20)

root.mainloop()