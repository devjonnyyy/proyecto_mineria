import tkinter as tk
from tkinter import ttk
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
    global sess, resultado_label

    if sess is None:
        messagebox.showwarning("Error", "Debes cargar un modelo primero.")
        return

    path = filedialog.askdirectory(title="Selecciona carpeta de prueba")
    file = filedialog.askopenfilename(title="Selecciona imagen de prueba", filetypes=[("Imágenes", "*.png;*.jpg;*.jpeg")])
    if path and file:
        slitc.process_image_from_route(path, file.split("/")[-1])
        result = slpsf.predict_single_file(sess, './predict_landmarks.csv')

        # Mostrar en un cuadro de diálogo
        messagebox.showinfo("Resultado de la predicción", f"📷 Se detectó la letra: {result}")


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
root.title("Sistema de Traducción de Lengua de Señas")
root.geometry("500x600")

style = ttk.Style()
style.configure("TButton", padding=10, font=("Segoe UI", 10))

# --- Sección de datos ---
frame_datos = ttk.LabelFrame(root, text="📂 Datos", padding=20)
frame_datos.pack(fill="both", padx=10, pady=10)

ttk.Button(frame_datos, text="Crear CSV desde imágenes", command=make_new_csv).pack(pady=5, fill='x')
ttk.Button(frame_datos, text="Análisis del CSV", command=run_csv_analysis).pack(pady=5, fill='x')

# --- Sección de modelo ---
frame_modelo = ttk.LabelFrame(root, text="🤖 Modelo ANN", padding=20)
frame_modelo.pack(fill="both", padx=10, pady=10)

ttk.Button(frame_modelo, text="Entrenar modelo", command=create_and_train_ANN).pack(pady=5, fill='x')
ttk.Button(frame_modelo, text="Cargar modelo", command=load_ANN).pack(pady=5, fill='x')
ttk.Button(frame_modelo, text="Probar con imagen", command=test_ANN).pack(pady=5, fill='x')
ttk.Button(frame_modelo, text="Probar en tiempo real", command=test_ANN_real_time).pack(pady=5, fill='x')

# Botón de salida
tk.Button(root, text="Salir", command=root.quit, bg="red", fg="white", font=("Segoe UI", 10, "bold")).pack(pady=20, fill='x')

# Área para mostrar resultados
resultado_label = ttk.Label(root, text="", wraplength=450, justify="left", foreground="blue")
resultado_label.pack(pady=10, padx=10)

root.mainloop()