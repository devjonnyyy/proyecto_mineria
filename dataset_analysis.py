import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analizar_csv(csv_path):
    # Cargar el archivo CSV
    df = pd.read_csv(csv_path)

    # Separar características y etiquetas
    X = df.iloc[:, :-1]  # Coordenadas
    y = df.iloc[:, -1]   # Etiquetas

    # --- Estadísticas básicas ---
    print("\n🔢 Número de muestras por clase:")
    print(y.value_counts())

    print("\n📊 Estadísticas descriptivas de coordenadas:")
    print(X.describe())

    if X.isnull().values.any():
        print("\n⚠️ Hay valores nulos en las coordenadas.")
    else:
        print("\n✅ No hay valores nulos.")

    # --- Visualizaciones ---
    # Histograma de la primera coordenada
    plt.figure(figsize=(10, 5))
    sns.histplot(X.iloc[:, 0], bins=30, kde=True)
    plt.title("Distribución de la coordenada X del primer landmark")
    plt.xlabel("Valor")
    plt.ylabel("Frecuencia")
    plt.grid(True)
    plt.show()

    # Conteo de clases
    plt.figure(figsize=(10, 5))
    y.value_counts().plot(kind='bar', color='skyblue')
    plt.title("Cantidad de muestras por clase")
    plt.xlabel("Clase")
    plt.ylabel("Cantidad")
    plt.grid(axis='y')
    plt.show()

    # Boxplot de las primeras 9 coordenadas
    plt.figure(figsize=(15, 5))
    sns.boxplot(data=X.iloc[:, :9])
    plt.title("Boxplot de las primeras coordenadas")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()
