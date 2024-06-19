import matplotlib.pyplot as plt
import pandas as pd

# Ruta al archivo CSV
ruta_csv = 'AccidentesBicicletas_2024.csv'

# Intentar leer el archivo CSV con una codificación diferente
try:
    # Leer el CSV
    df = pd.read_csv(ruta_csv, delimiter=';')

    # Contar las ocurrencias de cada marca
    alcohol_counts = df['positiva_alcohol'].value_counts()

    # Calcular el porcentaje respecto al total
    total = alcohol_counts.sum()
    marca_percentages = (alcohol_counts / total) * 100

    # Imprimir en la consola la lista de marcas con la cantidad de veces que se repiten y su porcentaje
    print("\nNº de veces positivos y porcentajes. S: positivos, N: negativos")
    for alcohol, count in alcohol_counts.items():
        porcentaje = marca_percentages[alcohol]
        print(f"{alcohol}: {count} veces ({porcentaje:.2f}%)")

    # Crear la gráfica de barras
    plt.figure(figsize=(10, 6))
    plt.pie(marca_percentages, labels=marca_percentages.index, autopct='%1.2f%%', startangle=140,
            colors=plt.cm.Paired.colors)
    plt.title('Cantidad Positivos Alcohol')
    plt.xlabel('Positivos: S, Negativos: N')
    plt.xticks(rotation=90)
    plt.show()

except FileNotFoundError:
    print(f"El archivo en la ruta {ruta_csv} no se encontró.")
except pd.errors.EmptyDataError:
    print("El archivo CSV está vacío.")
except pd.errors.ParserError:
    print("Error al analizar el archivo CSV.")
except UnicodeDecodeError as e:
    print(f"Error de decodificación: {e}")
except Exception as e:
    print(f"Ocurrió un error: {e}")
