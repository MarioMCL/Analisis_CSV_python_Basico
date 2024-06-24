import matplotlib.pyplot as plt
import pandas as pd

# Ruta al archivo CSV
ruta_csv = 'data/TAXI_Flota_corregido.csv'

# Intentar leer el archivo CSV con una codificación diferente
try:
    df = pd.read_csv(ruta_csv, delimiter=',', encoding='utf-8')
    # Puedes probar con otras codificaciones como 'latin1', 'utf-8', 'cp1252'

    # Imprimir la cabecera de la tabla
    print(df.head())

    # Asegúrate de que la columna 'Marca' existe en el DataFrame
    if 'Marca' in df.columns:
        # Contar las ocurrencias de cada marca
        marca_counts = df['Marca'].value_counts()

        # Crear la gráfica de barras
        plt.figure(figsize=(10, 6))
        marca_counts.plot(kind='bar', color='green')
        plt.title('Frecuencia de Marcas')
        plt.ylabel('Cantidad de Veces Repetida')
        plt.xticks(rotation=45)
        plt.show()

        # Imprimir en la consola la lista de marcas con la cantidad de veces que se repiten
        print("Lista de Marcas y Cantidad de Veces Repetidas:")
        for marca, count in marca_counts.items():
            print(f"{marca}: {count}")
        print()
    else:
        print("La columna 'Marca' no existe en el archivo CSV.")

    if 'Modelo' in df.columns:
        # Contar las ocurrencias de cada modelo
        modelo_counts = df['Modelo'].value_counts()

        # Crear la gráfica de barras
        plt.figure(figsize=(10, 6))
        modelo_counts.plot(kind='bar', color='blue')
        plt.title('Frecuencia de Modelos')
        plt.xlabel('Modelo')
        plt.ylabel('Cantidad de Veces Repetida')
        plt.xticks(rotation=80)
        plt.show()

        # Imprimir en la consola la lista de marcas con la cantidad de veces que se repiten
        print("Lista de Modelos y Cantidad de Veces Repetidas:")
        for modelo, count in modelo_counts.items():
            print(f"{modelo}: {count}")
    else:
        print("La columna 'Marca' no existe en el archivo CSV.")
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
