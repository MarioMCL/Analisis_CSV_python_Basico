import matplotlib.pyplot as plt
import pandas as pd

# Ruta al archivo CSV
ruta_csv = 'data/AccidentesBicicletas_2024.csv'

# Intentar leer el archivo CSV
try:
    # Leer el CSV
    df = pd.read_csv(ruta_csv, delimiter=';')

    # Contar las ocurrencias segun valor
    alcohol_counts = df['positiva_alcohol'].value_counts()

    # Calcular el porcentaje respecto al total
    total_alcohol = alcohol_counts.sum()
    alcohol_percentages = (alcohol_counts / total_alcohol) * 100

    # Imprimir en la consola la lista con la cantidad de veces que se repiten y su porcentaje
    print("\nNº de veces positivos y porcentajes. S: positivos, N: negativos")
    for alcohol, count in alcohol_counts.items():
        porcentaje = alcohol_percentages[alcohol]
        print(f"{alcohol}: {count} veces ({porcentaje:.2f}%)")

    # Crear la gráfica de sectores circulares
    plt.figure(figsize=(10, 6))
    plt.pie(alcohol_percentages, labels=alcohol_percentages.index, autopct='%1.2f%%', startangle=140,
            colors=plt.cm.Paired.colors)
    plt.title('Porcentajes Positivos Alcohol')
    plt.xlabel('Positivos: S, Negativos: N')
    plt.xticks(rotation=90)
    plt.show()

    # Contar las ocurrencias de cada sexo
    sexo_counts = df['sexo'].value_counts()

    # Calcular el porcentaje respecto al total
    total_sexo = sexo_counts.sum()
    sexo_percentages = (sexo_counts / total_sexo) * 100

    # Imprimir en la consola la lista con la cantidad de veces que se repiten y su porcentaje
    print("\nNº de veces positivos y porcentajes. S: positivos, N: negativos")
    for sexo, count in sexo_counts.items():
        porcentaje = sexo_percentages[sexo]
        print(f"{sexo}: {count} veces ({porcentaje:.2f}%)")

    # Crear la gráfica de sectores circulares
    plt.figure(figsize=(10, 6))
    plt.pie(sexo_percentages, labels=sexo_percentages.index, autopct='%1.2f%%', startangle=140,
            colors=plt.cm.Paired.colors)
    plt.title('Distribucion segun sexo')
    plt.xticks(rotation=90)
    plt.show()

    # Contar las ocurrencias segun rango de edad
    edad_counts = df['rango_edad'].value_counts()

    # Crear la gráfica de barras
    plt.figure(figsize=(10, 6))
    edad_counts.plot(kind='bar', color='orange')
    plt.title('Clasificacion por edad')
    plt.xlabel('Rangos edad')
    plt.ylabel('Cantidad de Veces Repetida')
    plt.xticks(rotation=45)
    plt.show()

    # Contar las ocurrencias de cada rango de edad
    edad_counts = df['rango_edad'].value_counts()

    # Calcular el porcentaje respecto al total
    total_edad = edad_counts.sum()
    edad_percentages = (edad_counts / total_edad) * 100

    # Crear la gráfica de sectores circulares
    plt.figure(figsize=(10, 6))
    plt.pie(edad_percentages, labels=edad_percentages.index, autopct='%1.2f%%', startangle=140,
            colors=plt.cm.Paired.colors)
    plt.title('Distribucion segun edad')
    plt.xticks(rotation=90)
    plt.show()

    # Imprimir en la consola la lista con la cantidad de veces que se repiten y su porcentaje
    print("\nNº de veces positivos y porcentajes. S: positivos, N: negativos")
    for edad, count in edad_counts.items():
        porcentaje = edad_percentages[edad]
        print(f"{edad}: {count} veces ({porcentaje:.2f}%)")

    # Contar las ocurrencias de cada rango de edad
    accidentes_counts = df['tipo_accidente'].value_counts()

    # Calcular el porcentaje respecto al total
    total_accidentes = accidentes_counts.sum()
    accidentes_percentages = (accidentes_counts / total_accidentes) * 100

    # Crear la gráfica de sectores circulares
    plt.figure(figsize=(10, 6))
    plt.pie(accidentes_percentages, labels=accidentes_percentages.index, autopct='%1.2f%%', startangle=140,
            colors=plt.cm.Paired.colors)
    plt.title('Distribucion segun tipo de accidente')
    plt.xticks(rotation=90)
    plt.show()

    # Imprimir en la consola la lista con la cantidad de veces que se repiten y su porcentaje
    print("\nNº de veces segun tipo de accidentes")
    for accidente, count in accidentes_counts.items():
        porcentaje = accidentes_percentages[accidente]
        print(f"{accidente}: {count} veces ({porcentaje:.2f}%)")



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
