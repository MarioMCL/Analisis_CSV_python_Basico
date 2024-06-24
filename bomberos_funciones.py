import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np

meses = {
    'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
    'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
}


def load_data():
    df2024 = pd.read_csv('data/Bomberos_2024.csv', encoding='utf-8')
    df2023 = pd.read_csv('data/Bomberos_2023.csv', encoding='utf-8')
    df2022 = pd.read_csv('data/Bomberos_2022.csv', encoding='utf-8')
    df2021 = pd.read_csv('data/Bomberos_2021.csv', encoding='utf-8')
    df2020 = pd.read_csv('data/Bomberos_2020.csv', encoding='utf-8')
    df2019 = pd.read_csv('data/Bomberos_2019.csv', encoding='utf-8')
    df2018 = pd.read_csv('data/Bomberos_2018.csv', encoding='utf-8')

    # Concatenar por filas
    df = pd.concat([df2024, df2023, df2022, df2021, df2020, df2019, df2018], axis=0)

    # Eliminar espacios en blanco en los nombres de las columnas
    df.columns = df.columns.str.strip()

    # Reemplazar espacios con guiones bajos
    df.columns = df.columns.str.replace(' ', '_')

    return df


def adapt_data(df):
    # Convertir los nombres de los meses a números

    # Convertir la columna MES a string y luego a número, manejando valores nulos
    df['MES'] = df['MES'].astype(str).str.lower().map(meses)

    # Asegurar que las columnas numéricas estén en el tipo correcto
    columnas_numericas = ['FUEGOS', 'DAÑOS_EN_CONSTRUCCION', 'SALVAMENTOS_Y_RESCATES', 'DAÑOS_POR_AGUA',
                          'INCIDENTES_DIVERSOS', 'SALIDAS_SIN_INTERVENCION', 'SERVICIOS_VARIOS', 'TOTAL']
    df[columnas_numericas] = df[columnas_numericas].apply(pd.to_numeric, errors='coerce')

    # Eliminar filas con valores nulos en columnas de interés
    df.dropna(subset=['AÑO', 'MES', 'TOTAL'], inplace=True)
    resultado = df.groupby(['MES', 'AÑO'])['TOTAL'].sum().reset_index()
    return resultado


def predictive_model(df_final, grado):
    # Paso 2: Entrenar el modelo de regresión polinomial
    # Seleccionar las columnas relevantes para el modelo
    X = df_final[['MES']]
    y = df_final['TOTAL']

    # Transformar las características para incluir términos polinomiales
    poly = PolynomialFeatures(degree=grado)  # Puedes ajustar el grado del polinomio
    X_poly = poly.fit_transform(X)
    poly.get_feature_names_out(['MES'])

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

    # Entrenar el modelo
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluar el modelo
    r2_score = model.score(X_test, y_test)
    print("R^2:", r2_score)

    # Paso 3: Hacer la predicción para los meses de 2025
    meses_2025 = ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 'julio', 'agosto', 'septiembre', 'octubre',
                  'noviembre', 'diciembre']
    predicciones = []
    # Calcular las predicciones para cada mes
    for mes in meses_2025:
        mes_numero = meses[mes.lower()]
        X_pred = poly.transform([[mes_numero]])
        prediccion = model.predict(X_pred)
        predicciones.append(prediccion[0])

    # Imprimir las predicciones para cada mes de 2025
    for i, mes in enumerate(meses_2025):
        print(f"Predicción para {mes} de 2025: {predicciones[i]:.0f} intervenciones")

    # Generar una secuencia de valores para X que cubra el rango completo
    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)  # 100 puntos entre el mínimo y máximo

    # Transformar los valores de X_range para incluir términos polinomiales
    X_range_poly = poly.transform(X_range)

    # Predicciones para la secuencia completa de X_range
    y_pred_range = model.predict(X_range_poly)

    # Graficar los resultados con la curva suave del modelo polinomial
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='green', label='Datos Reales')
    plt.plot(X_range, y_pred_range, color='red', linewidth=2, label='Modelo Polinomial')
    plt.xlabel('Mes')
    plt.ylabel('Total de Intervenciones')
    plt.title(f'Predicción de Intervenciones del Cuerpo de Bomberos (R^2 = {r2_score:.2f})(grado = {grado:.0f})')
    plt.legend()
    plt.show()

    #Codigo para crear graficas extras, por añadir
    """
    # Paso 2: Obtener la lista de meses únicos y categorías de incidencias
    categorias_incidencias = ['FUEGOS', 'DAÑOS_EN_CONSTRUCCION', 'SALVAMENTOS_Y_RESCATES', 'DAÑOS_POR_AGUA',
                          'INCIDENTES_DIVERSOS', 'SALIDAS_SIN_INTERVENCION', 'SERVICIOS_VARIOS']
    meses_unicos = [1, 2, 3, 4, 5,]

    # Paso 3: Crear un array para almacenar las sumas por mes
    sumas_por_mes = np.zeros((len(meses_unicos), len(categorias_incidencias)), dtype=int)

    # Paso 4: Calcular las sumas por mes y tipo de incidencia
    for i, mes in enumerate(meses_unicos):
        for j, categoria in enumerate(categorias_incidencias):
            sumas_por_mes[i, j] = df[df['MES'] == mes][categoria].sum()

    # Paso 5: Mostrar el array de sumas por mes
    print("Array de Sumas por Mes:")
    print(sumas_por_mes)

    # Configurar el gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.12  # Ancho de cada barra
    gap = 0.3  # Espacio entre grupos de barras (meses)
    num_bars = len(categorias_incidencias)
    index = np.arange(len(meses_2025))  # Índices de los grupos de barras

    # Crear barras para cada tipo de incidencia
    for i in range(num_bars):
        ax.bar(index + i * bar_width, sumas_por_mes[:, i], bar_width, label=categorias_incidencias[i])

    # Etiquetas y título
    ax.set_xlabel('Mes')
    ax.set_ylabel('Cantidad')
    ax.set_title('Cantidad de Incidencias por Mes y Tipo')
    ax.set_xticks(index + (num_bars - 1) * bar_width / 2)
    ax.set_xticklabels(meses_2025)
    ax.legend()

    plt.tight_layout()
    plt.show()

    # Graficar los resultados
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='green', label='Datos Reales')
    plt.plot(X, model.predict(poly.transform(X)), color='red', linewidth=2, label='Modelo Polinomial')
    plt.xlabel('Mes')
    plt.ylabel('Total de Intervenciones')
    plt.title(f'Predicción de Intervenciones del Cuerpo de Bomberos  {r2_score:.2f}')
    plt.legend()
    plt.show()

    # Visualizar la distribución de tipos de incidencias por mes
    incidencias_por_mes = df.groupby('MES').sum()[columnas_numericas].transpose()

    plt.figure(figsize=(12, 8))
    sns.heatmap(incidencias_por_mes, annot=True, cmap='Blues', fmt='.0f')
    plt.title('Tipos de Incidencias por Mes')
    plt.xlabel('Número de Mes')
    plt.ylabel('Tipo de Incidencia')
    plt.show()
    """

    """# Configurar el gráfico
    fig, ax = plt.subplots(figsize=(10, 6))

    # Crear barras apiladas
    for i in range(len(categorias_incidencias)):
        if i == 0:
            ax.bar(meses_2025, sumas_por_mes[:, i], label=categorias_incidencias[i])
        else:
            ax.bar(meses_2025, sumas_por_mes[:, i], bottom=np.sum(sumas_por_mes[:, :i], axis=1), label=categorias_incidencias[i])

    # Etiquetas y título
    ax.set_xlabel('Mes')
    ax.set_ylabel('Cantidad')
    ax.set_title('Cantidad de Incidencias por Mes y Tipo')
    ax.legend()

    plt.tight_layout()
    plt.show()"""