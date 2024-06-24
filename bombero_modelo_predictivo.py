import bomberos_funciones as bf

meses = {
    'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
    'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
}

grado = int(input('Ingrese la grado para el polinomio del modelo predictivo entre 2 y 10: '))
df_preview = bf.load_data()
df = bf.adapt_data(df_preview)
bf.predictive_model(df, grado)
