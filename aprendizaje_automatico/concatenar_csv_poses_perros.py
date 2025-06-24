import pandas as pd
import glob
import os

# LEER TODOS LOS CSV EN LA CARPETA DEL SCRIPT Y SUBCARPETAS, CONCATENARLOS OMITIENDO TRES FILAS Y COLUMNAS, AGREGAR UNA COLUMNA DE PROBABILIDAD Y GUARDAR EL RESULTADO.
def concatenar_csv_con_probabilidad():
    carpeta_csv = os.path.dirname(os.path.abspath(__file__))
    csv_files = glob.glob(os.path.join(carpeta_csv, '**', '*.csv'), recursive=True)
    data_frames = []
    for file in csv_files:
        df = pd.read_csv(file, header=None, skiprows=3)
        df = df.iloc[:, 3:]
        data_frames.append(df)
    concatenated_df = pd.concat(data_frames, axis=0, ignore_index=True)
    columns_list = []
    for i in range(concatenated_df.shape[1]):
        columns_list.append(concatenated_df.iloc[:, i])
        if (i + 1) % 2 == 0:
            columns_list.append(pd.Series([0.9] * concatenated_df.shape[0]))
    final_df = pd.concat(columns_list, axis=1)
    salida = os.path.join(carpeta_csv, 'poses_perros.csv')
    final_df.to_csv(salida, index=False, header=False)

# LLAMAR A LA FUNCIÓN PRINCIPAL PARA CONCATENAR LOS CSV.
concatenar_csv_con_probabilidad()
