import csv
import os
import pandas as pd
import numpy as np

def leer_csv(filename="esqueletos_de_los_perros.csv"):
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    df = pd.read_csv(filepath, header=None)
    data = df.iloc[:, :-1].values.tolist()
    etiquetas = df.iloc[:, -1].values.tolist()
    return data, etiquetas

def calcular_angulo(punto1, punto2, punto3, withers, tail_set):
    v1 = np.array([punto1[0] - punto2[0], punto1[1] - punto2[1]])
    v2 = np.array([punto3[0] - punto2[0], punto3[1] - punto2[1]])
    angulo_rad = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
    angulo_grados = np.degrees(angulo_rad) % 360
    if withers[0] > tail_set[0]:
        angulo_grados = 360 - angulo_grados
    return angulo_grados

def calcular_distancia(punto1, punto2):
    return np.linalg.norm(np.array(punto1) - np.array(punto2))

def generar_calculos_csv():
    frames, etiquetas = leer_csv()
    resultados = []

    for i, frame_data in enumerate(frames):
        bodyparts = [
            "Nose", "Withers", "Tail_Set", "Tail_Tip", "Right_Front_Elbow", "Right_Front_Wrist", "Right_Front_Paw",
            "Left_Front_Elbow", "Left_Front_Wrist", "Left_Front_Paw", "Right_Back_Elbow", "Right_Back_Wrist", "Right_Back_Paw",
            "Left_Back_Elbow", "Left_Back_Wrist", "Left_Back_Paw"
        ]
        coords = {}

        for j, part in enumerate(bodyparts):
            x = float(frame_data[3 * j])
            y = float(frame_data[3 * j + 1])
            confianza = float(frame_data[3 * j + 2])
            if confianza >= 0.75:
                coords[part] = (x, y)

        if "Withers" in coords and "Tail_Set" in coords:
            withers = coords["Withers"]
            tail_set = coords["Tail_Set"]
            longitud_corporal = calcular_distancia(withers, tail_set)
        else:
            continue

        angulos_resultados = []
        angulos = [
            ("Withers", "Tail_Set", "Tail_Tip"),
            ("Tail_Set", "Nose", "Withers"),
            ("Right_Front_Elbow", "Right_Front_Wrist", "Right_Front_Paw"),
            ("Left_Front_Elbow", "Left_Front_Wrist", "Left_Front_Paw"),
            ("Right_Back_Paw", "Right_Back_Wrist", "Right_Back_Elbow"),
            ("Left_Back_Paw", "Left_Back_Wrist", "Left_Back_Elbow")
        ]

        for p1, p2, p3 in angulos:
            if p1 in coords and p2 in coords and p3 in coords:
                angulo = calcular_angulo(coords[p1], coords[p2], coords[p3], withers, tail_set)
                angulos_resultados.append(angulo)

        distancias_resultados = []
        pares_distancia = [
            ("Left_Front_Paw", "Left_Back_Paw"),
            ("Right_Front_Paw", "Right_Back_Paw")
        ]

        for p1, p2 in pares_distancia:
            if p1 in coords and p2 in coords:
                distancia = calcular_distancia(coords[p1], coords[p2])
                distancia_normalizada = distancia / longitud_corporal
                distancias_resultados.append(distancia_normalizada)

        resultados.append(angulos_resultados + distancias_resultados + [etiquetas[i]])

    ruta_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "angulos_y_distancias_de_los_esqueletos_de_los_perros.csv")
    with open(ruta_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(resultados)

    print("Resultados guardados como angulos_y_distancias_de_los_esqueletos_de_los_perros.csv")

generar_calculos_csv()