import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def leer_csv(filename="esqueletos_perros.csv"):
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    df = pd.read_csv(filepath, header=None)
    datos = df.iloc[:, :-1].values.tolist()
    clases = df.iloc[:, -1].astype(int).tolist()
    return datos, clases

def obtener_etiqueta(clase):
    etiquetas = {
        1: "Acostado",
        2: "Parado en Cuatro Patas",
        3: "Parado en Dos Patas"
    }
    return etiquetas.get(clase, "Desconocido")

def calcular_angulo(punto1, punto2, punto3, withers, tail_set):
    v1 = np.array([punto1[0] - punto2[0], punto1[1] - punto2[1]])
    v2 = np.array([punto3[0] - punto2[0], punto3[1] - punto2[1]])
    angulo_rad = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
    angulo_grados = np.degrees(angulo_rad) % 360
    if withers[0] > tail_set[0]:
        angulo_grados = 360 - angulo_grados
    return angulo_grados

def dibujar_arco(ax, punto_central, punto_inicio, punto_fin, angulo, radio=20, withers=None, tail_set=None):
    if withers is not None and tail_set is not None and withers[0] > tail_set[0]:
        punto_inicio, punto_fin = punto_fin, punto_inicio
    v1 = np.array([punto_inicio[0] - punto_central[0], punto_inicio[1] - punto_central[1]])
    v2 = np.array([punto_fin[0] - punto_central[0], punto_fin[1] - punto_central[1]])
    ax.plot([punto_central[0], punto_inicio[0]], [punto_central[1], punto_inicio[1]], color='blue', linewidth=2)
    ax.plot([punto_central[0], punto_fin[0]], [punto_central[1], punto_fin[1]], color='red', linewidth=2)
    angulo_inicio = np.arctan2(v1[1], v1[0])
    angulo_fin = angulo_inicio + np.radians(angulo)
    radio_ajustado = min(radio, np.linalg.norm(v1) * 0.5, np.linalg.norm(v2) * 0.5)
    theta = np.linspace(angulo_inicio, angulo_fin, 30)
    arco_x = punto_central[0] + radio_ajustado * np.cos(theta)
    arco_y = punto_central[1] + radio_ajustado * np.sin(theta)
    ax.plot(arco_x, arco_y, color='red', linewidth=2)
    text_x = punto_central[0] + (radio_ajustado + 10) * np.cos((angulo_inicio + angulo_fin) / 2)
    text_y = punto_central[1] + (radio_ajustado + 10) * np.sin((angulo_inicio + angulo_fin) / 2)
    ax.text(text_x, text_y, f"{angulo:.2f}°", color='red', fontsize=10, fontweight='bold',
            fontfamily='Times New Roman', fontstyle='italic')

def ajustar_limites(ax, coords):
    if coords:
        x_vals, y_vals = zip(*coords.values())
        min_x, max_x = min(x_vals), max(x_vals)
        min_y, max_y = min(y_vals), max(y_vals)
        centro_x = (min_x + max_x) / 2
        centro_y = (min_y + max_y) / 2
        lado = max(max_x - min_x, max_y - min_y) + 100
        ax.set_xlim(centro_x - lado / 2, centro_x + lado / 2)
        ax.set_ylim(centro_y - lado / 2, centro_y + lado / 2)

def graficar_esqueleto(frame_data, ax, mostrar_calculos=False):
    bodyparts = [
        "Nose", "Withers", "Tail_Set", "Tail_Tip", "Right_Front_Elbow", "Right_Front_Wrist", "Right_Front_Paw",
        "Left_Front_Elbow", "Left_Front_Wrist", "Left_Front_Paw", "Right_Back_Elbow", "Right_Back_Wrist", "Right_Back_Paw",
        "Left_Back_Elbow", "Left_Back_Wrist", "Left_Back_Paw"
    ]
    skeleton = [
        ("Nose", "Withers"), ("Withers", "Tail_Set"), ("Tail_Set", "Tail_Tip"),
        ("Withers", "Right_Front_Elbow"), ("Withers", "Left_Front_Elbow"),
        ("Tail_Set", "Right_Back_Elbow"), ("Tail_Set", "Left_Back_Elbow"),
        ("Right_Front_Elbow", "Right_Front_Wrist"), ("Right_Front_Wrist", "Right_Front_Paw"),
        ("Left_Front_Elbow", "Left_Front_Wrist"), ("Left_Front_Wrist", "Left_Front_Paw"),
        ("Right_Back_Elbow", "Right_Back_Wrist"), ("Right_Back_Wrist", "Right_Back_Paw"),
        ("Left_Back_Elbow", "Left_Back_Wrist"), ("Left_Back_Wrist", "Left_Back_Paw")
    ]
    coords = {}
    if len(frame_data) != 48:
        return
    for i, part in enumerate(bodyparts):
        x = float(frame_data[3 * i])
        y = float(frame_data[3 * i + 1])
        confianza = float(frame_data[3 * i + 2])
        if confianza >= 0.75:
            coords[part] = (x, y)
            ax.scatter(x, y, color='black', s=100, zorder=3)
            ax.text(x + 10, y, part, fontsize=8,
                    fontfamily='Times New Roman', fontstyle='italic', color='black')

    for part1, part2 in skeleton:
        if part1 in coords and part2 in coords:
            x1, y1 = coords[part1]
            x2, y2 = coords[part2]
            ax.plot([x1, x2], [y1, y2], color='gray', linewidth=2, zorder=1)

    if not mostrar_calculos:
        ajustar_limites(ax, coords)
        
        ax.set_xlabel("Coordenadas X", fontsize=12, fontfamily='Times New Roman', fontstyle='italic')
        ax.set_ylabel("Coordenadas Y", fontsize=12, fontfamily='Times New Roman', fontstyle='italic')
        ax.tick_params(axis='both', which='major', labelsize=10)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily('Times New Roman')
            label.set_fontstyle('italic')
        
        return

    if "Withers" not in coords or "Tail_Set" not in coords:
        return

    withers = coords["Withers"]
    tail_set = coords["Tail_Set"]

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
            if angulo > 0:
                dibujar_arco(ax, coords[p2], coords[p1], coords[p3], angulo, withers=withers, tail_set=tail_set)

    pares_distancia = [
        ("Left_Front_Paw", "Left_Back_Paw"),
        ("Right_Front_Paw", "Right_Back_Paw")
    ]

    for p1, p2 in pares_distancia:
        if p1 in coords and p2 in coords:
            x1, y1 = coords[p1]
            x2, y2 = coords[p2]
            distancia = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))
            ax.plot([x1, x2], [y1, y2], color='purple', linestyle='dashed', linewidth=2, zorder=2)
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            ax.text(mid_x, mid_y, f"{distancia:.2f}", color='purple', fontsize=10, fontweight='bold',
                    fontfamily='Times New Roman', fontstyle='italic')

    ajustar_limites(ax, coords)
    
    ax.set_xlabel("Coordenadas X", fontsize=12, fontfamily='Times New Roman', fontstyle='italic')
    ax.set_ylabel("Coordenadas Y", fontsize=12, fontfamily='Times New Roman', fontstyle='italic')
    ax.tick_params(axis='both', which='major', labelsize=10)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('Times New Roman')
        label.set_fontstyle('italic')

def generar_imagenes_individuales():
    frames, clases = leer_csv()
    carpeta_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")
    carpeta_sin = os.path.join(carpeta_base, "esqueletos_graficados_sin_calculos")
    carpeta_con = os.path.join(carpeta_base, "esqueletos_graficados_con_calculos")
    os.makedirs(carpeta_sin, exist_ok=True)
    os.makedirs(carpeta_con, exist_ok=True)

    for i, (frame, clase) in enumerate(zip(frames, clases)):
        etiqueta = obtener_etiqueta(clase)
        numero = f"{i + 1:03d}"
        for mostrar, carpeta in [(False, carpeta_sin), (True, carpeta_con)]:
            fig, ax = plt.subplots(figsize=(6, 6))
            graficar_esqueleto(frame, ax, mostrar_calculos=mostrar)
            ax.set_title(f"Fotograma {numero} - {etiqueta}",
                         fontsize=12, fontfamily='Times New Roman', fontstyle='italic')
            ax.invert_yaxis()
            ax.set_aspect('equal')
            nombre = f"fotograma_{numero}.png"
            ruta = os.path.join(carpeta, nombre)
            plt.savefig(ruta, bbox_inches='tight')
            plt.close(fig)

    print(f"Se generaron {len(frames) * 2} imágenes en 'img/esqueletos_graficados_sin_calculos' y 'img/esqueletos_graficados_con_calculos'")

generar_imagenes_individuales()
