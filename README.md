# 🐶 DeepLabCutWeb

**DeepLabCutWeb** es una **aplicación web no oficial** desarrollada para realizar **detección automática de posturas** en perros de una raza específica, utilizando como motor de inferencia el framework [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut). Este sistema permite identificar y clasificar tres poses fundamentales —agachado, cuadrúpedo de pie y erguido sobre dos patas— a partir de videos cargados o secuencias en vivo; la aplicación está pensada como una herramienta interactiva para la evaluación comportamental automatizada.

> Este proyecto **no forma parte del repositorio oficial de DeepLabCut**, pero se apoya directamente en su modelo de estimación de poses para animales.

---

## 🚀 Características

- 🧠 Inferencia basada en DeepLabCut con modelo entrenado en una única raza canina  
- 🐾 Clasificación automática de tres poses clave  
- 🖥️ Interfaz web accesible localmente o desde navegador remoto  
- 📊 Visualización inmediata del resultado sobre el video  
- 📁 Compatible con cargas locales o rutas personalizadas

---

## 🛠️ Instalación

### Requisitos

- Python 3.10  
- DeepLabCut (versión recomendada estable)  
- Streamlit o Flask para la interfaz web  
- OpenCV, NumPy, pandas y otras dependencias comunes

```bash
git clone https://github.com/tu-usuario/DeepLabCutWeb.git
cd DeepLabCutWeb
pip install -r requirements.txt
