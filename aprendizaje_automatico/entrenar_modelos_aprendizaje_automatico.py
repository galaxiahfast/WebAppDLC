import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.linewidth': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'axes.edgecolor': 'gray',
    'grid.linestyle': '--',
    'grid.color': 'gray',
    'grid.alpha': 0.5,
    'figure.figsize': (15, 5),
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

dir_path = os.path.dirname(os.path.abspath(__file__))

ruta_csv = os.path.join(dir_path, "angulos_distancias_poses_perros.csv")
df = pd.read_csv(ruta_csv, header=None)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

models = [
    ('SVM', SVC(kernel='linear')),
    ('KNN', KNeighborsClassifier(n_neighbors=3)),
    ('Random Forest', RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=10, min_samples_leaf=5,
        max_features='sqrt', random_state=42
    )),
    ('Logistic Regression', LogisticRegression(max_iter=1000)),
    ('Decision Tree', DecisionTreeClassifier(max_depth=10, min_samples_split=10, min_samples_leaf=5)),
    ('Naive Bayes', GaussianNB())
]

accuracies_train = []
accuracies_val = []
errores_val = []

carpeta_modelos = os.path.join(dir_path, "modelos")
os.makedirs(carpeta_modelos, exist_ok=True)

for name, model in models:
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    model.fit(X_train, y_train)
    nombre_archivo = name.replace(' ', '_').lower() + ".pkl"
    ruta_modelo = os.path.join(carpeta_modelos, nombre_archivo)
    joblib.dump(model, ruta_modelo)
    acc_val = model.score(X_val, y_val)
    accuracies_train.append(np.mean(cv_scores))
    accuracies_val.append(acc_val)
    errores_val.append(1 - acc_val)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].barh([name for name, _ in models], accuracies_train, color=plt.cm.Greys(np.linspace(0.2, 0.8, len(models))))
axes[0].set_xlabel('Precisión en Entrenamiento')
axes[0].set_xlim(0, 1)
axes[0].set_title('Precisión Promedio en Entrenamiento')
for i, v in enumerate(accuracies_train):
    axes[0].text(v + 0.02, i, f'{v*100:.2f}%', va='center', ha='left', fontsize=10)
axes[1].barh([name for name, _ in models], accuracies_val, color=plt.cm.Greys(np.linspace(0.2, 0.8, len(models))))
axes[1].set_xlabel('Precisión en Validación')
axes[1].set_xlim(0, 1)
axes[1].set_title('Precisión en Validación')
for i, v in enumerate(accuracies_val):
    axes[1].text(v + 0.02, i, f'{v*100:.2f}%', va='center', ha='left', fontsize=10)
axes[2].barh([name for name, _ in models], errores_val, color=plt.cm.Greys(np.linspace(0.2, 0.8, len(models))))
axes[2].set_xlabel('Error en Validación')
axes[2].set_xlim(0, 1)
axes[2].set_title('Error Promedio en Validación')
for i, v in enumerate(errores_val):
    axes[2].text(v + 0.02, i, f'{v*100:.2f}%', va='center', ha='left', fontsize=10)

ruta_grafico = os.path.join(carpeta_modelos, "comparacion_modelos.png")
plt.tight_layout()
plt.savefig(ruta_grafico)
plt.close()

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, (name, model) in enumerate(models):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_train))
    ax = axes[i // 3, i % 3]
    disp.plot(ax=ax, cmap='Greys', values_format='d')
    ax.set_title(f'{name} - Matriz de Confusión')

plt.tight_layout()
ruta_matrices = os.path.join(carpeta_modelos, "matrices_confusion.png")
plt.savefig(ruta_matrices)
plt.close()

for name, model in models:
    i = [m[0] for m in models].index(name)
    print(f"Modelo: {name}")
    print(f"Precisión en entrenamiento: {accuracies_train[i]*100:.2f}%")
    print(f"Precisión en validación: {accuracies_val[i]*100:.2f}%")
    print(f"Diferencia (entrenamiento - validación): {(accuracies_train[i] - accuracies_val[i]) * 100:.2f}%")
    print('-' * 50)
