################## 1. IMPORTAR LIBRERÍAS ##########################
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import cv2  # para segmentación (Otsu, Canny, K-means)

################## 2. DEFINICIÓN DE LAS RUTAS #####################

# Ruta general donde está tu carpeta del dataset
Carpeta_General      = r"C:\Users\CECASIG\Downloads\archive (3)-20251210T192611Z-3-001\archive (3)"
Subcarpeta_NORMAL    = r"C:\TESIS\archive (3)-20251210T192611Z-3-001\archive (3)\datasets\normal"
Subcarpeta_CATARACT  = r"C:\TESIS\archive (3)-20251210T192611Z-3-001\archive (3)\datasets\cataract"

print("Ruta base:", Carpeta_General)
print("Carpetas usadas:")
print("  NORMAL   →", Subcarpeta_NORMAL)
print("  CATARACT →", Subcarpeta_CATARACT)

# Lista con rutas + etiqueta + nombre
# 0 = normal, 1 = cataract
RUTAS = [
    (Subcarpeta_NORMAL,   0, "normal"),
    (Subcarpeta_CATARACT, 1, "cataract")
]

############################################################
# 3. EXPLORACIÓN INICIAL + CARGA DE IMÁGENES
############################################################

tamaños  = []   # (ancho, alto) para explorar
clases   = []   # etiquetas 0/1 para explorar
formatos = []   # .png, .jpg, .jpeg
modos    = []   # RGB, L, etc.

# aquí guardaremos las imágenes para el modelo
X_list = []
y_list = []

print("\n=== EXPLORACIÓN INICIAL ===")

for ruta, etiqueta, nombre_clase in RUTAS:
    print(f"\nCarpeta: {ruta}  (clase: {nombre_clase})")

    if not os.path.isdir(ruta):
        print(f"⚠ ERROR: La carpeta NO existe: {ruta}")
        raise SystemExit

    for file in os.listdir(ruta):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            ruta_img = os.path.join(ruta, file)
            try:
                img = Image.open(ruta_img).convert("RGB")  # forzar RGB

                # ====== EXPLORACIÓN ======
                tamaños.append(img.size)        # (ancho, alto)
                clases.append(etiqueta)

                ext = os.path.splitext(file)[1].lower()
                formatos.append(ext)

                modos.append(img.mode)

                # ====== CARGA PARA MODELO ======
                X_list.append(np.array(img))
                y_list.append(etiqueta)

            except Exception as e:
                print("⚠ Imagen dañada ignorada:", ruta_img, "| Error:", e)

total_imgs = len(tamaños)
print("\nTotal de imágenes exploradas:", total_imgs)

# Aplanar las imágenes (convertir de (alto, ancho, 3) a un vector unidimensional)
X_flat = np.array(X_list).reshape(len(X_list), -1)  # (N, alto*ancho*3)

# Convertir las etiquetas a numpy array
y = np.array(y_list, dtype="int64")

# Mostrar forma de los datos antes de balanceo
print("Shape X (antes de balanceo):", X_flat.shape)
print("Shape y (antes de balanceo):", y.shape)

############################################################
# 4. APLICAR PCA PARA REDUCCIÓN DE DIMENSIONALIDAD
############################################################

# Reducir la dimensionalidad de las imágenes utilizando PCA
n_componentes = 50  # Puedes ajustar este número según sea necesario
pca = PCA(n_components=n_componentes)
X_pca = pca.fit_transform(X_flat)  # Reducir a 50 componentes principales

print("Shape X después de PCA:", X_pca.shape)

############################################################
# 5. APLICAR SMOTE PARA EL BALANCEO DE LAS CLASES
############################################################

print("\n=== BALANCEO DEL DATASET ===")

# Crear un objeto SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)

# Aplicar SMOTE a los datos reducidos con PCA
X_bal, y_bal = smote.fit_resample(X_pca, y)

# Verificar el balanceo de clases después de aplicar SMOTE
clases_bal, counts_bal = np.unique(y_bal, return_counts=True)
print("Después de aplicar SMOTE:", dict(zip(clases_bal, counts_bal)))

# A partir de aquí trabajamos con los datos balanceados
X = X_bal
y = y_bal

# Mostrar el balanceo de clases
clases_unicas, counts = np.unique(y, return_counts=True)
print(f"Distribución de clases después del balanceo: {dict(zip(clases_unicas, counts))}")

############################################################
# 6. LIMPIEZA DE DATOS (OPCIONAL) Y NORMALIZACIÓN
############################################################

# Normalización (píxeles entre 0 y 1)
X = X / np.max(X)  # Escalar entre 0 y 1

print(f"Shape de X después de la normalización: {X.shape}")

############################################################
# 7. DIVISIÓN TRAIN / TEST (EVITAR FUGA EN MODELOS)
############################################################

print("\n=== DIVISIÓN TRAIN / TEST ===")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("X_train:", X_train.shape, "| y_train:", y_train.shape)
print("X_test :", X_test.shape,  "| y_test :", y_test.shape)

############################################################
# 8. FUNCIONES DE SEGMENTACIÓN (si las necesitas)
############################################################
# Puedes agregar las funciones de segmentación (Otsu, Canny, K-means) aquí si las necesitas.
