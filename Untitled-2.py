
################## 1. IMPORTAR LIBRERÍAS ##########################
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
import cv2   # para segmentación (Otsu, Canny, K-means)S

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
# 0 = normal, 1 = catarata
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
                # Si quisieras forzar tamaño fijo, descomenta:
                # if img.size != (224, 224):
                #     img = img.resize((224, 224))

                X_list.append(np.array(img))
                y_list.append(etiqueta)

            except Exception as e:
                print("⚠ Imagen dañada ignorada:", ruta_img, "| Error:", e)

total_imgs = len(tamaños)
print("\nTotal de imágenes exploradas:", total_imgs)

if total_imgs > 0:
    anchos = np.array([w for (w, h) in tamaños])
    altos  = np.array([h for (w, h) in tamaños])

    print("\nDimensiones originales (ANCHO):")
    print("  mín   :", anchos.min())
    print("  máx   :", anchos.max())
    print("  media :", anchos.mean())

    print("\nDimensiones originales (ALTO):")
    print("  mín   :", altos.min())
    print("  máx   :", altos.max())
    print("  media :", altos.mean())

# Proporción de clases
conteo_clases = Counter(clases)
print("\nProporción de clases:")
for etiqueta, n in conteo_clases.items():
    nombre = "normal" if etiqueta == 0 else "cataract"
    porcentaje = n / total_imgs * 100 if total_imgs > 0 else 0
    print(f"  Clase {etiqueta} ({nombre}): {n} imágenes ({porcentaje:.2f}%)")

# Tipos de archivo
print("\nTipos de archivo encontrados (extensiones):")
print(Counter(formatos))

# Modos de imagen
print("\nModos de imagen encontrados:")
print(Counter(modos))

print("\n=== FIN DE LA EXPLORACIÓN INICIAL ===\n")

############################################################
# 4. CONVERSIÓN A ARRAYS X, y (DATOS CRUDOS)
############################################################

X = np.array(X_list, dtype="float32")   # (N, alto, ancho, 3)
y = np.array(y_list, dtype="int64")     # (N,)

print("Shape X (antes de balanceo):", X.shape)
print("Shape y (antes de balanceo):", y.shape)

############################################################
# 5. BALANCEO DE CLASES (OVERSAMPLING)
############################################################

print("\n=== BALANCEO DEL DATASET ===")

clases_unicas, counts = np.unique(y, return_counts=True)
print("Antes del balanceo:", dict(zip(clases_unicas, counts)))

max_count = counts.max()

X_bal = []
y_bal = []

for c in clases_unicas:
    idx = np.where(y == c)[0]                         # índices de esa clase
    idx_sel = np.random.choice(idx, max_count,        # oversampling
                               replace=True)
    X_bal.append(X[idx_sel])
    y_bal.append(y[idx_sel])

X_bal = np.concatenate(X_bal, axis=0)
y_bal = np.concatenate(y_bal, axis=0)

clases_bal, counts_bal = np.unique(y_bal, return_counts=True)
print("Después del balanceo:", dict(zip(clases_bal, counts_bal)))

# A partir de aquí trabajamos con los datos balanceados
X = X_bal
y = y_bal

def gini_index(y):
    clases_unicas, counts = np.unique(y, return_counts=True)
    proporciones = counts / len(y)
    gini = 1 - np.sum(proporciones ** 2)
    return gini

# Antes del balanceo
gini_before = gini_index(y)
print(f"Índice de Gini antes del balanceo: {gini_before:.4f}")

# Después del balanceo
gini_after = gini_index(y_bal)
print(f"Índice de Gini después del balanceo: {gini_after:.4f}")
