

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
Carpeta_General      = r"C:\TESIS\archive (3)"
Subcarpeta_NORMAL    = r"C:\TESIS\archive (3)\datasets\normal"
Subcarpeta_CATARACT  = r"C:\TESIS\archive (3)\datasets\cataract"

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

############################################################
# 6. LIMPIEZA DE DATOS
############################################################

print("\n=== LIMPIEZA DE LOS DATOS ===")

# Corregir valores imposibles
X = np.clip(X, 0, 255)

# Asegurar tipo de dato homogéneo
X = X.astype("float32")

print("Rango después de limpieza:", X.min(), "a", X.max())
print("dtype de X:", X.dtype)

############################################################
# 7. NORMALIZACIÓN (0–1)
############################################################

print("\n=== NORMALIZACIÓN (0–1) ===")

X = X / 255.0    # ahora píxeles en rango [0,1]

print("Rango después de normalizar:", X.min(), "a", X.max())

############################################################
# 8. DIVISIÓN TRAIN / TEST (EVITAR FUGA EN MODELOS)
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
# 9. FUNCIONES DE SEGMENTACIÓN
############################################################

def segmentacion_otsu(img_rgb):
    """
    Segmentación por umbralización de Otsu en escala de grises.
    Retorna una máscara binaria (0/255).
    """
    # Asegurar uint8 0–255
    if img_rgb.max() <= 1.0:
        img_u8 = (img_rgb * 255).astype("uint8")
    else:
        img_u8 = img_rgb.astype("uint8")

    gris = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gris, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def segmentacion_canny(img_rgb, t1=50, t2=150):
    """
    Segmentación por detección de bordes Canny.
    Retorna imagen de bordes (0/255).
    """
    if img_rgb.max() <= 1.0:
        img_u8 = (img_rgb * 255).astype("uint8")
    else:
        img_u8 = img_rgb.astype("uint8")

    gris = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gris, t1, t2)
    return edges


def segmentacion_kmeans(img_rgb, k=2):
    """
    Segmentación por K-means en el espacio de color.
    Retorna:
      - imagen segmentada en color
      - etiquetas (máscara) con los clusters.
    """
    if img_rgb.max() <= 1.0:
        img_u8 = (img_rgb * 255).astype("uint8")
    else:
        img_u8 = img_rgb.astype("uint8")

    Z = img_u8.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                10, 1.0)
    _, labels, centers = cv2.kmeans(
        Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    segmented_img = res.reshape(img_u8.shape)
    labels_2d = labels.reshape(img_u8.shape[:2])

    return segmented_img, labels_2d

############################################################
# 10. DEMO: ORIGINAL VS 3 MÉTODOS DE SEGMENTACIÓN
############################################################

print("\n=== DEMO DE SEGMENTACIÓN (1 IMAGEN DE TRAIN) ===")

idx_demo = 0  # puedes cambiar este índice
img_demo = X_train[idx_demo]        # está en [0,1]

# Asegurar imagen RGB uint8 para mostrar
img_demo_rgb = (img_demo * 255).astype("uint8")

# Aplicar los 3 métodos
mask_otsu   = segmentacion_otsu(img_demo_rgb)
edges_canny = segmentacion_canny(img_demo_rgb)
seg_kmeans, labels_k = segmentacion_kmeans(img_demo_rgb, k=2)

# Mostrar resultados
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(img_demo_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(mask_otsu, cmap="gray")
plt.title("Segmentación Otsu")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(edges_canny, cmap="gray")
plt.title("Bordes Canny")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(seg_kmeans)
plt.title("Segmentación K-means (k=2)")
plt.axis("off")

plt.tight_layout()
plt.show()
############################################################
# 11. CONSTRUIR DATASETS SEGMENTADOS PARA CADA TÉCNICA
############################################################

def construir_dataset_segmentado(X, metodo):
    """
    Aplica una técnica de segmentación a TODO un conjunto de imágenes.
    X está en [0,1], forma (N, H, W, 3).
    Devuelve un nuevo array segmentado.
    """
    X_seg = []

    for i, img in enumerate(X):
        img_rgb = (img * 255).astype("uint8")  # volver a 0–255 para cv2

        if metodo == "otsu":
            mask = segmentacion_otsu(img_rgb)          # (H, W) 0/255
            # lo dejamos como imagen 2D (añadimos eje canal)
            X_seg.append(mask.astype("float32") / 255.0)

        elif metodo == "canny":
            edges = segmentacion_canny(img_rgb)        # (H, W) 0/255
            X_seg.append(edges.astype("float32") / 255.0)

        elif metodo == "kmeans":
            seg_img, labels = segmentacion_kmeans(img_rgb, k=2)  # (H, W, 3)
            X_seg.append(seg_img.astype("float32") / 255.0)

        else:
            raise ValueError("Método no reconocido:", metodo)

        # Mensaje opcional de progreso
        if (i + 1) % 500 == 0:
            print(f"  {metodo}: procesadas {i+1} imágenes...")

    return np.array(X_seg, dtype="float32")


print("\n=== CREANDO DATASETS SEGMENTADOS PARA TRAIN Y TEST ===")

metodos = ["otsu", "canny", "kmeans"]

X_train_seg = {}
X_test_seg  = {}

for m in metodos:
    print(f"\n>> Aplicando {m.upper()} en TRAIN...")
    X_train_seg[m] = construir_dataset_segmentado(X_train, m)
    print(f"   Shape X_train_{m}:", X_train_seg[m].shape)

    print(f">> Aplicando {m.upper()} en TEST...")
    X_test_seg[m] = construir_dataset_segmentado(X_test, m)
    print(f"   Shape X_test_{m}:", X_test_seg[m].shape)
############################################################
# 13. ENTRENAR MODELOS CNN CON CADA TÉCNICA DE SEGMENTACIÓN
############################################################

import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn(input_shape):
    """
    Construye una CNN sencilla para clasificación binaria.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        layers.MaxPool2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPool2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPool2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid")   # salida binaria
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def asegurar_canal(X):
    """
    Si X es (N, H, W) la convierte a (N, H, W, 1).
    Si ya tiene canal, no cambia nada.
    """
    if X.ndim == 3:
        X = np.expand_dims(X, axis=-1)
    return X


# print("\n PREPARANDO DATASETS PARA CNN ")

# # Dataset original (ya está en [0,1] y tiene forma (N, H, W, 3))
# X_train_orig = X_train
# X_test_orig  = X_test

# # Dataset OTSU y CANNY: son imágenes en gris (N, H, W) → añadimos canal
# X_train_otsu   = asegurar_canal(X_train_seg["otsu"])
# X_test_otsu    = asegurar_canal(X_test_seg["otsu"])

# X_train_canny  = asegurar_canal(X_train_seg["canny"])
# X_test_canny   = asegurar_canal(X_test_seg["canny"])

# # Dataset KMEANS: ya es RGB (N, H, W, 3)
# X_train_kmeans = X_train_seg["kmeans"]
# X_test_kmeans  = X_test_seg["kmeans"]

# # Diccionario con todos los conjuntos a probar
# datasets_cnn = {
#     "original": (X_train_orig,  X_test_orig),
#     "otsu":     (X_train_otsu,  X_test_otsu),
#     "canny":    (X_train_canny, X_test_canny),
#     "kmeans":   (X_train_kmeans, X_test_kmeans),
# }

# resultados_cnn = {}

# EPOCHS = 10      # puedes subir/bajar
# BATCH_SIZE = 32  # puedes ajustar según la RAM

# for nombre, (Xtr, Xte) in datasets_cnn.items():
#     print(f"\n=== ENTRENANDO CNN CON IMÁGENES {nombre.upper()} ===")
#     print("  Shape X_train:", Xtr.shape)

#     # Construir modelo para este tipo de entrada
#     modelo = build_cnn(input_shape=Xtr.shape[1:])

#     # Entrenamiento
#     history = modelo.fit(
#         Xtr, y_train,
#         epochs=EPOCHS,
#         batch_size=BATCH_SIZE,
#         validation_data=(Xte, y_test),
#         verbose=1
#     )

#     # Evaluación en test
#     loss, acc = modelo.evaluate(Xte, y_test, verbose=0)
#     resultados_cnn[nombre] = acc
#     print(f"  Accuracy en TEST ({nombre}): {acc:.4f}")

# print("\n RESUMEN ACCURACY CNN POR TÉCNICA ")
# for nombre, acc in resultados_cnn.items():
#     print(f"{nombre.upper():8s} -> Accuracy: {acc:.4f}")
