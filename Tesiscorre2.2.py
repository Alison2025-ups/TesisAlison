###################################################################
# TESIS-DESEMPEÑO Y ANÁLISIS DE CARACTERÍSTICAS EN LA DETECCIÓN DE CATARATAS MEDIANTE MODELOS DE INTELIGENCIA ARTIFICIAL
# Alison Real
# INteligencia Artificial
# ################ 1. IMPORTAR LIBRERÍAS ##########################
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
import cv2

################## 2. DEFINICIÓN DE LAS RUTAS #####################
Carpeta_General      = r"C:\TESIS\archive (3)"
Subcarpeta_NORMAL    = r"C:\TESIS\archive (3)\datasets\normal"
Subcarpeta_CATARACT  = r"C:\TESIS\archive (3)\datasets\cataract"

RUTAS = [
    (Subcarpeta_NORMAL,   0, "normal"),
    (Subcarpeta_CATARACT, 1, "cataract")
]

label_map = {etq: name for _, etq, name in RUTAS}

###################################################################
# 3) EXPLORACIÓN DE DATOS
###################################################################

records = []  # aquí guardamos solo métricas/metadata (no alteramos imágenes)
errores = 0

def safe_open_image(path):
    """Abre imagen sin transformarla; devuelve PIL.Image o None si falla."""
    try:
        img = Image.open(path)
        img.load()
        return img
    except Exception:
        return None

for carpeta, etiqueta, nombre in RUTAS:
    print(f"\nLeyendo carpeta: {carpeta} | clase: {nombre}")

    if not os.path.isdir(carpeta):
        raise FileNotFoundError(f"No existe la carpeta: {carpeta}")

    for file in os.listdir(carpeta):
        if not file.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        path = os.path.join(carpeta, file)
        img_pil = safe_open_image(path)
        if img_pil is None:
            errores += 1
            continue

        mode = img_pil.mode               # 'RGB', 'RGBA', 'L', etc.
        w, h = img_pil.size
        ext = os.path.splitext(file)[1].lower()

        # Convertimos SOLO para medir (no guardamos como dataset procesado):
        img_np = np.array(img_pil)        # puede ser 2D (L) o 3D (RGB/RGBA)
        # Intensidad/luminancia para métricas de nitidez/entropía:
        if img_np.ndim == 2:
            lum = img_np.astype(np.float32)
            ch = 1
        else:
            ch = img_np.shape[2]
            # si es RGBA, ignoramos alfa para luminancia
            rgb = img_np[..., :3].astype(np.float32)
            lum = (0.299*rgb[...,0] + 0.587*rgb[...,1] + 0.114*rgb[...,2]).astype(np.float32)


        # Métricas de características
        # 3.1) Brillo y contraste robusto (percentiles)
        mean_lum = float(np.mean(lum))
        std_lum  = float(np.std(lum))
        p5, p50, p95 = [float(x) for x in np.percentile(lum, [5, 50, 95])]
        contr_rob = p95 - p5

        # 3.2) Nitidez (varianza del Laplaciano): catarata suele bajar nitidez
        lum32 = np.ascontiguousarray(lum, dtype=np.float32)
        lap = cv2.Laplacian(lum32, cv2.CV_32F)

        blur_var = float(lap.var())

        # 3.3) Entropía: catarata suele reducir detalle/variabilidad
        ent = float(shannon_entropy(lum))

        # 3.4) Saturación (solo si es RGB/RGBA): útil para ver variación por cámara/iluminación
        if img_np.ndim == 3 and img_np.shape[2] >= 3:
            rgb8 = img_np[..., :3].astype(np.uint8)
            hsv = cv2.cvtColor(rgb8, cv2.COLOR_RGB2HSV)
            sat_mean = float(np.mean(hsv[...,1]))
        else:
            sat_mean = np.nan

        records.append({
            "file": file,
            "path": path,
            "class": etiqueta,
            "class_name": nombre,
            "ext": ext,
            "mode": mode,
            "width": w,
            "height": h,
            "channels": ch,
            "mean_lum": mean_lum,
            "std_lum": std_lum,
            "p5": p5,
            "p50": p50,
            "p95": p95,
            "contrast_p95_p5": contr_rob,
            "blur_var_laplacian": blur_var,
            "entropy": ent,
            "sat_mean": sat_mean
        })

print("\nTotal imágenes analizadas:", len(records))
print("Imágenes con error (no leídas):", errores)

if len(records) == 0:
    raise SystemExit("No se cargaron imágenes. Revisa rutas/extensiones.")

###################################################################
# 3.5) RESUMEN PROFUNDO DEL DATASET (ESTRUCTURA + CALIDAD)
###################################################################
# Distribución por clase
clases = [r["class"] for r in records]
conteo = Counter(clases)
print("\n--- Distribución de clases ---")
for k, v in conteo.items():
    print(f"{label_map[k]:8s}: {v}")

plt.figure()
plt.bar([label_map[k] for k in conteo.keys()], list(conteo.values()))
plt.title("Distribución de clases")
plt.xlabel("Clase")
plt.ylabel("Cantidad")
plt.show()

# Modos, extensiones, canales
print("\n--- Modos de imagen ---")
print(Counter([r["mode"] for r in records]))
print("\n--- Extensiones ---")
print(Counter([r["ext"] for r in records]))
print("\n--- Canales ---")
print(Counter([r["channels"] for r in records]))

# Tamaños (ancho/alto)
widths  = np.array([r["width"] for r in records], dtype=np.int32)
heights = np.array([r["height"] for r in records], dtype=np.int32)
print("\n--- Tamaños (ORIGINALES) ---")
print("Ancho  -> min:", widths.min(),  "max:", widths.max(),  "mean:", round(widths.mean(),2))
print("Alto   -> min:", heights.min(), "max:", heights.max(), "mean:", round(heights.mean(),2))

plt.figure()
plt.hist(widths, bins=30, alpha=0.6, label="Ancho")
plt.hist(heights, bins=30, alpha=0.6, label="Alto")
plt.title("Distribución de dimensiones originales")
plt.xlabel("Pixeles")
plt.ylabel("Frecuencia")
plt.legend()
plt.show()

###################################################################
# 3.6) ANÁLISIS DE CARACTERÍSTICAS POR CLASE (PROFUNDO PARA TU TEMA)
###################################################################
def resumen_metrica_por_clase(key):
    for c in sorted(set(clases)):
        vals = np.array([r[key] for r in records if r["class"] == c], dtype=np.float64)
        print(f"{key:20s} | {label_map[c]:8s} -> mean={vals.mean():.3f} std={vals.std():.3f} "
              f"min={vals.min():.3f} max={vals.max():.3f}")

print("\n================= MÉTRICAS (POR CLASE) =================")
for k in ["mean_lum","std_lum","contrast_p95_p5","blur_var_laplacian","entropy"]:
    resumen_metrica_por_clase(k)

# Boxplots (comparación directa normal vs cataract)
def boxplot_por_clase(key, title):
    data = []
    labels = []
    for c in sorted(set(clases)):
        data.append([r[key] for r in records if r["class"] == c])
        labels.append(label_map[c])
    plt.figure()
    plt.boxplot(data, labels=labels, showfliers=True)
    plt.title(title)
    plt.ylabel(key)
    plt.show()

print("\n================= BOXPLOTS COMPARATIVOS =================")
boxplot_por_clase("contrast_p95_p5", "Contraste robusto (p95 - p5) por clase")
boxplot_por_clase("blur_var_laplacian", "Nitidez (Var. Laplaciano) por clase")
boxplot_por_clase("entropy", "Entropía por clase")
boxplot_por_clase("mean_lum", "Brillo medio (luminancia) por clase")

# Histogramas por clase (para ver desplazamientos por catarata)
def hist_por_clase(key, bins=40):
    plt.figure()
    for c in sorted(set(clases)):
        vals = np.array([r[key] for r in records if r["class"] == c], dtype=np.float64)
        plt.hist(vals, bins=bins, alpha=0.5, density=True, label=label_map[c])
    plt.title(f"Histograma (densidad) de {key} por clase")
    plt.xlabel(key)
    plt.ylabel("Densidad")
    plt.legend()
    plt.show()

print("\n================= HISTOGRAMAS POR CLASE =================")
hist_por_clase("contrast_p95_p5")
hist_por_clase("blur_var_laplacian")
hist_por_clase("entropy")

###################################################################
# 3.6) DETECCIÓN DE OUTLIERS DE CALIDAD (SIN PREPROCESAR)
###################################################################
print("\n================= OUTLIERS / CALIDAD =================")

blur_vals = np.array([r["blur_var_laplacian"] for r in records], dtype=np.float64)
mean_vals = np.array([r["mean_lum"] for r in records], dtype=np.float64)
contr_vals = np.array([r["contrast_p95_p5"] for r in records], dtype=np.float64)

# Umbrales robustos por percentiles
thr_blur_low  = np.percentile(blur_vals, 10)   # más borrosas
thr_dark      = np.percentile(mean_vals, 5)    # más oscuras
thr_bright    = np.percentile(mean_vals, 95)   # más claras
thr_contr_low = np.percentile(contr_vals, 10)  # bajo contraste

idx_blur  = [i for i,r in enumerate(records) if r["blur_var_laplacian"] <= thr_blur_low]
idx_dark  = [i for i,r in enumerate(records) if r["mean_lum"] <= thr_dark]
idx_bright= [i for i,r in enumerate(records) if r["mean_lum"] >= thr_bright]
idx_contr = [i for i,r in enumerate(records) if r["contrast_p95_p5"] <= thr_contr_low]

print(f"Umbral blur bajo (p10): {thr_blur_low:.3f} -> {len(idx_blur)} imágenes")
print(f"Umbral oscuras (p5):    {thr_dark:.3f} -> {len(idx_dark)} imágenes")
print(f"Umbral claras (p95):    {thr_bright:.3f} -> {len(idx_bright)} imágenes")
print(f"Umbral bajo contraste (p10): {thr_contr_low:.3f} -> {len(idx_contr)} imágenes")

def mostrar_ejemplos(indices, titulo, n=12):
    if len(indices) == 0:
        print("No hay ejemplos para:", titulo)
        return
    pick = np.random.choice(indices, size=min(n, len(indices)), replace=False)
    cols = 6
    rows = int(np.ceil(len(pick)/cols))
    plt.figure(figsize=(14, 3*rows))
    for k, idx in enumerate(pick, 1):
        r = records[idx]
        img = safe_open_image(r["path"])
        if img is None:
            continue
        plt.subplot(rows, cols, k)
        plt.imshow(img)  # mostramos original
        plt.axis("off")
        plt.title(f"{r['class_name']}\n{os.path.basename(r['file'])}", fontsize=8)
    plt.suptitle(titulo, y=0.98, fontsize=14)
    plt.show()

mostrar_ejemplos(idx_blur, "Ejemplos: imágenes con baja nitidez (candidatas a borrosas)")
mostrar_ejemplos(idx_dark, "Ejemplos: imágenes muy oscuras")
mostrar_ejemplos(idx_bright, "Ejemplos: imágenes muy claras/sobreexpuestas")
mostrar_ejemplos(idx_contr, "Ejemplos: imágenes de bajo contraste")

###################################################################
# 3.7) MUESTRAS ALEATORIAS POR CLASE (ORIGINALES)
###################################################################
print("\n================= MUESTRAS ALEATORIAS (ORIGINALES) =================")

def mostrar_muestras_por_clase(n=8):
    for c in sorted(set(clases)):
        idxs = [i for i,r in enumerate(records) if r["class"] == c]
        pick = np.random.choice(idxs, size=min(n, len(idxs)), replace=False)
        cols = 4
        rows = int(np.ceil(len(pick)/cols))
        plt.figure(figsize=(12, 3*rows))
        for k, idx in enumerate(pick, 1):
            r = records[idx]
            img = safe_open_image(r["path"])
            if img is None:
                continue
            plt.subplot(rows, cols, k)
            plt.imshow(img)
            plt.axis("off")
            plt.title(r["class_name"])
        plt.suptitle(f"Muestras originales - {label_map[c]}", y=0.98, fontsize=14)
        plt.show()

mostrar_muestras_por_clase(n=8)

print("\n================= FIN EXPLORACIÓN PROFUNDA (SIN PREPROCESAMIENTO) =================")

###################################################################
# REALIZACIÓN DE MATRÍZ X Y 
###################################################################
IMG_SIZE = (224, 224)

X_list, y_list = [], []

for carpeta, etiqueta, nombre in RUTAS:
    for file in os.listdir(carpeta):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(carpeta, file)
            img = safe_open_image(path)
            if img is None:
                continue

            img = img.convert("RGB").resize(IMG_SIZE)
            arr = np.array(img, dtype=np.uint8)   # (224,224,3)

            X_list.append(arr)
            y_list.append(etiqueta)

X = np.array(X_list, dtype=np.uint8)   # (N,224,224,3)
y = np.array(y_list, dtype=np.int64)   # (N,)

print("X:", X.shape, X.dtype)
print("y:", Counter(y))

###################################################################
# LIMPIEZA DE DATOS
###################################################################
print("\n=== LIMPIEZA DE LOS DATOS ===")

# Revisar NaN / Inf en X (NO en X_norm)
if np.isnan(X).any() or np.isinf(X).any():
    print("Se detectaron NaN/Inf. Se reemplazarán por valores válidos.")
    X = np.nan_to_num(X, nan=0.0, posinf=255.0, neginf=0.0)


# Rango válido para imágenes 8-bit
X = np.clip(X, 0, 255)

# Tipo homogéneo
X = X.astype(np.float32)

print("Rango después de limpieza:", float(X.min()), "a", float(X.max()))
print("dtype de X:", X.dtype)
print("shape de X:", X.shape)

############################################################
# NORMALIZACIÓN y ESTANDARIZACIÓN
############################################################

print("\nNORMALIZACIÓN [0,1]")

# Asegurar tipo
X = X.astype(np.float32)

# Normalización estándar para imágenes
X_norm = X / 255.0

print("Rango después de normalizar:", X_norm.min(), "a", X_norm.max())
print("dtype:", X_norm.dtype)
print("shape:", X_norm.shape)
############################################################

print("\nESTANDARIZACIÓN Z-SCORE")

# Calcular media y desviación sobre el TRAIN (recomendado)
mean = np.mean(X, axis=(0,1,2), keepdims=True)
std  = np.std(X, axis=(0,1,2), keepdims=True) + 1e-8  # evitar división por cero

X_std = (X - mean) / std

print("Media aproximada:", np.mean(X_std))
print("Std aproximada:", np.std(X_std))
print("dtype:", X_std.dtype)
print("shape:", X_std.shape)


###################################################################
# 4) PROCESAMIENTO DE IMÁGENES
###################################################################

print("\n================ PROCESAMIENTO DE IMÁGENES ================")

# ------------------------------------------------------------
# X está en (N, H, W, 3) uint8 [0,255]
# ------------------------------------------------------------
print("X original:", X.shape, X.dtype, X.min(), X.max())

###################################################################
# 4.1 CONVERSIÓN A ESCALA DE GRISES (uint8) – SOLO UNA VEZ
###################################################################
X_gray_u8 = (
    0.299 * X[..., 0] +
    0.587 * X[..., 1] +
    0.114 * X[..., 2]
).astype(np.uint8)

print("X_gray_u8:", X_gray_u8.shape, X_gray_u8.dtype,
      "rango:", X_gray_u8.min(), X_gray_u8.max())

###################################################################
# 4.2 NORMALIZACIÓN PARA CNN (float32 0–1)
###################################################################
X_gray_norm = X_gray_u8.astype(np.float32) / 255.0

print("X_gray_norm:", X_gray_norm.shape, X_gray_norm.dtype,
      "rango:", X_gray_norm.min(), X_gray_norm.max())

###################################################################
# 4.3 AGREGAR CANAL PARA CNN  (N,H,W,1)
###################################################################
X_cnn = X_gray_norm[..., np.newaxis]

print("X_cnn (CNN input):", X_cnn.shape, X_cnn.dtype)

####################################################################
####################################################################
# TESIS – DETECCIÓN DE CATARATAS CON CNN + EARLY STOPPING
####################################################################

# ==========================
# IMPORTACIONES
# ==========================
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

from rich.console import Console
from rich.table import Table
from rich import box

# ==========================
# FUNCIÓN: OVERSAMPLING
# ==========================
def oversample_train(Xtr, ytr, seed=42):
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(ytr, return_counts=True)
    max_count = counts.max()

    X_out, y_out = [], []
    for c in classes:
        idx = np.where(ytr == c)[0]
        sel = rng.choice(idx, size=max_count, replace=True)
        X_out.append(Xtr[sel])
        y_out.append(ytr[sel])

    Xb = np.concatenate(X_out)
    yb = np.concatenate(y_out)
    perm = rng.permutation(len(yb))
    return Xb[perm], yb[perm]

# ==========================
# FUNCIÓN: GINI
# ==========================
def gini_index(y_in):
    _, counts = np.unique(y_in, return_counts=True)
    p = counts / counts.sum()
    return 1 - np.sum(p**2)

# ==========================
# MODELO CNN REGULARIZADO
# ==========================
def make_model():
    model = keras.Sequential([
        layers.Input(shape=(224, 224, 1)),

        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ==========================
# PARÁMETROS
# ==========================
EPOCHS_MAX = 100       # early stopping decide cuándo parar
BATCH_SIZE = 8
console = Console()

# ==========================
# SE ASUME QUE YA EXISTEN:
# X_cnn -> (N,224,224,1)
# y     -> etiquetas
# ==========================

results = []

# ==========================
# SPLIT DE DATOS
# ==========================
X_temp, X_test, y_temp, y_test = train_test_split(
    X_cnn, y,
    test_size=0.15,
    stratify=y,
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.176,
    stratify=y_temp,
    random_state=42
)

print("Distribución TRAIN:", Counter(y_train))
print("Distribución VAL  :", Counter(y_val))
print("Distribución TEST :", Counter(y_test))

# ==========================
# GINI ANTES DEL BALANCEO
# ==========================
gini_before = gini_index(y_train)
print(f"Gini antes del balanceo: {gini_before:.4f}")

# ==========================
# BALANCEO TRAIN
# ==========================
X_train_bal, y_train_bal = oversample_train(X_train, y_train)

gini_after = gini_index(y_train_bal)
print(f"Gini después del balanceo: {gini_after:.4f}")

# ==========================
# EARLY STOPPING
# ==========================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# ==========================
# ENTRENAMIENTO
# ==========================
model = make_model()

history = model.fit(
    X_train_bal, y_train_bal,
    validation_data=(X_val, y_val),
    epochs=EPOCHS_MAX,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1
)

# ==========================
# EVALUACIÓN FINAL
# ==========================
y_prob = model.predict(X_test).ravel()
y_pred = (y_prob >= 0.5).astype(int)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
sens = recall_score(y_test, y_pred)
roc  = roc_auc_score(y_test, y_prob)

# ==========================
# TABLA RICH
# ==========================
table = Table(
    title="Resultados finales CNN con Early Stopping",
    box=box.DOUBLE,
    border_style="bright_green",
    header_style="bold bright_green"
)

table.add_column("Accuracy (%)", justify="center")
table.add_column("Precision (%)", justify="center")
table.add_column("Sensibilidad (%)", justify="center")
table.add_column("ROC-AUC", justify="center")
table.add_column("Gini antes", justify="center")
table.add_column("Gini después", justify="center")
table.add_column("Épocas reales", justify="center")

table.add_row(
    f"{acc*100:.2f}",
    f"{prec*100:.2f}",
    f"{sens*100:.2f}",
    f"{roc:.4f}",
    f"{gini_before:.3f}",
    f"{gini_after:.3f}",
    str(len(history.history["loss"]))
)

console.print(table)

# ==========================
# GRÁFICA DE ERROR
# ==========================
plt.figure(figsize=(7,5))
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], "--", label="Val")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.title("Convergencia del error (Early Stopping)")
plt.legend()
plt.grid(True)
plt.show()

# ==========================
# GRÁFICA DE DESEMPEÑO
# ==========================
plt.figure(figsize=(7,5))
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], "--", label="Val")
plt.xlabel("Épocas")
plt.ylabel("Accuracy")
plt.title("Convergencia del desempeño (Early Stopping)")
plt.legend()
plt.grid(True)
plt.show()
