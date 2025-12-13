# ############################################################
# # DETECCIÓN DE CATARATAS 
# # 1. Comprensión y preparación de datos
# # 2. Selección / extracción de características (PCA)
# # 3. Modelado supervisado (Regresión Logística + Naive Bayes)
# # 4. Modelado no supervisado (K-means)
# # 5. Interpretación y comunicación (matriz de confusión, ROC)
# ############################################################

# import os
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt

# from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn.cluster import KMeans
# from sklearn.metrics import (
#     accuracy_score,
#     f1_score,
#     roc_auc_score,
#     confusion_matrix,
#     classification_report,
#     RocCurveDisplay,
# )

# # ==========================================================
# # CONFIGURACIÓN
# # ==========================================================
# RUTA = r"C:\TESIS\archive (3)\datasets"
# CLASES = ["normal", "cataract"]  # 0 = normal, 1 = catarata
# IMG_SIZE = (224, 224)            # tamaño al que se redimensionan las imágenes

# # ==========================================================
# # 1. COMPRENSIÓN Y PREPARACIÓN DE DATOS
# # ==========================================================

# print("\n=== 1. COMPRENSIÓN Y PREPARACIÓN DE DATOS ===")

# # 1.1 Carga de imágenes
# X, y = [], []

# for label, clase in enumerate(CLASES):
#     carpeta = os.path.join(RUTA, clase)
#     for f in os.listdir(carpeta):
#         if f.lower().endswith((".jpg", ".jpeg", ".png")):
#             ruta_img = os.path.join(carpeta, f)
#             img = Image.open(ruta_img).convert("RGB").resize(IMG_SIZE)
#             X.append(np.array(img))
#             y.append(label)

# X = np.array(X)
# y = np.array(y)

# print(f"Imágenes cargadas: {len(X)}")
# print(f"Shape de X (n, alto, ancho, canales): {X.shape}")
# print(f"Shape de y: {y.shape}")
# print(f"Clases y conteos:", np.unique(y, return_counts=True))

# # 1.2 Exploración inicial: tipos y estadísticos
# print("\n--- 1.2 Exploración inicial ---")
# print(f"Tipo de datos de X: {X.dtype}")
# print(f"Tipo de datos de y: {y.dtype}")

# X_float = X.astype("float32")
# print("Estadísticos de intensidad de píxel (0-255):")
# print(f"  Mínimo:   {X_float.min():.2f}")
# print(f"  Máximo:   {X_float.max():.2f}")
# print(f"  Media:    {X_float.mean():.2f}")
# print(f"  Desv.Std: {X_float.std():.2f}")

# valores, conteos = np.unique(y, return_counts=True)
# proporciones = conteos / conteos.sum()
# print("\nDistribución de clases:")
# for v, c, p in zip(valores, conteos, proporciones):
#     print(f"  Clase {v} ({CLASES[v]}): {c} imágenes ({p*100:.2f} %)")

# # 1.3 Valores faltantes
# print("\n--- 1.3 Valores faltantes ---")
# print("NaN en X:", np.isnan(X_float).sum())
# print("NaN en y:", np.isnan(y.astype("float32")).sum())

# # 1.4 Limpieza básica
# print("\n--- 1.4 Limpieza ---")
# X_clean, y_clean = [], []
# for img, label in zip(X, y):
#     if img.max() == 0:      # imagen completamente negra
#         continue
#     if img.min() == 255:    # imagen completamente blanca
#         continue
#     if np.isnan(img).any(): # imagen corrupta
#         continue
#     X_clean.append(img)
#     y_clean.append(label)

# X = np.array(X_clean)
# y = np.array(y_clean)

# print("Imágenes después de limpieza:", X.shape[0])
# print("Distribución de clases tras limpieza:",
#       np.unique(y, return_counts=True))

# # 1.5 Escalado 0–1 (visión por computador)
# print("\n--- 1.5 Escalado (0–1) ---")
# X_scaled = X.astype("float32") / 255.0
# print("Nuevo rango de valores:", X_scaled.min(), "→", X_scaled.max())

# # 1.6 Evitar fuga de datos: split estratificado
# print("\n--- 1.6 Train/Test Split (evitar data leakage) ---")
# X_train_img, X_test_img, y_train, y_test = train_test_split(
#     X_scaled, y, test_size=0.2, random_state=42, stratify=y
# )

# print("Train imágenes:", X_train_img.shape)
# print("Test  imágenes:", X_test_img.shape)

# # ==========================================================
# # 2. SELECCIÓN / EXTRACCIÓN DE CARACTERÍSTICAS (PCA)
# # ==========================================================

# print("\n=== 2. SELECCIÓN / EXTRACCIÓN DE CARACTERÍSTICAS ===")

# # 2.1 Aplanar imágenes (224x224x3 -> 150528)
# X_train_flat = X_train_img.reshape(X_train_img.shape[0], -1)
# X_test_flat  = X_test_img.reshape(X_test_img.shape[0], -1)

# print("Shape imágenes aplanadas (train):", X_train_flat.shape)
# print("Shape imágenes aplanadas (test) :", X_test_flat.shape)

# # 2.2 Estandarización previa (StandardScaler)
# scaler = StandardScaler()
# X_train_std = scaler.fit_transform(X_train_flat)
# X_test_std  = scaler.transform(X_test_flat)

# print("Estandarización aplicada (StandardScaler).")

# # 2.3 PCA para explicar ~95% de la varianza
# pca = PCA(n_components=0.95, svd_solver="full")
# X_train_pca = pca.fit_transform(X_train_std)
# X_test_pca  = pca.transform(X_test_std)

# print("\nPCA aplicado:")
# print("  Nº de componentes:", pca.n_components_)
# print("  Varianza explicada total:",
#       round(pca.explained_variance_ratio_.sum(), 4))

# print("\nPrimeras 5 cargas (componentes):")
# print(pca.components_[:5])

# # ==========================================================
# # 3. MODELO SUPERVISADO
# #    - Regresión Logística
# #    - Naive Bayes
# # ==========================================================

# print("\n=== 3. MODELO SUPERVISADO ===")

# # 3.1 Regresión Logística
# print("\n--- 3.1 Regresión Logística ---")
# log_reg = LogisticRegression(max_iter=1000)

# # Validación cruzada (k=5) sobre train
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# cv_scores = cross_val_score(log_reg, X_train_pca, y_train, cv=cv,
#                             scoring="accuracy")
# print("CV Accuracy (k=5):", cv_scores.mean(), "±", cv_scores.std())

# # Entrenamiento final y evaluación en test
# log_reg.fit(X_train_pca, y_train)
# y_pred_lr = log_reg.predict(X_test_pca)
# y_proba_lr = log_reg.predict_proba(X_test_pca)[:, 1]

# acc_lr = accuracy_score(y_test, y_pred_lr)
# f1_lr  = f1_score(y_test, y_pred_lr)
# roc_lr = roc_auc_score(y_test, y_proba_lr)

# print("Accuracy (test):", acc_lr)
# print("F1 (test):      ", f1_lr)
# print("ROC-AUC (test):", roc_lr)
# print("\nReporte de clasificación (Regresión Logística):")
# print(classification_report(y_test, y_pred_lr, target_names=CLASES))

# # 3.2 Naive Bayes (Gaussian)
# print("\n--- 3.2 Naive Bayes (GaussianNB) ---")
# gnb = GaussianNB()
# gnb.fit(X_train_pca, y_train)

# y_pred_nb = gnb.predict(X_test_pca)
# y_proba_nb = gnb.predict_proba(X_test_pca)[:, 1]

# acc_nb = accuracy_score(y_test, y_pred_nb)
# f1_nb  = f1_score(y_test, y_pred_nb)
# roc_nb = roc_auc_score(y_test, y_proba_nb)

# print("Accuracy (test):", acc_nb)
# print("F1 (test):      ", f1_nb)
# print("ROC-AUC (test):", roc_nb)

# # ==========================================================
# # 4. MODELO NO SUPERVISADO (K-means)
# # ==========================================================

# print("\n=== 4. MODELO NO SUPERVISADO (K-means) ===")

# kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
# clusters = kmeans.fit_predict(X_train_pca)

# print("Tamaño de cada clúster (train):",
#       np.unique(clusters, return_counts=True))

# # Proyección 2D con las dos primeras PCs
# plt.figure(figsize=(6, 5))
# plt.scatter(
#     X_train_pca[:, 0], X_train_pca[:, 1],
#     c=clusters, cmap="viridis", s=10
# )
# plt.title("K-means sobre proyección PCA (train)")
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.colorbar(label="Cluster")
# plt.tight_layout()
# plt.show()

# # ==========================================================
# # 5. INTERPRETACIÓN Y COMUNICACIÓN
# #    - Matriz de confusión
# #    - Curva ROC
# # ==========================================================

# print("\n=== 5. INTERPRETACIÓN Y COMUNICACIÓN ===")

# # 5.1 Matriz de confusión (modelo principal: Regresión Logística)
# cm = confusion_matrix(y_test, y_pred_lr)
# print("\nMatriz de confusión (Logistic Regression):")
# print(cm)

# plt.figure(figsize=(4, 4))
# plt.imshow(cm, cmap="Blues")
# plt.title("Matriz de confusión - Regresión Logística")
# plt.xlabel("Predicción")
# plt.ylabel("Real")
# plt.xticks([0, 1], CLASES)
# plt.yticks([0, 1], CLASES)
# for i in range(2):
#     for j in range(2):
#         plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
# plt.tight_layout()
# plt.show()

# # 5.2 Curva ROC (Regresión Logística)
# RocCurveDisplay.from_predictions(y_test, y_proba_lr)
# plt.title("Curva ROC - Regresión Logística")
# plt.show()

# print("\nPipeline COMPLETO ejecutado correctamente.")

###########################################################
# DETECCIÓN DE CATARATAS 
# 1. Comprensión y preparación de datos
# 2. Balanceo de datos (SMOTE) ANTES del train/test
# 3. Selección / extracción de características (PCA)
# 4. Modelado supervisado (Regresión Logística + Naive Bayes)
# 5. Modelado no supervisado (K-means)
# 6. Interpretación y comunicación (matriz de confusión, ROC)
############################################################

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
)

from imblearn.over_sampling import SMOTE  # ← BALANCEADOR

# ==========================================================
# CONFIGURACIÓN
# ==========================================================
RUTA = r"C:\TESIS\archive (3)\datasets"
CLASES = ["normal", "cataract"]  # 0 = normal, 1 = catarata
IMG_SIZE = (224, 224)            # tamaño de redimensionamiento

# ==========================================================
# 1. COMPRENSIÓN Y PREPARACIÓN DE DATOS
# ==========================================================

print("\n=== 1. COMPRENSIÓN Y PREPARACIÓN DE DATOS ===")

# 1.1 Carga de imágenes
X, y = [], []

for label, clase in enumerate(CLASES):
    carpeta = os.path.join(RUTA, clase)
    for f in os.listdir(carpeta):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            ruta_img = os.path.join(carpeta, f)
            img = Image.open(ruta_img).convert("RGB").resize(IMG_SIZE)
            X.append(np.array(img))
            y.append(label)

X = np.array(X)
y = np.array(y)

print(f"Imágenes cargadas: {len(X)}")
print(f"Shape X: {X.shape}")
print(f"Clases:", np.unique(y, return_counts=True))

# 1.2 Explorar estadísticas
X_float = X.astype("float32")
print("\nDistribución original de clases:", np.unique(y, return_counts=True))

# 1.3 Limpieza básica
X_clean, y_clean = [], []
for img, label in zip(X, y):
    if img.max() == 0:   # completamente negra
        continue
    if img.min() == 255: # completamente blanca
        continue
    if np.isnan(img).any():  # NaNs
        continue
    X_clean.append(img)
    y_clean.append(label)

X = np.array(X_clean)
y = np.array(y_clean)
print("Tras limpieza:", np.unique(y, return_counts=True))

# 1.4 Escalado 0–1
X_scaled = X.astype("float32") / 255.0

# ==========================================================
# 1.7 BALANCEO CON SMOTE (ANTES DEL TRAIN/TEST SPLIT)
# ==========================================================
print("\n=== 1.7 Balanceo SMOTE (ANTES del split) ===")
print("Distribución antes de SMOTE:", np.unique(y, return_counts=True))

# Aplanar TODAS las imágenes para poder usar SMOTE
X_flat = X_scaled.reshape(X_scaled.shape[0], -1)

sm = SMOTE(random_state=42)
X_bal_flat, y_bal = sm.fit_resample(X_flat, y)

print("Distribución después de SMOTE:", np.unique(y_bal, return_counts=True))

# Reconstruir imágenes después del balanceo
X_bal = X_bal_flat.reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 3)

# ==========================================================
# 1.8 TRAIN/TEST SPLIT (DESPUÉS DE SMOTE)
# ==========================================================
X_train_img, X_test_img, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
)

print("\nTrain (balanceado):", np.unique(y_train, return_counts=True))
print("Test  (balanceado):", np.unique(y_test, return_counts=True))

# ==========================================================
# 2. PCA – EXTRACCIÓN DE CARACTERÍSTICAS
# ==========================================================
print("\n=== 2. PCA ===")

# Aplanar nuevamente para PCA
X_train_flat = X_train_img.reshape(X_train_img.shape[0], -1)
X_test_flat  = X_test_img.reshape(X_test_img.shape[0], -1)

# Estandarización
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train_flat)
X_test_std  = scaler.transform(X_test_flat)

# PCA al 95% de varianza
pca = PCA(n_components=0.95, svd_solver="full")
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca  = pca.transform(X_test_std)

print("Componentes PCA:", pca.n_components_)
print("Varianza explicada:", pca.explained_variance_ratio_.sum())

# ==========================================================
# 3. MODELOS SUPERVISADOS
# ==========================================================
print("\n=== 3. MODELOS SUPERVISADOS ===")

# ------------------------- #
# 3.1 REGRESIÓN LOGÍSTICA
# ------------------------- #
print("\n--- Regresión Logística ---")
log_reg = LogisticRegression(max_iter=1000)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(log_reg, X_train_pca, y_train, cv=cv, scoring="accuracy")
print("CV Accuracy:", cv_scores.mean())

log_reg.fit(X_train_pca, y_train)
y_pred_lr = log_reg.predict(X_test_pca)
y_proba_lr = log_reg.predict_proba(X_test_pca)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("F1:", f1_score(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_lr))
print(classification_report(y_test, y_pred_lr, target_names=CLASES))

# ------------------------- #
# 3.2 NAIVE BAYES
# ------------------------- #
print("\n--- Naive Bayes ---")
gnb = GaussianNB()
gnb.fit(X_train_pca, y_train)

y_pred_nb = gnb.predict(X_test_pca)
y_proba_nb = gnb.predict_proba(X_test_pca)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("F1:", f1_score(y_test, y_pred_nb))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_nb))

# ==========================================================
# 4. K-MEANS
# ==========================================================
print("\n=== 4. K-means ===")

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_train_pca)
print("Clusters:", np.unique(clusters, return_counts=True))

# ==========================================================
# 5. INTERPRETACIÓN FINAL
# ==========================================================
print("\n=== 5. MATRIZ DE CONFUSIÓN ===")

cm = confusion_matrix(y_test, y_pred_lr)
print(cm)

plt.imshow(cm, cmap="Blues")
plt.title("Matriz de Confusión - Logistic Regression")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.xticks([0, 1], CLASES)
plt.yticks([0, 1], CLASES)
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.show()

RocCurveDisplay.from_predictions(y_test, y_proba_lr)
plt.title("Curva ROC - Logistic Regression")
plt.show()

print("\nPipeline COMPLETO ejecutado correctamente con SMOTE antes del train/test.")
