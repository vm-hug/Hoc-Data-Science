import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import umap
from tensorflow.keras.datasets import fashion_mnist

# Load Fashion-MNIST
(X_train, y_train), (_, _) = fashion_mnist.load_data()

# Lấy 5000 mẫu
n_samples = 5000
X = X_train[:n_samples]
y = y_train[:n_samples]

# Flatten: (5000, 28, 28) → (5000, 784)
X = X.reshape(n_samples, -1)

# Normalize
X = X / 255.0

start = time.time()

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

time_pca = time.time() - start

start = time.time()

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate='auto',
    init='random',
    random_state=42
)
X_tsne = tsne.fit_transform(X)

time_tsne = time.time() - start

start = time.time()

umap_model = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    random_state=42
)
X_umap = umap_model.fit_transform(X)

time_umap = time.time() - start

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# ===== PCA =====
scatter1 = axes[0].scatter(
    X_pca[:, 0], X_pca[:, 1],
    c=y, cmap='tab10', s=5
)
axes[0].set_title("PCA (2D)")
cbar1 = plt.colorbar(scatter1, ax=axes[0])
cbar1.set_label("Class")

# ===== t-SNE =====
scatter2 = axes[1].scatter(
    X_tsne[:, 0], X_tsne[:, 1],
    c=y, cmap='tab10', s=5
)
axes[1].set_title("t-SNE (perplexity=30)")
cbar2 = plt.colorbar(scatter2, ax=axes[1])
cbar2.set_label("Class")

# ===== UMAP =====
scatter3 = axes[2].scatter(
    X_umap[:, 0], X_umap[:, 1],
    c=y, cmap='tab10', s=5
)
axes[2].set_title("UMAP (n_neighbors=15)")
cbar3 = plt.colorbar(scatter3, ax=axes[2])
cbar3.set_label("Class")

# ===== BẢNG THỜI GIAN =====
table_data = [
    ["PCA", f"{time_pca:.3f}", "Kém – chồng lấn nhiều"],
    ["t-SNE", f"{time_tsne:.3f}", "Rất tốt – cụm rõ"],
    ["UMAP", f"{time_umap:.3f}", "Tốt – gần t-SNE"]
]

columns = ["Method", "Runtime (seconds)", "Cluster Separation"]

table = plt.table(
    cellText=table_data,
    colLabels=columns,
    loc='bottom',
    cellLoc='left',
    bbox=[0.15, -0.45, 0.7, 0.25]
)

table.scale(1.5, 2)

# Chừa chỗ cho bảng phía dưới
plt.subplots_adjust(bottom=0.28)
plt.tight_layout()
plt.show()
