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

