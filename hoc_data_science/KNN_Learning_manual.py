import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ==========================================
# CUSTOM KNN CLASS (Đã sửa lỗi Minkowski)
# ==========================================
class CustomKNN:
    def __init__(self, n_neighbors=5, p=2, weights='uniform'):
        self.n_neighbors = n_neighbors
        self.p = p
        self.weights = weights

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def _minkowski_distance(self, x1, x2):
        # SỬA LỖI: Dùng phép lũy thừa (**) thay vì phép nhân (*)
        return np.sum(np.abs(x1 - x2) ** self.p) ** (1 / self.p)

    def predict(self, X):
        X = np.array(X)
        y_pred = [self._predict_single(x) for x in X]
        return np.array(y_pred)

    def _predict_single(self, x):
        distances = [self._minkowski_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.n_neighbors]

        k_nearest_labels = [self.y_train[i] for i in k_indices]
        k_nearest_distances = [distances[i] for i in k_indices]

        if self.weights == 'uniform':
            most_common = Counter(k_nearest_labels).most_common(1)
            return most_common[0][0]

        elif self.weights == 'distance':
            class_weights = {}
            for i in range(self.n_neighbors):
                label = k_nearest_labels[i]
                dist = k_nearest_distances[i]
                if dist == 0:
                    return label
                weight = 1.0 / dist
                class_weights[label] = class_weights.get(label, 0) + weight
            return max(class_weights, key=class_weights.get)
        else:
            raise ValueError("Tham số weights chỉ nhận 'uniform' hoặc 'distance'")


# ==========================================
# BƯỚC 1: Load dữ liệu 2D & Trực quan hóa
# ==========================================
# Sử dụng 2 đặc trưng đầu của tập Iris (Sepal length và Sepal width) để vẽ 2D
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

# Chia tập train/test để đo độ chính xác
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Hàm vẽ Decision Boundary
def plot_decision_boundary(X, y, model, ax, title):
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title(title)


# ==========================================
# BƯỚC 2 & 3: Code, Train và Vẽ Boundary cho K = 1, 5, 15, 50
# ==========================================
k_values = [1, 5, 15, 50]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
accuracy_results = {}

for i, k in enumerate(k_values):
    # Khởi tạo và train model
    knn = CustomKNN(n_neighbors=k, p=2, weights='uniform')
    knn.fit(X_train, y_train)

    # Tính accuracy trên tập test để đánh giá
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracy_results[k] = acc

    # Vẽ Decision Boundary
    plot_decision_boundary(X_train, y_train, knn, axes[i], f"KNN (K={k}) - Train Data\nTest Acc: {acc:.2f}")

plt.tight_layout()
plt.show()