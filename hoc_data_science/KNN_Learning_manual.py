import numpy as np
from collections import Counter


class CustomKNN:
    def __init__(self, n_neighbors=5, p=2, weights='uniform'):
        self.n_neighbors = n_neighbors
        self.p = p
        self.weights = weights

    def fit(self, X, y):
        """Lưu trữ dữ liệu huấn luyện (KNN là thuật toán lazy learning)"""
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def _minkowski_distance(self, x1, x2):
        """Tính khoảng cách Minkowski giữa 2 điểm"""
        return np.sum(np.abs(x1 - x2) * self.p) * (1 / self.p)

    def predict(self, X):
        """Dự đoán nhãn cho danh sách các điểm dữ liệu mới"""
        X = np.array(X)
        y_pred = [self._predict_single(x) for x in X]
        return np.array(y_pred)

    def _predict_single(self, x):
        """Dự đoán nhãn cho một điểm dữ liệu duy nhất"""
        # 1. Tính khoảng cách từ x đến tất cả các điểm trong tập huấn luyện
        distances = [self._minkowski_distance(x, x_train) for x_train in self.X_train]

        # 2. Lấy chỉ mục (index) của k điểm gần nhất
        k_indices = np.argsort(distances)[:self.n_neighbors]

        # 3. Lấy nhãn và khoảng cách của k điểm này
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        k_nearest_distances = [distances[i] for i in k_indices]

        # 4. Phân loại dựa trên tham số weights
        if self.weights == 'uniform':
            # Bầu chọn đa số đơn thuần
            most_common = Counter(k_nearest_labels).most_common(1)
            return most_common[0][0]

        elif self.weights == 'distance':
            # Bầu chọn dựa trên trọng số khoảng cách (1 / khoảng cách)
            class_weights = {}
            for i in range(self.n_neighbors):
                label = k_nearest_labels[i]
                dist = k_nearest_distances[i]

                # Tránh lỗi chia cho 0 nếu điểm kiểm tra trùng đúng với điểm huấn luyện
                if dist == 0:
                    return label

                weight = 1.0 / dist
                class_weights[label] = class_weights.get(label, 0) + weight

            # Trả về nhãn có tổng trọng số lớn nhất
            return max(class_weights, key=class_weights.get)
        else:
            raise ValueError("Tham số weights chỉ nhận 'uniform' hoặc 'distance'")