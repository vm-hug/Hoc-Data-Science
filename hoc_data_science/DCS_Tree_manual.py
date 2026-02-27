import numpy as np
import pandas as pd
from collections import Counter
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# ==========================================
# BƯỚC 2, 3, 4, 5: CÀI ĐẶT DECISION TREE THỦ CÔNG
# ==========================================
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class ManualDecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, criterion='gini'):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.criterion = criterion  # 'gini' hoặc 'entropy'
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Điều kiện dừng (Regularization: max_depth, min_samples)
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        best_feature, best_threshold = self._best_split(X, y, n_features)

        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        left = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(best_feature, best_threshold, left, right)

    def _best_split(self, X, y, n_features):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in range(n_features):
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                gain = self._calculate_gain(y, X_column, thr)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thr

        return split_idx, split_thresh

    def _calculate_gain(self, y, X_column, threshold):
        # Chọn hàm đo lường độ thuần nhất dựa trên cấu hình
        impurity_func = self._gini if self.criterion == 'gini' else self._entropy

        parent_impurity = impurity_func(y)

        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = impurity_func(y[left_idxs]), impurity_func(y[right_idxs])
        child_impurity = (n_l / n) * e_l + (n_r / n) * e_r

        # Information Gain / Gini Gain
        gain = parent_impurity - child_impurity
        return gain

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    # Bước 2: Cài đặt hàm tính Gini Impurity
    def _gini(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return 1 - np.sum(ps ** 2)

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    # Bước 7: Visualize cây quyết định dạng text
    def print_tree(self, feature_names=None, node=None, indent=""):
        if node is None:
            node = self.root

        if node.is_leaf_node():
            print(f"{indent}└── Predict: Class {node.value}")
            return

        feat_name = feature_names[node.feature] if feature_names else f"Feature {node.feature}"
        print(f"{indent}├── If {feat_name} <= {node.threshold:.4f}:")
        self.print_tree(feature_names, node.left, indent + "│   ")
        print(f"{indent}└── Else ({feat_name} > {node.threshold:.4f}):")
        self.print_tree(feature_names, node.right, indent + "    ")


# ==========================================
# THỰC THI LAB (RUN THE WORKFLOW)
# ==========================================
if __name__ == "__main__":
    # BƯỚC 1: Load dữ liệu Healthcare, EDA cơ bản
    print("--- BƯỚC 1: LOAD DỮ LIỆU HEALTHCARE (BREAST CANCER) ---")
    data = load_breast_cancer()

    # Chọn 4 features tiêu biểu để cây không quá sâu/phức tạp khi visualize
    selected_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area']
    feature_indices = [list(data.feature_names).index(f) for f in selected_features]

    X = data.data[:, feature_indices]
    y = data.target

    df = pd.DataFrame(X, columns=selected_features)
    df['target'] = y
    print("EDA Cơ bản (5 dòng đầu):")
    print(df.head())
    print(f"\nKích thước dữ liệu: {df.shape}\n")

    # Chia tập train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Khởi tạo tham số Regularization
    MAX_DEPTH = 3
    MIN_SAMPLES_SPLIT = 10
    CRITERION = 'gini'  # Bạn có thể đổi thành 'entropy'

    # BƯỚC 6: Xây dựng cây thủ công & so sánh với sklearn
    print(f"--- BƯỚC 6: HUẤN LUYỆN VÀ SO SÁNH ---")
    print(f"Tham số: max_depth={MAX_DEPTH}, min_samples_split={MIN_SAMPLES_SPLIT}, criterion='{CRITERION}'\n")

    # 1. Cây thủ công
    manual_clf = ManualDecisionTree(max_depth=MAX_DEPTH, min_samples_split=MIN_SAMPLES_SPLIT, criterion=CRITERION)
    manual_clf.fit(X_train, y_train)
    manual_pred = manual_clf.predict(X_test)
    manual_acc = accuracy_score(y_test, manual_pred)

    # 2. Cây Sklearn
    sklearn_clf = DecisionTreeClassifier(max_depth=MAX_DEPTH, min_samples_split=MIN_SAMPLES_SPLIT, criterion=CRITERION,
                                         random_state=42)
    sklearn_clf.fit(X_train, y_train)
    sklearn_pred = sklearn_clf.predict(X_test)
    sklearn_acc = accuracy_score(y_test, sklearn_pred)

    # Bảng so sánh
    print(f"{'Mô hình':<25} | {'Accuracy trên tập Test':<20}")
    print("-" * 50)
    print(f"{'Manual Decision Tree':<25} | {manual_acc:.4f}")
    print(f"{'Sklearn Decision Tree':<25} | {sklearn_acc:.4f}\n")

    # BƯỚC 7: Visualize cây thủ công
    print("--- BƯỚC 7: VISUALIZE CÂY QUYẾT ĐỊNH THỦ CÔNG (TEXT-BASED) ---")
    manual_clf.print_tree(feature_names=selected_features)