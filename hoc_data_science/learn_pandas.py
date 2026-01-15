import numpy as np

# 1. Tạo mảng NumPy
a = np.array([1, 2, 3, 4, 5])
b = np.array([10, 20, 30, 40, 50])

print("Mảng a:", a)
print("Mảng b:", b)

# 2. Phép toán trên mảng (vectorization)
print("\nCộng a + b:", a + b)
print("Nhân a * 2:", a * 2)
print("Bình phương a:", a ** 2)

# 3. Thống kê cơ bản
print("\nTổng a:", np.sum(a))
print("Trung bình a:", np.mean(a))
print("Giá trị lớn nhất:", np.max(a))
print("Độ lệch chuẩn:", np.std(a))

# 4. Tạo mảng 2 chiều
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print("\nMa trận:")
print(matrix)

# 5. Truy cập phần tử
print("\nPhần tử hàng 1 cột 2:", matrix[0, 1])

# 6. Reshape mảng
c = np.arange(1, 7)
print("\nMảng c:", c)

c_reshaped = c.reshape(2, 3)
print("Reshape thành 2x3:")
print(c_reshaped)
