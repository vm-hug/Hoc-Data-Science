import numpy as np

np.array([1,2,3,4])

np.array([3.14,4,5,6])

#convert sang kiểu float
np.array([1,2,3,4] , dtype='float32')

a2 = np.array([[1,2,3],[3,4,5]])
type(a2)

# hiển thị số hàng và số cột
a2.shape

# hiển thị số chiều của mảng
a2.ndim

a2.size

#Create numpy array from hàm có sẳn (zeros , ones , full , arange , linspace)
np.zeros([2,4] ,  dtype=int)

np.ones([3,5] , dtype=float)

#function arange (start , end , step)
np.arange(0 , 20, 2)

np.full((3,5) , 6.9)

np.linspace(0, 1 , 5)

#Random
# nếu mình muốn random ở đâu cx có giá trị giống nhau khi random như vậy thì sử dụng seed()

np.random.seed(34)

np.random.random((4,4))

np.random.normal(0,2,(3,3))

x2 = np.random.randint(0,10, size=6)

# slicing array
x1 = np.random.randint(20, size=6)
x1

x1[1:4]

# reshape & transpose
grid = np.arange(1,10)
grid.shape

grid.reshape((3,3))


np.vstack((x1,x2))
np.hstack([x1,x2])

#splitting of array
x = np.array([1,2,3,99,69,3,2,1])
np.split(x , [3,5])

#tính giá trị trung bình
np.mean()

#độ lệch chuẩn
np.std()

#nhân tích vô hướng
np.dot(arr1, arr2)

#hồi quy tuyến tính
np.random.seed(0)

sales_amounts = np.random.randint(20, size=(5,3))
sales_amounts

import pandas as pd

weekly_sales = pd.DataFrame(sales_amounts , index= ["Mon" , "Tues" , "Wed" , "Thus" , "Fri"],
                                            columns=["Almond butter" , "Peanut butter" , "Cashew butter"])
weekly_sales

prices = np.array([10,8,12])

total_prices = weekly_sales.dot(prices.T)
total_prices