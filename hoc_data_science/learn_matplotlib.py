# HỌC MATPLOTLIB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

var = plt.style.available
plt.style.use('seaborn-v0_8-darkgrid')

x = [1,2,3,5]
y = [4,5,6,7]
plt.plot(x,y , color = 'yellow')
#plt.show()

# Pylot Api & Object-Oriented(OO) API
x = np.linspace(0, 10 , 1000)
plt.plot(x , np.sin(x) , color="blue" , linestyle="dashed" , label="sin(x)");
plt.plot(x , np.cos(x) , color="red" , label="cos(x)")

plt.title("Biểu đồ sin(x) và cos(x)")
plt.xlabel("Bien X")
plt.ylabel("Sin(X)")
#plt.xlim([0,4])
#plt.ylim([-0.6,0.6])
plt.legend()
#plt.show()

# các kiểu đồ thị trong matplotlib (line , scatter , bar , hits , subplots())
# Object-Oriented API & Pylot API
# Pylot
plt.scatter(x , np.exp(x));

#OO API
rng = np.random.RandomState(0) # cục bộ từng object còn seed là toàn cục
x = rng.randn(100)
y = rng.randn(100)

sizes = 1000* rng.rand(100)
colors = rng.rand(100)

fig , ax = plt.subplots()
img1 = ax.scatter(x,y , s= sizes , c=colors , alpha=0.3 , cmap='viridis')
fig.colorbar(img1)

# bar
soft_drink_price = {"Tea" : 10,
                    "Coffe" : 20,
                    "Milk" : 30,
                    "Drink" : 40,
                    "Milkin Drink" : 50,
                    }
fig , ax = plt.subplots()
ax.bar(soft_drink_price.keys() , soft_drink_price.values())
ax.set(title= "Bach hoa xanh" , ylabel="Price $")

#histogram
np.random.seed(42)
student_hight = np.random.normal(170 , 10 , 250)
ax.hist(student_hight);

#subplots
fig , ax = plt.subplots(nrows=2 , ncols=2 , figsize=(10,5))
x = np.linspace(0, 10 , 1000)
x1 = np.random.randn(10)
x2 = np.random.randn(10)
ax[0,0].plot(x, np.sin(x))
ax[0,1].scatter(x1,x2)
ax[1,0].bar(soft_drink_price.keys() , soft_drink_price.values())
ax[1,1].hist(student_hight);