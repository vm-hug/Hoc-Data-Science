import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns' , None)
pd.set_option('display.width', None)

cities = pd.read_csv('california_cities.csv')
print(cities.head(5))

lat , lon = cities['latd'] , cities['longd']
population , area = cities['population_total'] , cities['area_total_km2']

print(plt.style.available)
plt.style.use('seaborn-v0_8-darkgrid')
plt.figure(figsize=(10,10))

plt.scatter(lon , lat , c= np.log10(population), s= area , linewidths=0 , alpha=0.5 , cmap='viridis')
plt.axis('equal')
plt.xlabel('longtitude')
plt.ylabel('latitude')
plt.colorbar(label='log10(population)')
plt.clim(3,7)

#create a legend for cities sizes
area_range = [50,100,300,500]
for area in area_range:
    plt.scatter([] , [] , s=area , label=str(area)+ 'km^2')
plt.legend()

plt.show()

