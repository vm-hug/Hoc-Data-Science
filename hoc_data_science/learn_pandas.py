import lamda
import pandas as pd
import numpy as np

#set option display
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

df = pd.read_csv("chipotle.tsv" , sep="\t")

# hiển thị 5 cột đầu tiên
print(df.head(5))

# hiển thị các thông tin kiểu dữ liệu của các cột trong data frame
print(df.info())

print(df.describe())

# loc vs iloc  (location) & (index location)
print(df.loc[(df.quantity == 15) | (df.item_name == "Nantucket Nectar")])
print(df.loc[(df.quantity == 2) & (df.item_name == "Nantucket Nectar")])

print(df.iloc[[10]]) # hiện thông tin của giá trị theo index
print(df.iloc[3:11])

print(df.iloc[3:5 , :-1])

df.item_price = df.item_price.apply(lambda x: float(x.replace('$','')))
df["total_price"] = df["quantity"]* df["item_price"]

print(df.total_price)

c= df.groupby("item_name")["quantity"].sum().sort_values()
print(c)
