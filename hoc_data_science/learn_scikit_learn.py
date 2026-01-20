import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# HỌC XỮ LÍ LÀM SẠCH DỮ LIỆU VÀ TIỀN DỮ LIỆU
data_df = pd.read_csv('./Data.csv')

#Data Imputation (Missing Data Replacement)
data_df.info()

# Vòng lặp kiểm tra xem là số thiếu dữ liệu trong data bằng vòng lặp
for col in data_df.columns:
    missing_data = data_df[col].isna().sum()
    missing_percent = missing_data/len(data_df) * 100
    print(f"Columns:{col} has {missing_percent}% data")

fig , ax = plt.subplots(figsize=(8,5))
sns.heatmap(data_df.isna() , cmap="Blues" , cbar=False , yticklabels=False)

X = data_df.iloc[: , :-1].values  # dữ liệu đầu vào
y = data_df.iloc[: , -1].values #Tagert của đầu ra

# XỮ LÍ PRE_PROCESS TIỀN XỮ LÍ
from sklearn.impute import SimpleImputer

#Create an instance of class SimpleImputer: np.nan is the empty value in the dataset
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer.fit(X[:, 1:3])
X[:,1:3] = imputer.transform(X[:, 1:3])

#Encode Category Data (Mã hóa dữ liệu danh mục)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

#Encode Independent variable (X)
ct = ColumnTransformer(transformers=[('encoder' , OneHotEncoder() , [0])] , remainder='passthrough')
X = ct.fit_transform(X)

#Encode Dependent variable (y)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Splitting the dataset(X=data, y= output) into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y , test_size=0.2)

#Feature Scaling (chính quy hóa dữ liệu)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train[: , 3:] = sc.fit_transform(X_train[: , 3:])
X_test[: , 3:] = sc.transform(X_test[: , 3:])

# TRAIN THỬ MỘT MÔ HÌNH MÁY HỌC

data = pd.read_csv('train.csv' , index_col="Id")
data.head()

features = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
X = data[features]
y = data['SalePrice']

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X ,y , train_size=0.8 , test_size=0.2 , random_state=0)

# Sử dụng Decision Tree để train
from sklearn.tree import DecisionTreeRegressor
dt_model = DecisionTreeRegressor(random_state=1)
dt_model.fit(X_train , y_train)
y_preds = dt_model.predict(X_test.head())

pd.DataFrame({'y': y_test.head(), 'y_preds': y_preds})

# Sử dụng RamdomForest để train
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(X_train, y_train)

rf_val_preds = rf_model.predict(X_test)

new_house = pd.DataFrame([{
    'LotArea': 6969,
    'YearBuilt': 2021,
    '1stFlrSF': 1000,
    '2ndFlrSF': 800,
    'FullBath': 4,
    'BedroomAbvGr': 6,
    'TotRmsAbvGrd': 7
}])

rf_model.predict(new_house)