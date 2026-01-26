import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv('./data/train.csv' , index_col = 'PassengerId')
test_df = pd.read_csv('./data/test.csv' , index_col = 'PassengerId')

print(train_df.columns)
print(test_df.columns)

print(train_df.head())
print(test_df.head())

print(train_df.describe())
print(test_df.describe())

train_df['Survived'] = train_df['Survived'].astype('category')
train_df.info()


#Hàm chuyển đổi các thuật tính sang categorical
features = ['Pclass' , 'Sex', 'SibSp' , 'Parch' , 'Embarked']
def convert_cat(df , features) :
  for feature in features :
    df[feature] = df[feature].astype('category')
convert_cat(train_df , features)
convert_cat(test_df , features)

#  Exploratory Data Analysis (EDA)
# EDA with data Categorical

train_df['Survived'].value_counts(normalize=True).to_frame()
sns.countplot(data = train_df , x = 'Sex' , hue='Survived' , palette="Greens");

cols = ['Sex','Embarked','Pclass','SibSp','Parch']

n_rows = 2
n_cols = 3

fig , ax = plt.subplots(n_rows , n_cols , figsize=(n_cols*3.5 , n_rows*3.5))

for r in range(0, n_rows):
  for c in range (0, n_cols):
     i = r*n_cols + c  # index to loop throught list "cols"
     if i < len(cols):
      ax_i = ax[r, c]
      sns.countplot(data= train_df , x = cols[i] , hue='Survived' , palette="Greens" , ax = ax_i)
      ax_i.set_title(f"Figure {i+1} : Survival Rate vs {cols[i]}")
      ax_i.legend(title='' , loc='upper left' , labels=['Not Survived' , 'Survived'])
ax.flat[-1].set_visible(False)
plt.tight_layout()
plt.show()

#EDA with data Numerical Features
#Age
sns.histplot(data = train_df , x = 'Age' , bins= 40 , hue='Survived' , palette="Blues" , kde= True)
sns.histplot(data = train_df , x = 'Fare' , bins = 40 , palette= 'Blues' , hue='Survived')

# To name for 0-25% quartile , 25-50 , 50-75 , 75-100
fare_categories = ['Economic' ,'Standard' ,'Expensive' ,'Luxury']
quartile_data =  pd.qcut(train_df['Fare'] , 4 , labels= fare_categories)

sns.countplot( x = quartile_data , hue = train_df['Survived'] , palette='Greens');

train_df['Name'].head(10)

import re # Regular Expression

def extract_title(name):
  p = re.compile(r",([\w\s]+)\.")
  return p.search(name).groups(1)[0].strip()

train_df['Title'] = train_df['Name'].apply(lambda name : extract_title(name))
train_df['Title'].value_counts()

test_df["Title"] = test_df["Name"].apply(lambda name: extract_title(name))
test_df["Title"].value_counts()

def group_title(title) :
  if title in ['Mr','Miss','Master','Mrs']:
    return title
  elif title == 'Ms':
    return "Miss"
  else :
    return "Others"

train_df['Title'] = train_df['Title'].apply(lambda title : group_title(title))
test_df['Title'] = test_df['Title'].apply(lambda title : group_title(title))
train_df['Title'].value_counts()

sns.countplot(data = train_df , x = "Title" , hue = "Survived");

# Gộp Family SibSp , Parch
train_df['Family_Size'] = train_df['SibSp'].astype(int) + train_df['Parch'].astype(int) + 1
test_df['Family_Size'] = test_df['SibSp'].astype(int) + test_df['Parch'].astype(int) + 1

# cut become 4 part (0,1] , (1 ,4] , (4,6] , (6,20]
train_df['Family_Cat'] = pd.cut(train_df['Family_Size'] , bins = [0,1,4,6,20] , labels= ['Solo' , 'Small' , 'Medium' , 'Large'])
test_df['Family_Cat'] =  pd.cut(test_df['Family_Size'] , bins = [0,1,4,6,20] , labels= ['Solo' , 'Small' , 'Medium' , 'Large'])
sns.countplot(data = train_df , x = 'Family_Cat' , hue = 'Survived')

#Data Wrangling
num_features = ['Age' , 'Fare']
cat_features = ['Sex' , 'Pclass' , 'Embarked' , 'Title' , 'Family_Cat']
features_cols = num_features + cat_features

def display_missing (df , features):
  n_rows = df.shape[0]
  for col in features :
    missing_count = df[col].isnull().sum()
    if missing_count > 0 :
      print(f"{col} has {missing_count*100/n_rows:.2f} % missing values")

display_missing(train_df , features_cols)
display_missing(test_df , features_cols)

train_df['Age'] = train_df.groupby(['Sex' , 'Pclass'] , observed=True)['Age'].transform(lambda x : x.fillna(x.median()))
test_df['Age'] = test_df.groupby(['Sex' , 'Pclass'] , observed=True)['Age'].transform(lambda x : x.fillna(x.median()))

print(display_missing(train_df , features_cols))

X = train_df[features_cols]
y = train_df['Survived']

X_test = test_df[features_cols]

# Preprocessing
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

#preprocess Pipeline
num_transformer = Pipeline(steps =[
    ('imputer' , SimpleImputer(strategy='median')),
    ('scaler' , StandardScaler())
])

cat_transformer = Pipeline(steps= [
    ('imputer' , SimpleImputer(strategy='most_frequent')),
    ('encoder' , OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num' , num_transformer , num_features),
    ('cat' , cat_transformer , cat_features)
])

preprocessor.fit(X)

X = preprocessor.transform(X)
X_test = preprocessor.transform(X_test)

from sklearn.model_selection import train_test_split

X_train , X_val , y_train , y_val = train_test_split(X , y , test_size=0.2)