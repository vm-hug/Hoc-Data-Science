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

# Binary classification
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score , recall_score , classification_report , confusion_matrix

log_reg = LogisticRegression(solver='liblinear' , max_iter=1000)
log_reg.fit(X_train , y_train)

log_reg.score(X_val, y_val)
y_pred = log_reg.predict(X_val)
precision_score(y_val , y_pred) , recall_score(y_val , y_pred)

print(classification_report(y_val, y_pred))

# Nâng bậc của linear y = ax1 + bx2 + bias -> y = ax1^2 + bx2^2 + cx1*x2 + bias sử dụng PolynomiaFeatures
poly = PolynomialFeatures(degree = 2)
poly_features_X_train = poly.fit_transform(X_train)
poly_features_X_val = poly.transform(X_val)

poly_log_reg = LogisticRegression(solver='liblinear' , max_iter=1000)
poly_log_reg.fit(poly_features_X_train , y_train)

poly_log_reg.score(poly_features_X_val , y_val)

#Decision Tree
decision_tree = DecisionTreeClassifier(criterion='entropy' , max_depth=8 , random_state= 2026)
decision_tree.fit(X_train , y_train)

decision_tree.score(X_val , y_val)

# Cross-Vadication
from sklearn.model_selection import cross_val_score
log_reg_cv = LogisticRegression(solver='liblinear' , max_iter=1000)
dt_cv = DecisionTreeClassifier(criterion='entropy' , max_depth=8 , random_state= 2026)

lr_scores = cross_val_score(log_reg_cv , X , y , scoring = 'accuracy' , cv = 5)
lr_scores.mean() , lr_scores.std()

dt_scores = cross_val_score(dt_cv , X , y , scoring = 'accuracy' , cv = 5)
dt_scores.mean() , dt_scores.std()

# Base line Model Comparison

from sklearn.svm import LinearSVC , SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier , ExtraTreesClassifier , AdaBoostClassifier
from xgboost import XGBClassifier
seed = 2026

models = [
    LinearSVC(max_iter=12000,random_state=seed),
    SVC(random_state=seed),
    KNeighborsClassifier(metric='minkowski' , p=2),
    LogisticRegression(solver='liblinear' , max_iter=1000),
    DecisionTreeClassifier(criterion='entropy' ,random_state= 2026),
    RandomForestClassifier(random_state=seed),
    GradientBoostingClassifier(random_state=seed),
    ExtraTreesClassifier(),
    AdaBoostClassifier(),
    XGBClassifier(use_label_encoder = False , eval_metric='logloss' ,random_state=seed)
]

from sklearn.model_selection import StratifiedKFold
def generate_baseline_results(models , X, y , metrics , cv =5 , plot_results = False):
  #define k-fold:
  kflod = StratifiedKFold(cv , shuffle=True , random_state=seed)
  entries = []
  for model in models:
    model_name = model.__class__.__name__
    scores = cross_val_score(model , X , y , scoring = metrics , cv = kflod)
    for flod_idx , score in enumerate(scores):
      entries.append((model_name , flod_idx , score))

  cv_df = pd.DataFrame(entries , columns=['model_name' , 'fold_idx' , 'accuracy_score'])

  if plot_results:
    sns.boxplot(x='model_name' , y='accuracy_score' , data=cv_df , color='skyblue' , showmeans= True)
    plt.title("Boxplot of Baseline Model Accuracy using 5-fold cross-validation")
    plt.xticks(rotation=45)
    plt.show()

  #Summary result
  mean = cv_df.groupby('model_name')['accuracy_score'].mean()
  std = cv_df.groupby('model_name')['accuracy_score'].std()

  baseline_results = pd.concat([mean , std] , axis=1 , ignore_index=True)
  baseline_results.columns = ['Mean' , 'Standard Deviation']

  baseline_results.sort_values(by=['Mean'] , ascending=False , inplace=True)

  return baseline_results

generate_baseline_results(models , X , y , metrics = 'accuracy' , cv=5 , plot_results= False)