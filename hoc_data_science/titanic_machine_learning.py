import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')

print(train_df.head())
print(test_df.head())