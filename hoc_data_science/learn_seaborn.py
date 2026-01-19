import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print(sns.get_dataset_names())
sns.set_theme()
tips_df = sns.load_dataset('tips')
tips_df.head()

sns.histplot(tips_df["total_bill"]);

sns.displot(data=tips_df , x='total_bill' , col='time' , kde=True);

cereal_df = pd.read_csv('cereal.csv')
cereal_df.head()

fields = ['shelf', 'weight' , 'cups' , 'rating', 'name', 'mfr', 'type']
cereal_df_new = cereal_df.drop(fields , axis=1)
cereal_df_new.head()

#corr() tính độ tương quan giữa các thuộc tính
cereal_corr = cereal_df_new.corr() #Get correlation data
print(cereal_corr)

# ones_like can build a matrix of booleans (True , False) with the same shape as our data
ones_corr = np.ones_like(cereal_corr, dtype=bool)
print(ones_corr)

# np's triu : return only upper triange matrix
mask =  np.triu(ones_corr)
print(mask)

sns.heatmap(
    data= cereal_corr,
    mask = mask
)

adjusted_mask = mask[1: , :-1]
adjusted_cereal_corr = cereal_corr.iloc[1: , :-1]

fig, ax = plt.subplots(figsize=(10,8))
#Change color with favorit
cmap = sns.diverging_palette(0,230, 90,60, as_cmap=True)

sns.heatmap(data= adjusted_cereal_corr , mask = adjusted_mask,
            annot=True, annot_kws={"fontsize":13} , fmt=".2f" , cmap=cmap,
            vmin=-1 , vmax=1 , linecolor='white' , linewidths=0.5);
yticks = [i.upper() for i in adjusted_cereal_corr.index]
xticks = [i.upper() for i in adjusted_cereal_corr.columns]

ax.set_yticklabels(yticks , rotation=0)
ax.set_xticklabels(xticks , rotation=90);


