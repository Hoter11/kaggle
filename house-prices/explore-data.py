import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import math
import seaborn as sns


def correlation(df):
    corr = df.corr()
    print(corr['SalePrice'].sort_values(ascending=False))


df = pd.read_csv("train.csv")
df = df.drop('Id', axis=1)

fig = plt.figure(figsize=(12, 18))

target = df.SalePrice
num_attributes = df.select_dtypes(exclude='object').drop(
    'SalePrice', axis=1).copy()

f = plt.figure(figsize=(12, 20))

for i in range(len(num_attributes.columns)):
    f.add_subplot(9, 4, i + 1)
    sns.scatterplot(num_attributes.iloc[:, i], target)

plt.tight_layout()
plt.show()

sys.exit()
for i in range(len(num_attributes.columns)):
    fig.add_subplot(9, 4, i + 1)
    sns.boxplot(y=num_attributes.iloc[:, i])

plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(12, 18))
for i in range(len(num_attributes.columns)):
    fig.add_subplot(9, 4, i + 1)
    sns.distplot(num_attributes.iloc[:, i].dropna())
    plt.xlabel(num_attributes.columns[i])

plt.tight_layout()
plt.show()

sys.exit()
correlation(df)


def encode_year(year):
    if np.isnan(year):
        return 0
    if year < 2000:
        return 1
    if year < 2010:
        return 2
    return 3


col = 'MasVnrArea'
#df = df[df[col] < 400]
df[col] = df[col].apply(lambda x: x if x == 0 else 1)
"""quality_pivot = df.pivot_table(index=col, values='SalePrice', aggfunc=np.mean)
quality_pivot.plot(kind='bar', color='blue')"""

# Next -> TotalBsmtSF
print(df[col].describe())
print(sorted(df[col].unique()))
plt.hist(df[col], color="red")
plt.show()
plt.pause(1000)
sys.exit()
"""
#df = df[['OverallQual', 'SalePrice']]
sale_median = df.groupby('SaleCondition').SalePrice.count()
sale_std = df.groupby('SaleCondition').SalePrice.std()

plt.scatter(df.SalePrice, df.SaleCondition, color='blue')
plt.show()
plt.pause(100000000)

nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
nulls

print(df.SalePrice.describe())
"""
print("Creating plot...")
plt.hist(df.SalePrice)
print("TEST")
#plt.scatter(sale_mean.index, sale_mean, color="blue")
#plt.scatter(sale_std.index, sale_std, color="green")
plt.ylabel('Price')
plt.xlabel('OverallQual')
plt.show()
plt.pause(100000000)
