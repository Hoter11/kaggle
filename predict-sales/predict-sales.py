 # -*- coding: utf-8 -*-

"""
Kaggle competition: Predict Future Sales
https://www.kaggle.com/c/competitive-data-science-predict-future-sales
"""

import sys
from joblib import dump, load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import GradientBoostingRegressor

# Parameters
train_model = True if sys.argv[1] == 'train' else False


# Read all datasets
print("Reading datasets...")
train = pd.read_csv("datasets/sales_train.csv")
test = pd.read_csv("datasets/test.csv")

train = train[int(train.shape[0]*0.1):]
test_ = train[:int(train.shape[0]*0.1)]

items = pd.read_csv("datasets/items.csv")
item_categories = pd.read_csv("datasets/item_categories.csv")
shops = pd.read_csv("datasets/shops.csv")

shops = np.sort(train.item_id.unique())

groups = train.groupby('date_block_num').groups
date_block_num = []
num_items = []
for _, g in train.groupby('date_block_num'):
    date_block_num.append(_)
    num_items.append(g.sum('item_id'))

item_cnt = train.groupby(['item_price']).agg({'item_cnt_day': sum}).sort_index()
item_cnt = item_cnt.nlargest(100, 'item_cnt_day').sort_index()

#plt.plot(item_cnt.index, item_cnt.values)
plt.plot(groups.keys(), [len(values) for values in groups.values()])
plt.ylabel('Number of items')
plt.xlabel('Months')
plt.show()

sys.exit()

if train_model:
    # Sum number of items sold monthly
    print("Preparing data...")
    train = train.groupby([
                        'shop_id',
                        'item_id',
                        'date_block_num'],
                        as_index=False).agg({'item_cnt_day': sum})

    # Keep just shop, item and number of items sold by month combinations
    train = train.drop('date_block_num', axis=1)

    # Shuffle train data
    train = train.sample(frac=1).reset_index(drop=True)
    print("Mean item/shop count by month: {}".format(
                                            train['item_cnt_day'].mean()))

    # Prepare input / output for training
    X = train[['shop_id', 'item_id']].values
    y = train['item_cnt_day'].values

    model = MLPClassifier(verbose=True)
    model = LinearRegression()
    model = LassoCV(max_iter=100, verbose=True)
    model = GradientBoostingRegressor(learning_rate=0.3, verbose=True)

    print("Training model...")
    """for i in range(0, 10):
        print("Iteration {}".format(i))
        model.partial_fit(X[:800000], y[:800000], classes=np.unique(y))
        model.partial_fit(X[800000:], y[800000:], classes=np.unique(y))

        # Save model
        dump(model, '.model-mlp')"""
    model.fit(X, y)
    print(model.score(X, y))
    dump(model, '.model-linear10')
else:
    model = load('.model-mlp')

# Predict testing data
print("Predicting results...")
X_test = test[['shop_id', 'item_id']].values
predict = model.predict(X_test)

# Create submission file
test.loc[:, 'item_cnt_month'] = pd.Series(predict)
results = test.drop(['shop_id', 'item_id'], axis=1)
results.to_csv('submission.csv', index=False)
