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

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor

# Parameters
train_model = True if sys.argv[1] == 'train' else False
month = int(sys.argv[2])

def create_plots(train):
    print("Calculating plots by month...")
    unique_items_month = train.groupby('date_block_num').item_id.nunique()
    unique_shops_month = train.groupby('date_block_num').shop_id.nunique()
    items_sold_month   = train.groupby('date_block_num').item_cnt_day.sum()

    print("Calculating revenue...")
    train['revenue'] = train.item_price * train.item_cnt_day
    revenue_month = train.groupby('date_block_num').revenue.sum()
    revenue_shop  = train.groupby('shop_id').revenue.sum()
    revenue_item  = train.groupby('item_id').revenue.sum()

    print("Calculating plots by shop...")
    unique_items_shop = train.groupby('shop_id').item_id.nunique()
    items_sold_shop   = train.groupby('shop_id').item_cnt_day.sum()

    print("Calculating plots by item...")
    unique_shops_item = train.groupby('item_id').shop_id.nunique()

    print("Creating plot...")
    plt.plot(revenue_item.index, revenue_item)
    plt.ylabel('Revenue (€)')
    plt.xlabel('Items')
    plt.show()
    plt.pause(100000000)

    sys.exit()

def investigate(train, month):
    unique_items = train.item_id.unique()
    unique_items_month = train.where(train['date_block_num'] > month).item_id.unique()
    train_unique = train[train.item_id.isin(unique_items_month)]

    print("Number of unique items: {}".format(len(unique_items)))
    print("Number of unique items after month {0}: {1}".format(
                                                month, len(unique_items_month)))
    print("Number of entries: {}".format(len(train.index)))
    print("Number of unique items that appear after month {0}: {1}".format(
                                                month, len(train_unique.index)))
    print("{} %".format(round(len(train_unique.index) / len(train.index) * 100), 2))

    sys.exit()


# Read all datasets
print("Reading datasets...")
train = pd.read_csv("datasets/sales_train.csv")
test = pd.read_csv("datasets/test.csv")

train = train[int(train.shape[0]*0.1):]
test_ = train[:int(train.shape[0]*0.1)]

items = pd.read_csv("datasets/items.csv")
item_categories = pd.read_csv("datasets/item_categories.csv")
shops = pd.read_csv("datasets/shops.csv")

# create_plots(train)
# investigate(train, month)

# Train just items that appear after month X
unique_items_month = train.where(train['date_block_num'] > month).item_id.unique()
train = train[train.item_id.isin(unique_items_month)]
train = train[train.item_cnt_day >= 0]

# Sum number of items sold monthly
print("Preparing data...")
train = train.groupby([
                    'shop_id',
                    'item_id',
                    'date_block_num'],
                    as_index=False).agg({'item_cnt_day': sum})

train_items = train.groupby([
                    'item_id',
                    'date_block_num'],
                    as_index=False).agg({'item_cnt_day': sum})

# Keep just shop, item and number of items sold by month combinations
train = train.drop('date_block_num', axis=1)
train = train.rename(columns={'item_cnt_day': 'item_cnt_month'})

train_items = train_items.drop('date_block_num', axis=1)
train_items = train_items.rename(columns={'item_cnt_day': 'item_cnt_month'})

# Shuffle train data
train = train.sample(frac=1).reset_index(drop=True)
train_items = train_items.sample(frac=1).reset_index(drop=True)
print("Mean item/shop count by month: {}".format(
                                        train['item_cnt_month'].mean()))

if train_model:
    # Prepare input / output for training
    X = train[['shop_id', 'item_id']].values
    y = train['item_cnt_month'].values
    X_items = train_items[['item_id']].values
    y_items = train_items['item_cnt_month'].values

    model = XGBRegressor()
    model_items = XGBRegressor()
    #model = DecisionTreeRegressor()

    print("Training model 1...")
    model.fit(X, y)
    print("Training model 2...")
    #model_items.fit(X_items, y_items)
    dump(model, 'models/.model-XGBoost-all-month{}'.format(month))
    #dump(model_items, 'models/.model-XGBoost-items-month{}'.format(month))

else:
    model = load('models/.model-XGBoost-all-month{}'.format(month))
    model_items = load('models/.model-XGBoost-items-month{}'.format(month))

# Predict testing data
print("Predicting results...")

predict = []
for index, row in test.iterrows():
    item = row['item_id']
    shop = row['shop_id']
    #item_price = train[(train['item_id'] == item) & (train['shop_id'] == shop)]['item_price'].mean()
    if item in train['item_id'] and shop in train['shop_id']:
        pred = model.predict([[shop, item]])
        predict.append(pred[0])
    elif item in train_items['item_id']:
        print("X")
        pred = model.predict([[item]])
        predict.append(pred[0])
    else:
        print("O")
        predict.append(float(0))

# Create submission file
test.loc[:, 'item_cnt_month'] = pd.Series(predict)
results = test.drop(['shop_id', 'item_id'], axis=1)
results.to_csv('submission.csv', index=False)
