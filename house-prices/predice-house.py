
"""
Kaggle competition: Predict House Sales
https://www.kaggle.com/c/house-prices-advanced-regression-techniques
"""

import sys
from joblib import dump
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import log10, pow
from random import randint

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datasetfunctions import correlation_matrix, show_info, downcast_dtypes

# Load data
print("Reading data...")
train = pd.read_csv('train.csv')
submission = pd.read_csv('test.csv')
submission = submission.fillna(0)

# Investigate most important features
# correlation_matrix(train, 'SalePrice', 10)
most_correlation = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea',
                    'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd',
                    'YearBuilt']
correlation_values = [0.79, 0.71, 0.64, 0.62, 0.61, 0.61, 0.56, 0.53, 0.52]
# train = train[most_correlation + ['SalePrice']]
# test = test[most_correlation]


# Prepare input / output for training
if 'Id' in train.columns:
    train = train.drop('Id', axis=1)
train = train.select_dtypes(include=['int64', 'float64'])
train.SalePrice = train.SalePrice.apply(lambda price: np.log(price))

acc = []
model_list = []
for _ in range(1,11):

    # Shuffle training data and create random seeds
    train = train.sample(frac=1).reset_index(drop=True)
    seed1 = randint(0, 100)
    seed2 = randint(0, 100)

    y = train['SalePrice'].values
    X = train.drop('SalePrice', axis=1).values
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=seed1)

    """
    # Regression model
    model = xgb.XGBRegressor(max_depth=3, min_child_weight=0.3,
                             subsample=1, eta=0.3,
                             num_round=10000, seed=seed2)

    model.fit(X_train, y_train, eval_metric='rmse')
    model_list.append(model)

    score = model.score(X_test, y_test)
    acc.append(score)
    """

    # DMatrix
    dtrain = xgb.DMatrix(X_train, y_train)
    param = {'max_depth': 10, 'eta': 0.2, 'silent': 1, 'num_round': 20000}
    param['nthread'] = 4
    param['eval_metric'] = 'mae'
    bst = xgb.train(param, dtrain)

    dtest = xgb.DMatrix(X_test)
    predict = bst.predict(dtest)
    print(type(predict), type(y_test))
    score = accuracy_score(y_test, predict)
    acc.append(score)

    print("Iteration {0: >2} | Accuracy {1: >4}%".format(
                                                    _, round(score * 100, 2)))

print("\nMean Accuracy: {}%".format(round(np.mean(acc) * 100, 2)))
print("Best model: {}".format(np.argmax(acc)+1))
sys.exit()
predict = model.predict(test.values)

"""
dtest = xgb.DMatrix(test_)
predict = bst.predict(dtest)
xgb.plot_importance(bst)
xgb.plot_tree(bst, num_trees=2)
plt.show()
"""
test['SalePrice'] = pd.Series(predict)
results = test[['Id', 'SalePrice']]
results['SalePrice'] = results['SalePrice'].apply(lambda price: pow(10, price))
results.to_csv('submission.csv', index=False)

"""
sales_price = test.SalePrice
print("Mean: {}".format(sales_price.mean()))
print("Median: {}".format(sales_price.median()))
print("Min: {}".format(sales_price.min()))
print("Max: {}".format(sales_price.max()))
plt.boxplot(sales_price)
plt.show()"""
