"""
Kaggle competition: Predict House Sales
https://www.kaggle.com/c/house-prices-advanced-regression-techniques
"""

import sys
from joblib import dump
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import log10, pow, sqrt
from random import randint

import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datasetfunctions import correlation_matrix, show_info, downcast_dtypes

pretty_metrics = {
    'neg_mean_squared_error': 'MSE',
    'neg_mean_absolute_error': 'MAE',
    'r2': 'R^2'
}
"""
MSZONE -> JUST 4 NANS ON TEST (need?)
"""


def clean_datasets(train, submission):
    object_columns = train.select_dtypes(include=['object'])
    for col in object_columns:
        dum = pd.get_dummies(train[col], prefix=col, dummy_na=True)
        train = train.merge(dum, left_index=True, right_index=True)
        submission = submission.merge(dum, left_index=True, right_index=True)

    return train, submission


# Load data
print("Reading data...")
train = pd.read_csv('train.csv')
submission = pd.read_csv('test.csv')

train, submission = clean_datasets(train, submission)
correlation_matrix(train, 'SalePrice')

# Investigate most important features
# correlation_matrix(train, 'SalePrice', 10)
most_correlation = [
    'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF',
    '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt'
]
correlation_values = [0.79, 0.71, 0.64, 0.62, 0.61, 0.61, 0.56, 0.53, 0.52]
# train = train[most_correlation + ['SalePrice']]
# submission = submission[most_correlation]

# Prepare input / output for training
if 'Id' in train.columns:
    train = train.drop('Id', axis=1)
train = train.select_dtypes(include=['int64', 'float64', 'uint8'])
train.SalePrice = train.SalePrice.apply(lambda price: np.log10(price))
# train = downcast_dtypes(train)

# Regression model
model_seed = randint(0, 100)
model = xgb.XGBRegressor(
    max_depth=5,
    min_child_weight=0.7,
    subsample=1,
    eta=0.15,
    num_round=100,
    seed=model_seed)
model = DecisionTreeRegressor(max_depth=5)
print(train.dtypes.unique())

results = []
model_list = []
for _ in range(1, 11):

    # Shuffle training data and create random seeds
    train = train.sample(frac=1).reset_index(drop=True)
    seed1 = randint(0, 100)
    seed2 = randint(0, 100)

    train.fillna(0)
    y = train['SalePrice'].values
    X = train.drop('SalePrice', axis=1).values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed1)

    # model.fit(X_train, y_train, eval_metric='mae')
    model.fit(X_train, y_train)
    model_list.append(model)

    kfold = model_selection.KFold(n_splits=10, random_state=seed2)
    chosen = 1
    metrics = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']

    score = model_selection.cross_val_score(
        model, X_test, y_test, cv=kfold, scoring=metrics[chosen])
    results.append(score.mean())

    print("Iteration {0: >2} | {1} {2: 2.4f} ({3: 1.3f} std)".format(
        _, pretty_metrics[metrics[chosen]], score.mean(), score.std()))

best_result = np.argmin([abs(x) for x in results])
print("\nMean {0}: {1: 2.4f}".format(pretty_metrics[metrics[chosen]],
                                     results[best_result]))
print("Best model: {}".format(best_result + 1))

best_model = model_list[np.argmin(results)]
# Predict test data just with columns used for training
y_predict = submission[train.drop('SalePrice', axis=1).columns].values
predict = best_model.predict(y_predict)

submission['SalePrice'] = pd.Series(predict)
results = submission[['Id', 'SalePrice']]
results.SalePrice = results.SalePrice.apply(lambda price: pow(10, price))
results.to_csv('submission.csv', index=False)
"""
sales_price = test.SalePrice
print("Mean: {}".format(sales_price.mean()))
print("Median: {}".format(sales_price.median()))
print("Min: {}".format(sales_price.min()))
print("Max: {}".format(sales_price.max()))
plt.boxplot(sales_price)
plt.show()"""
