
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
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datasetfunctions import correlation_matrix, show_info, downcast_dtypes

metric = {'neg_mean_squared_error': 'MSE', 'neg_mean_absolute_error': 'MAE', 'r2': 'R^2'}

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
train.SalePrice = train.SalePrice.apply(lambda price: np.log10(price))

results = []
model_list = []
for _ in range(1,11):

    # Shuffle training data and create random seeds
    train = train.sample(frac=1).reset_index(drop=True)
    seed1 = randint(0, 100)
    seed2 = randint(0, 100)
    seed3 = randint(0, 100)

    y = train['SalePrice'].values
    X = train.drop('SalePrice', axis=1).values
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=seed1)


    # Regression model
    model = xgb.XGBRegressor(max_depth=3, min_child_weight=0.3,
                             subsample=1, eta=0.3,
                             num_round=10000, seed=seed2)

    model.fit(X_train, y_train, eval_metric='mae')
    model_list.append(model)

    kfold = model_selection.KFold(n_splits=10, random_state=seed3)
    scoring = 'neg_mean_squared_error'
    #scoring = 'neg_mean_absolute_error'
    scoring = 'r2'
    score = model_selection.cross_val_score(model, X_test, y_test,
                                            cv=kfold, scoring=scoring)
    results.append(score.mean())
    """

    # DMatrix
    dtrain = xgb.DMatrix(X_train, y_train)
    param = {'max_depth': 10, 'eta': 0.2, 'silent': 1, 'num_round': 20000}
    param['nthread'] = 4
    param['eval_metric'] = 'mae'
    model = xgb.train(param, dtrain)
    model_list.append(model)

    dtest = xgb.DMatrix(X_test, y_test)
    y_predicted = model.predict(dtest)
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_predicted, s=20)
    # score = sqrt(mean_squared_error(y_test, y_predicted))
    score = mean_absolute_error(y_test, y_predicted)
    # score = r2_score(y_test, y_predicted)
    plt.title(''.join(['Predicted vs. Actual.', ' mae = ', str(score)]))
    plt.xlabel('Actual y')
    plt.ylabel('Predicted y')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)])
    plt.tight_layout()
    #plt.show()
    results.append(score)
    scoring = 'r2'"""


    print("Iteration {0: >2} | {1} {2: 2.4f} ({3: 1.3f} std)".format(
           _, metric[scoring], score.mean(), score.std()))

# print("\nMean MAE: {}".format(round(np.mean(results), 2)))
# print("Best model: {}".format(np.argmin(results)+1))

"""
best_model = model_list[np.argmin(results)]
submission_test = submission[train.drop('SalePrice', axis=1).columns]
dtest = xgb.DMatrix(submission_test.values)
predict = best_model.predict(dtest)
"""

best_model = model_list[np.argmin(results)]
predict = best_model.predict(submission[train.drop('SalePrice', axis=1).columns].values)

submission['SalePrice'] = pd.Series(predict)
results = submission[['Id', 'SalePrice']]
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
