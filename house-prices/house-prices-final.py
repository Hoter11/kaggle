import pandas as pd
import numpy as np
from random import randint
import math

import xgboost as xgb
from sklearn import model_selection
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model
from sklearn.linear_model import Lasso

import matplotlib.pyplot as plt
import seaborn as sns


def encode_year(year):
    if np.isnan(year):
        return 0
    if year < 2000:
        return 1990
    if year < 2010:
        return 2000
    return 2010


def drop_cols(df, cols):
    for col in df.columns:
        if col not in cols:
            df = df.drop(col, axis=1)
    return df


def most_corr_cols(df, alpha=0.4):
    corr = df.corr()
    corr_cols = corr['SalePrice'].sort_values(ascending=False)
    cols = []
    for col, corr_level in corr_cols.iteritems():
        if corr_level > alpha:
            cols.append(col)
    return cols


def feature_engineering(df, train=True):
    for col in df.select_dtypes(exclude=np.number).columns:
        df[col] = df[col].fillna("None")
        df[col] = df[[col]].apply(lambda x: pd.factorize(x)[0] + 1)

    if train:
        df.SalePrice = df.SalePrice.apply(lambda x: np.log(x))
        df = df.drop('Id', axis=1)
        df = df[df['GrLivArea'] < 4500]
        df = df[df['GarageArea'] < 1200]
        df = df[df['TotalBsmtSF'] < 3000]
        df = df[df['1stFlrSF'] < 2800]

    df.FullBath = df.FullBath.apply(lambda x: 1 if x == 0 else x)
    df.TotRmsAbvGrd = df.TotRmsAbvGrd.apply(lambda x: int(x / 2))
    df.YearBuilt = df.YearBuilt.apply(lambda x: int(math.ceil(x / 10.0)) * 10)
    df.YearRemodAdd = df.YearRemodAdd.apply(
        lambda x: int(math.ceil(x / 10.0)) * 10)
    df.GarageYrBlt = df.GarageYrBlt.apply(encode_year)
    df.MasVnrArea = df.MasVnrArea.apply(lambda x: x if x == 0 else 1)
    df.SaleCondition = df.SaleCondition.apply(lambda x: 1 if x == 3 else 0)
    df = df.select_dtypes(include=[np.number]).interpolate().dropna()

    df = df.fillna(df.median())
    return df


# Load data
print("Reading data...")
train = pd.read_csv('train.csv')
submission = pd.read_csv('test.csv')
train = feature_engineering(train)

cols = most_corr_cols(train, alpha=-10) + ['Id']
train = drop_cols(train, cols)
submission = drop_cols(submission, cols)

submission = feature_engineering(submission, train=False)

model_seed = randint(0, 100)
model = xgb.XGBRegressor(seed=model_seed)
#model = linear_model.LinearRegression()
#model = Lasso(alpha=0.00005, random_state=5)

# Shuffle training data
train = train.sample(frac=1).reset_index(drop=True)

y = train['SalePrice'].values
X = train.drop('SalePrice', axis=1).values

kfold = model_selection.KFold(n_splits=7, random_state=4, shuffle=False)

scores = []
for train_index, test_index in kfold.split(X):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[
        train_index], y[test_index]
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    #scores.append(mean_absolute_error(y_test, y_predict))
    scores.append(mean_absolute_error(np.exp(y_test), np.exp(y_predict)))

#print(f"Scores: {scores}")
print(f"Scores mean: {np.mean(scores)}")
print(f"Scores median: {np.median(scores)}")
y_predict = submission.drop('Id', axis=1).values
predict = model.predict(y_predict)

submission['SalePrice'] = pd.Series(predict)
results = submission[['Id', 'SalePrice']]
results.SalePrice = results.SalePrice.apply(lambda x: np.exp(x))
results.to_csv('submission.csv', index=False)
