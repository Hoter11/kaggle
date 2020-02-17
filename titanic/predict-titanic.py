# -*- coding: utf-8 -*-

"""
Titanic: Machine Learning from Disasters
https://www.kaggle.com/c/titanic/data
"""

import sys
from joblib import dump, load
import pandas as pd
import math
import matplotlib.pyplot as plt
from datasetfunctions import correlation_matrix, show_info, downcast_dtypes

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc


# Constants
titles = ['Mr.', 'Mrs.', 'Master.', 'Miss.', 'Don.',
          'Rev.', 'Dr.', 'Ms.', 'Dr.', 'Mlle.', 'Col.',
          'Capt.', 'Major.', 'Mme.', 'Countess.', 'Jonkheer.']
sex = ['male', 'female']
embarked = ['C', 'S']

objs = {'Name': titles, 'Sex': sex, 'Embarked': embarked}

# Parameters
train_model = True if sys.argv[1] == 'train' else False


def get_title(name):
    for title in titles:
        if title in name:
            return title
    return ''


def clean_dataset(df):
    df.Name = df.Name.apply(get_title)
    df.Name = df.Name.replace(['Lady', 'Countess', 'Capt', 'Col',
                               'Don', 'Dr', 'Major', 'Rev', 'Sir',
                               'Jonkheer', 'Dona'], 'Rare')

    df['Age'] = df['Age'].fillna(df['Age'].mean())
    # df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
    df = downcast_dtypes(df)

    df.loc[df['Age'] <= 16, 'Age'] = 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3
    df.loc[df['Age'] > 64, 'Age']

    df['Fare'].fillna(df['Fare'].dropna().median(), inplace=True)
    df.loc[df['Fare'] <= 7.91, 'Fare'] = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare']   = 2
    df.loc[df['Fare'] > 31, 'Fare'] = 3
    df['Fare'] = df['Fare'].astype(int)

    object_cols = df.select_dtypes(include=['object'])
    for col in object_cols:
        df[col] = df[col].astype('category', objs[col], ordered=True).cat.codes
    """dum = pd.get_dummies(df[col], prefix=col)
        df = df.merge(dum, left_index=True, right_index=True)
        df = df.drop(col, axis=1)

    for key in objs.keys():
        for elem in objs[key]:
            col = key+'_'+elem
            if col not in df.columns:
                df.loc[:, col] = 0"""
    return df


# Read all datasets
print("Reading datasets...")
X_train = pd.read_csv("datasets/train.csv")
X_test = pd.read_csv("datasets/test.csv")

print("Preparing data...")
drops = ['Ticket', 'Cabin', 'Embarked']
y_train = X_train.Survived
X_train = X_train.drop(drops + ['PassengerId', 'Survived'], axis=1)

# Feature Engineering
X_train = clean_dataset(X_train)
X_test = clean_dataset(X_test.drop(drops, axis=1))

if train_model:
    # Prepare input / output for training
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.25)
    X = X_train.values
    y = y_train.values

    model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                     max_depth=None, max_features='auto', max_leaf_nodes=None,
                                     min_impurity_split=1e-07, min_samples_leaf=1,
                                     min_samples_split=2, min_weight_fraction_leaf=0.0,
                                     n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
                                     verbose=0, warm_start=False)
    model = MLPClassifier()

    print("Training model...")
    model.fit(X, y)
    dump(model, 'models/.model-1')

    y_pred = model.predict(X_val)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_val, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    print("ROC AUC: {}".format(roc_auc))

    score = model.score(X_val, y_val)
    print("Validation score: {}".format(score))

else:
    model = load('models/.model-1')

# Predict testing data
print("Predicting results...")
predict = model.predict(X_test.drop('PassengerId', axis=1).values)

# Create submission file
X_test.loc[:, 'Survived'] = pd.Series(predict)
results = X_test[['PassengerId', 'Survived']]
results.to_csv('submission.csv', index=False)
print("Done!")
