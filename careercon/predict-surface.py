# -*- coding: utf-8 -*-

"""
Kaggle competition: CareerCon 2019 - Help Navigate Robots
https://www.kaggle.com/c/career-con-2019
"""

import sys
from joblib import dump, load
import pandas as pd
import matplotlib.pyplot as plt
from datasetfunctions import correlation_matrix, show_info, downcast_dtypes

from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

# Parameters
train_model = True if sys.argv[1] == 'train' else False

# Read all datasets
print("Reading datasets...")
X_train = pd.read_csv("datasets/X_train.csv")
y_train = pd.read_csv("datasets/y_train.csv")
X_test = pd.read_csv("datasets/X_test.csv")

print("Preparing data...")
X_train = X_train.set_index('series_id').join(y_train.set_index('series_id'))
X_train = X_train.reset_index()
y_train = X_train.surface
X_train = downcast_dtypes(X_train.drop(['group_id',
                                        'row_id',
                                        'surface',
                                        'measurement_number'], axis=1))

X_test = downcast_dtypes(X_test.drop(['row_id',
                                      'measurement_number'], axis=1))

X_train = X_train.drop('series_id', axis=1)
X_norm = (X_train - X_train.min())/(X_train.max() - X_train.min())
pca = PCA(n_components=2)  # 2-dimensional PCA
transformed = pd.DataFrame(pca.fit_transform(X_norm))

plt.scatter(transformed[y_train=='fine_concrete'][0], transformed[y_train=='fine_concrete'][1], label='fine_concrete', c='red')
plt.scatter(transformed[y_train=='concrete'][0], transformed[y_train=='concrete'][1], label='concrete', c='blue')
plt.scatter(transformed[y_train=='soft_tiles'][0], transformed[y_train=='soft_tiles'][1], label='soft_tiles', c='lightgreen')

plt.legend()
plt.show()
sys.exit()

if train_model:
    # Prepare input / output for training
    X = X_train.values
    y = y_train.values

    model = MLPClassifier()

    print("Training model...")
    model.fit(X, y)
    dump(model, 'models/.model-1')

else:
    model = load('models/.model-1')

# Predict testing data
print("Predicting results...")
predict = model.predict(X_test.drop('series_id', axis=1).values)

# Create submission file
X_test.loc[:, 'surface'] = pd.Series(predict)
X_test = X_test[['series_id', 'surface']]
X_test = X_test.groupby('series_id').agg(lambda x: x.value_counts().index[0])
X_test = X_test.reset_index()

results = X_test[['series_id', 'surface']]
results.to_csv('submission.csv', index=False)
print("Done!")
