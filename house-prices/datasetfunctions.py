
"""
This file contains a handful of useful code-snippets for dataset information
extraction.
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def correlation_matrix(dataset, target_variable, top_variables=10):
    """
    Plot correlation matrix of a dataset with a column as the target_variable.
    Only shows X best top_variables.

    Code from:
    https://www.kaggle.com/junkal/selecting-the-best-regression-model/data
    """
    # Correlation built towards 'target_variable' column
    correlation = dataset.corr(method='pearson')
    columns = correlation.nlargest(top_variables, target_variable).index

    # Create correlation map
    correlation_map = np.corrcoef(dataset[columns].values.T)

    # Show correlation map
    sns.set(font_scale=1.0)
    heatmap = sns.heatmap(correlation_map, cbar=True, annot=True,
                          square=True, fmt='.2f',
                          yticklabels=columns.values,
                          xticklabels=columns.values)
    plt.show()


def show_info(dataset):
    """
    Copy of code to view preamble information of a dataset.

    Source:
    https://www.kaggle.com/kyakovlev/1st-place-solution-part-1-hands-on-data/
    """
    print("----------Top-5- Record----------")
    print(dataset.head(5))
    print("-----------Information-----------")
    print(dataset.info())
    print("-----------Data Types-----------")
    print(dataset.dtypes)
    print("----------Missing value-----------")
    print(dataset.isnull().sum())
    print("----------Null value-----------")
    print(dataset.isna().sum())
    print("----------Shape of Data----------")
    print(dataset.shape)
    print('Number of duplicates:', len(dataset[dataset.duplicated()]))

def downcast_dtypes(df):
    """Downcast floating and integer types of a dataframe
    Source:
    https://www.kaggle.com/anqitu/feature-engineer-and-model-ensemble-top-10
    """
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    return df
