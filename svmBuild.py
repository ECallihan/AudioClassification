"""
This file contains methods to train and record results for Support Vector Machine algorithms.
"""
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn import svm
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import csv
import processing as pr

# Hyper-parameters
svm_results = 'svmResults.csv'
c_values = tuple([10**x for x in range(-2, 3, 2)])
kernels = (
        'linear',
        'rbf',
        'sigmoid',
        'poly'
)


def train_svm(csv_data=pr.csv_save, c_vals=c_values, kernelTypes=kernels):
    """
    This method takes the processed feature and label data and inputs them into multiple
    SVMs with different kernel types and correction values.
    :param csv_data: The file path to the feature and label data.
    :param c_vals: The collection of correction values.
    :param kernelTypes: Types of SVM kernels.
    :return: An array of the results.
    """
    data = pd.read_csv(csv_data)
    data = data.drop(['filename'], axis=1)
    features = np.array(data)
    X = features[:, :-1]
    Y = features[:, -1]
    scalar = StandardScaler()
    scalar.fit(X)
    X = scalar.transform(X)
    encoder = LabelEncoder()
    Y = encoder.fit_transform(Y)
    kfold = KFold(n_splits=10, shuffle=True)
    kernel_results = []

    for kernel in kernelTypes:
        for c_val in c_vals:
            print(f'Training with {kernel} kernel and {c_val} C-value')
            model = svm.SVC(kernel=kernel, gamma='scale', C=c_val)
            model = model.fit(X, Y)
            k_scores = cross_val_score(model, X, Y, cv=kfold)
            k_score = np.mean(k_scores)
            k_std = np.std(k_scores)
            print(f'Accuracy of 10-fold cross-validation: {k_score}')
            kernel_results.append([kernel,
                                   c_val,
                                   k_score,
                                   k_std])
    return kernel_results


def write_results(results, file_name=svm_results):
    """
    This method writes the cross-validation SVM results to a CSV file.
    :param results: The array of results from training and evaluation.
    :param file_name: Name of the CSV file storing the results data.
    :return: None
    """
    with open(file_name, mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['Kernel',
                         'C-Value',
                         'Mean_10_Fold_Score',
                         'STD_10_Fold_Score'])
        for row in results:
            writer.writerow(row)


def run():
    """
    This method runs all SVM model configurations and records the data to a CSV file.
    :return:
    """
    results = train_svm()
    write_results(results)
    print('KNN evaluations complete')


if __name__ == '__main__':
    run()

