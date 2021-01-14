"""
This file contains methods for training, testing and displaying the results of the
K-Nearest Neighbor algorithm using the UrbanSound8K dataset. The results are given for
80/20 split dropout evaluation and 10-fold cross-validation.
"""
import csv

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import processing as pr
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

knn_results = 'knnResults.csv'


def train_knn(dataset=pr.csv_save, kMax=41):
    """
    This method uses the preprocessed data from the processing file methods to train a KNN
    algorithm. The accuracy is tracked for both dropout and 10-fold cross-validation.
    :param dataset: preprocessed CSV file with the feature and label data.
    :param kMax: The limit on maximum neighbors. The program will stop processing once the
    number of neighbors reaches this value - 1.
    :return: Arrays of the dropout and kfold scores.
    """
    data = pd.read_csv(dataset)
    data = data.drop(['filename'], axis=1)
    features = np.array(data)
    X = features[:, :-1]
    Y = features[:, -1]
    # Scaling Data
    cvscaler = StandardScaler()
    cvscaler.fit(X)
    X = cvscaler.transform(X)
    encoder = LabelEncoder()
    Y = encoder.fit_transform(Y)
    # Cross validation with Kfold and dropout:
    kfold_scores = []
    dropout_scores = []
    for i in range(1, kMax):

        # Cross-fold validation
        print(f'results with {i} neighbors:')
        kfold = KFold(n_splits=10, shuffle=True)
        model_kfold = KNeighborsClassifier(n_neighbors=i)
        results_kfold = cross_val_score(model_kfold, X, Y, cv=kfold)
        kmean = results_kfold.mean()
        print("10 Fold Accuracy: %.2f%%" % (kmean * 100.0))
        kfold_scores.append(kmean)

        # Dropout Validation
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)
        model = KNeighborsClassifier(n_neighbors=i)
        model.fit(X_train, Y_train)
        if i == 1 or i == 40:
            y_pred = model.predict(X_test)
            print(confusion_matrix(Y_test, y_pred))
        result = model.score(X_test, Y_test)
        print("Dropout Accuracy: %.2f%%" % (result * 100.0))
        dropout_scores.append(result)
    return dropout_scores, kfold_scores


def write_results(cross_res, drop_res, file_name=knn_results):
    """
    This method writes the cross-validation SVM results to a CSV file.
    :param results: The array of results from training and evaluation.
    :param file_name: Name of the CSV file storing the results data.
    :return: None
    """
    with open(file_name, mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['K_value',
                         'cross_val_score',
                         'dropout_score'])
        for k in range(len(cross_res)):
            writer.writerow([k, cross_res[k], drop_res[k]])


def plot_knn(dropout, crossval, kMax=41):
    """
    This method plots dropout validation results against 10-fold cross-validation results
    by the number of neighbors used in the evaluation.
    :param dropout: Dropout accuracies array.
    :param crossval: Cross-validation accuracies array.
    :param kMax: Maximum number of neighbors to plot + 1.
    :return: None.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, kMax), crossval, color='red', linestyle='dashed', marker='o', markerfacecolor='blue',
             markersize='10', label='KFold')
    plt.plot(range(1, kMax), dropout, color='yellow', linestyle='dashed', marker='o', markerfacecolor='green',
             markersize='10', label='Dropout')
    plt.xlabel(f'K Neighbors')
    plt.ylabel(f'Accuracy')
    plt.title(f'Accuracy of Dropout and 10 Fold CV')
    plt.legend()
    plt.show()


def run():
    """
    This method builds trains the KNN models, records their results in a CSV file and plots the data.
    :return: None.
    """
    dropout, kfold = train_knn()
    write_results(kfold, dropout)
    plot_knn(dropout, kfold)


if __name__ == '__main__':
    run()

