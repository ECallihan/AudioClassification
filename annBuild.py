"""
This file is used to build Artificial Neural Networks for audio classification
It is setup to allow multiple learning rates and architectures to be iterated,
allowing comparisons between models to be made and recorded in a single run.

The results of each model's ten fold cross evaluation is stored in a csv file.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras import layers
from keras.models import Sequential, load_model, save_model
from keras.optimizers import Adam
import csv
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import processing as pr

data_path = pr.csv_save
result_path = 'annResults.csv'

lrs = (
    0.01,
    0.001,
    0.0001
)

archs = (
    [512, 256, 128, 64, 32, 10],
    [256, 128, 64, 32, 16, 10],
    [256, 128, 64, 32, 10],
    [512, 256, 128, 64, 10],
    [256, 128, 64, 10])


def train_ann(dataset=data_path, lrates=lrs, architechtures=archs):
    """
    This method uses the processed CSV data to build and train multiple Artificial Neural Networks. Each model
    consists of only dense layers and a softmax function. The layers of the model use the ReLU activation
    function and are compiled with a sparse categorical cross-entropy loss function.
    :param dataset: path to the processed CSV data file.
    :param lrates: The learning rates to be iterated over.
    :param architechtures: The node counts for the dense layers of the ANN. The final layer must be 10.
    :return: An array of results from the different architecture-hyperparameter configurations showing the
    accuracy of the different models.
    """
    data = pd.read_csv(dataset)

    # Encoding the Labels
    labels = data.iloc[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    # Scaling the Feature columns
    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype=float))
    kfold = KFold(n_splits=10, shuffle=True)

    kfold_scores = []
    rateNum = layerNum = 0
    for rate in lrates:
        rateNum += 1
        opt = Adam(lr=rate)
        print(f'\n\n\n\nStarting Learning Rate {rateNum}')

        for layer in architechtures:
            layerNum += 1
            fold_no = 1
            acc_per_fold = []
            loss_per_fold = []
            print(f'\n\n\n\nStarting Layer Architecture {layerNum}\n\n\n\n')

            for train, test in kfold.split(X, y):
                mod = Sequential()
                mod.add(layers.Dense(layer[0], activation='relu', input_shape=(train.shape[1],)))

                for output in range(1, len(layer) - 1):
                    mod.add(layers.Dense(layer[output], activation='relu'))

                mod.add(layers.Dense(layer[-1], activation='softmax'))
                mod.compile(optimizer=opt,
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])
                # Generate a print
                print('------------------------------------------------------------------------')
                print(f'Training for fold {fold_no} ...')

                history = mod.fit(X[train],
                                  y[train],
                                  validation_data=(X[test], y[test]),
                                  epochs=40,
                                  batch_size=64)

                y_pred = mod.predict(X[test])

                # Generate a print
                print('------------------------------------------------------------------------')
                print(f'Rate Number {rateNum}\n'
                      f'Architecture Number {layerNum}\n'
                      f'Training for fold {fold_no} ...')

                conf_matrix = confusion_matrix(y[test], np.argmax(y_pred, axis=1))
                print('ANN Confusion Matrix:')
                print('       Model 1       ')
                print(f'Learning Rate: {rate}')
                print(conf_matrix)
                scores = mod.evaluate(X[test], y[test], verbose=0)
                print(f'Score for fold {fold_no}: {mod.metrics_names[0]} of {scores[0]}; '
                      f'{mod.metrics_names[1]} of {scores[1] * 100}%')
                acc_per_fold.append(scores[1] * 100)
                loss_per_fold.append(scores[0])

                # Increase fold number
                fold_no = fold_no + 1
            # == Provide average scores ==
            print('------------------------------------------------------------------------')
            print('Score per fold')
            for i in range(0, len(acc_per_fold)):
                print('------------------------------------------------------------------------')
                print(f'> Fold {i + 1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
            print('------------------------------------------------------------------------')
            print('Average scores for all folds:')
            print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
            print(f'> Loss: {np.mean(loss_per_fold)}')
            print('------------------------------------------------------------------------')
            kfold_scores.append([rate, layerNum, np.mean(acc_per_fold), np.std(acc_per_fold), np.mean(loss_per_fold)])
    return kfold_scores


def write_results(result_data, save_path=result_path):
    """
    This method writes the results of the 10-fold cross-validation for the different model-architecture combinations.
    :param result_data: Score data from training and evaluation.
    :param save_path: The save path for the CSV file containing the results data.
    :return: None.
    """
    with open(save_path, 'w') as annResults:
        resultsWriter = csv.writer(annResults)
        resultsWriter.writerow(['Learning_Rate',
                                'Architecture_Number',
                                'Accuracy',
                                'Standard_Deviation',
                                'Loss'])
        for score in result_data:
            resultsWriter.writerow(score)


def run():
    """
    This method trains, evaluates and records accuracy for all ANN model configurations
    :return: None
    """
    results = train_ann()
    write_results(results)
    print("ANN evaluations complete")


if __name__ == '__main__':
    run()

