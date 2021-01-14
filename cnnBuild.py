"""
This file provides methods for shaping and training the preprocessed data for
convolutional neural networks. Multiple network variations can be iterated over to track
results and tune hyper-parameters. Progress may be tracked in the terminal and the results
for all network evaluations are saved in a csv file.
"""
import numpy as np
import processing as pr
import tensorflow.keras.optimizers as opts
import csv
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout
from tensorflow.keras.utils import to_categorical


# Hyper-parameters
input_dims = (16, 8, 1)  # Product must equal 128
learning_rates = (0.001, 0.0005, 0.0001)
epoch_counts = (75,)
batch_sizes = (64,)
splits = 10
save_file = "cnnResults.csv"

# Sample Architectures
model1 = Sequential()
model1.add(Conv2D(64, (3, 3), padding="same", activation="relu", input_shape=input_dims))
model1.add(MaxPool2D(pool_size=(2, 2)))
model1.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
model1.add(MaxPool2D(pool_size=(2, 2)))
model1.add(Dropout(0.1))
model1.add(Flatten())
model1.add(Dense(1024, activation="relu"))
model1.add(Dense(10, activation="softmax"))

model2 = Sequential()
model2.add(Conv2D(64, (3, 3), padding="same", activation="relu", input_shape=input_dims))
model2.add(MaxPool2D(pool_size=(2, 2)))
model2.add(Conv2D(64, (3, 3), padding="same", activation="relu", input_shape=input_dims))
model2.add(MaxPool2D(pool_size=(2, 2)))
model2.add(Conv2D(64, (3, 3), padding="same", activation="relu", input_shape=input_dims))
model2.add(MaxPool2D(pool_size=(2, 2)))
model2.add(Dropout(0.5))
model2.add(Flatten())
model2.add(Dense(64, activation='relu'))
model2.add(Dense(32, activation='relu'))
model2.add(Dense(10, activation='softmax'))

model3 = Sequential()
model3.add(Conv2D(64, (3, 3), padding="same", activation="tanh", input_shape=input_dims))
model3.add(MaxPool2D(pool_size=(2, 2)))
model3.add(Conv2D(128, (3, 3), padding="same", activation="tanh"))
model3.add(MaxPool2D(pool_size=(2, 2)))
model3.add(Conv2D(128, (3, 3), padding="same", activation="tanh"))
model3.add(MaxPool2D(pool_size=(2, 2)))
model3.add(Dropout(0.1))
model3.add(Flatten())
model3.add(Dense(1024, activation="tanh"))
model3.add(Dense(10, activation="softmax"))

model_list = (model1, model2, model3)


def shape_data(file_name=pr.np_save):
    """
    This method allows the flattened data values from melspectrogram images to be reshaped for
    training the networks.
    :param file_name: This is the name of the .npy file with the feature values and labels.
    :return: The feature vector and the label vector.
    """
    data = np.load(file_name, allow_pickle=True)
    X_ = data[:, 0]
    Y = data[:, 1]
    X = np.empty([8732, 128])
    for i in range(8732):
        X[i] = (X_[i])
    Y = to_categorical(Y)
    return X, Y


def train_models(models=model_list, file_name=pr.np_save, in_dims=input_dims, lrs=learning_rates, epochs=epoch_counts,
                 n_splits=splits, saveFile=save_file):
    """
    This method trains sequential convolutional neural network models with multiple variations of
    hyper-parameter configurations. The results of each model's k-fold evaluation is saved to a CSV file.
    :param models: The collection of CNN model architectures.
    :param file_name: The file where the feature and label data is stored.
    :param in_dims: The input dimensions of the CNN. The product of the values must equal 128.
    :param lrs: The collection of learning rate values.
    :param epochs: The collection of epoch count values.
    :param n_splits: The number of splits for k-fold cross-validation.
    :param saveFile: CSV file name where result data will be stored.
    :return: None
    """
    X, Y = shape_data(file_name)
    X = X.reshape((8732, in_dims[0], in_dims[1], in_dims[2]))
    kfold = KFold(n_splits=n_splits, shuffle=True)
    with open(saveFile, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['Model Number', 'Learning Rate', 'Number of epochs', 'Accuracy', 'STD', 'Loss'])
    for lr in lrs:
        opt = opts.Adam(learning_rate=lr)
        for batch_size in batch_sizes:
            for epoch_count in epochs:
                for model_num in range(len(models)):
                    fold_no = 0
                    model = models[model_num]
                    acc_per_fold = []
                    loss_per_fold = []
                    for train, test in kfold.split(X):
                        fold_no += 1
                        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
                        model.fit(X[train], Y[train], epochs=epoch_count, batch_size=batch_size,
                                  validation_data=(X[test], Y[test]))
                        model.summary()
                        scores = model.evaluate(X[test], Y[test])
                        print(
                            f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; '
                            f'{model.metrics_names[1]} of {scores[1] * 100}%')
                        acc_per_fold.append(scores[1] * 100)
                        loss_per_fold.append(scores[0])
                        if fold_no == 1:
                            predictions = model.predict(X[test])
                            preds = np.argmax(predictions, axis=1)
                            conf_matrix = confusion_matrix(np.argmax(Y[test], axis=1), preds)
                            print('Confusion Matrix for fold 1')
                            print(conf_matrix)
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
                    with open(saveFile, 'a') as file:
                        writer = csv.writer(file)
                        writer.writerow(
                            [model_num + 1, lr, epoch_count, np.mean(acc_per_fold), np.std(acc_per_fold),
                             np.mean(loss_per_fold)])


def run():
    shape_data()
    train_models(model_list)
    print("CNN evaluation complete.")


if __name__ == '__main__':
    run()
