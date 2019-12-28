import os
import sys
import zipfile
import glob
import urllib.request
import ipykernel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Dropout, Flatten
from keras.callbacks.callbacks import Callback

def load_data(folder, steps, label):
    """ load data in folder
        return (x[:, steps, 3], y)
    """
    def walk_through_files(path, ext='.txt'):
        """ walk trhough all files in 'path' with extension 'ext' """
        for filepath in glob.iglob(os.path.join("%s/*%s"%(path, ext))):
            yield filepath

    def load_file(filename, steps, label):
        """ load data file, and return (x[steps], y) """
        sample_rate = 32 # 32 samples/sec
        data = pd.read_csv(filename, sep=" ")
        data = data.values
        x, y = list(), list()
        # ignore the first and last 2 sec data, as the user may not start event
        # yet or have already stopped event.
        for i in range(sample_rate*2, len(data)-1-steps-sample_rate*2):
            x.append(data[i:i+steps, :])
            y.append(label)
        return x, y

    x_all, y_all = list(), list()
    for f in walk_through_files(folder):
        # load all files in folder
        x, y = load_file(f, steps, label)
        x_all += x
        y_all += y
    # x dimension ~ (batch, steps=steps, channels=3)
    return np.moveaxis(np.dstack(x_all), -1, 0), np.array(y_all)

def peek_data(path):
    for filepath in glob.iglob(os.path.join("%s/*.txt"%(path))):
        print(filepath)
        data = pd.read_csv(filepath, sep=" ")
        if not data.empty:
            plt.figure()
            plt.plot(data.values)
            plt.grid()
            plt.legend(['x', 'y', 'z'])
            plt.savefig("peek.eps")
            plt.savefig("peek.svg")
            plt.show()
            return

if __name__ == "__main__":

    folder = "HMP_Dataset-master/"
    # download data file if necessary
    if not os.path.isdir(folder):
        url = "https://github.com/wchill/HMP_Dataset/archive/master.zip"
        filename, _ = urllib.request.urlretrieve(url, "HMP_Dataset.zip")
        with zipfile.ZipFile(filename, 'r') as zf:
            zf.extractall()

    if not os.path.isdir(folder):
        print("can't download data file")
        sys.exit(0)

    # load data
    sample_rate = 32 # 32 samples/sec
    steps = 2*sample_rate
    x_walk, y_walk = load_data(folder+'walk', steps, 1)
    x_climb_chair, y_climb_chair = load_data(folder+'Climb_stairs', steps, 0)
    print(x_walk.shape, y_walk.shape)
    print(x_climb_chair.shape, y_climb_chair.shape)

    walk_train_len = int(len(x_walk)*0.8)
    climb_train_len = int(len(x_climb_chair)*0.8)

    x_train = np.concatenate((x_walk[:walk_train_len, :, :], x_climb_chair[:climb_train_len, :, :]))
    y_train = to_categorical(np.concatenate((y_walk[:walk_train_len], y_climb_chair[:climb_train_len])))

    x_test = np.concatenate((x_walk[walk_train_len:, :, :], x_climb_chair[climb_train_len:, :, :]))
    y_test = to_categorical(np.concatenate((y_walk[walk_train_len:], y_climb_chair[climb_train_len:])))

    # setup model
    n_timesteps, n_features = x_train.shape[1], x_train.shape[2]
    n_outputs = 2
    epochs, batch_size = 100, 32
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # training
    class TestCallback(Callback):
        def __init__(self, test_data):
            super(TestCallback, self).__init__()
            self.test_data = test_data
            self.accuracy = []

        def on_epoch_end(self, epoch, logs=None):
            # evaluate with test data
            x, y = self.test_data
            loss, acc = self.model.evaluate(x, y, verbose=0)
            print('Testing loss: {}, acc: {}\n'.format(loss, acc))
            # save training and test accuracy
            self.accuracy.append([logs["accuracy"], acc])

    tc = TestCallback((x_test, y_test))

    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[tc])

    # save model
    model.save('walk.h5')

    # show result
    np.save("walk", tc.accuracy)
    plt.figure()
    plt.plot(np.arange(epochs), tc.accuracy)
    plt.legend(['Train', 'Test'])
    plt.grid()
    plt.savefig("accuracy.eps")
    plt.savefig("accuracy.svg")
    plt.show()
