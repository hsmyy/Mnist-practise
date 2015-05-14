from __future__ import absolute_import
#from __future__ import print_function
import os
import struct
from array import array

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.regularizers import l2, l1
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.utils import np_utils, generic_utils
import numpy as np

class MNIST(object):
    def __init__(self, path='.'):
        self.path = path

        self.test_img_fname = 't10k-images-idx3-ubyte'
        self.test_lbl_fname = 't10k-labels-idx1-ubyte'

        self.train_img_fname = 'train-images-idx3-ubyte'
        self.train_lbl_fname = 'train-labels-idx1-ubyte'

        self.test_images = []
        self.test_labels = []

        self.train_images = []
        self.train_labels = []

    def load_testing(self):
        ims, labels = self.load(os.path.join(self.path, self.test_img_fname),
                         os.path.join(self.path, self.test_lbl_fname))

        self.test_images = ims
        self.test_labels = labels

        return ims, labels

    def load_training(self):
        ims, labels = self.load(os.path.join(self.path, self.train_img_fname),
                         os.path.join(self.path, self.train_lbl_fname))

        self.train_images = ims
        self.train_labels = labels

        return ims, labels

    @classmethod
    def load(cls, path_img, path_lbl):
        with open(path_lbl, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049,'
                    'got %d' % magic)

            labels = array("B", file.read())

        with open(path_img, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051,'
                    'got %d' % magic)

            image_data = array("B", file.read())

        images = []
        for i in xrange(size):
            images.append([0]*rows*cols)

        for i in xrange(size):
            images[i][:] = image_data[i*rows*cols : (i+1)*rows*cols]

        return images, labels

    def test(self):
        test_img, test_label = self.load_testing()
        train_img, train_label = self.load_training()
        assert len(test_img) == len(test_label)
        assert len(test_img) == 10000
        assert len(train_img) == len(train_label)
        assert len(train_img) == 60000
        print ("Showing num:" , train_label[0])
        print (self.display(train_img[0]))
        print
        return True

    @classmethod
    def display(cls, img, width=28):
        render = ''
        for i in range(len(img)):
            if i % width == 0: render += '\n'
            if img[i] > 200:
                render += '1'
            else:
                render += '0'
        return render

def DNN(X_train, Y_train, X_test, Y_test):
    batch_size = 64
    nb_classes = 10
    nb_epoch = 20
    np.random.seed(1337)
    X_train = X_train.reshape(60000,784)
    X_test = X_test.reshape(10000,784)
    X_train = X_train.astype("float32")
    X_test.astype("float32")
    X_train /= 255
    X_test /= 255
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test smaples')

    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)

    model = Sequential()
    model.add(Dense(784, 128))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128,128))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128,10))
    model.add(Activation('softmax'))

    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms)

    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=2, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

def CNN2(X_train, Y_train, X_test, Y_test, activation='relu'):
    batch_size = 64
    nb_classes = 10
    nb_epoch = 20
    np.random.seed(1337)
    X_train = X_train.reshape(60000,1, 28, 28)
    X_test = X_test.reshape(10000,1, 28, 28)
    X_train = X_train.astype("float32")
    X_test.astype("float32")
    #X_train /= 255
    #X_test /= 255
    print(X_train.shape, 'train samples')
    print(Y_train.shape, 'train labels')
    print(X_test.shape, 'test smaples')

    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)

    model = Sequential()
    model.add(Convolution2D(4, 1, 5, 5, border_mode='valid'))
    model.add(Activation(activation))
    model.add(Convolution2D(8, 4, 3, 3, border_mode='valid'))
    model.add(Activation(activation))
    model.add(MaxPooling2D(poolsize=(2, 2)))
    model.add(Convolution2D(16, 8, 3, 3, border_mode='valid'))
    model.add(Activation(activation))
    model.add(MaxPooling2D(poolsize=(2,2)))
    model.add(Flatten())
    model.add(Dense(16 * 4 * 4, 128, init='normal'))
    model.add(Activation(activation))
    model.add(Dense(128, nb_classes, init='normal'))
    model.add(Activation('softmax'))
    sgd = SGD(l2=0.0, lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=10, shuffle=True, verbose=1, show_accuracy=True, validation_split=0.2)
    score = model.evaluate(X_test, Y_test, batch_size=batch_size)
    print('Test score:', score)


def CNN(X_train, Y_train, X_test, Y_test):
    batch_size = 64
    nb_classes = 10
    nb_epoch = 20
    np.random.seed(1337)
    X_train = X_train.reshape(60000,1, 28, 28)
    X_test = X_test.reshape(10000,1, 28, 28)
    X_train = X_train.astype("float32")
    X_test.astype("float32")
    #X_train /= 255
    #X_test /= 255
    print(X_train.shape, 'train samples')
    print(Y_train.shape, 'train labels')
    print(X_test.shape, 'test smaples')

    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)

    model = Sequential()
    model.add(Convolution2D(20, 1, 4, 4))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(poolsize=(2, 2), ignore_border=False))
    model.add(Convolution2D(40, 20, 5, 5))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(poolsize=(3, 3)))
    model.add(Flatten())
    model.add(Dense(40 * 3 * 3, 150, init='normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(150, nb_classes, init='normal'))
    model.add(Activation('softmax'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=30)
    score = model.evaluate(X_test, Y_test, batch_size=batch_size)
    print('Test score:', score)

    

if __name__ == "__main__":
    print ('Testing')
    mn = MNIST('.')
    if mn.test():
        print ('Passed')
        #CNN2(np.array(mn.train_images), np.array(mn.train_labels), np.array(mn.test_images), np.array(mn.test_labels), 'tanh')
        CNN2(np.array(mn.train_images), np.array(mn.train_labels), np.array(mn.test_images), np.array(mn.test_labels), 'relu')
        # DNN(np.array(mn.train_images), np.array(mn.train_labels), np.array(mn.test_images), np.array(mn.test_labels))
