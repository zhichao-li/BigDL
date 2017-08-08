import unittest

from pyspark.context import SparkContext
from bigdl.util.common import create_spark_conf
from bigdl.util.common import init_engine

from bigdl.keras1.layers.core import *
from bigdl.keras1.engine.training import *



class TestWorkFlow(unittest.TestCase):
    def setUp(self):
        sparkConf = create_spark_conf()
        self.sc = SparkContext(master="local[4]", appName="test model",
                               conf=sparkConf)
        init_engine()

    def tearDown(self):
        self.sc.stop()

    def test_functional_api(self):
        input = Input(input_shape=(20, ))()
        dense1 = Dense(4, input_dim=20, activation="relu")(
            input)  # we need to add an Input if the parameter is empty.
        dense2 = Dense(2)(dense1)
        dense3 = Activation("relu")(dense2)
        model = Model(input=[input], output=[dense3])
        model.compile(optimizer="Adagrad", loss="categorical_crossentropy")

        # generate dummy data
        import numpy as np
        data = np.random.random((100, 20))
        labels = np.random.randint(low=1, high=2, size=(100, 1))

        model.fit(x=data, y=labels, batch_size=8, nb_epoch=10)

    def test_sequential_api(self):
        model = Sequential()
        model.add(Dense(4, input_dim=20, activation="relu"))
        model.add(Dense(2))
        model.add(Activation("relu"))
        model.compile(optimizer="Adagrad", loss="categorical_crossentropy")

        # generate dummy data
        import numpy as np
        data = np.random.random((100, 20))
        labels = np.random.randint(low=1, high=2, size=(100, 1))
        print(model.B)
        model.fit(x=data, y=labels, batch_size=8, nb_epoch=10)

    def test_mnist_mlp(self):
        import numpy as np
        np.random.seed(1337)  # for reproducibility

        from keras.datasets import mnist
        from bigdl.keras1.models import Sequential
        from bigdl.keras1.layers.core import Dense, Dropout, Activation
        from bigdl.keras1.optimizers import Adagrad
        from keras.utils import np_utils as keras_np_utils

        batch_size = 128
        nb_classes = 10
        nb_epoch = 20

        # the data, shuffled and split between train and test sets
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        X_train = X_train.reshape(60000, 784)
        X_test = X_test.reshape(10000, 784)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255
        print(X_train.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples')

        # TODO: need to sync the data format, bigdl only accept number for classnll instead of vector
        Y_train = y_train + 1
        Y_test = y_test + 1
        # convert class vectors to binary class matrices
        #Y_train = keras_np_utils.to_categorical(y_train, nb_classes)
        #Y_test = keras_np_utils.to_categorical(y_test, nb_classes)

        model = Sequential()

        model.add(Dense(512, input_dim=784))
        # #TODO: solve input_shape model.add(Dense(512, input_shape=(784,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(10))
        model.add(Activation('softmax'))

       # model.summary() # TODO: what's the meaning of summary????

        model.compile(loss='categorical_crossentropy',
                      optimizer=Adagrad(),
                      metrics=['accuracy'])

        # TODO: how to wrap history object?
        history = model.fit(X_train, Y_train,
                            batch_size=batch_size, nb_epoch=nb_epoch,
                            verbose=1, validation_data=(X_test, Y_test))
        print(model.B)
        #score = model.evaluate(X_test, Y_test, verbose=0)  # TODO: wrap evaluate method
        #print('Test score:', score[0])
        #print('Test accuracy:', score[1])