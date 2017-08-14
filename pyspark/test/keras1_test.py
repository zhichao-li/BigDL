import unittest

from pyspark.context import SparkContext
from bigdl.util.common import create_spark_conf
from bigdl.util.common import init_engine

from bigdl.keras1.layers.core import *
from bigdl.keras1.engine.training import *
from bigdl.keras1.layers.convolutional import *
from bigdl.keras1.visualize_util import *


class TestWorkFlow(unittest.TestCase):
    def setUp(self):
        sparkConf = create_spark_conf()
        self.sc = SparkContext(master="local[4]", appName="test model",
                               conf=sparkConf)
        init_engine()

    def tearDown(self):
        self.sc.stop()

# TODO: add test for multiple inputs.
    def test_visual(self):
        input = Input(input_shape=(20, ))()
        dense1 = Dense(4, input_dim=20, activation="relu")(
            input)  # we need to add an Input if the parameter is empty.
        dense2 = Dense(2)(dense1)
        activation = Activation("relu")
        out1 = activation(dense1)
        out2 = activation(dense2)

        model = Model(input=[input], output=[out1, out2])
        plot(model, to_file='model.png', show_shapes=True, show_layer_names=True)

    def test_node(self):
        input = Input(input_shape=(20, ))()
        dense1 = Dense(4, input_dim=20, activation="relu")(input)
        dense2 = Dense(2)(dense1)
        inbound_nodes = dense2.inbound_nodes()
        print inbound_nodes

    def test_merge(self):
        input1 = Input(input_shape=(20, ), name="input1")()
        input2 = Input(input_shape=(20, ))()
        merge_output = merge([input1, input2], mode='concat', concat_axis=1)
        model = Model(input=[input1, input2], output=[merge_output])
        plot(model, to_file='merge.png', show_shapes=True, show_layer_names=True)
        assert(merge_output.output_shape == (None, 40))

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

    def test_convolution2D(self):
        # apply a 3x3 convolution with 64 output filters on a 256x256 image:
        model = Sequential()
        model.add(Convolution2D(64, 3, 3,
                                border_mode='valid',
                                input_shape=(3, 256, 256)))
        # now model.output_shape == (None, 64, 256, 256)

        # add a 3x3 convolution on top, with 32 output filters:
        model.add(Convolution2D(32, 3, 3, border_mode='same'))
        # now model.output_shape == (None, 32, 256, 256)

    def test_reshape(self):
        from bigdl.nn.layer import *
        from bigdl.nn.criterion import *
        import numpy as np
        reshape = Reshape([6, 1], None)
        input = np.random.rand(32, 2, 3)
        output = reshape.forward(input)
        print output[0].shape

        # from bigdl.nn.layer import *
        # import numpy as np
        #
        # module = Linear(3, 5)
        #
        # print(module.forward(np.arange(1, 4, 1)))

    def test_flatten(self):
        model = Sequential()
        model.add(Convolution2D(64, 3, 3,
                                border_mode='valid',
                                input_shape=(3, 20, 20)))
        # now model.output_shape == (None, 64, 18, 18)
        model.add(Flatten())
        output = model.B.forward(np.random.random_sample((2, 3, 20, 20))) # batch_size: 2
        assert((2, 20736) == output.shape)


    def test_keras_merge(self):
        from keras.models import Model, Sequential
        from keras.layers import Input, Dense, Merge
        model1 = Sequential()
        model1.add(Dense(32, input_dim=32))

        model2 = Sequential()
        model2.add(Dense(32, input_dim=32))

        merged_model = Sequential()
        merged_model.add(Merge([model1, model2], mode='concat', concat_axis=1))
        merged_model


    def test_mnist_cnn(self):
        from keras.datasets import mnist
        from bigdl.keras1.models import Sequential
        from bigdl.keras1.layers import Dense, Dropout, Activation, Flatten
        from bigdl.keras1.layers import Convolution2D, MaxPooling2D
        from keras.utils import np_utils
        from bigdl.keras1.optimizers import Adagrad

        batch_size = 128
        nb_classes = 10
        nb_epoch = 1

        # input image dimensions
        img_rows, img_cols = 28, 28
        # number of convolutional filters to use
        nb_filters = 32
        # size of pooling area for max pooling
        pool_size = (2, 2)
        # convolution kernel size
        kernel_size = (3, 3)

        # the data, shuffled and split between train and test sets
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)


        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255
        print('X_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        #Y_train = np_utils.to_categorical(y_train, nb_classes) #TODO: modify CLASSNLL criterion to meet this format
        #Y_test = np_utils.to_categorical(y_test, nb_classes)

        Y_train = y_train + 1
        Y_test = y_test + 1

        model = Sequential()

        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                                border_mode='valid',
                                input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer=Adagrad(),
                      metrics=['accuracy'])
        plot(model, to_file='model.png', show_shapes=True, show_layer_names=True)
        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                  verbose=1, validation_data=(X_test, Y_test))
        score = model.evaluate(X_test, Y_test, verbose=0)
        print(" score: " + str(score[0]))