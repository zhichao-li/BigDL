#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from __future__ import print_function
import bigdl.nn.layer as bigdl_layer
import bigdl.optim.optimizer  as bigdl_optimizer
import bigdl.util.common as bigdl_common
import numpy as np
import pytest
import shutil
import tempfile
from numpy.testing import assert_allclose
import bigdl.keras1.backend as bigdl_backend
from bigdl.keras1.converter import ModelLoader

from keras.layers import *
from keras.models import Sequential, Model
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.layers import Dense, Dropout, Input, Lambda
from keras.models import model_from_json
from keras.optimizers import RMSprop
from keras.utils import np_utils
from bigdl.util.common import create_tmp_path
from bigdl.keras1.converter import DefinitionLoader
from bigdl.keras1.backend import use_bigdl_backend

from bigdl.keras1.test_keras_base import TestBase



class TestLayer(TestBase):

    def test_lenet(self):
        # assuming channel first
        input_shape = [1, 28, 28]
        b_input_shape = input_shape[:]
        b_input_shape.insert(0, 2)
        nb_classes = 10

        # input image dimensions
        img_rows, img_cols = 28, 28
        # number of convolutional filters to use
        nb_filters = 32
        # size of pooling area for max pooling
        pool_size = (2, 2)
        # convolution kernel size
        kernel_size = (3, 3)

        model = Sequential()

        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                                border_mode='valid',
                                input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adadelta',
                      metrics=['accuracy'])

        input_data = np.random.random_sample(b_input_shape)
        kr = model.predict(input_data, batch_size=2)
        self._modelTest(input_data, model, dump_weights=True)

    def test_quick_demo(self):
        '''This example demonstrates the use of Convolution1D for text classification.

        Gets to 0.89 test accuracy after 2 epochs.
        90s/epoch on Intel i5 2.4Ghz CPU.
        10s/epoch on Tesla K40 GPU.

        '''

        import numpy as np
        np.random.seed(1337)  # for reproducibility

        from keras.preprocessing import sequence
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, Activation
        from keras.layers import Embedding
        from keras.layers import Convolution1D, GlobalMaxPooling1D
        from keras.datasets import imdb

        # set parameters:
        max_features = 5000
        maxlen = 400
        batch_size = 32
        embedding_dims = 50
        nb_filter = 250
        filter_length = 3
        hidden_dims = 250
        nb_epoch = 2

        print('Loading data...')
        (X_train, y_train), (X_test, y_test) = imdb.load_data(
            path="/Users/lizhichao/god/data/imdb_full.pkl", nb_words=max_features)
        print(len(X_train), 'train sequences')
        print(len(X_test), 'test sequences')

        print('Pad sequences (samples x time)')
        X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
        X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
        print('X_train shape:', X_train.shape)
        print('X_test shape:', X_test.shape)

        print('Build model...')
        model = Sequential()

        # we start off with an efficient embedding layer which maps
        # our vocab indices into embedding_dims dimensions
        # model.add(Embedding(max_features,
        #                     embedding_dims,
        #                     input_length=maxlen,
        #                     dropout=0.2)) # Exception if specify Dropout

        model.add(Embedding(max_features,
                            embedding_dims,
                            input_length=maxlen)) # Exception if specify Dropout

        # we add a Convolution1D, which will learn nb_filter
        # word group filters of size filter_length:
        model.add(Convolution1D(nb_filter=nb_filter,
                                filter_length=filter_length,
                                border_mode='valid',
                                activation='relu',
                                subsample_length=1))
        # we use max pooling:
        # model.add(GlobalMaxPooling1D()) #TODO: Why there's exception for GlobalMaxPooling1D?
        model.add(GlobalAveragePooling1D())


        # We add a vanilla hidden layer:
        model.add(Dense(hidden_dims))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))

        # We project onto a single unit output layer, and squash it with a sigmoid:
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model = use_bigdl_backend(model)

        # model.fit(X_train, y_train,
        #           batch_size=batch_size,
        #           nb_epoch=nb_epoch,
        #           validation_data=(X_test, y_test))
        # 2017-09-22 15:53:45 INFO  DistriOptimizer$:657 - Top1Accuracy is Accuracy(correct: 21557, count: 25000, accuracy: 0.86228)
        # this result is from GlobalAveragePooling not GlobalMaxPooling.
        model.predict(X_test) # OK
        model.evaluate(X_test, y_test) # TODO: would keras print the result out while evaluating?
        print(model)


if __name__ == "__main__":
    pytest.main([__file__])
