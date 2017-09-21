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
from bigdl.keras1.backend import ModelLoader

from keras.layers import Dense, Dropout, Input, Activation
from keras.models import Sequential, Model
import numpy as np
np.random.seed(1337)  # for reproducibility
# from keras.layers.core import Dense, Dropout, Activation, Input
from keras.layers import Dense, Dropout, Input, Lambda
from keras.optimizers import RMSprop
from keras.utils import np_utils




class TestModelOp():

    def test_run_keras_example(self):
        input1 = Input(shape=(20,))
        dense = Dense(10)(input1)
        activation = Activation('relu')(dense)
        dense2 = Dense(10, activation='relu')(activation)
        dense3 = Dense(5)(dense2)
        activation2 = Activation('softmax')(dense3)
        model = Model(input=input1, output=activation2)
        model.compile(loss='categorical_crossentropy',
                      optimizer='Adagrad',
                      metrics=['accuracy'])

        bigdl_common.init_engine()
        bigdl_backend.use_bigdl_backend(model)
        input_data = np.random.random([4, 20])
        output_data = np.random.randint(1, 5, [4, 5])
        model.fit(input_data, output_data, batch_size=4, nb_epoch=2)
    def test_run_keras_linear_regression(self):
        input1 = Input(shape=(3,))
        dense = Dense(1, bias=False)(input1)
        model = Model(input=input1, output=dense)
        model.compile(loss='mse',
                      optimizer='sgd') # lr should be 0.01 otherwise the result is bad.

        bigdl_common.init_engine()
        bigdl_backend.use_bigdl_backend(model)
        input_data = np.random.uniform(0, 1, (1000, 3))
        expected_W = np.array([1, 2, 3]).transpose()
        output_data = np.dot(input_data, expected_W)
        model.fit(input_data, output_data, batch_size=4, nb_epoch=10)
        W = model.get_weights()
        assert_allclose(expected_W, W[0][0], rtol=1e-1)


if __name__ == "__main__":
    pytest.main([__file__])
