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
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.layers import Dense, Dropout, Input, Lambda
from keras.optimizers import RMSprop
from keras.utils import np_utils




class TestLayer():

    def __modelTestSingleLayer(self, input_data, output_layer):
        input_node = Input(shape=input_data.shape[1:])
        out = output_layer(input_node)
        keras_model = Model(input=input_node, output=out)
        keras_model_path = bigdl_backend.create_tmp_path()
        keras_model_json_path = keras_model_path + ".json"
        with open(keras_model_json_path, "w") as json_file:
            json_file.write(keras_model.to_json())
        print("json path: " + keras_model_json_path)
        bigdl_model = bigdl_backend.DefinitionLoader(keras_model_json_path).to_bigdl()
        bigdl_output = bigdl_model.forward(input_data)
        keras_output = keras_model.predict(input_data)
        assert bigdl_output.shape == keras_output.shape
        # assert_allclose(bigdl_output, keras_output, rtol=1e-2)
        #  init result is not the same, so we disable it for now
        if output_layer.W:
            keras_model_hdf5_path = keras_model_path + ".hdf5"
            keras_model.save(keras_model_hdf5_path)
            print("hdf5 path: " + keras_model_hdf5_path)
            ModelLoader.load_weights(bigdl_model, keras_model, keras_model_hdf5_path)
            bigdl_output2 = bigdl_model.forward(input_data)
            self.bigdl_assert_allclose(bigdl_model.get_weights()[0], keras_model.get_weights()[0], rtol=1e-4)
            self.bigdl_assert_allclose(bigdl_model.get_weights()[1], keras_model.get_weights()[1], rtol=1e-4)
            assert_allclose(bigdl_output2, keras_output, rtol=1e-1) # TODO: increase the presision?

    def bigdl_assert_allclose(self, a, b, rtol=1e-7):
        if a.shape != b.shape:
            a = a.squeeze() # bigdl has a leading 1 for conv2d
            b = b.squeeze()
        if a.shape != b.shape:
            a = a.transpose() # for Dense in keras and linear in bigdl has diff order

        assert_allclose(a, b, rtol)


    def test_dense(self):
        input_data = np.random.random_sample([1, 10])
        dense = Dense(2, init='one', activation="relu")
        self.__modelTestSingleLayer(input_data,  dense)

    def test_flatten(self):
        input_data = np.random.random_sample([1, 2, 3])
        layer = Flatten()
        self.__modelTestSingleLayer(input_data, layer)

    def test_conv2D(self):
        input_data = np.random.random_sample([1, 3, 256, 256])
        layer = Convolution2D(64, 3, 3,
                    border_mode='same',
                                input_shape=(3, 256, 256))
        self.__modelTestSingleLayer(input_data,  layer)



if __name__ == "__main__":
    pytest.main([__file__])
