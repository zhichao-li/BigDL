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

from keras.layers import Dense, Dropout, Input, Activation
from keras.models import Sequential, Model
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.layers import Dense, Dropout, Input, Lambda
from keras.optimizers import RMSprop
from keras.utils import np_utils
from bigdl.util.common import create_tmp_path
from bigdl.keras1.converter import DefinitionLoader




class TestLayer():
    def __dump_keras(self, keras_model, dump_weights=False):
        keras_model_path = create_tmp_path()
        keras_model_json_path = keras_model_path + ".json"
        keras_model_hdf5_path = keras_model_path + ".hdf5"
        with open(keras_model_json_path, "w") as json_file:
            json_file.write(keras_model.to_json())
        print("json path: " + keras_model_json_path)
        if dump_weights:
            keras_model.save(keras_model_hdf5_path)
            print("hdf5 path: " + keras_model_hdf5_path)
        return keras_model_json_path, keras_model_hdf5_path

    def __modelTestSingleLayer(self,
                               input_data,
                               output_layer,
                               dump_weights=False,
                               predict_precision=1e-4):
        input_node = Input(shape=input_data.shape[1:])
        out = output_layer(input_node)
        keras_model = Model(input=input_node, output=out)

        keras_model_json_path, keras_model_hdf5_path = self.__dump_keras(keras_model, dump_weights)
        bigdl_model = DefinitionLoader.from_json(keras_model_json_path).to_bigdl()
        bigdl_output = bigdl_model.forward(input_data)
        keras_output = keras_model.predict(input_data)
        assert bigdl_output.shape == keras_output.shape
        # assert_allclose(bigdl_output, keras_output, rtol=1e-2)
        #  init result is not the same, so we disable it for now
        if dump_weights: # load weights if possible
            ModelLoader.load_weights(bigdl_model, keras_model, keras_model_hdf5_path)
            bweights = bigdl_model.get_weights()
            kweights = keras_model.get_weights()
            self.bigdl_assert_allclose(bweights[0], kweights[0], rtol=1e-4)
            if isinstance(bweights, list) and len(bweights) > 1: # if has bias
                self.bigdl_assert_allclose(bweights[1], kweights[1], rtol=1e-4)

        bigdl_output2 = bigdl_model.forward(input_data)

        # TODO: increase the presision?
        assert_allclose(bigdl_output2, keras_output, rtol=predict_precision)
        # np.testing.assert_array_almost_equal(bigdl_output2, keras_output)

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
        self.__modelTestSingleLayer(input_data,  dense, dump_weights=True)

    # Add more test case?
    def test_conv1D(self):
        input_data = np.random.random_sample([1, 10, 32])
        layer = Convolution1D(64, 3, border_mode='valid', input_shape=(10, 32))
        self.__modelTestSingleLayer(input_data, layer, dump_weights=True, predict_precision=1e-1)

        layer = Convolution1D(64, 3, border_mode='same', input_shape=(10, 32))
        self.__modelTestSingleLayer(input_data, layer, dump_weights=True, predict_precision=1e-1)

    def test_conv2D(self):
        input_data = np.random.random_sample([1, 3, 256, 256])
        layer = Convolution2D(64, 3, 3,
                    border_mode='same',
                                input_shape=(3, 256, 256))
        self.__modelTestSingleLayer(input_data,
                                    layer,
                                    dump_weights=True)

    def test_maxpooling2d(self): # TODO: implement create_flatten
        input_data = np.random.random_sample([1, 3, 20, 20])
        layer = MaxPooling2D(pool_size=[3, 3], strides=[2,2], border_mode="valid")
        self.__modelTestSingleLayer(input_data, layer)

    def test_flatten(self):
        input_data = np.random.random_sample([1, 2, 3])
        layer = Flatten()
        self.__modelTestSingleLayer(input_data, layer)

    # TODO: Support share weights training.
    def test_multiple_inputs_share_weights(self):
        with pytest.raises(Exception) as excinfo:
            input_node1 = Input(shape=[3, 16, 16])
            input_node2 = Input(shape=[3, 32, 32])
            conv2d = Convolution2D(5, 3, 3,
                                   border_mode='same')
            conv1 = conv2d(input_node1)
            conv2 = conv2d(input_node2)
            out1 = Flatten()(conv1)
            out2 = Flatten()(conv2)
            model1 = Model(input=[input_node1, input_node2], output=[out1, out2])
            tensor1, tensor2 = model1([input_node1, input_node2])
            out3 = Dense(7)(tensor1)
            out4 = Dense(8)(tensor2)
            model2 = Model(input=[input_node1, input_node2], output=[out3, out4])
            def_path, w_path = self.__dump_keras(model2)
            bigdl_model = DefinitionLoader.from_json(def_path).to_bigdl()


        assert str(excinfo.value) == """We don't support shared weights style for now"""

if __name__ == "__main__":
    pytest.main([__file__])
