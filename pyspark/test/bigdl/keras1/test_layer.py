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

import numpy as np
import keras
import pytest
from keras.layers import *
from keras.models import Sequential, Model

from bigdl.keras1.converter import ModelLoader

np.random.seed(1337)  # for reproducibility
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.layers import Dense, Input
from keras.models import model_from_json
from bigdl.util.common import create_tmp_path
from bigdl.keras1.converter import DefinitionLoader
from test.bigdl.test_utils import BigDLTestCase


class TestLayer(BigDLTestCase):

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

    def __generate_model(self, input_data, output_layer):
        def without_batch(batch_shape):
            return batch_shape[1:]
        if isinstance(output_layer, keras.engine.Merge):  # it's a list in case of Merge Layer
            assert isinstance(list, input_data)
            input_tensor = [Input(shape=without_batch(input_data.shape)) for i in input_data]
        else:
            input_tensor = Input(shape=without_batch(input_data.shape))
        out_tensor = output_layer(input_tensor)
        return Model(input=input_tensor, output=out_tensor)

    def __generate_sequence(self, input_data, output_layer):
        seq = Sequential()
        seq.add(output_layer)
        return seq

    def __modelTestSingleLayer(self,
                               input_data,
                               output_layer,
                               functional_api=True,
                               dump_weights=False,
                               is_training=False,
                               rtol=1e-7,
                               atol=1e-7):
        if functional_api:
            keras_model = self.__generate_model(input_data, output_layer)
        else:
            keras_model = self.__generate_sequence(input_data, output_layer)

        keras_model_json_path, keras_model_hdf5_path = self.__dump_keras(keras_model, dump_weights)
        bigdl_model = DefinitionLoader.from_json(keras_model_json_path).to_bigdl()
        if not is_training:
            bigdl_model.evaluate()
        bigdl_output = bigdl_model.forward(input_data)
        keras_output = keras_model.predict(input_data)
        assert bigdl_output.shape == keras_output.shape
        #  init result is not the same, so we disable it for now
        if dump_weights: # load weights if possible
            ModelLoader.load_weights(bigdl_model, keras_model, keras_model_hdf5_path)
            bweights = bigdl_model.get_weights()
            kweights = keras_model.get_weights()
            self.bigdl_assert_allclose(bweights[0], kweights[0], rtol, atol)
            if isinstance(bweights, list) and len(bweights) > 1: # if has bias
                self.bigdl_assert_allclose(bweights[1], kweights[1], rtol=rtol, atol=atol)

        bigdl_output2 = bigdl_model.forward(input_data)
        self.assert_allclose(bigdl_output2,
                                              keras_output,
                                              rtol=rtol,
                                              atol=atol)

    def bigdl_assert_allclose(self, a, b, rtol=1e-7, atol=1e-7):
        if a.shape != b.shape and a.shape[0] == 1:
            a = a.squeeze(0) # bigdl has a leading 1 for conv2d
            b = b
        if a.shape != b.shape:
            a = a.transpose() # for Dense in keras and linear in bigdl has diff order
        self.assert_allclose(a, b, rtol, atol)

    def _load_keras(self, json_path, hdf5_path):
        with open(json_path, "r") as jp:
            kmodel = model_from_json(jp.read())
        kmodel.load_weights(hdf5_path)
        bmodel = ModelLoader.load_def_from_json(json_path)
        ModelLoader.load_weights(bmodel, kmodel, hdf5_path) # TODO: refactor reability of this api
        return kmodel, bmodel

    def test_dense(self):
        input_data = np.random.random_sample([1, 10])
        dense = Dense(2, init='one', activation="relu")
        self.__modelTestSingleLayer(input_data,  dense, dump_weights=True)

    def test_embedding(self):
        # Test index start from 0
        input_data = np.array([[0, 1, 2, 99], [0, 4, 5, 99]])
        layer = Embedding(input_dim = 100, # index [0,99]
                          output_dim = 64, # vector dim
                          init='uniform',
                          input_length=None,
                          W_regularizer=None, activity_regularizer=None,
                          W_constraint=None,
                          mask_zero=False,
                          weights=None, dropout=0.)
        self.__modelTestSingleLayer(input_data,  layer, dump_weights=True)
        # Random input
        input_data = np.random.randint(100, size=(10, 128)) # batch: 20, seqlen 128
        self.__modelTestSingleLayer(input_data, layer, dump_weights=True)

        # TODO: add test that exception would be raised if input_lenght == 6
        with pytest.raises(Exception) as excinfo:
            layer = Embedding(input_dim=100,  # index [0,99]
                              output_dim=64,  # vector dim
                              init='uniform',
                              input_length=111,
                              W_regularizer=None, activity_regularizer=None,
                              W_constraint=None,
                              mask_zero=False,
                              weights=None, dropout=0.)
            input_data = np.random.randint(100, size=(10, 128))  # batch: 20, seqlen 128
            self.__modelTestSingleLayer(input_data, layer, dump_weights=True)
        assert str(excinfo.value) == """The input_length doesn't match: 128 vs 111"""

    # Add more test case?
    def test_conv1D(self):
        input_data = np.random.random_sample([1, 10, 32])
        layer = Convolution1D(64, 3, border_mode='valid', input_shape=(10, 32))
        self.__modelTestSingleLayer(input_data, layer, dump_weights=True)

        layer = Convolution1D(64, 3, border_mode='same', input_shape=(10, 32))
        self.__modelTestSingleLayer(input_data, layer, dump_weights=True)


    def test_conv2D(self):
        image_dim_orders = ["tf", "th"]
        modes = ["valid", "same"]
        for order in image_dim_orders:
            keras.backend.set_image_dim_ordering(order)
            print("Testing with %s order" % keras.backend.image_dim_ordering())
            for mode in modes:
                print("Testing with mode %s" % mode)
                input_data = np.random.random_sample([1,3, 128, 128])
                layer = Convolution2D(64, 3, 3,
                            border_mode=mode,
                                        input_shape=(128, 128, 3))
                self.__modelTestSingleLayer(input_data,
                                            layer,
                                            dump_weights=True, rtol=1e-5, atol=1e-5)
                # # Test if alias works or not
                layer = Conv2D(64, 3, 3,
                            border_mode=mode,
                                        input_shape=(3, 128, 128))
                self.__modelTestSingleLayer(input_data,
                                            layer,
                                            dump_weights=True, rtol=1e-5, atol=1e-5)


    def test_maxpooling2d(self):
        input_data = np.random.random_sample([1, 3, 20, 20])
        layer = MaxPooling2D(pool_size=[3, 3], strides=[2,2], border_mode="valid")
        self.__modelTestSingleLayer(input_data, layer)

    def test_maxpooling1d(self):
        input_data = np.random.random_sample([1, 3, 20])
        layer = MaxPooling1D(pool_length=2, stride=None, border_mode='valid')
        self.__modelTestSingleLayer(input_data, layer)

    def test_globalmaxpooling2d(self):
        input_data = np.random.random_sample([1, 3, 20, 20])
        layer = GlobalMaxPooling2D()
        self.__modelTestSingleLayer(input_data, layer)

    def test_globalmaxpooling1d(self):
        input_data = np.random.random_sample([1, 3, 20])
        layer = GlobalMaxPooling1D()
        self.__modelTestSingleLayer(input_data, layer)

    def test_averagepooling2d(self):
        input_data = np.random.random_sample([1, 3, 20, 20])
        layer = AveragePooling2D(pool_size=[3, 3], strides=[2,2], border_mode="valid")
        self.__modelTestSingleLayer(input_data, layer)

    def test_averagepooling1d(self):
        input_data = np.random.random_sample([1, 3, 20])
        layer = AveragePooling1D(pool_length=2, stride=None, border_mode='valid')
        self.__modelTestSingleLayer(input_data, layer)

    def test_globalaveragepooling2d(self):
        input_data = np.random.random_sample([1, 3, 20, 20])
        layer = GlobalAveragePooling2D()
        self.__modelTestSingleLayer(input_data, layer)

    def test_globalaveragepooling1d(self):
        input_data = np.random.random_sample([1, 3, 20])
        layer = GlobalAveragePooling1D() # TODO: add dim_ordering as parameter?
        self.__modelTestSingleLayer(input_data, layer)

    def test_batchnormalization(self):
        # TODO: test training stage result, as the calc logic is not the same for mode 0
        image_dim_orders = ["tf", "th"]
        modes = ["valid", "same"]
        for order in image_dim_orders:
            keras.backend.set_image_dim_ordering(order)
            print("Testing with %s order" % keras.backend.image_dim_ordering())
            for mode in modes:
                print("Testing with mode %s" % mode)
                input_data = np.random.random_sample([2, 3, 20, 20])
                layer = BatchNormalization(input_shape=(3, 20, 20))
                self.__modelTestSingleLayer(input_data, layer, dump_weights=True)

    def test_flatten(self):
        input_data = np.random.random_sample([1, 2, 3])
        layer = Flatten()
        self.__modelTestSingleLayer(input_data, layer)

    def test_reshape(self):
        input_data = np.random.random_sample([1, 3, 5, 4])
        layer = Reshape(target_shape=(3, 20))
        self.__modelTestSingleLayer(input_data, layer)

    def test_merge_concat(self):
        inputLayer1 = InputLayer(input_shape=(3, 6, 7))
        inputLayer2 = InputLayer(input_shape=(3, 6, 8))
        # the index including batch and start from zero which is the index to be merge
        layer = Merge([inputLayer1, inputLayer2], mode='concat', concat_axis=3)
        input_data = [np.random.random_sample([2, 3, 6, 7]), np.random.random([2, 3, 6, 8])]
        self.__modelTestSingleLayer(input_data, layer, functional_api=False)

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
