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
from bigdl.keras1.converter import *
from test.bigdl.test_utils import BigDLTestCase, TestModels


class TestLayer(BigDLTestCase):

    def test_dense(self):
        input_data = np.random.random_sample([1, 10])
        dense = Dense(2, init='one', activation="relu")
        self.modelTestSingleLayer(input_data,  dense, dump_weights=True)

    def test_embedding(self):
        # Test index start from 0
        input_data = np.array([[0, 1, 2, 99], [0, 4, 5, 99]])
        layer = Embedding(input_dim=100,  # index [0,99]
                          output_dim=64,  # vector dim
                          init='uniform',
                          input_length=None,
                          W_regularizer=None, activity_regularizer=None,
                          W_constraint=None,
                          mask_zero=False,
                          weights=None, dropout=0.)
        self.modelTestSingleLayer(input_data,  layer, dump_weights=True)
        # Random input
        input_data = np.random.randint(100, size=(10, 128))  # batch: 20, seqlen 128
        self.modelTestSingleLayer(input_data, layer, dump_weights=True)

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
            self.modelTestSingleLayer(input_data, layer, dump_weights=True)
        assert str(excinfo.value) == """The input_length doesn't match: 128 vs 111"""

    # Add more test case?
    def test_conv1D(self):
        input_data = np.random.random_sample([1, 10, 32])
        # layer = Convolution1D(64, 3, border_mode='valid', input_shape=(10, 32))
        # self.modelTestSingleLayer(input_data, layer, dump_weights=True)
        # #
        # layer = Convolution1D(64, 3, border_mode='same', input_shape=(10, 32))
        # self.modelTestSingleLayer(input_data, layer, dump_weights=True)

        layer = Convolution1D(64, 3, border_mode='same', input_shape=(10, 32))
        self.modelTestSingleLayer(input_data, layer, dump_weights=True)

    def _load_keras(self, json_path, hdf5_path):
        with open(json_path, "r") as jp:
            kmodel = model_from_json(jp.read())
        kmodel.load_weights(hdf5_path)
        bmodel = ModelLoader.load_def_from_json(json_path)
        ModelLoader.load_weights(bmodel, kmodel, hdf5_path)  # TODO: refactor reability of this api
        return kmodel, bmodel

    def test_conv2D(self):
        image_dim_orders = ["tf", "th"]
        modes = ["valid", "same"]
        for order in image_dim_orders:
            keras.backend.set_image_dim_ordering(order)
            print("Testing with %s order" % keras.backend.image_dim_ordering())
            for mode in modes:
                print("Testing with mode %s" % mode)
                input_data = np.random.random_sample([1, 3, 128, 128])
                layer = Convolution2D(64, 1, 20,
                                      border_mode=mode,
                                      input_shape=(128, 128, 3))
                self.modelTestSingleLayer(input_data,
                                            layer,
                                            dump_weights=True, rtol=1e-5, atol=1e-5)
        # Test if alias works or not
        layer = Conv2D(64, 3, 1,
                          border_mode="valid",
                       input_shape=(3, 128, 128))
        self.modelTestSingleLayer(input_data,
                                    layer,
                                    dump_weights=True, rtol=1e-5, atol=1e-5)

    def test_maxpooling2d(self):
        input_data = np.random.random_sample([1, 3, 20, 20])
        layer = MaxPooling2D(pool_size=[3, 3], strides=[2, 2], border_mode="valid")
        self.modelTestSingleLayer(input_data, layer)

    def test_maxpooling1d(self):
        input_data = np.random.random_sample([1, 3, 20])
        layer = MaxPooling1D(pool_length=2, stride=None, border_mode='valid')
        self.modelTestSingleLayer(input_data, layer)

    def test_globalmaxpooling2d(self):
        input_data = np.random.random_sample([1, 3, 20, 20])
        layer = GlobalMaxPooling2D()
        self.modelTestSingleLayer(input_data, layer)

    def test_globalmaxpooling1d(self):
        input_data = np.random.random_sample([1, 3, 20])
        layer = GlobalMaxPooling1D()
        self.modelTestSingleLayer(input_data, layer)

    def test_averagepooling2d(self):
        input_data = np.random.random_sample([1, 3, 20, 20])
        layer = AveragePooling2D(pool_size=[3, 3], strides=[2, 2], border_mode="valid")
        self.modelTestSingleLayer(input_data, layer)

    def test_averagepooling1d(self):
        input_data = np.random.random_sample([1, 3, 20])
        layer = AveragePooling1D(pool_length=2, stride=None, border_mode='valid')
        self.modelTestSingleLayer(input_data, layer)

    def test_globalaveragepooling2d(self):
        input_data = np.random.random_sample([1, 3, 20, 20])
        layer = GlobalAveragePooling2D()
        self.modelTestSingleLayer(input_data, layer, rtol=1e-6, atol=1e-6)

    def test_globalaveragepooling1d(self):
        input_data = np.random.random_sample([1, 3, 20])
        layer = GlobalAveragePooling1D()  # TODO: add dim_ordering as parameter?
        self.modelTestSingleLayer(input_data, layer)

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
                self.modelTestSingleLayer(input_data, layer, dump_weights=True)

    def test_flatten(self):
        input_data = np.random.random_sample([1, 2, 3])
        layer = Flatten()
        self.modelTestSingleLayer(input_data, layer)

    def test_reshape(self):
        input_data = np.random.random_sample([1, 3, 5, 4])
        layer = Reshape(target_shape=(3, 20))
        self.modelTestSingleLayer(input_data, layer)

    def test_merge_concat(self):
        # input_data1 = np.random.random_sample([2, 3, 5])
        # input_data2 = np.random.random_sample([2, 3, 6])
        # model1 = Sequential()
        # model1.add(Dense(20, input_dim=2))
        # model1.add(Dense(20, input_dim=2))
        #
        # model2 = Sequential()
        # model2.add(Input(input_dim=32))
        #
        # merged_model = Sequential()
        # merged_model.add(Merge([model1, model2], mode='concat', concat_axis=0))

        inputLayer1 = InputLayer(input_shape=(3, 6, 7))
        inputLayer2 = InputLayer(input_shape=(3, 6, 8))
        inputLayer3 = InputLayer(input_shape=(3, 6, 9))

        layer = Merge([inputLayer1, inputLayer2, inputLayer3], mode='concat', concat_axis=3)
        # the index including batch and start from zero, and it's the index to be merge
        input_data = [np.random.random_sample([2, 3, 6, 7]),
                      np.random.random([2, 3, 6, 8]),
                      np.random.random([2, 3, 6, 9])]
        self.modelTestSingleLayer(input_data, layer, functional_api=False)

    def test_merge_sum(self):
        inputLayer1 = InputLayer(input_shape=(3, 6, 7))
        inputLayer2 = InputLayer(input_shape=(3, 6, 7))
        inputLayer3 = InputLayer(input_shape=(3, 6, 7))

        layer = Merge([inputLayer1, inputLayer2, inputLayer3], mode='sum')
        # the index including batch and start from zero, and it's the index to be merge
        input_data = [np.random.random_sample([2, 3, 6, 7]),
                      np.random.random([2, 3, 6, 7]),
                      np.random.random([2, 3, 6, 7])]
        self.modelTestSingleLayer(input_data, layer, functional_api=False)

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
        assert str(excinfo.value) == """Convolution2D doesn't support multiple inputs with shared weights"""  # noqa

if __name__ == "__main__":
    pytest.main([__file__])
