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

    def __generate_model(self, input_data, output_layer):
        input_node = [Input(shape=input_data.shape[1:]) for i in input_data] # it's a list incase of Merge Layer
        out = output_layer(input_node)
        return Model(input=input_node, output=out)

    def __generate_sequence(self, input_data, output_layer):
        seq = Sequential()
        seq.add(output_layer)
        return seq

    def __modelTestSingleLayer(self,
                               input_data,
                               output_layer,
                               functional_api=True,
                               dump_weights=False,
                               predict_precision=1e-4):
        if functional_api:
            keras_model = self.__generate_model(input_data, output_layer)
        else:
            keras_model = self.__generate_sequence(input_data, output_layer)

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
        d = ~np.isclose(bigdl_output2, keras_output, rtol=predict_precision)
        np.where(d)
        assert_allclose(bigdl_output2, keras_output, rtol=predict_precision)
        # np.testing.assert_array_almost_equal(bigdl_output2, keras_output)

    def bigdl_assert_allclose(self, a, b, rtol=1e-7):
        if a.shape != b.shape and a.shape[0] == 1:
            a = a.squeeze(0) # bigdl has a leading 1 for conv2d
            b = b
        if a.shape != b.shape:
            a = a.transpose() # for Dense in keras and linear in bigdl has diff order
        assert_allclose(a, b, rtol)

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
        # TODO: increase the precision
        self.__modelTestSingleLayer(input_data, layer, dump_weights=True, predict_precision=1e-1)

        layer = Convolution1D(64, 3, border_mode='same', input_shape=(10, 32))
        self.__modelTestSingleLayer(input_data, layer, dump_weights=True, predict_precision=1e-1)


    def test_conv2D(self):
        # TODO: the shape of weights is not the same if using theano backend.
        input_data = np.random.random_sample([1, 3, 128, 128])
        # json_path = "/tmp/bigdlYUOHqN.json"
        # hdf5_path = "/tmp/bigdlYUOHqN.hdf5"
        # kmodel, bmodel = self._load_keras(json_path, hdf5_path)
        # koutput = kmodel.predict(input_data)
        # boutput = bmodel.forward(input_data)
        # assert_allclose(boutput, koutput, rtol=1e-4)



        layer = Convolution2D(64, 4, 4,
                    border_mode='valid')
        self.__modelTestSingleLayer(input_data,
                                    layer,
                                    dump_weights=True, predict_precision=1e-4)
        # Test if alias works or not
        # layer = Conv2D(64, 3, 3,
        #             border_mode='same',
        #                         input_shape=(3, 128, 128))
        # self.__modelTestSingleLayer(input_data,
        #                             layer,
        #                             dump_weights=True, predict_precision=1e-1)


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

    def test_batchnormalization(self): # test with channel first
        # For debug: bigdl_model.executions()[1].value.runningMean()
        # TODO mode=0 would fail 0 vs 2? oh, keras predict is in predict stage not the same as training.
        input_data = np.random.random_sample([2, 3, 20, 20])
        layer = BatchNormalization(input_shape=(3, 20, 20), epsilon=1e-3, mode=2, axis=1, momentum=0.99,
                 weights=None, beta_init='zero', gamma_init='one',
                 gamma_regularizer=None, beta_regularizer=None)
        # TODO: increase the precision, 1e-4 would fail
        self.__modelTestSingleLayer(input_data, layer, dump_weights=True, predict_precision=1e-3)

    def test_flatten(self):
        input_data = np.random.random_sample([1, 2, 3])
        layer = Flatten()
        self.__modelTestSingleLayer(input_data, layer)

    def test_reshape(self):
        input_data = np.random.random_sample([1, 3, 5, 4])
        layer = Reshape(target_shape=(3, 20))
        self.__modelTestSingleLayer(input_data, layer)

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
        layer = Merge([inputLayer1, inputLayer2], mode='concat', concat_axis=3) # the index including batch and start from zero, and it's the index to be merge
        input_data = [np.random.random_sample([2, 3, 6, 7]), np.random.random([2, 3, 6, 8])]
        # r = seq.predict(input_data)
        # print(r)
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
