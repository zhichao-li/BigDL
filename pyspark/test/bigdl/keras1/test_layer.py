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
        keras_model_json_path = bigdl_backend.create_tmp_path() + ".json"
        with open(keras_model_json_path, "w") as json_file:
            json_file.write(keras_model.to_json())
        print("json path: " + keras_model_json_path)
        bigdl_model = ModelLoader.load_definition(keras_model_json_path)
        bigdl_output = bigdl_model.forward(input_data)
        keras_output = keras_model.predict(input_data)
        assert_allclose(bigdl_output, keras_output, rtol=1e-5)

    def test_dense(self):
        input_data = np.random.random_sample([1, 10])
        dense = Dense(2, init='one', activation="relu")
        self.__modelTestSingleLayer(input_data,  dense)

    def test_conv2D(self):
        input_data = np.random.random_sample([1, 3, 256, 256])
        layer = Convolution2D(64, 3, 3,
                    border_mode='same',
                                input_shape=(3, 256, 256))
        self.__modelTestSingleLayer(input_data,  layer)



if __name__ == "__main__":
    pytest.main([__file__])
