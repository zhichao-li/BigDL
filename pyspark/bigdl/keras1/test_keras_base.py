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


class TestBase():

    def _bigdl_assert_allclose(self, a, b, rtol=1e-7):
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

    def _dump_keras(self, keras_model, dump_weights=False):
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

    def _modelTest(self,
                   input_data,
                   keras_model,
                   dump_weights=False,
                predict_precision=1e-4):
        keras_model_json_path, keras_model_hdf5_path = self._dump_keras(keras_model, dump_weights)
        bigdl_model = DefinitionLoader.from_json(keras_model_json_path).to_bigdl()
        bigdl_output = bigdl_model.forward(input_data)
        keras_output = keras_model.predict(input_data)
        assert bigdl_output.shape == keras_output.shape
        # assert_allclose(bigdl_output, keras_output, rtol=1e-2)
        #  init result is not the same, so we disable it for now
        if dump_weights:  # load weights if possible
            ModelLoader.load_weights(bigdl_model, keras_model, keras_model_hdf5_path)
            bweights = bigdl_model.get_weights()
            kweights = keras_model.get_weights()
            self._bigdl_assert_allclose(bweights[0], kweights[0], rtol=1e-4)
            if isinstance(bweights, list) and len(bweights) > 1:  # if has bias
                self._bigdl_assert_allclose(bweights[1], kweights[1], rtol=1e-4)

        bigdl_output2 = bigdl_model.forward(input_data)

        # TODO: increase the presision?
        d = ~np.isclose(bigdl_output2, keras_output, rtol=predict_precision)
        np.where(d)
        assert_allclose(bigdl_output2, keras_output, rtol=predict_precision)
        # np.testing.assert_array_almost_equal(bigdl_output2, keras_output)
