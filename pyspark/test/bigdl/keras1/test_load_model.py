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




class TestLoadModel():

    def __simple_mlp_model_single(self):
        input1 = Input(shape=(3,))
        dense = Dense(2)(input1)
        model = Model(input=input1, output=dense)
        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(),
                      metrics=['accuracy'])
        keras_model_path = bigdl_backend.create_tmp_path()
        keras_model_path_json = keras_model_path + ".json"
        keras_model_path_hdf5 = keras_model_path + ".hdf5"

        with open(keras_model_path_json, "w") as json_file:
            json_file.write(model.to_json())
        model.save(keras_model_path_hdf5)
        return model, keras_model_path_json, keras_model_path_hdf5

    def __simple_mlp_model(self):
        input1 = Input(shape=(20,))
        dense = Dense(10)(input1)
        activation = Activation('relu')(dense)
        dense2 = Dense(10, activation='relu')(activation)
        dense3 = Dense(5)(dense2)
        # activation2 = Activation('softmax')(dense3)
        model = Model(input=input1, output=dense3)
        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(),
                      metrics=['accuracy'])
        keras_model_path = bigdl_backend.create_tmp_path()
        keras_model_path_json = keras_model_path + ".json"
        keras_model_path_hdf5 = keras_model_path + ".hdf5"

        with open(keras_model_path_json, "w") as json_file:
            json_file.write(model.to_json())
        model.save(keras_model_path_hdf5)
        return model, keras_model_path_json, keras_model_path_hdf5



    def test_load_Graph(self):
        pass

    def test_load_weights(self):
        kmodel, keras_model_path_json, keras_model_path_hdf5 = self.__simple_mlp_model()
        bmodel = ModelLoader.load_definition(keras_model_path_json)
        ModelLoader.load_weights(bmodel,
                                       kmodel,

                                       keras_model_path_hdf5)
        input = np.random.sample([1, 20])
        boutput = bmodel.forward(input)
        koutput = kmodel.predict(input)
        assert_allclose(boutput, koutput, rtol=1e-5)

    def test_load_weights_single_layer(self):
        kmodel, keras_model_path_json, keras_model_path_hdf5 = self.__simple_mlp_model_single()
        bmodel = ModelLoader.load_definition(keras_model_path_json)
        ModelLoader.load_weights(bmodel,
                                       kmodel,
                                       keras_model_path_hdf5)
        input = np.random.sample([1, 3])
        boutput = bmodel.forward(input)
        koutput = kmodel.predict(input)
        assert_allclose(boutput, koutput, rtol=1e-5)

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
        bigdl_backend.install_bigdl_backend(model)
        input_data = np.random.random([4, 20])
        output_data = np.random.randint(1, 5, [4, 5])
        model.fit(input_data, output_data, batch_size=4, nb_epoch=2)


if __name__ == "__main__":
    pytest.main([__file__])
