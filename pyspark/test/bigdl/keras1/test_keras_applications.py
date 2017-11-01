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
import keras
import numpy as np
from keras.applications import *
from keras.applications import music_tagger_crnn
from bigdl.keras1.converter import ModelLoader

class TestKerasApplications():

    def transform_and_compare(self, input_data, kmodel, weights_path):
        # transform a keras model into a bigdl model and compare the result of two models given the same input
        bmodel = ModelLoader.load_def_from_kmodel(kmodel)
        ModelLoader.load_weights(bmodel, kmodel, weights_path)
        keras_output = kmodel.predict(input_data)
        bigdl_output = bmodel.forward(input_data)
        np.testing.assert_allclose(keras_output, bigdl_output, rtol=1e-6, atol=1e-6)
        print("Test Passed.")

    def test_vgg16(self, weights_path):
        keras.backend.set_image_dim_ordering("tf")
        kmodel = vgg16.VGG16(include_top=False, input_shape=(224, 224, 3))
        input_data = np.random.random([2, 224, 224, 3])
        self.transform_and_compare(input_data, kmodel, weights_path)

    def test_vgg19(self, weights_path):
        keras.backend.set_image_dim_ordering("tf")
        kmodel = vgg19.VGG19(include_top=False, input_shape=(224, 224, 3))
        input_data = np.random.random([2, 224, 224, 3])
        self.transform_and_compare(input_data, kmodel, weights_path)

    def test_inception_v3(self, weights_path):
        keras.backend.set_image_dim_ordering("th")
        kmodel = inception_v3.InceptionV3(include_top=False, input_shape=(3, 299, 299))
        input_data = np.random.random([2, 3, 299, 299])
        bmodel = ModelLoader.load_def_from_kmodel(kmodel)
        ModelLoader.load_weights(bmodel, kmodel, weights_path)
        keras_output = kmodel.predict(input_data)
        bmodel.evaluate()
        bigdl_output = bmodel.forward(input_data)
        np.testing.assert_allclose(keras_output, bigdl_output, rtol=1e-6, atol=1e-6)
        print("Test Passed.")

if __name__ == "__main__":
    # vgg16_weights_path = "/home/kai/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    # TestKerasApplications().test_vgg16(vgg16_weights_path)
    # vgg19_weights_path = "/home/kai/.keras/models/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5"
    # TestKerasApplications().test_vgg16(vgg19_weights_path)
    inception_v3_weights_path = "/home/kai/.keras/models/inception_v3_weights_th_dim_ordering_th_kernels_notop.h5"
    # inception_v3_weights_path = "/home/kai/.keras/models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
    TestKerasApplications().test_inception_v3(inception_v3_weights_path)