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




class TestLoadDefinition():

    def test_load_mlp(self):
        pass


if __name__ == "__main__":
    pytest.main([__file__])
