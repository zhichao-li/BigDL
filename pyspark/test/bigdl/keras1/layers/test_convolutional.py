import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras.utils.np_utils import conv_input_length
from bigdl.keras1.utils.test_utils import *
import bigdl.keras1.layers.convolutional as convolutional
import bigdl.util.common as bigdl_common
_convolution_border_modes = ['valid', 'same']


#@keras_test
def test_convolution_2d():
    nb_samples = 2
    nb_filter = 2
    stack_size = 3
    nb_row = 10
    nb_col = 6

    for border_mode in _convolution_border_modes:
        for subsample in [(1, 1), (2, 2)]:
            if border_mode == 'same' and subsample != (1, 1):
                continue

            layer_test(convolutional.Convolution2D,
                       kwargs={'nb_filter': nb_filter,
                               'nb_row': 3,
                               'nb_col': 3,
                               'border_mode': border_mode,
                               'subsample': subsample},
                       input_shape=(nb_samples, nb_row, nb_col, stack_size))

            layer_test(convolutional.Convolution2D,
                       kwargs={'nb_filter': nb_filter,
                               'nb_row': 3,
                               'nb_col': 3,
                               'border_mode': border_mode,
                               'W_regularizer': 'l2',
                               'b_regularizer': 'l2',
                               'activity_regularizer': 'activity_l2',
                               'subsample': subsample},
                       input_shape=(nb_samples, nb_row, nb_col, stack_size))



if __name__ == '__main__':
    bigdl_common.init_engine()
    pytest.main([__file__])
